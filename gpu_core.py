"""Shared core of the GPU monitor: sampling, history, logging and reduction.

Everything here is independent of how the data is displayed, so the matplotlib
window, the web dashboard and the headless mode all sit on top of it unchanged.
"""

import hashlib
import math
import subprocess
import sys
import threading
import time
from collections import namedtuple
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pynvml as nvml

# ===================== CONFIGURATION =====================
GPU_INDEX = 0                   # Which GPU to monitor
PLOT_INTERVAL_MS = 250          # How often the figure is redrawn
MAX_HISTORY_SECONDS = 48 * 3600 # Longest window offered, and how much is retained

# Sample rates offered by the "Sample rate" selector, in Hz.
SAMPLE_RATES = (10.0, 4.0, 2.0, 1.0, 0.5, 0.2)
DEFAULT_RATE_HZ = 4.0

# Window lengths offered by the "Window" selector, in seconds.
WINDOW_LENGTHS = (60, 300, 900, 3600, 6 * 3600, 12 * 3600, 24 * 3600, 48 * 3600)
DEFAULT_WINDOW_SECONDS = 60

# Rendering budget. Long windows hold far more samples than the screen has
# pixels, so they are reduced to an envelope (see `envelope`) instead of being
# drawn point by point.
PLOT_BUCKETS = 800              # Envelope buckets when the window is oversampled
SMOOTH_MAX_POINTS = 300         # Above this, curves are drawn without smoothing
MARKER_MAX_POINTS = 300         # Above this, individual samples are not marked
SMOOTHING_STEPS = 8             # Interpolated points per segment (<=1 disables)

PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs"
# =========================================================

# One color per metric, reused for its smoothed line, its raw samples and its fill.
COLOR_SM = "#00FFFF"
COLOR_BUS = "#FF00FF"
COLOR_FB = "#FFD700"
COLOR_GTEMP = "#FF4500"
COLOR_MTEMP = "#FF85C0"
COLOR_POWER = "#32CD32"
COLOR_RX = "#FFA500"
COLOR_TX = "#1E90FF"

FILL_ALPHA = 0.12               # Shading under every curve
RAW_LIGHTEN = 0.45              # How much lighter the raw-sample line is drawn

# Metric columns, in storage order. `t` is kept apart because it needs full
# float64 precision over a 48 hour run.
METRICS = ("sm", "bus", "fb", "gtemp", "mtemp", "power", "rx", "tx")
Sample = namedtuple("Sample", ("t",) + METRICS)


class History:
    """Chronological, fixed-duration store of samples, safe across threads.

    Samples stay contiguous and in order: when the buffer fills up, the oldest
    eighth is dropped in one move. That keeps reads to a plain slice, at the
    cost of an occasional memmove (roughly once every six hours at 4 Hz).
    """

    def __init__(self, rate_hz: float) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self._allocate(rate_hz)

    def _allocate(self, rate_hz: float) -> None:
        # 15% of slack over the longest window, so a full 48 h stays available
        # right after the oldest eighth is dropped.
        capacity = max(16, int(MAX_HISTORY_SECONDS * rate_hz * 1.15))
        times = np.zeros(capacity, dtype=np.float64)
        values = np.zeros((capacity, len(METRICS)), dtype=np.float32)

        if self._count:  # Carry over as much recent history as still fits.
            keep = min(self._count, capacity)
            times[:keep] = self._times[self._count - keep:self._count]
            values[:keep] = self._values[self._count - keep:self._count]
            self._count = keep

        self._times, self._values, self._capacity = times, values, capacity

    def resize(self, rate_hz: float) -> None:
        with self._lock:
            self._allocate(rate_hz)

    def append(self, sample: Sample) -> None:
        with self._lock:
            if self._count == self._capacity:
                drop = self._capacity // 8
                self._times[:-drop] = self._times[drop:]
                self._values[:-drop] = self._values[drop:]
                self._count -= drop
            self._times[self._count] = sample.t
            self._values[self._count] = sample[1:]
            self._count += 1

    def span(self) -> tuple[float, float]:
        """Oldest and newest timestamps currently stored."""
        with self._lock:
            if not self._count:
                return 0.0, 0.0
            return float(self._times[0]), float(self._times[self._count - 1])

    def view(self, start: float, end: float) -> dict[str, np.ndarray]:
        """Copy of every metric between two timestamps, keyed by metric name."""
        with self._lock:
            times = self._times[:self._count]
            lo = int(np.searchsorted(times, start, side="left"))
            hi = int(np.searchsorted(times, end, side="right"))
            window = {"t": times[lo:hi].copy()}
            values = self._values[lo:hi]
            window.update({name: values[:, i].astype(np.float64)
                           for i, name in enumerate(METRICS)})
        return window


class SampleLog:
    """CSV log holding every sample taken during one run of the monitor."""

    def __init__(self, sampler: "GpuSampler") -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        started = datetime.now()
        self.path = LOG_DIR / f"gpu_monitor_{started:%Y%m%d_%H%M%S}.csv"

        # Line buffered, so a log left behind by a crash is still complete.
        self._file = open(self.path, "w", buffering=1)
        self._file.write(
            f"# gpu: {sampler.name}\n"
            f"# started: {started.isoformat(timespec='seconds')}\n"
            f"# vram_total_mb: {sampler.total_memory / (1 << 20):.0f}\n"
            f"# power_cap_w: {sampler.power_limit:.0f}\n"
            f"# temp_target_c: {sampler.temp_target:.0f}\n"
            f"# temp_slowdown_c: {sampler.temp_slowdown:.0f}\n"
            f"# temp_shutdown_c: {sampler.temp_shutdown:.0f}\n"
            "timestamp,elapsed_s,sm,mem,fb,gtemp,mtemp,pwr,rxpci,txpci\n"
        )

    def write(self, sample: Sample) -> None:
        # NaN (an unsupported metric) is logged as an empty field.
        fields = ("" if math.isnan(v) else f"{v:.3f}" for v in sample[1:])
        self._file.write(f"{datetime.now().isoformat(timespec='milliseconds')},"
                         f"{sample.t:.3f}," + ",".join(fields) + "\n")

    def close(self) -> None:
        self._file.close()


class GpuSampler:
    """Background NVML poller feeding the history buffer and the log."""

    def __init__(self, index: int = GPU_INDEX) -> None:
        nvml.nvmlInit()
        self.index = index
        self.handle = nvml.nvmlDeviceGetHandleByIndex(index)

        name = nvml.nvmlDeviceGetName(self.handle)
        self.name = name.decode() if isinstance(name, bytes) else name
        self.total_memory = float(nvml.nvmlDeviceGetMemoryInfo(self.handle).total)

        # Static limits, read once: they never change while the monitor runs.
        self.power_limit = self._read_power_limit()
        self.temp_slowdown = self._read_threshold("NVML_TEMPERATURE_THRESHOLD_SLOWDOWN")
        self.temp_shutdown = self._read_threshold("NVML_TEMPERATURE_THRESHOLD_SHUTDOWN")
        self.temp_target = self._read_boost_target()

        # Consumer boards usually do not expose the GDDR temperature at all, so
        # probe once and drop the curve entirely when it is unsupported.
        self.has_mtemp = not math.isnan(self._read_memory_temp())

        self.errors = 0
        self.rate_hz = DEFAULT_RATE_HZ
        self.history = History(self.rate_hz)
        self.log = SampleLog(self)
        self._start = time.monotonic()

    # ---------------- NVML readings ----------------

    def _read_power_limit(self) -> float:
        """Enforced power cap in W, or NaN when the board does not report one."""
        try:
            return nvml.nvmlDeviceGetEnforcedPowerLimit(self.handle) / 1000.0
        except nvml.NVMLError:
            return math.nan

    def _read_threshold(self, constant: str) -> float:
        """Temperature threshold in C for the given NVML constant name."""
        try:
            return float(nvml.nvmlDeviceGetTemperatureThreshold(self.handle,
                                                                getattr(nvml, constant)))
        except (nvml.NVMLError, AttributeError):
            return math.nan

    def _read_boost_target(self) -> float:
        """GPU Boost thermal target in C.

        NVML has no getter for it, so it is parsed once from `nvidia-smi -q`.
        This is the soft limit the boost algorithm aims to stay below, and the
        one that actually trims clocks under sustained load.
        """
        try:
            output = subprocess.run(
                ["nvidia-smi", "-q", "-d", "TEMPERATURE", "-i", str(self.index)],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return math.nan

        for line in output.splitlines():
            if "GPU Target Temperature" in line:
                value = line.split(":", 1)[1].strip().split()[0]
                return float(value) if value.replace(".", "", 1).isdigit() else math.nan
        return math.nan

    def _read_memory_temp(self) -> float:
        """GDDR temperature in C, NaN when the driver does not expose it."""
        try:
            field = nvml.nvmlDeviceGetFieldValues(self.handle, [nvml.NVML_FI_DEV_MEMORY_TEMP])[0]
            if field.nvmlReturn == nvml.NVML_SUCCESS:
                return float(field.value.siVal)
        except (nvml.NVMLError, AttributeError):
            pass
        return math.nan

    def _read_sample(self) -> Sample:
        util = nvml.nvmlDeviceGetUtilizationRates(self.handle)
        memory = nvml.nvmlDeviceGetMemoryInfo(self.handle)
        return Sample(
            t=time.monotonic() - self._start,
            sm=float(util.gpu),
            bus=float(util.memory),
            # dmon reports `fb` in MB; as a share of total VRAM it can live on
            # the same percentage axis as the other two utilization curves.
            fb=100.0 * memory.used / self.total_memory,
            gtemp=float(nvml.nvmlDeviceGetTemperature(self.handle, nvml.NVML_TEMPERATURE_GPU)),
            mtemp=self._read_memory_temp() if self.has_mtemp else math.nan,
            power=nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0,
            # NVML reports PCIe throughput in KB/s.
            rx=nvml.nvmlDeviceGetPcieThroughput(self.handle, nvml.NVML_PCIE_UTIL_RX_BYTES) / 1024.0,
            tx=nvml.nvmlDeviceGetPcieThroughput(self.handle, nvml.NVML_PCIE_UTIL_TX_BYTES) / 1024.0,
        )

    # ---------------- Sampling loop ----------------

    def set_rate(self, rate_hz: float) -> None:
        """Change the sampling rate, re-sizing the history to keep 48 h of it."""
        self.rate_hz = rate_hz
        self.history.resize(rate_hz)

    def run(self) -> None:
        """Poll the GPU forever on a drift-free schedule."""
        next_tick = time.monotonic()
        while True:
            try:
                sample = self._read_sample()
                self.history.append(sample)
                self.log.write(sample)
            except Exception as error:  # noqa: BLE001
                # A sampler thread that dies leaves every front-end frozen with
                # no hint as to why, so nothing here is allowed to end the loop.
                self.errors += 1
                if self.errors == 1:
                    print(f"sampling error, further ones only counted: {error!r}",
                          file=sys.stderr)

            next_tick = max(next_tick + 1.0 / self.rate_hz, time.monotonic())
            time.sleep(max(0.0, next_tick - time.monotonic()))

    def start(self) -> None:
        threading.Thread(target=self.run, daemon=True).start()


class ViewState:
    """Which slice of the history is on screen, and whether it tracks the live edge."""

    def __init__(self, window_seconds: float) -> None:
        self.window = float(window_seconds)
        self.right_edge = 0.0
        self.following = True

    def bounds(self, oldest: float, newest: float) -> tuple[float, float]:
        """Visible time range, clamped to the history actually available."""
        if self.following:
            self.right_edge = newest
        else:
            # Held views stay put. Clamping up to a full window of history here
            # would drag the view forward whenever less than one window has been
            # recorded, which is exactly when a pause must still hold.
            self.right_edge = min(max(self.right_edge, oldest), newest)
        # Panning forward onto the newest sample resumes live following.
        if self.right_edge >= newest:
            self.following = True
        return self.right_edge - self.window, self.right_edge

    def pan(self, seconds: float) -> None:
        self.right_edge += seconds
        self.following = False

    def set_window(self, window_seconds: float) -> None:
        self.window = float(window_seconds)


def envelope(x: np.ndarray, y: np.ndarray, buckets: int = PLOT_BUCKETS):
    """Reduce a long series to a min/max envelope of real samples.

    Averaging would erase a one-sample glitch on a 48 hour window. Keeping the
    minimum and the maximum of each bucket, in the order they were measured,
    draws every spike at full height while cutting the point count by orders of
    magnitude. Both returned points are values that were actually sampled.
    """
    n = x.size
    if n <= 2 * buckets:
        return x, y

    per_bucket = n // buckets
    head = per_bucket * buckets
    xs = x[:head].reshape(buckets, per_bucket)
    ys = y[:head].reshape(buckets, per_bucket)

    rows = np.arange(buckets)
    lo, hi = ys.argmin(axis=1), ys.argmax(axis=1)
    first_is_low = (lo <= hi)[:, None]

    pairs_x = np.where(first_is_low, np.stack((xs[rows, lo], xs[rows, hi]), axis=1),
                       np.stack((xs[rows, hi], xs[rows, lo]), axis=1))
    pairs_y = np.where(first_is_low, np.stack((ys[rows, lo], ys[rows, hi]), axis=1),
                       np.stack((ys[rows, hi], ys[rows, lo]), axis=1))

    # Samples past the last whole bucket are kept as they are.
    return (np.concatenate((pairs_x.ravel(), x[head:])),
            np.concatenate((pairs_y.ravel(), y[head:])))


def parse_duration(text: str) -> float:
    """Read a window width such as "90", "45s", "10m", "1 min" or "1.5h" as seconds.

    Every spelling `format_window` produces is accepted, so whatever the box
    shows can be edited and submitted again unchanged.
    """
    text = text.strip().lower().replace(" ", "")
    # "min" is tried before "m" so it is not read as a bare minute suffix.
    for suffix, scale in (("min", 60.0), ("h", 3600.0), ("m", 60.0), ("s", 1.0)):
        if text.endswith(suffix):
            return float(text[:-len(suffix)]) * scale
    return float(text)


def format_elapsed(value: float, _pos=None) -> str:
    """Render an elapsed time in seconds compactly enough for a 48 h axis."""
    sign = "-" if value < 0 else ""
    seconds = int(abs(value))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{sign}{hours}h{minutes:02d}m"
    if minutes:
        return f"{sign}{minutes}m{secs:02d}s"
    return f"{sign}{secs}s"


def format_window(seconds: float) -> str:
    """Compact <num><unit> spelling of a duration: the form the boxes accept."""
    if seconds >= 3600:
        return f"{seconds / 3600:g}h"
    if seconds >= 60:
        return f"{seconds / 60:g}m"
    return f"{seconds:g}s"


def format_limits(low: float, high: float) -> str:
    """Spell an axis range the way the limit boxes show it."""
    return f"{low:g}{LIMIT_SEPARATOR}{high:g}"


def parse_limits(text, fallback: tuple[float, float]) -> tuple[float, float]:
    """Read a "min;max" axis range, falling back when the text is unusable.

    Only ';' separates the pair. A comma is the decimal separator in many
    locales, so splitting on it would read "0,5" as two numbers instead of one.
    """
    try:
        low, high = (float(part) for part in str(text).split(LIMIT_SEPARATOR))
        return (low, high) if high > low else fallback
    except (ValueError, TypeError):
        return fallback


LIMIT_SEPARATOR = ";"             # Separates min from max in the limit boxes
DEFAULT_BW_AXIS = (0.0, 3000.0)   # MB/s; RX bursts on this bus reach ~2.5 GB/s


def default_temp_axis(sampler) -> tuple[float, float]:
    """Round temperature bounds that still show the shutdown threshold.

    Round bounds matter on the thermal plot: temperature and power share it
    through a twin axis, and only tick steps that divide evenly let the two
    grids fall on the same lines.
    """
    top = 100.0 if math.isnan(sampler.temp_shutdown) else sampler.temp_shutdown + 2.0
    return 20.0, math.ceil(top / 20.0) * 20.0


def default_power_axis(sampler) -> tuple[float, float]:
    """Round power bounds that still show the enforced cap."""
    top = 400.0 if math.isnan(sampler.power_limit) else sampler.power_limit * 1.08
    return 0.0, math.ceil(top / 100.0) * 100.0


STARTED_AT = datetime.now()


@lru_cache(maxsize=1)
def build_id() -> str:
    """Identity of the code this process is actually running.

    Three separate questions hide behind "am I looking at the right build?":
    the commit answers which version, the source digest answers whether
    uncommitted edits are included, and the start time answers whether the
    process was restarted since those edits were made. A stale server answers
    the first two correctly and still shows old behaviour, so all three are
    reported together.
    """
    try:
        def git(*args: str) -> str:
            return subprocess.run(["git", "-C", str(PROJECT_DIR), *args], check=True,
                                  capture_output=True, text=True, timeout=5).stdout.strip()

        commit = git("rev-parse", "--short", "HEAD")
        commit += "-dirty" if git("status", "--porcelain") else ""
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"

    digest = hashlib.sha256()
    for path in sorted(PROJECT_DIR.glob("*.py")):
        digest.update(path.read_bytes())

    return (f"commit {commit}, src {digest.hexdigest()[:6]}, "
            f"started {STARTED_AT:%Y-%m-%d %H:%M:%S}")
