"""Real-time NVIDIA GPU monitor.

Samples the GPU through NVML (nvidia-ml-py) and draws three live plots:
utilization, thermals + power, and PCIe bandwidth. Every sample is also
appended to a timestamped CSV log created in the project folder at startup.

NVML is used instead of `nvidia-smi dmon` because dmon's sampling period is an
integer number of seconds, while this monitor samples several times per second.
Legends keep the nvidia-smi names in parentheses, so every curve maps back to
what `nvidia-smi dmon` / `nvidia-smi -q` report for the same quantity.

Interactive controls:
  * Sample rate and window length are selectable on the right-hand panel.
  * The bottom slider pans the visible window; the left/right arrow keys pan it
    by a quarter window, and panning to the right edge resumes live following.
  * Every y axis and the window width are fixed by text boxes: axes never
    rescale on their own, so a curve keeps the same shape as values change.
"""

import math
import subprocess
import threading
import time
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pynvml as nvml
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import RadioButtons, Slider, TextBox

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

LOG_DIR = Path(__file__).resolve().parent / "logs"
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
                ["nvidia-smi", "-q", "-d", "TEMPERATURE", "-i", str(GPU_INDEX)],
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
            except nvml.NVMLError:
                pass  # Transient driver hiccup: skip this tick and keep going.

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
        earliest_edge = min(oldest + self.window, newest)
        if self.following:
            self.right_edge = newest
        self.right_edge = min(max(self.right_edge, earliest_edge), newest)
        # Panning forward onto the newest sample resumes live following.
        if self.right_edge >= newest:
            self.following = True
        return self.right_edge - self.window, self.right_edge

    def pan(self, seconds: float) -> None:
        self.right_edge += seconds
        self.following = False

    def set_window(self, window_seconds: float) -> None:
        self.window = float(window_seconds)


# ===================== DRAWING HELPERS =====================

def lighten(color: str, amount: float = RAW_LIGHTEN) -> tuple[float, float, float]:
    """Blend a color towards white, for the raw-sample line."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1.0 - r) * amount, g + (1.0 - g) * amount, b + (1.0 - b) * amount)


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


def smooth(x: np.ndarray, y: np.ndarray):
    """Catmull-Rom spline through every sample, without pulling in SciPy.

    The spline is parameterized by sample index and applied to both axes, so
    uneven sampling intervals do not distort the curve. The result is clipped to
    the observed range to keep the spline's overshoot from inventing values.
    """
    n = x.size
    if SMOOTHING_STEPS <= 1 or n < 3 or n > SMOOTH_MAX_POINTS:
        return x, y

    idx = np.arange(n)
    p0, p1 = np.clip(idx - 1, 0, n - 1)[:-1], idx[:-1]
    p2, p3 = np.clip(idx + 1, 0, n - 1)[:-1], np.clip(idx + 2, 0, n - 1)[:-1]

    t = np.linspace(0.0, 1.0, SMOOTHING_STEPS, endpoint=False)[None, :]
    t2, t3 = t * t, t * t * t
    b0, b1 = -0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0
    b2, b3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2

    def interpolate(v: np.ndarray) -> np.ndarray:
        segments = v[p0, None] * b0 + v[p1, None] * b1 + v[p2, None] * b2 + v[p3, None] * b3
        return np.append(segments.ravel(), v[-1])

    return interpolate(x), np.clip(interpolate(y), y.min(), y.max())


def draw_series(ax, x, y, color, label, baseline):
    """Draw one metric: smoothed line, shaded area, and the real samples."""
    x, y = envelope(x, y)
    sx, sy = smooth(x, y)
    line, = ax.plot(sx, sy, color=color, linewidth=2.0, label=label, zorder=3)
    ax.fill_between(sx, sy, baseline, color=color, alpha=FILL_ALPHA, linewidth=0, zorder=2)
    # The real samples stay visible on a lighter line of the same color, so the
    # smoothing never hides what was actually measured. Markers are dropped once
    # they would merge into a solid band.
    ax.plot(x, y, color=lighten(color), linewidth=0.8, alpha=0.9, zorder=4,
            marker="o" if x.size <= MARKER_MAX_POINTS else "None", markersize=2.5)
    return line


def draw_limit(ax, value, label, color, linestyle=":", x=0.01, ha="left", va="bottom"):
    """Draw a hardware limit as a horizontal line labelled in place.

    The label sits at a caller-chosen horizontal slot: thresholds only a few
    degrees apart would otherwise print on top of each other.
    """
    if math.isnan(value):
        return
    ax.axhline(value, color=color, linestyle=linestyle, linewidth=1.2, alpha=0.8, zorder=1)
    ax.text(x, value, f" {label} ", color=color, fontsize=7.5, alpha=0.95,
            ha=ha, va=va, transform=ax.get_yaxis_transform(), zorder=5)


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


def legend_above(ax, ncol: int, handles=None):
    """Put a legend in the margin above the axes, where it hides no data."""
    labels = [h.get_label() for h in handles] if handles else None
    args = (handles, labels) if handles else ()
    ax.legend(*args, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=ncol,
              fontsize=8, frameon=False, borderaxespad=0.0)


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
    """Label for the window-length selector."""
    if seconds >= 3600:
        return f"{seconds / 3600:g} h"
    return f"{seconds / 60:g} min"


# ===================== FIGURE =====================

sampler = GpuSampler()
sampler.start()
view = ViewState(DEFAULT_WINDOW_SECONDS)

plt.style.use("dark_background")
# The arrow keys pan the time window, so they are taken back from matplotlib's
# default "previous/next view" bindings.
for key, binding in (("left", "keymap.back"), ("right", "keymap.forward")):
    plt.rcParams[binding] = [k for k in plt.rcParams[binding] if k != key]

fig = plt.figure(figsize=(14, 9))
fig.canvas.manager.set_window_title(f"NVIDIA GPU Monitor - {sampler.name}")
grid = fig.add_gridspec(3, 1, left=0.07, right=0.75, top=0.92, bottom=0.13, hspace=0.30)
ax_util = fig.add_subplot(grid[0])
ax_temp = fig.add_subplot(grid[1], sharex=ax_util)
ax_bw = fig.add_subplot(grid[2], sharex=ax_util)

# Power shares the thermal plot through a second y axis: both tell the same
# story of how much headroom the card has left. Created once, cleared per frame.
ax_power = ax_temp.twinx()

# ---- Controls ----
# Matplotlib has no combo box, so the two selectors are radio button groups.
ax_rate = fig.add_axes((0.815, 0.72, 0.17, 0.22))
ax_rate.set_title("Sample rate", fontsize=9, color="white")
rate_buttons = RadioButtons(ax_rate, tuple(f"{hz:g} Hz" for hz in SAMPLE_RATES),
                            active=SAMPLE_RATES.index(DEFAULT_RATE_HZ))

ax_window = fig.add_axes((0.815, 0.44, 0.17, 0.25))
ax_window.set_title("Window", fontsize=9, color="white")
window_buttons = RadioButtons(ax_window, tuple(format_window(s) for s in WINDOW_LENGTHS),
                              active=WINDOW_LENGTHS.index(DEFAULT_WINDOW_SECONDS))

for buttons in (rate_buttons, window_buttons):
    for label in buttons.labels:
        label.set_fontsize(8)


def make_text_box(rect, label, initial):
    """Text box with its caption above it, to fit the narrow control column."""
    box = TextBox(fig.add_axes(rect), label, initial=initial,
                  color="#1c1c1c", hovercolor="#2a2a2a")
    box.label.set_position((0.0, 1.30))
    box.label.set_horizontalalignment("left")
    box.label.set_fontsize(8)
    box.text_disp.set_color("white")
    box.text_disp.set_fontsize(9)
    return box


class LimitControl:
    """Fixed "min,max" bounds for one y axis, edited through a text box.

    The axes deliberately never rescale themselves: an axis that follows its
    data hides exactly what a monitor is meant to show, since every trace ends
    up filling the same height whatever the values are.
    """

    def __init__(self, rect, label, low, high):
        self.low, self.high = float(low), float(high)
        self._syncing = False
        self.box = make_text_box(rect, label, self._text())
        self.box.on_submit(self._on_submit)

    def _text(self) -> str:
        return f"{self.low:g},{self.high:g}"

    def _on_submit(self, text: str) -> None:
        if self._syncing:
            return
        try:
            low, high = (float(part) for part in text.replace(",", " ").split())
            if high > low:
                self.low, self.high = low, high
        except ValueError:
            pass  # Unparsable entry: fall through and restore the current bounds.
        self._syncing = True
        self.box.set_val(self._text())
        self._syncing = False

    def apply(self, ax) -> float:
        """Pin the axis to these bounds and return the baseline for its fills."""
        ax.set_ylim(self.low, self.high)
        return self.low


class WindowControl:
    """Exact window width, edited through a text box and driven by the presets."""

    def __init__(self, rect, label, seconds):
        self._syncing = False
        self.box = make_text_box(rect, label, format_window(seconds))
        self.box.on_submit(self._on_submit)

    def _on_submit(self, text: str) -> None:
        if self._syncing:
            return
        try:
            view.set_window(min(max(parse_duration(text), 1.0), MAX_HISTORY_SECONDS))
        except ValueError:
            pass
        self.show(view.window)

    def show(self, seconds: float) -> None:
        """Refresh the text without re-entering the submit handler."""
        self._syncing = True
        self.box.set_val(format_window(seconds))
        self._syncing = False


BOX_HEIGHT = 0.035
BOX_RECTS = [(0.815, 0.36 - i * 0.068, 0.17, BOX_HEIGHT) for i in range(5)]

window_box = WindowControl(BOX_RECTS[0], "X window (s/m/h)", DEFAULT_WINDOW_SECONDS)
util_limits = LimitControl(BOX_RECTS[1], "Utilization y (%)", 0, 105)
temp_limits = LimitControl(BOX_RECTS[2], "Temperature y (C)", 20,
                           (100.0 if math.isnan(sampler.temp_shutdown) else sampler.temp_shutdown) + 5)
power_limits = LimitControl(BOX_RECTS[3], "Power y (W)", 0,
                            400.0 if math.isnan(sampler.power_limit)
                            else round(sampler.power_limit * 1.1, -1))
bw_limits = LimitControl(BOX_RECTS[4], "PCIe y (MB/s)", 0, 2000)
TEXT_BOXES = (window_box.box, util_limits.box, temp_limits.box, power_limits.box, bw_limits.box)

ax_slider = fig.add_axes((0.07, 0.04, 0.68, 0.025))
pan_slider = Slider(ax_slider, "Pan", 0.0, 1.0, valinit=1.0)
pan_slider.valtext.set_fontsize(8)
pan_slider.label.set_fontsize(8)

_syncing = False  # Guards the slider callback while the slider is updated in code


def on_rate(label: str) -> None:
    sampler.set_rate(float(label.split()[0]))


def on_window(label: str) -> None:
    seconds = WINDOW_LENGTHS[[format_window(s) for s in WINDOW_LENGTHS].index(label)]
    view.set_window(seconds)
    window_box.show(seconds)


def on_pan(value: float) -> None:
    if _syncing:
        return
    view.right_edge = value
    # Snapping to the right edge of the slider resumes live following.
    view.following = value >= pan_slider.valmax - 1e-9


def on_key(event) -> None:
    # Arrow keys belong to whichever text box is being edited, if any.
    if event.key not in ("left", "right") or any(
            getattr(box, "capturekeystrokes", False) for box in TEXT_BOXES):
        return
    view.pan(-0.25 * view.window if event.key == "left" else 0.25 * view.window)


rate_buttons.on_clicked(on_rate)
window_buttons.on_clicked(on_window)
pan_slider.on_changed(on_pan)
fig.canvas.mpl_connect("key_press_event", on_key)


def sync_slider(oldest: float, newest: float) -> None:
    """Keep the slider range in step with the history that exists so far."""
    global _syncing
    low = min(oldest + view.window, newest)
    _syncing = True
    pan_slider.valmin, pan_slider.valmax = low, max(newest, low + 1e-6)
    pan_slider.ax.set_xlim(pan_slider.valmin, pan_slider.valmax)
    pan_slider.set_val(view.right_edge)
    pan_slider.valtext.set_text("live" if view.following
                                else f"-{format_elapsed(newest - view.right_edge)}")
    _syncing = False


def update_plot(_frame):
    oldest, newest = sampler.history.span()
    start, end = view.bounds(oldest, newest)
    data = sampler.history.view(start, end)
    if data["t"].size < 2:
        return  # Wait until the visible window holds something to draw.

    sync_slider(oldest, newest)
    t = data["t"]
    status = "live" if view.following else "PAUSED"

    # ---- 1. Utilization ----
    fig.suptitle(f"{sampler.name}  -  {sampler.rate_hz:g} Hz  -  "
                 f"{format_window(view.window)} window  -  {status}", fontsize=11)

    ax_util.clear()
    base = util_limits.apply(ax_util)
    ax_util.set_ylabel("Utilization (%)")
    draw_series(ax_util, t, data["sm"], COLOR_SM, "GPU-Utilization (sm)", base)
    draw_series(ax_util, t, data["bus"], COLOR_BUS, "Bus-Utilization (mem)", base)
    draw_series(ax_util, t, data["fb"], COLOR_FB, "Memory-Utilization (fb)", base)
    legend_above(ax_util, ncol=3)
    ax_util.grid(True, linestyle="--", alpha=0.3)

    # ---- 2. Temperature (left axis) and power (right axis) ----
    ax_temp.clear()
    ax_power.clear()
    base = temp_limits.apply(ax_temp)
    ax_temp.set_ylabel("Temperature (C)")
    handles = [draw_series(ax_temp, t, data["gtemp"], COLOR_GTEMP, "Core-Temp (gtemp)", base)]
    if sampler.has_mtemp:
        handles.append(draw_series(ax_temp, t, data["mtemp"], COLOR_MTEMP,
                                   "Memory-Temp (mtemp)", base))

    # The three thresholds behave very differently: the boost target only trims
    # clocks gradually, slowdown is the hard hardware clamp, shutdown is the
    # emergency stop.
    draw_limit(ax_temp, sampler.temp_target, "Boost target (GPU Target Temperature)",
               "#FFD700", x=0.01)
    draw_limit(ax_temp, sampler.temp_slowdown, "HW slowdown (GPU Slowdown Temp)",
               "#FF6347", "--", x=0.34)
    draw_limit(ax_temp, sampler.temp_shutdown, "Shutdown (GPU Shutdown Temp)",
               "#FF0000", "-", x=0.67)

    base = power_limits.apply(ax_power)
    # clear() resets a twin axis back to the left-hand side.
    ax_power.yaxis.set_label_position("right")
    ax_power.yaxis.tick_right()
    ax_power.set_ylabel("Power (W)")
    handles.append(draw_series(ax_power, t, data["power"], COLOR_POWER, "Power-Draw (pwr)", base))
    draw_limit(ax_power, sampler.power_limit, "Power cap (enforced.power.limit)",
               "#7CFC00", "--", x=0.99, ha="right", va="top")

    # Both axes feed a single legend, otherwise the two would overlap.
    legend_above(ax_temp, ncol=3, handles=handles)
    ax_temp.grid(True, linestyle="--", alpha=0.3)

    # ---- 3. PCIe bandwidth ----
    ax_bw.clear()
    # RX bursts reach the GB/s range on this bus, so the default ceiling is set
    # well above the old fixed 500 MB/s that used to clip them.
    base = bw_limits.apply(ax_bw)
    ax_bw.set_ylabel("PCIe bandwidth (MB/s)")
    ax_bw.set_xlabel("Elapsed time")
    draw_series(ax_bw, t, data["rx"], COLOR_RX, "RX-Bandwidth (rxpci)", base)
    draw_series(ax_bw, t, data["tx"], COLOR_TX, "TX-Bandwidth (txpci)", base)
    legend_above(ax_bw, ncol=2)
    ax_bw.grid(True, linestyle="--", alpha=0.3)

    ax_bw.set_xlim(start, end)
    ax_bw.xaxis.set_major_formatter(FuncFormatter(format_elapsed))
    # Only the bottom plot of the shared-x column carries tick labels.
    for ax in (ax_util, ax_temp):
        ax.tick_params(labelbottom=False)


def main() -> None:
    print(f"Monitoring {sampler.name} at {sampler.rate_hz:g} Hz")
    print(f"  power cap        : {sampler.power_limit:.0f} W")
    print(f"  boost target     : {sampler.temp_target:.0f} C")
    print(f"  hardware slowdown: {sampler.temp_slowdown:.0f} C")
    print(f"  shutdown         : {sampler.temp_shutdown:.0f} C")
    if not sampler.has_mtemp:
        print("  memory temp (mtemp): not exposed by this driver/board, curve disabled")
    print(f"  logging to       : {sampler.log.path}")

    # Kept in a local to stop the animation from being garbage collected.
    ani = animation.FuncAnimation(fig, update_plot, interval=PLOT_INTERVAL_MS,
                                  cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        del ani
        sampler.log.close()
        nvml.nvmlShutdown()


if __name__ == "__main__":
    main()
