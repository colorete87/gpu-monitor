# gpu-monitor

Real-time monitor for a single NVIDIA GPU: utilization, thermals with power, and
PCIe bandwidth, sampled several times per second and logged to CSV.

One sampler feeds three interchangeable front-ends — a desktop window, a web
dashboard, and a headless logger — so the same tool works whether you are
sitting at the machine or reaching it over SSH.

## Running it

`start.sh` launches everything through [uv](https://docs.astral.sh/uv/) in an
ephemeral environment: nothing is installed system-wide, and each mode pulls
only the dependencies it needs.

```bash
./start.sh              # desktop window (matplotlib)
./start.sh --web        # dashboard on http://127.0.0.1:8050 (Dash)
./start.sh --no-gui     # sample and log only, no interface
```

Requires an NVIDIA driver (`nvidia-smi` on `PATH`) and `uv`.

### Over SSH

The desktop window needs X11 forwarding and ships a full bitmap per frame, which
crawls at 4 Hz. Prefer the dashboard and tunnel the port:

```bash
ssh -L 8050:localhost:8050 <host> '<path>/gpu-monitor/start.sh --web'
```

Then open <http://localhost:8050>. For an unattended run, `--no-gui` writes the
same CSV with no display at all.

### Options

```
--web                serve the dashboard instead of opening a window
--no-gui             only sample and log
--rate HZ            samples per second (default 4)
--window DURATION    visible window, e.g. "90", "10m", "6h" (default 60)
--gpu INDEX          which GPU to monitor (default 0)
--host / --port      web mode bind address (default 127.0.0.1:8050)
--quiet              --no-gui: drop the periodic status line
```

## What it plots

Legends carry the `nvidia-smi` name of each quantity in parentheses, so every
curve maps back to what `nvidia-smi dmon` and `nvidia-smi -q` report.

| Plot | Curves |
| --- | --- |
| Utilization (%) | `sm` compute, `mem` memory-controller, `fb` frame buffer as a share of total VRAM |
| Temperature (°C) and power (W), twin axes | `gtemp` core, `mtemp` memory when the board exposes it, `pwr` draw |
| PCIe bandwidth (MB/s) | `rxpci`, `txpci` |

The thermal plot draws the board's own limits, read from the hardware at startup
rather than hardcoded: the GPU Boost target, the hardware slowdown threshold and
the shutdown threshold, plus the enforced power cap. These three temperatures
mean very different things — the boost target only trims clocks gradually and is
the one you will actually see act under sustained load, slowdown is the hard
clamp, shutdown is the emergency stop.

Most consumer boards do not expose the GDDR temperature at all. When
`nvidia-smi -q -d TEMPERATURE` reports `Memory Current Temp: N/A`, the `mtemp`
curve is dropped and the startup banner says so. On boards that do report it,
memory is often the real thermal bottleneck while the core looks comfortable.

## Controls

Both graphical front-ends share the same model: **the text boxes are the single
source of truth for every axis, and no axis ever rescales itself.** An axis that
follows its data hides exactly what a monitor is meant to show, since every
trace ends up filling the same height whatever the values are.

- **Axis limits** are typed as `min;max`. The separator is a semicolon because a
  comma is the decimal separator in many locales.
- **Window width** is typed as `<number><unit>`, with `s`, `m` or `h` — `90s`,
  `10m`, `6h`, up to 48 h.
- **Panning** uses the slider at the bottom. In the desktop window the left and
  right arrow keys pan a quarter window per press; in the browser the slider
  takes the arrow keys once it has focus. Reaching the live edge resumes
  following the latest samples.
- **Pause** stops the display from advancing while you examine something.
  Sampling and logging carry on regardless.
- **Zoom in / Zoom out** halve and double the time window. They move the time
  axis only: the y axes are pinned to the boxes, and a zoom that moved them too
  would silently overwrite limits you set on purpose.
- **Autoscale** is a toggle. While it is on the y bounds are refitted to the
  visible samples and written into the boxes, so the boxes never stop describing
  what is on screen; switching it off leaves the last fit in place.
- **Reset axes** restores the defaults: the four y ranges, a one minute window,
  and the pan back to live.

Ranges changed with the mouse — drag to box zoom, drag an axis to pan it, wheel
to zoom — are copied back into the boxes, rounded outward to whole numbers so a
rounded bound never clips the samples it was meant to frame.

The web header shows a build id: the commit, a digest of the sources, and the
time this process started. A server left running from before an edit reports the
right commit and still runs the old code, so the start time is the field that
makes that visible.

## Logging

Every run writes `logs/gpu_monitor_<timestamp>.csv`, in every mode. The file is
line buffered, so a log left behind by a crash is still complete.

```
# gpu: NVIDIA GeForce RTX 3090
# started: 2026-08-28T07:20:45
# vram_total_mb: 24576
# power_cap_w: 370
# temp_target_c: 83
# temp_slowdown_c: 95
# temp_shutdown_c: 98
timestamp,elapsed_s,sm,mem,fb,gtemp,mtemp,pwr,rxpci,txpci
2026-08-28T07:20:45.115,0.002,100.000,65.000,34.578,87.000,,266.085,373.411,293.493
```

`fb` is a percentage of total VRAM, `pwr` is watts, `rxpci`/`txpci` are MB/s, and
an unsupported metric is left empty. The comment lines are readable by
`pandas.read_csv(path, comment="#")`. `logs/` is gitignored.

## How it works

**Sampling goes through NVML** (`nvidia-ml-py`) rather than `nvidia-smi dmon`.
It is the same telemetry — `nvidia-smi` is a front-end over `libnvidia-ml.so` —
without a text table to parse, and `dmon` cannot sample faster than once per
second, since its `-d` takes whole seconds. Note that the driver refreshes some
counters on its own internal window, so sampling at 10 Hz yields repeated values
in a staircase rather than more detail.

**Long windows are reduced by a min/max envelope.** 48 hours at 4 Hz is roughly
700k samples per curve, far more than the screen has pixels. Averaging them
would erase a one-sample glitch, which is often the thing worth seeing, so each
bucket contributes its minimum and its maximum in the order they were measured.
Both are real samples, and a single-sample spike still draws at full height
across a 48 hour window.

**History is a contiguous fixed-duration buffer.** When it fills, the oldest
eighth is dropped in one move, which keeps reads to a plain slice at the cost of
an occasional memmove — roughly every seven hours at 4 Hz.

## Layout

```
gpu_core.py       sampling, history, logging, envelope, formatting
gpu_view_mpl.py   desktop window
gpu_view_web.py   web dashboard
gpu_monitor.py    command line entry point
start.sh          uv launcher
```

`gpu_core.py` depends on nothing but numpy and NVML, so the front-ends are
independent of one another and are imported lazily: `--no-gui` pulls neither
matplotlib nor Dash, and `--web` never pulls matplotlib.
