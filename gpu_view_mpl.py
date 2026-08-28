"""Matplotlib front-end: a live window with radio buttons, text boxes and a pan slider."""

import math
from types import SimpleNamespace

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox

from gpu_core import (COLOR_BUS, COLOR_FB, COLOR_GTEMP, COLOR_MTEMP, COLOR_POWER, COLOR_RX,
                      COLOR_SM, COLOR_TX, DEFAULT_BW_AXIS, DEFAULT_RATE_HZ, DEFAULT_UTIL_AXIS,
                      DEFAULT_WINDOW_SECONDS, FILL_ALPHA,
                      MARKER_MAX_POINTS, MAX_HISTORY_SECONDS, PLOT_INTERVAL_MS, RAW_LIGHTEN,
                      SAMPLE_RATES, SMOOTHING_STEPS, SMOOTH_MAX_POINTS, WINDOW_LENGTHS,
                      default_power_axis, default_temp_axis, envelope, format_elapsed,
                      format_limits, format_window, parse_duration, parse_limits)

def lighten(color: str, amount: float = RAW_LIGHTEN) -> tuple[float, float, float]:
    """Blend a color towards white, for the raw-sample line."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1.0 - r) * amount, g + (1.0 - g) * amount, b + (1.0 - b) * amount)


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


def legend_above(ax, ncol: int, handles=None):
    """Put a legend in the margin above the axes, where it hides no data."""
    labels = [h.get_label() for h in handles] if handles else None
    args = (handles, labels) if handles else ()
    ax.legend(*args, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=ncol,
              fontsize=8, frameon=False, borderaxespad=0.0)


def run(sampler, view, show: bool = True):
    """Open the live window and block until it is closed.

    With `show=False` the figure is built and drawn once, then returned along
    with its controls instead of being displayed: that renders a frame to a
    file without a display, and is what the tests drive.
    """
    plt.style.use("dark_background")
    # The arrow keys pan the time window, so they are taken back from matplotlib's
    # default "previous/next view" bindings.
    for key, binding in (("left", "keymap.back"), ("right", "keymap.forward")):
        plt.rcParams[binding] = [k for k in plt.rcParams[binding] if k != key]

    fig = plt.figure(figsize=(14, 11))
    fig.canvas.manager.set_window_title(f"NVIDIA GPU Monitor - {sampler.name}")
    grid = fig.add_gridspec(3, 1, left=0.07, right=0.75, top=0.94, bottom=0.11, hspace=0.55)
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
        """Fixed "min;max" bounds for one y axis, edited through a text box.

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
            return format_limits(self.low, self.high)

        def _on_submit(self, text: str) -> None:
            if self._syncing:
                return
            # An unparsable entry falls back to the bounds already in force.
            self.set_bounds(*parse_limits(text, (self.low, self.high)), force=True)

        def set_bounds(self, low: float, high: float, force: bool = False) -> None:
            """Adopt a range and show it in the box, without re-entering submit."""
            if not force and (low, high) == (self.low, self.high):
                return
            self.low, self.high = float(low), float(high)
            self._syncing = True
            self.box.set_val(self._text())
            self._syncing = False
            state.dirty = True

        def adopt(self, ax) -> None:
            """Copy a range the mouse changed on `ax` back into the box."""
            if not state.redrawing:
                self.set_bounds(*ax.get_ylim())

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
                state.dirty = True
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
    util_limits = LimitControl(BOX_RECTS[1], "Utilization y (%)", *DEFAULT_UTIL_AXIS)
    temp_limits = LimitControl(BOX_RECTS[2], "Temperature y (C)", *default_temp_axis(sampler))
    power_limits = LimitControl(BOX_RECTS[3], "Power y (W)", *default_power_axis(sampler))
    bw_limits = LimitControl(BOX_RECTS[4], "PCIe y (MB/s)", *DEFAULT_BW_AXIS)
    TEXT_BOXES = (window_box.box, util_limits.box, temp_limits.box, power_limits.box, bw_limits.box)

    # Autoupdate can be stopped while navigating, so the window stops sliding
    # out from under the region being examined. Sampling and logging carry on.
    state = SimpleNamespace(paused=False, dirty=True, redrawing=False)

    ax_pause = fig.add_axes((0.020, 0.033, 0.055, 0.030))
    pause_button = Button(ax_pause, "Pause", color="#1c1c1c", hovercolor="#2a2a2a")
    pause_button.label.set_fontsize(8)
    pause_button.label.set_color("white")

    def on_pause(_event) -> None:
        state.paused = not state.paused
        pause_button.label.set_text("Play" if state.paused else "Pause")
        # Resuming returns to the live edge, otherwise the button would look
        # dead: the window it froze would simply stay where it was left.
        view.following = not state.paused
        state.dirty = True

    pause_button.on_clicked(on_pause)

    ax_slider = fig.add_axes((0.13, 0.04, 0.62, 0.025))
    pan_slider = Slider(ax_slider, "Pan", 0.0, 1.0, valinit=1.0)
    pan_slider.valtext.set_fontsize(8)
    pan_slider.label.set_fontsize(8)

    _syncing = False  # Guards the slider callback while the slider is updated in code


    def on_rate(label: str) -> None:
        sampler.set_rate(float(label.split()[0]))
        state.dirty = True


    def on_window(label: str) -> None:
        seconds = WINDOW_LENGTHS[[format_window(s) for s in WINDOW_LENGTHS].index(label)]
        view.set_window(seconds)
        window_box.show(seconds)
        state.dirty = True


    def on_pan(value: float) -> None:
        if _syncing:
            return
        view.right_edge = value
        # Snapping to the right edge of the slider resumes live following, unless
        # autoupdate is off.
        view.following = value >= pan_slider.valmax - 1e-9 and not state.paused
        state.dirty = True


    def on_key(event) -> None:
        # Arrow keys belong to whichever text box is being edited, if any.
        if event.key not in ("left", "right") or any(
                getattr(box, "capturekeystrokes", False) for box in TEXT_BOXES):
            return
        view.pan(-0.25 * view.window if event.key == "left" else 0.25 * view.window)
        state.dirty = True


    rate_buttons.on_clicked(on_rate)
    window_buttons.on_clicked(on_window)
    pan_slider.on_changed(on_pan)
    fig.canvas.mpl_connect("key_press_event", on_key)


    def sync_slider(oldest: float, newest: float) -> None:
        """Keep the slider range in step with the history that exists so far."""
        nonlocal _syncing
        # The slider spans the oldest full window up to live, widened when the
        # view is held further back than that.
        low = min(oldest + view.window, newest, view.right_edge)
        _syncing = True
        pan_slider.valmin, pan_slider.valmax = low, max(newest, low + 1e-6)
        pan_slider.ax.set_xlim(pan_slider.valmin, pan_slider.valmax)
        pan_slider.set_val(view.right_edge)
        pan_slider.valtext.set_text("live" if view.following
                                    else f"-{format_elapsed(newest - view.right_edge)}")
        _syncing = False


    def adopt_xlim(ax) -> None:
        """A mouse zoom on the time axis becomes the new window width."""
        if state.redrawing:
            return
        start, end = ax.get_xlim()
        width = max(end - start, 1.0)
        if abs(width - view.window) < 1e-6:
            return
        view.set_window(min(width, MAX_HISTORY_SECONDS))
        view.right_edge = end
        view.following = False
        window_box.show(view.window)
        state.dirty = True


    def capture_axis_limits() -> None:
        """Re-arm the handlers that copy mouse-driven ranges into the boxes.

        Axes.clear() drops its callback registry, so this runs once per frame,
        after the axes have been redrawn from the values already in the boxes.
        Clearing the flag here is what ends the redraw: until then the handlers
        ignore everything, because clearing one axes resets the limits of every
        axes sharing its x, which would otherwise read as a mouse zoom.
        """
        for ax, control in ((ax_util, util_limits), (ax_temp, temp_limits),
                            (ax_power, power_limits), (ax_bw, bw_limits)):
            ax.callbacks.connect("ylim_changed", control.adopt)
        ax_bw.callbacks.connect("xlim_changed", adopt_xlim)
        state.redrawing = False


    def update_plot(_frame):
        if state.paused and not state.dirty:
            return  # Autoupdate is off: nothing moves until a control changes.
        state.dirty = False
        if state.paused:
            view.following = False

        oldest, newest = sampler.history.span()
        start, end = view.bounds(oldest, newest)
        data = sampler.history.view(start, end)
        if data["t"].size < 2:
            return  # Wait until the visible window holds something to draw.

        state.redrawing = True
        sync_slider(oldest, newest)
        t = data["t"]
        status = "PAUSED" if state.paused else "live" if view.following else "browsing"

        # ---- 1. Utilization ----
        # t= is a heartbeat: it advances on every redraw, so a frozen display is
        # obvious at a glance instead of being guesswork.
        fig.suptitle(f"{sampler.name}  -  {sampler.rate_hz:g} Hz  -  "
                     f"{format_window(view.window)} window  -  {status}"
                     f"  -  t={format_elapsed(newest)}", fontsize=11)

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
        # Short labels in their own horizontal slots: the full nvidia-smi
        # field names are wide enough to collide with each other.
        draw_limit(ax_temp, sampler.temp_target, f"Boost target ({sampler.temp_target:.0f}C)",
                   "#FFD700", x=0.01)
        draw_limit(ax_temp, sampler.temp_slowdown, f"Slowdown ({sampler.temp_slowdown:.0f}C)",
                   "#FF6347", "--", x=0.30)
        draw_limit(ax_temp, sampler.temp_shutdown, f"Shutdown ({sampler.temp_shutdown:.0f}C)",
                   "#FF0000", "-", x=0.58, va="top")

        base = power_limits.apply(ax_power)
        # clear() resets a twin axis back to the left-hand side.
        ax_power.yaxis.set_label_position("right")
        ax_power.yaxis.tick_right()
        ax_power.set_ylabel("Power (W)")
        handles.append(draw_series(ax_power, t, data["power"], COLOR_POWER, "Power-Draw (pwr)", base))
        draw_limit(ax_power, sampler.power_limit, f"Power cap ({sampler.power_limit:.0f}W)",
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
        capture_axis_limits()

    if not show:
        update_plot(0)
        return SimpleNamespace(fig=fig, update=update_plot, on_key=on_key, pan=pan_slider,
                               window_box=window_box, util=util_limits, temp=temp_limits,
                               power=power_limits, bw=bw_limits, state=state,
                               toggle=on_pause, axes=(ax_util, ax_temp, ax_power, ax_bw))

    # Kept in a local to stop the animation from being garbage collected.
    ani = animation.FuncAnimation(fig, update_plot, interval=PLOT_INTERVAL_MS,
                                  cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        del ani
