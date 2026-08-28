"""Web front-end: a Dash dashboard served over HTTP.

Built for a machine reached over SSH, where no X display is available: the
monitor runs headless on the GPU host and the plots are opened in a browser
through an SSH tunnel.

Long windows are reduced server-side by `gpu_core.envelope`, so the browser
only ever receives a couple of thousand points per curve no matter how much
history is on screen, and single-sample glitches survive the reduction.
"""

import math

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from gpu_core import (COLOR_BUS, COLOR_FB, COLOR_GTEMP, COLOR_MTEMP, COLOR_POWER, COLOR_RX,
                      COLOR_SM, COLOR_TX, DEFAULT_BW_AXIS, FILL_ALPHA, MARKER_MAX_POINTS,
                      RAW_LIGHTEN,
                      DEFAULT_UTIL_AXIS, DEFAULT_WINDOW_SECONDS, MAX_HISTORY_SECONDS,
                      SAMPLE_RATES, SMOOTH_MAX_POINTS,
                      default_power_axis, default_temp_axis, envelope, format_elapsed,
                      build_id, format_limits, format_window, parse_duration,
                      parse_limits)

WEB_INTERVAL_MS = 1000     # How often the browser asks for a new figure
PLOT_HEIGHT = 1080         # Total figure height in pixels
PLOT_SPACING = 0.13        # Gap between the stacked plots, as a fraction of the height

# Y axes of the three rows, in the order of the limit boxes. Row 2 carries two:
# temperature on the primary axis and power on the secondary one.
AXIS_OF_BOX = ("yaxis", "yaxis2", "yaxis3", "yaxis4")
X_AXES = ("xaxis", "xaxis2", "xaxis3")

# What each y axis draws, so that "Autoscale" can be answered from the data.
# The toolbar only reports that autorange was switched on, never the bounds it
# arrived at, so they have to be worked out here.
METRICS_OF_AXIS = {"yaxis": ("sm", "bus", "fb"), "yaxis2": ("gtemp", "mtemp"),
                   "yaxis3": ("power",), "yaxis4": ("rx", "tx")}
AUTOSCALE_MARGIN = 0.05
ZOOM_STEP = 2.0            # How much one Zoom in / Zoom out press changes the window
X_TICKS = 8                # Tick count on the shared time axis

PAGE_STYLE = {"backgroundColor": "#111111", "color": "#dddddd", "fontFamily": "sans-serif",
              "padding": "12px", "minHeight": "100vh"}
FIELD_STYLE = {"width": "100%", "backgroundColor": "#1c1c1c", "color": "#ffffff",
               "border": "1px solid #444", "borderRadius": "3px", "padding": "5px"}
CAPTION_STYLE = {"fontSize": "11px", "color": "#999999", "marginBottom": "3px"}
BUTTON_STYLE = {"backgroundColor": "#1c1c1c", "color": "#ffffff", "border": "1px solid #444",
                "borderRadius": "3px", "padding": "6px 18px", "cursor": "pointer"}


def _rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def fill_color(color: str, alpha: float = FILL_ALPHA) -> str:
    """Translucent version of a curve's color, for the area beneath it."""
    r, g, b = _rgb(color)
    return f"rgba({r},{g},{b},{alpha})"


def lighten(color: str, amount: float = RAW_LIGHTEN) -> str:
    """Blend a color towards white, for the markers on the real samples."""
    r, g, b = (int(c + (255 - c) * amount) for c in _rgb(color))
    return f"rgb({r},{g},{b})"


def autoscale_range(data, metrics):
    """Bounds that fit the visible samples of the given metrics, with margin."""
    columns = [data[name] for name in metrics if name in data and data[name].size]
    if not columns:
        return None
    values = np.concatenate(columns)
    values = values[np.isfinite(values)]      # mtemp is all-NaN when unsupported
    if not values.size:
        return None

    low, high = float(values.min()), float(values.max())
    if high <= low:
        high = low + 1.0
    pad = (high - low) * AUTOSCALE_MARGIN
    return math.floor((low - pad) * 100) / 100, math.ceil((high + pad) * 100) / 100


def legend_of(row: int) -> str:
    """Name of the legend that sits above the given row."""
    return "legend" if row == 1 else f"legend{row}"


def add_series(fig, row, x, y, color, label, secondary_y=False):
    """Add one metric: line, shaded area, and markers on the real samples."""
    x, y = envelope(x, y)
    detailed = x.size <= MARKER_MAX_POINTS
    # Splines are SVG-only, and WebGL is what keeps a long window responsive,
    # so the smooth rendering is used exactly while the point count is small.
    trace = go.Scatter if x.size <= SMOOTH_MAX_POINTS else go.Scattergl
    shape = "spline" if trace is go.Scatter else "linear"
    fig.add_trace(
        trace(x=x, y=y, name=label, legend=legend_of(row),
              mode="lines+markers" if detailed else "lines",
              line=dict(color=color, width=2, shape=shape),
              marker=dict(color=lighten(color), size=4),
              fill="tozeroy", fillcolor=fill_color(color),
              hovertemplate=f"{label}: %{{y:.1f}}<extra></extra>"),
        row=row, col=1, secondary_y=secondary_y)


def add_limit(fig, row, value, label, color, secondary_y=False, position="top left"):
    """Draw a hardware limit as a labelled horizontal line."""
    if math.isnan(value):
        return
    fig.add_hline(y=value, line=dict(color=color, width=1, dash="dot"),
                  annotation_text=label, annotation_position=position,
                  annotation_font=dict(size=10, color=color),
                  row=row, col=1, secondary_y=secondary_y)


def build_figure(sampler, view, limits) -> go.Figure:
    """Render the three stacked plots for the window currently in view."""
    oldest, newest = sampler.history.span()
    start, end = view.bounds(oldest, newest)
    data = sampler.history.view(start, end)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=PLOT_SPACING,
                        specs=[[{}], [{"secondary_y": True}], [{}]])
    if data["t"].size >= 2:
        t = data["t"]
        add_series(fig, 1, t, data["sm"], COLOR_SM, "GPU-Utilization (sm)")
        add_series(fig, 1, t, data["bus"], COLOR_BUS, "Bus-Utilization (mem)")
        add_series(fig, 1, t, data["fb"], COLOR_FB, "Memory-Utilization (fb)")

        add_series(fig, 2, t, data["gtemp"], COLOR_GTEMP, "Core-Temp (gtemp)")
        if sampler.has_mtemp:
            add_series(fig, 2, t, data["mtemp"], COLOR_MTEMP, "Memory-Temp (mtemp)")
        add_series(fig, 2, t, data["power"], COLOR_POWER, "Power-Draw (pwr)", secondary_y=True)

        add_series(fig, 3, t, data["rx"], COLOR_RX, "RX-Bandwidth (rxpci)")
        add_series(fig, 3, t, data["tx"], COLOR_TX, "TX-Bandwidth (txpci)")

    # The three thresholds behave very differently: the boost target only trims
    # clocks gradually, slowdown is the hard hardware clamp, shutdown is the
    # emergency stop.
    add_limit(fig, 2, sampler.temp_target, f"Boost target ({sampler.temp_target:.0f}C)",
              "#FFD700", position="bottom left")
    add_limit(fig, 2, sampler.temp_slowdown, f"Slowdown ({sampler.temp_slowdown:.0f}C)",
              "#FF6347", position="top left")
    add_limit(fig, 2, sampler.temp_shutdown, f"Shutdown ({sampler.temp_shutdown:.0f}C)",
              "#FF0000", position="top right")
    add_limit(fig, 2, sampler.power_limit, f"Power cap ({sampler.power_limit:.0f}W)", "#7CFC00",
              secondary_y=True, position="bottom right")

    # Axes are pinned to the values in the boxes and never rescale themselves.
    # autorange has to be turned off explicitly: setting a range does not clear
    # it, and while it is on plotly overrides the range sent here and keeps
    # reporting autorange, which had the boxes recomputed on every refresh.
    fig.update_yaxes(range=list(limits["util"]), autorange=False,
                     title_text="Utilization (%)", row=1, col=1)
    fig.update_yaxes(range=list(limits["temp"]), autorange=False,
                     title_text="Temperature (C)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(range=list(limits["power"]), autorange=False, tickmode="sync",
                     title_text="Power (W)", row=2, col=1, secondary_y=True)
    fig.update_yaxes(range=list(limits["bw"]), autorange=False,
                     title_text="PCIe bandwidth (MB/s)", row=3, col=1)

    ticks = np.linspace(start, end, X_TICKS)
    fig.update_xaxes(range=[start, end], autorange=False, tickvals=ticks,
                     ticktext=[format_elapsed(v) for v in ticks])
    fig.update_xaxes(title_text="Elapsed time", row=3, col=1)
    # uirevision is deliberately set on the legend alone. On the layout it also
    # covers the axes, and an unchanged value makes plotly.js keep the ranges it
    # already has and discard the ones sent here: the window stops advancing and
    # the plots look frozen while the data keeps scrolling out of view.
    fig.update_layout(template="plotly_dark", height=PLOT_HEIGHT, hovermode="x unified",
                      margin=dict(l=70, r=70, t=40, b=40))

    # One legend per row, parked just above the plot it describes. The rows own
    # unrelated quantities, so a single shared legend would force the reader to
    # work out which of eight entries belongs to the plot being read.
    for row, axis in enumerate(("yaxis", "yaxis2", "yaxis4"), start=1):
        top = fig.layout[axis].domain[1]
        fig.update_layout({legend_of(row): dict(
            orientation="h", x=0.0, xanchor="left", y=top + 0.012, yanchor="bottom",
            font=dict(size=11), bgcolor="rgba(0,0,0,0)", uirevision="legend")})
    return fig


def build_layout(sampler, view):
    column = {"flex": "1", "minWidth": "150px"}
    return html.Div(style=PAGE_STYLE, children=[
        html.Div(style={"display": "flex", "alignItems": "baseline", "gap": "14px"}, children=[
            html.H3(sampler.name, style={"margin": "0 0 8px 0"}),
            html.Span(id="status", style={"color": "#888888", "fontSize": "13px"}),
            # Static: it identifies the running process, so a stale server that
            # was never restarted is visible without reading the terminal.
            html.Span(build_id(), style={"color": "#666666", "fontSize": "11px",
                                         "marginLeft": "auto"}),
        ]),
        html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                        "marginBottom": "10px"}, children=[
            html.Div(style=column, children=[
                html.Div("Sample rate", style=CAPTION_STYLE),
                dcc.Dropdown(id="rate", value=sampler.rate_hz, clearable=False,
                             options=[{"label": f"{hz:g} Hz", "value": hz} for hz in SAMPLE_RATES]),
            ]),
            html.Div(style=column, children=[
                html.Div("Window (<num>s/m/h)", style=CAPTION_STYLE),
                dcc.Input(id="window", value=format_window(view.window), debounce=True,
                          style=FIELD_STYLE)]),
            html.Div(style=column, children=[
                html.Div("Utilization y (%)", style=CAPTION_STYLE),
                dcc.Input(id="ylim-util", debounce=True, style=FIELD_STYLE,
                          value=format_limits(*DEFAULT_UTIL_AXIS))]),
            html.Div(style=column, children=[
                html.Div("Temperature y (C)", style=CAPTION_STYLE),
                dcc.Input(id="ylim-temp", debounce=True, style=FIELD_STYLE,
                          value=format_limits(*default_temp_axis(sampler)))]),
            html.Div(style=column, children=[
                html.Div("Power y (W)", style=CAPTION_STYLE),
                dcc.Input(id="ylim-power", debounce=True, style=FIELD_STYLE,
                          value=format_limits(*default_power_axis(sampler)))]),
            html.Div(style=column, children=[
                html.Div("PCIe y (MB/s)", style=CAPTION_STYLE),
                dcc.Input(id="ylim-bw", debounce=True, style=FIELD_STYLE,
                          value=format_limits(*DEFAULT_BW_AXIS))]),
        ]),
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px"}, children=[
            html.Button("Pause", id="playpause", n_clicks=0, style=BUTTON_STYLE),
            html.Button("Zoom in", id="zoom-in", n_clicks=0, style=BUTTON_STYLE),
            html.Button("Zoom out", id="zoom-out", n_clicks=0, style=BUTTON_STYLE),
            html.Button("Autoscale: off", id="autoscale", n_clicks=0, style=BUTTON_STYLE),
            html.Button("Reset axes", id="reset", n_clicks=0, style=BUTTON_STYLE),
            html.Div("Pan (0 = live; the slider also moves with the arrow keys)",
                     style={**CAPTION_STYLE, "marginBottom": "0"}),
        ]),
        dcc.Slider(id="pan", min=-1, max=0, value=0, step=0.1, marks=None,
                   tooltip={"placement": "bottom"}),
        # The modebar is replaced by the buttons above, but the interactions it
        # used to host stay: drag to box zoom, drag an axis to pan it, wheel to
        # zoom. None of those live in the modebar.
        dcc.Graph(id="graph", config={"displayModeBar": False, "scrollZoom": True,
                                      "displaylogo": False}),
        dcc.Interval(id="tick", interval=WEB_INTERVAL_MS),
        dcc.Store(id="auto", data=False),
    ])


def run(sampler, view, host: str, port: int) -> None:
    """Serve the dashboard and block until interrupted."""
    # update_title defaults to "Updating...", which would rewrite the browser tab
    # title on every single refresh.
    app = Dash(__name__, title=f"GPU Monitor - {sampler.name}", update_title=None)
    app.layout = build_layout(sampler, view)

    @app.callback(
        Output("tick", "disabled"), Output("playpause", "children"), Output("pan", "value"),
        Input("playpause", "n_clicks"), State("pan", "value"))
    def toggle_autoupdate(clicks, pan):
        """Stop the refresh loop so the window holds still while navigating."""
        paused = bool(clicks) and clicks % 2 == 1
        # Resuming snaps the pan slider back to the live edge, otherwise the
        # button would look dead: the window it froze would stay where it was.
        return paused, ("Play" if paused else "Pause"), (pan if paused else 0)

    @app.callback(
        Output("ylim-util", "value"), Output("ylim-temp", "value"),
        Output("ylim-power", "value"), Output("ylim-bw", "value"), Output("window", "value"),
        Input("graph", "relayoutData"),
        State("ylim-util", "value"), State("ylim-temp", "value"),
        State("ylim-power", "value"), State("ylim-bw", "value"), State("window", "value"),
        prevent_initial_call=True)
    def adopt_mouse_ranges(relayout, *shown):
        """Copy a range dragged with the mouse into the box that owns that axis.

        The boxes stay the single source of truth, so a zoom only survives the
        next refresh once it lands in them. A value that already matches is
        returned as no_update, which is also what stops this from feeding itself
        through the redraw it triggers.
        """
        if not relayout:
            raise PreventUpdate

        visible = None  # Read only if an autoscale actually needs the data.
        updates = []
        for axis, current in zip(AXIS_OF_BOX, shown):
            low, high = relayout.get(f"{axis}.range[0]"), relayout.get(f"{axis}.range[1]")
            if low is not None and high is not None:
                text = format_limits(float(low), float(high))
            elif relayout.get(f"{axis}.autorange"):
                if visible is None:
                    visible = sampler.history.view(view.right_edge - view.window,
                                                   view.right_edge)
                bounds = autoscale_range(visible, METRICS_OF_AXIS[axis])
                text = format_limits(*bounds) if bounds else None
            else:
                text = None
            updates.append(text if text is not None and text != current else no_update)

        # A drag on any of the shared time axes becomes the new window width;
        # autoscaling it means showing every sample held.
        dragged = next((a for a in X_AXES if f"{a}.range[0]" in relayout), None)
        if dragged is not None:
            span = float(relayout[f"{dragged}.range[1]"]) - float(relayout[f"{dragged}.range[0]"])
        elif any(relayout.get(f"{axis}.autorange") for axis in X_AXES):
            oldest, newest = sampler.history.span()
            span = newest - oldest
        else:
            span = None

        if span is None:
            updates.append(no_update)
        else:
            text = format_window(min(max(span, 1.0), MAX_HISTORY_SECONDS))
            updates.append(text if text != shown[-1] else no_update)

        if all(update is no_update for update in updates):
            raise PreventUpdate
        return updates

    @app.callback(
        Output("window", "value", allow_duplicate=True),
        Input("zoom-in", "n_clicks"), Input("zoom-out", "n_clicks"),
        State("window", "value"), prevent_initial_call=True)
    def zoom_time_axis(_in, _out, window):
        """Widen or narrow the time window.

        Zooming here is deliberately one-dimensional: the y axes are pinned to
        the boxes, and a zoom that moved them too would silently overwrite
        limits that were set on purpose.
        """
        try:
            current = parse_duration(window)
        except (ValueError, TypeError):
            current = DEFAULT_WINDOW_SECONDS
        factor = 1 / ZOOM_STEP if ctx.triggered_id == "zoom-in" else ZOOM_STEP
        return format_window(min(max(current * factor, 1.0), MAX_HISTORY_SECONDS))

    @app.callback(
        Output("auto", "data"), Output("autoscale", "children"),
        Input("autoscale", "n_clicks"), prevent_initial_call=True)
    def toggle_autoscale(clicks):
        """Switch continuous y autoscaling on and off."""
        on = bool(clicks) and clicks % 2 == 1
        return on, f"Autoscale: {'on' if on else 'off'}"

    @app.callback(
        Output("ylim-util", "value", allow_duplicate=True),
        Output("ylim-temp", "value", allow_duplicate=True),
        Output("ylim-power", "value", allow_duplicate=True),
        Output("ylim-bw", "value", allow_duplicate=True),
        Input("tick", "n_intervals"), Input("auto", "data"),
        State("ylim-util", "value"), State("ylim-temp", "value"),
        State("ylim-power", "value"), State("ylim-bw", "value"),
        prevent_initial_call=True)
    def apply_autoscale(_tick, on, *shown):
        """While autoscaling, keep refitting the y axes through the boxes.

        The bounds are written into the boxes rather than applied behind them,
        so the boxes never stop describing what is on screen, and switching
        autoscale off just leaves the last fit in place.
        """
        if not on:
            raise PreventUpdate
        visible = sampler.history.view(view.right_edge - view.window, view.right_edge)
        updates = []
        for axis, current in zip(AXIS_OF_BOX, shown):
            bounds = autoscale_range(visible, METRICS_OF_AXIS[axis])
            text = format_limits(*bounds) if bounds else None
            updates.append(text if text is not None and text != current else no_update)
        if all(update is no_update for update in updates):
            raise PreventUpdate
        return updates

    @app.callback(
        Output("ylim-util", "value", allow_duplicate=True),
        Output("ylim-temp", "value", allow_duplicate=True),
        Output("ylim-power", "value", allow_duplicate=True),
        Output("ylim-bw", "value", allow_duplicate=True),
        Output("window", "value", allow_duplicate=True),
        Output("pan", "value", allow_duplicate=True),
        Input("reset", "n_clicks"), prevent_initial_call=True)
    def reset_axes(_clicks):
        """Put every axis, the window and the pan back to the opening values."""
        return (format_limits(*DEFAULT_UTIL_AXIS), format_limits(*default_temp_axis(sampler)),
                format_limits(*default_power_axis(sampler)), format_limits(*DEFAULT_BW_AXIS),
                format_window(DEFAULT_WINDOW_SECONDS), 0)

    @app.callback(
        Output("graph", "figure"), Output("pan", "min"), Output("status", "children"),
        Input("tick", "n_intervals"), Input("tick", "disabled"), Input("rate", "value"),
        Input("window", "value"), Input("pan", "value"), Input("ylim-util", "value"),
        Input("ylim-temp", "value"), Input("ylim-power", "value"), Input("ylim-bw", "value"))
    def refresh(_tick, paused, rate, window, pan, util_text, temp_text, power_text, bw_text):
        if rate and rate != sampler.rate_hz:
            sampler.set_rate(float(rate))
        try:
            view.set_window(min(max(parse_duration(window), 1.0), MAX_HISTORY_SECONDS))
        except (ValueError, TypeError):
            pass  # Half-typed entry: keep the window already in force.

        oldest, newest = sampler.history.span()
        # The slider counts seconds behind the live edge, so 0 always means
        # "live" however much history has piled up behind it.
        reach = -max(0.0, (newest - oldest) - view.window)
        # While paused the view is only re-anchored by the slider itself:
        # editing a limit must not drag the window back to the live edge.
        if not paused or ctx.triggered_id == "pan":
            view.right_edge = newest + min(0.0, float(pan or 0.0))
        view.following = (pan or 0.0) >= 0.0 and not paused

        limits = {"util": parse_limits(util_text, (0.0, 105.0)),
                  "temp": parse_limits(temp_text, default_temp_axis(sampler)),
                  "power": parse_limits(power_text, default_power_axis(sampler)),
                  "bw": parse_limits(bw_text, DEFAULT_BW_AXIS)}
        figure = build_figure(sampler, view, limits)
        # t= is a heartbeat: it advances on every refresh, so a frozen display
        # is obvious at a glance instead of being guesswork.
        status = (f"{sampler.rate_hz:g} Hz - {format_window(view.window)} window - "
                  + ("PAUSED" if paused else "live" if view.following
                     else f"browsing at -{format_elapsed(newest - view.right_edge)}")
                  + f" - t={format_elapsed(newest)}")
        return figure, reach, status

    print(f"  dashboard on     : http://{host}:{port}")
    if host in ("127.0.0.1", "localhost"):
        print(f"  over SSH, tunnel : ssh -N -L {port}:localhost:{port} <this-host>")
    app.run(host=host, port=port)
