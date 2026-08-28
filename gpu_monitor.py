"""NVIDIA GPU monitor.

One sampler, three front-ends:

    ./start.sh                 desktop window (matplotlib)
    ./start.sh --web           dashboard served over HTTP (Dash)
    ./start.sh --no-gui        sample and log only, no interface

Every mode writes the same timestamped CSV under `logs/`, so `--no-gui` is the
one to leave running on a headless box and plot afterwards.
"""

import argparse
import sys
import time

import pynvml as nvml

from gpu_core import (DEFAULT_RATE_HZ, DEFAULT_WINDOW_SECONDS, GPU_INDEX, MAX_HISTORY_SECONDS,
                      SAMPLE_RATES, GpuSampler, ViewState, build_id, format_elapsed,
                      parse_duration)

STATUS_INTERVAL_SECONDS = 2.0   # How often --no-gui prints a status line


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    front_end = parser.add_mutually_exclusive_group()
    front_end.add_argument("--web", action="store_true",
                           help="serve the dashboard over HTTP instead of opening a window")
    front_end.add_argument("--no-gui", dest="no_gui", action="store_true",
                           help="only sample and log, with no interface at all")

    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ, metavar="HZ",
                        help=f"samples per second (the selectors offer {SAMPLE_RATES})")
    parser.add_argument("--window", default=str(DEFAULT_WINDOW_SECONDS), metavar="DURATION",
                        help='width of the visible window, e.g. "90", "10m" or "6h"')
    parser.add_argument("--gpu", type=int, default=GPU_INDEX, metavar="INDEX",
                        help="which GPU to monitor")
    parser.add_argument("--host", default="127.0.0.1", help="web mode: address to bind")
    parser.add_argument("--port", type=int, default=8050, help="web mode: port to bind")
    parser.add_argument("--quiet", action="store_true",
                        help="--no-gui: do not print the periodic status line")
    return parser.parse_args()


def print_banner(sampler: GpuSampler) -> None:
    print(f"Monitoring {sampler.name} at {sampler.rate_hz:g} Hz")
    print(f"  power cap        : {sampler.power_limit:.0f} W")
    print(f"  boost target     : {sampler.temp_target:.0f} C")
    print(f"  hardware slowdown: {sampler.temp_slowdown:.0f} C")
    print(f"  shutdown         : {sampler.temp_shutdown:.0f} C")
    if not sampler.has_mtemp:
        print("  memory temp (mtemp): not exposed by this driver/board, curve disabled")
    print(f"  logging to       : {sampler.log.path}")
    print(f"  running          : {build_id()}")


def run_headless(sampler: GpuSampler, quiet: bool) -> None:
    """Keep sampling into the log, reporting the latest reading now and then."""
    print("  press Ctrl+C to stop")
    while True:
        time.sleep(STATUS_INTERVAL_SECONDS)
        if quiet:
            continue
        _, newest = sampler.history.span()
        latest = sampler.history.view(newest, newest)
        if latest["t"].size == 0:
            continue
        print(f"[{format_elapsed(newest):>8}] "
              f"sm {latest['sm'][-1]:3.0f}%  mem {latest['bus'][-1]:3.0f}%  "
              f"fb {latest['fb'][-1]:4.1f}%  |  {latest['gtemp'][-1]:3.0f}C  |  "
              f"{latest['power'][-1]:4.0f}W  |  "
              f"rx {latest['rx'][-1]:6.0f}  tx {latest['tx'][-1]:6.0f} MB/s")


def main() -> None:
    # Status lines must reach a pipe as they happen, not in block-buffered
    # bursts: `--no-gui | tee run.log` is the expected way to run this.
    sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    window = min(max(parse_duration(args.window), 1.0), MAX_HISTORY_SECONDS)

    sampler = GpuSampler(index=args.gpu)
    sampler.set_rate(args.rate)          # Sizes the history before sampling starts.
    view = ViewState(window)
    sampler.start()
    print_banner(sampler)

    try:
        # The front-ends are imported lazily, so a mode never pays for the
        # dependencies of the other two.
        if args.no_gui:
            run_headless(sampler, args.quiet)
        elif args.web:
            import gpu_view_web
            gpu_view_web.run(sampler, view, args.host, args.port)
        else:
            import gpu_view_mpl
            gpu_view_mpl.run(sampler, view)
    except KeyboardInterrupt:
        print()
    finally:
        sampler.log.close()
        nvml.nvmlShutdown()
        print(f"log written to {sampler.log.path}")


if __name__ == "__main__":
    main()
