#!/usr/bin/env bash
#
# start.sh - Launch the NVIDIA GPU monitor through uv.
#
# uv builds an ephemeral, isolated environment holding the required
# dependencies, so nothing has to be installed system-wide.

set -euo pipefail

# Resolve paths relative to this script so it can be launched from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="${SCRIPT_DIR}/gpu_monitor.py"

# Interpreter used for the ephemeral environment. uv downloads it on the
# first run if it is not already available locally. Overridable from the
# environment, e.g. PYTHON_VERSION=3.13 ./start.sh
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

# Runtime dependencies. Each front-end pulls only what it needs, so --no-gui
# stays light and the web mode does not drag in matplotlib.
DEPENDENCIES=(nvidia-ml-py numpy)
case " $* " in
    *" --no-gui "*) ;;
    *" --web "*)    DEPENDENCIES+=(dash plotly) ;;
    *)              DEPENDENCIES+=(matplotlib) ;;
esac

err() { printf 'error: %s\n' "$*" >&2; }
warn() { printf 'warning: %s\n' "$*" >&2; }

if ! command -v uv >/dev/null 2>&1; then
    err "uv is not installed: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

if [[ ! -f "${APP}" ]]; then
    err "cannot find ${APP}"
    exit 1
fi

# The monitor feeds on 'nvidia-smi dmon'; without it there is nothing to plot.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    err "nvidia-smi not found: an NVIDIA driver installation is required."
    exit 1
fi

# Only the desktop window needs a graphical session; --web and --no-gui do not.
if [[ " ${DEPENDENCIES[*]} " == *" matplotlib "* \
      && -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    warn "neither DISPLAY nor WAYLAND_DISPLAY is set; the plot window may fail to open."
    warn "over SSH, try './start.sh --web' and tunnel the port, or './start.sh --no-gui'."
fi

# Turn the dependency list into the repeated --with flags uv expects.
with_flags=()
for dep in "${DEPENDENCIES[@]}"; do
    with_flags+=(--with "${dep}")
done

# --no-project stops uv from picking up an unrelated pyproject.toml that may
# live in a parent directory. Extra arguments are forwarded to the script.
exec uv run --no-project \
    --python "${PYTHON_VERSION}" \
    "${with_flags[@]}" \
    "${APP}" "$@"
