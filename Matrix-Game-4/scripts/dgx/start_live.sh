#!/usr/bin/env bash
# Start, stop or inspect the live Matrix-Game stack on the DGX: MediaMTX (shared video relay, one instance for all scenes) and one
# `matrix-game serve` process per scene, each on its own fixed port, so more than one scene can run at once and you pick which to play
# by which port you open. Idempotent: a scene that is already listening is left alone; only stops processes this script itself started.
#
# Ports: racing 8025, castle 8026, ancient-egypt 8027 (MediaMTX/WHEP 8890 and ICE 8190 are shared by all three).
#
# Usage:   bash scripts/dgx/start_live.sh [start|stop|status] [--scene racing|castle|ancient-egypt] [--preset realtime|quality|rtx5090]
#                                         [--fps 24] [--prompt-mode fixed|dynamic]
#   start   launch whatever is not running yet for the given --scene (default racing) and wait until its page answers
#   stop    stop the given --scene's server (default: all three) plus MediaMTX, but only what this script started (pid files in runtime/live/)
#   status  show ports and pids for MediaMTX and all three scenes
# `--prompt-mode dynamic` is accepted by the server for racing only.
# Needs MediaMTX (~/bin/mediamtx or on PATH), GStreamer with the x264 and RTSP plugins, and the racing resources (see docs/dgx_racing_runs.md).
# Logs: runtime/live/mediamtx.log and runtime/live/serve-<scene>.log. The GPU is checked first: it refuses to start with less than
# MIN_FREE_MIB (default 30000) free.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

action="start"
if [[ $# -gt 0 && "$1" != --* ]]; then
  action="$1"
  shift
fi
scene=""
preset="realtime"
fps=24
prompt_mode="fixed"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene) scene="${2:?--scene needs a name}"; shift 2 ;;
    --preset) preset="${2:?--preset needs a name}"; shift 2 ;;
    --fps) fps="${2:?--fps needs a number}"; shift 2 ;;
    --prompt-mode) prompt_mode="${2:?--prompt-mode needs fixed or dynamic}"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

scene_port() {
  case "$1" in
    racing) echo 8025 ;;
    castle) echo 8026 ;;
    ancient-egypt) echo 8027 ;;
    *) echo "Unknown scene: $1 (racing, castle or ancient-egypt)" >&2; exit 2 ;;
  esac
}
all_scenes=(racing castle ancient-egypt)

state="runtime/live"
mkdir -p "${state}"

listening() { ss -ltn | awk '{print $4}' | grep -q ":$1\$"; }
alive() { [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null; }

case "${action}" in
  status)
    for port in 8555 8890 8190; do
      listening "${port}" && echo "port ${port}: listening" || echo "port ${port}: closed"
    done
    alive "${state}/mediamtx.pid" && echo "mediamtx: started by this script, pid $(cat "${state}/mediamtx.pid")" || echo "mediamtx: no live pid file"
    for name in "${all_scenes[@]}"; do
      port="$(scene_port "${name}")"
      if listening "${port}"; then echo "${name} (port ${port}): listening"; else echo "${name} (port ${port}): closed"; fi
      alive "${state}/serve-${name}.pid" && echo "  started by this script, pid $(cat "${state}/serve-${name}.pid")" || echo "  no live pid file"
    done
    ;;
  stop)
    targets=("${all_scenes[@]}")
    [[ -n "${scene}" ]] && targets=("${scene}")
    for name in "${targets[@]}"; do
      scene_port "${name}" > /dev/null
      if alive "${state}/serve-${name}.pid"; then
        pid="$(cat "${state}/serve-${name}.pid")"
        echo "stopping ${name} (pid ${pid})"
        kill "${pid}"
        for _ in $(seq 1 30); do kill -0 "${pid}" 2>/dev/null || break; sleep 1; done
      else
        echo "${name}: not started by this script, left alone"
      fi
      rm -f "${state}/serve-${name}.pid"
    done
    if [[ -z "${scene}" ]]; then
      if alive "${state}/mediamtx.pid"; then
        pid="$(cat "${state}/mediamtx.pid")"
        echo "stopping mediamtx (pid ${pid})"
        kill "${pid}"
      else
        echo "mediamtx: not started by this script, left alone"
      fi
      rm -f "${state}/mediamtx.pid"
    fi
    ;;
  start)
    scene="${scene:-racing}"
    port="$(scene_port "${scene}")"
    mediamtx_bin="$(command -v mediamtx || true)"
    [[ -n "${mediamtx_bin}" ]] || mediamtx_bin="${HOME}/bin/mediamtx"
    [[ -x "${mediamtx_bin}" ]] || { echo "ERROR: mediamtx not found. Install it first (see docs/dgx_racing_runs.md)." >&2; exit 1; }
    [[ -x .venv/bin/python ]] || { echo "ERROR: .venv is missing. Run scripts/dgx/setup_dgx.sh first." >&2; exit 1; }

    if listening 8555 || listening 8890; then
      echo "MediaMTX ports are already listening; not starting another"
    else
      nohup "${mediamtx_bin}" configs/mediamtx.yml > "${state}/mediamtx.log" 2>&1 < /dev/null &
      echo $! > "${state}/mediamtx.pid"
      echo "started MediaMTX (pid $!)"
      sleep 2
    fi

    if listening "${port}"; then
      echo "${scene} already answers on port ${port}; not starting another"
    else
      min_free="${MIN_FREE_MIB:-30000}"
      free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
      echo "GPU free: ${free_mib} MiB (needs at least ${min_free})"
      if [[ "${free_mib}" -lt "${min_free}" ]]; then
        echo "ERROR: not enough free GPU memory; the GPU is shared. Not starting." >&2
        exit 1
      fi
      scene_prompt_mode="${prompt_mode}"
      [[ "${scene}" != "racing" ]] && scene_prompt_mode="fixed"
      CUDA_VISIBLE_DEVICES=0 PATH="${root}/.venv/bin:${PATH}" PYTHON="${root}/.venv/bin/python" nohup bash scripts/serve.sh \
        --scene "${scene}" --resources resources --preset "${preset}" --fps "${fps}" --prompt-mode "${scene_prompt_mode}" --port "${port}" \
        > "${state}/serve-${scene}.log" 2>&1 < /dev/null &
      echo $! > "${state}/serve-${scene}.pid"
      echo "started ${scene} (pid $!, port ${port}): ${preset}, ${fps} fps, ${scene_prompt_mode} prompt; loading the model"
      for _ in $(seq 1 180); do
        listening "${port}" && break
        kill -0 "$(cat "${state}/serve-${scene}.pid")" 2>/dev/null || { echo "ERROR: ${scene} exited; see ${state}/serve-${scene}.log" >&2; tail -5 "${state}/serve-${scene}.log" >&2; exit 1; }
        sleep 1
      done
      listening "${port}" || { echo "ERROR: ${scene}'s page did not come up within 180 s; see ${state}/serve-${scene}.log" >&2; exit 1; }
    fi
    echo "ready: forward ports 8025, 8026, 8027, 8890 and 8190 and open http://127.0.0.1:${port}/"
    ;;
  *)
    echo "Unknown action: ${action} (start, stop or status)" >&2
    exit 2
    ;;
esac
