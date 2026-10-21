#!/usr/bin/env bash
# Render a recorded low-poly clip with the Matrix-Game 4.0 City Racing scene at the "quality" preset (Wan VAE encoding/decoding, two
# denoising steps, native DiT: slower and cleaner than the realtime preset) and a custom sink image, so the car and look come from your
# reference photo instead of the stock gray sedan. Thin wrapper around run_racing_yellow_taxi.sh with the checks and summary added;
# same prerequisites (scripts/dgx/setup_dgx.sh, scripts/dgx/download_scene.sh racing).
#
# Usage:   bash scripts/dgx/run_taxi_quality.sh [--input CLIP] [--sink-image IMAGE] [--output DIR]
#   --input       recorded low-poly clip                    [captures/taxi-lanes.mp4]
#   --sink-image  reference photo, shown as output frame 0  [captures/taxi_sink.jpg]
#   --output      output folder, must not exist yet         [outputs/taxi_quality_<timestamp>]
# Prints the render time, frames per second and the path of rendered.mp4. The log is <output>.render.log.
set -euo pipefail

self="$(readlink -f "${BASH_SOURCE[0]}")"
root="$(cd "$(dirname "${self}")/../.." && pwd)"
cd "${root}"

input="captures/taxi-lanes.mp4"
sink="captures/taxi_sink.jpg"
output="outputs/taxi_quality_$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) input="${2:?--input needs a file}"; shift 2 ;;
    --sink-image) sink="${2:?--sink-image needs a file}"; shift 2 ;;
    --output) output="${2:?--output needs a folder}"; shift 2 ;;
    -h|--help) sed -n '2,11p' "${self}"; exit 0 ;;
    *) echo "ERROR: unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
done

[[ -f "${input}" ]] || { echo "ERROR: input clip not found: ${input}" >&2; exit 2; }
if [[ ! -f "${sink}" ]]; then
  echo "ERROR: sink image not found: ${sink}" >&2
  echo "Copy one from your PC, for example:" >&2
  echo "  scp <image> $(whoami)@$(hostname -I | awk '{print $1}'):${root}/${sink}" >&2
  exit 2
fi
[[ ! -e "${output}" ]] || { echo "ERROR: output folder already exists: ${output}" >&2; exit 2; }

echo "input: ${input}"
echo "sink image: ${sink}"
echo "preset: quality (Wan VAE, 2 denoising steps, native DiT)"
start=${SECONDS}
bash scripts/dgx/run_racing_yellow_taxi.sh --input "${input}" --sink-image "${sink}" --preset quality --output "${output}"
elapsed=$((SECONDS - start))

[[ -f "${output}/rendered.mp4" ]] || { echo "ERROR: ${output}/rendered.mp4 was not produced; see ${output}.render.log" >&2; exit 1; }
.venv/bin/python - "${output}/render.json" "${elapsed}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
frames, seconds = report.get("frames"), report.get("seconds")
total = float(sys.argv[2])
line = f"whole run: {total:.0f} s including model load"
if frames and seconds:
    line += f" | generation: {frames} frames in {seconds:.1f} s = {frames / seconds:.1f} fps"
print(line)
PY
echo "output: ${root}/${output}/rendered.mp4"
