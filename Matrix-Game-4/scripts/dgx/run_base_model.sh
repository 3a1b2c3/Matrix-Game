#!/usr/bin/env bash
# Render a recorded clip with the Matrix-Game 4.0 BASE model instead of a scene checkpoint. The scene checkpoints (racing, castle, ancient-egypt)
# are fine-tunes of the open Wan2.2-Fun-5B-Control weights; this script converts those base weights into the layout the Matrix-Game code
# loads (825 tensors, "model." prefix, without the unused ref_conv tensors; the layout was checked against the racing checkpoint) and renders
# with them, using the racing scene profile (caption, first-frame sink, 864x480). Nothing from the racing fine-tune is used.
# Details and caveats: docs/base_model.md. Needs scripts/dgx/setup_dgx.sh and scripts/dgx/download_base.sh to have completed.
#
# Usage:   bash scripts/dgx/run_base_model.sh --input CLIP [--sink-image IMAGE | --no-sink-image] [--preset quality|realtime]
#                                              [--prompt TEXT] [--output DIR] [--download] [--rebuild] [--convert-only]
#   --input         recorded low-poly clip (required unless --convert-only)
#   --sink-image    reference photo shown as output frame 0                      [resources/models/racing/sink.png, the stock reference]
#   --no-sink-image render without a reference image
#   --preset        quality (Wan VAE, 2 steps) or realtime (TAEHV, 1 step)        [quality]
#   --prompt        override the fixed scene caption
#   --output        output folder, must not exist yet                            [outputs/base_<timestamp>]
#   --download      fetch the base weights (10 GB, Hugging Face alibaba-pai/Wan2.2-Fun-5B-Control) when missing
#   --rebuild       convert the base weights again even if the converted file exists
#   --convert-only  prepare resources/models/base/model.safetensors and stop; no render
# Environment: MIN_FREE_MIB  free GPU memory required, in MiB                    [30000]
set -euo pipefail

self="$(readlink -f "${BASH_SOURCE[0]}")"
root="$(cd "$(dirname "${self}")/../.." && pwd)"
cd "${root}"

input=""
sink=""
no_sink=0
preset="quality"
prompt=""
output="outputs/base_$(date +%Y%m%d_%H%M%S)"
download=0
rebuild=0
convert_only=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) input="${2:?--input needs a file}"; shift 2 ;;
    --sink-image) sink="${2:?--sink-image needs a file}"; shift 2 ;;
    --no-sink-image) no_sink=1; shift ;;
    --preset) preset="${2:?--preset needs a name}"; shift 2 ;;
    --prompt) prompt="${2:?--prompt needs text}"; shift 2 ;;
    --output) output="${2:?--output needs a folder}"; shift 2 ;;
    --download) download=1; shift ;;
    --rebuild) rebuild=1; shift ;;
    --convert-only) convert_only=1; shift ;;
    -h|--help) sed -n '2,21p' "${self}"; exit 0 ;;
    *) echo "ERROR: unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
done

tool=".venv/bin/matrix-game"
python_bin=".venv/bin/python"
[[ -x "${tool}" ]] || { echo "ERROR: ${tool} not found. Run scripts/dgx/setup_dgx.sh first." >&2; exit 1; }
[[ "${preset}" == "quality" || "${preset}" == "realtime" ]] || { echo "ERROR: --preset must be quality or realtime" >&2; exit 2; }
for required in resources/base/models_t5_umt5-xxl-enc-bf16.pth resources/base/Wan2.2_VAE.pth resources/base/taew2_2.pth resources/racing/assets; do
  [[ -e "${required}" ]] || { echo "ERROR: ${required} is missing. Run: bash scripts/dgx/download_base.sh and download_scene.sh racing" >&2; exit 1; }
done

# Step 1: the base weights, converted once into the layout the Matrix-Game code loads.
source_dir="resources_base_src"
source_file="${source_dir}/diffusion_pytorch_model.safetensors"
converted="resources/models/base/model.safetensors"
if (( rebuild )) || [[ ! -f "${converted}" ]]; then
  if [[ ! -f "${source_file}" ]]; then
    (( download )) || { echo "ERROR: ${source_file} is missing. Re-run with --download to fetch it (10 GB)." >&2; exit 1; }
    echo "downloading the base weights (10 GB) from alibaba-pai/Wan2.2-Fun-5B-Control"
    mkdir -p "${source_dir}"
    "${python_bin}" -c "from huggingface_hub import hf_hub_download; hf_hub_download('alibaba-pai/Wan2.2-Fun-5B-Control', 'diffusion_pytorch_model.safetensors', revision='b8bc1a65ab71d054ba4636dc0dac104aa4df2686', local_dir='${source_dir}')"
  fi
  echo "converting the base weights -> ${converted}"
  mkdir -p "$(dirname "${converted}")"
  "${python_bin}" - "${source_file}" "${converted}" "resources/models/racing/model.safetensors" <<'PY'
import json
import os
import struct
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

source, destination, reference = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def header(path):
    with open(path, "rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        table = json.loads(stream.read(size))
    table.pop("__metadata__", None)
    return table


tensors = {}
with safe_open(source, framework="pt") as stream:
    for key in stream.keys():
        if key.startswith("ref_conv."):
            continue  # the Fun-Control reference convolution is not part of the Matrix-Game layout
        tensors["model." + key] = stream.get_tensor(key)
if len(tensors) != 825:
    raise SystemExit(f"expected 825 tensors after conversion, got {len(tensors)}")
if reference.is_file():
    expected = {key: value["shape"] for key, value in header(reference).items()}
    actual = {key: list(value.shape) for key, value in tensors.items()}
    if expected != actual:
        raise SystemExit("converted tensor names or shapes differ from the racing checkpoint layout")
    print("layout matches the racing checkpoint: 825 tensors, same names and shapes")
temporary = destination.with_suffix(".tmp")
save_file(tensors, str(temporary), metadata={"format": "pt"})
os.replace(temporary, destination)
print(f"wrote {destination} ({destination.stat().st_size / 1e9:.2f} GB)")
PY
else
  echo "converted base weights already present: ${converted} (use --rebuild to convert again)"
fi
(( convert_only )) && exit 0

# Step 2: render with the base weights and the racing scene profile.
[[ -n "${input}" ]] || { echo "ERROR: --input is required (see --help)" >&2; exit 2; }
[[ -f "${input}" ]] || { echo "ERROR: input clip not found: ${input}" >&2; exit 2; }
[[ ! -e "${output}" ]] || { echo "ERROR: output folder already exists: ${output}" >&2; exit 2; }
sink_args=()
if (( no_sink )); then
  sink_args=(--no-sink-image)
elif [[ -n "${sink}" ]]; then
  [[ -f "${sink}" ]] || { echo "ERROR: sink image not found: ${sink}" >&2; exit 2; }
  sink_args=(--sink-image "${sink}")
fi
prompt_args=()
[[ -z "${prompt}" ]] || prompt_args=(--prompt "${prompt}")

min_free="${MIN_FREE_MIB:-30000}"
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
echo "GPU free: ${free_mib} MiB (needs at least ${min_free})"
(( free_mib >= min_free )) || { echo "ERROR: not enough free GPU memory; the GPU is shared. Not starting." >&2; exit 1; }

mkdir -p "$(dirname "${output}")"
echo "model: ${converted} (Wan2.2-Fun-5B-Control base, no scene fine-tune)"
echo "input: ${input} | preset: ${preset} | output: ${output}"
start=${SECONDS}
"${tool}" render --scene racing --input "${input}" --resources resources --checkpoint "${converted}" --preset "${preset}" --prompt-mode fixed --output "${output}" "${sink_args[@]}" "${prompt_args[@]}" 2>&1 | tee "${output}.render.log"
elapsed=$((SECONDS - start))
[[ -f "${output}/rendered.mp4" ]] || { echo "ERROR: ${output}/rendered.mp4 was not produced; see ${output}.render.log" >&2; exit 1; }
"${python_bin}" - "${output}/render.json" "${elapsed}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
frames, seconds = report.get("frames"), report.get("seconds")
line = f"whole run: {float(sys.argv[2]):.0f} s including model load"
if frames and seconds:
    line += f" | generation: {frames} frames in {seconds:.1f} s = {frames / seconds:.1f} fps"
print(line)
PY
echo "output: ${root}/${output}/rendered.mp4"
