#!/usr/bin/env bash
# Download and verify only the shared Matrix-Game 4.0 base files at the pinned release revision: the umT5 text encoder, the Wan2.2 VAE,
# the umT5 tokenizer files and the TAEHV fast codec under resources/base, plus the small shared game assets. No scene checkpoint is
# fetched (download_scene.sh adds those, 10 GB each). The text encoder and VAE are first taken from the Hugging Face cache when a copy
# with the manifest's SHA256 exists (link_cached_base.py hard-links it, no extra disk), and whatever is missing is downloaded by
# `matrix-game download` from a manifest that lists the shared bundle only. Resumable and safe to run again: files that already verify
# are skipped. Needs scripts/dgx/setup_dgx.sh to have completed.
#
# Sizes: the shared files are 14.2 GB and the shared game assets about 20 MB.
#
# Usage:   bash scripts/dgx/download_base.sh [--verify-only]
#   --verify-only   check the installed base files against the manifests; download nothing
# Environment: MIN_FREE_GB   free disk required before downloading, in GB   [20]
set -euo pipefail

self="$(readlink -f "${BASH_SOURCE[0]}")"
root="$(cd "$(dirname "${self}")/../.." && pwd)"
cd "${root}"

verify_only=0
case "${1:-}" in
  "") ;;
  --verify-only) verify_only=1 ;;
  -h|--help) sed -n '2,13p' "${self}"; exit 0 ;;
  *) echo "ERROR: unknown option $1 (see --help)" >&2; exit 2 ;;
esac

revision="4419accdb7bb7881234c5914de8cd915266e0325"
repo="Skywork/Matrix-Game-4.0"
tool=".venv/bin/matrix-game"
[[ -x "${tool}" ]] || { echo "ERROR: ${tool} not found. Run scripts/dgx/setup_dgx.sh first." >&2; exit 1; }

# Manifests reduced to the shared bundle, so the downloader fetches no scene checkpoint.
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
.venv/bin/python - "${work}" <<'PY'
import json
import sys
from pathlib import Path

work = Path(sys.argv[1])
for name in ("models", "game-assets"):
    manifest = json.loads(Path("releases", f"{name}.json").read_text())
    manifest["bundles"] = {"shared": manifest["bundles"]["shared"]}
    (work / f"{name}.json").write_text(json.dumps(manifest))
PY

if (( ! verify_only )); then
  min_free="${MIN_FREE_GB:-20}"
  free_gb="$(df -BG --output=avail . | tail -1 | tr -dc '0-9')"
  echo "free disk here: ${free_gb} GB (needs at least ${min_free} GB if nothing is cached)"
  if (( free_gb < min_free )); then
    echo "ERROR: not enough free disk. Set MIN_FREE_GB lower if the Hugging Face cache already holds the base files." >&2
    exit 1
  fi
  .venv/bin/python scripts/dgx/link_cached_base.py --resources resources
fi

# `--scene racing` only selects which scene bundle is added to the shared one; the reduced manifests contain none.
for manifest in models game-assets; do
  echo "===== ${manifest} (shared base) ====="
  if (( ! verify_only )); then
    "${tool}" download --manifest "${work}/${manifest}.json" --repo-id "${repo}" --repo-type model --revision "${revision}" --scene racing --resources resources
  fi
  "${tool}" verify --manifest "${work}/${manifest}.json" --scene racing --resources resources > /dev/null
  echo "${manifest}: verified"
done

echo
echo "Base files under resources/base ($(du -sh resources/base | cut -f1)):"
ls -l resources/base | awk 'NR>1 {printf "  %7.2f GB  %s\n", $5/1e9, $9}'
echo "Add a scene with: bash scripts/dgx/download_scene.sh <racing|castle|ancient-egypt>"
