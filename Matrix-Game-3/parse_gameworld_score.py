"""
Parse GameWorldScore eval_results.json files into a CSV.

Usage:
    python parse_gameworld_score.py                          # latest result in out/gameworld_score
    python parse_gameworld_score.py path/to/eval_results.json
    python parse_gameworld_score.py out/gameworld_score      # picks latest *_eval_results.json in folder
    python parse_gameworld_score.py --all                    # merge all runs in out/gameworld_score

Output: <results_stem>.csv  (same folder as the json)
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path


def _find_latest(folder: Path) -> Path:
    candidates = sorted(folder.glob("*_eval_results.json"))
    if not candidates:
        raise FileNotFoundError(f"No *_eval_results.json found in {folder}")
    return candidates[-1]


def _stem(video_path: str) -> str:
    """Return filename without extension."""
    return Path(video_path).stem


def parse_eval_results(json_path: Path) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return (overall_scores, per_video_scores).

    overall_scores  : {dimension: float}
    per_video_scores: {video_stem: {dimension: float}}
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    overall: dict[str, float] = {}
    per_video: dict[str, dict[str, float]] = {}

    for dim, payload in data.items():
        if not isinstance(payload, list) or len(payload) < 2:
            continue
        overall[dim] = float(payload[0])
        video_list = payload[1]
        if not isinstance(video_list, list):
            continue
        for entry in video_list:
            stem = _stem(entry["video_path"])
            per_video.setdefault(stem, {})
            per_video[stem][dim] = float(entry["video_results"])

    return overall, per_video


def write_csv(json_path: Path, overall: dict, per_video: dict) -> Path:
    dims = list(overall.keys())
    out_path = json_path.with_suffix(".csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video"] + dims)
        # summary row first
        writer.writerow(["OVERALL"] + [f"{overall[d]:.6f}" for d in dims])
        # per-video rows sorted by name
        for stem in sorted(per_video.keys()):
            row = [stem] + [f"{per_video[stem].get(d, ''):.6f}" if per_video[stem].get(d) is not None else "" for d in dims]
            writer.writerow(row)

    return out_path


def main():
    args = sys.argv[1:]

    if "--all" in args:
        folder = Path("out/gameworld_score")
        files = sorted(folder.glob("*_eval_results.json"))
        if not files:
            print(f"No eval_results.json found in {folder}")
            sys.exit(1)
        for f in files:
            overall, per_video = parse_eval_results(f)
            out = write_csv(f, overall, per_video)
            print(f"  {f.name} → {out.name}")
        return

    if args:
        p = Path(args[0])
        json_path = _find_latest(p) if p.is_dir() else p
    else:
        json_path = _find_latest(Path("out/gameworld_score"))

    overall, per_video = parse_eval_results(json_path)
    out = write_csv(json_path, overall, per_video)

    print(f"Written: {out}")
    print()
    print(f"{'Dimension':<28} {'Score':>8}")
    print("-" * 38)
    for dim, score in overall.items():
        print(f"  {dim:<26} {score:.4f}")

    # Also print action-following if present
    act_path = json_path.parent / "mg3_action_control_results.json"
    if act_path.exists():
        import json as _json
        act = _json.loads(act_path.read_text())["overall"]
        print(f"\n  {'action_following (prec)':<26} {act['precision']:.4f}")
        print(f"  {'action_following (rec)':<26} {act['recall']:.4f}")
        print(f"  {'action_following (f1)':<26} {act['f1']:.4f}")


if __name__ == "__main__":
    main()
