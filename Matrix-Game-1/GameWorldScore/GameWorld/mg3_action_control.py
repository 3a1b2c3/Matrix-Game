"""
MG3 Action-Following Evaluator
================================
Uses optical flow to estimate the per-frame action taken in a generated video,
then compares against the ground-truth keyboard_condition saved as a .action.npy
sidecar alongside each video.

Keyboard layout (indices 0-3):
    0 = forward   1 = back   2 = left   3 = right
Mouse layout (indices 0-1):
    0 = pitch (up/down camera)   1 = yaw (left/right camera)

Optical flow → action mapping
------------------------------
For a rear-chase camera in a driving game:
    mean_flow_y < -thr  →  forward  (scene flows upward → car moving forward)
    mean_flow_y >  thr  →  back
    mean_flow_x >  thr  →  left turn (scene shifts right → car going left)
    mean_flow_x < -thr  →  right turn
    mean_yaw    >  thr  →  camera_r
    mean_yaw    < -thr  →  camera_l

All binary: threshold = FLOW_THRESH (tunable via env MG3_FLOW_THRESH).
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


FLOW_THRESH = float(os.environ.get("MG3_FLOW_THRESH", "0.5"))
# keyboard_condition dim used for evaluation (skip dim 4-5 which are unused)
KEY_NAMES = ["forward", "back", "left", "right"]
KEY_INDICES = [0, 1, 2, 3]


def _optical_flow_actions(video_path: str, n_frames: int | None = None) -> np.ndarray:
    """Return (T-1, 4) binary array of predicted keyboard actions via optical flow."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()

    if n_frames is not None:
        frames = frames[:n_frames]

    T = len(frames)
    if T < 2:
        return np.zeros((0, 4), dtype=np.float32)

    preds = []
    for i in range(T - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i + 1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        # flow shape (H, W, 2): flow[...,0]=dx  flow[...,1]=dy
        mean_dx = float(flow[..., 0].mean())
        mean_dy = float(flow[..., 1].mean())

        pred = np.zeros(4, dtype=np.float32)
        if mean_dy < -FLOW_THRESH:
            pred[0] = 1.0   # forward
        if mean_dy >  FLOW_THRESH:
            pred[1] = 1.0   # back
        if mean_dx >  FLOW_THRESH:
            pred[2] = 1.0   # left  (scene →right means car →left)
        if mean_dx < -FLOW_THRESH:
            pred[3] = 1.0   # right
        preds.append(pred)

    return np.stack(preds)   # (T-1, 4)


def _gt_actions(sidecar_path: str, n_frames: int | None = None) -> np.ndarray:
    """Load ground-truth keyboard condition from .action.npy sidecar.

    Returns (T, 4) float array, first dimension clipped to n_frames-1 to match
    optical flow output.
    """
    data = np.load(sidecar_path, allow_pickle=True).item()
    kb = data["keyboard"]   # (T, 6) or (T, 4)
    kb = kb[:, :4]          # keep only forward/back/left/right
    # binarize (condition is already 0/1 scaled by key_strength)
    gt = (kb > 0.1).astype(np.float32)
    if n_frames is not None:
        gt = gt[:n_frames]
    # align to flow length (T-1)
    return gt[:-1] if gt.shape[0] > 1 else gt


def compute_mg3_action_control(videos_path: str, output_path: str | None = None):
    """Evaluate action following for all videos that have .action.npy sidecars.

    Returns (overall_precision, per_video_results).
    """
    video_files = sorted(Path(videos_path).rglob("*.mp4"))
    results = []

    for vf in tqdm(video_files, desc="mg3_action_control"):
        sidecar = vf.with_suffix(".action.npy")
        if not sidecar.exists():
            continue

        pred = _optical_flow_actions(str(vf))
        gt   = _gt_actions(str(sidecar), n_frames=pred.shape[0] + 1)

        n = min(pred.shape[0], gt.shape[0])
        if n == 0:
            continue

        pred_n, gt_n = pred[:n], gt[:n]

        per_key = {}
        for ki, kname in enumerate(KEY_NAMES):
            gt_k  = gt_n[:, ki]
            pr_k  = pred_n[:, ki]
            tp = float(((pr_k == 1) & (gt_k == 1)).sum())
            fp = float(((pr_k == 1) & (gt_k == 0)).sum())
            fn = float(((pr_k == 0) & (gt_k == 1)).sum())
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            per_key[kname] = {"precision": prec, "recall": rec, "f1": f1}

        macro_prec = float(np.mean([per_key[k]["precision"] for k in KEY_NAMES]))
        macro_rec  = float(np.mean([per_key[k]["recall"]    for k in KEY_NAMES]))
        macro_f1   = float(np.mean([per_key[k]["f1"]        for k in KEY_NAMES]))

        results.append({
            "video_path":  str(vf),
            "precision":   macro_prec,
            "recall":      macro_rec,
            "f1":          macro_f1,
            "per_key":     per_key,
        })

    if not results:
        print("[mg3_action_control] No videos with .action.npy sidecars found.")
        return 0.0, []

    overall_prec = float(np.mean([r["precision"] for r in results]))
    overall_rec  = float(np.mean([r["recall"]    for r in results]))
    overall_f1   = float(np.mean([r["f1"]        for r in results]))

    print(f"[mg3_action_control] {len(results)} videos")
    print(f"  precision : {overall_prec:.4f}")
    print(f"  recall    : {overall_rec:.4f}")
    print(f"  f1        : {overall_f1:.4f}")

    if output_path:
        out_json = os.path.join(output_path, "mg3_action_control_results.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "overall": {"precision": overall_prec, "recall": overall_rec, "f1": overall_f1},
                "per_video": results,
            }, f, indent=2)
        print(f"  saved -> {out_json}")

        out_csv = os.path.join(output_path, "mg3_action_control_results.csv")
        cols = (
            ["video", "precision", "recall", "f1"] +
            [f"{k}_{m}" for k in KEY_NAMES for m in ("precision", "recall", "f1")]
        )
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            # summary row
            w.writerow(
                ["OVERALL", f"{overall_prec:.6f}", f"{overall_rec:.6f}", f"{overall_f1:.6f}"] +
                [""] * (len(KEY_NAMES) * 3)
            )
            for r in results:
                row = [
                    Path(r["video_path"]).stem,
                    f"{r['precision']:.6f}",
                    f"{r['recall']:.6f}",
                    f"{r['f1']:.6f}",
                ]
                for k in KEY_NAMES:
                    pk = r["per_key"].get(k, {})
                    row += [
                        f"{pk.get('precision', ''):.6f}" if pk else "",
                        f"{pk.get('recall', ''):.6f}"    if pk else "",
                        f"{pk.get('f1', ''):.6f}"        if pk else "",
                    ]
                w.writerow(row)
        print(f"  saved -> {out_csv}")

    return overall_prec, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_path", required=True)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--flow_thresh", type=float, default=FLOW_THRESH)
    args = parser.parse_args()
    FLOW_THRESH = args.flow_thresh
    compute_mg3_action_control(args.videos_path, args.output_path)
