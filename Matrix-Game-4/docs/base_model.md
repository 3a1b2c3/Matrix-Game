# Rendering with the base model

This page describes how to render with the **base model** that the Matrix-Game 4.0 scene checkpoints were fine-tuned from, instead of a scene
checkpoint (racing, castle, ancient-egypt). It covers what the base model is, how it differs from the scene checkpoints, the script that runs
it, and what is and is not known about the results.

## What the base model is

The three released checkpoints (`resources/models/<scene>/model.safetensors`, 10.0 GB each, bf16) are described by their `inference.json` as
`Wan2.2-Fun-5B-Control causal`. The open Wan2.2-Fun-5B-Control weights are published as
`alibaba-pai/Wan2.2-Fun-5B-Control` on Hugging Face; its `diffusion_pytorch_model.safetensors` is also 10.0 GB (bf16, 5.0 B parameters).
The shared files in `resources/base` (umT5 text encoder, Wan2.2 VAE) come from the same repository, at the revision pinned in
`releases/models.json` (`b8bc1a65ab71d054ba4636dc0dac104aa4df2686`). There is no separate "base transformer" in the Matrix-Game 4.0 release.

## How it differs from a scene checkpoint

Compared on the DGX, header by header:

| | Wan2.2-Fun-5B-Control base | Matrix-Game racing checkpoint |
| --- | --- | --- |
| Tensors | 827 | 825 |
| Names | `blocks.0.cross_attn.k.weight` | `model.blocks.0.cross_attn.k.weight` |
| Shapes | all 825 shared tensors identical | identical |
| Extra tensors | `ref_conv.weight`, `ref_conv.bias` | none |
| Precision | bf16 | bf16 |
| Training | General video generation with control input | Fine-tuned for causal, chunk-by-chunk streaming of one game's low-poly video |

So the base weights fit Matrix-Game's model code once the names get the `model.` prefix and the two `ref_conv` tensors are dropped. The
Matrix-Game loader takes the checkpoint path from `--checkpoint` and does not verify it against `model.json`.

The scene fine-tune is what teaches the model to follow its own game's low-poly input over time. A scene checkpoint rendering a clip from a
different game can drift away from the input (see the notes in `docs/dgx_racing_runs.md`). Rendering with the base model removes the racing
fine-tune from the result. Whether the base weights follow a low-poly input *better*, worse or not at all in the causal pipeline is **not known**:
no render with them has been evaluated yet.

## The script

`scripts/dgx/run_base_model.sh`, run from `Matrix-Game-4/` on the DGX.

```bash
bash scripts/dgx/run_base_model.sh --input captures/taxi-lanes-gb.mp4
```

What it does:

1. Checks that the shared base files (`resources/base`) and `resources/racing/assets` exist.
2. Converts the base weights once into `resources/models/base/model.safetensors`: prefix `model.`, `ref_conv.*` removed, 825 tensors. It refuses
   to write unless the names and shapes match the racing checkpoint's layout. The source is `resources_base_src/diffusion_pytorch_model.safetensors`;
   `--download` fetches it (10 GB, pinned revision) when missing.
3. Runs `matrix-game render --scene racing --checkpoint resources/models/base/model.safetensors ...`. The racing **scene profile** (caption,
   first-frame sink, 864x480 frame size) is reused because the code needs a profile; the racing **checkpoint is not loaded**.
4. Prints the whole-run time and the generation speed from the render report.

| Option | Meaning | Default |
| --- | --- | --- |
| `--input CLIP` | Recorded low-poly clip | required |
| `--sink-image IMAGE` | Reference photo shown as output frame 0 | `resources/models/racing/sink.png` |
| `--no-sink-image` | Render without a reference image | off |
| `--preset quality\|realtime` | Wan VAE and 2 steps, or TAEHV and 1 step | `quality` |
| `--prompt TEXT` | Override the fixed scene caption | the racing caption |
| `--output DIR` | Output folder, must not exist yet | `outputs/base_<timestamp>` |
| `--download` | Fetch the base weights when missing | off |
| `--rebuild` | Convert again even if the converted file exists | off |
| `--convert-only` | Prepare the converted file and stop | off |

The default sink `sink.png` is a racing **image** (a gray sedan), not the racing checkpoint. Pass `--sink-image` with your own reference or
`--no-sink-image` to avoid it. The racing caption also names a sedan; use `--prompt` for a different subject.

Output: `outputs/base_<timestamp>/rendered.mp4`, `render.json` and `outputs/base_<timestamp>.render.log`.

## Files and sizes

| File | Size | Notes |
| --- | --- | --- |
| `resources_base_src/diffusion_pytorch_model.safetensors` | 10.0 GB | Source download, kept so the conversion can be repeated |
| `resources/models/base/model.safetensors` | 10.0 GB | Converted base weights, 825 tensors |
| `resources/base/*` | 14.2 GB | Shared encoders and codecs, unchanged |

Both folders are outside Git, like the rest of `resources/`.

## Status

- The conversion was run on the DGX: 825 tensors, names and shapes identical to the racing checkpoint, written in 4 s.
- No render with the base weights has been run or evaluated yet. Speed should match the scene checkpoints (same architecture and size), about
  14 fps generation for the quality preset and about 40 fps for realtime at 864x480 on the GB300, measured with the racing checkpoint.
- For comparison, renders of the same gray-box lane clip with the **racing** checkpoint are in `outputs/from_dgx` on the PC (and
  `outputs/gb_basic_*`, `gb_race_*` on the DGX): the stock sink with the realtime preset collapsed after about 5 s, and the taxi sink with the
  quality preset stayed coherent.

## Related

- `scripts/dgx/download_base.sh`: downloads and verifies the shared base files in `resources/base`.
- `scripts/dgx/run_taxi_quality.sh`: racing checkpoint, quality preset, custom sink image.
- `scripts/capture_graybox.py`: records the taxi game as a gray-box input clip.
- `docs/dgx_racing_runs.md`: notes on earlier DGX runs and known issues.
