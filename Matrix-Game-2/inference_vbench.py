import os
import argparse
import sys
#os.environ["TORCH_USE_FLASH_ATTENTION"] = "0"
import torch
import numpy as np

# Disable flash attention
torch.backends.cuda.enable_flash_sdp(False)


from omegaconf import OmegaConf
from torchvision.transforms import v2
from diffusers.utils import load_image
from einops import rearrange
from pipeline import CausalInferencePipeline
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.visualize import process_video
from utils.misc import set_seed
from utils.conditions import *
from utils.wan_wrapper import WanDiffusionWrapper
from safetensors.torch import load_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/inference_yaml/inference_universal.yaml", help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to the checkpoint")
    parser.add_argument("--img_path", type=str, default="demo_images/universal/0000.png", help="Path to the image")
    parser.add_argument("--output_folder", type=str, default="outputs/", help="Output folder")
    parser.add_argument("--num_output_frames", type=int, default=150,
                        help="Number of output latent frames (21 → 81 video frames for VBench)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--pretrained_model_path", type=str, default="Matrix-Game-2.0", help="Path to the VAE model folder")
    # VBench batch mode
    parser.add_argument("--vbench_output_dir", type=str, default=None,
                        help="VBench output directory (activates batch mode); videos saved as {caption}-{idx}.mp4")
    parser.add_argument("--vbench_crop_dir", type=str,
                        default=r"C:\workspace\world\VBench\vbench2_beta_i2v\vbench2_beta_i2v\data\crop",
                        help="VBench crop image directory")
    parser.add_argument("--vbench_info_json", type=str,
                        default=r"C:\workspace\world\VBench\vbench2_beta_i2v\vbench2_beta_i2v\data\i2v-bench-info.json",
                        help="Path to i2v-bench-info.json")
    parser.add_argument("--image_types", type=str, default="scenery,indoor",
                        help="Comma-separated image types to include (default: scenery,indoor)")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Videos to generate per prompt for VBench (default: 5)")
    parser.add_argument("--fps_log", type=str, default=None,
                        help="Path for FPS/timing log (default: {vbench_output_dir}/fps_log.txt)")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Number of denoising steps; overrides denoising_step_list in config")
    args = parser.parse_args()
    return args

class InteractiveGameInference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        self.weight_dtype = torch.bfloat16

        self._init_config()
        self._init_models()

        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _init_config(self):
        self.config = OmegaConf.load(self.args.config_path)
        if getattr(self.args, 'num_steps', None):
            n = self.args.num_steps
            step_list = [round(1000 * (n - i) / n) for i in range(n)]
            self.config.denoising_step_list = step_list
            print(f"[MG2] denoising_step_list overridden to {step_list} ({n} steps)")

    def _init_models(self):
        # Initialize pipeline
        generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}), is_causal=True)
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load(os.path.join(self.args.pretrained_model_path, "Wan2.1_VAE.pth"), map_location="cpu")
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if 'decoder.' in key or 'conv2' in key:
                decoder_state_dict[key] = value
        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(self.device, torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()
        if os.name != 'nt':  # Disable torch.compile if running on Windows
            current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")
        pipeline = CausalInferencePipeline(self.config, generator=generator, vae_decoder=current_vae_decoder)
        if self.args.checkpoint_path:
            print("Loading Pretrained Model...")
            state_dict = load_file(self.args.checkpoint_path)
            pipeline.generator.load_state_dict(state_dict)

        self.pipeline = pipeline.to(device=self.device, dtype=self.weight_dtype)
        self.pipeline.vae_decoder.to(torch.float16)

        vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae.to(self.device, self.weight_dtype)

    def _resizecrop(self, image, th, tw):
        w, h = image.size
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        image = image.crop((left, top, right, bottom))
        return image
    
    def generate_videos(self):
        mode = self.config.pop('mode')
        assert mode in ['universal', 'gta_drive', 'templerun']

        image = load_image(self.args.img_path)
        image = self._resizecrop(image, 352, 640)
        image = self.frame_process(image)[None, :, None, :, :].to(dtype=self.weight_dtype, device=self.device)
        # Encode the input image as the first latent
        padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (self.args.num_output_frames - 1), 1, 1)
        img_cond = torch.concat([image, padding_video], dim=2)
        tiler_kwargs={"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1) 
        visual_context = self.vae.clip.encode_video(image)
        sampled_noise = torch.randn(
            [1, 16,self.args.num_output_frames, 44, 80], device=self.device, dtype=self.weight_dtype
        )
        num_frames = (self.args.num_output_frames - 1) * 4 + 1
        
        conditional_dict = {
            "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
            "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype)
        }
        
        if mode == 'universal':
            cond_data = Bench_actions_universal(num_frames)
            mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            conditional_dict['mouse_cond'] = mouse_condition
        elif mode == 'gta_drive':
            cond_data = Bench_actions_gta_drive(num_frames)
            mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            conditional_dict['mouse_cond'] = mouse_condition
        else:
            cond_data = Bench_actions_templerun(num_frames)
        keyboard_condition = cond_data['keyboard_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
        conditional_dict['keyboard_cond'] = keyboard_condition
        
        with torch.no_grad():
            videos = self.pipeline.inference(
                noise=sampled_noise,
                conditional_dict=conditional_dict,
                return_latents=False,
                mode=mode,
                profile=False
            )

        videos_tensor = torch.cat(videos, dim=1)
        videos = rearrange(videos_tensor, "B T C H W -> B T H W C")
        videos = ((videos.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
        video = np.ascontiguousarray(videos)
        mouse_icon = 'assets/images/mouse.png'
        if mode != 'templerun':
            config = (
                keyboard_condition[0].float().cpu().numpy(),
                mouse_condition[0].float().cpu().numpy()
            )
        else:
            config = (
                keyboard_condition[0].float().cpu().numpy()
            )
        process_video(video.astype(np.uint8), self.args.output_folder+f'/demo.mp4', config, mouse_icon, mouse_scale=0.1, process_icon=False, mode=mode)
        process_video(video.astype(np.uint8), self.args.output_folder+f'/demo_icon.mp4', config, mouse_icon, mouse_scale=0.1, process_icon=True, mode=mode)
        print("Done")

    def run_vbench(self):
        import glob as _glob
        import json, random, time
        from diffusers.utils import export_to_video

        mode = 'universal'
        num_frames = (self.args.num_output_frames - 1) * 4 + 1

        with open(self.args.vbench_info_json) as f:
            entries = json.load(f)

        allowed_types = {t.strip() for t in self.args.image_types.split(',') if t.strip()} if self.args.image_types else None
        if allowed_types:
            entries = [e for e in entries if e.get('type', '') in allowed_types]

        crop_dir = self.args.vbench_crop_dir
        folders = sorted(d for d in os.listdir(crop_dir) if os.path.isdir(os.path.join(crop_dir, d)))
        if not folders:
            raise ValueError(f"No subdirectories in: {crop_dir}")
        crop_folder = os.path.join(crop_dir, folders[0])

        os.makedirs(self.args.vbench_output_dir, exist_ok=True)
        fps_log = self.args.fps_log or os.path.join(self.args.vbench_output_dir, "fps_log.txt")

        fps_lines = [
            "Matrix-Game-2 VBench FPS Log",
            f"types={self.args.image_types}  samples={self.args.num_samples}  "
            f"num_output_frames={self.args.num_output_frames}  video_frames={num_frames}  seed_base={self.args.seed}",
            "",
        ]
        total_gen_t = 0.0
        total_videos = 0
        n_skipped = 0

        # Pre-scan: count already-done and total to-do
        total_todo = 0
        n_prescan_done = 0
        for _entry in entries:
            _caption = _entry.get('caption', _entry['file_name'])
            _img = os.path.join(crop_folder, _entry['file_name'])
            if not os.path.exists(_img):
                continue
            for si in range(self.args.num_samples):
                if _glob.glob(os.path.join(self.args.vbench_output_dir, f"{_caption}-{si}-*.mp4")):
                    n_prescan_done += 1
                else:
                    total_todo += 1
        print(f"[MG2-VBench] {len(entries)} entries after type filter ({self.args.image_types}) | "
              f"{n_prescan_done} already done | {total_todo} to generate")

        for entry_idx, entry in enumerate(entries):
            file_name = entry['file_name']
            caption = entry.get('caption', file_name)
            image_path = os.path.join(crop_folder, file_name)

            if not os.path.exists(image_path):
                print(f"[MG2-VBench] Skip {entry_idx}: not found: {image_path}")
                continue

            prompt_skipped = 0
            for sample_idx in range(self.args.num_samples):
                if _glob.glob(os.path.join(self.args.vbench_output_dir, f"{caption}-{sample_idx}-*.mp4")):
                    n_skipped += 1
                    prompt_skipped += 1
                    continue

                sample_seed = random.randint(0, 2**32 - 1)
                set_seed(sample_seed)
                vbench_path = os.path.join(self.args.vbench_output_dir, f"{caption}-{sample_idx}-{sample_seed}.mp4")
                print(f"[MG2-VBench] {entry_idx}/{len(entries)} s{sample_idx} seed={sample_seed}: {caption[:60]}")
                t0 = time.time()

                image = load_image(image_path)
                image = self._resizecrop(image, 352, 640)
                image = self.frame_process(image)[None, :, None, :, :].to(dtype=self.weight_dtype, device=self.device)

                padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (self.args.num_output_frames - 1), 1, 1)
                img_cond = torch.concat([image, padding_video], dim=2)
                tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
                img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)
                mask_cond = torch.ones_like(img_cond)
                mask_cond[:, :, 1:] = 0
                cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
                visual_context = self.vae.clip.encode_video(image)
                sampled_noise = torch.randn(
                    [1, 16, self.args.num_output_frames, 44, 80], device=self.device, dtype=self.weight_dtype
                )

                conditional_dict = {
                    "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
                    "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype),
                }
                cond_data = Bench_actions_universal(num_frames)
                mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
                keyboard_condition = cond_data['keyboard_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
                conditional_dict['mouse_cond'] = mouse_condition
                conditional_dict['keyboard_cond'] = keyboard_condition

                with torch.no_grad():
                    videos = self.pipeline.inference(
                        noise=sampled_noise,
                        conditional_dict=conditional_dict,
                        return_latents=False,
                        mode=mode,
                        profile=False,
                    )

                videos_tensor = torch.cat(videos, dim=1)
                videos_np = rearrange(videos_tensor, "B T C H W -> B T H W C")
                videos_np = ((videos_np.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
                frames = [videos_np[i] / 255.0 for i in range(videos_np.shape[0])]
                export_to_video(frames, vbench_path, fps=24)

                elapsed = time.time() - t0
                gen_fps = num_frames / elapsed
                total_gen_t += elapsed
                total_videos += 1
                line = f"{caption}-{sample_idx}-{sample_seed} | {num_frames}f | {elapsed:.1f}s | gen_fps={gen_fps:.3f}"
                fps_lines.append(line)
                avg_t = total_gen_t / total_videos
                remaining = total_todo - total_videos
                eta_s = int(avg_t * remaining)
                eta_str = f"{eta_s // 3600}h {(eta_s % 3600) // 60}m {eta_s % 60}s"
                pct = 100.0 * total_videos / max(total_todo, 1)
                print(f"[MG2-VBench] [{total_videos}/{total_todo}] {pct:.1f}% | "
                      f"prompt {entry_idx+1}/{len(entries)} s{sample_idx+1}/{self.args.num_samples} | "
                      f"{elapsed:.1f}s | ETA: {eta_str} | skipped so far: {n_skipped}")
            if prompt_skipped > 0:
                print(f"[MG2-VBench] Skipped prompt {entry_idx+1}/{len(entries)}: "
                      f"{prompt_skipped}/{self.args.num_samples} samples exist — {caption[:60]}")

        fps_lines.append("")
        if total_videos > 0:
            avg_t = total_gen_t / total_videos
            fps_lines.append(
                f"Total: {total_videos} generated  {n_skipped} skipped  "
                f"avg_time={avg_t:.1f}s  avg_gen_fps={num_frames / avg_t:.3f}"
            )
        else:
            avg_t = 0.0
            fps_lines.append(f"Total: 0 generated  {n_skipped} skipped")
        with open(fps_log, 'w') as f:
            f.write('\n'.join(fps_lines) + '\n')
        print(f"[MG2-VBench] Done. FPS log: {fps_log}")

        # Write stats summary txt
        import datetime as _dt
        stats_file = os.path.join(os.path.dirname(self.args.vbench_output_dir), "vbench_stats.txt")
        total_s = int(total_gen_t)
        with open(stats_file, 'w') as _sf:
            _sf.write("Matrix-Game-2 VBench Stats\n")
            _sf.write("==========================\n")
            _sf.write(f"Date:           {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            _sf.write(f"\n=== Settings ===\n")
            _sf.write(f"Image types:    {self.args.image_types}\n")
            _sf.write(f"Num samples:    {self.args.num_samples}\n")
            _sf.write(f"Lat frames:     {self.args.num_output_frames}\n")
            _sf.write(f"Video frames:   {num_frames}\n")
            _sf.write(f"Num steps:      {getattr(self.args, 'num_steps', None) or 'config default'}\n")
            _sf.write(f"Seed:           {self.args.seed}\n")
            _sf.write(f"\n=== Performance ===\n")
            _sf.write(f"Generated:      {total_videos}\n")
            _sf.write(f"Skipped:        {n_skipped}\n")
            _sf.write(f"Total time:     {total_s // 3600}h {(total_s % 3600) // 60}m {total_s % 60}s ({total_s}s)\n")
            if total_videos > 0:
                _sf.write(f"Avg per video:  {avg_t:.1f}s\n")
                _sf.write(f"Avg gen fps:    {num_frames / avg_t:.3f}\n")
            _sf.write(f"\n=== Output ===\n")
            _sf.write(f"Videos dir:     {self.args.vbench_output_dir}\n")
            _sf.write(f"FPS log:        {fps_log}\n")
        print(f"[MG2-VBench] Stats: {stats_file}")


def main():
    """Main entry point for video generation."""
    args = parse_args()
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.pretrained_model_path):
        args.pretrained_model_path = os.path.join(_script_dir, args.pretrained_model_path)
    if not os.path.isabs(args.config_path):
        args.config_path = os.path.join(_script_dir, args.config_path)
    if args.vbench_output_dir and not os.path.isabs(args.vbench_output_dir):
        args.vbench_output_dir = os.path.join(_script_dir, args.vbench_output_dir)
    set_seed(args.seed)

    if args.vbench_output_dir:
        import glob as _glob, json as _json
        with open(args.vbench_info_json) as _f:
            _entries = _json.load(_f)
        if args.image_types:
            _allowed = {t.strip() for t in args.image_types.split(',') if t.strip()}
            _entries = [e for e in _entries if e.get('type', '') in _allowed]
        _out_dir = args.vbench_output_dir
        _n_done = sum(
            1 for e in _entries
            for si in range(args.num_samples)
            if _glob.glob(os.path.join(_out_dir, f"{e.get('caption', e['file_name'])}-{si}-*.mp4"))
        )
        _n_todo = len(_entries) * args.num_samples - _n_done
        print(f"[MG2-VBench] Output dir  : {args.vbench_output_dir}")
        print(f"[MG2-VBench] Prompts     : {len(_entries)} ({args.image_types})")
        print(f"[MG2-VBench] To generate : {_n_todo}  |  already done: {_n_done}")
        print(f"[MG2-VBench] Loading models...")

    pipeline = InteractiveGameInference(args)
    if args.vbench_output_dir:
        pipeline.run_vbench()
    else:
        os.makedirs(args.output_folder, exist_ok=True)
        pipeline.generate_videos()

if __name__ == "__main__":
    main()