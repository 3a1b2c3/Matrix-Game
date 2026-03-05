import os
from huggingface_hub import hf_hub_download

# Output directory for downloaded checkpoints
output_dir = "Matrix-Game-2.0"
os.makedirs(output_dir, exist_ok=True)

# Hugging Face repository and files to download
checkpoints = [
    ("Skywork/Matrix-Game-2.0", "base_distilled_model/base_distill.safetensors"),
    # ("Skywork/Matrix-Game-2.0", "gta_distilled_model/gta_keyboard2dim.safetensors"),
    ("Skywork/Matrix-Game-2.0", "Wan2.1_VAE.pth"),
]

for repo_id, filename in checkpoints:
    print(f"Downloading {filename} from {repo_id}...")
    file_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=output_dir)
    print(f"Saved to {file_path}")

print("All checkpoints downloaded to", output_dir)