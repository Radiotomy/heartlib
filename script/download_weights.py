import os
from huggingface_hub import snapshot_download

def download_models():
    base_path = "./ckpt"
    os.makedirs(base_path, exist_ok=True)
    
    print("Downloading HeartMuLa 3B Weights...")
    snapshot_download(
        repo_id="HeartMuLa/HeartMuLa-oss-3B-happy-new-year",
        local_dir=os.path.join(base_path, "HeartMuLa-oss-3B"),
        ignore_patterns=["*.md", "*.txt"]
    )
    
    print("Downloading HeartCodec Weights...")
    snapshot_download(
        repo_id="HeartMuLa/HeartCodec-oss-20260123",
        local_dir=os.path.join(base_path, "HeartCodec-oss"),
        ignore_patterns=["*.md", "*.txt"]
    )
    
    print("Download complete. Weights baked into image.")

if __name__ == "__main__":
    download_models()
