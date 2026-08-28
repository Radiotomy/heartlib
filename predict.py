import os
import time
import torch
import torch.nn as nn
import torchaudio
from cog import BasePredictor, Input, Path
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

import sys
sys.path.append('/src')
sys.path.append('/src/src')

from heartlib.heartmula.modeling_heartmula import HeartMuLaModel
from heartlib.heartcodec.modeling_heartcodec import HeartCodecModel

class Predictor(BasePredictor):
    def setup(self):
        print("[SETUP] Starting Predictor initialization...")
        start_setup = time.time()
        
        self.mula_device = "cuda:0"
        self.codec_device = "cuda:0"
        
        print("[SETUP] Locating or downloading model weights...")
        mula_path = snapshot_download(repo_id="HeartMuLa/HeartMuLa-oss-3B")
        codec_path = snapshot_download(repo_id="HeartCodec-oss")
        print(f"[SETUP] Weights verified locally in {time.time() - start_setup:.2f}s")
        
        print("[SETUP] Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(mula_path)
        
        print("[SETUP] Loading HeartMuLa 3B model (bf16)...")
        self.mula_model = HeartMuLaModel.from_pretrained(
            mula_path, 
            torch_dtype=torch.bfloat16
        ).to(self.mula_device).eval()
        
        print("[SETUP] Loading HeartCodec model (fp32)...")
        self.codec_model = HeartCodecModel.from_pretrained(
            codec_path, 
            torch_dtype=torch.float32
        ).to(self.codec_device).eval()
        
        print(f"[SETUP] Initialization fully completed in {time.time() - start_setup:.2f}s. System Ready.")

    def predict(
        self,
        prompt_tags: str = Input(description="Formatted style tags wrapped in <tag>...</tag>", default="<tag>Pop, Acoustic Guitar, Warm, Joyful</tag>"),
        lyrics: str = Input(description="Lyrics annotated with structural markers", default="[Intro]\n[Verse]\nWalking down the road tonight...\n[Chorus]\nSinging under neon light..."),
        cfg_scale: float = Input(description="CFG scale for HeartMuLa", default=1.5, ge=1.0, le=4.0),
        temperature: float = Input(description="Sampling temperature", default=1.0, ge=0.1, le=2.0),
        top_k: int = Input(description="Top-k sampling threshold", default=50, ge=1, le=200),
        codec_cfg_scale: float = Input(description="Flow Matching scale for HeartCodec", default=1.25, ge=1.0, le=2.0),
        max_duration_seconds: int = Input(description="Maximum song duration in seconds", default=180, ge=10, le=360)
    ) -> Path:
        start_time = time.time()
        
        formatted_prompt = f"{prompt_tags.strip()}\n{lyrics.strip()}"
        input_tokens = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(self.mula_device)
        
        print(f"Generating audio up to {max_duration_seconds}s...")
        with torch.inference_mode():
            audio_tokens = self.mula_model.generate(
                input_ids=input_tokens,
                max_duration_s=max_duration_seconds,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_k=top_k
            )
            
            print(f"Decoding tokens via Flow Matching (codec_cfg={codec_cfg_scale})...")
            waveform = self.codec_model.decode(
                tokens=audio_tokens,
                cfg_scale=codec_cfg_scale,
                steps=10
            )

        output_path = "/tmp/output_heartmula.wav"
        torchaudio.save(output_path, waveform.cpu(), sample_rate=48000)
        
        elapsed = time.time() - start_time
        print(f"Generation completed in {elapsed:.2f}s")
        
        return Path(output_path)
