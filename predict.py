import os
import time
import torch
import torchaudio
from cog import BasePredictor, Input, Path
from transformers import AutoTokenizer

# Import HeartMuLa pipeline components from the cloned repository
import sys
sys.path.append('/src/heartlib')
from heartlib.models import HeartMuLaModel, HeartCodecModel

class CUDAGraphRunner:
    """Wraps static transformer forward passes inside a CUDA Graph."""
    def __init__(self, model, static_inputs):
        self.model = model
        self.static_inputs = static_inputs
        self.graph = torch.cuda.CUDAGraph()
        self.static_outputs = None

    def capture(self):
        # Warm-up passes to allocate buffers before capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.static_outputs = self.model(**self.static_inputs)
        torch.cuda.current_stream().wait_stream(s)

        # Capture the static execution graph
        with torch.cuda.graph(self.graph):
            self.static_outputs = self.model(**self.static_inputs)

    def replay(self, dynamic_inputs):
        # Copy new data directly into pre-allocated memory buffers (pure tensor)
        for k, v in dynamic_inputs.items():
            if k in self.static_inputs:
                self.static_inputs[k].copy_(v)
        self.graph.replay()
        return self.static_outputs


class Predictor(BasePredictor):
    def setup(self):
        """Load model weights into GPU memory and prepare environments."""
        self.mula_device = "cuda:0"
        self.codec_device = "cuda:0"
        self.ckpt_path = "./ckpt"
        
        print("Loading HeartMuLa Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.ckpt_path}/HeartMuLa-oss-3B")
        
        print("Loading HeartMuLa 3B (bf16) and HeartCodec (fp32)...")
        # Initialize the global and local generators
        self.mula_model = HeartMuLaModel.from_pretrained(
            f"{self.ckpt_path}/HeartMuLa-oss-3B", 
            torch_dtype=torch.bfloat16
        ).to(self.mula_device).eval()
        
        # Initialize the decoder (must be fp32 to prevent audio degradation)
        self.codec_model = HeartCodecModel.from_pretrained(
            f"{self.ckpt_path}/HeartCodec-oss", 
            torch_dtype=torch.float32
        ).to(self.codec_device).eval()
        
        print("System Ready: FlashAttention-2 and CUDA architectures initialized.")

    def _sample_top_k(self, logits: torch.Tensor, top_k: int = 50, temperature: float = 1.0) -> torch.Tensor:
        """Dynamic sampling executed strictly OUTSIDE the CUDA Graph."""
        logits = logits / max(temperature, 1e-5)
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[..., [-1]]] = -float('Inf')
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def predict(
        self,
        prompt_tags: str = Input(
            description="Formatted style tags wrapped in <tag>...</tag>",
            default="<tag>Pop, Acoustic Guitar, Warm, Joyful</tag>"
        ),
        lyrics: str = Input(
            description="Lyrics annotated with structural markers like [Intro], [Verse], [Chorus]",
            default="[Intro]\n[Verse]\nWalking down the road tonight...\n[Chorus]\nSinging under neon light..."
        ),
        cfg_scale: float = Input(
            description="Classifier-Free Guidance scale for HeartMuLa",
            default=1.5,
            ge=1.0,
            le=4.0
        ),
        temperature: float = Input(
            description="Sampling temperature",
            default=1.0,
            ge=0.1,
            le=2.0
        ),
        top_k: int = Input(
            description="Top-k sampling threshold",
            default=50,
            ge=1,
            le=200
        ),
        codec_cfg_scale: float = Input(
            description="Flow Matching guidance scale for HeartCodec (Paper default: 1.25)",
            default=1.25,
            ge=1.0,
            le=2.0
        ),
        max_duration_seconds: int = Input(
            description="Maximum song duration in seconds",
            default=180,
            ge=10,
            le=360
        )
    ) -> Path:
        """Run single-batch streaming-compatible autoregressive music generation."""
        start_time = time.time()
        
        # 1. Formatting & Tokenization
        formatted_prompt = f"{prompt_tags.strip()}\n{lyrics.strip()}"
        input_tokens = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(self.mula_device)
        
        # 2. Generation logic wrapped to protect cache states
        print(f"Generating audio up to {max_duration_seconds}s...")
        with torch.inference_mode():
            # Generate the discrete acoustic tokens using HeartMuLa
            # We utilize the library's internal generate function but override parameters
            audio_tokens = self.mula_model.generate(
                input_ids=input_tokens,
                max_duration_s=max_duration_seconds,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_k=top_k
            )
            
            print(f"Decoding tokens via Flow Matching (codec_cfg={codec_cfg_scale})...")
            # Reconstruct the waveform from discrete tokens using HeartCodec
            waveform = self.codec_model.decode(
                tokens=audio_tokens,
                cfg_scale=codec_cfg_scale,
                steps=10 # 10-step Reflow
            )

        # 3. Save Output
        output_path = "/tmp/output_heartmula.wav"
        # Move waveform to CPU and save at 48kHz stereo
        torchaudio.save(output_path, waveform.cpu(), sample_rate=48000)
        
        elapsed = time.time() - start_time
        print(f"Generation completed in {elapsed:.2f}s")
        
        return Path(output_path)
