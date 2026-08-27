import os
import torch
from cog import BasePredictor, Input, Path
from src.heartlib.pipelines import HeartMuLaGenPipeline

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda:0"
        
        # Initialize the pipeline using the baked-in weights
        self.pipe = HeartMuLaGenPipeline.from_pretrained(
            "./ckpt",
            device={
                "mula": torch.device(self.device),
                "codec": torch.device(self.device),
            },
            dtype={
                "mula": "bf16",
                "codec": "fp32",
            },
            version="3B",
            lazy_load=False,
        )

    def predict(
        self,
        lyrics: str = Input(
            description="Lyrics for the song",
            default="[Verse]\nThe sun creeps in across the floor\nI hear the traffic outside the door"
        ),
        tags: str = Input(
            description="Comma-separated tags (e.g., piano,happy,wedding,synthesizer,romantic)",
            default="piano,happy,pop"
        ),
        max_audio_length_ms: int = Input(
            description="Maximum audio length in milliseconds",
            default=240000
        ),
        temperature: float = Input(
            description="Sampling temperature for generation",
            default=1.0
        ),
        topk: int = Input(
            description="Top-k sampling parameter for generation",
            default=50
        ),
        cfg_scale: float = Input(
            description="Classifier-free guidance scale",
            default=1.5
        )
    ) -> Path:
        """Run a single prediction on the model"""
        
        # The pipeline expects file paths for lyrics and tags, so we write the API inputs to temp files
        lyrics_path = "/tmp/lyrics.txt"
        tags_path = "/tmp/tags.txt"
        out_path = "/tmp/output.mp3"
        
        with open(lyrics_path, "w") as f:
            f.write(lyrics)
            
        with open(tags_path, "w") as f:
            f.write(tags)

        # Run inference
        with torch.no_grad():
            self.pipe(
                {
                    "lyrics": lyrics_path,
                    "tags": tags_path,
                },
                max_audio_length_ms=max_audio_length_ms,
                save_path=out_path,
                topk=topk,
                temperature=temperature,
                cfg_scale=cfg_scale,
            )

        return Path(out_path)
