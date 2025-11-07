from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional
import uvicorn

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

# Import IndexTTS
from indextts.infer_v2 import IndexTTS2
from tools.download_files import download_model_from_huggingface

# Initialize FastAPI app
app = FastAPI(title="IndexTTS API", version="2.0")

# Configuration
MODEL_DIR = "./checkpoints"
OUTPUT_DIR = "./outputs"
USE_FP16 = False
USE_DEEPSPEED = False
USE_CUDA_KERNEL = False

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Global TTS model instance
tts_model = None


@app.on_event("startup")
async def startup_event():
    """Initialize the TTS model on startup"""
    global tts_model
    
    # Download model if needed
    download_model_from_huggingface(
        os.path.join(current_dir, "checkpoints"),
        os.path.join(current_dir, "checkpoints", "hf_cache")
    )
    
    # Initialize TTS model
    tts_model = IndexTTS2(
        model_dir=MODEL_DIR,
        cfg_path=os.path.join(MODEL_DIR, "config.yaml"),
        use_fp16=USE_FP16,
        use_deepspeed=USE_DEEPSPEED,
        use_cuda_kernel=USE_CUDA_KERNEL,
    )
    print("TTS model loaded successfully!")


@app.post("/generate")
async def generate_speech(
    reference_audio: UploadFile = File(..., description="Reference audio file for voice cloning"),
    text: str = Form(..., description="Text to synthesize"),
    emo_weight: float = Form(0.8, description="Emotion weight (0.0-1.0)"),
    max_text_tokens: int = Form(120, description="Max tokens per segment"),
    do_sample: bool = Form(True, description="Enable sampling"),
    temperature: float = Form(0.8, description="Temperature (0.1-2.0)"),
    top_p: float = Form(0.8, description="Top-p sampling (0.0-1.0)"),
    top_k: int = Form(30, description="Top-k sampling"),
    num_beams: int = Form(3, description="Number of beams"),
    repetition_penalty: float = Form(10.0, description="Repetition penalty"),
    length_penalty: float = Form(0.0, description="Length penalty"),
    max_mel_tokens: int = Form(1500, description="Maximum mel tokens")
):
    """
    Generate speech from text using reference audio for voice cloning.
    
    - **reference_audio**: Audio file for voice reference (WAV, MP3, etc.)
    - **text**: Text to convert to speech
    - **emo_weight**: Control emotional intensity (0.0-1.0)
    - **max_text_tokens**: Maximum tokens per generation segment
    """
    
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Save uploaded reference audio temporarily
        timestamp = int(time.time())
        ref_audio_path = os.path.join(OUTPUT_DIR, f"ref_{timestamp}.wav")
        
        with open(ref_audio_path, "wb") as f:
            content = await reference_audio.read()
            f.write(content)
        
        # Generate output path
        output_path = os.path.join(OUTPUT_DIR, f"output_{timestamp}.wav")
        
        # Prepare generation parameters
        kwargs = {
            "do_sample": bool(do_sample),
            "top_p": float(top_p),
            "top_k": int(top_k) if int(top_k) > 0 else None,
            "temperature": float(temperature),
            "length_penalty": float(length_penalty),
            "num_beams": int(num_beams),
            "repetition_penalty": float(repetition_penalty),
            "max_mel_tokens": int(max_mel_tokens),
        }
        
        # Generate speech
        output = tts_model.infer(
            spk_audio_prompt=ref_audio_path,
            text=text,
            output_path=output_path,
            emo_audio_prompt=None,
            emo_alpha=emo_weight * 0.8,  # Normalize for better UX
            emo_vector=None,
            use_emo_text=False,
            emo_text=None,
            use_random=False,
            verbose=False,
            max_text_tokens_per_segment=int(max_text_tokens),
            **kwargs
        )
        
        # Clean up reference audio
        if os.path.exists(ref_audio_path):
            os.remove(ref_audio_path)
        
        # Return generated audio file
        if os.path.exists(output):
            return FileResponse(
                output,
                media_type="audio/wav",
                filename=f"generated_{timestamp}.wav",
                headers={"X-Generation-Time": str(timestamp)}
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
            
    except Exception as e:
        # Clean up on error
        if os.path.exists(ref_audio_path):
            os.remove(ref_audio_path)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Check if the API is running and model is loaded"""
    return {
        "status": "healthy" if tts_model is not None else "model_not_loaded",
        "model_loaded": tts_model is not None
    }


@app.get("/")
async def root():
    """API information"""
    return {
        "name": "IndexTTS API",
        "version": "2.0",
        "endpoints": {
            "/generate": "POST - Generate speech from text and reference audio",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
