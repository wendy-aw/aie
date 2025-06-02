from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import torch
import torchaudio
import tempfile
import os
import logging
import time
import asyncio
import multiprocessing
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Configuration constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
REQUEST_TIMEOUT = 60  # seconds
INFERENCE_TIMEOUT = 30  # seconds
LOG_LEVEL = logging.INFO  # Change to logging.DEBUG for verbose logging

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('asr_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Helper function to wrap async operations with timeout
async def with_timeout(coro, timeout_seconds, operation_name):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error(
            f"Operation '{operation_name}' timed out after {timeout_seconds}s"
        )
        raise HTTPException(
            status_code=408,
            detail=f"Request timeout: {operation_name}"
        )

# Create FastAPI app
app = FastAPI(
    title="ASR Microservice",
    description="Automatic Speech Recognition API",
    version="1.0.0"
)

# Load model and processor from Hugging Face
logger.info("Loading Wav2Vec2 model and processor...")
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-large-960h"
)
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-960h"
)
logger.info("Model and processor loaded successfully")

# Task 2b: Check if service is working
@app.get("/ping")
async def ping():
    """Ping endpoint to check if service is working."""
    logger.info("Ping endpoint called")
    return {"message": "pong"}

# Task 2c: ASR endpoint to transcribe audio
@app.post("/asr")
async def asr_transcribe(file: UploadFile = File(...)):
    """ASR endpoint to transcribe audio."""
    start_time = time.time()
    request_id = int(time.time() * 1000000)
    
    logger.info(f"ASR request {request_id} started - filename: {file.filename}, content_type: {file.content_type}")
    
    try:
        # Wrap entire request in timeout
        return await with_timeout(
            _process_asr_request(file, request_id, start_time),
            REQUEST_TIMEOUT,
            "ASR request processing"
        )
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(
            f"Request {request_id} [{file.filename}] failed after {total_time:.2f}s - Error: {str(e)}",
            exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

async def _process_asr_request(file: UploadFile, request_id: int, start_time: float):
    """Process ASR request with timeout and temporary file handling."""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(
            delete=False, suffix='.mp3'
        ) as temp_file:
            content = await file.read()
            
            # Check file size limit
            file_size = len(content)
            if file_size > MAX_FILE_SIZE:
                logger.warning(
                    f"Request {request_id} [{file.filename}] - File too large: {file_size} bytes"
                )
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB"
                )
            
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.debug(
            f"Request {request_id} [{file.filename}] - File saved, size: {file_size} bytes"
        )
        
        # Load audio with torchaudio
        waveform, sample_rate = torchaudio.load(temp_file_path)
        logger.debug(
            f"Request {request_id} [{file.filename}] - Audio loaded, sample_rate: {sample_rate}"
        )
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
            logger.debug(
                f"Request {request_id} [{file.filename}] - Resampled to 16kHz"
            )
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            logger.debug(
                f"Request {request_id} [{file.filename}] - Converted to mono"
            )
        
        # Calculate duration
        duration = str(round(waveform.shape[1] / sample_rate, 1))
        
        # Prepare input for model
        input_values = processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_values
        
        # Perform inference with timeout
        logits = await with_timeout(
            _run_inference(input_values),
            INFERENCE_TIMEOUT,
            "Model inference"
        )
        
        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        total_time = time.time() - start_time
        logger.info(
            f"Request {request_id} [{file.filename}] completed in {total_time:.2f}s, duration: {duration}s"
        )
        
        return {
            "transcription": transcription,
            "duration": duration
        }
        
    except HTTPException:
        # Clean up temp file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
        raise
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
        total_time = time.time() - start_time
        logger.error(
            f"Request {request_id} [{file.filename}] failed after {total_time:.2f}s - Error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))

async def _run_inference(input_values):
    with torch.no_grad():
        return model(input_values).logits

if __name__ == "__main__":
    num_workers = (2 * multiprocessing.cpu_count()) + 1
    uvicorn.run(
        app, host="0.0.0.0", port=8001, workers=num_workers
    )