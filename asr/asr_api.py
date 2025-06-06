from typing import Dict, Any, List, Optional, Union
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchaudio
import tempfile
import os
import logging
import time
import asyncio
import transformers
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Configuration from environment variables
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE_MB', '50')) * 1024 * 1024
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT_SECONDS', '60'))
INFERENCE_TIMEOUT = int(os.getenv('INFERENCE_TIMEOUT_SECONDS', '30'))
LOG_LEVEL = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper())

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/asr_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Helper function to wrap async operations with timeout
async def with_timeout(coro: Any, timeout_seconds: int, operation_name: str) -> Any:
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

# Detect device for inference
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
logger.info(f"Using device: {device}")

# Load model and processor from Hugging Face
process_id = os.getpid()
logger.info(f"Loading Wav2Vec2 model and processor... (PID: {process_id})")

# Suppress transformers warnings about unused weights
transformers.logging.set_verbosity_error()

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-large-960h"
)
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-960h"
).to(device)
logger.info(f"Model and processor loaded successfully on {device} (PID: {process_id})")

# Task 2b: Check if service is working
@app.get("/ping")
async def ping() -> Dict[str, str]:
    """Ping endpoint to check if service is working."""
    return {"message": "pong"}

# Task 2c: ASR endpoint to transcribe audio (single file or batch)
@app.post("/asr")
async def asr_transcribe(
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None)
) -> Union[Dict[str, str], Dict[str, List[Dict[str, Any]]]]:
    """ASR endpoint to transcribe audio - supports single file or batch processing."""
    start_time = time.time()
    request_id = int(time.time() * 1000000)
    
    # Validate input
    if file and files:
        raise HTTPException(status_code=400, detail="Send either 'file' or 'files', not both")
    if not file and not files:
        raise HTTPException(status_code=400, detail="No files provided. Send either 'file' (single) or 'files' (batch)")
    
    try:
        if file:
            # Single file processing
            logger.info(f"ASR request {request_id} started - single file: {file.filename}, content_type: {file.content_type}")
            return await with_timeout(
                _process_asr_request(file, request_id, start_time),
                REQUEST_TIMEOUT,
                "ASR request processing"
            )
        else:
            # Batch processing
            logger.info(f"ASR batch request {request_id} started - {len(files)} files")
            return await with_timeout(
                _process_batch_request(files, request_id, start_time),
                REQUEST_TIMEOUT * len(files),  # Scale timeout with number of files
                "ASR batch request processing"
            )
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        file_info = file.filename if file else f"{len(files)} files"
        logger.error(
            f"Request {request_id} [{file_info}] failed after {total_time:.2f}s - Error: {str(e)}",
            exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

async def _process_asr_request(file: UploadFile, request_id: int, start_time: float) -> Dict[str, str]:
    """Process ASR request with timeout and temporary file handling."""
    try:
        # Process audio file using shared function
        audio_result = await _process_audio_file(file)
        
        if audio_result["status"] == "error":
            error_msg = audio_result["error"]
            logger.warning(f"Request {request_id} [{file.filename}] - {error_msg}")
            
            # Map common errors to appropriate HTTP status codes
            if "Only MP3" in error_msg:
                raise HTTPException(status_code=415, detail="Only MP3 audio files are supported")
            elif "File too large" in error_msg:
                raise HTTPException(status_code=413, detail=error_msg)
            elif "Empty or invalid" in error_msg or "Invalid sample rate" in error_msg:
                raise HTTPException(status_code=422, detail="Corrupted or invalid audio file")
            else:
                raise HTTPException(status_code=422, detail="Failed to process audio file")
        
        # Extract waveform data
        waveform = audio_result["waveform"]
        duration = str(audio_result["duration"])
        
        logger.debug(f"Request {request_id} [{file.filename}] - Audio processed successfully")
        
        # Prepare input for model
        input_values = processor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_values.to(device)
        
        # Perform inference with timeout
        logits = await with_timeout(
            _run_inference(input_values),
            INFERENCE_TIMEOUT,
            "Model inference"
        )
        
        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        total_time = time.time() - start_time
        logger.info(
            f"Request {request_id} [{file.filename}] completed in {total_time:.2f}s, duration: {duration}s"
        )
        
        return {
            "transcription": transcription,
            "duration": duration
        }
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(
            f"Request {request_id} [{file.filename}] failed after {total_time:.2f}s - Error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))

async def _process_batch_request(files: List[UploadFile], request_id: int, start_time: float) -> Dict[str, List[Dict[str, Any]]]:
    """Process batch ASR request with true batch inference."""
    logger.info(f"Processing batch {request_id} with {len(files)} files using batch inference")
    
    # Step 1: Load and preprocess all audio files using shared function
    waveforms = []
    file_info = []
    
    for i, file in enumerate(files):
        audio_result = await _process_audio_file(file)
        
        if audio_result["status"] == "success":
            # Store processed waveform and metadata
            waveforms.append(audio_result["waveform"])
            file_info.append({
                "filename": file.filename,
                "duration": audio_result["duration"],
                "processing_time": audio_result["processing_time"],
                "status": "success"
            })
            logger.debug(f"Batch {request_id} - File {i+1}/{len(files)} [{file.filename}] loaded successfully")
        else:
            file_info.append({
                "filename": file.filename or f"file_{i}",
                "status": "error",
                "error": audio_result["error"],
                "processing_time": audio_result["processing_time"]
            })
            logger.warning(f"Batch {request_id} - File {i+1}/{len(files)} [{file.filename}] failed to load: {audio_result['error']}")
    
    # Step 2: Perform batch inference on valid waveforms
    transcriptions = []
    if waveforms:
        try:
            inference_start = time.time()
            
            # Pad waveforms to same length for batching
            max_length = max(w.shape[0] for w in waveforms)
            padded_waveforms = []
            
            for w in waveforms:
                if w.shape[0] < max_length:
                    padding = max_length - w.shape[0]
                    w = torch.nn.functional.pad(w, (0, padding))
                padded_waveforms.append(w)
            
            # Stack into batch tensor [batch_size, seq_length]
            batch_waveforms = torch.stack(padded_waveforms)
            
            # Process entire batch with processor
            input_values = processor(
                batch_waveforms.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            ).input_values.to(device)
            
            # Single inference call for all files
            with torch.no_grad():
                logits = model(input_values).logits
            
            # Decode all predictions at once
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = processor.batch_decode(predicted_ids)
            
            inference_time = time.time() - inference_start
            logger.info(f"Batch {request_id} - Batch inference completed in {inference_time:.2f}s for {len(waveforms)} files")
            
        except Exception as e:
            logger.error(f"Batch {request_id} - Batch inference failed: {str(e)}")
            # Mark all successful files as failed due to inference error
            for info in file_info:
                if info["status"] == "success":
                    info["status"] = "error"
                    info["error"] = f"Batch inference failed: {str(e)}"
    
    # Step 3: Build final results
    results = []
    transcription_idx = 0
    
    for info in file_info:
        if info["status"] == "error":
            results.append({
                "filename": info["filename"],
                "status": "error",
                "transcription": "",
                "error": info["error"],
                "processing_time": info["processing_time"]
            })
        else:
            results.append({
                "filename": info["filename"],
                "status": "success",
                "transcription": transcriptions[transcription_idx] if transcription_idx < len(transcriptions) else "",
                "duration": info["duration"],
                "processing_time": info["processing_time"]
            })
            transcription_idx += 1
    
    # Log final results
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r["status"] == "success")
    logger.info(f"Batch request {request_id} completed in {total_time:.2f}s - {successful}/{len(files)} successful")
    
    return {"results": results}

async def _process_audio_file(file: UploadFile) -> Dict[str, Any]:
    """Process a single audio file and return waveform data or error info."""
    file_start_time = time.time()
    temp_file_path = None
    
    try:
        # Validate file format
        if not file.filename or not file.filename.lower().endswith('.mp3'):
            raise ValueError("Only MP3 audio files are supported")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            content = await file.read()
            
            # Check file size
            if len(content) > MAX_FILE_SIZE:
                raise ValueError(f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB")
            
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Load audio
        waveform, sample_rate = torchaudio.load(temp_file_path)
        
        # Validate audio
        if waveform.size(0) == 0 or waveform.size(1) == 0:
            raise ValueError("Empty or invalid audio data")
        if sample_rate <= 0:
            raise ValueError("Invalid sample rate")
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        return {
            "status": "success",
            "waveform": waveform.squeeze(0),  # Remove channel dimension
            "duration": round(waveform.shape[1] / sample_rate, 1),
            "processing_time": round(time.time() - file_start_time, 2)
        }
        
    except Exception as e:
        # Clean up temp file if it exists
        if temp_file_path:
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        return {
            "status": "error", 
            "error": str(e),
            "processing_time": round(time.time() - file_start_time, 2)
        }

async def _run_inference(input_values: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(input_values).logits

