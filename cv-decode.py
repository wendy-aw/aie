#!/usr/bin/env python3
"""
cv-decode.py - Transcribe Common Voice MP3 files using ASR API

This script calls the ASR API to transcribe all MP3 files in the cv-valid-dev folder.
MP3 files can be downloaded from https://www.dropbox.com/scl/fi/i9yvfqpf7p8uye5o8k1sj/common_voice.zip
?rlkey=lz3dtjuhekc3xw4jnoeoqy5yu&dl=0
"""

import os
import json
import time
import logging
import asyncio
import aiohttp
import aiofiles
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import argparse
from tqdm.asyncio import tqdm

# Configuration
API_BASE_URL = f"http://localhost:{os.getenv('PORT', 8001)}" # ASR API base URL
DATA_FOLDER = "cv-valid-dev" # Folder containing MP3 files
INPUT_CSV = "cv-valid-dev.csv" # Input CSV file with audio filenames
OUTPUT_CSV = "cv-valid-dev-updated.csv" # Output CSV file with transcriptions
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1  # seconds

# Calculate optimal concurrency based on API workers
API_WORKERS = int(os.getenv('WORKERS', '1'))
DEFAULT_CONCURRENT = API_WORKERS * 2 if API_WORKERS > 1 else 2  # 2x workers or min 2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cv-decode.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def check_api_health() -> bool:
    """Check if ASR API is running and healthy."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(API_BASE_URL + "/ping", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"API health check: {data}")
                    return True
                else:
                    logger.error(f"API health check failed: {response.status}")
                    return False
    except Exception as e:
        logger.error(f"API health check error: {str(e)}")
        return False


async def transcribe_file(session: aiohttp.ClientSession, file_path: Path) -> Dict[str, Any]:
    """Transcribe a single MP3 file using the ASR API."""
    file_name = file_path.name
    start_time = time.time()
    
    for attempt in range(RETRY_ATTEMPTS):
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                file_content = await f.read()
            
            # Prepare multipart form data
            data = aiohttp.FormData()
            data.add_field('file', 
                          file_content, 
                          filename=file_name, 
                          content_type='audio/mpeg')
            
            # Make API request
            async with session.post(API_BASE_URL + "/asr", data=data, timeout=120) as response:
                if response.status == 200:
                    result = await response.json()
                    duration = time.time() - start_time
                    
                    logger.info(f"✓ {file_name} transcribed in {duration:.2f}s")
                    
                    return {
                        "file": file_name,
                        "status": "success",
                        "transcription": result.get("transcription", ""),
                        "audio_duration": result.get("duration", ""),
                        "processing_time": round(duration, 2),
                        "attempt": attempt + 1
                    }
                else:
                    error_text = await response.text()
                    logger.warning(f"✗ {file_name} failed (attempt {attempt + 1}): {response.status} - {error_text}")
                    
                    if attempt == RETRY_ATTEMPTS - 1:
                        return {
                            "file": file_name,
                            "status": "error",
                            "transcription": "",  # Empty transcription for failed files
                            "error": f"HTTP {response.status}: {error_text}",
                            "processing_time": round(time.time() - start_time, 2),
                            "attempts": RETRY_ATTEMPTS
                        }
                    
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    
        except asyncio.TimeoutError:
            logger.warning(f"✗ {file_name} timeout (attempt {attempt + 1})")
            if attempt == RETRY_ATTEMPTS - 1:
                return {
                    "file": file_name,
                    "status": "error",
                    "transcription": "",  # Empty transcription for failed files
                    "error": "Request timeout",
                    "processing_time": round(time.time() - start_time, 2),
                    "attempts": RETRY_ATTEMPTS
                }
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            
        except Exception as e:
            logger.error(f"✗ {file_name} error (attempt {attempt + 1}): {str(e)}")
            if attempt == RETRY_ATTEMPTS - 1:
                return {
                    "file": file_name,
                    "status": "error",
                    "transcription": "",  # Empty transcription for failed files
                    "error": str(e),
                    "processing_time": round(time.time() - start_time, 2),
                    "attempts": RETRY_ATTEMPTS
                }
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))


async def process_files_batch(mp3_filenames: List[str], folder_path: Path, concurrent: int) -> Dict[str, str]:
    """Process MP3 file names from CSV and return transcription results."""
    connector = aiohttp.TCPConnector(limit=concurrent)
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes
    
    transcriptions = {}
    success_count = 0
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent)
        
        # Progress bar setup
        progress_bar = tqdm(
            total=len(mp3_filenames), 
            desc="Transcribing files",
            unit="file",
            dynamic_ncols=True,
            colour="green"
        )
        
        async def process_with_semaphore_and_progress(file_name: str):
            nonlocal success_count
            async with semaphore:
                file_path = folder_path / file_name
                
                if file_path.exists():
                    result = await transcribe_file(session, file_path)
                else: 
                    logger.warning(f"File not found: {file_name}")
                    result = {
                        "file": file_name,
                        "status": "error",
                        "transcription": "",
                        "error": "File not found"
                    }
                
                # Update progress bar
                progress_bar.update(1)
                
                # Update success counter and progress bar description
                if result["status"] == "success":
                    success_count += 1
                    progress_bar.set_postfix({
                        "Success": f"{success_count}/{progress_bar.n}",
                        "Current": result["file"][:20] + "..." if len(result["file"]) > 20 else result["file"]
                    })
                else:
                    progress_bar.set_postfix({
                        "Success": f"{success_count}/{progress_bar.n}",
                        "Error": result["file"][:20] + "..." if len(result["file"]) > 20 else result["file"]
                    })
                
                return result
        
        # Process all files concurrently (semaphore controls actual concurrency)
        tasks = [process_with_semaphore_and_progress(file_name) for file_name in mp3_filenames]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store results
            error_count = 0
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task error: {result}")
                    error_count += 1
                else:
                    transcriptions[result["file"]] = result["transcription"]
                    
                    if result["status"] != "success":
                        error_count += 1
        
        finally:
            progress_bar.close()
    
    logger.info(f"Transcription completed: {success_count}/{len(mp3_filenames)} successful, {error_count} errors")
    return transcriptions


def load_csv_and_get_files(csv_path: str) -> List[str]:
    """Load CSV file and extract MP3 filenames."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV with {len(df)} rows")
    
    # Extract filenames and ensure they have .mp3 extension
    filenames = df['filename'].tolist()
    mp3_filenames = []
    
    for filename in filenames:
        if pd.isna(filename):
            continue
        
        # Extract just the filename part, removing any directory path
        filename = str(filename)
        if '/' in filename:
            filename = filename.split('/')[-1]
        
        # Ensure .mp3 extension
        if not filename.lower().endswith('.mp3'):
            filename = filename + '.mp3'
        
        mp3_filenames.append(filename)
    
    logger.info(f"Found {len(mp3_filenames)} MP3 file names in CSV")
    return mp3_filenames


def save_updated_csv(original_csv: str, transcriptions: Dict[str, str], output_csv: str):
    """Save updated CSV with transcription column."""
    # Load original CSV
    df = pd.read_csv(original_csv)
    logger.info(f"Loading original CSV: {original_csv}")
    
    # Add transcription column
    transcription_texts = []
    for _, row in df.iterrows():
        filename = row["filename"]
        if pd.isna(filename):
            transcription_texts.append("")
            continue
            
        # Extract just the filename part for lookup
        lookup_name = str(filename)
        if '/' in lookup_name:
            lookup_name = lookup_name.split('/')[-1]
        
        # Ensure .mp3 extension for lookup
        if not lookup_name.lower().endswith('.mp3'):
            lookup_name = lookup_name + '.mp3'
        
        # Get transcription
        transcription = transcriptions.get(lookup_name, "")
        transcription_texts.append(transcription)
    
    # Add new column
    df['generated_text'] = transcription_texts
    
    # Save updated CSV
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved updated CSV: {output_csv}")
    logger.info(f"Added transcriptions for {sum(1 for t in transcription_texts if t)} files")


async def main():
    """Main function to orchestrate the transcription process."""
    parser = argparse.ArgumentParser(description="Transcribe Common Voice MP3 files using ASR API and update CSV")
    parser.add_argument("--csv", default=INPUT_CSV, help="Input CSV file with audio filenames")
    parser.add_argument("--folder", default=DATA_FOLDER, help="Folder containing MP3 files")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV file with transcriptions")
    parser.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENT, help=f"Max concurrent requests (default: {DEFAULT_CONCURRENT} based on {API_WORKERS} API workers)")
    parser.add_argument("--n_files", type=int, help="Limit number of files to process")
    
    args = parser.parse_args()
    logger.info("Starting Common Voice CSV transcription process")
    logger.info(f"Input CSV: {args.csv}")
    logger.info(f"Audio folder: {args.folder}")
    logger.info(f"Output CSV: {args.output}")
    logger.info(f"Max concurrent requests: {args.concurrent}")
    if args.n_files:
        logger.info(f"Limiting to {args.n_files} files")
    
    # Check API health
    logger.info("Checking API health...")
    if not await check_api_health():
        logger.error("ASR API is not available. Please start the API server first.")
        return 1
    
    start_time = time.time()
    
    # Load CSV and get filenames
    try:
        mp3_filenames = load_csv_and_get_files(args.csv)
        if not mp3_filenames:
            logger.error(f"No MP3 file names found in CSV: {args.csv}")
            return 1
        
        # Apply file limit if specified
        if args.n_files and args.n_files < len(mp3_filenames):
            mp3_filenames = mp3_filenames[:args.n_files]
            logger.info(f"Limiting transcription to first {args.n_files} files")
            
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1
    
    # Process files
    folder_path = Path(args.folder)
    if not folder_path.exists():
        logger.error(f"Audio data folder not found: {args.folder}")
        return 1
    
    logger.info(f"Starting transcription of {len(mp3_filenames)} files...")
    transcriptions = await process_files_batch(mp3_filenames, folder_path, args.concurrent)
    
    # Save updated CSV
    try:
        save_updated_csv(args.csv, transcriptions, args.output)
    except Exception as e:
        logger.error(f"Error saving CSV: {str(e)}")
        return 1
    
    # Summary
    total_time = time.time() - start_time
    success_count = sum(1 for t in transcriptions.values() if t.strip())
    total_count = len(mp3_filenames)
    
    logger.info("=" * 50)
    logger.info("TRANSCRIPTION COMPLETE")
    logger.info(f"Total files: {total_count}")
    logger.info(f"Successful transcriptions: {success_count}")
    logger.info(f"Empty/failed transcriptions: {total_count - success_count}")
    logger.info(f"Success rate: {success_count/total_count*100:.1f}%")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Average time per file: {total_time/total_count:.2f} seconds")
    logger.info(f"Updated CSV saved to: {args.output}")
    logger.info("=" * 50)
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)