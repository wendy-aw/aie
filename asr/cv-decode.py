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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
API_PORT = int(os.getenv('PORT', '8001'))
API_BASE_URL = os.getenv('API_BASE_URL', f"http://localhost:{API_PORT}")
DATA_FOLDER = os.getenv('DATA_FOLDER', 'cv-valid-dev')
INPUT_CSV = os.getenv('INPUT_CSV', 'cv-valid-dev.csv')
OUTPUT_CSV = os.getenv('OUTPUT_CSV', 'cv-valid-dev-updated.csv')
RETRY_ATTEMPTS = int(os.getenv('RETRY_ATTEMPTS', '3'))
RETRY_DELAY = int(os.getenv('RETRY_DELAY', '1'))

# Calculate optimal concurrency based on API workers
API_WORKERS = int(os.getenv('WORKERS', '1'))
DEFAULT_CONCURRENT = int(os.getenv('DEFAULT_CONCURRENT', str(API_WORKERS * 2 if API_WORKERS > 1 else 2)))
DEFAULT_BATCH_SIZE = int(os.getenv('DEFAULT_BATCH_SIZE', '5'))

# Configure logging
LOG_LEVEL = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper())
logging.basicConfig(
    level=LOG_LEVEL,
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


async def transcribe_batch(session: aiohttp.ClientSession, file_paths: List[Path], batch_id: int) -> List[Dict[str, Any]]:
    """Transcribe multiple MP3 files using the ASR batch API."""
    start_time = time.time()
    batch_size = len(file_paths)
    
    for attempt in range(RETRY_ATTEMPTS):
        try:
            # Prepare multipart form data for batch
            data = aiohttp.FormData()
            file_info = []
            
            # Load all files in the batch
            for file_path in file_paths:
                try:
                    async with aiofiles.open(file_path, 'rb') as f:
                        file_content = await f.read()
                        data.add_field('files', 
                                      file_content, 
                                      filename=file_path.name, 
                                      content_type='audio/mpeg')
                        file_info.append((file_path.name, True))
                except Exception as e:
                    # File couldn't be read
                    file_info.append((file_path.name, False, str(e)))
            
            # Make batch API request
            timeout = aiohttp.ClientTimeout(total=120 * batch_size)  # Scale timeout with batch size
            async with session.post(API_BASE_URL + "/asr", data=data, timeout=timeout) as response:
                if response.status == 200:
                    batch_result = await response.json()
                    duration = time.time() - start_time
                    
                    # Process batch results
                    results = []
                    if "results" in batch_result:
                        # Batch response format
                        for result in batch_result["results"]:
                            results.append({
                                "file": result.get("filename", "unknown"),
                                "status": result.get("status", "error"),
                                "transcription": result.get("transcription", ""),
                                "audio_duration": result.get("duration", ""),
                                "processing_time": result.get("processing_time", round(duration, 2)),
                                "attempt": attempt + 1
                            })
                    else:
                        # Single file response format (fallback)
                        results.append({
                            "file": file_paths[0].name if file_paths else "unknown",
                            "status": "success",
                            "transcription": batch_result.get("transcription", ""),
                            "audio_duration": batch_result.get("duration", ""),
                            "processing_time": round(duration, 2),
                            "attempt": attempt + 1
                        })
                    
                    # Add any files that couldn't be read
                    for file_name, success, *error in file_info:
                        if not success:
                            results.append({
                                "file": file_name,
                                "status": "error",
                                "transcription": "",
                                "error": error[0] if error else "File read error",
                                "processing_time": round(time.time() - start_time, 2),
                                "attempt": attempt + 1
                            })
                    
                    return results
                    
                else:
                    error_text = await response.text()
                    
                    if attempt == RETRY_ATTEMPTS - 1:
                        # Return error for all files in batch
                        results = []
                        for file_path in file_paths:
                            results.append({
                                "file": file_path.name,
                                "status": "error",
                                "transcription": "",
                                "error": f"HTTP {response.status}: {error_text}",
                                "processing_time": round(time.time() - start_time, 2),
                                "attempts": RETRY_ATTEMPTS
                            })
                        return results
                    
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    
        except asyncio.TimeoutError:
            if attempt == RETRY_ATTEMPTS - 1:
                results = []
                for file_path in file_paths:
                    results.append({
                        "file": file_path.name,
                        "status": "error",
                        "transcription": "",
                        "error": "Request timeout",
                        "processing_time": round(time.time() - start_time, 2),
                        "attempts": RETRY_ATTEMPTS
                    })
                return results
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            
        except Exception as e:
            if attempt == RETRY_ATTEMPTS - 1:
                results = []
                for file_path in file_paths:
                    results.append({
                        "file": file_path.name,
                        "status": "error",
                        "transcription": "",
                        "error": str(e),
                        "processing_time": round(time.time() - start_time, 2),
                        "attempts": RETRY_ATTEMPTS
                    })
                return results
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))


async def process_files_batch(mp3_filenames: List[str], folder_path: Path, concurrent: int, batch_size: int = DEFAULT_BATCH_SIZE) -> Dict[str, str]:
    """Process MP3 file names from CSV using batch API requests."""
    connector = aiohttp.TCPConnector(limit=concurrent)
    timeout = aiohttp.ClientTimeout(total=600)  # Longer timeout for batch requests
    
    transcriptions = {}
    success_count = 0
    
    # Split files into batches
    file_batches = []
    for i in range(0, len(mp3_filenames), batch_size):
        batch_files = mp3_filenames[i:i+batch_size]
        batch_paths = [folder_path / filename for filename in batch_files]
        file_batches.append((i // batch_size, batch_paths))
    
    logger.info(f"Processing {len(mp3_filenames)} files in {len(file_batches)} batches of size {batch_size}")
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Create semaphore to limit concurrent batch requests
        semaphore = asyncio.Semaphore(concurrent)
        
        # Progress bar setup
        progress_bar = tqdm(
            total=len(mp3_filenames), 
            desc="Transcribing batches",
            unit="file",
            dynamic_ncols=True,
            colour="green"
        )
        
        async def process_batch_with_semaphore(batch_id: int, file_paths: List[Path]):
            nonlocal success_count
            async with semaphore:
                # Filter existing files
                existing_files = []
                missing_files = []
                
                for file_path in file_paths:
                    if file_path.exists():
                        existing_files.append(file_path)
                    else:
                        missing_files.append(file_path)
                
                batch_results = []
                
                # Process existing files with batch API
                if existing_files:
                    batch_results = await transcribe_batch(session, existing_files, batch_id)
                
                # Add missing files as errors
                for missing_file in missing_files:
                    batch_results.append({
                        "file": missing_file.name,
                        "status": "error",
                        "transcription": "",
                        "error": "File not found"
                    })
                
                # Update progress and success count
                batch_success_count = 0
                for result in batch_results:
                    if result["status"] == "success":
                        batch_success_count += 1
                        success_count += 1
                    
                    # Update progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "Success": f"{success_count}/{progress_bar.n}",
                        "Batch": f"{batch_id+1}/{len(file_batches)}"
                    })
                
                return batch_results
        
        # Process all batches concurrently
        batch_tasks = [
            process_batch_with_semaphore(batch_id, file_paths) 
            for batch_id, file_paths in file_batches
        ]
        
        try:
            batch_results_list = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Flatten results and store transcriptions
            error_count = 0
            
            for batch_results in batch_results_list:
                if isinstance(batch_results, Exception):
                    logger.error(f"Batch processing error: {batch_results}")
                    error_count += 1
                else:
                    for result in batch_results:
                        transcriptions[result["file"]] = result["transcription"]
                        if result["status"] != "success":
                            error_count += 1
        
        finally:
            progress_bar.close()
    
    logger.info(f"Batch transcription completed: {success_count}/{len(mp3_filenames)} successful, {error_count} errors")
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
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Number of files per batch (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--n_files", type=int, help="Limit number of files to process")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.batch_size <= 0:
        logger.error("batch_size must be greater than 0")
        return 1
    if args.n_files is not None and args.n_files <= 0:
        logger.error("n_files must be greater than 0")
        return 1
    
    logger.info("Starting Common Voice CSV transcription process")
    logger.info(f"Input CSV: {args.csv}")
    logger.info(f"Audio folder: {args.folder}")
    logger.info(f"Output CSV: {args.output}")
    logger.info(f"Max concurrent requests: {args.concurrent}")
    logger.info(f"Batch size: {args.batch_size}")
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
    transcriptions = await process_files_batch(mp3_filenames, folder_path, args.concurrent, args.batch_size)
    
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