"""
Unit tests for cv-decode.py functionality.
"""
import pytest
import asyncio
import aiohttp
import pandas as pd
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import tempfile
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the module under test (need to handle the hyphen in filename)
import importlib.util
cv_decode_path = Path(__file__).parent.parent.parent / "cv-decode.py"
spec = importlib.util.spec_from_file_location("cv_decode", cv_decode_path)
cv_decode = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv_decode)


class TestAPIHealthCheck:
    """Test API health check functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_api_health_success(self):
        """Test successful API health check."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"message": "pong"})
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await cv_decode.check_api_health()
            
            assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_api_health_failure(self):
        """Test API health check failure."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 500
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await cv_decode.check_api_health()
            
            assert result is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_api_health_exception(self):
        """Test API health check with connection exception."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.side_effect = aiohttp.ClientError("Connection failed")
            
            result = await cv_decode.check_api_health()
            
            assert result is False


class TestBatchTranscription:
    """Test batch transcription functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_transcribe_batch_success(self, temp_dir: Path, batch_audio_files: list[Path]):
        """Test successful batch transcription."""
        # Create a mock session
        mock_session = AsyncMock()
        
        # Mock successful API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "results": [
                {
                    "filename": "test_audio_0.mp3",
                    "status": "success",
                    "transcription": "hello world",
                    "duration": "1.0",
                    "processing_time": 0.5
                },
                {
                    "filename": "test_audio_1.mp3",
                    "status": "success",
                    "transcription": "test audio",
                    "duration": "1.0",
                    "processing_time": 0.6
                },
                {
                    "filename": "test_audio_2.mp3",
                    "status": "success",
                    "transcription": "another test",
                    "duration": "1.0",
                    "processing_time": 0.4
                }
            ]
        })
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        result = await cv_decode.transcribe_batch(mock_session, batch_audio_files, batch_id=0)
        
        assert len(result) == 3
        assert all(r["status"] == "success" for r in result)
        assert all("transcription" in r for r in result)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_transcribe_batch_http_error(self, batch_audio_files: list[Path]):
        """Test batch transcription with HTTP error."""
        mock_session = AsyncMock()
        
        # Mock HTTP error response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        result = await cv_decode.transcribe_batch(mock_session, batch_audio_files, batch_id=0)
        
        assert len(result) == len(batch_audio_files)
        assert all(r["status"] == "error" for r in result)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_transcribe_batch_timeout(self, batch_audio_files: list[Path]):
        """Test batch transcription with timeout."""
        mock_session = AsyncMock()
        mock_session.post.side_effect = asyncio.TimeoutError()
        
        result = await cv_decode.transcribe_batch(mock_session, batch_audio_files, batch_id=0)
        
        assert len(result) == len(batch_audio_files)
        assert all(r["status"] == "error" for r in result)
        assert all("timeout" in r["error"].lower() for r in result)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_transcribe_batch_retry_logic(self, batch_audio_files: list[Path]):
        """Test batch transcription retry logic."""
        mock_session = AsyncMock()
        
        # First two attempts fail, third succeeds
        mock_responses = [
            AsyncMock(status=500),
            AsyncMock(status=503),
            AsyncMock(status=200, json=AsyncMock(return_value={
                "results": [{"filename": f.name, "status": "success", "transcription": "test"} 
                           for f in batch_audio_files]
            }))
        ]
        
        mock_session.post.return_value.__aenter__.side_effect = mock_responses
        
        with patch('asyncio.sleep'):  # Speed up the test
            result = await cv_decode.transcribe_batch(mock_session, batch_audio_files, batch_id=0)
        
        assert len(result) == len(batch_audio_files)
        assert all(r["status"] == "success" for r in result)


class TestCSVProcessing:
    """Test CSV file processing functionality."""

    @pytest.mark.unit
    def test_load_csv_and_get_files_success(self, sample_csv_file: Path):
        """Test successful CSV loading."""
        filenames = cv_decode.load_csv_and_get_files(str(sample_csv_file))
        
        assert len(filenames) == 3
        assert all(name.endswith('.mp3') for name in filenames)
        assert 'sample-000000.mp3' in filenames

    @pytest.mark.unit
    def test_load_csv_and_get_files_missing_file(self, temp_dir: Path):
        """Test CSV loading with missing file."""
        missing_file = temp_dir / "missing.csv"
        
        with pytest.raises(FileNotFoundError):
            cv_decode.load_csv_and_get_files(str(missing_file))

    @pytest.mark.unit
    def test_load_csv_and_get_files_add_extension(self, temp_dir: Path):
        """Test CSV loading adds .mp3 extension when missing."""
        csv_content = """filename,text
sample-000000,hello world
sample-000001,test audio"""
        
        csv_file = temp_dir / "test.csv"
        csv_file.write_text(csv_content)
        
        filenames = cv_decode.load_csv_and_get_files(str(csv_file))
        
        assert len(filenames) == 2
        assert all(name.endswith('.mp3') for name in filenames)

    @pytest.mark.unit
    def test_load_csv_and_get_files_handle_paths(self, temp_dir: Path):
        """Test CSV loading handles file paths correctly."""
        csv_content = """filename,text
folder/sample-000000.mp3,hello world
another/folder/sample-000001.mp3,test audio"""
        
        csv_file = temp_dir / "test.csv"
        csv_file.write_text(csv_content)
        
        filenames = cv_decode.load_csv_and_get_files(str(csv_file))
        
        assert len(filenames) == 2
        assert 'sample-000000.mp3' in filenames
        assert 'sample-000001.mp3' in filenames

    @pytest.mark.unit
    def test_save_updated_csv(self, temp_dir: Path, sample_csv_file: Path):
        """Test saving updated CSV with transcriptions."""
        transcriptions = {
            'sample-000000.mp3': 'transcribed text one',
            'sample-000001.mp3': 'transcribed text two',
            'sample-000002.mp3': 'transcribed text three'
        }
        
        output_file = temp_dir / "output.csv"
        
        cv_decode.save_updated_csv(str(sample_csv_file), transcriptions, str(output_file))
        
        # Read the output file and verify
        df = pd.read_csv(output_file)
        assert 'generated_text' in df.columns
        assert len(df) == 3
        assert df['generated_text'].iloc[0] == 'transcribed text one'


class TestBatchProcessing:
    """Test batch processing workflow."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_files_batch_success(self, temp_dir: Path, batch_audio_files: list[Path]):
        """Test successful batch processing."""
        # Create MP3 filenames list
        mp3_filenames = [f.name for f in batch_audio_files]
        
        # Mock successful API responses
        with patch('cv_decode.transcribe_batch') as mock_transcribe:
            mock_transcribe.return_value = [
                {
                    "file": filename,
                    "status": "success",
                    "transcription": f"transcription for {filename}",
                    "audio_duration": "1.0",
                    "processing_time": 0.5,
                    "attempt": 1
                }
                for filename in mp3_filenames
            ]
            
            transcriptions = await cv_decode.process_files_batch(
                mp3_filenames, temp_dir, concurrent=2, batch_size=2
            )
            
            assert len(transcriptions) == len(mp3_filenames)
            assert all(filename in transcriptions for filename in mp3_filenames)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_files_batch_missing_files(self, temp_dir: Path):
        """Test batch processing with missing files."""
        mp3_filenames = ['missing1.mp3', 'missing2.mp3']
        
        transcriptions = await cv_decode.process_files_batch(
            mp3_filenames, temp_dir, concurrent=1, batch_size=1
        )
        
        # Should return empty transcriptions for missing files
        assert len(transcriptions) == len(mp3_filenames)
        assert all(transcriptions[filename] == "" for filename in mp3_filenames)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_files_batch_mixed_results(self, temp_dir: Path, batch_audio_files: list[Path]):
        """Test batch processing with mixed success/failure results."""
        # Create one existing file and add one missing file
        existing_files = [batch_audio_files[0].name]
        missing_files = ['missing.mp3']
        mp3_filenames = existing_files + missing_files
        
        with patch('cv_decode.transcribe_batch') as mock_transcribe:
            # Mock response for existing file only
            mock_transcribe.return_value = [
                {
                    "file": existing_files[0],
                    "status": "success",
                    "transcription": "success transcription",
                }
            ]
            
            transcriptions = await cv_decode.process_files_batch(
                mp3_filenames, temp_dir, concurrent=1, batch_size=1
            )
            
            assert len(transcriptions) == 2
            assert transcriptions[existing_files[0]] == "success transcription"
            assert transcriptions[missing_files[0]] == ""


class TestMainFunction:
    """Test main function and argument parsing."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_main_function_success(self, temp_dir: Path, sample_csv_file: Path, batch_audio_files: list[Path]):
        """Test successful main function execution."""
        # Copy sample CSV to temp directory
        csv_file = temp_dir / "test.csv"
        csv_file.write_text(sample_csv_file.read_text())
        
        # Mock all the dependencies
        with patch('cv_decode.check_api_health', return_value=True), \
             patch('cv_decode.process_files_batch') as mock_process, \
             patch('cv_decode.save_updated_csv'), \
             patch('sys.argv', ['cv-decode.py', '--csv', str(csv_file), '--folder', str(temp_dir), '--n_files', '2']):
            
            # Mock successful processing
            mock_process.return_value = {
                'sample-000000.mp3': 'transcription one',
                'sample-000001.mp3': 'transcription two'
            }
            
            result = await cv_decode.main()
            
            assert result == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_main_function_api_unhealthy(self, temp_dir: Path, sample_csv_file: Path):
        """Test main function with unhealthy API."""
        csv_file = temp_dir / "test.csv"
        csv_file.write_text(sample_csv_file.read_text())
        
        with patch('cv_decode.check_api_health', return_value=False), \
             patch('sys.argv', ['cv-decode.py', '--csv', str(csv_file)]):
            
            result = await cv_decode.main()
            
            assert result == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_main_function_invalid_batch_size(self):
        """Test main function with invalid batch size."""
        with patch('sys.argv', ['cv-decode.py', '--batch_size', '0']):
            result = await cv_decode.main()
            assert result == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_main_function_invalid_n_files(self):
        """Test main function with invalid n_files."""
        with patch('sys.argv', ['cv-decode.py', '--n_files', '-1']):
            result = await cv_decode.main()
            assert result == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_main_function_missing_csv(self, temp_dir: Path):
        """Test main function with missing CSV file."""
        missing_csv = temp_dir / "missing.csv"
        
        with patch('sys.argv', ['cv-decode.py', '--csv', str(missing_csv)]):
            result = await cv_decode.main()
            assert result == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_main_function_missing_folder(self, sample_csv_file: Path, temp_dir: Path):
        """Test main function with missing audio folder."""
        csv_file = temp_dir / "test.csv"
        csv_file.write_text(sample_csv_file.read_text())
        
        missing_folder = temp_dir / "missing_folder"
        
        with patch('cv_decode.check_api_health', return_value=True), \
             patch('sys.argv', ['cv-decode.py', '--csv', str(csv_file), '--folder', str(missing_folder)]):
            
            result = await cv_decode.main()
            assert result == 1


class TestConfigurationLoading:
    """Test environment configuration loading."""

    @pytest.mark.unit
    def test_environment_variable_loading(self, monkeypatch):
        """Test that environment variables are loaded correctly."""
        # Set test environment variables
        monkeypatch.setenv('PORT', '9000')
        monkeypatch.setenv('WORKERS', '8')
        monkeypatch.setenv('DEFAULT_BATCH_SIZE', '10')
        
        # Reload the module to pick up new environment variables
        importlib.reload(cv_decode)
        
        assert cv_decode.API_PORT == 9000
        assert cv_decode.API_WORKERS == 8
        assert cv_decode.DEFAULT_BATCH_SIZE == 10

    @pytest.mark.unit
    def test_default_values(self, monkeypatch):
        """Test default values when environment variables are not set."""
        # Clear relevant environment variables
        for var in ['PORT', 'WORKERS', 'DEFAULT_BATCH_SIZE']:
            monkeypatch.delenv(var, raising=False)
        
        # Reload the module
        importlib.reload(cv_decode)
        
        assert cv_decode.API_PORT == 8001  # default
        assert cv_decode.DEFAULT_BATCH_SIZE == 5  # default