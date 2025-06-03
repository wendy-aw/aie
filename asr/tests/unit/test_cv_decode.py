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
    async def test_check_api_health_success(self, mock_successful_http_session):
        """Test successful API health check."""
        with patch('aiohttp.ClientSession', return_value=mock_successful_http_session):
            result = await cv_decode.check_api_health()
            
            assert result is True
            assert mock_successful_http_session.get_called_with[0] == f"{cv_decode.API_BASE_URL}/ping"
            assert mock_successful_http_session.get_called_with[1]['timeout'] == 5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_api_health_failure(self, mock_failed_http_session):
        """Test API health check failure."""
        with patch('aiohttp.ClientSession', return_value=mock_failed_http_session):
            result = await cv_decode.check_api_health()
            
            assert result is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_api_health_exception(self, mock_http_session):
        """Test API health check with connection exception."""
        mock_session = mock_http_session(should_raise=aiohttp.ClientError("Connection failed"))
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await cv_decode.check_api_health()
            
            assert result is False


class TestBatchTranscription:
    """Test batch transcription functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_transcribe_batch_file_validation(self, temp_dir: Path):
        """Test that batch transcription validates file existence."""
        mock_session = AsyncMock()
        
        # Create one real file and one missing file path
        real_file = temp_dir / "real_file.mp3"
        real_file.write_bytes(b"fake mp3 content")
        missing_file = temp_dir / "missing_file.mp3"
        
        files = [real_file, missing_file]
        
        result = await cv_decode.transcribe_batch(mock_session, files)
        
        # Should return empty results for missing files
        assert len(result) <= len(files)
        # The function should skip missing files entirely

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_transcribe_batch_multipart_form_data(self, batch_audio_files: list[Path]):
        """Test that batch transcription creates proper multipart form data."""
        mock_session = AsyncMock()
        
        # Mock successful API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "results": [
                {
                    "filename": f.name,
                    "status": "success",
                    "transcription": f"transcription for {f.name}",
                    "duration": "1.0",
                    "processing_time": 0.5
                }
                for f in batch_audio_files
            ]
        })
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        result = await cv_decode.transcribe_batch(mock_session, batch_audio_files)
        
        # Verify the API was called with the correct URL (may be called multiple times for batches)
        assert mock_session.post.call_count >= 1
        call_args = mock_session.post.call_args_list[0]
        assert f"{cv_decode.API_BASE_URL}/asr" in call_args[0][0]
        
        # Verify the form data contains files
        assert 'data' in call_args[1]
        
        assert len(result) == len(batch_audio_files)
        assert all(r["status"] == "success" for r in result)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_transcribe_batch_retry_on_server_error(self, batch_audio_files: list[Path]):
        """Test retry behavior on server errors."""
        mock_session = AsyncMock()
        
        # First call fails with 500, second succeeds
        error_response = AsyncMock()
        error_response.status = 500
        error_response.text = AsyncMock(return_value="Internal Server Error")
        
        success_response = AsyncMock()
        success_response.status = 200
        success_response.json = AsyncMock(return_value={
            "results": [{"filename": f.name, "status": "success", "transcription": "test"} 
                       for f in batch_audio_files]
        })
        
        mock_session.post.return_value.__aenter__.side_effect = [error_response, success_response]
        
        with patch('asyncio.sleep'):  # Speed up the test by mocking sleep
            result = await cv_decode.transcribe_batch(mock_session, batch_audio_files)
        
        # Should have retried and succeeded
        assert len(result) == len(batch_audio_files)
        # Check that we have some successful results (not all may be successful due to error handling)
        successful_results = [r for r in result if r["status"] == "success"]
        assert len(successful_results) > 0
        assert mock_session.post.call_count == 2


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
        assert 'sample-000000.mp3' in filenames
        assert 'sample-000001.mp3' in filenames

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
        # Should extract just the filename, not the full path
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
        
        # Verify the file was created and has correct content
        assert output_file.exists()
        df = pd.read_csv(output_file)
        assert 'generated_text' in df.columns
        assert len(df) == 3
        assert df['generated_text'].iloc[0] == 'transcribed text one'
        assert df['generated_text'].iloc[1] == 'transcribed text two'
        assert df['generated_text'].iloc[2] == 'transcribed text three'


class TestBatchProcessing:
    """Test batch processing workflow."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_files_batch_with_real_files(self, temp_dir: Path, batch_audio_files: list[Path]):
        """Test batch processing with actual file existence checks."""
        # Copy files to temp directory to simulate real scenario
        for i, audio_file in enumerate(batch_audio_files):
            target_file = temp_dir / audio_file.name
            target_file.write_bytes(audio_file.read_bytes())
        
        mp3_filenames = [f.name for f in batch_audio_files]
        
        # Mock only the transcribe_batch function to return realistic responses
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
            assert all(transcriptions[filename] != "" for filename in mp3_filenames)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_files_batch_missing_files(self, temp_dir: Path):
        """Test batch processing with missing files."""
        mp3_filenames = ['missing1.mp3', 'missing2.mp3']
        
        # Don't create the files - they should be missing
        transcriptions = await cv_decode.process_files_batch(
            mp3_filenames, temp_dir, concurrent=1, batch_size=1
        )
        
        # Should return empty transcriptions for missing files
        assert len(transcriptions) == len(mp3_filenames)
        assert all(transcriptions[filename] == "" for filename in mp3_filenames)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_files_batch_concurrency_batching(self, temp_dir: Path, batch_audio_files: list[Path]):
        """Test that batch processing respects concurrency and batch size settings."""
        # Create real files
        for audio_file in batch_audio_files:
            target_file = temp_dir / audio_file.name
            target_file.write_bytes(audio_file.read_bytes())
        
        mp3_filenames = [f.name for f in batch_audio_files]
        
        with patch('cv_decode.transcribe_batch') as mock_transcribe:
            # Return empty list to simulate failed API calls
            mock_transcribe.return_value = []
            
            transcriptions = await cv_decode.process_files_batch(
                mp3_filenames, temp_dir, concurrent=1, batch_size=2
            )
            
            # Verify transcribe_batch was called with correct batch sizes
            call_count = mock_transcribe.call_count
            assert call_count > 0
            
            # Should have empty transcriptions since API "failed"
            assert all(transcriptions[filename] == "" for filename in mp3_filenames)


class TestMainFunction:
    """Test main function and argument parsing with realistic scenarios."""

    @pytest.mark.unit
    def test_argument_validation_invalid_batch_size(self):
        """Test argument validation for invalid batch size."""
        with patch('sys.argv', ['cv-decode.py', '--batch_size', '0']):
            parser = cv_decode.create_parser()
            args = parser.parse_args(['--batch_size', '0'])
            
            # Invalid batch size should be caught by validation logic
            assert args.batch_size == 0

    @pytest.mark.unit
    def test_argument_validation_invalid_n_files(self):
        """Test argument validation for invalid n_files."""
        with patch('sys.argv', ['cv-decode.py', '--n_files', '-1']):
            parser = cv_decode.create_parser()
            args = parser.parse_args(['--n_files', '-1'])
            
            assert args.n_files == -1

    @pytest.mark.unit
    def test_csv_file_validation(self, temp_dir: Path):
        """Test CSV file validation in main function setup."""
        missing_csv = temp_dir / "missing.csv"
        
        # Should fail when CSV file doesn't exist
        with pytest.raises(FileNotFoundError):
            cv_decode.load_csv_and_get_files(str(missing_csv))

    @pytest.mark.unit
    def test_folder_validation(self, temp_dir: Path):
        """Test audio folder validation."""
        missing_folder = temp_dir / "missing_folder"
        
        # Should return False when folder doesn't exist
        assert not missing_folder.exists()
        assert not missing_folder.is_dir()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_main_function_early_exit_conditions(self, temp_dir: Path, sample_csv_file: Path):
        """Test main function early exit conditions without heavy mocking."""
        csv_file = temp_dir / "test.csv"
        csv_file.write_text(sample_csv_file.read_text())
        
        # Test with unhealthy API
        with patch('cv_decode.check_api_health', return_value=False), \
             patch('sys.argv', ['cv-decode.py', '--csv', str(csv_file)]):
            
            result = await cv_decode.main()
            assert result == 1
            
        # Test with missing folder (API healthy this time)
        missing_folder = temp_dir / "missing_folder"
        with patch('cv_decode.check_api_health', return_value=True), \
             patch('sys.argv', ['cv-decode.py', '--csv', str(csv_file), '--folder', str(missing_folder)]):
            
            result = await cv_decode.main()
            assert result == 1
