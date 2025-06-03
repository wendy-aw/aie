"""
Unit tests for audio processing functions in asr_api.py
"""
import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import UploadFile, HTTPException
import io

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from asr_api import _process_audio_file, _run_inference


class TestAudioProcessing:
    """Test audio processing functions."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_audio_file_success(self, sample_audio_file: Path):
        """Test successful audio file processing."""
        
        # Create UploadFile from sample audio
        with open(sample_audio_file, 'rb') as f:
            content = f.read()
        
        upload_file = UploadFile(
            filename="test_audio.mp3",
            file=io.BytesIO(content)
        )
        
        result = await _process_audio_file(upload_file)
        
        assert result["status"] == "success"
        assert "waveform" in result
        assert "duration" in result
        assert "processing_time" in result
        assert isinstance(result["waveform"], torch.Tensor)
        assert result["duration"] > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_audio_file_invalid_format(self):
        """Test audio file processing with invalid format."""
        upload_file = UploadFile(
            filename="test_audio.txt",
            file=io.BytesIO(b"not audio data")
        )
        
        result = await _process_audio_file(upload_file)
        
        assert result["status"] == "error"
        assert "Only MP3" in result["error"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_audio_file_too_large(self, temp_dir: Path):
        """Test audio file processing with file too large."""
        # Create a large file
        large_content = b'0' * (60 * 1024 * 1024)  # 60MB
        
        upload_file = UploadFile(
            filename="large_audio.mp3",
            file=io.BytesIO(large_content)
        )
        
        result = await _process_audio_file(upload_file)
        
        assert result["status"] == "error"
        assert "File too large" in result["error"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_audio_file_corrupted(self):
        """Test audio file processing with corrupted file."""
        upload_file = UploadFile(
            filename="corrupted_audio.mp3",
            file=io.BytesIO(b'fake corrupted mp3 content')
        )
        
        with patch('torchaudio.load') as mock_load:
            mock_load.side_effect = RuntimeError("Invalid audio file")
            
            result = await _process_audio_file(upload_file)
            
            assert result["status"] == "error"
            assert "Invalid audio file" in result["error"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_audio_file_empty_audio(self, empty_audio_file: Path):
        """Test audio file processing with empty audio data."""
        with open(empty_audio_file, 'rb') as f:
            content = f.read()
        
        upload_file = UploadFile(
            filename="empty_audio.mp3",
            file=io.BytesIO(content)
        )
        with patch('torchaudio.load') as mock_load:
            mock_load.side_effect = Exception("Empty audio file")
            result = await _process_audio_file(upload_file)
            
            assert result["status"] == "error"
            # Should fail when trying to load the empty file
            assert "error" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_audio_file_resampling(self, high_sample_rate_audio_file: Path):
        """Test audio file processing with resampling."""
        
        with open(high_sample_rate_audio_file, 'rb') as f:
            content = f.read()
        
        upload_file = UploadFile(
            filename="high_sample_rate_audio.mp3",
            file=io.BytesIO(content)
        )
        
        result = await _process_audio_file(upload_file)
        
        assert result["status"] == "success"
        assert "waveform" in result
        assert "duration" in result
        
        # Verify the waveform was resampled to 16kHz
        waveform = result["waveform"]
        duration = result["duration"]
        assert isinstance(waveform, torch.Tensor)
        
        # Calculate the effective sample rate from waveform length and duration
        # If resampling worked correctly, it should be 16000 Hz
        effective_sample_rate = waveform.shape[0] / duration
        assert abs(effective_sample_rate - 16000) < 100  # Allow small tolerance

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_audio_file_stereo_to_mono(self, stereo_audio_file: Path):
        """Test audio file processing with stereo to mono conversion."""
        
        with open(stereo_audio_file, 'rb') as f:
            content = f.read()
        
        upload_file = UploadFile(
            filename="stereo_audio.mp3",
            file=io.BytesIO(content)
        )
        
        result = await _process_audio_file(upload_file)
        
        assert result["status"] == "success"
        assert "waveform" in result
        assert "duration" in result
        
        # Verify the result is mono (1-dimensional waveform)
        waveform = result["waveform"]
        assert isinstance(waveform, torch.Tensor)
        # After stereo-to-mono conversion, waveform should be 1D
        assert waveform.dim() == 1
        # Verify reasonable duration (approximately 2 seconds)
        assert 1.5 < result["duration"] < 2.5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_inference(self, mock_processor_and_model):
        """Test model inference function."""
        mock_processor, mock_model = mock_processor_and_model
        
        input_values = torch.randn(1, 32000)
        
        logits = await _run_inference(input_values)
        
        assert isinstance(logits, torch.Tensor)
        mock_model.assert_called_once_with(input_values)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_audio_file_temp_file_cleanup(self, sample_audio_file: Path):
        """Test that temporary files are properly cleaned up."""
        
        with open(sample_audio_file, 'rb') as f:
            content = f.read()
        
        upload_file = UploadFile(
            filename="test_audio.mp3",
            file=io.BytesIO(content)
        )
        
        # Track the specific temp file path that gets created
        created_temp_files = []
        
        # Patch tempfile.NamedTemporaryFile to capture the file path
        original_named_temp_file = tempfile.NamedTemporaryFile
        
        def track_temp_file(*args, **kwargs):
            temp_file = original_named_temp_file(*args, **kwargs)
            created_temp_files.append(temp_file.name)
            return temp_file
        
        with patch('tempfile.NamedTemporaryFile', side_effect=track_temp_file):
            result = await _process_audio_file(upload_file)
            
            assert result["status"] == "success"
            
            # Verify the specific temp file was cleaned up
            assert len(created_temp_files) == 1
            temp_file_path = Path(created_temp_files[0])
            assert not temp_file_path.exists()  # File should be deleted

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_audio_file_exception_cleanup(self, sample_audio_file: Path):
        """Test that temporary files are cleaned up even on exceptions."""
        with open(sample_audio_file, 'rb') as f:
            content = f.read()
        
        upload_file = UploadFile(
            filename="test_audio.mp3",
            file=io.BytesIO(content)
        )
        
        # Track the specific temp file path that gets created
        created_temp_files = []
        
        # Patch tempfile.NamedTemporaryFile to capture the file path
        original_named_temp_file = tempfile.NamedTemporaryFile
        
        def track_temp_file(*args, **kwargs):
            temp_file = original_named_temp_file(*args, **kwargs)
            created_temp_files.append(temp_file.name)
            return temp_file
        
        with patch('tempfile.NamedTemporaryFile', side_effect=track_temp_file), \
             patch('torchaudio.load', side_effect=Exception("Unexpected error")):
            
            result = await _process_audio_file(upload_file)
            
            assert result["status"] == "error"
            
            # Verify the specific temp file was cleaned up even on error
            assert len(created_temp_files) == 1
            temp_file_path = Path(created_temp_files[0])
            assert not temp_file_path.exists()  # File should be deleted even on error

    @pytest.mark.unit
    def test_invalid_sample_rate(self, sample_audio_file: Path):
        """Test audio processing with invalid sample rate."""
        
        with open(sample_audio_file, 'rb') as f:
            content = f.read()
        
        upload_file = UploadFile(
            filename="test_audio.mp3",
            file=io.BytesIO(content)
        )
        
        async def test_func():
            # Mock torchaudio.load to return invalid sample rate
            with patch('torchaudio.load', return_value=(torch.randn(1, 32000), -1)):
                result = await _process_audio_file(upload_file)
                
                assert result["status"] == "error"
                assert "Invalid sample rate" in result["error"]
        
        # Run the async test
        import asyncio
        asyncio.run(test_func())