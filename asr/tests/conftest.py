"""
Test configuration and fixtures for ASR microservice tests.
"""
import os
import tempfile
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient
import torch
import torchaudio
import numpy as np

# Import the app for testing
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asr_api import app


@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for the FastAPI app."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_audio_file(temp_dir: Path) -> Path:
    """Create a sample MP3 audio file for testing."""
    # Generate a simple sine wave
    sample_rate = 16000
    duration = 2.0  # seconds
    frequency = 440  # Hz (A4 note)
    
    # Create sine wave
    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
    
    # Save as temporary wav file first
    wav_file = temp_dir / "test_audio.wav"
    torchaudio.save(str(wav_file), audio, sample_rate)
    
    # Convert to mp3 (for now, just use wav file as mp3 for testing)
    mp3_file = temp_dir / "test_audio.mp3"
    mp3_file.write_bytes(wav_file.read_bytes())
    
    return mp3_file


@pytest.fixture
def large_audio_file(temp_dir: Path) -> Path:
    """Create a large audio file for testing file size limits."""
    # Create a large dummy file (> 50MB)
    large_file = temp_dir / "large_audio.mp3"
    with open(large_file, 'wb') as f:
        # Write 60MB of zeros
        f.write(b'0' * (60 * 1024 * 1024))
    
    return large_file


@pytest.fixture
def empty_audio_file(temp_dir: Path) -> Path:
    """Create an empty audio file for testing."""
    empty_file = temp_dir / "empty_audio.mp3"
    empty_file.touch()
    
    return empty_file


@pytest.fixture
def high_sample_rate_audio_file(temp_dir: Path) -> Path:
    """Create an audio file with 44.1kHz sample rate that requires resampling to 16kHz."""
    import torch
    import torchaudio
    
    # Create a synthetic audio signal at 44.1kHz
    sample_rate = 44100
    duration = 2.0  # 2 seconds
    frequency = 440  # A4 note
    
    # Generate sine wave
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
    
    # Save as audio file (MP3 format)
    audio_file = temp_dir / "high_sample_rate_audio.mp3"
    torchaudio.save(str(audio_file), waveform, sample_rate)
    
    return audio_file


@pytest.fixture
def stereo_audio_file(temp_dir: Path) -> Path:
    """Create a stereo audio file for testing stereo-to-mono conversion."""
    import torch
    import torchaudio
    
    # Create a synthetic stereo audio signal at 16kHz
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    frequency_left = 440   # A4 note in left channel
    frequency_right = 523  # C5 note in right channel
    
    # Generate different sine waves for left and right channels
    t = torch.linspace(0, duration, int(sample_rate * duration))
    left_channel = torch.sin(2 * torch.pi * frequency_left * t)
    right_channel = torch.sin(2 * torch.pi * frequency_right * t)
    
    # Create stereo waveform (2 channels)
    waveform = torch.stack([left_channel, right_channel], dim=0)
    
    # Save as audio file
    audio_file = temp_dir / "stereo_audio.mp3"
    torchaudio.save(str(audio_file), waveform, sample_rate)
    
    return audio_file


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    test_env = {
        'MAX_FILE_SIZE_MB': '50',
        'REQUEST_TIMEOUT_SECONDS': '60',
        'INFERENCE_TIMEOUT_SECONDS': '30',
        'LOG_LEVEL': 'ERROR',
        'HOST': '0.0.0.0',
        'PORT': '8001',
        'WORKERS': '1'
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing cv-decode functionality."""
    return """filename,text,up_votes,down_votes,age,gender,accent,duration
sample-000000.mp3,hello world,1,0,,,,
sample-000001.mp3,this is a test,2,0,,,,
sample-000002.mp3,another test file,1,0,,,,"""


@pytest.fixture
def sample_csv_file(temp_dir: Path, sample_csv_data: str) -> Path:
    """Create a sample CSV file for testing."""
    csv_file = temp_dir / "test_data.csv"
    csv_file.write_text(sample_csv_data)
    return csv_file


@pytest.fixture
def mock_model_response():
    """Mock model response for testing."""
    return {
        "transcription": "hello world",
        "duration": "2.0"
    }


@pytest.fixture
def batch_audio_files(temp_dir: Path) -> list[Path]:
    """Create multiple audio files for batch testing."""
    files = []
    for i in range(3):
        # Generate different frequency sine waves
        sample_rate = 16000
        duration = 1.0
        frequency = 440 + (i * 100)  # Different frequencies
        
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
        
        wav_file = temp_dir / f"test_audio_{i}.wav"
        torchaudio.save(str(wav_file), audio, sample_rate)
        
        mp3_file = temp_dir / f"test_audio_{i}.mp3"
        mp3_file.write_bytes(wav_file.read_bytes())
        files.append(mp3_file)
    
    return files


@pytest.fixture
def large_batch_audio_files(temp_dir: Path) -> list[Path]:
    """Create 50 audio files for large batch testing."""
    files = []
    for i in range(50):
        # Generate different frequency sine waves
        sample_rate = 16000
        duration = 1.0
        frequency = 440 + (i * 50)  # Different frequencies
        
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
        
        wav_file = temp_dir / f"test_audio_large_{i}.wav"
        torchaudio.save(str(wav_file), audio, sample_rate)
        
        mp3_file = temp_dir / f"test_audio_large_{i}.mp3"
        mp3_file.write_bytes(wav_file.read_bytes())
        files.append(mp3_file)
    
    return files


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Configure logging for tests."""
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("torchaudio").setLevel(logging.ERROR)


@pytest.fixture
def mock_processor_and_model(mocker):
    """Mock the Wav2Vec2 processor and model for faster testing."""
    # Mock processor
    mock_processor = mocker.MagicMock()
    mock_processor.return_value.input_values = torch.randn(1, 32000)
    mock_processor.batch_decode.return_value = ["hello world"]
    
    # Mock model
    mock_model = mocker.MagicMock()
    mock_logits = torch.randn(1, 100, 32)  # batch_size, seq_len, vocab_size
    mock_model.return_value.logits = mock_logits
    mock_model.to.return_value = mock_model
    
    # Patch the global processor and model
    mocker.patch('asr_api.processor', mock_processor)
    mocker.patch('asr_api.model', mock_model)
    
    return mock_processor, mock_model


class MockHttpResponse:
    """Mock HTTP response for aiohttp tests."""
    def __init__(self, status=200, json_data=None):
        self.status = status
        self._json_data = json_data or {}
        
    async def json(self):
        return self._json_data
    
    async def text(self):
        return "Mock response text"
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class MockHttpSession:
    """Mock HTTP session for aiohttp tests."""
    def __init__(self, response_status=200, response_json=None, should_raise=None, response_sequence=None):
        self.response_status = response_status
        self.response_json = response_json or {}
        self.should_raise = should_raise
        self.response_sequence = response_sequence or []  # List of (status, json, exception) tuples
        self.call_count = 0
        self.get_called_with = None
        self.post_called_with = None
        
    def get(self, url, **kwargs):
        self.get_called_with = (url, kwargs)
        return self._get_response()
        
    def post(self, url, **kwargs):
        self.post_called_with = (url, kwargs)
        return self._get_response()
        
    def _get_response(self):
        """Get response based on call count and sequence."""
        if self.response_sequence and self.call_count < len(self.response_sequence):
            status, json_data, exception = self.response_sequence[self.call_count]
            self.call_count += 1
            if exception:
                raise exception
            return MockHttpResponse(status, json_data)
        else:
            # Use default response
            if self.should_raise:
                raise self.should_raise
            return MockHttpResponse(self.response_status, self.response_json)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


@pytest.fixture
def mock_http_session():
    """Create a mock HTTP session for testing."""
    return MockHttpSession


@pytest.fixture
def mock_successful_http_session():
    """Create a mock HTTP session that returns successful responses."""
    return MockHttpSession(response_status=200, response_json={"message": "pong"})


@pytest.fixture
def mock_failed_http_session():
    """Create a mock HTTP session that returns error responses."""
    return MockHttpSession(response_status=500)


@pytest.fixture
def cv_decode_module():
    """Import and return the cv-decode module for testing."""
    import importlib.util
    
    cv_decode_path = Path(__file__).parent.parent / "cv-decode.py"
    spec = importlib.util.spec_from_file_location("cv_decode", cv_decode_path)
    cv_decode = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cv_decode)
    
    return cv_decode