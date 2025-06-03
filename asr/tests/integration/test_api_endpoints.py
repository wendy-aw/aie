"""
Integration tests for ASR API endpoints.
"""
import pytest
import io
from pathlib import Path
from fastapi.testclient import TestClient
from httpx import AsyncClient


class TestPingEndpoint:
    """Test ping endpoint."""

    def test_ping_endpoint_sync(self, client: TestClient):
        """Test ping endpoint synchronously."""
        response = client.get("/ping")
        
        assert response.status_code == 200
        assert response.json() == {"message": "pong"}

    @pytest.mark.asyncio
    async def test_ping_endpoint_async(self, async_client: AsyncClient):
        """Test ping endpoint asynchronously."""
        response = await async_client.get("/ping")
        
        assert response.status_code == 200
        assert response.json() == {"message": "pong"}


class TestASREndpoint:
    """Test ASR transcription endpoint."""

    def test_asr_endpoint_no_files(self, client: TestClient):
        """Test ASR endpoint with no files provided."""
        response = client.post("/asr")
        
        assert response.status_code == 400
        assert "No files provided" in response.json()["detail"]

    def test_asr_endpoint_both_file_and_files(self, client: TestClient, sample_audio_file: Path):
        """Test ASR endpoint with both file and files parameters."""
        with open(sample_audio_file, 'rb') as f:
            files = {
                'file': ('test.mp3', f, 'audio/mpeg'),
                'files': ('test2.mp3', f, 'audio/mpeg')
            }
            response = client.post("/asr", files=files)
        
        assert response.status_code == 400
        assert "Send either 'file' or 'files', not both" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_asr_single_file_success(self, async_client: AsyncClient, sample_audio_file: Path, mock_processor_and_model):
        """Test successful single file transcription."""
        mock_processor, mock_model = mock_processor_and_model
        
        with open(sample_audio_file, 'rb') as f:
            files = {'file': ('test.mp3', f, 'audio/mpeg')}
            
            with pytest.patch('asr_api.torchaudio.load') as mock_load:
                mock_load.return_value = (pytest.torch.randn(1, 32000), 16000)
                
                response = await async_client.post("/asr", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "transcription" in data
        assert "duration" in data

    def test_asr_single_file_invalid_format(self, client: TestClient, temp_dir: Path):
        """Test single file transcription with invalid format."""
        # Create a text file instead of MP3
        text_file = temp_dir / "test.txt"
        text_file.write_text("This is not an MP3 file")
        
        with open(text_file, 'rb') as f:
            files = {'file': ('test.txt', f, 'text/plain')}
            response = client.post("/asr", files=files)
        
        assert response.status_code == 415
        assert "Only MP3 audio files are supported" in response.json()["detail"]

    def test_asr_single_file_too_large(self, client: TestClient, large_audio_file: Path):
        """Test single file transcription with file too large."""
        with open(large_audio_file, 'rb') as f:
            files = {'file': ('large.mp3', f, 'audio/mpeg')}
            response = client.post("/asr", files=files)
        
        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_asr_batch_success(self, async_client: AsyncClient, batch_audio_files: list[Path], mock_processor_and_model):
        """Test successful batch file transcription."""
        mock_processor, mock_model = mock_processor_and_model
        
        files = []
        for i, audio_file in enumerate(batch_audio_files):
            with open(audio_file, 'rb') as f:
                files.append(('files', (f'test_{i}.mp3', f.read(), 'audio/mpeg')))
        
        with pytest.patch('asr_api.torchaudio.load') as mock_load:
            mock_load.return_value = (pytest.torch.randn(1, 16000), 16000)
            
            response = await async_client.post("/asr", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == len(batch_audio_files)
        
        for result in data["results"]:
            assert "filename" in result
            assert "status" in result
            if result["status"] == "success":
                assert "transcription" in result
                assert "duration" in result

    @pytest.mark.asyncio
    async def test_asr_batch_mixed_results(self, async_client: AsyncClient, temp_dir: Path, mock_processor_and_model):
        """Test batch transcription with mixed success/failure results."""
        mock_processor, mock_model = mock_processor_and_model
        
        # Create one valid file and one invalid file
        valid_file = temp_dir / "valid.mp3"
        valid_file.write_bytes(b"fake mp3 data")
        
        invalid_file = temp_dir / "invalid.txt"
        invalid_file.write_text("not audio")
        
        files = [
            ('files', ('valid.mp3', valid_file.read_bytes(), 'audio/mpeg')),
            ('files', ('invalid.txt', invalid_file.read_bytes(), 'text/plain'))
        ]
        
        with pytest.patch('asr_api.torchaudio.load') as mock_load:
            mock_load.return_value = (pytest.torch.randn(1, 16000), 16000)
            
            response = await async_client.post("/asr", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2
        
        # Check that we have both success and error results
        statuses = [result["status"] for result in data["results"]]
        assert "success" in statuses
        assert "error" in statuses

    def test_asr_endpoint_empty_file(self, client: TestClient, empty_audio_file: Path):
        """Test ASR endpoint with empty file."""
        with open(empty_audio_file, 'rb') as f:
            files = {'file': ('empty.mp3', f, 'audio/mpeg')}
            response = client.post("/asr", files=files)
        
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_asr_endpoint_timeout(self, async_client: AsyncClient, sample_audio_file: Path):
        """Test ASR endpoint timeout handling."""
        with open(sample_audio_file, 'rb') as f:
            files = {'file': ('test.mp3', f, 'audio/mpeg')}
            
            # Mock a timeout scenario
            with pytest.patch('asr_api.with_timeout') as mock_timeout:
                mock_timeout.side_effect = pytest.HTTPException(status_code=408, detail="Request timeout")
                
                response = await async_client.post("/asr", files=files)
        
        assert response.status_code == 408

    def test_asr_endpoint_server_error(self, client: TestClient, sample_audio_file: Path):
        """Test ASR endpoint server error handling."""
        with open(sample_audio_file, 'rb') as f:
            files = {'file': ('test.mp3', f, 'audio/mpeg')}
            
            # Mock an unexpected server error
            with pytest.patch('asr_api._process_asr_request') as mock_process:
                mock_process.side_effect = Exception("Unexpected error")
                
                response = client.post("/asr", files=files)
        
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_asr_endpoint_concurrent_requests(self, async_client: AsyncClient, sample_audio_file: Path, mock_processor_and_model):
        """Test multiple concurrent requests to ASR endpoint."""
        mock_processor, mock_model = mock_processor_and_model
        
        import asyncio
        
        async def make_request():
            with open(sample_audio_file, 'rb') as f:
                files = {'file': ('test.mp3', f, 'audio/mpeg')}
                
                with pytest.patch('asr_api.torchaudio.load') as mock_load:
                    mock_load.return_value = (pytest.torch.randn(1, 32000), 16000)
                    
                    return await async_client.post("/asr", files=files)
        
        # Make 3 concurrent requests
        tasks = [make_request() for _ in range(3)]
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "transcription" in data
            assert "duration" in data


class TestErrorHandling:
    """Test error handling across the API."""

    def test_404_endpoint(self, client: TestClient):
        """Test 404 error for non-existent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client: TestClient):
        """Test 405 error for wrong HTTP method."""
        response = client.get("/asr")  # Should be POST
        assert response.status_code == 405

    @pytest.mark.asyncio
    async def test_malformed_request(self, async_client: AsyncClient):
        """Test malformed request handling."""
        # Send invalid multipart data
        response = await async_client.post(
            "/asr",
            headers={"Content-Type": "multipart/form-data"},
            content=b"invalid multipart data"
        )
        
        assert response.status_code == 422


class TestHealthAndMetrics:
    """Test health and monitoring endpoints."""

    def test_ping_response_time(self, client: TestClient):
        """Test that ping endpoint responds quickly."""
        import time
        
        start_time = time.time()
        response = client.get("/ping")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond in less than 1 second

    @pytest.mark.asyncio
    async def test_ping_concurrent_requests(self, async_client: AsyncClient):
        """Test ping endpoint under concurrent load."""
        import asyncio
        
        async def ping_request():
            return await async_client.get("/ping")
        
        # Make 10 concurrent ping requests
        tasks = [ping_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            assert response.json() == {"message": "pong"}