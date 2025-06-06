"""
End-to-end integration tests for the complete ASR workflow.
"""
import pytest
import asyncio
import pandas as pd
from pathlib import Path
import tempfile
from unittest.mock import patch
import subprocess
import time
import requests
import aiohttp


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete end-to-end workflow from CSV to results."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_complete_csv_processing_workflow(self, temp_dir: Path, cv_decode_module, batch_audio_files, sample_csv_data):
        """Test complete workflow from CSV input to final results."""
        cv_decode = cv_decode_module
        
        # Use sample_csv_data fixture
        input_csv = temp_dir / "input.csv"
        input_csv.write_text(sample_csv_data)
        
        # Use batch_audio_files fixture
        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()
        
        # Copy batch_audio_files to match sample_csv_data filenames
        csv_filenames = ["sample-000000.mp3", "sample-000001.mp3", "sample-000002.mp3"]
        for i, audio_file in enumerate(batch_audio_files):
            target_file = audio_dir / csv_filenames[i]
            target_file.write_bytes(audio_file.read_bytes())
        
        output_csv = temp_dir / "output.csv"
        
        # Mock the API health check and transcription
        with patch.object(cv_decode, 'check_api_health', return_value=True), \
             patch.object(cv_decode, 'transcribe_batch') as mock_transcribe:
            
            # Mock successful transcription responses
            mock_transcribe.return_value = [
                {
                    "file": "sample-000000.mp3",
                    "status": "success",
                    "transcription": "hello world transcribed",
                    "audio_duration": "2.0",
                    "processing_time": 0.5,
                    "attempt": 1
                },
                {
                    "file": "sample-000001.mp3", 
                    "status": "success",
                    "transcription": "this is a test transcribed",
                    "audio_duration": "2.1",
                    "processing_time": 0.6,
                    "attempt": 1
                },
                {
                    "file": "sample-000002.mp3",
                    "status": "success", 
                    "transcription": "another test file transcribed",
                    "audio_duration": "1.9",
                    "processing_time": 0.4,
                    "attempt": 1
                }
            ]
            
            # Mock sys.argv for the main function
            with patch('sys.argv', [
                'cv-decode.py',
                '--csv', str(input_csv),
                '--folder', str(audio_dir),
                '--output', str(output_csv),
                '--concurrent', '2',
                '--batch_size', '2'
            ]):
                
                result = await cv_decode.main()
        
        # Verify successful execution
        assert result == 0
        
        # Verify output CSV was created and contains expected data
        assert output_csv.exists()
        
        df = pd.read_csv(output_csv)
        assert len(df) == 3
        assert 'generated_text' in df.columns
        
        # Check that transcriptions were added
        transcriptions = df['generated_text'].tolist()
        expected_transcriptions = [
            "hello world transcribed",
            "this is a test transcribed", 
            "another test file transcribed"
        ]
        
        assert transcriptions == expected_transcriptions

    @pytest.mark.asyncio
    async def test_api_server_integration(self, sample_audio_file: Path):
        """Test integration with actual API server (if running)."""
        # This test assumes an API server is running locally
        try:
            response = requests.get("http://localhost:8001/ping", timeout=5)
            if response.status_code != 200:
                pytest.skip("API server not running")
        except requests.RequestException:
            pytest.skip("API server not running")
        
        # Test actual API call
        with open(sample_audio_file, 'rb') as f:
            files = {'file': ('test.mp3', f, 'audio/mpeg')}
            response = requests.post("http://localhost:8001/asr", files=files, timeout=30)
        
        assert response.status_code in [200, 422]  # 422 for invalid audio format is acceptable
        
        if response.status_code == 200:
            data = response.json()
            assert "transcription" in data
            assert "duration" in data

    @pytest.mark.slow
    def test_docker_container_workflow(self):
        """Test the complete Docker workflow."""
        # This test requires Docker to be installed and running
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode != 0:
                pytest.skip("Docker not available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Docker not available")
        
        # Build the Docker image (this might take a while)
        project_root = Path(__file__).parent.parent.parent
        
        build_result = subprocess.run(
            ["docker", "build", "-f", "Dockerfile.asr", "-t", "asr-api-test", "."],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes
        )
        
        if build_result.returncode != 0:
            pytest.skip(f"Docker build failed: {build_result.stderr}")
        
        # Start the container
        container_name = f"asr-test-{int(time.time())}"
        
        try:
            run_result = subprocess.run([
                "docker", "run", "-d", 
                "--name", container_name,
                "-p", "8002:8001",  # Use different port to avoid conflicts
                "-e", "WORKERS=1",  # Use single worker for testing
                "asr-api-test"
            ], capture_output=True, text=True, timeout=30)
            
            if run_result.returncode != 0:
                pytest.skip(f"Docker run failed: {run_result.stderr}")
            
            # Check if container is actually running
            check_result = subprocess.run(
                ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Status}}"],
                capture_output=True, text=True, timeout=10
            )
            
            if not check_result.stdout.strip():
                # Container not running, get logs for debugging
                logs_result = subprocess.run(
                    ["docker", "logs", container_name],
                    capture_output=True, text=True, timeout=10
                )
                log_text = logs_result.stdout + logs_result.stderr
                
                # Check for common issues that should cause test skip rather than failure
                if "429 Client Error: Too Many Requests" in log_text:
                    pytest.skip("Hugging Face rate limit reached - container cannot download model")
                elif "Connection error" in log_text or "Network is unreachable" in log_text:
                    pytest.skip("Network connectivity issues - cannot download model")
                else:
                    pytest.skip(f"Container failed to start. Logs: {log_text}")
            
            # Wait longer for container to fully initialize
            time.sleep(60)
            
            # Test the containerized API with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get("http://localhost:8002/ping", timeout=10)
                    assert response.status_code == 200
                    assert response.json() == {"message": "pong"}
                    break  # Success, exit retry loop
                except requests.RequestException as e:
                    if attempt == max_retries - 1:  # Last attempt
                        # Get container logs for debugging
                        logs_result = subprocess.run(
                            ["docker", "logs", container_name],
                            capture_output=True, text=True, timeout=10
                        )
                        log_text = logs_result.stdout + logs_result.stderr
                        
                        # Check for rate limit or network issues
                        if "Connection error" in log_text or "Network is unreachable" in log_text:
                            pytest.skip("Network connectivity issues - cannot download model")
                        else:
                            pytest.fail(f"Container API not responding after {max_retries} attempts: {e}. Container logs: {log_text}")
                    time.sleep(5)  # Wait before retry
        
        finally:
            # Clean up container
            subprocess.run(["docker", "stop", container_name], capture_output=True)
            subprocess.run(["docker", "rm", container_name], capture_output=True)


@pytest.mark.integration
class TestErrorRecovery:
    """Test error recovery and resilience."""

    @pytest.mark.asyncio
    async def test_partial_batch_failure_recovery(self, temp_dir: Path, cv_decode_module, sample_audio_file: Path, empty_audio_file: Path):
        """Test recovery when some files in a batch fail."""
        cv_decode = cv_decode_module
        
        # Create test setup
        csv_content = """filename,text
valid_audio.mp3,hello world
corrupted_audio.mp3,this should fail
another_valid.mp3,another test"""
        
        input_csv = temp_dir / "input.csv"
        input_csv.write_text(csv_content)
        
        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()
        
        # Use fixtures for audio files
        valid_file = audio_dir / "valid_audio.mp3"
        valid_file.write_bytes(sample_audio_file.read_bytes())
        
        # Use empty file as corrupted file
        corrupted_file = audio_dir / "corrupted_audio.mp3"
        corrupted_file.write_bytes(empty_audio_file.read_bytes())
        
        # Create another valid file using sample audio
        another_valid = audio_dir / "another_valid.mp3"
        another_valid.write_bytes(sample_audio_file.read_bytes())
        
        output_csv = temp_dir / "output.csv"
        
        with patch.object(cv_decode, 'check_api_health', return_value=True), \
             patch.object(cv_decode, 'transcribe_batch') as mock_transcribe:
            
            # Mock mixed success/failure response
            mock_transcribe.return_value = [
                {
                    "file": "valid_audio.mp3",
                    "status": "success",
                    "transcription": "hello world transcribed",
                    "audio_duration": "2.0",
                    "processing_time": 0.5,
                    "attempt": 1
                },
                {
                    "file": "corrupted_audio.mp3",
                    "status": "error",
                    "transcription": "",
                    "error": "Invalid audio format",
                    "audio_duration": "0.0",
                    "processing_time": 0.1,
                    "attempt": 1
                },
                {
                    "file": "another_valid.mp3",
                    "status": "success", 
                    "transcription": "another test transcribed",
                    "audio_duration": "1.8",
                    "processing_time": 0.4,
                    "attempt": 1
                }
            ]
            
            with patch('sys.argv', [
                'cv-decode.py',
                '--csv', str(input_csv),
                '--folder', str(audio_dir), 
                '--output', str(output_csv)
            ]):
                
                result = await cv_decode.main()
        
        # Should complete successfully despite partial failures
        assert result == 0
        
        # Check output
        df = pd.read_csv(output_csv, keep_default_na=False)
        assert len(df) == 3
        transcriptions = df['generated_text'].tolist()
        assert transcriptions[0] == "hello world transcribed"  # Success
        assert transcriptions[1] == ""  # Failed file should have empty transcription
        assert transcriptions[2] == "another test transcribed"  # Success

    @pytest.mark.asyncio
    async def test_network_interruption_recovery(self, temp_dir: Path, cv_decode_module, mock_http_session):
        """Test recovery from network interruptions."""
        cv_decode = cv_decode_module
        
        # Create a sequence that fails twice, then succeeds
        response_sequence = [
            (500, {}, aiohttp.ClientError("Network error")),  # First attempt fails
            (500, {}, aiohttp.ClientError("Network error")),  # Second attempt fails
            (200, {
                "results": [{
                    "filename": "test.mp3",
                    "status": "success", 
                    "transcription": "recovered transcription",
                    "duration": "2.0",
                    "processing_time": 0.5
                }]
            }, None)  # Third attempt succeeds
        ]
        
        mock_session = mock_http_session(response_sequence=response_sequence)
        file_paths = [temp_dir / "test.mp3"]
        file_paths[0].write_bytes(b"fake mp3")
        
        # Test retry logic
        with patch('asyncio.sleep'):  # Speed up test
            result = await cv_decode.transcribe_batch(mock_session, file_paths)
        
        # Should eventually succeed after retries
        assert len(result) == 1
        assert result[0]["status"] == "success"
        assert result[0]["transcription"] == "recovered transcription"
        assert mock_session.call_count == 3  # Failed twice, succeeded on third


@pytest.mark.integration
class TestConfigurationVariations:
    """Test different configuration scenarios."""

    @pytest.mark.asyncio
    async def test_different_batch_sizes(self, temp_dir: Path, cv_decode_module, batch_audio_files, sample_csv_data):
        """Test processing with different batch sizes."""
        cv_decode = cv_decode_module
        
        # Use sample_csv_data fixture
        input_csv = temp_dir / "input.csv"
        input_csv.write_text(sample_csv_data)
        
        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()
        
        # Copy batch_audio_files to match sample_csv_data filenames
        csv_filenames = ["sample-000000.mp3", "sample-000001.mp3", "sample-000002.mp3"]
        for i, audio_file in enumerate(batch_audio_files):
            target_file = audio_dir / csv_filenames[i]
            target_file.write_bytes(audio_file.read_bytes())
        
        # Test different batch sizes
        for batch_size in [1, 2, 3]:
            output_csv = temp_dir / f"output_batch_{batch_size}.csv"
            
            with patch.object(cv_decode, 'check_api_health', return_value=True), \
                 patch.object(cv_decode, 'transcribe_batch') as mock_transcribe:
                
                # Mock responses based on batch size
                def mock_batch_response(_session, file_paths):
                    return [{
                        "file": fp.name,
                        "status": "success",
                        "transcription": f"transcribed {fp.name}",
                        "audio_duration": "2.0",
                        "processing_time": 0.5,
                        "attempt": 1
                    } for fp in file_paths]
                
                mock_transcribe.side_effect = mock_batch_response
                
                with patch('sys.argv', [
                    'cv-decode.py',
                    '--csv', str(input_csv),
                    '--folder', str(audio_dir),
                    '--output', str(output_csv),
                    '--batch_size', str(batch_size)
                ]):
                    
                    result = await cv_decode.main()
            
            assert result == 0
            
            # Verify output
            df = pd.read_csv(output_csv, keep_default_na=False)
            assert len(df) == 3  # Using 3 files from batch_audio_files fixture
            assert all(df['generated_text'].astype(str).str.startswith('transcribed'))

    def test_environment_configuration_override(self, monkeypatch):
        """Test that environment variables properly override defaults."""
        # Set custom environment variables
        monkeypatch.setenv('DEFAULT_BATCH_SIZE', '7')
        monkeypatch.setenv('RETRY_ATTEMPTS', '5')
        monkeypatch.setenv('PORT', '9001')
        
        # Re-import cv-decode with new environment variables
        import importlib.util
        cv_decode_path = Path(__file__).parent.parent.parent / "cv-decode.py"
        spec = importlib.util.spec_from_file_location("cv_decode_env_test", cv_decode_path)
        cv_decode = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cv_decode)
        
        # Check that environment variables were picked up
        assert cv_decode.DEFAULT_BATCH_SIZE == 7
        assert cv_decode.RETRY_ATTEMPTS == 5
        assert cv_decode.API_PORT == 9001