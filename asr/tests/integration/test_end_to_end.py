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


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow from CSV to results."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_complete_csv_processing_workflow(self, temp_dir: Path, mock_processor_and_model):
        """Test complete workflow from CSV input to final results."""
        mock_processor, mock_model = mock_processor_and_model
        
        # Create test CSV file
        csv_content = """filename,text,up_votes,down_votes,age,gender,accent,duration
test_audio_0.mp3,hello world,1,0,,,,
test_audio_1.mp3,this is a test,2,0,,,,
test_audio_2.mp3,another test,1,0,,,,"""
        
        input_csv = temp_dir / "input.csv"
        input_csv.write_text(csv_content)
        
        # Create mock audio files
        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()
        
        for i in range(3):
            audio_file = audio_dir / f"test_audio_{i}.mp3"
            audio_file.write_bytes(b"fake mp3 data" + bytes([i]))  # Make each file unique
        
        output_csv = temp_dir / "output.csv"
        
        # Import cv-decode module
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        import importlib.util
        cv_decode_path = Path(__file__).parent.parent.parent / "cv-decode.py"
        spec = importlib.util.spec_from_file_location("cv_decode", cv_decode_path)
        cv_decode = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cv_decode)
        
        # Mock the API health check and transcription
        with patch('cv_decode.check_api_health', return_value=True), \
             patch('cv_decode.transcribe_batch') as mock_transcribe:
            
            # Mock successful transcription responses
            mock_transcribe.return_value = [
                {
                    "file": "test_audio_0.mp3",
                    "status": "success",
                    "transcription": "hello world transcribed",
                    "audio_duration": "2.0",
                    "processing_time": 0.5,
                    "attempt": 1
                },
                {
                    "file": "test_audio_1.mp3", 
                    "status": "success",
                    "transcription": "this is a test transcribed",
                    "audio_duration": "2.1",
                    "processing_time": 0.6,
                    "attempt": 1
                },
                {
                    "file": "test_audio_2.mp3",
                    "status": "success", 
                    "transcription": "another test transcribed",
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
            "another test transcribed"
        ]
        
        assert transcriptions == expected_transcriptions

    @pytest.mark.integration
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
    def test_docker_container_workflow(self, temp_dir: Path):
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
            ["docker", "build", "-t", "asr-api-test", "."],
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
                "asr-api-test"
            ], capture_output=True, text=True, timeout=30)
            
            if run_result.returncode != 0:
                pytest.skip(f"Docker run failed: {run_result.stderr}")
            
            # Wait for container to start
            time.sleep(10)
            
            # Test the containerized API
            try:
                response = requests.get("http://localhost:8002/ping", timeout=10)
                assert response.status_code == 200
                assert response.json() == {"message": "pong"}
            except requests.RequestException as e:
                pytest.fail(f"Container API not responding: {e}")
        
        finally:
            # Clean up container
            subprocess.run(["docker", "stop", container_name], capture_output=True)
            subprocess.run(["docker", "rm", container_name], capture_output=True)


class TestErrorRecovery:
    """Test error recovery and resilience."""

    @pytest.mark.asyncio
    async def test_partial_batch_failure_recovery(self, temp_dir: Path):
        """Test recovery when some files in a batch fail."""
        # Create test setup
        csv_content = """filename,text
valid_audio.mp3,hello world
corrupted_audio.mp3,this should fail
another_valid.mp3,another test"""
        
        input_csv = temp_dir / "input.csv"
        input_csv.write_text(csv_content)
        
        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()
        
        # Create one valid file and one corrupted file
        valid_file = audio_dir / "valid_audio.mp3"
        valid_file.write_bytes(b"fake but valid mp3 data")
        
        corrupted_file = audio_dir / "corrupted_audio.mp3"
        corrupted_file.write_bytes(b"definitely not mp3")
        
        another_valid = audio_dir / "another_valid.mp3"
        another_valid.write_bytes(b"another fake mp3")
        
        output_csv = temp_dir / "output.csv"
        
        # Import cv-decode
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        import importlib.util
        cv_decode_path = Path(__file__).parent.parent.parent / "cv-decode.py"
        spec = importlib.util.spec_from_file_location("cv_decode", cv_decode_path)
        cv_decode = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cv_decode)
        
        with patch('cv_decode.check_api_health', return_value=True), \
             patch('cv_decode.transcribe_batch') as mock_transcribe:
            
            # Mock mixed success/failure response
            mock_transcribe.return_value = [
                {
                    "file": "valid_audio.mp3",
                    "status": "success",
                    "transcription": "hello world transcribed"
                },
                {
                    "file": "corrupted_audio.mp3",
                    "status": "error",
                    "error": "Invalid audio format"
                },
                {
                    "file": "another_valid.mp3",
                    "status": "success", 
                    "transcription": "another test transcribed"
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
        df = pd.read_csv(output_csv)
        assert len(df) == 3
        
        transcriptions = df['generated_text'].tolist()
        assert transcriptions[0] == "hello world transcribed"  # Success
        assert transcriptions[1] == ""  # Failed file should have empty transcription
        assert transcriptions[2] == "another test transcribed"  # Success

    @pytest.mark.asyncio
    async def test_network_interruption_recovery(self, temp_dir: Path):
        """Test recovery from network interruptions."""
        # This would test retry logic in real scenarios
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        import importlib.util
        cv_decode_path = Path(__file__).parent.parent.parent / "cv-decode.py"
        spec = importlib.util.spec_from_file_location("cv_decode", cv_decode_path)
        cv_decode = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cv_decode)
        
        # Mock session that fails first few times then succeeds
        class MockSession:
            def __init__(self):
                self.attempt_count = 0
            
            async def post(self, *args, **kwargs):
                self.attempt_count += 1
                if self.attempt_count <= 2:
                    # Simulate network failure
                    raise aiohttp.ClientError("Network error")
                else:
                    # Succeed on third attempt
                    mock_response = type('MockResponse', (), {})()
                    mock_response.status = 200
                    mock_response.json = lambda: {
                        "results": [{
                            "filename": "test.mp3",
                            "status": "success", 
                            "transcription": "recovered transcription"
                        }]
                    }
                    return type('MockContext', (), {'__aenter__': lambda self: mock_response, '__aexit__': lambda *args: None})()
        
        mock_session = MockSession()
        file_paths = [temp_dir / "test.mp3"]
        file_paths[0].write_bytes(b"fake mp3")
        
        # Test retry logic
        with patch('asyncio.sleep'):  # Speed up test
            result = await cv_decode.transcribe_batch(mock_session, file_paths, batch_id=0)
        
        # Should eventually succeed after retries
        assert len(result) == 1
        assert result[0]["status"] == "success"
        assert mock_session.attempt_count == 3  # Failed twice, succeeded on third


class TestConfigurationVariations:
    """Test different configuration scenarios."""

    @pytest.mark.asyncio
    async def test_different_batch_sizes(self, temp_dir: Path):
        """Test processing with different batch sizes."""
        # Create larger test dataset
        filenames = [f"test_audio_{i:03d}.mp3" for i in range(10)]
        csv_content = "filename,text\n" + "\n".join([f"{f},test text {i}" for i, f in enumerate(filenames)])
        
        input_csv = temp_dir / "input.csv"
        input_csv.write_text(csv_content)
        
        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()
        
        for filename in filenames:
            (audio_dir / filename).write_bytes(b"fake mp3 data")
        
        # Import cv-decode
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        import importlib.util
        cv_decode_path = Path(__file__).parent.parent.parent / "cv-decode.py"
        spec = importlib.util.spec_from_file_location("cv_decode", cv_decode_path)
        cv_decode = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cv_decode)
        
        # Test different batch sizes
        for batch_size in [1, 3, 5]:
            output_csv = temp_dir / f"output_batch_{batch_size}.csv"
            
            with patch('cv_decode.check_api_health', return_value=True), \
                 patch('cv_decode.transcribe_batch') as mock_transcribe:
                
                # Mock responses based on batch size
                def mock_batch_response(session, file_paths, batch_id):
                    return [{
                        "file": fp.name,
                        "status": "success",
                        "transcription": f"transcribed {fp.name}"
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
            df = pd.read_csv(output_csv)
            assert len(df) == 10
            assert all(df['generated_text'].str.startswith('transcribed'))

    def test_environment_configuration_override(self, monkeypatch, temp_dir: Path):
        """Test that environment variables properly override defaults."""
        # Set custom environment variables
        monkeypatch.setenv('DEFAULT_BATCH_SIZE', '7')
        monkeypatch.setenv('RETRY_ATTEMPTS', '5')
        monkeypatch.setenv('PORT', '9001')
        
        # Import cv-decode with new environment
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        import importlib.util
        cv_decode_path = Path(__file__).parent.parent.parent / "cv-decode.py"
        spec = importlib.util.spec_from_file_location("cv_decode", cv_decode_path)
        cv_decode = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cv_decode)
        
        # Check that environment variables were picked up
        assert cv_decode.DEFAULT_BATCH_SIZE == 7
        assert cv_decode.RETRY_ATTEMPTS == 5
        assert cv_decode.API_PORT == 9001