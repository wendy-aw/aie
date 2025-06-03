"""
Performance and load tests for ASR API.
"""
import pytest
import asyncio
import time
import torch
from httpx import AsyncClient
from pathlib import Path
from unittest.mock import patch


class TestPerformance:
    """Test performance characteristics of the ASR API."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_single_file_response_time(self, async_client: AsyncClient, sample_audio_file: Path):
        """Test single file transcription response time."""
        
        with open(sample_audio_file, 'rb') as f:
            files = {'file': ('test.mp3', f, 'audio/mpeg')}
            
            with patch('asr_api.torchaudio.load') as mock_load:
                mock_load.return_value = (torch.randn(1, 32000), 16000)
                
                start_time = time.time()
                response = await async_client.post("/asr", files=files)
                end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        
        # Should respond within 5 seconds (generous for CI/testing)
        assert response_time < 5.0
        print(f"Single file response time: {response_time:.2f}s")

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self, async_client: AsyncClient, large_batch_audio_files: list[Path]):
        """Test that batch processing is more efficient than individual requests."""
        
        # Time individual requests
        individual_start = time.time()
        individual_responses = []
        
        for audio_file in large_batch_audio_files:
            with open(audio_file, 'rb') as f:
                files = {'file': (audio_file.name, f, 'audio/mpeg')}
                
                with patch('asr_api.torchaudio.load') as mock_load:
                    mock_load.return_value = (torch.randn(1, 16000), 16000)
                    
                    response = await async_client.post("/asr", files=files)
                    individual_responses.append(response)
        
        individual_end = time.time()
        individual_time = individual_end - individual_start
        
        # Time batch request
        batch_start = time.time()
        
        files = []
        for audio_file in large_batch_audio_files:
            with open(audio_file, 'rb') as f:
                files.append(('files', (audio_file.name, f.read(), 'audio/mpeg')))
        
        with patch('asr_api.torchaudio.load') as mock_load:
            mock_load.return_value = (torch.randn(1, 16000), 16000)
            
            batch_response = await async_client.post("/asr", files=files)
        
        batch_end = time.time()
        batch_time = batch_end - batch_start
        
        # Verify all requests succeeded
        assert all(r.status_code == 200 for r in individual_responses)
        assert batch_response.status_code == 200
        
        print(f"Individual requests time: {individual_time:.2f}s")
        print(f"Batch request time: {batch_time:.2f}s")
        print(f"Efficiency gain: {individual_time / batch_time:.2f}x")
        
        # Batch should be faster (or at least not significantly slower)
        assert batch_time <= individual_time * 1.2  # Allow 20% margin

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, async_client: AsyncClient, sample_audio_file: Path):
        """Test handling of multiple concurrent requests."""
        
        async def make_request(request_id: int):
            with open(sample_audio_file, 'rb') as f:
                files = {'file': (f'test_{request_id}.mp3', f, 'audio/mpeg')}
                
                with patch('asr_api.torchaudio.load') as mock_load:
                    mock_load.return_value = (torch.randn(1, 32000), 16000)
                    
                    start_time = time.time()
                    response = await async_client.post("/asr", files=files)
                    end_time = time.time()
                    
                    return response, end_time - start_time
        
        # Test with 5 concurrent requests
        num_requests = 5
        start_time = time.time()
        
        tasks = [make_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Verify all requests succeeded
        for response, request_time in results:
            assert response.status_code == 200
        
        response_times = [request_time for _, request_time in results]
        avg_response_time = sum(response_times) / len(response_times)
        
        print(f"Concurrent requests: {num_requests}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average response time: {avg_response_time:.2f}s")
        print(f"Requests per second: {num_requests / total_time:.2f}")
        
        # Should handle concurrent requests reasonably well
        assert total_time < num_requests * 3  # Not more than 3s per request on average

    @pytest.mark.slow
    def test_memory_usage_stability(self, client, sample_audio_file: Path):
        """Test that memory usage remains stable over multiple requests."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make 10 requests and monitor memory
        memory_samples = [initial_memory]
        
        for i in range(10):
            with open(sample_audio_file, 'rb') as f:
                files = {'file': (f'test_{i}.mp3', f, 'audio/mpeg')}
                
                with patch('asr_api.torchaudio.load') as mock_load:
                    mock_load.return_value = (torch.randn(1, 32000), 16000)
                    
                    response = client.post("/asr", files=files)
                    assert response.status_code == 200
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(current_memory)
        
        final_memory = memory_samples[-1]
        max_memory = max(memory_samples)
        
        print(f"Initial memory: {initial_memory:.1f}MB")
        print(f"Final memory: {final_memory:.1f}MB")
        print(f"Max memory: {max_memory:.1f}MB")
        print(f"Memory increase: {final_memory - initial_memory:.1f}MB")
        
        # Memory shouldn't grow excessively (allow for some fluctuation)
        memory_growth = final_memory - initial_memory
        assert memory_growth < 100, f"Memory grew by {memory_growth:.1f}MB, which is too much"

    @pytest.mark.asyncio
    async def test_ping_endpoint_performance(self, async_client: AsyncClient):
        """Test ping endpoint performance under load."""
        async def ping_request():
            start_time = time.time()
            response = await async_client.get("/ping")
            end_time = time.time()
            return response.status_code == 200, end_time - start_time
        
        # Make 50 concurrent ping requests
        num_requests = 50
        start_time = time.time()
        
        tasks = [ping_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        success_count = sum(1 for success, _ in results if success)
        response_times = [rt for _, rt in results]
        avg_response_time = sum(response_times) / len(response_times)
        
        print(f"Ping requests: {num_requests}")
        print(f"Success rate: {success_count}/{num_requests} ({success_count/num_requests*100:.1f}%)")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average response time: {avg_response_time*1000:.1f}ms")
        print(f"Requests per second: {num_requests / total_time:.1f}")
        
        # All ping requests should succeed
        assert success_count == num_requests
        
        # Ping should be very fast
        assert avg_response_time < 0.1  # Less than 100ms average
        
        # Should handle high throughput
        assert num_requests / total_time > 100  # More than 100 RPS


class TestScalability:
    """Test scalability characteristics."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_batch_size_scaling(self, async_client: AsyncClient, large_batch_audio_files: list[Path]):
        """Test how performance scales with batch size."""
        
        batch_sizes = [1, 10, 20]
        results = {}
        
        for batch_size in batch_sizes:
            files = []
            for i in range(batch_size):
                audio_file = large_batch_audio_files[i % len(large_batch_audio_files)]
                with open(audio_file, 'rb') as f:
                    files.append(('files', (f'test_{i}.mp3', f.read(), 'audio/mpeg')))
            
            with patch('asr_api.torchaudio.load') as mock_load:
                mock_load.return_value = (torch.randn(1, 16000), 16000)
                
                start_time = time.time()
                response = await async_client.post("/asr", files=files)
                end_time = time.time()
            
            assert response.status_code == 200
            
            processing_time = end_time - start_time
            time_per_file = processing_time / batch_size
            
            results[batch_size] = {
                'total_time': processing_time,
                'time_per_file': time_per_file
            }
            
            print(f"Batch size {batch_size}: {processing_time:.2f}s total, {time_per_file:.2f}s per file")
        
        # Larger batches should be more efficient per file
        # Compare the largest batch (20) to the smallest (1)
        assert results[20]['time_per_file'] <= results[1]['time_per_file'] * 1.5
        
        # Also verify that medium batches show some efficiency gain
        assert results[10]['time_per_file'] <= results[1]['time_per_file'] * 1.2


class TestStressTest:
    """Stress tests for the API."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_rapid_sequential_requests(self, async_client: AsyncClient, sample_audio_file: Path):
        """Test handling rapid sequential requests."""
        
        num_requests = 20
        success_count = 0
        start_time = time.time()
        
        for i in range(num_requests):
            with open(sample_audio_file, 'rb') as f:
                files = {'file': (f'test_{i}.mp3', f, 'audio/mpeg')}
                
                with patch('asr_api.torchaudio.load') as mock_load:
                    mock_load.return_value = (torch.randn(1, 32000), 16000)
                    
                    try:
                        response = await async_client.post("/asr", files=files)
                        if response.status_code == 200:
                            success_count += 1
                    except Exception as e:
                        print(f"Request {i} failed: {e}")
        
        total_time = time.time() - start_time
        
        print(f"Sequential requests: {num_requests}")
        print(f"Success rate: {success_count}/{num_requests} ({success_count/num_requests*100:.1f}%)")
        print(f"Total time: {total_time:.2f}s")
        print(f"Requests per second: {num_requests / total_time:.1f}")
        
        # Should handle rapid requests with high success rate
        assert success_count >= num_requests * 0.9  # At least 90% success rate