"""
Performance and load tests for YouTube Comment Sentiment Analysis system.

This module contains performance tests, load tests, and benchmarks
to ensure the system can handle large datasets efficiently.

Author: GitHub Copilot
Date: 2025
"""

import pytest
import time
import asyncio
import tempfile
import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import psutil
import memory_profiler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.scrapers.youtube_scraper import CommentScraper
from src.processors.text_processor import TextPreprocessor
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.utils.data_manager import DataManager


class TestPerformance:
    """Performance tests for system components."""
    def setup_method(self):
        """Set up performance test fixtures."""
        self.processor = TextPreprocessor()
        self.analyzer = SentimentAnalyzer()
        
        # Create temp database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.data_manager = DataManager(db_path=self.temp_db.name)
    
    def teardown_method(self):
        """Clean up performance test fixtures."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_text_processing_performance(self):
        """Test text processing performance with large dataset."""
        # Generate test data
        test_texts = [
            f"This is test comment number {i} with some emoji 😊 and URLs https://example.com"
            for i in range(1000)
        ]
        
        # Measure processing time
        start_time = time.time()
        
        for text in test_texts:
            result = self.processor.process_text(text)
            assert 'cleaned_text' in result
            assert 'features' in result
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 1000 texts in reasonable time (< 30 seconds)
        assert processing_time < 30.0, f"Processing took too long: {processing_time:.2f}s"
        
        # Calculate average processing time per text
        avg_time = processing_time / len(test_texts)
        print(f"Average processing time per text: {avg_time*1000:.2f}ms")
        
        assert avg_time < 0.05, f"Average processing time too high: {avg_time:.3f}s"
    
    def test_sentiment_analysis_performance(self):
        """Test sentiment analysis performance with different models."""
        test_texts = [
            "This is an amazing video!",
            "Not impressed with this content.",
            "This is okay, nothing special.",
            "Absolutely terrible and disappointing.",
            "Great work, keep it up!"
        ] * 100  # 500 texts total
        
        models = ["vader", "textblob"]
        
        for model in models:
            start_time = time.time()
            
            for text in test_texts:
                result = self.analyzer.analyze_sentiment(text, model=model)
                assert 'label' in result
                assert 'confidence' in result
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"{model} model: {processing_time:.2f}s for {len(test_texts)} texts")
            
            # Should be reasonably fast
            assert processing_time < 60.0, f"{model} took too long: {processing_time:.2f}s"
    
    def test_batch_analysis_performance(self):
        """Test batch analysis performance."""
        test_texts = [
            f"Test comment {i} with various sentiments and content."
            for i in range(1000)
        ]
        
        # Test batch processing
        start_time = time.time()
        results = self.analyzer.analyze_batch(test_texts, model="vader")
        end_time = time.time()
        
        batch_time = end_time - start_time
        
        # Test individual processing
        start_time = time.time()
        individual_results = []
        for text in test_texts:
            result = self.analyzer.analyze_sentiment(text, model="vader")
            individual_results.append(result)
        end_time = time.time()
        
        individual_time = end_time - start_time
        
        print(f"Batch processing: {batch_time:.2f}s")
        print(f"Individual processing: {individual_time:.2f}s")
        print(f"Speedup: {individual_time/batch_time:.2f}x")
        
        # Batch should be faster or at least not much slower
        assert batch_time <= individual_time * 1.2, "Batch processing should be efficient"
        
        # Results should be consistent
        assert len(results) == len(individual_results)
    
    def test_database_performance(self):
        """Test database operations performance."""
        video_id = "performance_test"
        
        # Generate large dataset
        comments = []
        for i in range(5000):
            comments.append({
                'text': f'Performance test comment {i} with various content and length.',
                'author': f'TestUser{i}',
                'timestamp': f'2024-01-01T{i%24:02d}:00:00',
                'likes': i % 100,
                'cid': f'comment_{i}'
            })
        
        # Test bulk insert performance
        start_time = time.time()
        saved_count = self.data_manager.save_comments(video_id, comments)
        insert_time = time.time() - start_time
        
        assert saved_count == len(comments)
        print(f"Inserted {saved_count} comments in {insert_time:.2f}s")
        print(f"Insert rate: {saved_count/insert_time:.0f} comments/sec")
        
        # Should handle large inserts efficiently
        assert insert_time < 10.0, f"Insert took too long: {insert_time:.2f}s"
        
        # Test bulk load performance
        start_time = time.time()
        loaded_comments = self.data_manager.load_comments(video_id)
        load_time = time.time() - start_time
        
        assert len(loaded_comments) == len(comments)
        print(f"Loaded {len(loaded_comments)} comments in {load_time:.2f}s")
        print(f"Load rate: {len(loaded_comments)/load_time:.0f} comments/sec")
        
        # Should load efficiently
        assert load_time < 5.0, f"Load took too long: {load_time:.2f}s"
    
    @pytest.mark.skipif(not hasattr(memory_profiler, 'profile'), 
                       reason="memory_profiler not available")
    def test_memory_usage(self):
        """Test memory usage during processing."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large dataset
        large_texts = [
            f"This is a very long comment with lots of text and content. "
            f"Comment number {i} contains various information and details. " * 10
            for i in range(1000)
        ]
        
        # Process all texts
        results = []
        for text in large_texts:
            processed = self.processor.process_text(text)
            sentiment = self.analyzer.analyze_sentiment(processed['cleaned_text'], model="vader")
            results.append({'processed': processed, 'sentiment': sentiment})
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Peak memory: {peak_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        
        # Clean up
        del results
        del large_texts
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_released = peak_memory - final_memory
        
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Memory released: {memory_released:.1f} MB")
        
        # Memory usage should be reasonable
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f} MB"
    
    def test_concurrent_processing(self):
        """Test concurrent processing performance."""
        test_texts = [
            f"Concurrent test comment {i} with various content."
            for i in range(500)
        ]
        
        # Sequential processing
        start_time = time.time()
        sequential_results = []
        for text in test_texts:
            result = self.analyzer.analyze_sentiment(text, model="vader")
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Concurrent processing
        start_time = time.time()
        
        def process_text(text):
            return self.analyzer.analyze_sentiment(text, model="vader")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            concurrent_results = list(executor.map(process_text, test_texts))
        
        concurrent_time = time.time() - start_time
        
        print(f"Sequential processing: {sequential_time:.2f}s")
        print(f"Concurrent processing: {concurrent_time:.2f}s")
        print(f"Speedup: {sequential_time/concurrent_time:.2f}x")
        
        # Results should be consistent
        assert len(concurrent_results) == len(sequential_results)
        
        # Concurrent should be faster (or at least not much slower due to overhead)
        assert concurrent_time <= sequential_time * 1.1, "Concurrent processing should be efficient"


class TestScalability:
    """Scalability tests for different data sizes."""
    def setup_method(self):
        """Set up scalability test fixtures."""
        self.processor = TextPreprocessor()
        self.analyzer = SentimentAnalyzer()
    
    @pytest.mark.parametrize("data_size", [100, 500, 1000, 2000])
    def test_processing_scalability(self, data_size):
        """Test processing performance with different data sizes."""
        test_texts = [
            f"Scalability test comment {i} with content and details."
            for i in range(data_size)
        ]
        
        # Measure processing time
        start_time = time.time()
        
        for text in test_texts:
            processed = self.processor.process_text(text)
            sentiment = self.analyzer.analyze_sentiment(processed['cleaned_text'], model="vader")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        rate = data_size / processing_time
        
        print(f"Data size: {data_size}, Time: {processing_time:.2f}s, Rate: {rate:.1f} items/sec")
        
        # Performance should scale reasonably
        expected_max_time = data_size * 0.01  # 10ms per item max
        assert processing_time < expected_max_time, f"Processing too slow for {data_size} items"
    
    def test_memory_scalability(self):
        """Test memory usage with increasing data sizes."""
        import psutil
        
        process = psutil.Process()
        memory_usage = []
        
        for size in [100, 500, 1000, 2000]:
            # Generate data
            texts = [f"Memory test {i}" for i in range(size)]
            
            # Measure memory before processing
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # Process data
            results = []
            for text in texts:
                processed = self.processor.process_text(text)
                results.append(processed)
            
            # Measure memory after processing
            peak_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = peak_memory - initial_memory
            
            memory_usage.append((size, memory_increase))
            
            # Clean up
            del results
            del texts
        
        # Check memory scaling
        for i, (size, memory) in enumerate(memory_usage):
            print(f"Size: {size}, Memory increase: {memory:.1f} MB")
            
            if i > 0:
                prev_size, prev_memory = memory_usage[i-1]
                size_ratio = size / prev_size
                memory_ratio = memory / prev_memory if prev_memory > 0 else 1
                
                # Memory should scale roughly linearly (not exponentially)
                assert memory_ratio < size_ratio * 2, f"Memory scaling too aggressive: {memory_ratio:.2f}x"


class TestStressTest:
    """Stress tests for system limits."""
    
    def setup_method(self):
        """Set up stress test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.data_manager = DataManager(db_path=self.temp_db.name)
        self.analyzer = SentimentAnalyzer()
    
    def teardown_method(self):
        """Clean up stress test fixtures."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_large_text_processing(self):
        """Test processing of very large text inputs."""
        # Create very long text
        large_text = "This is a very long comment. " * 1000  # ~28KB text
        
        start_time = time.time()
        result = self.analyzer.analyze_sentiment(large_text, model="vader")
        processing_time = time.time() - start_time
        
        assert 'label' in result
        assert 'confidence' in result
        print(f"Large text ({len(large_text)} chars) processed in {processing_time:.3f}s")
        
        # Should handle large texts reasonably
        assert processing_time < 5.0, f"Large text processing too slow: {processing_time:.3f}s"
    
    def test_database_stress(self):
        """Test database under stress conditions."""
        video_id = "stress_test"
        
        # Insert data in multiple batches
        total_inserted = 0
        batch_size = 1000
        num_batches = 10
        
        for batch in range(num_batches):
            comments = []
            for i in range(batch_size):
                comment_id = batch * batch_size + i
                comments.append({
                    'text': f'Stress test comment {comment_id} with content.',
                    'author': f'StressUser{comment_id}',
                    'timestamp': f'2024-01-01T{comment_id%24:02d}:00:00',
                    'likes': comment_id % 1000,
                    'cid': f'stress_comment_{comment_id}'
                })
            
            start_time = time.time()
            saved = self.data_manager.save_comments(f"{video_id}_{batch}", comments)
            batch_time = time.time() - start_time
            
            total_inserted += saved
            print(f"Batch {batch+1}: {saved} comments in {batch_time:.2f}s")
            
            # Each batch should complete in reasonable time
            assert batch_time < 5.0, f"Batch {batch} took too long: {batch_time:.2f}s"
        
        print(f"Total inserted: {total_inserted} comments")
        assert total_inserted == batch_size * num_batches
    
    def test_error_recovery(self):
        """Test system behavior under error conditions."""
        # Test with invalid/malformed data
        invalid_texts = [
            "",  # Empty text
            None,  # None value
            "a" * 100000,  # Extremely long text
            "🎉" * 1000,  # Lots of emojis
            "\n\n\n\t\t\t   ",  # Only whitespace
            "http://example.com " * 500,  # Lots of URLs
        ]
        
        for i, text in enumerate(invalid_texts):
            try:
                if text is not None:
                    result = self.analyzer.analyze_sentiment(str(text), model="vader")
                    assert 'label' in result, f"Failed to get label for text {i}"
                    print(f"Text {i}: {result['label']} (confidence: {result['confidence']:.3f})")
                else:
                    # Should handle None gracefully
                    result = self.analyzer.analyze_sentiment("", model="vader")
                    assert result['label'] == 'neutral'
            except Exception as e:
                pytest.fail(f"System failed on invalid text {i}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
