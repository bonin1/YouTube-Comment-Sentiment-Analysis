import functools
import time
import logging
from typing import Callable, Any, Optional

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Retry decorator with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier for delay
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        logging.error(f"Function {func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    
                    logging.warning(f"Attempt {attempts} failed for {func.__name__}: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator

def timing(func: Callable) -> Callable:
    """
    Decorator to measure function execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logging.debug(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    return wrapper

def cache_result(ttl: Optional[float] = None):
    """
    Simple caching decorator with optional TTL
    
    Args:
        ttl: Time to live in seconds (None for no expiration)
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = str(args) + str(sorted(kwargs.items()))
            
            # Check if result is cached and not expired
            if key in cache:
                result, timestamp = cache[key]
                if ttl is None or (time.time() - timestamp) < ttl:
                    return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            
            return result
        
        return wrapper
    return decorator
