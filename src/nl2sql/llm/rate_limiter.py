"""Token bucket rate limiter for LLM API calls."""

from __future__ import annotations

import logging
import threading
import time
from functools import wraps
from typing import Callable, TypeVar

from .config import RateLimitConfig

logger = logging.getLogger(__name__)
T = TypeVar("T")


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter with configurable parameters.

    Uses a token bucket algorithm where:
    - Tokens are added at a steady rate (calls_per_minute / 60 per second)
    - Burst limit sets the maximum tokens (bucket size)
    - Each API call consumes one token
    - If no tokens available, waits until one is available
    """

    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter.

        Parameters
        ----------
        config : RateLimitConfig
            Rate limiting configuration
        """
        self.config = config
        self.tokens = float(config.burst_limit)
        self.max_tokens = float(config.burst_limit)
        self.refill_rate = config.calls_per_minute / 60.0  # tokens per second
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + new_tokens)
        self.last_refill = now

    def acquire(self, blocking: bool = True) -> bool:
        """
        Acquire a token for an API call.

        Parameters
        ----------
        blocking : bool
            If True, wait until a token is available.
            If False, return immediately.

        Returns
        -------
        bool
            True if token acquired, False if not (only when blocking=False)
        """
        max_attempts = 10

        for attempt in range(max_attempts):
            with self._lock:
                self._refill()

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True

                if not blocking:
                    return False

                # Calculate wait time for next token
                tokens_needed = 1.0 - self.tokens
                wait_time = tokens_needed / self.refill_rate

            # Wait outside the lock
            logger.debug(
                f"Rate limited, waiting {wait_time:.2f}s for token (attempt {attempt + 1}/{max_attempts})"
            )
            time.sleep(wait_time)

        # Failed to acquire after max attempts
        logger.error(f"Failed to acquire token after {max_attempts} attempts")
        return False

    def get_wait_time(self) -> float:
        """
        Get estimated wait time until a token is available.

        Returns
        -------
        float
            Seconds until next token available, 0 if available now
        """
        with self._lock:
            self._refill()
            if self.tokens >= 1.0:
                return 0.0
            tokens_needed = 1.0 - self.tokens
            return tokens_needed / self.refill_rate


def rate_limited(limiter: TokenBucketRateLimiter) -> Callable:
    """
    Decorator to apply rate limiting to a function.

    Parameters
    ----------
    limiter : TokenBucketRateLimiter
        Rate limiter instance to use

    Returns
    -------
    Callable
        Decorated function with rate limiting
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            limiter.acquire()
            return func(*args, **kwargs)

        return wrapper

    return decorator


class RateLimiterRegistry:
    """
    Registry to manage rate limiters per provider.

    Ensures each provider has a single shared rate limiter instance.
    """

    _limiters: dict[str, TokenBucketRateLimiter] = {}
    _lock = threading.Lock()

    @classmethod
    def get_or_create(
        cls, provider_name: str, config: RateLimitConfig
    ) -> TokenBucketRateLimiter:
        """
        Get or create a rate limiter for a provider.

        Parameters
        ----------
        provider_name : str
            Name of the provider
        config : RateLimitConfig
            Rate limit configuration

        Returns
        -------
        TokenBucketRateLimiter
            Rate limiter for the provider
        """
        with cls._lock:
            if provider_name not in cls._limiters:
                cls._limiters[provider_name] = TokenBucketRateLimiter(config)
            return cls._limiters[provider_name]

    @classmethod
    def clear(cls) -> None:
        """Clear all rate limiters. Useful for testing."""
        with cls._lock:
            cls._limiters.clear()
