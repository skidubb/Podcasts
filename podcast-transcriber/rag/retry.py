"""Shared retry helper with exponential backoff for API calls."""

import time


def retry_with_backoff(
    fn,
    *args,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    retryable: tuple = (Exception,),
    should_retry=None,
    label: str = "operation",
    **kwargs,
):
    """Call fn(*args, **kwargs), retrying on retryable exceptions.

    Delay doubles each attempt (2s, 4s, 8s, ...) capped at max_delay.
    should_retry(exc) can veto a retry for exceptions that match
    retryable but are known-permanent (e.g. HTTP 4xx other than 429).
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except retryable as e:
            last_error = e
            if should_retry is not None and not should_retry(e):
                raise
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                print(f"  {label} failed ({e}); retrying in {delay:.0f}s "
                      f"(attempt {attempt + 2}/{max_retries})...")
                time.sleep(delay)
    raise last_error


def is_retryable_status(exc) -> bool:
    """True if an API error carries a retryable HTTP status (429/5xx)."""
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status is None:
        return True  # connection errors etc. — worth retrying
    return status == 429 or status >= 500
