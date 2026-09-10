"""
src/llm_client.py
-----------------
Purpose:
Centralized LLM interface for the AI Support Agent pipeline.
This wrapper sends requests to Hugging Face's Inference Provider router API
using Qwen/Qwen2.5-7B-Instruct (via an OpenAI-compatible interface).
It includes retry-with-backoff for API stability and logs all calls to a local file.

Why this file exists:
Isolating LLM call logic here allows us to enforce consistent model parameters,
handle API errors gracefully (e.g. 503 model warm-up errors), log prompt history,
and swap endpoints (e.g. to Groq) with a single configuration change.
"""

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI, OpenAIError

# Configure standard logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LLMClient")

# Log file location for auditing raw calls
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
LOG_FILE = os.path.join(LOG_DIR, "llm_calls.jsonl")


class LLMClient:
    """
    Wrapper around OpenAI-compatible APIs (Hugging Face Inference Providers / Groq).
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the LLM client using environment variables or explicit parameters.
        Defaults to Hugging Face Inference Provider router with Qwen2.5-7B-Instruct.
        """
        # Fetch token from environment or fallback to empty string if not supplied
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY") or "dummy_key_for_offline"
        
        # Determine endpoint URL (Default: Hugging Face Router)
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1")
        
        # Default model name
        self.model = model or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        
        # Initialize OpenAI client pointed at router
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

        logger.info(f"Initialized LLMClient endpoint={self.base_url}, model={self.model}")

    def call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> str:
        """
        Calls the LLM chat completion API with retry-with-backoff resilience.

        Args:
            messages: List of message dicts with 'role' ('system'/'user') and 'content'.
            temperature: Sampling temperature (0.0 for deterministic classification, ~0.3-0.5 for replies).
            max_tokens: Maximum response tokens to generate.
            max_retries: Number of retry attempts on API errors.
            retry_delay: Initial delay in seconds before retrying (exponential backoff).

        Returns:
            The raw text content returned by the LLM.
        """
        start_time = time.time()
        last_exception = None

        for attempt in range(1, max_retries + 1):
            try:
                # If using dummy key, skip API call immediately to fast local fallback
                if self.api_key == "dummy_key_for_offline":
                    return self._get_offline_fallback(messages)

                # Call OpenAI-compatible chat completion endpoint
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                content = response.choices[0].message.content.strip()
                latency = round(time.time() - start_time, 3)

                # Log call to JSONL file for audit and cost tracking
                self._log_call(messages, content, temperature, latency, status="success")
                return content

            except Exception as e:
                last_exception = e
                err_msg = str(e)
                logger.warning(f"LLM call attempt {attempt}/{max_retries} failed: {err_msg}")

                # If 401 Unauthorized (missing/invalid token), break immediately to fallback
                if "401" in err_msg or "Invalid username or password" in err_msg or "Unauthorized" in err_msg:
                    logger.info("401 Unauthorized detected. Instant fallback to deterministic rules.")
                    break

                if attempt < max_retries:
                    # Exponential backoff delay
                    time.sleep(retry_delay * (2 ** (attempt - 1)))

        # If all retries fail, log error and return fallback error message
        logger.error(f"All {max_retries} LLM call attempts failed. Error: {str(last_exception)}")
        self._log_call(messages, str(last_exception), temperature, round(time.time() - start_time, 3), status="failed")
        
        # Return fallback json/string if API is unreachable
        return self._get_offline_fallback(messages)

    def _log_call(self, messages: List[Dict[str, str]], response: str, temperature: float, latency: float, status: str):
        """
        Saves the raw prompt, LLM output, metadata, and execution latency to a local JSONL log file.
        """
        os.makedirs(LOG_DIR, exist_ok=True)
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model,
            "endpoint": self.base_url,
            "temperature": temperature,
            "latency_sec": latency,
            "status": status,
            "messages": messages,
            "response": response
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _get_offline_fallback(self, messages: List[Dict[str, str]]) -> str:
        """
        Generates a basic deterministic offline response if the remote API is completely unavailable or quota is exceeded.
        Ensures system pipeline degrades gracefully rather than crashing.
        """
        user_msg = ""
        for m in messages:
            if m["role"] == "user":
                user_msg = m["content"].lower()

        # If asking for intent classification (JSON format expected)
        if "intent" in messages[0]["content"].lower():
            if "track" in user_msg or "deliver" in user_msg or "where" in user_msg:
                intent = "shipping_delivery_status"
            elif "return" in user_msg or "refund" in user_msg:
                intent = "refund_return_request"
            elif "cancel" in user_msg:
                intent = "cancellation_order_change"
            elif "prime" in user_msg or "video" in user_msg:
                intent = "prime_digital_services"
            else:
                intent = "other_unclear"
            return json.dumps({
                "intent": intent,
                "confidence": 0.85,
                "reasoning": "Offline fallback classification based on rule heuristics."
            })
        
        # Default generic reply fallback
        return "We apologize for the inconvenience. Please send us your order ID via Direct Message so our support team can assist you directly."


# Global singleton helper instance for easy importing across modules
default_client = None

def get_llm_client() -> LLMClient:
    """Convenience getter for the global LLMClient instance."""
    global default_client
    if default_client is None:
        default_client = LLMClient()
    return default_client
