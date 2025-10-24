"""Ollama LLM API client for generating responses."""
import requests
import time
from typing import Optional, Dict, Any, List
from loguru import logger
from src.config import config

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self):
        """Initialize the Ollama client."""
        self.api_url = config.ollama.api_url
        self.model = config.ollama.model
        self.timeout = config.ollama.timeout
        self.max_tokens = config.ollama.max_tokens
        self.temperature = config.ollama.temperature
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        
        logger.info(f"Initialized Ollama client with API: {self.api_url}")
    
    def _make_request_with_retry(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make HTTP request with exponential backoff retry logic.
        
        Args:
            url: API endpoint URL
            payload: Request payload
            
        Returns:
            Response JSON
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                # Ensure proper UTF-8 encoding
                response.encoding = 'utf-8'
                return response.json()
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit error
                    if attempt < self.max_retries - 1:
                        # Exponential backoff: 2s, 4s, 8s
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limited (429). Retrying in {wait_time}s... (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Max retries reached for rate limiting")
                        raise
                else:
                    # Other HTTP errors - don't retry
                    raise
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Request timeout. Retrying in {wait_time}s... (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise
        
        raise Exception("Max retries exceeded")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Generate a response from the Ollama API with retry logic.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for instructions
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stream: Whether to stream the response
            
        Returns:
            Generated text response
        """
        try:
            # Prepare the request payload
            payload = {
                "model": "gemma3:4b",  # Use the actual model name from the API
                "prompt": prompt,
                "options": {
                    "temperature": temperature or self.temperature,
                    "num_predict": max_tokens or self.max_tokens,
                },
                "stream": stream
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            # Make API request with retry logic
            result = self._make_request_with_retry(
                f"{self.api_url}/api/generate",
                payload
            )
            
            # Parse response
            generated_text = result.get("response", "")
            
            logger.debug(f"Generated response: {generated_text[:100]}...")
            return generated_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return "I apologize, but I'm having trouble connecting to my language model. Please try again later."
        except Exception as e:
            logger.error(f"Unexpected error in generate: {e}")
            return "I apologize, but something went wrong. Please try again."
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response using chat format with retry logic.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Generated text response
        """
        try:
            payload = {
                "model": "gemma3:4b",  # Use the actual model name from the API
                "messages": messages,
                "options": {
                    "temperature": temperature or self.temperature,
                    "num_predict": max_tokens or self.max_tokens,
                },
                "stream": False
            }
            
            # Make API request with retry logic
            result = self._make_request_with_retry(
                f"{self.api_url}/api/chat",
                payload
            )
            
            generated_text = result.get("message", {}).get("content", "")
            
            logger.debug(f"Chat response: {generated_text[:100]}...")
            return generated_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama chat API: {e}")
            return "I apologize, but I'm having trouble connecting to my language model. Please try again later."
        except Exception as e:
            logger.error(f"Unexpected error in chat: {e}")
            return "I apologize, but something went wrong. Please try again."
    
    def health_check(self) -> bool:
        """
        Check if the Ollama API is accessible.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.api_url}/", timeout=10)
            is_healthy = response.status_code == 200
            logger.info(f"Ollama API health check: {'OK' if is_healthy else 'FAILED'}")
            return is_healthy
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False