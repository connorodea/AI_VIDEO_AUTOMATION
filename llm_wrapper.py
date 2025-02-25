# models/llm_wrapper.py

import json
import os
from typing import Dict, List, Optional, Union, Any

class LLMProvider:
    """
    A wrapper for various LLM providers (OpenAI, Anthropic, Llama, etc.).
    Handles API communication and provides a unified interface.
    """
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize the LLM provider.
        
        Args:
            provider: The LLM provider to use ('openai', 'anthropic', 'huggingface', etc.)
            model: The specific model to use
            api_key: API key for the provider (if None, will look for it in environment variables)
        """
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or self._get_api_key(provider)
        self.client = self._initialize_client()
        
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from environment variables or config file."""
        # Try environment variables first
        env_var_name = f"{provider.upper()}_API_KEY"
        if env_var_name in os.environ:
            return os.environ[env_var_name]
        
        # Try config file
        try:
            with open("config/api_keys.json", "r") as f:
                keys = json.load(f)
                return keys.get(provider)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        print(f"Warning: No API key found for {provider}. You'll need to provide it explicitly.")
        return None
    
    def _initialize_client(self) -> Any:
        """Initialize the client for the selected provider."""
        if self.provider == "openai":
            try:
                from openai import OpenAI
                return OpenAI(api_key=self.api_key)
            except ImportError:
                print("Error: OpenAI package not installed. Run 'pip install openai'")
        
        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                return Anthropic(api_key=self.api_key)
            except ImportError:
                print("Error: Anthropic package not installed. Run 'pip install anthropic'")
        
        elif self.provider == "huggingface":
            try:
                from huggingface_hub import InferenceClient
                return InferenceClient(token=self.api_key)
            except ImportError:
                print("Error: Hugging Face package not installed. Run 'pip install huggingface_hub'")
        
        elif self.provider == "local":
            # Local models like Llama.cpp
            try:
                # This is a placeholder - actual implementation depends on how you run local models
                return {"model_path": self.model}
            except Exception as e:
                print(f"Error initializing local model: {e}")
        
        return None
    
    def generate(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.7, 
                 system_message: Optional[str] = None, **kwargs) -> str:
        """
        Generate text using the configured LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter for generation
            system_message: Optional system message (for models that support it)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text
        """
        if self.provider == "openai":
            return self._generate_openai(prompt, max_tokens, temperature, system_message, **kwargs)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, max_tokens, temperature, system_message, **kwargs)
        elif self.provider == "huggingface":
            return self._generate_huggingface(prompt, max_tokens, temperature, **kwargs)
        elif self.provider == "local":
            return self._generate_local(prompt, max_tokens, temperature, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_openai(self, prompt: str, max_tokens: int, temperature: float, 
                        system_message: Optional[str], **kwargs) -> str:
        """Generate text using OpenAI's API."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating with OpenAI: {e}")
            return f"Error: {str(e)}"
    
    def _generate_anthropic(self, prompt: str, max_tokens: int, temperature: float, 
                           system_message: Optional[str], **kwargs) -> str:
        """Generate text using Anthropic's API."""
        if not self.client:
            raise ValueError("Anthropic client not initialized")
        
        try:
            if system_message:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_message,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                )
            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                )
            return response.content[0].text
        except Exception as e:
            print(f"Error generating with Anthropic: {e}")
            return f"Error: {str(e)}"
    
    def _generate_huggingface(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate text using Hugging Face inference API."""
        if not self.client:
            raise ValueError("Hugging Face client not initialized")
        
        try:
            response = self.client.text_generation(
                prompt=prompt,
                model=self.model,
                max_new_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response
        except Exception as e:
            print(f"Error generating with Hugging Face: {e}")
            return f"Error: {str(e)}"
    
    def _generate_local(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate text using a local model."""
        # This is a placeholder - implement based on your local setup
        try:
            # Example for llama.cpp server
            import requests
            response = requests.post(
                "http://localhost:8080/completion",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    **kwargs
                }
            )
            return response.json()["content"]
        except Exception as e:
            print(f"Error generating with local model: {e}")
            return f"Error: {str(e)}"
    
    def get_token_count(self, text: str) -> int:
        """
        Estimate token count for a given text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Estimated token count
        """
        # A very rough estimate - about 4 characters per token for English
        # For production, use a proper tokenizer for your specific model
        return len(text) // 4
    
    def stream_generate(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.7,
                       system_message: Optional[str] = None, **kwargs):
        """
        Stream generated text using the configured LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter for generation
            system_message: Optional system message (for models that support it)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generator yielding text chunks
        """
        if self.provider == "openai":
            return self._stream_openai(prompt, max_tokens, temperature, system_message, **kwargs)
        elif self.provider == "anthropic":
            return self._stream_anthropic(prompt, max_tokens, temperature, system_message, **kwargs)
        else:
            # Fall back to non-streaming for unsupported providers
            yield self.generate(prompt, max_tokens, temperature, system_message, **kwargs)
    
    def _stream_openai(self, prompt: str, max_tokens: int, temperature: float, 
                      system_message: Optional[str], **kwargs):
        """Stream text using OpenAI's API."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"Error streaming with OpenAI: {e}")
            yield f"Error: {str(e)}"
    
    def _stream_anthropic(self, prompt: str, max_tokens: int, temperature: float, 
                         system_message: Optional[str], **kwargs):
        """Stream text using Anthropic's API."""
        if not self.client:
            raise ValueError("Anthropic client not initialized")
        
        try:
            if system_message:
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_message,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                ) as stream:
                    for text in stream.text_stream:
                        yield text
            else:
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                ) as stream:
                    for text in stream.text_stream:
                        yield text
        except Exception as e:
            print(f"Error streaming with Anthropic: {e}")
            yield f"Error: {str(e)}"
