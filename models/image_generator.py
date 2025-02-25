# models/image_generator.py

import os
import json
import requests
import base64
import tempfile
from io import BytesIO
import logging
from typing import Dict, List, Optional, Any
import time
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('image_generator')

class ImageGenerator:
    """
    Generates images using AI models (Stable Diffusion, DALL-E, etc.).
    """

    def __init__(self, config_path: str = "config/default_settings.json"):
        """
        Initialize the ImageGenerator with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.image_gen_config = self.config.get("image_generation", {})
        
        # Default provider (can be "openai", "stability", "replicate", or "local")
        self.provider = self.image_gen_config.get("provider", "openai")
        
        # Model settings
        self.model = self.image_gen_config.get("model", "dall-e-3") if self.provider == "openai" else "stable-diffusion-xl"
        
        # Default parameters
        self.default_width = self.image_gen_config.get("width", 1024)
        self.default_height = self.image_gen_config.get("height", 1024)
        self.default_style = self.image_gen_config.get("style", "photorealistic")
        self.default_quality = self.image_gen_config.get("quality", "standard")
        
        # Load API keys
        self.api_keys = self._load_api_keys()
        
        # Provider handlers
        self.provider_handlers = {
            "openai": self._generate_openai,
            "stability": self._generate_stability,
            "replicate": self._generate_replicate,
            "local": self._generate_local
        }

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
            return {
                "image_generation": {
                    "provider": "openai",
                    "model": "dall-e-3",
                    "width": 1024,
                    "height": 1024,
                    "style": "photorealistic",
                    "quality": "standard"
                }
            }
    
    def _load_api_keys(self) -> Dict:
        """Load API keys from the config file."""
        api_keys = {}
        
        try:
            with open("config/api_keys.json", "r") as f:
                keys = json.load(f)
                
                # Map config keys to providers
                key_mapping = {
                    "openai": "openai",
                    "stability": "stability_ai",
                    "replicate": "replicate"
                }
                
                for provider, key_name in key_mapping.items():
                    if key_name in keys:
                        api_keys[provider] = keys[key_name]
                    else:
                        logger.warning(f"No API key found for {provider}")
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning("Failed to load API keys")
        
        return api_keys
    
    def generate_image(
        self,
        prompt: str,
        output_path: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        style: Optional[str] = None,
        quality: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Generate an image using the configured provider.
        
        Args:
            prompt: The prompt to generate an image from
            output_path: Path to save the generated image
            width: Width of the image
            height: Height of the image
            provider: Provider to use (openai, stability, replicate, local)
            model: Model to use (provider-specific)
            style: Style of the image (provider-specific)
            quality: Quality level (provider-specific)
            negative_prompt: Things to avoid in the image
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Path to the generated image, or None if generation failed
        """
        # Use provided values or defaults
        width = width or self.default_width
        height = height or self.default_height
        provider = provider or self.provider
        model = model or self.model
        style = style or self.default_style
        quality = quality or self.default_quality
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Check if we have an API key for the provider
        if provider not in self.api_keys and provider != "local":
            logger.error(f"No API key found for {provider}")
            return None
        
        # Check if we have a handler for the provider
        if provider not in self.provider_handlers:
            logger.error(f"Unsupported provider: {provider}")
            return None
        
        # Call the appropriate provider handler
        try:
            return self.provider_handlers[provider](
                prompt=prompt,
                output_path=output_path,
                width=width,
                height=height,
                model=model,
                style=style,
                quality=quality,
                negative_prompt=negative_prompt,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            return None
    
    def _generate_openai(
        self,
        prompt: str,
        output_path: str,
        width: int,
        height: int,
        model: str,
        style: str,
        quality: str,
        **kwargs
    ) -> Optional[str]:
        """
        Generate an image using OpenAI (DALL-E).
        
        Args:
            prompt: The prompt to generate an image from
            output_path: Path to save the generated image
            width: Width of the image
            height: Height of the image
            model: Model to use (e.g., "dall-e-3")
            style: Style of the image (e.g., "vivid", "natural")
            quality: Quality level (e.g., "standard", "hd")
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Path to the generated image, or None if generation failed
        """
        api_key = self.api_keys.get("openai")
        if not api_key:
            logger.error("No API key for OpenAI")
            return None
        
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            # Map size based on width and height
            # DALL-E 3 supports only specific sizes: 1024x1024, 1024x1792, 1792x1024
            if width == height:
                size = "1024x1024"
            elif width > height:
                size = "1792x1024"
            else:
                size = "1024x1792"
            
            # Generate the image
            response = client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                style=style,
                n=1
            )
            
            # Download the image
            image_url = response.data[0].url
            response = requests.get(image_url)
            response.raise_for_status()
            
            # Save the image
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error generating image with OpenAI: {str(e)}")
            return None
    
    def _generate_stability(
        self,
        prompt: str,
        output_path: str,
        width: int,
        height: int,
        model: str,
        style: str,
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Generate an image using Stability AI.
        
        Args:
            prompt: The prompt to generate an image from
            output_path: Path to save the generated image
            width: Width of the image
            height: Height of the image
            model: Model to use (e.g., "stable-diffusion-xl")
            style: Style of the image
            negative_prompt: Things to avoid in the image
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Path to the generated image, or None if generation failed
        """
        api_key = self.api_keys.get("stability")
        if not api_key:
            logger.error("No API key for Stability AI")
            return None
        
        try:
            # Stability AI API endpoint
            if model == "stable-diffusion-xl":
                url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
            else:
                url = f"https://api.stability.ai/v1/generation/{model}/text-to-image"
            
            # Prepare the request
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            # Round dimensions to the nearest 64 pixels
            width = round(width / 64) * 64
            height = round(height / 64) * 64
            
            # Stability has size limitations
            width = min(max(width, 512), 1536)
            height = min(max(height, 512), 1536)
            
            data = {
                "text_prompts": [{"text": prompt, "weight": 1.0}],
                "cfg_scale": kwargs.get("cfg_scale", 7.0),
                "samples": 1,
                "steps": kwargs.get("steps", 30),
                "width": width,
                "height": height
            }
            
            # Add negative prompt if provided
            if negative_prompt:
                data["text_prompts"].append({"text": negative_prompt, "weight": -1.0})
            
            # Add style preset if available
            if style:
                data["style_preset"] = style
            
            # Make the request
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            image_data = result["artifacts"][0]["base64"]
            
            # Decode and save the image
            image_bytes = base64.b64decode(image_data)
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error generating image with Stability AI: {str(e)}")
            return None
    
    def _generate_replicate(
        self,
        prompt: str,
        output_path: str,
        width: int,
        height: int,
        model: str,
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Generate an image using Replicate.
        
        Args:
            prompt: The prompt to generate an image from
            output_path: Path to save the generated image
            width: Width of the image
            height: Height of the image
            model: Model to use (e.g., "stability-ai/sdxl")
            negative_prompt: Things to avoid in the image
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Path to the generated image, or None if generation failed
        """
        api_key = self.api_keys.get("replicate")
        if not api_key:
            logger.error("No API key for Replicate")
            return None
        
        try:
            # Replicate API endpoint
            url = "https://api.replicate.com/v1/predictions"
            
            # Prepare the request
            headers = {
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json"
            }
            
            # Set the model ID
            if model == "stable-diffusion-xl":
                model_id = "stability-ai/sdxl:2b017d9b67edd2ee1401238df49d75da53c523f36e363605806eaa352496785d"
            else:
                model_id = model
            
            # Prepare the input parameters
            input_data = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_outputs": 1,
                "scheduler": "K_EULER_ANCESTRAL",
                "num_inference_steps": kwargs.get("steps", 30)
            }
            
            # Add negative prompt if provided
            if negative_prompt:
                input_data["negative_prompt"] = negative_prompt
            
            # Create the prediction
            data = {
                "version": model_id,
                "input": input_data
            }
            
            # Start the prediction
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            prediction = response.json()
            
            # Wait for the prediction to complete
            prediction_id = prediction["id"]
            poll_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
            
            while True:
                response = requests.get(poll_url, headers=headers)
                response.raise_for_status()
                prediction = response.json()
                
                if prediction["status"] == "succeeded":
                    break
                elif prediction["status"] == "failed":
                    logger.error(f"Replicate prediction failed: {prediction.get('error')}")
                    return None
                
                time.sleep(1)
            
            # Get the output image URL
            image_url = prediction["output"][0]
            
            # Download the image
            response = requests.get(image_url)
            response.raise_for_status()
            
            # Save the image
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error generating image with Replicate: {str(e)}")
            return None
    
    def _generate_local(
        self,
        prompt: str,
        output_path: str,
        width: int,
        height: int,
        model: str,
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Generate an image using a local Stable Diffusion model.
        
        Args:
            prompt: The prompt to generate an image from
            output_path: Path to save the generated image
            width: Width of the image
            height: Height of the image
            model: Model to use (path to local model)
            negative_prompt: Things to avoid in the image
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Path to the generated image, or None if generation failed
        """
        try:
            # Check if diffusers is installed
            try:
                from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
                import torch
            except ImportError:
                logger.error("diffusers and torch are required for local image generation")
                return None
            
            # Load the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Use default model path if not specified
            if model == "stable-diffusion-xl":
                model_path = "stabilityai/stable-diffusion-xl-base-1.0"
            else:
                model_path = model
            
            # Create the pipeline
            pipe = StableDiffusionPipeline.from_pretrained(model_path)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(device)
            
            # Enable attention slicing if on CPU
            if device == "cpu":
                pipe.enable_attention_slicing()
            
            # Generate the image
            generator = torch.Generator(device=device).manual_seed(kwargs.get("seed", random.randint(0, 2**32 - 1)))
            
            # Set the parameters
            params = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": kwargs.get("steps", 30),
                "guidance_scale": kwargs.get("cfg_scale", 7.0),
                "generator": generator
            }
            
            # Generate the image
            image = pipe(**params).images[0]
            
            # Save the image
            image.save(output_path)
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error generating image locally: {str(e)}")
            return None
