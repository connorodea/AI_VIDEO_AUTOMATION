# data_providers/stock_image_api.py

import os
import json
import requests
import time
import logging
from typing import Dict, List, Optional, Any
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stock_image_api')

class StockImageAPI:
    """
    Interface to stock image APIs (Unsplash, Pexels, Pixabay, etc.) for retrieving image content.
    """

    def __init__(self, config_path: str = "config/default_settings.json"):
        """
        Initialize the StockImageAPI with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.providers = self.config.get("visual_assets", {}).get("providers", ["unsplash", "pexels", "pixabay"])
        
        # Load API keys
        self.api_keys = self._load_api_keys()
        
        # Initialize providers
        self.provider_handlers = {
            "unsplash": self._search_unsplash,
            "pexels": self._search_pexels,
            "pixabay": self._search_pixabay
        }
        
        # Cache directory for search results
        self.cache_dir = os.path.join("cache", "search_results")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Cache expiration (24 hours)
        self.cache_expiration = 86400

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
            return {
                "visual_assets": {
                    "providers": ["unsplash", "pexels", "pixabay"],
                    "cache_dir": "cache/media"
                }
            }
    
    def _load_api_keys(self) -> Dict:
        """Load API keys from the config file."""
        api_keys = {}
        
        try:
            with open("config/api_keys.json", "r") as f:
                keys = json.load(f)
                
                for provider in self.providers:
                    key_name = f"{provider}"
                    if key_name in keys:
                        api_keys[provider] = keys[key_name]
                    else:
                        logger.warning(f"No API key found for {provider}")
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning("Failed to load API keys")
        
        return api_keys
    
    def search(self, 
              query: str, 
              max_results: int = 10, 
              min_width: int = 1280, 
              min_height: int = 720,
              orientation: str = "landscape",
              color: Optional[str] = None,
              **kwargs) -> List[Dict]:
        """
        Search for images across all configured providers.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            min_width: Minimum image width
            min_height: Minimum image height
            orientation: Image orientation (landscape, portrait, square)
            color: Dominant color (if supported by provider)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of image information dictionaries
        """
        logger.info(f"Searching for images: {query}")
        
        # Check if we have cached results
        cache_key = self._generate_cache_key(
            query=query,
            min_width=min_width,
            min_height=min_height,
            orientation=orientation,
            color=color,
            **kwargs
        )
        
        cached_results = self._get_cached_results(cache_key)
        if cached_results:
            logger.info(f"Found {len(cached_results)} cached results for '{query}'")
            # Return up to max_results entries
            return cached_results[:max_results]
        
        # Aggregate results from all providers
        all_results = []
        results_per_provider = max(1, max_results // len(self.providers))
        
        # Shuffle providers to distribute load
        providers = list(self.providers)
        random.shuffle(providers)
        
        for provider in providers:
            if provider in self.provider_handlers and provider in self.api_keys:
                try:
                    logger.info(f"Searching {provider} for '{query}'")
                    provider_results = self.provider_handlers[provider](
                        query=query,
                        max_results=results_per_provider,
                        min_width=min_width,
                        min_height=min_height,
                        orientation=orientation,
                        color=color,
                        **kwargs
                    )
                    
                    if provider_results:
                        all_results.extend(provider_results)
                        logger.info(f"Found {len(provider_results)} results from {provider}")
                except Exception as e:
                    logger.error(f"Error searching {provider}: {str(e)}")
            else:
                logger.warning(f"Provider {provider} not configured or missing API key")
        
        # Cache the results
        if all_results:
            self._cache_results(cache_key, all_results)
        
        # Shuffle and return up to max_results entries
        random.shuffle(all_results)
        return all_results[:max_results]
    
    def _search_unsplash(self, 
                        query: str, 
                        max_results: int = 10, 
                        min_width: int = 1280, 
                        min_height: int = 720,
                        orientation: str = "landscape",
                        color: Optional[str] = None,
                        **kwargs) -> List[Dict]:
        """
        Search for images on Unsplash.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            min_width: Minimum image width
            min_height: Minimum image height
            orientation: Image orientation (landscape, portrait, square)
            color: Dominant color (if supported by provider)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of image information dictionaries
        """
        api_key = self.api_keys.get("unsplash")
        if not api_key:
            logger.error("No API key for Unsplash")
            return []
        
        # Unsplash API URL
        url = "https://api.unsplash.com/search/photos"
        
        # Unsplash API parameters
        params = {
            "query": query,
            "per_page": min(30, max_results),
            "orientation": orientation,
            "client_id": api_key
        }
        
        # Add color filter if specified
        if color:
            params["color"] = color
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            images = data.get("results", [])
            
            results = []
            for image in images:
                urls = image.get("urls", {})
                width = image.get("width", 0)
                height = image.get("height", 0)
                
                # Check if it meets our requirements
                if width >= min_width and height >= min_height:
                    results.append({
                        "id": f"unsplash_{image.get('id')}",
                        "url": image.get("links", {}).get("html"),
                        "download_url": urls.get("raw") or urls.get("full") or urls.get("regular"),
                        "width": width,
                        "height": height,
                        "author": image.get("user", {}).get("name"),
                        "source": "unsplash"
                    })
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching Unsplash: {str(e)}")
            return []
    
    def _search_pexels(self, 
                      query: str, 
                      max_results: int = 10, 
                      min_width: int = 1280, 
                      min_height: int = 720,
                      orientation: str = "landscape",
                      color: Optional[str] = None,
                      **kwargs) -> List[Dict]:
        """
        Search for images on Pexels.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            min_width: Minimum image width
            min_height: Minimum image height
            orientation: Image orientation (landscape, portrait, square)
            color: Dominant color (if supported by provider)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of image information dictionaries
        """
        api_key = self.api_keys.get("pexels")
        if not api_key:
            logger.error("No API key for Pexels")
            return []
        
        headers = {
            "Authorization": api_key
        }
        
        # Pexels API URL
        url = "https://api.pexels.com/v1/search"
        
        # Pexels API parameters
        params = {
            "query": query,
            "per_page": min(80, max_results * 2),  # Fetch more to filter
            "orientation": orientation,
            "size": "large"  # This is a rough filter, we'll filter more precisely later
        }
        
        # Add color filter if specified
        if color:
            params["color"] = color
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            images = data.get("photos", [])
            
            results = []
            for image in images:
                src = image.get("src", {})
                width = image.get("width", 0)
                height = image.get("height", 0)
                
                # Check if it meets our requirements
                if width >= min_width and height >= min_height:
                    results.append({
                        "id": f"pexels_{image.get('id')}",
                        "url": image.get("url"),
                        "download_url": src.get("original") or src.get("large2x") or src.get("large"),
                        "width": width,
                        "height": height,
                        "author": image.get("photographer"),
                        "source": "pexels"
                    })
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching Pexels: {str(e)}")
            return []
    
    def _search_pixabay(self, 
                       query: str, 
                       max_results: int = 10, 
                       min_width: int = 1280, 
                       min_height: int = 720,
                       orientation: str = "landscape",
                       color: Optional[str] = None,
                       **kwargs) -> List[Dict]:
        """
        Search for images on Pixabay.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            min_width: Minimum image width
            min_height: Minimum image height
            orientation: Image orientation (landscape, portrait, square)
            color: Dominant color (if supported by provider)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of image information dictionaries
        """
        api_key = self.api_keys.get("pixabay")
        if not api_key:
            logger.error("No API key for Pixabay")
            return []
        
        # Pixabay API URL
        url = "https://pixabay.com/api/"
        
        # Map orientation to Pixabay's format
        pixabay_orientation = {
            "landscape": "horizontal",
            "portrait": "vertical",
            "square": "horizontal"  # Pixabay doesn't have square option
        }.get(orientation, "horizontal")
        
        # Pixabay API parameters
        params = {
            "key": api_key,
            "q": query,
            "per_page": min(50, max_results * 2),  # Fetch more to filter
            "orientation": pixabay_orientation,
            "min_width": min_width,
            "min_height": min_height,
            "image_type": "photo"
        }
        
        # Add color filter if specified
        if color:
            params["colors"] = color
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            images = data.get("hits", [])
            
            results = []
            for image in images:
                # Check if it meets our requirements
                width = image.get("imageWidth", 0)
                height = image.get("imageHeight", 0)
                
                if width >= min_width and height >= min_height:
                    results.append({
                        "id": f"pixabay_{image.get('id')}",
                        "url": image.get("pageURL"),
                        "download_url": image.get("largeImageURL"),
                        "width": width,
                        "height": height,
                        "author": image.get("user"),
                        "source": "pixabay"
                    })
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching Pixabay: {str(e)}")
            return []
    
    def _generate_cache_key(self, **kwargs) -> str:
        """Generate a cache key from search parameters."""
        # Sort the kwargs to ensure consistent cache keys
        sorted_kwargs = sorted(kwargs.items())
        
        # Create a string representation
        key_parts = []
        for k, v in sorted_kwargs:
            if v is not None:
                key_parts.append(f"{k}={v}")
        
        # Join with underscores and convert to a safe filename
        key = "_".join(key_parts)
        
        # Make it filesystem safe
        import hashlib
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cached_results(self, cache_key: str) -> Optional[List[Dict]]:
        """Get cached search results if available and not expired."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            # Check if the cache is still valid
            file_age = time.time() - os.path.getmtime(cache_file)
            
            if file_age < self.cache_expiration:
                try:
                    with open(cache_file, "r") as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError):
                    # If there's an error reading the cache, ignore it
                    pass
        
        return None
    
    def _cache_results(self, cache_key: str, results: List[Dict]) -> None:
        """Cache search results for future use."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, "w") as f:
                json.dump(results, f)
        except IOError as e:
            logger.error(f"Error caching results: {str(e)}")
