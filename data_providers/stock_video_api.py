# data_providers/stock_video_api.py

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
logger = logging.getLogger('stock_video_api')

class StockVideoAPI:
    """
    Interface to stock video APIs (Pexels, Pixabay, etc.) for retrieving video content.
    """

    def __init__(self, config_path: str = "config/default_settings.json"):
        """
        Initialize the StockVideoAPI with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.providers = self.config.get("visual_assets", {}).get("providers", ["pexels", "pixabay"])
        
        # Load API keys
        self.api_keys = self._load_api_keys()
        
        # Initialize providers
        self.provider_handlers = {
            "pexels": self._search_pexels,
            "pixabay": self._search_pixabay,
            "videvo": self._search_videvo
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
                    "providers": ["pexels", "pixabay"],
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
              min_duration: float = 5.0,
              max_duration: float = 30.0,
              orientation: str = "landscape",
              **kwargs) -> List[Dict]:
        """
        Search for videos across all configured providers.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            min_width: Minimum video width
            min_height: Minimum video height
            min_duration: Minimum video duration in seconds
            max_duration: Maximum video duration in seconds
            orientation: Video orientation (landscape, portrait, square)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of video information dictionaries
        """
        logger.info(f"Searching for videos: {query}")
        
        # Check if we have cached results
        cache_key = self._generate_cache_key(
            query=query,
            min_width=min_width,
            min_height=min_height,
            min_duration=min_duration,
            max_duration=max_duration,
            orientation=orientation,
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
                        min_duration=min_duration,
                        max_duration=max_duration,
                        orientation=orientation,
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
    
    def _search_pexels(self, 
                      query: str, 
                      max_results: int = 10, 
                      min_width: int = 1280, 
                      min_height: int = 720,
                      min_duration: float = 5.0,
                      max_duration: float = 30.0,
                      orientation: str = "landscape",
                      **kwargs) -> List[Dict]:
        """
        Search for videos on Pexels.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            min_width: Minimum video width
            min_height: Minimum video height
            min_duration: Minimum video duration in seconds
            max_duration: Maximum video duration in seconds
            orientation: Video orientation (landscape, portrait, square)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of video information dictionaries
        """
        api_key = self.api_keys.get("pexels")
        if not api_key:
            logger.error("No API key for Pexels")
            return []
        
        headers = {
            "Authorization": api_key
        }
        
        # Pexels API URL
        url = "https://api.pexels.com/videos/search"
        
        # Pexels API parameters
        params = {
            "query": query,
            "per_page": min(80, max_results * 2),  # Fetch more to filter
            "orientation": orientation,
            "size": "large"  # This is a rough filter, we'll filter more precisely later
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            videos = data.get("videos", [])
            
            results = []
            for video in videos:
                # Get the best quality video file
                video_files = video.get("video_files", [])
                
                # Sort by height descending to get the highest quality first
                video_files.sort(key=lambda x: x.get("height", 0), reverse=True)
                
                for video_file in video_files:
                    width = video_file.get("width", 0)
                    height = video_file.get("height", 0)
                    
                    # Check if it meets our requirements
                    if width >= min_width and height >= min_height:
                        duration = video.get("duration", 0)
                        
                        if min_duration <= duration <= max_duration:
                            results.append({
                                "id": f"pexels_{video.get('id')}",
                                "url": video.get("url"),
                                "download_url": video_file.get("link"),
                                "width": width,
                                "height": height,
                                "duration": duration,
                                "author": video.get("user", {}).get("name"),
                                "source": "pexels"
                            })
                            break  # We found a suitable file for this video
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching Pexels: {str(e)}")
            return []
    
    def _search_pixabay(self, 
                       query: str, 
                       max_results: int = 10, 
                       min_width: int = 1280, 
                       min_height: int = 720,
                       min_duration: float = 5.0,
                       max_duration: float = 30.0,
                       orientation: str = "landscape",
                       **kwargs) -> List[Dict]:
        """
        Search for videos on Pixabay.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            min_width: Minimum video width
            min_height: Minimum video height
            min_duration: Minimum video duration in seconds
            max_duration: Maximum video duration in seconds
            orientation: Video orientation (landscape, portrait, square)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of video information dictionaries
        """
        api_key = self.api_keys.get("pixabay")
        if not api_key:
            logger.error("No API key for Pixabay")
            return []
        
        # Pixabay API URL
        url = "https://pixabay.com/api/videos/"
        
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
            "min_height": min_height
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            videos = data.get("hits", [])
            
            results = []
            for video in videos:
                # Get the best quality video file
                videos_urls = video.get("videos", {})
                
                # Try to get the large size first
                video_file = videos_urls.get("large", {})
                if not video_file:
                    video_file = videos_urls.get("medium", {})
                if not video_file:
                    video_file = videos_urls.get("small", {})
                
                if video_file:
                    width = video_file.get("width", 0)
                    height = video_file.get("height", 0)
                    
                    # Check if it meets our requirements
                    if width >= min_width and height >= min_height:
                        duration = video.get("duration", 0)
                        
                        if min_duration <= duration <= max_duration:
                            results.append({
                                "id": f"pixabay_{video.get('id')}",
                                "url": video.get("pageURL"),
                                "download_url": video_file.get("url"),
                                "width": width,
                                "height": height,
                                "duration": duration,
                                "author": video.get("user"),
                                "source": "pixabay"
                            })
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching Pixabay: {str(e)}")
            return []
    
    def _search_videvo(self, 
                      query: str, 
                      max_results: int = 10, 
                      min_width: int = 1280, 
                      min_height: int = 720,
                      min_duration: float = 5.0,
                      max_duration: float = 30.0,
                      orientation: str = "landscape",
                      **kwargs) -> List[Dict]:
        """
        Search for videos on Videvo.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            min_width: Minimum video width
            min_height: Minimum video height
            min_duration: Minimum video duration in seconds
            max_duration: Maximum video duration in seconds
            orientation: Video orientation (landscape, portrait, square)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of video information dictionaries
        """
        api_key = self.api_keys.get("videvo")
        if not api_key:
            logger.error("No API key for Videvo")
            return []
        
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        # Videvo API URL
        url = "https://api.videvo.net/search"
        
        # Videvo API parameters
        params = {
            "query": query,
            "sort_by": "popular",
            "page": 1,
            "page_size": min(30, max_results * 2),  # Fetch more to filter
            "content_type": "video"
        }
        
        # Add resolution filter if available
        if min_width >= 3840 and min_height >= 2160:
            params["res"] = "4k"
        elif min_width >= 1920 and min_height >= 1080:
            params["res"] = "hd"
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            videos = data.get("data", [])
            
            results = []
            for video in videos:
                # Get the clip details
                clip_id = video.get("clip_id")
                clip_url = f"https://api.videvo.net/clip/{clip_id}"
                
                clip_response = requests.get(clip_url, headers=headers)
                clip_response.raise_for_status()
                
                clip_data = clip_response.json()
                
                # Get the download URL
                download_url = clip_data.get("downloads", {}).get("mp4", {}).get("url")
                
                if download_url:
                    width = clip_data.get("width", 0)
                    height = clip_data.get("height", 0)
                    
                    # Check if it meets our requirements
                    if width >= min_width and height >= min_height:
                        duration = clip_data.get("duration", 0)
                        
                        if min_duration <= duration <= max_duration:
                            results.append({
                                "id": f"videvo_{clip_id}",
                                "url": clip_data.get("url"),
                                "download_url": download_url,
                                "width": width,
                                "height": height,
                                "duration": duration,
                                "author": clip_data.get("author_name"),
                                "source": "videvo"
                            })
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching Videvo: {str(e)}")
            return []
    
    def _generate_cache_key(self, **kwargs) -> str:
        """Generate a cache key from search parameters."""
        # Sort the kwargs to ensure consistent cache keys
        sorted_kwargs = sorted(kwargs.items())
        
        # Create a string representation
        key_parts = []
        for k, v in sorted_kwargs:
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
