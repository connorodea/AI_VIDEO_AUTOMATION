# core/media_selector.py

import json
import os
import time
import random
import re
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from urllib.parse import urlparse

# Optional imports (imported at runtime if needed)
# from models.image_generator import ImageGenerator
# from data_providers.stock_video_api import StockVideoAPI
# from data_providers.stock_image_api import StockImageAPI


class MediaSelector:
    """
    Selects and retrieves visual assets (images, videos) for each script segment.
    Sources include stock media APIs, web scraping, local media, and AI-generated assets.
    """

    def __init__(self, config_path: str = "config/default_settings.json"):
        """
        Initialize the MediaSelector with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.media_config = self.config.get("visual_assets", {})
        self.providers = self.media_config.get("providers", ["pexels", "pixabay", "unsplash"])
        self.prefer_video = self.media_config.get("prefer_video_over_image", True)
        self.max_assets_per_segment = self.media_config.get("max_assets_per_segment", 3)
        self.min_resolution = self.media_config.get("min_resolution", {"width": 1280, "height": 720})
        self.aspect_ratio = self.media_config.get("aspect_ratio", "16:9")
        self.cache_dir = self.media_config.get("cache_dir", "cache/media")
        
        # Initialize API clients (lazy loading)
        self._video_api = None
        self._image_api = None
        self._image_generator = None
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using default settings.")
            return {
                "visual_assets": {
                    "providers": ["pexels", "pixabay", "unsplash"],
                    "max_assets_per_segment": 3,
                    "prefer_video_over_image": True,
                    "min_resolution": {
                        "width": 1280,
                        "height": 720
                    },
                    "aspect_ratio": "16:9",
                    "cache_dir": "cache/media"
                }
            }
    
    @property
    def video_api(self):
        """Lazy-load the video API client."""
        if self._video_api is None:
            try:
                from data_providers.stock_video_api import StockVideoAPI
                self._video_api = StockVideoAPI()
            except ImportError:
                # Create a minimal implementation if module doesn't exist yet
                print("Warning: StockVideoAPI module not found. Using a minimal implementation.")
                self._video_api = MinimalStockAPI("video")
        return self._video_api
    
    @property
    def image_api(self):
        """Lazy-load the image API client."""
        if self._image_api is None:
            try:
                from data_providers.stock_image_api import StockImageAPI
                self._image_api = StockImageAPI()
            except ImportError:
                # Create a minimal implementation if module doesn't exist yet
                print("Warning: StockImageAPI module not found. Using a minimal implementation.")
                self._image_api = MinimalStockAPI("image")
        return self._image_api
    
    @property
    def image_generator(self):
        """Lazy-load the image generator."""
        if self._image_generator is None:
            try:
                from models.image_generator import ImageGenerator
                self._image_generator = ImageGenerator()
            except ImportError:
                # Create a minimal implementation if module doesn't exist yet
                print("Warning: ImageGenerator module not found. Using a minimal implementation.")
                self._image_generator = DummyImageGenerator()
        return self._image_generator
    
    def select_media_for_script(self, script_data: Dict, output_dir: str) -> Dict:
        """
        Select and download media assets for each segment in a script.
        
        Args:
            script_data: Script data from ScriptGenerator
            output_dir: Directory to save downloaded media files
            
        Returns:
            Dictionary with paths to selected media files for each segment
        """
        os.makedirs(output_dir, exist_ok=True)
        
        result = {
            "segments": []
        }
        
        # Process each segment
        for i, segment in enumerate(script_data["segments"]):
            print(f"Selecting media for segment {i+1}/{len(script_data['segments'])}: {segment['title']}")
            
            # Create keywords from the segment content
            keywords = self._extract_keywords(segment["title"], segment["content"])
            
            # Select media for this segment
            segment_result = self.select_media_for_segment(
                segment_title=segment["title"],
                segment_content=segment["content"],
                keywords=keywords,
                segment_index=i,
                output_dir=os.path.join(output_dir, f"segment_{i:03d}"),
                prefer_video=self.prefer_video
            )
            
            # Add segment info to result
            result["segments"].append({
                "segment_index": i,
                "segment_id": segment.get("id", str(i)),
                "timestamp": segment["timestamp"],
                "time_in_seconds": segment["time_in_seconds"],
                "title": segment["title"],
                "media": segment_result
            })
        
        return result
    
    def select_media_for_segment(
        self, 
        segment_title: str, 
        segment_content: str,
        keywords: List[str],
        segment_index: int,
        output_dir: str,
        prefer_video: bool = True
    ) -> Dict:
        """
        Select and download media assets for a single script segment.
        
        Args:
            segment_title: Title of the segment
            segment_content: Content of the segment
            keywords: List of keywords extracted from the segment
            segment_index: Index of the segment in the script
            output_dir: Directory to save downloaded media files
            prefer_video: Whether to prefer videos over images
            
        Returns:
            Dictionary with paths to selected media files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        result = {
            "videos": [],
            "images": [],
            "generated_images": []
        }
        
        # Try to get videos first if preferred
        if prefer_video:
            # Try with different keywords until we get enough videos
            for keyword in keywords[:5]:  # Try up to 5 keywords
                videos = self._find_and_download_videos(
                    keyword, 
                    output_dir, 
                    max_count=self.max_assets_per_segment - len(result["videos"])
                )
                
                # Add to result
                for video in videos:
                    if video not in result["videos"]:
                        result["videos"].append(video)
                
                # Break if we have enough videos
                if len(result["videos"]) >= self.max_assets_per_segment:
                    break
        
        # Get images if we need more assets
        if len(result["videos"]) < self.max_assets_per_segment:
            needed_images = self.max_assets_per_segment - len(result["videos"])
            
            # Try with different keywords until we get enough images
            for keyword in keywords[:5]:  # Try up to 5 keywords
                images = self._find_and_download_images(
                    keyword, 
                    output_dir, 
                    max_count=needed_images - len(result["images"])
                )
                
                # Add to result
                for image in images:
                    if image not in result["images"]:
                        result["images"].append(image)
                
                # Break if we have enough images
                if len(result["images"]) >= needed_images:
                    break
        
        # Generate images with AI if we still need more assets
        if len(result["videos"]) + len(result["images"]) < self.max_assets_per_segment:
            needed_generated = self.max_assets_per_segment - len(result["videos"]) - len(result["images"])
            
            # Create prompts for image generation
            prompts = self._create_image_prompts(segment_title, segment_content, needed_generated)
            
            # Generate images
            for i, prompt in enumerate(prompts):
                try:
                    gen_image_path = self.image_generator.generate_image(
                        prompt=prompt,
                        output_path=os.path.join(output_dir, f"generated_{i:03d}.png"),
                        width=self.min_resolution["width"],
                        height=self.min_resolution["height"]
                    )
                    
                    if gen_image_path:
                        result["generated_images"].append({
                            "path": gen_image_path,
                            "prompt": prompt
                        })
                except Exception as e:
                    print(f"Error generating image: {e}")
        
        return result
    
    def _extract_keywords(self, title: str, content: str) -> List[str]:
        """
        Extract keywords from segment title and content.
        
        Args:
            title: Segment title
            content: Segment content
            
        Returns:
            List of keywords
        """
        # Combine title and content
        text = f"{title} {content}"
        
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()
        
        # Split into words
        words = text.split()
        
        # Remove common stop words
        stop_words = set([
            "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
            "when", "where", "how", "that", "this", "to", "in", "on", "at", "for",
            "with", "about", "by", "from", "up", "down", "of", "off", "over", "under"
        ])
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count word frequency
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Extract top keywords
        keywords = [word for word, count in sorted_words[:20]]
        
        # Add some keyword combinations for better search results
        keyword_combinations = []
        if len(keywords) >= 2:
            for i in range(min(5, len(keywords) - 1)):
                keyword_combinations.append(f"{keywords[i]} {keywords[i+1]}")
        
        # Add the title as a keyword (if not too long)
        if len(title.strip()) <= 50:
            keywords.insert(0, title.strip())
        
        # Add combinations to the beginning of the list
        keywords = keyword_combinations + keywords
        
        return keywords
    
    def _find_and_download_videos(self, keyword: str, output_dir: str, max_count: int = 3) -> List[Dict]:
        """
        Find and download videos for a keyword.
        
        Args:
            keyword: Search keyword
            output_dir: Directory to save downloaded videos
            max_count: Maximum number of videos to download
            
        Returns:
            List of dictionaries with video information
        """
        results = []
        
        # Search for videos
        try:
            videos = self.video_api.search(
                query=keyword,
                max_results=max_count * 3,  # Get more results to filter
                min_width=self.min_resolution["width"],
                min_height=self.min_resolution["height"]
            )
            
            # Filter videos
            filtered_videos = []
            for video in videos:
                # Check if it meets our requirements
                if self._check_video_requirements(video):
                    filtered_videos.append(video)
                
                # Stop if we have enough
                if len(filtered_videos) >= max_count:
                    break
            
            # Download videos
            for i, video in enumerate(filtered_videos[:max_count]):
                try:
                    # Make filename safe
                    safe_keyword = re.sub(r'[^a-zA-Z0-9]', '_', keyword)
                    filename = f"{safe_keyword}_{i:03d}.mp4"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Download the video
                    download_path = self._download_file(video["download_url"], filepath)
                    
                    if download_path:
                        results.append({
                            "type": "video",
                            "path": download_path,
                            "source": video.get("source", "stock"),
                            "query": keyword,
                            "metadata": {
                                "width": video.get("width"),
                                "height": video.get("height"),
                                "duration": video.get("duration"),
                                "author": video.get("author"),
                                "url": video.get("url")
                            }
                        })
                except Exception as e:
                    print(f"Error downloading video {video.get('url')}: {e}")
        
        except Exception as e:
            print(f"Error searching for videos with keyword '{keyword}': {e}")
        
        return results
    
    def _find_and_download_images(self, keyword: str, output_dir: str, max_count: int = 3) -> List[Dict]:
        """
        Find and download images for a keyword.
        
        Args:
            keyword: Search keyword
            output_dir: Directory to save downloaded images
            max_count: Maximum number of images to download
            
        Returns:
            List of dictionaries with image information
        """
        results = []
        
        # Search for images
        try:
            images = self.image_api.search(
                query=keyword,
                max_results=max_count * 3,  # Get more results to filter
                min_width=self.min_resolution["width"],
                min_height=self.min_resolution["height"]
            )
            
            # Filter images
            filtered_images = []
            for image in images:
                # Check if it meets our requirements
                if self._check_image_requirements(image):
                    filtered_images.append(image)
                
                # Stop if we have enough
                if len(filtered_images) >= max_count:
                    break
            
            # Download images
            for i, image in enumerate(filtered_images[:max_count]):
                try:
                    # Make filename safe
                    safe_keyword = re.sub(r'[^a-zA-Z0-9]', '_', keyword)
                    filename = f"{safe_keyword}_{i:03d}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Download the image
                    download_path = self._download_file(image["download_url"], filepath)
                    
                    if download_path:
                        results.append({
                            "type": "image",
                            "path": download_path,
                            "source": image.get("source", "stock"),
                            "query": keyword,
                            "metadata": {
                                "width": image.get("width"),
                                "height": image.get("height"),
                                "author": image.get("author"),
                                "url": image.get("url")
                            }
                        })
                except Exception as e:
                    print(f"Error downloading image {image.get('url')}: {e}")
        
        except Exception as e:
            print(f"Error searching for images with keyword '{keyword}': {e}")
        
        return results
    
    def _check_video_requirements(self, video: Dict) -> bool:
        """Check if a video meets the requirements."""
        # Check resolution
        if video.get("width") < self.min_resolution["width"] or video.get("height") < self.min_resolution["height"]:
            return False
        
        # Check aspect ratio (approximately)
        if "width" in video and "height" in video:
            video_ratio = video["width"] / video["height"]
            
            if self.aspect_ratio == "16:9":
                expected_ratio = 16 / 9
            elif self.aspect_ratio == "4:3":
                expected_ratio = 4 / 3
            elif self.aspect_ratio == "1:1":
                expected_ratio = 1
            else:
                # Parse custom ratio
                parts = self.aspect_ratio.split(":")
                if len(parts) == 2:
                    expected_ratio = float(parts[0]) / float(parts[1])
                else:
                    expected_ratio = 16 / 9  # Default to 16:9
            
            # Allow some flexibility in aspect ratio (±10%)
            ratio_diff = abs(video_ratio - expected_ratio) / expected_ratio
            if ratio_diff > 0.1:
                return False
        
        # Check duration (if available)
        min_duration = self.media_config.get("min_video_duration", 5)
        max_duration = self.media_config.get("max_video_duration", 30)
        
        if "duration" in video:
            if video["duration"] < min_duration or video["duration"] > max_duration:
                return False
        
        return True
    
    def _check_image_requirements(self, image: Dict) -> bool:
        """Check if an image meets the requirements."""
        # Check resolution
        if image.get("width") < self.min_resolution["width"] or image.get("height") < self.min_resolution["height"]:
            return False
        
        # Check aspect ratio (approximately)
        if "width" in image and "height" in image:
            image_ratio = image["width"] / image["height"]
            
            if self.aspect_ratio == "16:9":
                expected_ratio = 16 / 9
            elif self.aspect_ratio == "4:3":
                expected_ratio = 4 / 3
            elif self.aspect_ratio == "1:1":
                expected_ratio = 1
            else:
                # Parse custom ratio
                parts = self.aspect_ratio.split(":")
                if len(parts) == 2:
                    expected_ratio = float(parts[0]) / float(parts[1])
                else:
                    expected_ratio = 16 / 9  # Default to 16:9
            
            # Allow some flexibility in aspect ratio (±10%)
            ratio_diff = abs(image_ratio - expected_ratio) / expected_ratio
            if ratio_diff > 0.1:
                return False
        
        return True
    
    def _download_file(self, url: str, output_path: str) -> Optional[str]:
        """
        Download a file from a URL.
        
        Args:
            url: URL to download
            output_path: Path to save the file
            
        Returns:
            Path to the downloaded file, or None if download failed
        """
        try:
            # Check if file already exists
            if os.path.exists(output_path):
                return output_path
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Download the file
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return output_path
        except Exception as e:
            print(f"Error downloading file from {url}: {e}")
            return None
    
    def _create_image_prompts(self, title: str, content: str, count: int) -> List[str]:
        """
        Create prompts for image generation.
        
        Args:
            title: Segment title
            content: Segment content
            count: Number of prompts to create
            
        Returns:
            List of prompts
        """
        # Extract a few sentences from the content
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Create base prompt from title
        base_prompt = f"{title.strip()}"
        
        # Add style and quality parameters
        style_suffix = ", high quality, photorealistic, professional, detailed"
        
        # Create variations
        prompts = []
        
        # Add title as first prompt
        prompts.append(f"{base_prompt}{style_suffix}")
        
        # Add sentences as additional prompts
        for sentence in sentences[:count-1]:
            if len(sentence) > 10 and len(sentence) < 200:  # Skip very short or long sentences
                prompts.append(f"{sentence.strip()}{style_suffix}")
        
        # If we still need more prompts, create combinations
        while len(prompts) < count:
            if len(sentences) >= 2:
                i, j = random.sample(range(len(sentences)), 2)
                combined = f"{sentences[i].strip()} {sentences[j].strip()}"
                if len(combined) < 200:  # Skip if too long
                    prompts.append(f"{combined}{style_suffix}")
            else:
                # If we don't have enough sentences, just duplicate the title
                prompts.append(f"{base_prompt}{style_suffix}")
            
            # Avoid infinite loop
            if len(prompts) >= count or len(prompts) >= 10:
                break
        
        return prompts[:count]


class MinimalStockAPI:
    """A minimal implementation of a stock media API for testing."""
    
    def __init__(self, media_type: str = "image"):
        self.media_type = media_type
    
    def search(self, query: str, max_results: int = 10, **kwargs) -> List[Dict]:
        """Minimal search implementation that returns placeholder results."""
        print(f"MinimalStockAPI: Searching for {self.media_type}s with query '{query}'")
        
        results = []
        
        # Create placeholder results
        for i in range(max_results):
            if self.media_type == "image":
                results.append({
                    "id": f"placeholder-{i}",
                    "url": f"https://example.com/image-{i}.jpg",
                    "download_url": f"https://placekitten.com/{kwargs.get('min_width', 1280)}/{kwargs.get('min_height', 720)}",
                    "width": kwargs.get("min_width", 1280),
                    "height": kwargs.get("min_height", 720),
                    "author": "Placeholder Author",
                    "source": "placeholder"
                })
            else:  # video
                results.append({
                    "id": f"placeholder-{i}",
                    "url": f"https://example.com/video-{i}.mp4",
                    "download_url": f"https://example.com/video-{i}.mp4",  # This won't work in reality
                    "width": kwargs.get("min_width", 1280),
                    "height": kwargs.get("min_height", 720),
                    "duration": 10,
                    "author": "Placeholder Author",
                    "source": "placeholder"
                })
        
        print(f"MinimalStockAPI: Found {len(results)} placeholder {self.media_type}s")
        return results


class DummyImageGenerator:
    """A dummy image generator that creates placeholder images."""
    
    def generate_image(self, prompt: str, output_path: str, width: int = 1024, height: int = 1024, **kwargs) -> Optional[str]:
        """Generate a placeholder image."""
        print(f"DummyImageGenerator: Generating image with prompt '{prompt}'")
        
        try:
            # Create a placeholder image using requests to placekitten.com
            response = requests.get(f"https://placekitten.com/{width}/{height}", timeout=30)
            response.raise_for_status()
            
            # Save the image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            print(f"DummyImageGenerator: Generated placeholder image at {output_path}")
            return output_path
        except Exception as e:
            print(f"DummyImageGenerator: Error generating image: {e}")
            return None
