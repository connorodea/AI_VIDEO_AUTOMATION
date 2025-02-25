# data_providers/youtube_api.py

import os
import json
import logging
import time
import http.client
import httplib2
import random
import string
from typing import Dict, List, Optional, Any, Tuple

# Google API client libraries
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google.auth.exceptions import RefreshError
except ImportError:
    logging.warning("Google API client libraries not installed. Run 'pip install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2'")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('youtube_api')

class YouTubeAPI:
    """
    Integration with YouTube API for uploading videos and managing channels.
    """
    
    # OAuth scopes required for YouTube uploads
    SCOPES = ["https://www.googleapis.com/auth/youtube.upload", 
              "https://www.googleapis.com/auth/youtube"]
    
    def __init__(self, config_path: str = "config/default_settings.json"):
        """
        Initialize the YouTube API integration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.youtube_config = self.config.get("youtube", {})
        
        # YouTube API settings
        self.credentials_dir = self.youtube_config.get("credentials_dir", "config")
        self.credentials_file = self.youtube_config.get("credentials_file", "youtube_credentials.json")
        self.client_secrets_file = self.youtube_config.get("client_secrets_file", "client_secrets.json")
        self.token_file = os.path.join(self.credentials_dir, "youtube_token.json")
        
        # Default upload settings
        self.default_category = self.youtube_config.get("category", "22")  # 22 is "People & Blogs"
        self.default_privacy = self.youtube_config.get("privacy", "private")
        self.auto_tags = self.youtube_config.get("generate_tags", True)
        self.max_tags = self.youtube_config.get("max_tags", 15)
        
        # Initialize the YouTube API client
        self.youtube = None
        self._initialize_api()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
            return {
                "youtube": {
                    "credentials_dir": "config",
                    "credentials_file": "youtube_credentials.json",
                    "client_secrets_file": "client_secrets.json",
                    "category": "22",
                    "privacy": "private",
                    "generate_tags": True,
                    "max_tags": 15
                }
            }
    
    def _initialize_api(self) -> None:
        """Initialize the YouTube API client."""
        try:
            # Check if credentials are available and valid
            credentials = None
            if os.path.exists(self.token_file):
                try:
                    credentials = Credentials.from_authorized_user_info(
                        json.load(open(self.token_file, 'r')), 
                        self.SCOPES
                    )
                except Exception as e:
                    logger.error(f"Error loading credentials: {e}")
            
            # If credentials don't exist or are invalid, create new ones
            if not credentials or not credentials.valid:
                if credentials and credentials.expired and credentials.refresh_token:
                    try:
                        credentials.refresh(Request())
                    except RefreshError:
                        os.remove(self.token_file)
                        credentials = None
                
                # If still no valid credentials, run the OAuth flow
                if not credentials:
                    client_secrets_path = os.path.join(self.credentials_dir, self.client_secrets_file)
                    if not os.path.exists(client_secrets_path):
                        logger.error(f"Client secrets file not found at {client_secrets_path}")
                        logger.info("Please download the client secrets file from the Google Cloud Console")
                        logger.info("See README.md for instructions")
                        return
                    
                    flow = InstalledAppFlow.from_client_secrets_file(
                        client_secrets_path, 
                        self.SCOPES
                    )
                    credentials = flow.run_local_server(port=0)
                    
                    # Save the credentials for future use
                    with open(self.token_file, 'w') as token:
                        token.write(credentials.to_json())
            
            # Build the YouTube API client
            self.youtube = build('youtube', 'v3', credentials=credentials)
            logger.info("YouTube API initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize YouTube API: {e}")
            self.youtube = None
    
    def upload_video(self, 
                    video_path: str, 
                    title: str, 
                    description: str = None, 
                    tags: List[str] = None,
                    category: str = None,
                    privacy: str = None,
                    thumbnail_path: str = None,
                    notify_subscribers: bool = False,
                    **kwargs) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Upload a video to YouTube.
        
        Args:
            video_path: Path to the video file
            title: Video title
            description: Video description
            tags: List of tags
            category: Video category ID
            privacy: Privacy status (private, unlisted, public)
            thumbnail_path: Path to the thumbnail image
            notify_subscribers: Whether to notify subscribers
            **kwargs: Additional parameters
            
        Returns:
            Tuple containing success status, video ID, and video URL
        """
        if not self.youtube:
            logger.error("YouTube API not initialized")
            return False, None, None
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return False, None, None
        
        # Use defaults if not provided
        category = category or self.default_category
        privacy = privacy or self.default_privacy
        
        # Generate description if not provided
        if not description and kwargs.get("script_data"):
            description = self._generate_description(kwargs.get("script_data"))
        
        # Generate tags if not provided
        if not tags and self.auto_tags and kwargs.get("script_data"):
            tags = self._generate_tags(kwargs.get("script_data"))
        
        # Prepare the request body
        body = {
            "snippet": {
                "title": title,
                "description": description or "",
                "tags": tags or [],
                "categoryId": category
            },
            "status": {
                "privacyStatus": privacy,
                "selfDeclaredMadeForKids": False
            }
        }
        
        # Set notification settings
        if privacy == "public":
            body["status"]["publishAt"] = None  # Publish immediately
            if not notify_subscribers:
                body["status"]["notifySubscribers"] = False
        
        # Set other optional parameters
        if kwargs.get("language"):
            body["snippet"]["defaultLanguage"] = kwargs.get("language")
            body["snippet"]["defaultAudioLanguage"] = kwargs.get("language")
        
        if kwargs.get("embeddable") is not None:
            body["status"]["embeddable"] = kwargs.get("embeddable")
        
        if kwargs.get("license"):
            body["status"]["license"] = kwargs.get("license")
        
        # Prepare the media file
        media = MediaFileUpload(
            video_path,
            mimetype="application/octet-stream",
            resumable=True
        )
        
        try:
            # Insert the video
            logger.info(f"Uploading video: {title}")
            insert_request = self.youtube.videos().insert(
                part=",".join(body.keys()),
                body=body,
                media_body=media
            )
            
            # Execute the request with progress tracking
            response = self._execute_with_progress(insert_request)
            
            if response:
                video_id = response.get("id")
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                logger.info(f"Video uploaded successfully: {video_url}")
                
                # Upload thumbnail if provided
                if thumbnail_path and os.path.exists(thumbnail_path):
                    self._set_thumbnail(video_id, thumbnail_path)
                
                return True, video_id, video_url
            else:
                logger.error("Failed to upload video")
                return False, None, None
        
        except HttpError as e:
            logger.error(f"HTTP error during upload: {e.content.decode()}")
            return False, None, None
        
        except Exception as e:
            logger.error(f"Error uploading video: {e}")
            return False, None, None
    
    def _execute_with_progress(self, request):
        """Execute a request with progress tracking."""
        response = None
        error = None
        retry = 0
        max_retries = 10
        
        while response is None:
            try:
                status, response = request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    logger.info(f"Upload progress: {progress}%")
            except HttpError as e:
                if e.resp.status in [500, 502, 503, 504] and retry < max_retries:
                    retry += 1
                    sleep_time = random.randint(1, 2 ** retry)
                    logger.warning(f"Retrying upload (attempt {retry}/{max_retries}) in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    error = e
                    break
            except Exception as e:
                error = e
                break
        
        if error:
            logger.error(f"Error during upload: {error}")
            return None
        
        return response
    
    def _set_thumbnail(self, video_id: str, thumbnail_path: str) -> bool:
        """Set a custom thumbnail for a video."""
        try:
            with open(thumbnail_path, "rb") as thumbnail_file:
                self.youtube.thumbnails().set(
                    videoId=video_id,
                    media_body=MediaFileUpload(thumbnail_path, resumable=False)
                ).execute()
            
            logger.info("Thumbnail set successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error setting thumbnail: {e}")
            return False
    
    def _generate_description(self, script_data: Dict) -> str:
        """Generate a video description from script data."""
        description = ""
        
        # Add title as the first line
        if "metadata" in script_data and "topic" in script_data["metadata"]:
            description += f"{script_data['metadata']['topic']}\n\n"
        
        # Add a general description
        description += "This video was created using AI Video Generator.\n\n"
        
        # Add content sections
        description += "CONTENTS:\n"
        
        # Add timestamps if available
        if "segments" in script_data:
            for segment in script_data["segments"]:
                if "timestamp" in segment and "title" in segment:
                    description += f"{segment['timestamp']} - {segment['title']}\n"
        
        # Add a call to action
        description += "\nIf you found this video helpful, please like, subscribe, and share!"
        
        return description
    
    def _generate_tags(self, script_data: Dict) -> List[str]:
        """Generate tags from script data."""
        tags = []
        
        # Add topic as a tag
        if "metadata" in script_data and "topic" in script_data["metadata"]:
            topic = script_data["metadata"]["topic"]
            tags.append(topic)
            
            # Add individual words from the topic
            words = topic.split()
            for word in words:
                if len(word) > 3 and word.lower() not in [tag.lower() for tag in tags]:
                    tags.append(word)
        
        # Add keywords if available
        if "metadata" in script_data and "keywords" in script_data["metadata"]:
            keywords = script_data["metadata"]["keywords"]
            if keywords:
                for keyword in keywords:
                    if keyword.lower() not in [tag.lower() for tag in tags]:
                        tags.append(keyword)
        
        # Extract important words from segment titles
        if "segments" in script_data:
            for segment in script_data["segments"]:
                if "title" in segment:
                    words = segment["title"].split()
                    for word in words:
                        if len(word) > 3 and word.lower() not in [tag.lower() for tag in tags]:
                            tags.append(word)
        
        # Limit the number of tags
        return tags[:self.max_tags]
    
    def get_channel_info(self) -> Optional[Dict]:
        """Get information about the authenticated user's channel."""
        if not self.youtube:
            logger.error("YouTube API not initialized")
            return None
        
        try:
            response = self.youtube.channels().list(
                part="snippet,statistics",
                mine=True
            ).execute()
            
            if response["items"]:
                channel = response["items"][0]
                return {
                    "id": channel["id"],
                    "title": channel["snippet"]["title"],
                    "description": channel["snippet"].get("description", ""),
                    "subscribers": channel["statistics"].get("subscriberCount", "0"),
                    "views": channel["statistics"].get("viewCount", "0"),
                    "videos": channel["statistics"].get("videoCount", "0"),
                    "thumbnail": channel["snippet"].get("thumbnails", {}).get("default", {}).get("url", "")
                }
            else:
                logger.error("No channel found for the authenticated user")
                return None
        
        except Exception as e:
            logger.error(f"Error getting channel info: {e}")
            return None
    
    def list_videos(self, max_results: int = 50) -> List[Dict]:
        """List videos from the authenticated user's channel."""
        if not self.youtube:
            logger.error("YouTube API not initialized")
            return []
        
        try:
            response = self.youtube.videos().list(
                part="snippet,statistics,status",
                myRating="like",
                maxResults=max_results
            ).execute()
            
            videos = []
            for item in response.get("items", []):
                videos.append({
                    "id": item["id"],
                    "title": item["snippet"]["title"],
                    "publish_date": item["snippet"]["publishedAt"],
                    "views": item["statistics"].get("viewCount", "0"),
                    "likes": item["statistics"].get("likeCount", "0"),
                    "privacy": item["status"]["privacyStatus"],
                    "url": f"https://www.youtube.com/watch?v={item['id']}"
                })
            
            return videos
        
        except Exception as e:
            logger.error(f"Error listing videos: {e}")
            return []
    
    def create_playlist(self, title: str, description: str = "", privacy: str = "private") -> Optional[str]:
        """Create a new playlist."""
        if not self.youtube:
            logger.error("YouTube API not initialized")
            return None
        
        try:
            playlist = self.youtube.playlists().insert(
                part="snippet,status",
                body={
                    "snippet": {
                        "title": title,
                        "description": description
                    },
                    "status": {
                        "privacyStatus": privacy
                    }
                }
            ).execute()
            
            playlist_id = playlist["id"]
            logger.info(f"Playlist created: {playlist_id}")
            return playlist_id
        
        except Exception as e:
            logger.error(f"Error creating playlist: {e}")
            return None
    
    def add_video_to_playlist(self, playlist_id: str, video_id: str) -> bool:
        """Add a video to a playlist."""
        if not self.youtube:
            logger.error("YouTube API not initialized")
            return False
        
        try:
            self.youtube.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {
                            "kind": "youtube#video",
                            "videoId": video_id
                        }
                    }
                }
            ).execute()
            
            logger.info(f"Video {video_id} added to playlist {playlist_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding video to playlist: {e}")
            return False
    
    def check_auth_status(self) -> bool:
        """Check if the API is properly authenticated."""
        if not self.youtube:
            return False
        
        try:
            # Try to list a channel to verify auth
            response = self.youtube.channels().list(
                part="snippet",
                mine=True
            ).execute()
            
            return "items" in response and len(response["items"]) > 0
        
        except Exception:
            return False
    
    def revoke_credentials(self) -> bool:
        """Revoke the API credentials."""
        if os.path.exists(self.token_file):
            try:
                os.remove(self.token_file)
                logger.info("YouTube API credentials revoked")
                self.youtube = None
                return True
            except Exception as e:
                logger.error(f"Error revoking credentials: {e}")
                return False
        
        return True


# Helper functions for setting up OAuth credentials

def generate_client_secrets_file(client_id: str, client_secret: str, output_path: str = "config/client_secrets.json") -> bool:
    """
    Generate a client secrets file for YouTube API OAuth.
    
    Args:
        client_id: OAuth client ID
        client_secret: OAuth client secret
        output_path: Path to save the client secrets file
        
    Returns:
        True if successful, False otherwise
    """
    client_secrets = {
        "installed": {
            "client_id": client_id,
            "project_id": "ai-video-generator",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": client_secret,
            "redirect_uris": ["http://localhost", "urn:ietf:wg:oauth:2.0:oob"]
        }
    }
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(client_secrets, f, indent=2)
        
        logger.info(f"Client secrets file created at {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating client secrets file: {e}")
        return False
