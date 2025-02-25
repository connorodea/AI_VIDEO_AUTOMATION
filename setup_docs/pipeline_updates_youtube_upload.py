# This script contains the updates needed to integrate YouTube uploads into the pipeline.py file

# 1. Add a new function to handle YouTube uploads in the Pipeline class

def _handle_youtube_upload(self, project_state: Dict, **kwargs) -> Optional[Dict]:
    """Handle YouTube video upload stage."""
    logger.info("Uploading video to YouTube")
    
    # Check if video assembly or final export was successful
    if (project_state["stages"]["final_export"]["status"] != "completed" and 
        project_state["stages"]["video_assembly"]["status"] != "completed"):
        logger.error("Cannot upload video without completed video assembly or final export")
        return None
    
    # Get video path
    video_path = None
    if project_state["stages"]["final_export"]["status"] == "completed":
        video_path = project_state["stages"]["final_export"]["output"]["output_path"]
    else:
        video_path = project_state["stages"]["video_assembly"]["output"]["output_path"]
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None
    
    # Get script data
    script_data = project_state["stages"]["script_generation"]["output"]["script_data"]
    
    # Extract YouTube options from kwargs
    youtube_options = kwargs.get("youtube_options", {})
    
    # Set default title and privacy
    title = youtube_options.get("title", script_data["metadata"]["topic"])
    privacy = youtube_options.get("privacy", self.config.get("youtube", {}).get("privacy", "private"))
    
    # Generate description and tags if not provided
    description = youtube_options.get("description")
    tags = youtube_options.get("tags")
    
    try:
        # Import the YouTubeAPI class
        from data_providers.youtube_api import YouTubeAPI
        
        # Initialize the YouTube API
        youtube_api = YouTubeAPI()
        
        # Check if API is authenticated
        if not youtube_api.check_auth_status():
            logger.error("YouTube API not authenticated. Please set up OAuth credentials.")
            return None
        
        # Upload the video
        success, video_id, video_url = youtube_api.upload_video(
            video_path=video_path,
            title=title,
            description=description,
            tags=tags,
            privacy=privacy,
            script_data=script_data,
            **youtube_options
        )
        
        if success:
            logger.info(f"Video uploaded successfully: {video_url}")
            
            # Add to a playlist if specified
            playlist_id = youtube_options.get("playlist_id")
            if playlist_id:
                youtube_api.add_video_to_playlist(playlist_id, video_id)
            
            return {
                "video_id": video_id,
                "video_url": video_url,
                "privacy": privacy,
                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            logger.error("Failed to upload video to YouTube")
            return None
    
    except ImportError:
        logger.error("YouTubeAPI module not found. Make sure the YouTube API dependencies are installed.")
        return None
    
    except Exception as e:
        logger.error(f"Error uploading video to YouTube: {e}")
        return None


# 2. Add the new stage to the default_stage_sequence in the Pipeline class __init__ method:

self.default_stage_sequence = [
    "script_generation",
    "voice_generation",
    "media_selection",
    "subtitle_generation",
    "video_assembly",
    "audio_processing",
    "final_export",
    "youtube_upload"  # Add this new stage
]


# 3. Add the stage handler to the stage_handlers dictionary in the Pipeline class __init__ method:

self.stage_handlers = {
    "script_generation": self._handle_script_generation,
    "voice_generation": self._handle_voice_generation,
    "media_selection": self._handle_media_selection,
    "subtitle_generation": self._handle_subtitle_generation,
    "video_assembly": self._handle_video_assembly,
    "audio_processing": self._handle_audio_processing,
    "final_export": self._handle_final_export,
    "youtube_upload": self._handle_youtube_upload  # Add this new handler
}


# 4. Add the new stage to the dependencies dictionary in the generate_video method:

"dependencies": {
    "script_generation": [],
    "voice_generation": ["script_generation"],
    "media_selection": ["script_generation"],
    "subtitle_generation": ["script_generation", "voice_generation"],
    "video_assembly": ["script_generation", "media_selection"],
    "audio_processing": ["voice_generation"],
    "final_export": ["video_assembly", "audio_processing", "subtitle_generation"],
    "youtube_upload": ["final_export"]  # Add this new dependency
}


# 5. Initialize an empty stage output in the project_state dictionary in the generate_video method:

"stages": {
    "script_generation": {"status": "pending", "output": None, "errors": None},
    "voice_generation": {"status": "pending", "output": None, "errors": None},
    "media_selection": {"status": "pending", "output": None, "errors": None},
    "subtitle_generation": {"status": "pending", "output": None, "errors": None},
    "video_assembly": {"status": "pending", "output": None, "errors": None},
    "audio_processing": {"status": "pending", "output": None, "errors": None},
    "final_export": {"status": "pending", "output": None, "errors": None},
    "youtube_upload": {"status": "pending", "output": None, "errors": None}  # Add this new stage
}
