#!/usr/bin/env python3
# example_full_pipeline.py
# This script demonstrates a complete end-to-end video generation pipeline

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('example_pipeline')

# Add parent directory to path to make imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the pipeline and components
from core.pipeline import Pipeline
from core.script_generator import ScriptGenerator
from core.voice_generator import VoiceGenerator
from core.media_selector import MediaSelector
from data_providers.youtube_api import YouTubeAPI

def setup_environment():
    """Set up necessary directories and configuration files."""
    # Create required directories
    dirs = [
        "config",
        "output",
        "output/scripts",
        "output/audio",
        "output/media",
        "output/video",
        "output/projects",
        "assets",
        "assets/music",
        "assets/overlays",
        "cache",
        "cache/media"
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Check if we have API keys configured
    if not os.path.exists("config/api_keys.json"):
        logger.warning("API keys not configured. Please add your API keys to config/api_keys.json")
        sample_keys = {
            "openai": "YOUR_OPENAI_API_KEY",
            "elevenlabs": "YOUR_ELEVENLABS_API_KEY",
            "pexels": "YOUR_PEXELS_API_KEY",
            "pixabay": "YOUR_PIXABAY_API_KEY",
            "unsplash": "YOUR_UNSPLASH_API_KEY"
        }
        
        with open("config/api_keys.json", "w") as f:
            json.dump(sample_keys, f, indent=2)

def run_full_pipeline(topic, output_name=None):
    """Run a complete pipeline from script generation to YouTube upload."""
    pipeline = Pipeline()
    
    # Define options for different stages
    options = {
        "script_options": {
            "tone": "informative",
            "include_timestamps": True,
            "timestamp_interval": 30
        },
        "voice_options": {
            "speech_rate": 1.0,
            "enhance_voice": True
        },
        "media_options": {
            "prefer_video_over_image": True,
            "max_assets_per_segment": 3
        },
        "video_options": {
            "apply_ken_burns": True,
            "transition_duration": 0.8
        },
        "audio_options": {
            "add_music": True,
            "music_volume": 0.15,
            "normalize_audio": True
        },
        "youtube_options": {
            "privacy": "private",  # private, unlisted, public
            "category": "22",  # 22 = People & Blogs
            "generate_tags": True
        }
    }
    
    # Run the pipeline
    project_state = pipeline.generate_video(
        topic=topic,
        output_name=output_name,
        script_type="educational",
        duration=3,  # 3 minutes
        tone="informative",
        voice="adam",  # ElevenLabs voice
        keywords=["tutorial", "learning", "education"],
        **options
    )
    
    # Print the results
    if project_state["status"] == "completed":
        logger.info("Pipeline completed successfully!")
        
        # Print script info
        if project_state["stages"]["script_generation"]["status"] == "completed":
            script_info = project_state["stages"]["script_generation"]["output"]
            logger.info(f"Script generated: {script_info['script_text']}")
            logger.info(f"Word count: {script_info['word_count']}")
            logger.info(f"Estimated duration: {script_info['estimated_duration_seconds'] // 60}:{script_info['estimated_duration_seconds'] % 60:02d}")
        
        # Print video info
        if project_state["stages"]["final_export"]["status"] == "completed":
            video_info = project_state["stages"]["final_export"]["output"]
            logger.info(f"Final video: {video_info['output_path']}")
            logger.info(f"Duration: {video_info['duration']}s")
            logger.info(f"Resolution: {video_info['resolution']}")
        
        # Print YouTube info
        if project_state["stages"]["youtube_upload"]["status"] == "completed":
            youtube_info = project_state["stages"]["youtube_upload"]["output"]
            logger.info(f"YouTube video: {youtube_info['video_url']}")
            logger.info(f"Privacy: {youtube_info['privacy']}")
    
    elif project_state["status"] == "partially_completed":
        logger.warning("Pipeline partially completed.")
        
        # Print completed stages
        for stage, info in project_state["stages"].items():
            if info["status"] == "completed":
                logger.info(f"Stage {stage} completed")
            elif info["status"] == "failed":
                logger.error(f"Stage {stage} failed: {info.get('errors')}")
    
    else:
        logger.error(f"Pipeline failed with status: {project_state['status']}")
    
    return project_state

def demo_component_by_component(topic):
    """Run each component individually to demonstrate the step-by-step process."""
    logger.info("Starting component-by-component demonstration")
    
    # Step 1: Script Generation
    logger.info("Step 1: Script Generation")
    script_generator = ScriptGenerator()
    
    script_data = script_generator.generate_script(
        topic=topic,
        script_type="educational",
        duration=3,
        tone="informative",
        keywords=["tutorial", "learning", "education"]
    )
    
    script_path = os.path.join("output", "scripts", "example_script.txt")
    script_generator.export_to_file(script_data, script_path)
    
    logger.info(f"Script generated: {script_path}")
    logger.info(f"Word count: {script_data['word_count']}")
    
    # Step 2: Voice Generation
    logger.info("Step 2: Voice Generation")
    voice_generator = VoiceGenerator()
    
    audio_dir = os.path.join("output", "audio", "example")
    os.makedirs(audio_dir, exist_ok=True)
    
    audio_data = voice_generator.generate_audio_for_script(
        script_data=script_data,
        output_dir=audio_dir,
        voice="adam",
        combine=True
    )
    
    logger.info(f"Audio generated: {audio_data['full_audio']}")
    
    # Step 3: Media Selection
    logger.info("Step 3: Media Selection")
    media_selector = MediaSelector()
    
    media_dir = os.path.join("output", "media", "example")
    os.makedirs(media_dir, exist_ok=True)
    
    media_data = media_selector.select_media_for_script(
        script_data=script_data,
        output_dir=media_dir
    )
    
    total_videos = sum(len(segment["media"]["videos"]) for segment in media_data["segments"])
    total_images = sum(len(segment["media"]["images"]) for segment in media_data["segments"])
    
    logger.info(f"Media selected: {total_videos} videos, {total_images} images")
    
    # Steps 4-7: Use the pipeline for remaining steps
    logger.info("Steps 4-7: Using pipeline for remaining steps")
    
    # Save intermediate data to temp files for pipeline to use
    os.makedirs("output/temp", exist_ok=True)
    
    with open("output/temp/script_data.json", "w") as f:
        json.dump(script_data, f)
    
    with open("output/temp/audio_data.json", "w") as f:
        json.dump(audio_data, f)
    
    with open("output/temp/media_data.json", "w") as f:
        json.dump(media_data, f)
    
    # Initialize pipeline
    pipeline = Pipeline()
    
    # Create a partial project state
    project_state = {
        "project_id": "example_demo",
        "topic": topic,
        "output_name": "component_demo",
        "script_type": "educational",
        "duration": 3,
        "tone": "informative",
        "voice": "adam",
        "keywords": ["tutorial", "learning", "education"],
        "created_at": "2025-03-01 12:00:00",
        "project_dir": os.path.join("output", "projects", "component_demo"),
        "status": "running",
        "progress": 30,
        "stages": {
            "script_generation": {
                "status": "completed", 
                "output": {
                    "script_data": script_data,
                    "script_text": script_path,
                    "word_count": script_data["word_count"],
                    "estimated_duration_seconds": script_data["estimated_duration_seconds"]
                }
            },
            "voice_generation": {
                "status": "completed",
                "output": audio_data
            },
            "media_selection": {
                "status": "completed",
                "output": media_data
            },
            "subtitle_generation": {"status": "pending", "output": None},
            "video_assembly": {"status": "pending", "output": None},
            "audio_processing": {"status": "pending", "output": None},
            "final_export": {"status": "pending", "output": None},
            "youtube_upload": {"status": "pending", "output": None}
        }
    }
    
    # Make sure the project directory exists
    os.makedirs(project_state["project_dir"], exist_ok=True)
    
    # Save the project state
    with open(os.path.join(project_state["project_dir"], "project_state.json"), "w") as f:
        json.dump(project_state, f, indent=2)
    
    # Resume the pipeline from subtitle generation
    logger.info("Resuming pipeline from subtitle generation")
    result = pipeline.resume_project(
        project_dir=project_state["project_dir"],
        start_from_stage="subtitle_generation"
    )
    
    # Print results
    if result["status"] == "completed" or result["status"] == "partially_completed":
        # Print video info if available
        if result["stages"]["final_export"]["status"] == "completed":
            video_info = result["stages"]["final_export"]["output"]
            logger.info(f"Final video: {video_info['output_path']}")
            logger.info(f"Duration: {video_info['duration']}s")
            logger.info(f"Resolution: {video_info['resolution']}")
        
        # Check which stages completed
        for stage, info in result["stages"].items():
            if info["status"] == "completed":
                logger.info(f"Stage {stage} completed")
            elif info["status"] == "failed":
                logger.error(f"Stage {stage} failed: {info.get('errors')}")
    else:
        logger.error(f"Pipeline failed with status: {result['status']}")
    
    return result

def test_youtube_integration():
    """Test YouTube API integration."""
    logger.info("Testing YouTube API integration")
    
    # Initialize the YouTube API
    youtube_api = YouTubeAPI()
    
    # Check if authenticated
    if youtube_api.check_auth_status():
        logger.info("YouTube API authenticated successfully")
        
        # Get channel info
        channel_info = youtube_api.get_channel_info()
        if channel_info:
            logger.info(f"Channel: {channel_info['title']}")
            logger.info(f"Subscribers: {channel_info['subscribers']}")
            logger.info(f"Videos: {channel_info['videos']}")
        
        # List videos
        videos = youtube_api.list_videos(max_results=5)
        logger.info(f"Found {len(videos)} videos")
        for video in videos[:3]:  # Show the first 3
            logger.info(f"- {video['title']} ({video['views']} views)")
        
        return True
    else:
        logger.error("YouTube API authentication failed")
        logger.info("Please follow the setup instructions to authenticate with YouTube")
        return False

def main():
    """Main function to run the examples."""
    setup_environment()
    
    print("\nAI Video Generator - Complete Pipeline Example\n")
    print("Choose an option:")
    print("1. Run full pipeline (script to video)")
    print("2. Demo component-by-component")
    print("3. Test YouTube API integration")
    print("q. Quit")
    
    choice = input("\nEnter your choice (1-3, q): ")
    
    if choice == '1':
        topic = input("Enter a topic for your video: ")
        if not topic:
            topic = "The Fascinating World of Artificial Intelligence"
        
        print(f"\nGenerating video about: {topic}")
        run_full_pipeline(topic)
    
    elif choice == '2':
        topic = input("Enter a topic for your video: ")
        if not topic:
            topic = "How to Learn Programming: A Beginner's Guide"
        
        print(f"\nDemonstrating components for topic: {topic}")
        demo_component_by_component(topic)
    
    elif choice == '3':
        print("\nTesting YouTube API integration")
        test_youtube_integration()
    
    elif choice.lower() == 'q':
        print("Exiting...")
        return
    
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
