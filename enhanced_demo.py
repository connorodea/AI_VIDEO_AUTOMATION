#!/usr/bin/env python3
# enhanced_demo.py

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Ensure imports work regardless of where the script is run from
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from core.pipeline import Pipeline
from core.script_generator import ScriptGenerator
from core.voice_generator import VoiceGenerator
from core.media_selector import MediaSelector


def setup_environment():
    """Set up necessary directories and configuration files."""
    # Create required directories
    os.makedirs("config", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("cache/media", exist_ok=True)
    
    # Create default config if it doesn't exist
    if not os.path.exists("config/default_settings.json"):
        # We'll use a simpler version for the demo
        default_config = {
            "default_llm_provider": "openai",
            "default_llm_model": "gpt-4",
            
            "script": {
                "min_length": 300,
                "max_length": 1500,
                "tone": "informative",
                "include_timestamps": True,
                "timestamp_interval": 30
            },
            
            "voiceover": {
                "provider": "elevenlabs",
                "default_voice": "adam",
                "speech_rate": 1.0,
                "stability": 0.5,
                "clarity": 0.75
            },
            
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
            },
            
            "output_dir": "output"
        }
        
        with open("config/default_settings.json", "w") as f:
            json.dump(default_config, f, indent=2)
    
    # Create API keys template if it doesn't exist
    if not os.path.exists("config/api_keys.json"):
        api_keys_template = {
            "openai": "YOUR_OPENAI_API_KEY",
            "anthropic": "YOUR_ANTHROPIC_API_KEY",
            "elevenlabs": "YOUR_ELEVENLABS_API_KEY",
            "pexels": "YOUR_PEXELS_API_KEY",
            "pixabay": "YOUR_PIXABAY_API_KEY"
        }
        
        with open("config/api_keys.json", "w") as f:
            json.dump(api_keys_template, f, indent=2)


def demo_script_generation(args):
    """Demo the script generation functionality."""
    print("\n===== SCRIPT GENERATION DEMO =====")
    
    # Initialize the script generator
    script_generator = ScriptGenerator()
    
    # Generate the script
    print(f"Generating script about: {args.topic}")
    script_data = script_generator.generate_script(
        topic=args.topic,
        script_type=args.type,
        duration=args.duration,
        tone=args.tone,
        keywords=args.keywords.split(",") if args.keywords else None
    )
    
    # Create a filename based on the topic
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_topic = "".join(c if c.isalnum() or c in " _-" else "_" for c in args.topic)
    sanitized_topic = sanitized_topic.replace(" ", "_")
    filename = f"{timestamp}_{sanitized_topic[:50]}.txt"
    output_path = os.path.join("output/scripts", filename)
    
    # Export the script to a file
    saved_path = script_generator.export_to_file(script_data, output_path)
    print(f"Script saved to: {saved_path}")
    
    # Print a summary
    print("\nScript Summary:")
    print(f"Word count: {script_data['word_count']}")
    print(f"Estimated duration: {script_data['estimated_duration_seconds'] // 60}:{script_data['estimated_duration_seconds'] % 60:02d}")
    print(f"Number of segments: {len(script_data['segments'])}")
    
    # Print the first segment as a preview
    if script_data["segments"]:
        first_segment = script_data["segments"][0]
        print("\nPreview (first segment):")
        print(f"[{first_segment['timestamp']}] {first_segment['title']}")
        print(first_segment['content'][:150] + "..." if len(first_segment['content']) > 150 else first_segment['content'])
    
    return script_data, saved_path


def demo_voice_generation(script_data, args):
    """Demo the voice generation functionality."""
    if not args.voice:
        print("\n===== SKIPPING VOICE GENERATION (no voice specified) =====")
        print("To generate voice, use --voice parameter.")
        return None
    
    print("\n===== VOICE GENERATION DEMO =====")
    
    # Initialize the voice generator
    voice_generator = VoiceGenerator()
    
    # Get available voices
    available_voices = voice_generator.get_available_voices()
    if available_voices:
        print(f"Available voices: {', '.join([v.get('name', v.get('id', 'Unknown')) for v in available_voices[:5]])}...")
    
    # Create an output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output/audio", f"{timestamp}_demo")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate audio
    print(f"Generating audio with voice: {args.voice}")
    
    audio_data = voice_generator.generate_audio_for_script(
        script_data=script_data,
        output_dir=output_dir,
        voice=args.voice,
        combine=True
    )
    
    print(f"Generated {len(audio_data['segments'])} audio segments.")
    
    if audio_data.get('full_audio'):
        print(f"Full audio saved to: {audio_data['full_audio']}")
    
    return audio_data


def demo_media_selection(script_data, args):
    """Demo the media selection functionality."""
    if not args.media:
        print("\n===== SKIPPING MEDIA SELECTION (--media flag not set) =====")
        print("To select media, use --media flag.")
        return None
    
    print("\n===== MEDIA SELECTION DEMO =====")
    
    # Initialize the media selector
    media_selector = MediaSelector()
    
    # Create an output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output/media", f"{timestamp}_demo")
    os.makedirs(output_dir, exist_ok=True)
    
    # Select media
    print("Selecting media for script segments...")
    
    media_data = media_selector.select_media_for_script(
        script_data=script_data,
        output_dir=output_dir
    )
    
    # Print summary
    print("\nMedia Selection Summary:")
    total_videos = sum(len(segment["media"]["videos"]) for segment in media_data["segments"])
    total_images = sum(len(segment["media"]["images"]) for segment in media_data["segments"])
    total_generated = sum(len(segment["media"]["generated_images"]) for segment in media_data["segments"])
    
    print(f"Selected {total_videos} videos, {total_images} images, and {total_generated} generated images.")
    print(f"Media files saved to: {output_dir}")
    
    return media_data


def demo_full_pipeline(args):
    """Demo the full pipeline functionality."""
    if not args.pipeline:
        print("\n===== SKIPPING FULL PIPELINE (--pipeline flag not set) =====")
        print("To run the full pipeline, use --pipeline flag.")
        return None
    
    print("\n===== FULL PIPELINE DEMO =====")
    
    # Initialize the pipeline
    pipeline = Pipeline()
    
    # Run the pipeline
    print(f"Running full pipeline for: {args.topic}")
    
    try:
        project_state = pipeline.generate_video(
            topic=args.topic,
            script_type=args.type,
            duration=args.duration,
            tone=args.tone,
            voice=args.voice,
            keywords=args.keywords.split(",") if args.keywords else None
        )
        
        print("\nPipeline Execution Summary:")
        print(f"Project ID: {project_state['project_id']}")
        print(f"Project directory: {project_state['project_dir']}")
        print(f"Project status: {project_state['status']}")
        
        # Print stage statuses
        print("\nStage statuses:")
        for stage, info in project_state["stages"].items():
            print(f"  {stage}: {info['status']}")
        
        return project_state
    
    except NotImplementedError as e:
        print(f"\nNote: {e}")
        print("Some components of the pipeline are not yet implemented. Only the available components were executed.")
        return None
    
    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        return None


def main():
    """Main function for the enhanced demo script."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="AI Video Generator Enhanced Demo")
    parser.add_argument("topic", help="Topic for the video script")
    parser.add_argument(
        "--type", 
        choices=["standard", "educational", "entertainment"], 
        default="standard",
        help="Type of script to generate"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=5,
        help="Target video duration in minutes"
    )
    parser.add_argument(
        "--tone", 
        default="informative",
        help="Tone of the script (informative, casual, professional, etc.)"
    )
    parser.add_argument(
        "--keywords", 
        help="Comma-separated list of keywords to include in the script"
    )
    parser.add_argument(
        "--voice",
        help="Voice to use for audio generation"
    )
    parser.add_argument(
        "--media",
        action="store_true",
        help="Enable media selection"
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run the full pipeline"
    )
    
    args = parser.parse_args()
    
    # Set up environment
    setup_environment()
    
    # Demo each component
    script_data, _ = demo_script_generation(args)
    audio_data = demo_voice_generation(script_data, args)
    media_data = demo_media_selection(script_data, args)
    project_state = demo_full_pipeline(args)
    
    print("\n===== DEMO COMPLETED =====")
    print("Check the 'output' directory for generated files.")


if __name__ == "__main__":
    main()
