#!/usr/bin/env python3
# demo.py

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

# Ensure imports work regardless of where the script is run from
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.script_generator import ScriptGenerator
from models.llm_wrapper import LLMProvider


def setup_environment():
    """Set up necessary directories and configuration files."""
    # Create required directories
    os.makedirs("config", exist_ok=True)
    os.makedirs("output/scripts", exist_ok=True)
    
    # Create default config if it doesn't exist
    if not os.path.exists("config/default_settings.json"):
        default_config = {
            "default_llm_provider": "openai",
            "default_llm_model": "gpt-4",
            "script_min_length": 300,
            "script_max_length": 1500,
            "tone": "informative",
            "include_timestamps": True,
            "timestamp_interval": 30,  # seconds
            "script_structure": ["intro", "main_points", "conclusion"]
        }
        
        with open("config/default_settings.json", "w") as f:
            json.dump(default_config, f, indent=2)
    
    # Create API keys template if it doesn't exist
    if not os.path.exists("config/api_keys.json"):
        api_keys_template = {
            "openai": "YOUR_OPENAI_API_KEY",
            "anthropic": "YOUR_ANTHROPIC_API_KEY",
            "huggingface": "YOUR_HUGGINGFACE_API_KEY"
        }
        
        with open("config/api_keys.json", "w") as f:
            json.dump(api_keys_template, f, indent=2)


def generate_script(args):
    """Generate a script based on the provided topic."""
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


def main():
    """Main function for the demo script."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="AI Video Generator Demo")
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
    
    args = parser.parse_args()
    
    # Set up environment
    setup_environment()
    
    # Generate a script
    generate_script(args)


if __name__ == "__main__":
    main()
