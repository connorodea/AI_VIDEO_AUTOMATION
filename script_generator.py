# core/script_generator.py

import json
import os
import time
from typing import Dict, List, Optional, Tuple, Union

from models.llm_wrapper import LLMProvider


class ScriptGenerator:
    """
    Generates video scripts from topics or keywords using AI language models.
    """

    def __init__(self, config_path: str = "config/default_settings.json"):
        """
        Initialize the ScriptGenerator with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.llm_provider = LLMProvider(
            provider=self.config.get("default_llm_provider", "openai"),
            model=self.config.get("default_llm_model", "gpt-4")
        )
        
        # Load prompt templates
        self.prompt_templates = self._load_prompt_templates()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using default settings.")
            return {
                "default_llm_provider": "openai",
                "default_llm_model": "gpt-4",
                "script_min_length": 300,
                "script_max_length": 1500,
                "tone": "informative",
                "include_timestamps": True,
                "timestamp_interval": 30,  # seconds
                "script_structure": ["intro", "main_points", "conclusion"],
                "prompt_templates_path": "utils/prompt_templates.py"
            }
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates for different script types."""
        return {
            "standard": """
                Create a YouTube script about {topic}. 
                The script should be engaging, informative, and well-structured.
                The target video length is {duration} minutes.
                Tone: {tone}
                
                Include the following sections:
                - A captivating introduction that hooks the viewer
                - Clear, well-organized main points
                - A compelling conclusion with a call to action
                
                {additional_instructions}
                
                Format the script with proper narration instructions and timestamps every {timestamp_interval} seconds.
                Example format:
                
                [0:00] INTRO
                Hey viewers, welcome to this video about [topic]...
                
                [0:30] FIRST POINT
                Let's dive into the first aspect of [topic]...
            """,
            
            "educational": """
                Create an educational YouTube script about {topic}.
                The script should be clear, structured, and present information in a logical sequence.
                The target video length is {duration} minutes.
                Tone: {tone}
                
                Include the following sections:
                - An introduction explaining why this topic matters
                - Step-by-step explanation of key concepts
                - Practical examples or applications
                - A summary of main takeaways
                
                {additional_instructions}
                
                Format the script with timestamps every {timestamp_interval} seconds.
            """,
            
            "entertainment": """
                Create an entertaining YouTube script about {topic}.
                The script should be engaging, fun, and maintain viewer interest throughout.
                The target video length is {duration} minutes.
                Tone: {tone}
                
                Include:
                - A hook that immediately grabs attention
                - Interesting stories or facts about the topic
                - Engaging transitions between segments
                - A memorable ending
                
                {additional_instructions}
                
                Format with timestamps every {timestamp_interval} seconds.
            """
        }
    
    def generate_script(
        self, 
        topic: str, 
        script_type: str = "standard", 
        duration: int = 5,
        tone: Optional[str] = None,
        include_timestamps: Optional[bool] = None,
        timestamp_interval: Optional[int] = None,
        additional_instructions: str = "",
        keywords: List[str] = None
    ) -> Dict:
        """
        Generate a complete video script based on the provided topic.
        
        Args:
            topic: The main topic for the video
            script_type: Type of script (standard, educational, entertainment)
            duration: Target video duration in minutes
            tone: Tone of the script (informative, casual, professional, etc.)
            include_timestamps: Whether to include timestamps in the script
            timestamp_interval: Interval for timestamps in seconds
            additional_instructions: Any specific instructions to add to the prompt
            keywords: List of keywords to emphasize in the script
            
        Returns:
            Dict containing the generated script and metadata
        """
        # Use config defaults if not specified
        tone = tone or self.config.get("tone", "informative")
        include_timestamps = include_timestamps if include_timestamps is not None else self.config.get("include_timestamps", True)
        timestamp_interval = timestamp_interval or self.config.get("timestamp_interval", 30)
        
        # Prepare prompt
        prompt_template = self.prompt_templates.get(script_type, self.prompt_templates["standard"])
        
        # Add keywords to prompt if provided
        keyword_text = ""
        if keywords and len(keywords) > 0:
            keyword_text = f"Include the following keywords in the script: {', '.join(keywords)}"
        
        # Format the prompt
        prompt = prompt_template.format(
            topic=topic,
            duration=duration,
            tone=tone,
            timestamp_interval=timestamp_interval,
            additional_instructions=f"{additional_instructions}\n{keyword_text}".strip()
        )
        
        # Generate the script
        generated_content = self.llm_provider.generate(prompt)
        
        # Process the response (extract timestamps, segments, etc.)
        script_data = self._process_script(generated_content, include_timestamps)
        
        # Add metadata
        script_data["metadata"] = {
            "topic": topic,
            "type": script_type,
            "target_duration": duration,
            "tone": tone,
            "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "keywords": keywords
        }
        
        return script_data
    
    def _process_script(self, raw_script: str, include_timestamps: bool) -> Dict:
        """
        Process the raw script to extract segments, timestamps, etc.
        
        Args:
            raw_script: The raw script text from the LLM
            include_timestamps: Whether timestamps were requested
            
        Returns:
            Dict with processed script data
        """
        result = {
            "raw_script": raw_script,
            "segments": []
        }
        
        if include_timestamps:
            # Extract timestamps and segments using a simple regex approach
            import re
            
            # Find timestamp patterns like [0:00] or [00:00]
            timestamp_pattern = r'\[(\d+:\d+)\](.*?)(?=\[\d+:\d+\]|$)'
            matches = re.findall(timestamp_pattern, raw_script, re.DOTALL)
            
            if matches:
                for timestamp, content in matches:
                    minutes, seconds = map(int, timestamp.split(':'))
                    time_in_seconds = minutes * 60 + seconds
                    
                    # Find a title/section name if present (often in ALL CAPS after timestamp)
                    title_match = re.match(r'\s*([A-Z][A-Z\s]+):(.*)', content, re.DOTALL)
                    
                    if title_match:
                        title, content = title_match.groups()
                    else:
                        title = f"Segment at {timestamp}"
                    
                    segment = {
                        "timestamp": timestamp,
                        "time_in_seconds": time_in_seconds,
                        "title": title.strip(),
                        "content": content.strip()
                    }
                    
                    result["segments"].append(segment)
            else:
                # If no timestamps found, treat as a single segment
                result["segments"].append({
                    "timestamp": "0:00",
                    "time_in_seconds": 0,
                    "title": "Full Script",
                    "content": raw_script.strip()
                })
        else:
            # If timestamps weren't requested, treat as a single segment
            result["segments"].append({
                "timestamp": "0:00",
                "time_in_seconds": 0,
                "title": "Full Script",
                "content": raw_script.strip()
            })
        
        # Calculate word count for timing estimates
        result["word_count"] = len(raw_script.split())
        
        # Estimate duration (assuming 150 words per minute)
        result["estimated_duration_seconds"] = int(result["word_count"] / 150 * 60)
        
        return result
    
    def regenerate_segment(
        self, 
        script_data: Dict,
        segment_index: int,
        instructions: str
    ) -> Dict:
        """
        Regenerate a specific segment of the script with new instructions.
        
        Args:
            script_data: The original script data
            segment_index: Index of the segment to regenerate
            instructions: Specific instructions for the regeneration
            
        Returns:
            Updated script data
        """
        if segment_index >= len(script_data["segments"]):
            raise IndexError(f"Segment index {segment_index} is out of range")
        
        segment = script_data["segments"][segment_index]
        
        prompt = f"""
        Rewrite the following segment of a YouTube script according to these instructions:
        {instructions}
        
        ORIGINAL SEGMENT:
        {segment['content']}
        
        Keep the same timestamp [{segment['timestamp']}] and maintain consistency with the rest of the script.
        """
        
        # Generate new content for the segment
        new_content = self.llm_provider.generate(prompt)
        
        # Update the segment
        script_data["segments"][segment_index]["content"] = new_content.strip()
        
        # Recalculate word count and duration estimate
        raw_script = "\n\n".join([f"[{s['timestamp']}] {s['title']}\n{s['content']}" for s in script_data["segments"]])
        script_data["raw_script"] = raw_script
        script_data["word_count"] = len(raw_script.split())
        script_data["estimated_duration_seconds"] = int(script_data["word_count"] / 150 * 60)
        
        return script_data
    
    def export_to_file(self, script_data: Dict, output_path: str) -> str:
        """
        Export the script to a file.
        
        Args:
            script_data: The script data to export
            output_path: Path to save the script
            
        Returns:
            The path to the saved file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            # Write metadata
            f.write(f"# Script: {script_data['metadata']['topic']}\n")
            f.write(f"# Type: {script_data['metadata']['type']}\n")
            f.write(f"# Generated: {script_data['metadata']['generation_time']}\n")
            f.write(f"# Estimated Duration: {script_data['estimated_duration_seconds'] // 60}:{script_data['estimated_duration_seconds'] % 60:02d}\n")
            f.write(f"# Word Count: {script_data['word_count']}\n\n")
            
            # Write segments
            for segment in script_data["segments"]:
                f.write(f"[{segment['timestamp']}] {segment['title']}\n")
                f.write(f"{segment['content']}\n\n")
        
        return output_path
