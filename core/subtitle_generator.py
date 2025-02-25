# core/subtitle_generator.py

import json
import os
import subprocess
import tempfile
import shutil
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('subtitle_generator')

class SubtitleGenerator:
    """
    Generates subtitles for videos based on script content and audio timing.
    """

    def __init__(self, config_path: str = "config/default_settings.json"):
        """
        Initialize the SubtitleGenerator with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.subtitle_config = self.config.get("subtitle", {})
        
        # Subtitle settings
        self.enabled = self.subtitle_config.get("enabled", True)
        self.font = self.subtitle_config.get("font", "Roboto")
        self.font_size = self.subtitle_config.get("font_size", 30)
        self.position = self.subtitle_config.get("position", "bottom")
        self.background_opacity = self.subtitle_config.get("background_opacity", 0.5)
        self.max_chars_per_line = self.subtitle_config.get("max_chars_per_line", 42)
        
        # Check if necessary tools are available
        self._check_dependencies()
        
        # Initialize temp directory for intermediate files
        self.temp_dir = None

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
            return {
                "subtitle": {
                    "enabled": True,
                    "font": "Roboto",
                    "font_size": 30,
                    "position": "bottom",
                    "background_opacity": 0.5,
                    "max_chars_per_line": 42
                }
            }
    
    def _check_dependencies(self) -> None:
        """Check if necessary tools for subtitle generation are available."""
        try:
            # Check for FFmpeg
            result = subprocess.run(['ffmpeg', '-version'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True, 
                                   check=False)
            if result.returncode != 0:
                logger.warning("FFmpeg is not available. Some subtitle features may not work.")
            
            # Check for Whisper (optional)
            try:
                import whisper
                self.whisper_available = True
                logger.info("Whisper is available for speech-to-text")
            except ImportError:
                self.whisper_available = False
                logger.info("Whisper is not available. Manual transcription will be used.")
        except FileNotFoundError:
            logger.warning("FFmpeg is not available. Some subtitle features may not work.")
    
    def generate_subtitles_for_script(
        self,
        script_data: Dict,
        output_dir: str,
        audio_data: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Generate subtitles for a script.
        
        Args:
            script_data: Script data from ScriptGenerator
            output_dir: Directory to save the output subtitles
            audio_data: Optional audio data from VoiceGenerator for timing
            **kwargs: Additional options
            
        Returns:
            Dictionary with subtitle information
        """
        logger.info("Generating subtitles")
        
        if not self.enabled:
            logger.info("Subtitles are disabled in config")
            return {"enabled": False}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a temporary directory for intermediate files
        self.temp_dir = tempfile.mkdtemp()
        logger.debug("Created temporary directory: %s", self.temp_dir)
        
        try:
            # Generate SRT subtitles
            if audio_data and self.whisper_available:
                # Use Whisper for speech-to-text if audio is available
                subtitles_file = self._generate_subtitles_from_audio(audio_data, output_dir)
            else:
                # Generate subtitles from script if no audio or Whisper
                subtitles_file = self._generate_subtitles_from_script(script_data, audio_data, output_dir)
            
            # Also generate VTT format (for web video)
            vtt_file = self._convert_srt_to_vtt(subtitles_file, output_dir)
            
            # Generate styled subtitles (for burning into video)
            styled_file = self._create_styled_subtitles(subtitles_file, output_dir)
            
            return {
                "enabled": True,
                "subtitles_file": subtitles_file,
                "vtt_file": vtt_file,
                "styled_file": styled_file,
                "format": "srt"
            }
        
        except Exception as e:
            logger.error("Error generating subtitles: %s", str(e), exc_info=True)
            return {"enabled": False, "error": str(e)}
        
        finally:
            self._cleanup()
    
    def _generate_subtitles_from_script(
        self, 
        script_data: Dict, 
        audio_data: Optional[Dict], 
        output_dir: str
    ) -> str:
        """
        Generate subtitles based on script content.
        
        Args:
            script_data: Script data
            audio_data: Audio data (if available)
            output_dir: Output directory
            
        Returns:
            Path to the generated SRT file
        """
        logger.info("Generating subtitles from script")
        
        subtitles_file = os.path.join(output_dir, "subtitles.srt")
        
        with open(subtitles_file, 'w') as f:
            subtitle_index = 1
            current_time = 0
            
            for segment in script_data.get("segments", []):
                content = segment["content"]
                
                # Estimate duration based on word count if audio data not available
                if audio_data:
                    # Try to find corresponding audio segment
                    segment_audio = None
                    for audio_segment in audio_data.get("segments", []):
                        if audio_segment.get("segment_index") == segment.get("segment_index"):
                            segment_audio = audio_segment
                            break
                    
                    if segment_audio and os.path.exists(segment_audio["audio_path"]):
                        # Get duration from audio file
                        duration = self._get_audio_duration(segment_audio["audio_path"])
                    else:
                        # Fallback to estimation
                        words = len(content.split())
                        duration = words / 2.5  # Assuming 2.5 words per second
                else:
                    # Estimate duration from word count
                    words = len(content.split())
                    duration = words / 2.5  # Assuming 2.5 words per second
                
                # Split content into subtitle chunks
                chunks = self._split_into_subtitle_chunks(content)
                
                # Calculate time per chunk
                time_per_chunk = duration / len(chunks) if chunks else 0
                
                # Write each chunk as a subtitle
                chunk_time = current_time
                for chunk in chunks:
                    start_time = self._format_time(chunk_time)
                    chunk_time += time_per_chunk
                    end_time = self._format_time(chunk_time)
                    
                    # Write the subtitle entry
                    f.write(f"{subtitle_index}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{chunk}\n\n")
                    
                    subtitle_index += 1
                
                current_time += duration
        
        return subtitles_file
    
    def _generate_subtitles_from_audio(self, audio_data: Dict, output_dir: str) -> str:
        """
        Generate subtitles using Whisper speech-to-text on audio.
        
        Args:
            audio_data: Audio data
            output_dir: Output directory
            
        Returns:
            Path to the generated SRT file
        """
        logger.info("Generating subtitles from audio using Whisper")
        
        # Import Whisper
        import whisper
        
        subtitles_file = os.path.join(output_dir, "subtitles.srt")
        
        # Load Whisper model
        model = whisper.load_model("base")
        
        # Process each audio segment or the full audio if available
        if audio_data.get("full_audio") and os.path.exists(audio_data["full_audio"]):
            # Transcribe the full audio
            logger.info("Transcribing full audio")
            result = model.transcribe(audio_data["full_audio"])
            
            # Write the segments to SRT format
            with open(subtitles_file, 'w') as f:
                for i, segment in enumerate(result["segments"]):
                    start_time = self._format_time(segment["start"])
                    end_time = self._format_time(segment["end"])
                    text = segment["text"].strip()
                    
                    f.write(f"{i+1}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
        
        elif audio_data.get("segments"):
            # Process individual segments
            all_segments = []
            
            for i, segment in enumerate(audio_data["segments"]):
                if os.path.exists(segment["audio_path"]):
                    logger.info("Transcribing segment %d", i)
                    result = model.transcribe(segment["audio_path"])
                    
                    # Adjust timestamps to account for segment position
                    start_offset = segment.get("time_in_seconds", 0)
                    
                    for s in result["segments"]:
                        s["start"] += start_offset
                        s["end"] += start_offset
                        all_segments.append(s)
            
            # Sort segments by start time
            all_segments.sort(key=lambda x: x["start"])
            
            # Write the segments to SRT format
            with open(subtitles_file, 'w') as f:
                for i, segment in enumerate(all_segments):
                    start_time = self._format_time(segment["start"])
                    end_time = self._format_time(segment["end"])
                    text = segment["text"].strip()
                    
                    f.write(f"{i+1}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
        
        else:
            logger.warning("No audio data found for transcription")
            # Fall back to script-based subtitles
            return self._generate_subtitles_from_script({}, audio_data, output_dir)
        
        return subtitles_file
    
    def _split_into_subtitle_chunks(self, text: str) -> List[str]:
        """
        Split text into subtitle-sized chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of subtitle chunks
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # If text is short enough, return it as is
        if len(text) <= self.max_chars_per_line:
            return [text]
        
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        for sentence in sentences:
            # If the sentence is too long, break it up
            if len(sentence) > self.max_chars_per_line:
                # If we have something in the current chunk, add it
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Break the long sentence into chunks
                words = sentence.split()
                sentence_chunk = ""
                
                for word in words:
                    if len(sentence_chunk) + len(word) + 1 <= self.max_chars_per_line:
                        sentence_chunk += " " + word if sentence_chunk else word
                    else:
                        chunks.append(sentence_chunk)
                        sentence_chunk = word
                
                if sentence_chunk:
                    current_chunk = sentence_chunk
            
            # Otherwise, try to add the sentence to the current chunk
            elif len(current_chunk) + len(sentence) + 1 <= self.max_chars_per_line:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                # Current chunk is full, start a new one
                chunks.append(current_chunk)
                current_chunk = sentence
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _convert_srt_to_vtt(self, srt_file: str, output_dir: str) -> str:
        """
        Convert SRT to VTT format.
        
        Args:
            srt_file: Path to the SRT file
            output_dir: Output directory
            
        Returns:
            Path to the VTT file
        """
        vtt_file = os.path.join(output_dir, "subtitles.vtt")
        
        with open(srt_file, 'r') as f_in, open(vtt_file, 'w') as f_out:
            # Write VTT header
            f_out.write("WEBVTT\n\n")
            
            content = f_in.read()
            
            # Replace SRT timestamp format with VTT format
            # SRT: 00:00:10,500 --> 00:00:13,000
            # VTT: 00:00:10.500 --> 00:00:13.000
            content = re.sub(r'(\d\d:\d\d:\d\d),(\d\d\d)', r'\1.\2', content)
            
            f_out.write(content)
        
        return vtt_file
    
    def _create_styled_subtitles(self, srt_file: str, output_dir: str) -> str:
        """
        Create styled subtitle file for FFmpeg.
        
        Args:
            srt_file: Path to the SRT file
            output_dir: Output directory
            
        Returns:
            Path to the styled subtitles file
        """
        styled_file = os.path.join(output_dir, "styled_subtitles.ass")
        
        # Convert SRT to ASS (Advanced SubStation Alpha) format with styling
        cmd = [
            "ffmpeg",
            "-y",
            "-i", srt_file,
            styled_file
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            # Now modify the ASS file to add our custom styling
            with open(styled_file, 'r') as f:
                content = f.read()
            
            # Add custom styling based on our configuration
            style_section = f"[V4+ Styles]\n"
            style_section += f"Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            style_section += f"Style: Default,{self.font},{self.font_size},&H00FFFFFF,&H000000FF,&H00000000,&H{int(self.background_opacity * 255):02X}000000,0,0,0,0,100,100,0,0,1,1,1,{2 if self.position.lower() == 'top' else 1},10,10,10,1\n\n"
            
            # Replace the style section in the ASS file
            content = re.sub(r'\[V4\+ Styles\].*?\[Events\]', style_section + "[Events]", content, flags=re.DOTALL)
            
            with open(styled_file, 'w') as f:
                f.write(content)
            
            return styled_file
        
        except subprocess.CalledProcessError as e:
            logger.error("Error creating styled subtitles: %s", e.stderr)
            return srt_file  # Return the original SRT as fallback
    
    def _format_time(self, seconds: float) -> str:
        """
        Format time in SRT format (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """
        Get the duration of an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Duration in seconds
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        
        if result.returncode != 0:
            logger.error("Error getting audio duration: %s", result.stderr)
            return 0.0
        
        try:
            return float(result.stdout.strip())
        except ValueError:
            logger.error("Error parsing audio duration")
            return 0.0
    
    def _cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.debug("Cleaned up temporary directory: %s", self.temp_dir)
            except Exception as e:
                logger.error("Error cleaning up temporary directory: %s", str(e))
            finally:
                self.temp_dir = None
