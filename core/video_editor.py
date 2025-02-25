# core/video_editor.py

import json
import os
import subprocess
import random
import tempfile
import shutil
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('video_editor')

class VideoEditor:
    """
    Video editing and composition module that assembles media assets into a complete video.
    Features include transitions, Ken Burns effects, text overlays, and more.
    """

    def __init__(self, config_path: str = "config/default_settings.json"):
        """
        Initialize the VideoEditor with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.video_config = self.config.get("video_editing", {})
        
        # Output settings
        self.output_resolution = self.video_config.get("output_resolution", {"width": 1920, "height": 1080})
        self.fps = self.video_config.get("fps", 30)
        self.codec = self.video_config.get("codec", "h264")
        self.bitrate = self.video_config.get("bitrate", "5M")
        
        # Transition settings
        self.transition_config = self.video_config.get("transitions", {})
        self.default_transition = self.transition_config.get("default", "fade")
        self.transition_duration = self.transition_config.get("duration", 0.8)
        self.random_transitions = self.transition_config.get("random_variation", False)
        
        # Ken Burns settings
        self.ken_burns_config = self.video_config.get("ken_burns", {})
        self.ken_burns_enabled = self.ken_burns_config.get("enabled", True)
        self.ken_burns_min_zoom = self.ken_burns_config.get("min_zoom", 1.0)
        self.ken_burns_max_zoom = self.ken_burns_config.get("max_zoom", 1.2)
        self.ken_burns_min_duration = self.ken_burns_config.get("min_duration", 5)
        self.ken_burns_max_duration = self.ken_burns_config.get("max_duration", 10)
        
        # Overlay settings
        self.overlay_config = self.video_config.get("overlay", {})
        self.logo_config = self.overlay_config.get("logo", {})
        self.grain_config = self.overlay_config.get("grain", {})
        self.text_style = self.overlay_config.get("text_style", {})
        
        # Check if FFmpeg is available
        self._check_ffmpeg()
        
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
                "video_editing": {
                    "output_resolution": {"width": 1920, "height": 1080},
                    "fps": 30,
                    "codec": "h264",
                    "bitrate": "5M",
                    "transitions": {
                        "default": "fade",
                        "duration": 0.8,
                        "random_variation": False
                    },
                    "ken_burns": {
                        "enabled": True,
                        "min_zoom": 1.0,
                        "max_zoom": 1.2,
                        "min_duration": 5,
                        "max_duration": 10
                    },
                    "overlay": {
                        "logo": {
                            "enabled": False,
                            "path": "assets/logo.png",
                            "position": "bottom-right",
                            "opacity": 0.8
                        },
                        "grain": {
                            "enabled": False,
                            "strength": 0.2
                        },
                        "text_style": {
                            "font": "Montserrat",
                            "color": "#FFFFFF",
                            "shadow": True
                        }
                    }
                }
            }
    
    def _check_ffmpeg(self):
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True, 
                                   check=False)
            if result.returncode != 0:
                logger.error("FFmpeg is not available. Please install FFmpeg to use the video editor.")
                raise RuntimeError("FFmpeg is not available")
            else:
                logger.info("FFmpeg is available")
        except FileNotFoundError:
            logger.error("FFmpeg is not available. Please install FFmpeg to use the video editor.")
            raise RuntimeError("FFmpeg is not available")
    
    def assemble_video(
        self,
        script_data: Dict,
        media_data: Dict,
        output_dir: str,
        output_name: str,
        voice_data: Optional[Dict] = None,
        subtitles_data: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Assemble a video from script, media assets, and audio.
        
        Args:
            script_data: Script data from ScriptGenerator
            media_data: Media data from MediaSelector
            output_dir: Directory to save the output video
            output_name: Name of the output video file
            voice_data: Optional voice data from VoiceGenerator
            subtitles_data: Optional subtitles data from SubtitleGenerator
            **kwargs: Additional options
            
        Returns:
            Dictionary with information about the assembled video
        """
        logger.info("Assembling video: %s", output_name)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a temporary directory for intermediate files
        self.temp_dir = tempfile.mkdtemp()
        logger.debug("Created temporary directory: %s", self.temp_dir)
        
        try:
            # 1. Create an edit decision list (EDL) - timeline of clips
            edl = self._create_edit_decision_list(script_data, media_data, voice_data)
            
            # 2. Process each media item (resize, apply effects, etc.)
            processed_media = self._process_media_items(edl)
            
            # 3. Generate the video assembly script (FFmpeg commands)
            if voice_data and voice_data.get("full_audio"):
                # Use the full audio file
                audio_file = voice_data["full_audio"]
            else:
                # Create a silent audio track
                audio_file = self._create_silent_audio(edl["total_duration"])
            
            # 4. Assemble the final video
            output_path = os.path.join(output_dir, f"{output_name}.mp4")
            self._assemble_final_video(processed_media, audio_file, output_path)
            
            # 5. Add subtitles if provided
            if subtitles_data and subtitles_data.get("subtitles_file"):
                output_path_with_subs = os.path.join(output_dir, f"{output_name}_with_subs.mp4")
                self._add_subtitles(output_path, subtitles_data["subtitles_file"], output_path_with_subs)
                output_path = output_path_with_subs
            
            # 6. Clean up temporary files
            self._cleanup()
            
            return {
                "output_path": output_path,
                "duration": edl["total_duration"],
                "resolution": f"{self.output_resolution['width']}x{self.output_resolution['height']}",
                "fps": self.fps
            }
        
        except Exception as e:
            logger.error("Error assembling video: %s", str(e), exc_info=True)
            self._cleanup()
            raise
    
    def create_final_video(
        self,
        draft_video_path: str,
        output_dir: str,
        output_name: str,
        enhanced_audio_path: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Create the final video by combining the draft video with enhanced audio.
        
        Args:
            draft_video_path: Path to the draft video
            output_dir: Directory to save the output video
            output_name: Name of the output video file
            enhanced_audio_path: Path to the enhanced audio file
            **kwargs: Additional options
            
        Returns:
            Dictionary with information about the final video
        """
        logger.info("Creating final video: %s", output_name)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if draft video exists
        if not os.path.exists(draft_video_path):
            raise FileNotFoundError(f"Draft video not found: {draft_video_path}")
        
        # Set output path
        output_path = os.path.join(output_dir, f"{output_name}.mp4")
        
        # If no enhanced audio provided, just copy the draft video
        if not enhanced_audio_path:
            logger.info("No enhanced audio provided, copying draft video")
            shutil.copy2(draft_video_path, output_path)
        else:
            # Check if enhanced audio exists
            if not os.path.exists(enhanced_audio_path):
                raise FileNotFoundError(f"Enhanced audio not found: {enhanced_audio_path}")
            
            # Replace audio in the draft video
            logger.info("Replacing audio in draft video")
            self._replace_audio(draft_video_path, enhanced_audio_path, output_path)
        
        # Get video information
        video_info = self._get_video_info(output_path)
        
        return {
            "output_path": output_path,
            "duration": video_info.get("duration", 0),
            "resolution": video_info.get("resolution", f"{self.output_resolution['width']}x{self.output_resolution['height']}"),
            "fps": video_info.get("fps", self.fps)
        }
    
    def _create_edit_decision_list(self, script_data: Dict, media_data: Dict, voice_data: Optional[Dict] = None) -> Dict:
        """
        Create an edit decision list (timeline) based on script and media data.
        
        Args:
            script_data: Script data
            media_data: Media data
            voice_data: Voice data
            
        Returns:
            Edit decision list with clip information
        """
        logger.debug("Creating edit decision list")
        
        edl = {
            "clips": [],
            "total_duration": 0
        }
        
        # If we have voice data, use segment durations from the audio files
        if voice_data and voice_data.get("segments"):
            # Build a clip for each script segment with its corresponding audio
            for i, segment_info in enumerate(voice_data["segments"]):
                # Get the corresponding media data
                segment_media = None
                for media_segment in media_data.get("segments", []):
                    if media_segment.get("segment_index") == segment_info.get("segment_index"):
                        segment_media = media_segment
                        break
                
                if not segment_media:
                    logger.warning("No media found for segment %d", i)
                    continue
                
                # Get audio duration
                audio_duration = self._get_audio_duration(segment_info["audio_path"])
                
                # Create a clip for this segment
                clip = {
                    "segment_index": i,
                    "start_time": edl["total_duration"],
                    "duration": audio_duration,
                    "audio_path": segment_info["audio_path"],
                    "media": segment_media["media"],
                    "title": segment_info["title"],
                    "timestamp": segment_info["timestamp"]
                }
                
                edl["clips"].append(clip)
                edl["total_duration"] += audio_duration
        else:
            # No voice data, use estimated durations based on script
            words_per_second = 2.5  # Typical speaking rate
            
            for i, segment in enumerate(script_data.get("segments", [])):
                # Get the corresponding media data
                segment_media = None
                for media_segment in media_data.get("segments", []):
                    if media_segment.get("segment_index") == i:
                        segment_media = media_segment
                        break
                
                if not segment_media:
                    logger.warning("No media found for segment %d", i)
                    continue
                
                # Estimate duration based on word count
                word_count = len(segment["content"].split())
                estimated_duration = word_count / words_per_second
                
                # Ensure minimum duration
                estimated_duration = max(estimated_duration, 3.0)
                
                # Create a clip for this segment
                clip = {
                    "segment_index": i,
                    "start_time": edl["total_duration"],
                    "duration": estimated_duration,
                    "audio_path": None,
                    "media": segment_media["media"],
                    "title": segment["title"],
                    "timestamp": segment["timestamp"]
                }
                
                edl["clips"].append(clip)
                edl["total_duration"] += estimated_duration
        
        logger.info("Created edit decision list with %d clips, total duration: %.2f seconds", 
                   len(edl["clips"]), edl["total_duration"])
        
        return edl
    
    def _process_media_items(self, edl: Dict) -> List[Dict]:
        """
        Process all media items in the edit decision list.
        
        Args:
            edl: Edit decision list
            
        Returns:
            List of processed media items
        """
        logger.debug("Processing media items")
        
        processed_items = []
        
        for clip in edl["clips"]:
            # Determine which media to use for this clip
            clip_media = []
            
            # First priority: videos
            if clip["media"]["videos"]:
                for video_info in clip["media"]["videos"]:
                    clip_media.append({
                        "type": "video",
                        "path": video_info["path"],
                        "duration": self._get_video_duration(video_info["path"]),
                        "metadata": video_info.get("metadata", {})
                    })
            
            # Second priority: images
            if clip["media"]["images"] and (not clip_media or len(clip_media) < 3):
                for image_info in clip["media"]["images"]:
                    clip_media.append({
                        "type": "image",
                        "path": image_info["path"],
                        "duration": None,  # Will be calculated based on clip duration
                        "metadata": image_info.get("metadata", {})
                    })
            
            # Third priority: generated images
            if clip["media"]["generated_images"] and (not clip_media or len(clip_media) < 3):
                for image_info in clip["media"]["generated_images"]:
                    clip_media.append({
                        "type": "image",
                        "path": image_info["path"],
                        "duration": None,  # Will be calculated based on clip duration
                        "prompt": image_info.get("prompt", ""),
                        "generated": True
                    })
            
            # If we have no media, create a placeholder
            if not clip_media:
                placeholder_path = self._create_placeholder_image(
                    f"Segment {clip['segment_index'] + 1}: {clip['title']}",
                    os.path.join(self.temp_dir, f"placeholder_{clip['segment_index']}.png")
                )
                clip_media.append({
                    "type": "image",
                    "path": placeholder_path,
                    "duration": None,
                    "placeholder": True
                })
            
            # Process each media item
            clip_duration = clip["duration"]
            media_items = []
            
            remaining_duration = clip_duration
            
            for i, media in enumerate(clip_media):
                if media["type"] == "video":
                    # For videos, determine how much to use
                    video_duration = media["duration"]
                    use_duration = min(video_duration, remaining_duration)
                    
                    # Process the video (crop, resize, etc.)
                    processed_path = self._process_video(
                        media["path"],
                        os.path.join(self.temp_dir, f"proc_video_{clip['segment_index']}_{i}.mp4"),
                        start_time=0,
                        duration=use_duration
                    )
                    
                    media_items.append({
                        "type": "video",
                        "path": processed_path,
                        "start_time": clip["start_time"] + (clip_duration - remaining_duration),
                        "duration": use_duration,
                        "original_path": media["path"]
                    })
                    
                    remaining_duration -= use_duration
                    
                    # If we've used up all the duration, stop adding media
                    if remaining_duration <= 0:
                        break
                elif media["type"] == "image":
                    # For images, determine duration
                    if i == len(clip_media) - 1:
                        # Last media item gets all remaining duration
                        use_duration = remaining_duration
                    else:
                        # Otherwise, split remaining duration evenly among remaining items
                        remaining_items = len(clip_media) - i
                        use_duration = remaining_duration / remaining_items
                    
                    # Process the image (resize, apply Ken Burns effect)
                    processed_path = self._process_image(
                        media["path"],
                        os.path.join(self.temp_dir, f"proc_image_{clip['segment_index']}_{i}.mp4"),
                        duration=use_duration,
                        apply_ken_burns=self.ken_burns_enabled and not media.get("placeholder", False)
                    )
                    
                    media_items.append({
                        "type": "video",  # Note: After processing, images become video clips
                        "path": processed_path,
                        "start_time": clip["start_time"] + (clip_duration - remaining_duration),
                        "duration": use_duration,
                        "original_path": media["path"],
                        "is_image": True
                    })
                    
                    remaining_duration -= use_duration
            
            # Add the processed media items for this clip
            for item in media_items:
                processed_items.append(item)
        
        logger.info("Processed %d media items", len(processed_items))
        return processed_items
    
    def _process_image(self, image_path: str, output_path: str, duration: float, apply_ken_burns: bool = True) -> str:
        """
        Process an image for video inclusion.
        
        Args:
            image_path: Path to the input image
            output_path: Path for the output video
            duration: Duration of the output video
            apply_ken_burns: Whether to apply Ken Burns effect
            
        Returns:
            Path to the processed video
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Prepare output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Determine video resolution
        width = self.output_resolution["width"]
        height = self.output_resolution["height"]
        
        if apply_ken_burns:
            # Apply Ken Burns effect (zooming and panning)
            
            # Randomly choose zoom direction (in or out)
            zoom_in = random.choice([True, False])
            
            if zoom_in:
                # Zoom in effect
                start_scale = 1.0
                end_scale = random.uniform(self.ken_burns_min_zoom, self.ken_burns_max_zoom)
            else:
                # Zoom out effect
                start_scale = random.uniform(self.ken_burns_min_zoom, self.ken_burns_max_zoom)
                end_scale = 1.0
            
            # Random starting position (for panning)
            # Values between 0 and (scale-1) represent the top-left corner position
            start_x = random.uniform(0, max(0, start_scale - 1) * 0.5)
            start_y = random.uniform(0, max(0, start_scale - 1) * 0.5)
            
            # Random ending position (for panning)
            end_x = random.uniform(0, max(0, end_scale - 1) * 0.5)
            end_y = random.uniform(0, max(0, end_scale - 1) * 0.5)
            
            # Build the filter complex string for Ken Burns effect
            filter_complex = (
                f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
                f"zoompan=z='if(lte(on,1),{start_scale},{start_scale}+((on-1)/({self.fps}*{duration}))*({end_scale}-{start_scale}))':"
                f"x='if(lte(on,1),{start_x}*iw,{start_x}*iw+((on-1)/({self.fps}*{duration}))*({end_x}*iw-{start_x}*iw))':"
                f"y='if(lte(on,1),{start_y}*ih,{start_y}*ih+((on-1)/({self.fps}*{duration}))*({end_y}*ih-{start_y}*ih))':"
                f"d={int(self.fps * duration)}:s={width}x{height}"
            )
        else:
            # Simple scale and pad to fit the frame
            filter_complex = (
                f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"loop=loop={int(self.fps * duration)}:size=1:start=0"
            )
        
        # Execute FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",
            "-i", image_path,
            "-filter_complex", filter_complex,
            "-c:v", "libx264",
            "-t", str(duration),
            "-pix_fmt", "yuv420p",
            "-r", str(self.fps),
            output_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        
        if result.returncode != 0:
            logger.error("Error processing image: %s", result.stderr.decode())
            raise RuntimeError(f"FFmpeg error: {result.stderr.decode()}")
        
        return output_path
    
    def _process_video(self, video_path: str, output_path: str, start_time: float = 0, duration: Optional[float] = None) -> str:
        """
        Process a video for inclusion in the final video.
        
        Args:
            video_path: Path to the input video
            output_path: Path for the output video
            start_time: Start time in the source video
            duration: Duration to extract (if None, use the entire video)
            
        Returns:
            Path to the processed video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Prepare output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Determine video resolution
        width = self.output_resolution["width"]
        height = self.output_resolution["height"]
        
        # Build the filter complex string for scaling and padding
        filter_complex = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
        
        # Build the FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time)
        ]
        
        if duration is not None:
            cmd.extend(["-t", str(duration)])
        
        cmd.extend([
            "-i", video_path,
            "-filter_complex", filter_complex,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(self.fps),
            "-an",  # No audio
            output_path
        ])
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        
        if result.returncode != 0:
            logger.error("Error processing video: %s", result.stderr.decode())
            raise RuntimeError(f"FFmpeg error: {result.stderr.decode()}")
        
        return output_path
    
    def _create_placeholder_image(self, text: str, output_path: str) -> str:
        """
        Create a placeholder image with text.
        
        Args:
            text: Text to display on the placeholder
            output_path: Path for the output image
            
        Returns:
            Path to the created image
        """
        # Prepare output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Determine image resolution
        width = self.output_resolution["width"]
        height = self.output_resolution["height"]
        
        # Wrap text to fit width
        wrapped_text = '\n'.join([text[i:i+30] for i in range(0, len(text), 30)])
        
        # Build the FFmpeg command to create a placeholder image
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s={width}x{height}:d=1",
            "-vf", f"drawtext=fontfile=/path/to/font.ttf:text='{wrapped_text}':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2",
            "-frames:v", "1",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            
            if result.returncode != 0:
                logger.warning("Error creating placeholder with text: %s", result.stderr.decode())
                # Fall back to a simple color image
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-f", "lavfi",
                    "-i", f"color=c=black:s={width}x{height}:d=1",
                    "-frames:v", "1",
                    output_path
                ]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except Exception as e:
            logger.error("Error creating placeholder: %s", str(e))
            # Create an even simpler fallback using a solid color
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (width, height), color='black')
            draw = ImageDraw.Draw(img)
            draw.text((width/2, height/2), text, fill='white')
            img.save(output_path)
        
        return output_path
    
    def _assemble_final_video(self, media_items: List[Dict], audio_path: str, output_path: str) -> str:
        """
        Assemble the final video from processed media items and audio.
        
        Args:
            media_items: List of processed media items
            audio_path: Path to the audio file
            output_path: Path for the output video
            
        Returns:
            Path to the assembled video
        """
        logger.info("Assembling final video")
        
        # Sort media items by start time
        media_items.sort(key=lambda x: x["start_time"])
        
        # Create a temporary file for the concatenation list
        concat_file = os.path.join(self.temp_dir, "concat_list.txt")
        
        # Create a temporary file for each clip with the correct duration
        clip_files = []
        
        # Process each media item
        for i, item in enumerate(media_items):
            # Create a file entry for this item
            with open(concat_file, "a" if i > 0 else "w") as f:
                f.write(f"file '{item['path']}'\n")
                f.write(f"duration {item['duration']}\n")
            
            clip_files.append(item["path"])
        
        # Add the last file again (FFmpeg concat demuxer quirk)
        if media_items:
            with open(concat_file, "a") as f:
                f.write(f"file '{media_items[-1]['path']}'\n")
        
        # Create a temporary video file without audio
        temp_video = os.path.join(self.temp_dir, "temp_video.mp4")
        
        # Execute FFmpeg to concatenate the video clips
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(self.fps),
            "-preset", "medium",
            "-crf", "23",
            "-movflags", "+faststart",
            temp_video
        ]
        
        logger.debug("Executing FFmpeg concatenation: %s", " ".join(cmd))
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        
        if result.returncode != 0:
            logger.error("Error concatenating video clips: %s", result.stderr.decode())
            raise RuntimeError(f"FFmpeg error: {result.stderr.decode()}")
        
        # Now combine the video with the audio
        cmd = [
            "ffmpeg",
            "-y",
            "-i", temp_video,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-strict", "experimental",
            "-shortest",
            output_path
        ]
        
        logger.debug("Executing FFmpeg audio merge: %s", " ".join(cmd))
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        
        if result.returncode != 0:
            logger.error("Error adding audio to video: %s", result.stderr.decode())
            raise RuntimeError(f"FFmpeg error: {result.stderr.decode()}")
        
        return output_path
    
    def _create_silent_audio(self, duration: float) -> str:
        """
        Create a silent audio track.
        
        Args:
            duration: Duration of the silent audio in seconds
            
        Returns:
            Path to the silent audio file
        """
        silent_audio_path = os.path.join(self.temp_dir, "silent_audio.aac")
        
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r=44100:cl=stereo",
            "-t", str(duration),
            "-c:a", "aac",
            "-b:a", "128k",
            silent_audio_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        
        if result.returncode != 0:
            logger.error("Error creating silent audio: %s", result.stderr.decode())
            raise RuntimeError(f"FFmpeg error: {result.stderr.decode()}")
        
        return silent_audio_path
    
    def _add_subtitles(self, video_path: str, subtitles_path: str, output_path: str) -> str:
        """
        Add subtitles to a video.
        
        Args:
            video_path: Path to the input video
            subtitles_path: Path to the subtitles file
            output_path: Path for the output video
            
        Returns:
            Path to the video with subtitles
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vf", f"subtitles='{subtitles_path}'",
            "-c:a", "copy",
            output_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        
        if result.returncode != 0:
            logger.error("Error adding subtitles: %s", result.stderr.decode())
            raise RuntimeError(f"FFmpeg error: {result.stderr.decode()}")
        
        return output_path
    
    def _replace_audio(self, video_path: str, audio_path: str, output_path: str) -> str:
        """
        Replace the audio track in a video.
        
        Args:
            video_path: Path to the input video
            audio_path: Path to the new audio
            output_path: Path for the output video
            
        Returns:
            Path to the video with replaced audio
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        
        if result.returncode != 0:
            logger.error("Error replacing audio: %s", result.stderr.decode())
            raise RuntimeError(f"FFmpeg error: {result.stderr.decode()}")
        
        return output_path
    
    def _get_video_info(self, video_path: str) -> Dict:
        """
        Get information about a video file.
        
        Args:
            video_path: Path to the video
            
        Returns:
            Dictionary with video information
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration",
            "-of", "json",
            video_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        
        if result.returncode != 0:
            logger.error("Error getting video info: %s", result.stderr)
            return {}
        
        try:
            info = json.loads(result.stdout)
            stream = info.get("streams", [{}])[0]
            
            # Parse frame rate (e.g., "30000/1001")
            fps = stream.get("r_frame_rate", "").split("/")
            if len(fps) == 2 and fps[1] != "0":
                fps_value = float(fps[0]) / float(fps[1])
            else:
                fps_value = self.fps
            
            return {
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "fps": fps_value,
                "duration": float(stream.get("duration", 0)),
                "resolution": f"{stream.get('width', 0)}x{stream.get('height', 0)}"
            }
        except (json.JSONDecodeError, IndexError, KeyError, ValueError) as e:
            logger.error("Error parsing video info: %s", str(e))
            return {}
    
    def _get_video_duration(self, video_path: str) -> float:
        """
        Get the duration of a video file.
        
        Args:
            video_path: Path to the video
            
        Returns:
            Duration in seconds
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        
        if result.returncode != 0:
            logger.error("Error getting video duration: %s", result.stderr)
            return 0.0
        
        try:
            return float(result.stdout.strip())
        except ValueError:
            logger.error("Error parsing video duration")
            return 0.0
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """
        Get the duration of an audio file.
        
        Args:
            audio_path: Path to the audio
            
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
