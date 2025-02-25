# core/audio_processor.py

import json
import os
import subprocess
import tempfile
import shutil
import random
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('audio_processor')

class AudioProcessor:
    """
    Processes and enhances audio for videos, including voice enhancement,
    background music selection and mixing, and audio normalization.
    """

    def __init__(self, config_path: str = "config/default_settings.json"):
        """
        Initialize the AudioProcessor with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.audio_config = self.config.get("audio", {})
        
        # Music settings
        self.music_config = self.audio_config.get("music", {})
        self.music_enabled = self.music_config.get("enabled", True)
        self.music_volume = self.music_config.get("volume", 0.2)
        self.music_fade_in = self.music_config.get("fade_in", 2.0)
        self.music_fade_out = self.music_config.get("fade_out", 3.0)
        self.music_sources = self.music_config.get("sources", ["musicbed", "artlist", "epidemic_sound"])
        
        # Normalization settings
        self.norm_config = self.audio_config.get("normalization", {})
        self.target_lufs = self.norm_config.get("target_lufs", -14)
        self.true_peak = self.norm_config.get("true_peak", -1.0)
        
        # Voice enhancement settings
        self.voice_config = self.audio_config.get("voice_enhancement", {})
        self.voice_enhance_enabled = self.voice_config.get("enabled", True)
        self.noise_reduction = self.voice_config.get("noise_reduction", 0.3)
        self.compression = self.voice_config.get("compression", 0.4)
        self.eq_config = self.voice_config.get("eq", {})
        
        # Default music directory
        self.default_music_dir = os.path.join("assets", "music")
        if not os.path.exists(self.default_music_dir):
            os.makedirs(self.default_music_dir, exist_ok=True)
        
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
                "audio": {
                    "music": {
                        "enabled": True,
                        "volume": 0.2,
                        "fade_in": 2.0,
                        "fade_out": 3.0,
                        "sources": ["musicbed", "artlist", "epidemic_sound"]
                    },
                    "normalization": {
                        "target_lufs": -14,
                        "true_peak": -1.0
                    },
                    "voice_enhancement": {
                        "enabled": True,
                        "noise_reduction": 0.3,
                        "compression": 0.4,
                        "eq": {
                            "enabled": True,
                            "preset": "voice"
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
                logger.error("FFmpeg is not available. Please install FFmpeg to use the audio processor.")
                raise RuntimeError("FFmpeg is not available")
            else:
                logger.info("FFmpeg is available")
        except FileNotFoundError:
            logger.error("FFmpeg is not available. Please install FFmpeg to use the audio processor.")
            raise RuntimeError("FFmpeg is not available")
    
    def process_audio(
        self,
        voice_audio_path: str,
        output_dir: str,
        add_music: bool = True,
        music_path: Optional[str] = None,
        music_mood: Optional[str] = None,
        enhance_voice: Optional[bool] = None,
        normalize_audio: bool = True,
        **kwargs
    ) -> Dict:
        """
        Process audio for a video.
        
        Args:
            voice_audio_path: Path to the voice audio file
            output_dir: Directory to save the processed audio
            add_music: Whether to add background music
            music_path: Path to a specific music file to use (if None, one will be selected)
            music_mood: Mood for music selection (e.g., "upbeat", "calm", "dramatic")
            enhance_voice: Whether to enhance the voice audio
            normalize_audio: Whether to normalize the final audio
            **kwargs: Additional options
            
        Returns:
            Dictionary with information about the processed audio
        """
        logger.info("Processing audio")
        
        # Check if voice audio exists
        if not os.path.exists(voice_audio_path):
            raise FileNotFoundError(f"Voice audio not found: {voice_audio_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a temporary directory for intermediate files
        self.temp_dir = tempfile.mkdtemp()
        logger.debug("Created temporary directory: %s", self.temp_dir)
        
        try:
            # Get the voice audio duration
            voice_duration = self._get_audio_duration(voice_audio_path)
            
            # 1. Enhance voice audio if enabled
            if enhance_voice is None:
                enhance_voice = self.voice_enhance_enabled
            
            if enhance_voice:
                logger.info("Enhancing voice audio")
                enhanced_voice_path = os.path.join(self.temp_dir, "enhanced_voice.wav")
                self._enhance_voice(voice_audio_path, enhanced_voice_path)
            else:
                enhanced_voice_path = voice_audio_path
            
            # 2. Select and process background music if enabled
            if add_music and self.music_enabled:
                logger.info("Adding background music")
                
                # Select music file
                if music_path and os.path.exists(music_path):
                    selected_music_path = music_path
                else:
                    selected_music_path = self._select_music(music_mood, voice_duration)
                
                # Process the music (trim, fade, adjust volume)
                if selected_music_path:
                    processed_music_path = os.path.join(self.temp_dir, "processed_music.wav")
                    self._process_music(selected_music_path, processed_music_path, voice_duration)
                    
                    # Mix voice and music
                    mixed_output_path = os.path.join(self.temp_dir, "mixed_audio.wav")
                    self._mix_audio(enhanced_voice_path, processed_music_path, mixed_output_path)
                    
                    work_audio_path = mixed_output_path
                else:
                    logger.warning("No suitable music found, using voice audio only")
                    work_audio_path = enhanced_voice_path
            else:
                work_audio_path = enhanced_voice_path
            
            # 3. Normalize audio if enabled
            output_path = os.path.join(output_dir, "processed_audio.wav")
            
            if normalize_audio:
                logger.info("Normalizing audio")
                self._normalize_audio(work_audio_path, output_path)
            else:
                # Just copy the audio if not normalizing
                shutil.copy2(work_audio_path, output_path)
            
            # Also create an MP3 version for convenience
            mp3_path = os.path.join(output_dir, "processed_audio.mp3")
            self._convert_to_mp3(output_path, mp3_path)
            
            return {
                "output_path": output_path,
                "mp3_path": mp3_path,
                "duration": self._get_audio_duration(output_path),
                "enhanced_voice": enhance_voice,
                "added_music": add_music and self.music_enabled,
                "normalized": normalize_audio
            }
        
        except Exception as e:
            logger.error("Error processing audio: %s", str(e), exc_info=True)
            raise
        
        finally:
            self._cleanup()
    
    def _enhance_voice(self, input_path: str, output_path: str) -> str:
        """
        Enhance voice audio with noise reduction, EQ, and compression.
        
        Args:
            input_path: Path to the input audio
            output_path: Path for the output audio
            
        Returns:
            Path to the enhanced audio
        """
        # Create a complex filter for voice enhancement
        filter_complex = []
        
        # Add highpass filter to remove rumble
        filter_complex.append(f"highpass=f=80")
        
        # Add noise reduction if enabled
        if self.noise_reduction > 0:
            nr_amount = min(1.0, max(0.0, self.noise_reduction))
            filter_complex.append(f"afftdn=nr={nr_amount*10}:nf={nr_amount*5}")
        
        # Add EQ if enabled
        if self.eq_config.get("enabled", True):
            preset = self.eq_config.get("preset", "voice")
            
            if preset == "voice":
                # Voice EQ preset (boost clarity, cut mud)
                filter_complex.append(f"equalizer=f=100:t=h:width=200:g=-3")  # Cut low rumble
                filter_complex.append(f"equalizer=f=250:t=o:width=300:g=-2")  # Cut mud
                filter_complex.append(f"equalizer=f=1200:t=o:width=600:g=2")  # Boost presence
                filter_complex.append(f"equalizer=f=4000:t=o:width=1000:g=3")  # Boost clarity
                filter_complex.append(f"equalizer=f=8000:t=h:width=2000:g=-1")  # Cut hiss
            elif preset == "male_voice":
                # Male voice preset
                filter_complex.append(f"equalizer=f=120:t=h:width=200:g=-2")
                filter_complex.append(f"equalizer=f=250:t=o:width=200:g=-3")
                filter_complex.append(f"equalizer=f=2500:t=o:width=1000:g=3")
                filter_complex.append(f"equalizer=f=7000:t=o:width=2000:g=2")
            elif preset == "female_voice":
                # Female voice preset
                filter_complex.append(f"equalizer=f=90:t=h:width=100:g=-4")
                filter_complex.append(f"equalizer=f=200:t=o:width=200:g=-2")
                filter_complex.append(f"equalizer=f=3000:t=o:width=1000:g=3")
                filter_complex.append(f"equalizer=f=8000:t=o:width=2000:g=1")
        
        # Add compression if enabled
        if self.compression > 0:
            comp_amount = min(1.0, max(0.0, self.compression))
            filter_complex.append(f"compand=attacks=0.01:decays=0.5:points=-80/-80|-60/-40|-40/-30|-30/-20|-20/-15|-10/-10|0/0:soft-knee=6:gain=2")
        
        # Add de-esser
        filter_complex.append(f"highshelf=f=8000:g=-6")
        
        # Add final limiter
        filter_complex.append(f"alimiter=limit=0.95:level=true:attack=5:release=50")
        
        # Join the filters
        filter_string = ','.join(filter_complex)
        
        # Execute FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-af", filter_string,
            "-c:a", "pcm_s16le",
            output_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        
        if result.returncode != 0:
            logger.error("Error enhancing voice: %s", result.stderr.decode())
            return input_path  # Return the original file as fallback
        
        return output_path
    
    def _select_music(self, mood: Optional[str] = None, duration: Optional[float] = None) -> Optional[str]:
        """
        Select a suitable background music track.
        
        Args:
            mood: Desired mood for the music
            duration: Approximate duration needed
            
        Returns:
            Path to the selected music file
        """
        music_dir = self.default_music_dir
        
        # Get a list of all music files
        music_files = []
        for root, _, files in os.walk(music_dir):
            for file in files:
                if file.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg')):
                    path = os.path.join(root, file)
                    music_files.append(path)
        
        if not music_files:
            logger.warning("No music files found in %s", music_dir)
            return None
        
        # If mood is specified, try to match it
        if mood:
            mood_matches = []
            for file in music_files:
                # Check if the filename contains the mood
                if mood.lower() in os.path.basename(file).lower():
                    mood_matches.append(file)
            
            if mood_matches:
                # Randomly select from mood matches
                return random.choice(mood_matches)
        
        # Otherwise, just pick a random file
        return random.choice(music_files)
    
    def _process_music(self, music_path: str, output_path: str, target_duration: float) -> str:
        """
        Process music for background use (trim, fade, adjust volume).
        
        Args:
            music_path: Path to the music file
            output_path: Path for the processed music
            target_duration: Target duration for the music
            
        Returns:
            Path to the processed music
        """
        # Get the music duration
        music_duration = self._get_audio_duration(music_path)
        
        # Determine if we need to loop the music
        if music_duration < target_duration:
            # We need to loop the music
            loop_count = int(target_duration / music_duration) + 1
            
            # Build a complex filter for looping
            filter_complex = f"aloop=loop={loop_count}:size=2e+09"
            
            # Add fades and duration limit
            filter_complex += f",afade=t=in:st=0:d={self.music_fade_in}"
            filter_complex += f",afade=t=out:st={target_duration - self.music_fade_out}:d={self.music_fade_out}"
            filter_complex += f",atrim=0:{target_duration}"
            
            # Add volume adjustment
            filter_complex += f",volume={self.music_volume}"
            
            # Execute FFmpeg command
            cmd = [
                "ffmpeg",
                "-y",
                "-i", music_path,
                "-af", filter_complex,
                "-c:a", "pcm_s16le",
                output_path
            ]
        else:
            # Music is long enough, just need to trim and fade
            # Determine where to start in the music (random if music is much longer)
            if music_duration > target_duration + 30:
                # Pick a random start point, but leave enough room for the full duration
                max_start = music_duration - target_duration - 5
                start_time = random.uniform(0, max_start)
            else:
                start_time = 0
            
            # Build a filter for trimming, fading and volume adjustment
            filter_complex = f"atrim={start_time}:{start_time + target_duration}"
            filter_complex += f",afade=t=in:st=0:d={self.music_fade_in}"
            filter_complex += f",afade=t=out:st={target_duration - self.music_fade_out}:d={self.music_fade_out}"
            filter_complex += f",volume={self.music_volume}"
            
            # Execute FFmpeg command
            cmd = [
                "ffmpeg",
                "-y",
                "-i", music_path,
                "-af", filter_complex,
                "-c:a", "pcm_s16le",
                output_path
            ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        
        if result.returncode != 0:
            logger.error("Error processing music: %s", result.stderr.decode())
            return music_path  # Return the original file as fallback
        
        return output_path
    
    def _mix_audio(self, voice_path: str, music_path: str, output_path: str) -> str:
        """
        Mix voice and music audio.
        
        Args:
            voice_path: Path to the voice audio
            music_path: Path to the music audio
            output_path: Path for the mixed audio
            
        Returns:
            Path to the mixed audio
        """
        # Execute FFmpeg command to mix audio
        cmd = [
            "ffmpeg",
            "-y",
            "-i", voice_path,
            "-i", music_path,
            "-filter_complex", "amix=inputs=2:duration=longest",
            "-c:a", "pcm_s16le",
            output_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        
        if result.returncode != 0:
            logger.error("Error mixing audio: %s", result.stderr.decode())
            return voice_path  # Return the voice audio as fallback
        
        return output_path
    
    def _normalize_audio(self, input_path: str, output_path: str) -> str:
        """
        Normalize audio to target loudness.
        
        Args:
            input_path: Path to the input audio
            output_path: Path for the normalized audio
            
        Returns:
            Path to the normalized audio
        """
        # Build the filter for loudness normalization
        filter_complex = f"loudnorm=I={self.target_lufs}:TP={self.true_peak}:LRA=7"
        
        # Execute FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-af", filter_complex,
            "-c:a", "pcm_s16le",
            output_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        
        if result.returncode != 0:
            logger.error("Error normalizing audio: %s", result.stderr.decode())
            return input_path  # Return the input file as fallback
        
        return output_path
    
    def _convert_to_mp3(self, input_path: str, output_path: str) -> str:
        """
        Convert audio to MP3 format.
        
        Args:
            input_path: Path to the input audio
            output_path: Path for the MP3 audio
            
        Returns:
            Path to the MP3 audio
        """
        # Execute FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-c:a", "libmp3lame",
            "-q:a", "2",
            output_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        
        if result.returncode != 0:
            logger.error("Error converting to MP3: %s", result.stderr.decode())
            return input_path  # Return the input file as fallback
        
        return output_path
    
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
