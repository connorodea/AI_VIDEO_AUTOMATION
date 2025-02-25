# core/voice_generator.py

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, BinaryIO


class VoiceGenerator:
    """
    Generates AI voiceovers from script text using various TTS providers.
    Supports ElevenLabs, Google Cloud TTS, Amazon Polly, and Coqui TTS.
    """

    def __init__(self, config_path: str = "config/default_settings.json"):
        """
        Initialize the VoiceGenerator with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.voice_config = self.config.get("voiceover", {})
        self.provider = self.voice_config.get("provider", "elevenlabs").lower()
        self.default_voice = self.voice_config.get("default_voice", "adam")
        self.speech_rate = self.voice_config.get("speech_rate", 1.0)
        self.stability = self.voice_config.get("stability", 0.5)
        self.clarity = self.voice_config.get("clarity", 0.75)
        self.client = self._initialize_client()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using default settings.")
            return {
                "voiceover": {
                    "provider": "elevenlabs",
                    "default_voice": "adam",
                    "speech_rate": 1.0,
                    "stability": 0.5,
                    "clarity": 0.75
                }
            }
    
    def _initialize_client(self):
        """Initialize the client for the selected TTS provider."""
        if self.provider == "elevenlabs":
            try:
                from elevenlabs import set_api_key
                api_key = self._get_api_key("elevenlabs")
                if api_key:
                    set_api_key(api_key)
                return True
            except ImportError:
                print("Error: ElevenLabs package not installed. Run 'pip install elevenlabs'")
        
        elif self.provider == "google":
            try:
                from google.cloud import texttospeech
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self._get_google_credentials_path()
                return texttospeech.TextToSpeechClient()
            except ImportError:
                print("Error: Google Cloud TTS package not installed. Run 'pip install google-cloud-texttospeech'")
        
        elif self.provider == "aws":
            try:
                import boto3
                return boto3.client('polly', 
                                   aws_access_key_id=self._get_api_key("aws_access_key"), 
                                   aws_secret_access_key=self._get_api_key("aws_secret_key"),
                                   region_name=self.voice_config.get("aws_region", "us-east-1"))
            except ImportError:
                print("Error: Boto3 package not installed. Run 'pip install boto3'")
        
        elif self.provider == "coqui":
            try:
                from TTS.api import TTS
                model_name = self.voice_config.get("coqui_model", "tts_models/en/vctk/vits")
                return TTS(model_name=model_name)
            except ImportError:
                print("Error: Coqui TTS package not installed. Run 'pip install TTS'")
        
        return None
    
    def _get_api_key(self, key_name: str) -> Optional[str]:
        """Get API key from environment variables or config file."""
        # Try environment variables first
        env_var_name = f"{key_name.upper()}_API_KEY"
        if env_var_name in os.environ:
            return os.environ[env_var_name]
        
        # Try config file
        try:
            with open("config/api_keys.json", "r") as f:
                keys = json.load(f)
                return keys.get(key_name)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        print(f"Warning: No API key found for {key_name}. You'll need to provide it explicitly.")
        return None
    
    def _get_google_credentials_path(self) -> str:
        """Get the path to Google Cloud credentials file."""
        # Check environment variable
        if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            return os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        
        # Check config
        try:
            with open("config/api_keys.json", "r") as f:
                keys = json.load(f)
                return keys.get("google_credentials_path", "config/google_credentials.json")
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        # Default to a path in the config directory
        return "config/google_credentials.json"
    
    def generate_audio(self, 
                      text: str, 
                      output_path: str,
                      voice: Optional[str] = None,
                      rate: Optional[float] = None,
                      stability: Optional[float] = None,
                      clarity: Optional[float] = None,
                      **kwargs) -> str:
        """
        Generate audio from text using the configured TTS provider.
        
        Args:
            text: The text to convert to speech
            output_path: Path to save the audio file
            voice: Voice ID/name to use (provider-specific)
            rate: Speech rate multiplier (1.0 = normal speed)
            stability: Voice stability (0.0-1.0, provider-specific)
            clarity: Voice clarity/similarity (0.0-1.0, provider-specific)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Path to the saved audio file
        """
        # Use default values if not provided
        voice = voice or self.default_voice
        rate = rate if rate is not None else self.speech_rate
        stability = stability if stability is not None else self.stability
        clarity = clarity if clarity is not None else self.clarity
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if self.provider == "elevenlabs":
            return self._generate_elevenlabs(text, output_path, voice, rate, stability, clarity, **kwargs)
        elif self.provider == "google":
            return self._generate_google(text, output_path, voice, rate, **kwargs)
        elif self.provider == "aws":
            return self._generate_aws(text, output_path, voice, rate, **kwargs)
        elif self.provider == "coqui":
            return self._generate_coqui(text, output_path, voice, rate, **kwargs)
        else:
            raise ValueError(f"Unsupported TTS provider: {self.provider}")
    
    def _generate_elevenlabs(self, text, output_path, voice, rate, stability, clarity, **kwargs):
        """Generate audio using ElevenLabs API."""
        try:
            from elevenlabs import generate, save, Voice, VoiceSettings
            
            # Create voice settings
            voice_settings = VoiceSettings(
                stability=stability,
                similarity_boost=clarity,
                style=kwargs.get("style", 0.0),
                use_speaker_boost=kwargs.get("speaker_boost", True)
            )
            
            # Generate audio
            audio = generate(
                text=text,
                voice=Voice(
                    voice_id=voice,
                    settings=voice_settings
                ),
                model=kwargs.get("model", "eleven_monolingual_v1")
            )
            
            # Save audio
            save(audio, output_path)
            
            return output_path
        except Exception as e:
            print(f"Error generating audio with ElevenLabs: {e}")
            raise
    
    def _generate_google(self, text, output_path, voice, rate, **kwargs):
        """Generate audio using Google Cloud TTS."""
        try:
            from google.cloud import texttospeech
            
            # Configure input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Build voice params
            language_code = kwargs.get("language_code", "en-US")
            voice_params = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice,
                ssml_gender=kwargs.get("gender", texttospeech.SsmlVoiceGender.NEUTRAL)
            )
            
            # Select audio config
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=rate,
                pitch=kwargs.get("pitch", 0.0),
                volume_gain_db=kwargs.get("volume", 0.0),
                effects_profile_id=kwargs.get("effects_profile", ["small-bluetooth-speaker-class-device"])
            )
            
            # Generate audio
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config
            )
            
            # Save audio
            with open(output_path, "wb") as f:
                f.write(response.audio_content)
            
            return output_path
        except Exception as e:
            print(f"Error generating audio with Google Cloud TTS: {e}")
            raise
    
    def _generate_aws(self, text, output_path, voice, rate, **kwargs):
        """Generate audio using Amazon Polly."""
        try:
            # Configure parameters
            params = {
                'Text': text,
                'OutputFormat': 'mp3',
                'VoiceId': voice,
                'Engine': kwargs.get("engine", "neural"),
                'LanguageCode': kwargs.get("language_code", "en-US")
            }
            
            # Add rate if specified
            if rate != 1.0:
                params['SpeechMarkTypes'] = ['ssml']
                # Wrap text in SSML with rate
                params['Text'] = f'<speak><prosody rate="{int(rate * 100)}%">{text}</prosody></speak>'
                params['TextType'] = 'ssml'
            
            # Generate speech
            response = self.client.synthesize_speech(**params)
            
            # Save audio
            with open(output_path, "wb") as f:
                f.write(response['AudioStream'].read())
            
            return output_path
        except Exception as e:
            print(f"Error generating audio with Amazon Polly: {e}")
            raise
    
    def _generate_coqui(self, text, output_path, voice, rate, **kwargs):
        """Generate audio using Coqui TTS (local model)."""
        try:
            speaker = kwargs.get("speaker", voice)
            
            # Generate audio
            self.client.tts_to_file(
                text=text,
                file_path=output_path,
                speaker=speaker,
                speed=rate
            )
            
            return output_path
        except Exception as e:
            print(f"Error generating audio with Coqui TTS: {e}")
            raise
    
    def generate_audio_for_script(self, 
                                 script_data: Dict, 
                                 output_dir: str,
                                 voice: Optional[str] = None,
                                 combine: bool = False,
                                 **kwargs) -> Dict:
        """
        Generate audio for each segment in a script.
        
        Args:
            script_data: Script data from ScriptGenerator
            output_dir: Directory to save audio files
            voice: Voice to use for all segments
            combine: Whether to combine all segments into a single audio file
            **kwargs: Additional parameters for generate_audio
            
        Returns:
            Dictionary with paths to generated audio files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        result = {
            "segments": [],
            "full_audio": None
        }
        
        # Generate audio for each segment
        for i, segment in enumerate(script_data["segments"]):
            segment_filename = f"segment_{i:03d}_{segment['timestamp'].replace(':', '_')}.mp3"
            segment_path = os.path.join(output_dir, segment_filename)
            
            # Generate audio for this segment
            audio_path = self.generate_audio(
                text=segment["content"],
                output_path=segment_path,
                voice=voice,
                **kwargs
            )
            
            # Add to result
            result["segments"].append({
                "segment_index": i,
                "segment_id": segment.get("id", str(i)),
                "timestamp": segment["timestamp"],
                "time_in_seconds": segment["time_in_seconds"],
                "title": segment["title"],
                "audio_path": audio_path
            })
        
        # Combine all segments if requested
        if combine and result["segments"]:
            try:
                import pydub
                
                # Concatenate all audio segments
                combined_audio = pydub.AudioSegment.empty()
                for segment_info in result["segments"]:
                    audio_segment = pydub.AudioSegment.from_mp3(segment_info["audio_path"])
                    combined_audio += audio_segment
                
                # Export combined audio
                combined_path = os.path.join(output_dir, "full_audio.mp3")
                combined_audio.export(combined_path, format="mp3")
                result["full_audio"] = combined_path
            except ImportError:
                print("Warning: pydub not installed. Could not combine audio segments.")
                print("Run 'pip install pydub' to enable audio combining.")
        
        return result
    
    def get_available_voices(self) -> List[Dict]:
        """
        Get a list of available voices for the current provider.
        
        Returns:
            List of voice information dictionaries
        """
        if self.provider == "elevenlabs":
            try:
                from elevenlabs import voices
                return [{"id": v.voice_id, "name": v.name, "category": "elevenlabs"} for v in voices()]
            except Exception as e:
                print(f"Error getting ElevenLabs voices: {e}")
                return []
        
        elif self.provider == "google":
            try:
                from google.cloud import texttospeech
                response = self.client.list_voices()
                return [{"id": v.name, "name": v.name, "language_codes": v.language_codes, "gender": str(v.ssml_gender)} for v in response.voices]
            except Exception as e:
                print(f"Error getting Google Cloud voices: {e}")
                return []
        
        elif self.provider == "aws":
            try:
                response = self.client.describe_voices()
                return [{"id": v["Id"], "name": v["Name"], "language": v["LanguageCode"], "gender": v["Gender"], "engine": v["SupportedEngines"]} for v in response["Voices"]]
            except Exception as e:
                print(f"Error getting Amazon Polly voices: {e}")
                return []
        
        elif self.provider == "coqui":
            try:
                return [{"id": speaker, "name": speaker} for speaker in self.client.speakers]
            except Exception as e:
                print(f"Error getting Coqui TTS voices: {e}")
                return []
        
        return []
