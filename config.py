# config/config_manager.py

import os
import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages configuration for the YouTube video automation system.
    Handles loading from files, environment variables, and provides defaults.
    """
    
    def __init__(self, config_dir: str = "config", env_prefix: str = "YT_AUTO"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
            env_prefix: Prefix for environment variables
        """
        self.config_dir = config_dir
        self.env_prefix = env_prefix
        self.config = {}
        
        # Load default configuration
        self._load_default_config()
        
        # Override with environment-specific config if available
        env = os.environ.get(f"{env_prefix}_ENV", "development")
        self._load_environment_config(env)
        
        # Override with environment variables
        self._load_from_env()
        
    def _load_default_config(self) -> None:
        """Load the default configuration from file."""
        default_config_path = os.path.join(self.config_dir, "default.json")
        if os.path.exists(default_config_path):
            try:
                with open(default_config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info("Loaded default configuration")
            except Exception as e:
                logger.error(f"Error loading default configuration: {str(e)}")
                # Initialize with basic defaults
                self._set_defaults()
        else:
            logger.warning(f"Default configuration file not found at {default_config_path}")
            self._set_defaults()
    
    def _load_environment_config(self, env: str) -> None:
        """Load environment-specific configuration."""
        env_config_path = os.path.join(self.config_dir, f"{env}.json")
        if os.path.exists(env_config_path):
            try:
                with open(env_config_path, 'r') as f:
                    env_config = json.load(f)
                    # Deep merge the configurations
                    self._deep_merge(self.config, env_config)
                logger.info(f"Loaded {env} configuration")
            except Exception as e:
                logger.error(f"Error loading {env} configuration: {str(e)}")
    
    def _load_from_env(self) -> None:
        """
        Override configuration settings from environment variables.
        Environment variables should be in the format:
        YT_AUTO_SECTION_SUBSECTION_KEY=value
        
        For example:
        YT_AUTO_SCRIPT_GENERATOR_AI_PROVIDER=openai
        """
        prefix = f"{self.env_prefix}_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and split by underscore
                config_path = key[len(prefix):].lower().split('_')
                
                # Convert string value to appropriate type
                if value.lower() == 'true':
                    typed_value = True
                elif value.lower() == 'false':
                    typed_value = False
                elif value.isdigit():
                    typed_value = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                    typed_value = float(value)
                else:
                    typed_value = value
                
                # Set the value in the nested config
                self._set_nested_value(self.config, config_path, typed_value)
    
    def _set_nested_value(self, config: Dict, path: list, value: Any) -> None:
        """Set a value in a nested dictionary using a path."""
        if len(path) == 1:
            config[path[0]] = value
        else:
            if path[0] not in config:
                config[path[0]] = {}
            self._set_nested_value(config[path[0]], path[1:], value)
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> None:
        """
        Recursively merge dict2 into dict1.
        Values in dict2 override those in dict1.
        """
        for key, value in dict2.items():
            if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                self._deep_merge(dict1[key], value)
            else:
                dict1[key] = value
    
    def _set_defaults(self) -> None:
        """Set default configuration values."""
        self.config = {
            "script_generator": {
                "ai_provider": "openai",
                "max_tokens": 4000,
                "template_dir": "modules/script_generator/templates",
                "default_template": "educational.j2"
            },
            "voice_generator": {
                "provider": "elevenlabs",
                "voice_style": "professional",
                "speaking_rate": 1.0
            },
            "media_selector": {
                "sources": ["pexels", "pixabay", "unsplash"],
                "ai_generation": {
                    "enabled": False,
                    "provider": "stable_diffusion"
                }
            },
            "video_editor": {
                "resolution": "1080p",
                "fps": 30,
                "transitions": {
                    "default": "fade",
                    "duration": 1.0
                },
                "color_grading": {
                    "enabled": True,
                    "lut": "cinematic_1"
                }
            },
            "subtitles": {
                "enabled": True,
                "provider": "whisper",
                "font": "Roboto",
                "font_size": 32
            },
            "audio": {
                "music": {
                    "enabled": True,
                    "volume": 0.2
                },
                "voiceover": {
                    "volume": 1.0
                }
            },
            "api_keys": {
                "openai": "",
                "anthropic": "",
                "elevenlabs": "",
                "pexels": "",
                "pixabay": ""
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Dot-separated path to the configuration value
            default: Default value if not found
            
        Returns:
            The configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Dot-separated path to the configuration value
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
    
    def save(self, config_file: str = "user.json") -> None:
        """
        Save the current configuration to a file.
        
        Args:
            config_file: Name of the file to save to
        """
        config_path = os.path.join(self.config_dir, config_file)
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")

# Create example default configuration
def create_example_config():
    """Create an example default configuration file."""
    os.makedirs("config", exist_ok=True)
    config = {
        "script_generator": {
            "ai_provider": "openai",
            "max_tokens": 4000,
            "template_dir": "modules/script_generator/templates",
            "default_template": "educational.j2"
        },
        "voice_generator": {
            "provider": "elevenlabs",
            "voice_style": "professional",
            "speaking_rate": 1.0
        },
        "media_selector": {
            "sources": ["pexels", "pixabay", "unsplash"],
            "ai_generation": {
                "enabled": False,
                "provider": "stable_diffusion"
            }
        },
        "video_editor": {
            "resolution": "1080p",
            "fps": 30,
            "transitions": {
                "default": "fade",
                "duration": 1.0
            },
            "color_grading": {
                "enabled": True,
                "lut": "cinematic_1"
            }
        },
        "subtitles": {
            "enabled": True,
            "provider": "whisper",
            "font": "Roboto",
            "font_size": 32
        },
        "audio": {
            "music": {
                "enabled": True,
                "volume": 0.2
            },
            "voiceover": {
                "volume": 1.0
            }
        },
        "api_keys": {
            "openai": "",
            "anthropic": "",
            "elevenlabs": "",
            "pexels": "",
            "pixabay": ""
        }
    }
    
    with open("config/default.json", "w") as f:
        json.dump(config, f, indent=2)
