# core/pipeline.py

import json
import os
import time
import uuid
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pipeline')

class Pipeline:
    """
    Main workflow orchestrator for the video generation pipeline.
    Coordinates the execution of all components in the correct sequence.
    """

    def __init__(self, config_path: str = "config/default_settings.json"):
        """
        Initialize the Pipeline with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Initialize components (lazy loading)
        self._script_generator = None
        self._voice_generator = None
        self._media_selector = None
        self._video_editor = None
        self._subtitle_generator = None
        self._audio_processor = None
        
        # Output directories
        self.output_base_dir = self.config.get("output_dir", "output")
        self.projects_dir = os.path.join(self.output_base_dir, "projects")
        
        # Create output directories
        os.makedirs(self.output_base_dir, exist_ok=True)
        os.makedirs(self.projects_dir, exist_ok=True)
        
        # Pipeline stage handlers (for flexibility in execution order)
        self.stage_handlers = {
            "script_generation": self._handle_script_generation,
            "voice_generation": self._handle_voice_generation,
            "media_selection": self._handle_media_selection,
            "subtitle_generation": self._handle_subtitle_generation,
            "video_assembly": self._handle_video_assembly,
            "audio_processing": self._handle_audio_processing,
            "final_export": self._handle_final_export
        }
        
        # Default stage sequence
        self.default_stage_sequence = [
            "script_generation",
            "voice_generation",
            "media_selection",
            "subtitle_generation",
            "video_assembly",
            "audio_processing",
            "final_export"
        ]
        
        logger.info("Pipeline initialized with config from %s", config_path)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Config file %s not found. Using default settings.", config_path)
            return {
                "output_dir": "output"
            }
    
    # Lazy loading properties for components
    
    @property
    def script_generator(self):
        """Lazy-load the script generator."""
        if self._script_generator is None:
            try:
                from core.script_generator import ScriptGenerator
                self._script_generator = ScriptGenerator(self.config_path)
                logger.info("ScriptGenerator initialized")
            except ImportError as e:
                logger.error("Failed to import ScriptGenerator: %s", e)
                raise ImportError("ScriptGenerator module is required for the pipeline")
        return self._script_generator
    
    @property
    def voice_generator(self):
        """Lazy-load the voice generator."""
        if self._voice_generator is None:
            try:
                from core.voice_generator import VoiceGenerator
                self._voice_generator = VoiceGenerator(self.config_path)
                logger.info("VoiceGenerator initialized")
            except ImportError as e:
                logger.error("Failed to import VoiceGenerator: %s", e)
                self._voice_generator = DummyVoiceGenerator()
                logger.warning("Using DummyVoiceGenerator instead")
        return self._voice_generator
    
    @property
    def media_selector(self):
        """Lazy-load the media selector."""
        if self._media_selector is None:
            try:
                from core.media_selector import MediaSelector
                self._media_selector = MediaSelector(self.config_path)
                logger.info("MediaSelector initialized")
            except ImportError as e:
                logger.error("Failed to import MediaSelector: %s", e)
                self._media_selector = DummyMediaSelector()
                logger.warning("Using DummyMediaSelector instead")
        return self._media_selector
    
    @property
    def video_editor(self):
        """Lazy-load the video editor."""
        if self._video_editor is None:
            try:
                from core.video_editor import VideoEditor
                self._video_editor = VideoEditor(self.config_path)
                logger.info("VideoEditor initialized")
            except ImportError as e:
                logger.debug("VideoEditor module not found: %s", e)
                self._video_editor = DummyVideoEditor()
                logger.info("Using DummyVideoEditor instead")
        return self._video_editor
    
    @property
    def subtitle_generator(self):
        """Lazy-load the subtitle generator."""
        if self._subtitle_generator is None:
            try:
                from core.subtitle_generator import SubtitleGenerator
                self._subtitle_generator = SubtitleGenerator(self.config_path)
                logger.info("SubtitleGenerator initialized")
            except ImportError as e:
                logger.debug("SubtitleGenerator module not found: %s", e)
                self._subtitle_generator = DummySubtitleGenerator()
                logger.info("Using DummySubtitleGenerator instead")
        return self._subtitle_generator
    
    @property
    def audio_processor(self):
        """Lazy-load the audio processor."""
        if self._audio_processor is None:
            try:
                from core.audio_processor import AudioProcessor
                self._audio_processor = AudioProcessor(self.config_path)
                logger.info("AudioProcessor initialized")
            except ImportError as e:
                logger.debug("AudioProcessor module not found: %s", e)
                self._audio_processor = DummyAudioProcessor()
                logger.info("Using DummyAudioProcessor instead")
        return self._audio_processor
    
    # Main pipeline methods
    
    def generate_video(
        self,
        topic: str,
        output_name: Optional[str] = None,
        script_type: str = "standard",
        duration: int = 5,
        tone: Optional[str] = None,
        voice: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        stage_sequence: Optional[List[str]] = None,
        start_from_stage: Optional[str] = None,
        skip_stages: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
        """
        Generate a complete video from a topic.
        
        Args:
            topic: The main topic for the video
            output_name: Name for the output files (if None, will be derived from topic)
            script_type: Type of script (standard, educational, entertainment)
            duration: Target video duration in minutes
            tone: Tone of the script (informative, casual, professional, etc.)
            voice: Voice to use for the voiceover
            keywords: List of keywords to emphasize in the script
            stage_sequence: Custom sequence of stages to execute (defaults to standard sequence)
            start_from_stage: Stage to start execution from (skips previous stages)
            skip_stages: List of stages to skip
            **kwargs: Additional parameters for the components
            
        Returns:
            Dictionary with project information and output paths
        """
        # Create a unique project ID
        project_id = str(uuid.uuid4())[:8]
        
        # Create a sanitized output name if not provided
        if not output_name:
            sanitized_topic = "".join(c if c.isalnum() or c in " _-" else "_" for c in topic)
            sanitized_topic = sanitized_topic.replace(" ", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"{timestamp}_{sanitized_topic[:50]}"
        
        # Create project directories
        project_dir = os.path.join(self.projects_dir, output_name)
        
        # Check if project directory already exists
        if os.path.exists(project_dir):
            i = 1
            while os.path.exists(f"{project_dir}_{i}"):
                i += 1
            project_dir = f"{project_dir}_{i}"
        
        # Create subdirectories
        script_dir = os.path.join(project_dir, "script")
        audio_dir = os.path.join(project_dir, "audio")
        media_dir = os.path.join(project_dir, "media")
        subtitles_dir = os.path.join(project_dir, "subtitles")
        video_dir = os.path.join(project_dir, "video")
        
        for directory in [project_dir, script_dir, audio_dir, media_dir, subtitles_dir, video_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize project state
        project_state = {
            "project_id": project_id,
            "topic": topic,
            "output_name": output_name,
            "script_type": script_type,
            "duration": duration,
            "tone": tone,
            "voice": voice,
            "keywords": keywords,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "project_dir": project_dir,
            "status": "initializing",
            "progress": 0,
            "stages": {
                "script_generation": {"status": "pending", "output": None, "errors": None},
                "voice_generation": {"status": "pending", "output": None, "errors": None},
                "media_selection": {"status": "pending", "output": None, "errors": None},
                "subtitle_generation": {"status": "pending", "output": None, "errors": None},
                "video_assembly": {"status": "pending", "output": None, "errors": None},
                "audio_processing": {"status": "pending", "output": None, "errors": None},
                "final_export": {"status": "pending", "output": None, "errors": None}
            },
            "dependencies": {
                "script_generation": [],
                "voice_generation": ["script_generation"],
                "media_selection": ["script_generation"],
                "subtitle_generation": ["script_generation", "voice_generation"],
                "video_assembly": ["script_generation", "media_selection"],
                "audio_processing": ["voice_generation"],
                "final_export": ["video_assembly", "audio_processing", "subtitle_generation"]
            },
            "options": kwargs
        }
        
        # Save initial project state
        self._save_project_state(project_state, project_dir)
        
        # Determine stage sequence to run
        if stage_sequence is None:
            stage_sequence = self.default_stage_sequence
        
        # Apply start_from_stage if specified
        if start_from_stage and start_from_stage in stage_sequence:
            start_index = stage_sequence.index(start_from_stage)
            stage_sequence = stage_sequence[start_index:]
        
        # Apply skip_stages if specified
        if skip_stages:
            stage_sequence = [stage for stage in stage_sequence if stage not in skip_stages]
        
        # Flag to track if we should continue pipeline execution
        continue_pipeline = True
        
        # Execute each stage in sequence
        for i, stage in enumerate(stage_sequence):
            # Check dependencies
            dependencies_met = True
            for dep in project_state["dependencies"][stage]:
                if dep not in stage_sequence[:i] and project_state["stages"][dep]["status"] != "completed":
                    logger.warning(f"Skipping {stage} because dependency {dep} is not completed")
                    project_state["stages"][stage]["status"] = "skipped"
                    project_state["stages"][stage]["errors"] = f"Dependency {dep} not completed"
                    dependencies_met = False
                    continue_pipeline = False
                    break
            
            if not dependencies_met:
                self._save_project_state(project_state, project_dir)
                continue
            
            # Calculate progress percentage
            progress_step = 100 / len(stage_sequence)
            project_state["progress"] = min(100, int(i * progress_step))
            project_state["status"] = f"executing_{stage}"
            self._save_project_state(project_state, project_dir)
            
            if continue_pipeline:
                try:
                    # Execute the stage
                    if stage in self.stage_handlers:
                        logger.info(f"Executing stage: {stage}")
                        result = self.stage_handlers[stage](project_state, **kwargs)
                        
                        if result is not None:
                            project_state["stages"][stage]["status"] = "completed"
                            project_state["stages"][stage]["output"] = result
                        else:
                            project_state["stages"][stage]["status"] = "skipped"
                            project_state["stages"][stage]["errors"] = "Stage handler returned None"
                            continue_pipeline = False
                    else:
                        logger.warning(f"Unknown stage: {stage}")
                        project_state["stages"][stage]["status"] = "skipped"
                        project_state["stages"][stage]["errors"] = "Unknown stage"
                
                except Exception as e:
                    logger.error(f"Error in stage {stage}: {str(e)}", exc_info=True)
                    project_state["stages"][stage]["status"] = "failed"
                    project_state["stages"][stage]["errors"] = str(e)
                    continue_pipeline = False
            
            # Save state after each stage
            self._save_project_state(project_state, project_dir)
        
        # Update final project state
        if all(info["status"] == "completed" for stage, info in project_state["stages"].items() 
              if stage in stage_sequence):
            project_state["status"] = "completed"
        elif any(info["status"] == "failed" for stage, info in project_state["stages"].items() 
               if stage in stage_sequence):
            project_state["status"] = "failed"
        else:
            project_state["status"] = "partially_completed"
        
        project_state["progress"] = 100
        project_state["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_project_state(project_state, project_dir)
        
        logger.info(f"Pipeline execution completed with status: {project_state['status']}")
        
        return project_state
    
    # Stage handlers
    
    def _handle_script_generation(self, project_state: Dict, **kwargs) -> Dict:
        """Handle script generation stage."""
        logger.info("Generating script for topic: %s", project_state["topic"])
        
        script_dir = os.path.join(project_state["project_dir"], "script")
        
        # Extract script options from kwargs
        script_options = kwargs.get("script_options", {})
        
        # Generate script
        script_data = self.script_generator.generate_script(
            topic=project_state["topic"],
            script_type=project_state["script_type"],
            duration=project_state["duration"],
            tone=project_state["tone"],
            keywords=project_state["keywords"] or [],
            **script_options
        )
        
        # Save script
        script_path = os.path.join(script_dir, "script.json")
        with open(script_path, "w") as f:
            json.dump(script_data, f, indent=2)
        
        # Also save as text for easy reading
        text_script_path = os.path.join(script_dir, "script.txt")
        self.script_generator.export_to_file(script_data, text_script_path)
        
        # Return output data
        return {
            "script_json": script_path,
            "script_text": text_script_path,
            "word_count": script_data["word_count"],
            "estimated_duration_seconds": script_data["estimated_duration_seconds"],
            "script_data": script_data
        }
    
    def _handle_voice_generation(self, project_state: Dict, **kwargs) -> Optional[Dict]:
        """Handle voice generation stage."""
        logger.info("Generating voiceover")
        
        # Check if script generation was successful
        if project_state["stages"]["script_generation"]["status"] != "completed":
            logger.error("Cannot generate voice without completed script")
            return None
        
        audio_dir = os.path.join(project_state["project_dir"], "audio")
        
        # Get script data
        script_data = project_state["stages"]["script_generation"]["output"]["script_data"]
        
        # Extract voice options from kwargs
        voice_options = kwargs.get("voice_options", {})
        
        # Generate voice
        try:
            voice_data = self.voice_generator.generate_audio_for_script(
                script_data=script_data,
                output_dir=audio_dir,
                voice=project_state["voice"],
                combine=True,
                **voice_options
            )
            
            return voice_data
        except NotImplementedError:
            logger.warning("Voice generation not implemented yet")
            return None
        except Exception as e:
            logger.error(f"Error generating voice: {e}", exc_info=True)
            return None
    
    def _handle_media_selection(self, project_state: Dict, **kwargs) -> Optional[Dict]:
        """Handle media selection stage."""
        logger.info("Selecting media assets")
        
        # Check if script generation was successful
        if project_state["stages"]["script_generation"]["status"] != "completed":
            logger.error("Cannot select media without completed script")
            return None
        
        media_dir = os.path.join(project_state["project_dir"], "media")
        
        # Get script data
        script_data = project_state["stages"]["script_generation"]["output"]["script_data"]
        
        # Extract media options from kwargs
        media_options = kwargs.get("media_options", {})
        
        # Select media
        try:
            media_data = self.media_selector.select_media_for_script(
                script_data=script_data,
                output_dir=media_dir,
                **media_options
            )
            
            return media_data
        except NotImplementedError:
            logger.warning("Media selection not implemented yet")
            return None
        except Exception as e:
            logger.error(f"Error selecting media: {e}", exc_info=True)
            return None
    
    def _handle_subtitle_generation(self, project_state: Dict, **kwargs) -> Optional[Dict]:
        """Handle subtitle generation stage."""
        logger.info("Generating subtitles")
        
        # Check dependencies
        if (project_state["stages"]["script_generation"]["status"] != "completed" or
            project_state["stages"]["voice_generation"]["status"] != "completed"):
            logger.error("Cannot generate subtitles without completed script and voice")
            return None
        
        subtitles_dir = os.path.join(project_state["project_dir"], "subtitles")
        
        # Get data from previous stages
        script_data = project_state["stages"]["script_generation"]["output"]["script_data"]
        voice_data = project_state["stages"]["voice_generation"]["output"]
        
        # Extract subtitle options from kwargs
        subtitle_options = kwargs.get("subtitle_options", {})
        
        # Generate subtitles
        try:
            subtitles_data = self.subtitle_generator.generate_subtitles_for_script(
                script_data=script_data,
                audio_data=voice_data,
                output_dir=subtitles_dir,
                **subtitle_options
            )
            
            return subtitles_data
        except NotImplementedError:
            logger.warning("Subtitle generation not implemented yet")
            return None
        except Exception as e:
            logger.error(f"Error generating subtitles: {e}", exc_info=True)
            return None
    
    def _handle_video_assembly(self, project_state: Dict, **kwargs) -> Optional[Dict]:
        """Handle video assembly stage."""
        logger.info("Assembling video")
        
        # Check dependencies
        if (project_state["stages"]["script_generation"]["status"] != "completed" or
            project_state["stages"]["media_selection"]["status"] != "completed"):
            logger.error("Cannot assemble video without completed script and media")
            return None
        
        video_dir = os.path.join(project_state["project_dir"], "video")
        
        # Get data from previous stages
        script_data = project_state["stages"]["script_generation"]["output"]["script_data"]
        media_data = project_state["stages"]["media_selection"]["output"]
        
        # Get voice data if available
        voice_data = None
        if project_state["stages"]["voice_generation"]["status"] == "completed":
            voice_data = project_state["stages"]["voice_generation"]["output"]
        
        # Get subtitles data if available
        subtitles_data = None
        if project_state["stages"]["subtitle_generation"]["status"] == "completed":
            subtitles_data = project_state["stages"]["subtitle_generation"]["output"]
        
        # Extract video options from kwargs
        video_options = kwargs.get("video_options", {})
        
        # Assemble video
        try:
            video_data = self.video_editor.assemble_video(
                script_data=script_data,
                voice_data=voice_data,
                media_data=media_data,
                subtitles_data=subtitles_data,
                output_dir=video_dir,
                output_name=f"{project_state['output_name']}_draft",
                **video_options
            )
            
            return video_data
        except NotImplementedError:
            logger.warning("Video assembly not implemented yet")
            return None
        except Exception as e:
            logger.error(f"Error assembling video: {e}", exc_info=True)
            return None
    
    def _handle_audio_processing(self, project_state: Dict, **kwargs) -> Optional[Dict]:
        """Handle audio processing stage."""
        logger.info("Processing audio")
        
        # Check if voice generation was successful
        if project_state["stages"]["voice_generation"]["status"] != "completed":
            logger.error("Cannot process audio without completed voice generation")
            return None
        
        audio_processed_dir = os.path.join(project_state["project_dir"], "audio_processed")
        os.makedirs(audio_processed_dir, exist_ok=True)
        
        # Get voice data
        voice_data = project_state["stages"]["voice_generation"]["output"]
        
        # Extract audio options from kwargs
        audio_options = kwargs.get("audio_options", {})
        
        # Process audio
        try:
            processed_audio = self.audio_processor.process_audio(
                voice_audio_path=voice_data["full_audio"],
                output_dir=audio_processed_dir,
                **audio_options
            )
            
            return processed_audio
        except NotImplementedError:
            logger.warning("Audio processing not implemented yet")
            return None
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            return None
    
    def _handle_final_export(self, project_state: Dict, **kwargs) -> Optional[Dict]:
        """Handle final video export stage."""
        logger.info("Creating final video export")
        
        # Check dependencies
        if project_state["stages"]["video_assembly"]["status"] != "completed":
            logger.error("Cannot create final export without completed video assembly")
            return None
        
        video_dir = os.path.join(project_state["project_dir"], "video")
        
        # Get video data
        video_data = project_state["stages"]["video_assembly"]["output"]
        
        # Get processed audio if available, otherwise use voice audio
        audio_path = None
        if project_state["stages"]["audio_processing"]["status"] == "completed":
            audio_path = project_state["stages"]["audio_processing"]["output"]["output_path"]
        elif project_state["stages"]["voice_generation"]["status"] == "completed":
            audio_path = project_state["stages"]["voice_generation"]["output"]["full_audio"]
        
        # Extract export options from kwargs
        export_options = kwargs.get("export_options", {})
        
        # Create final video
        try:
            final_video = self.video_editor.create_final_video(
                draft_video_path=video_data["output_path"],
                enhanced_audio_path=audio_path,
                output_dir=video_dir,
                output_name=project_state["output_name"],
                **export_options
            )
            
            return final_video
        except NotImplementedError:
            logger.warning("Final video export not implemented yet")
            return None
        except Exception as e:
            logger.error(f"Error creating final video: {e}", exc_info=True)
            return None
    
    # Project management methods
    
    def _save_project_state(self, project_state: Dict, project_dir: str) -> None:
        """Save the current project state to a JSON file."""
        state_path = os.path.join(project_dir, "project_state.json")
        with open(state_path, "w") as f:
            json.dump(project_state, f, indent=2)
    
    def load_project(self, project_dir: str) -> Dict:
        """
        Load a project from a directory.
        
        Args:
            project_dir: Path to the project directory
            
        Returns:
            Project state dictionary
        """
        state_path = os.path.join(project_dir, "project_state.json")
        try:
            with open(state_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Project not found at {project_dir}")
    
    def resume_project(self, project_dir: str, 
                      start_from_stage: Optional[str] = None,
                      skip_stages: Optional[List[str]] = None) -> Dict:
        """
        Resume a previously started project.
        
        Args:
            project_dir: Path to the project directory
            start_from_stage: Stage to start execution from (skips previous stages)
            skip_stages: List of stages to skip
            
        Returns:
            Updated project state
        """
        # Load project state
        project_state = self.load_project(project_dir)
        
        # Determine which stages need to be executed
        completed_stages = [
            stage for stage, info in project_state["stages"].items()
            if info["status"] == "completed"
        ]
        
        # Find first incomplete stage if not specified
        if start_from_stage is None:
            for stage in self.default_stage_sequence:
                if stage not in completed_stages:
                    start_from_stage = stage
                    break
        
        # If all stages are complete, return the project state
        if start_from_stage is None:
            logger.info("Project is already completed.")
            return project_state
        
        # Resume project execution
        logger.info(f"Resuming project from stage: {start_from_stage}")
        
        return self.generate_video(
            topic=project_state["topic"],
            output_name=project_state["output_name"],
            script_type=project_state["script_type"],
            duration=project_state["duration"],
            tone=project_state["tone"],
            voice=project_state["voice"],
            keywords=project_state["keywords"],
            start_from_stage=start_from_stage,
            skip_stages=skip_stages,
            **project_state.get("options", {})
        )
    
    def list_projects(self) -> List[Dict]:
        """
        List all projects in the projects directory.
        
        Returns:
            List of project summary dictionaries
        """
        projects = []
        
        for project_name in os.listdir(self.projects_dir):
            project_dir = os.path.join(self.projects_dir, project_name)
            
            if os.path.isdir(project_dir):
                try:
                    # Load project state
                    project_state = self.load_project(project_dir)
                    
                    # Create summary
                    summary = {
                        "project_id": project_state.get("project_id"),
                        "name": project_name,
                        "topic": project_state.get("topic"),
                        "created_at": project_state.get("created_at"),
                        "completed_at": project_state.get("completed_at"),
                        "status": project_state.get("status"),
                        "progress": project_state.get("progress"),
                        "directory": project_dir
                    }
                    
                    projects.append(summary)
                except Exception as e:
                    # Skip invalid projects
                    logger.warning(f"Error loading project {project_name}: {e}")
        
        # Sort by creation date (newest first)
        return sorted(projects, 
                     key=lambda p: p.get("created_at", ""), 
                     reverse=True)
    
    def get_project_details(self, project_id: str) -> Optional[Dict]:
        """
        Get detailed information about a project.
        
        Args:
            project_id: Project ID to look for
            
        Returns:
            Project state dictionary or None if not found
        """
        for project in self.list_projects():
            if project.get("project_id") == project_id:
                return self.load_project(project["directory"])
        
        return None
    
    def delete_project(self, project_id: str, confirm: bool = True) -> bool:
        """
        Delete a project and all its files.
        
        Args:
            project_id: Project ID to delete
            confirm: Whether to require confirmation
            
        Returns:
            True if project was deleted, False otherwise
        """
        project = self.get_project_details(project_id)
        
        if not project:
            logger.error(f"Project {project_id} not found")
            return False
        
        project_dir = project.get("project_dir")
        
        if not project_dir or not os.path.exists(project_dir):
            logger.error(f"Project directory not found for project {project_id}")
            return False
        
        if confirm:
            logger.warning(f"Deleting project {project_id} from {project_dir}")
            # In a real application, you would prompt for confirmation here
        
        try:
            shutil.rmtree(project_dir)
            logger.info(f"Project {project_id} deleted")
            return True
        except Exception as e:
            logger.error(f"Error deleting project {project_id}: {e}")
            return False
    
    def export_project(self, project_id: str, export_dir: str) -> Optional[str]:
        """
        Export a project to a zip file.
        
        Args:
            project_id: Project ID to export
            export_dir: Directory to save the export
            
        Returns:
            Path to the exported zip file or None if export failed
        """
        project = self.get_project_details(project_id)
        
        if not project:
            logger.error(f"Project {project_id} not found")
            return None
        
        project_dir = project.get("project_dir")
        
        if not project_dir or not os.path.exists(project_dir):
            logger.error(f"Project directory not found for project {project_id}")
            return None
        
        try:
            os.makedirs(export_dir, exist_ok=True)
            
            # Create a zip file
            import zipfile
            from datetime import datetime
            
            export_name = f"{project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            export_path = os.path.join(export_dir, export_name)
            
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(project_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.dirname(project_dir))
                        zipf.write(file_path, arcname)
            
            logger.info(f"Project {project_id} exported to {export_path}")
            return export_path
        except Exception as e:
            logger.error(f"Error exporting project {project_id}: {e}")
            return None
    
    def import_project(self, zip_path: str) -> Optional[str]:
        """
        Import a project from a zip file.
        
        Args:
            zip_path: Path to the zip file
            
        Returns:
            Imported project ID or None if import failed
        """
        try:
            import zipfile
            import tempfile
            
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract the zip file
                with zipfile.ZipFile(zip_path, 'r') as zipf:
                    zipf.extractall(temp_dir)
                
                # Find the project state file
                for root, _, files in os.walk(temp_dir):
                    if "project_state.json" in files:
                        # Load the project state
                        with open(os.path.join(root, "project_state.json"), "r") as f:
                            project_state = json.load(f)
                        
                        # Create a new project name
                        project_id = project_state.get("project_id", "imported")
                        output_name = f"imported_{project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        # Create the project directory
                        project_dir = os.path.join(self.projects_dir, output_name)
                        os.makedirs(project_dir, exist_ok=True)
                        
                        # Copy project files
                        for item in os.listdir(root):
                            s = os.path.join(root, item)
                            d = os.path.join(project_dir, item)
                            if os.path.isdir(s):
                                shutil.copytree(s, d, dirs_exist_ok=True)
                            else:
                                shutil.copy2(s, d)
                        
                        # Update project state
                        project_state["project_dir"] = project_dir
                        project_state["imported_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Save project state
                        with open(os.path.join(project_dir, "project_state.json"), "w") as f:
                            json.dump(project_state, f, indent=2)
                        
                        logger.info(f"Project imported to {project_dir}")
                        return project_state.get("project_id")
                
                logger.error("No project state file found in the zip file")
                return None
        except Exception as e:
            logger.error(f"Error importing project: {e}")
            return None


# Dummy classes for components that are not yet implemented

class DummyVoiceGenerator:
    """Placeholder for the VoiceGenerator when it's not available."""
    
    def generate_audio_for_script(self, **kwargs):
        """Placeholder for voice generation."""
        raise NotImplementedError("VoiceGenerator is not yet implemented")
    
    def get_available_voices(self):
        """Placeholder for getting available voices."""
        return []


class DummyMediaSelector:
    """Placeholder for the MediaSelector when it's not available."""
    
    def select_media_for_script(self, **kwargs):
        """Placeholder for media selection."""
        raise NotImplementedError("MediaSelector is not yet implemented")


class DummyVideoEditor:
    """Placeholder for the VideoEditor when it's not available."""
    
    def assemble_video(self, **kwargs):
        """Placeholder for video assembly."""
        raise NotImplementedError("VideoEditor is not yet implemented")
    
    def create_final_video(self, **kwargs):
        """Placeholder for final video creation."""
        raise NotImplementedError("VideoEditor is not yet implemented")


class DummySubtitleGenerator:
    """Placeholder for the SubtitleGenerator when it's not available."""
    
    def generate_subtitles_for_script(self, **kwargs):
        """Placeholder for subtitle generation."""
        raise NotImplementedError("SubtitleGenerator is not yet implemented")


class DummyAudioProcessor:
    """Placeholder for the AudioProcessor when it's not available."""
    
    def process_audio(self, **kwargs):
        """Placeholder for audio processing."""
        raise NotImplementedError("AudioProcessor is not yet implemented")
