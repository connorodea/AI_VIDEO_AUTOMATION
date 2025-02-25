# ui/web/app.py

import os
import sys
import json
import time
import threading
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

# Add parent directory to path to make imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(parent_dir)

# Import Flask and core components
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from core.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(parent_dir, "web_ui.log"))
    ]
)
logger = logging.getLogger('web_ui')

# Initialize Flask app
app = Flask(__name__, 
           static_folder=os.path.join(current_dir, "static"),
           template_folder=os.path.join(current_dir, "templates"))

# Global variables
pipeline = Pipeline()
active_projects = {}  # Dictionary to track active projects

@app.route('/')
def index():
    """Render the main page."""
    # Get list of projects
    projects = pipeline.list_projects()
    
    return render_template('index.html', projects=projects, active_projects=active_projects)

@app.route('/create', methods=['GET', 'POST'])
def create_project():
    """Handle project creation."""
    if request.method == 'POST':
        # Get form data
        topic = request.form.get('topic')
        script_type = request.form.get('script_type', 'standard')
        duration = int(request.form.get('duration', 5))
        tone = request.form.get('tone')
        voice = request.form.get('voice')
        keywords = request.form.get('keywords', '')
        
        if keywords:
            keywords = [k.strip() for k in keywords.split(',')]
        else:
            keywords = None
        
        # Generate a random project ID for tracking
        project_id = f"proj_{int(time.time())}"
        
        # Add to active projects
        active_projects[project_id] = {
            "topic": topic,
            "status": "starting",
            "progress": 0,
            "time_started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "output_dir": None
        }
        
        # Start a thread to run the pipeline
        thread = threading.Thread(
            target=run_pipeline_in_thread,
            args=(project_id, topic, script_type, duration, tone, voice, keywords)
        )
        thread.daemon = True
        thread.start()
        
        return redirect(url_for('project_status', project_id=project_id))
    
    return render_template('create.html')

@app.route('/project/<project_id>')
def project_status(project_id):
    """Display project status."""
    # Check if it's an active project
    if project_id in active_projects:
        return render_template('project_status.html', 
                              project_id=project_id, 
                              project=active_projects[project_id])
    
    # Check if it's a completed project
    project = pipeline.get_project_details(project_id)
    if project:
        return render_template('project_details.html', project=project)
    
    # Project not found
    return render_template('error.html', message=f"Project {project_id} not found")

@app.route('/api/project/<project_id>/status')
def api_project_status(project_id):
    """API endpoint to get project status."""
    # Check if it's an active project
    if project_id in active_projects:
        return jsonify(active_projects[project_id])
    
    # Check if it's a completed project
    project = pipeline.get_project_details(project_id)
    if project:
        return jsonify({
            "status": project.get("status"),
            "progress": project.get("progress"),
            "time_started": project.get("created_at"),
            "time_completed": project.get("completed_at"),
            "topic": project.get("topic"),
            "output_dir": project.get("project_dir")
        })
    
    # Project not found
    return jsonify({"error": "Project not found"}), 404

@app.route('/api/projects')
def api_projects():
    """API endpoint to get list of projects."""
    projects = pipeline.list_projects()
    return jsonify(projects)

@app.route('/api/project/<project_id>/details')
def api_project_details(project_id):
    """API endpoint to get project details."""
    project = pipeline.get_project_details(project_id)
    if project:
        return jsonify(project)
    return jsonify({"error": "Project not found"}), 404

@app.route('/api/project/<project_id>/delete', methods=['POST'])
def api_delete_project(project_id):
    """API endpoint to delete a project."""
    success = pipeline.delete_project(project_id, confirm=False)
    return jsonify({"success": success})

@app.route('/api/project/<project_id>/resume', methods=['POST'])
def api_resume_project(project_id):
    """API endpoint to resume a project."""
    try:
        # Get form data
        start_from = request.form.get('start_from')
        skip_stages = request.form.get('skip_stages')
        
        if skip_stages:
            skip_stages = skip_stages.split(',')
        else:
            skip_stages = None
        
        # Generate a tracking ID
        tracking_id = f"resume_{int(time.time())}"
        
        # Get project details
        project = pipeline.get_project_details(project_id)
        
        # Add to active projects
        active_projects[tracking_id] = {
            "topic": project.get("topic"),
            "status": "resuming",
            "progress": project.get("progress", 0),
            "time_started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "output_dir": project.get("project_dir"),
            "original_project_id": project_id
        }
        
        # Start a thread to resume the pipeline
        thread = threading.Thread(
            target=resume_pipeline_in_thread,
            args=(tracking_id, project_id, start_from, skip_stages)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({"success": True, "tracking_id": tracking_id})
    except Exception as e:
        logger.error(f"Error resuming project: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Handle settings configuration."""
    config_path = os.path.join(parent_dir, "config", "default_settings.json")
    api_keys_path = os.path.join(parent_dir, "config", "api_keys.json")
    
    if request.method == 'POST':
        # Handle API keys update
        if 'update_api_keys' in request.form:
            try:
                api_keys = {}
                
                # Get all form fields that start with "api_key_"
                for key, value in request.form.items():
                    if key.startswith("api_key_") and value:
                        api_name = key[8:]  # Remove "api_key_" prefix
                        api_keys[api_name] = value
                
                # Save to file
                with open(api_keys_path, "w") as f:
                    json.dump(api_keys, f, indent=2)
                
                return render_template('settings.html', message="API keys updated successfully")
            except Exception as e:
                logger.error(f"Error updating API keys: {str(e)}")
                return render_template('settings.html', error=f"Error updating API keys: {str(e)}")
        
        # Handle config update
        elif 'update_config' in request.form:
            try:
                # Load existing config
                with open(config_path, "r") as f:
                    config = json.load(f)
                
                # Update script settings
                if "script" not in config:
                    config["script"] = {}
                
                config["script"]["tone"] = request.form.get("script_tone", "informative")
                config["script"]["include_timestamps"] = "script_timestamps" in request.form
                config["script"]["timestamp_interval"] = int(request.form.get("script_timestamp_interval", 30))
                
                # Update voiceover settings
                if "voiceover" not in config:
                    config["voiceover"] = {}
                
                config["voiceover"]["provider"] = request.form.get("voice_provider", "elevenlabs")
                config["voiceover"]["default_voice"] = request.form.get("voice_default", "adam")
                config["voiceover"]["speech_rate"] = float(request.form.get("voice_rate", 1.0))
                
                # Update media settings
                if "visual_assets" not in config:
                    config["visual_assets"] = {}
                
                providers = []
                for p in ["pexels", "pixabay", "unsplash"]:
                    if f"media_provider_{p}" in request.form:
                        providers.append(p)
                
                config["visual_assets"]["providers"] = providers
                config["visual_assets"]["prefer_video_over_image"] = "media_prefer_video" in request.form
                config["visual_assets"]["max_assets_per_segment"] = int(request.form.get("media_max_assets", 3))
                
                # Save to file
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                
                return render_template('settings.html', message="Settings updated successfully")
            except Exception as e:
                logger.error(f"Error updating settings: {str(e)}")
                return render_template('settings.html', error=f"Error updating settings: {str(e)}")
    
    # Load current settings
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception:
        config = {}
    
    # Load current API keys
    try:
        with open(api_keys_path, "r") as f:
            api_keys = json.load(f)
    except Exception:
        api_keys = {}
    
    return render_template('settings.html', config=config, api_keys=api_keys)

@app.route('/media/<path:filename>')
def serve_media(filename):
    """Serve media files."""
    return send_from_directory(os.path.join(parent_dir, "output"), filename)

def run_pipeline_in_thread(project_id, topic, script_type, duration, tone, voice, keywords):
    """Run the pipeline in a separate thread."""
    try:
        # Update status
        active_projects[project_id]["status"] = "running"
        
        # Run the pipeline
        project_state = pipeline.generate_video(
            topic=topic,
            script_type=script_type,
            duration=duration,
            tone=tone,
            voice=voice,
            keywords=keywords
        )
        
        # Update status
        active_projects[project_id]["status"] = project_state["status"]
        active_projects[project_id]["progress"] = project_state["progress"]
        active_projects[project_id]["project_id"] = project_state["project_id"]
        active_projects[project_id]["output_dir"] = project_state["project_dir"]
        active_projects[project_id]["time_completed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Keep the project in active projects list for 1 hour, then clean up
        def cleanup():
            time.sleep(3600)  # 1 hour
            if project_id in active_projects:
                del active_projects[project_id]
        
        cleanup_thread = threading.Thread(target=cleanup)
        cleanup_thread.daemon = True
        cleanup_thread.start()
    
    except Exception as e:
        logger.error(f"Error in pipeline thread: {str(e)}")
        active_projects[project_id]["status"] = "failed"
        active_projects[project_id]["error"] = str(e)

def resume_pipeline_in_thread(tracking_id, project_id, start_from, skip_stages):
    """Resume a pipeline in a separate thread."""
    try:
        # Update status
        active_projects[tracking_id]["status"] = "running"
        
        # Try to get project path
        project = pipeline.get_project_details(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        project_dir = project.get("project_dir")
        if not project_dir:
            raise ValueError(f"Project directory not found for project {project_id}")
        
        # Resume the pipeline
        project_state = pipeline.resume_project(
            project_dir=project_dir,
            start_from_stage=start_from,
            skip_stages=skip_stages
        )
        
        # Update status
        active_projects[tracking_id]["status"] = project_state["status"]
        active_projects[tracking_id]["progress"] = project_state["progress"]
        active_projects[tracking_id]["project_id"] = project_state["project_id"]
        active_projects[tracking_id]["output_dir"] = project_state["project_dir"]
        active_projects[tracking_id]["time_completed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Keep the project in active projects list for 1 hour, then clean up
        def cleanup():
            time.sleep(3600)  # 1 hour
            if tracking_id in active_projects:
                del active_projects[tracking_id]
        
        cleanup_thread = threading.Thread(target=cleanup)
        cleanup_thread.daemon = True
        cleanup_thread.start()
    
    except Exception as e:
        logger.error(f"Error in resume thread: {str(e)}")
        active_projects[tracking_id]["status"] = "failed"
        active_projects[tracking_id]["error"] = str(e)

if __name__ == '__main__':
    # Ensure necessary directories and config exist
    os.makedirs(os.path.join(parent_dir, "config"), exist_ok=True)
    os.makedirs(os.path.join(parent_dir, "output"), exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
