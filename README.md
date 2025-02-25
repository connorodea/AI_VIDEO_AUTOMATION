# AI-Powered YouTube Video Automation Software

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A comprehensive AI-driven system for automatically generating professional-quality YouTube videos from just a topic or keyword. This software leverages cutting-edge AI technologies to handle the entire video production workflow: script writing, voiceover generation, media selection, video editing, and final rendering.

## ✨ Features

- **Script Generation**: Creates engaging, structured scripts using OpenAI GPT-4 or Claude
- **AI Voiceovers**: Converts scripts to natural-sounding speech using ElevenLabs, Google TTS, or Amazon Polly
- **Media Selection**: Automatically sources relevant visuals from Pexels, Pixabay, and Unsplash 
- **AI Image Generation**: Creates custom visuals using DALL-E or Stable Diffusion when needed
- **Automated Video Editing**: Handles scene transitions, Ken Burns effects, and timing
- **Audio Processing**: Adds background music, enhances voice quality, and normalizes audio
- **Subtitle Generation**: Creates synchronized captions for better accessibility
- **End-to-end Automation**: From script to final YouTube-ready video with minimal human input

## 🛠️ Technical Architecture

The system is built with a modular architecture that separates concerns and allows for easy extension:

```
ai_video_generator/
├── core/                  # Core pipeline components
│   ├── pipeline.py        # Main orchestration
│   ├── script_generator.py
│   ├── voice_generator.py
│   ├── media_selector.py
│   ├── video_editor.py
│   ├── subtitle_generator.py
│   └── audio_processor.py
│
├── models/                # AI model interfaces
│   ├── llm_wrapper.py     # Language models (GPT-4, Claude)
│   ├── tts_wrapper.py     # Text-to-speech models
│   └── image_generator.py # Image generation (DALL-E, SD)
│
├── data_providers/        # External API integrations
│   ├── stock_video_api.py # Pexels, Pixabay
│   ├── stock_image_api.py # Unsplash, etc.
│   └── music_library_api.py
│
├── utils/                 # Utility functions
├── ui/                    # User interfaces
│   ├── cli/               # Command-line interface
│   └── web/               # Web interface
│
└── config/                # Configuration files
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ 
- FFmpeg installed on your system
- API keys for services (setup instructions below)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ai-video-generator.git
   cd ai-video-generator
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up configuration:
   ```bash
   # Edit config/api_keys.json with your API keys
   # Edit config/default_settings.json for customization (optional)
   ```

### Setting Up API Keys

The system requires API keys for various services. You can set these up in the `config/api_keys.json` file or through the web UI settings page:

- **OpenAI API**: For script generation (GPT-4) and image generation (DALL-E)
- **ElevenLabs API**: For realistic voiceovers 
- **Pexels API**: For stock videos and images
- **Pixabay API**: For additional stock media
- **Unsplash API**: For high-quality stock images

All APIs offer free tiers that are suitable for testing the system.

## 📊 Usage

### Command Line Interface

Create a video with basic parameters:
```bash
python main.py create "The Future of Artificial Intelligence" --type educational --duration 5 --voice adam
```

List all your projects:
```bash
python main.py list
```

Get details for a specific project:
```bash
python main.py details <project_id>
```

Resume a failed or incomplete project:
```bash
python main.py resume <project_path> --start-from media_selection
```

### Web Interface

Start the web server:
```bash
cd ui/web
python app.py
```

Then open your browser and navigate to http://localhost:5000

The web interface provides:
- Project creation with advanced options
- Real-time progress tracking
- Project management dashboard
- Settings configuration
- Video preview and download

## 🎬 Example Output

Here's what you can expect from the generated videos:

1. **Educational Content**: Clear, structured explanations with relevant visuals, optimized for learning
2. **Entertainment Videos**: Engaging, dynamic content with smooth transitions and pacing
3. **Marketing Material**: Professional promotional videos with consistent branding

## 🔧 Advanced Configuration

The system is highly configurable through the `config/default_settings.json` file:

- **Video quality**: Resolution, frame rate, and bitrate
- **Ken Burns settings**: Animation speed and zoom levels
- **Audio processing**: Music volume, voice enhancement
- **Visual styles**: Color grading, overlays, transitions

## 🔄 Development Workflow

1. The system starts by generating a script based on your topic
2. It converts the script to audio using the selected voice
3. For each segment, it finds relevant media from stock providers
4. It generates custom images for concepts that lack stock media
5. The video editor assembles everything with transitions and effects
6. Audio processing enhances sound quality and adds background music
7. The final video is rendered in YouTube-ready format

## 🤝 Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md) for details.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- OpenAI, Anthropic, and ElevenLabs for their powerful AI APIs
- Pexels, Pixabay, and Unsplash for their stock media APIs
- The FFmpeg team for their incredible video processing library

## 📧 Contact

For questions or feedback, please open an issue or contact the maintainers directly.

---

*Note: This software is intended for creating legitimate content. Please respect copyright and terms of service for all integrated APIs.*
