# Setting Up YouTube API Integration

This guide will walk you through the process of setting up the YouTube API integration for AI Video Generator, allowing the system to automatically upload generated videos to your YouTube channel.

## Prerequisites

1. A Google Account with a YouTube channel
2. A Google Cloud Platform (GCP) project
3. API credentials for the YouTube Data API

## Step 1: Create a Google Cloud Platform Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top of the page
3. Click on "New Project"
4. Name your project (e.g., "AI Video Generator") and click "Create"
5. Make sure your new project is selected in the project dropdown

## Step 2: Enable the YouTube Data API

1. In the Google Cloud Console, go to the Navigation Menu (☰) > "APIs & Services" > "Library"
2. Search for "YouTube Data API v3"
3. Click on the API in the results
4. Click "Enable"

## Step 3: Create OAuth Credentials

1. In the Google Cloud Console, go to the Navigation Menu (☰) > "APIs & Services" > "Credentials"
2. Click "Create Credentials" and select "OAuth client ID"
3. If prompted, configure the OAuth consent screen:
   - User Type: External
   - App name: "AI Video Generator"
   - User support email: your email
   - Developer contact information: your email
   - Scopes: add the YouTube Data API scopes (for testing, you can add them later)
   - Click "Save and Continue" through the steps, then "Back to Dashboard"

4. Now, create the OAuth client ID:
   - Application type: "Desktop app"
   - Name: "AI Video Generator Desktop Client"
   - Click "Create"

5. You'll receive your client ID and client secret. Click "Download JSON"
6. Rename the downloaded file to `client_secrets.json`

## Step 4: Configure AI Video Generator

### Method 1: Place the credentials file directly

1. Create a directory called `config` in your AI Video Generator root directory (if it doesn't exist already)
2. Move the `client_secrets.json` file into the `config` directory

### Method 2: Use the helper function in the code

```python
from data_providers.youtube_api import generate_client_secrets_file

# Replace with your actual client ID and secret
client_id = "YOUR_CLIENT_ID"
client_secret = "YOUR_CLIENT_SECRET"

# Generate the client secrets file
generate_client_secrets_file(client_id, client_secret)
```

## Step 5: Authentication

The first time you run the YouTube upload feature, you'll be prompted to authenticate:

1. A browser window will open with a Google sign-in page
2. Sign in with the Google account associated with your YouTube channel
3. Review the permissions and click "Allow"

After successful authentication, a token will be saved to `config/youtube_token.json` for future use.

## Using the YouTube Upload Feature

### Via the Command Line

```bash
python main.py create "My Video Topic" --type educational --duration 3 --youtube
```

You can also specify YouTube options:

```bash
python main.py create "My Video Topic" --youtube-options '{"privacy": "unlisted", "tags": ["AI", "Tutorial"]}'
```

### Via the Web Interface

1. Go to the Create Video page
2. Fill in your video details
3. In the advanced options, check "Upload to YouTube when complete"
4. Optionally configure privacy settings, playlist, etc.

## Troubleshooting

### Authentication Issues

If you encounter authentication issues:

1. Delete the `config/youtube_token.json` file
2. Run the upload process again to trigger a new authentication flow

### API Quota Limits

YouTube API has daily quota limits. If you exceed them:

1. Wait until the quota resets (usually 24 hours)
2. Consider creating a new GCP project for additional quota

### Verification Requirements

For production use, Google may require your app to go through verification:

1. In the Google Cloud Console, go to "APIs & Services" > "OAuth consent screen"
2. Click "Edit App"
3. Follow the verification process

For testing and personal use, you can use the app without verification, but it will have a "Unverified App" warning during authentication.

## Advanced Configuration

You can configure additional YouTube settings in the `config/default_settings.json` file:

```json
{
  "youtube": {
    "credentials_dir": "config",
    "client_secrets_file": "client_secrets.json",
    "category": "22",
    "privacy": "private",
    "generate_tags": true,
    "max_tags": 15,
    "auto_upload": false
  }
}
```

- `category`: YouTube category ID (e.g., "22" for "People & Blogs")
- `privacy`: Default privacy setting ("private", "unlisted", or "public")
- `generate_tags`: Whether to automatically generate tags from the script
- `max_tags`: Maximum number of tags to include
- `auto_upload`: Whether to upload to YouTube by default
