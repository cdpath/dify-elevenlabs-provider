# ElevenLabs Provider for Dify

A Dify plugin for integrating ElevenLabs Text-to-Speech and Speech-to-Text services.

## Features

- **Text-to-Speech (TTS)**: Convert text to high-quality, natural-sounding speech using ElevenLabs' advanced voice synthesis technology.
- **Speech-to-Text (STT)**: Transcribe audio to text with accurate speech recognition.

## Setup

1. Install the plugin in your Dify instance
2. Configure the plugin with your ElevenLabs API key
3. Use the ElevenLabs models in your Dify applications

## Requirements

- An [ElevenLabs account](https://elevenlabs.io/)
- ElevenLabs API key, available from your [ElevenLabs account page](https://elevenlabs.io/app/settings/api-keys)

## Models

### Text-to-Speech

- **Model**: `eleven_monolingual_v1`
- **Voices**: Rachel, Domi, Bella, Antoni, Thomas, Josh, Arnold, Adam, Sam
- **Default Voice**: Rachel

### Speech-to-Text

- **Model**: `speech-to-text`
- **Mode**: transcription
- **Supported File Extensions**: mp3, mp4, mpeg, mpga, m4a, wav, webm
- **File Upload Limit**: 25MB

## License

This plugin is provided under the MIT license.

