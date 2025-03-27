from collections.abc import Generator
from typing import Optional, Dict, Type
import io
import logging
import json

from dify_plugin import TTSModel
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeBadRequestError,
    InvokeAuthorizationError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings


class DifyElevenlabsProviderText2SpeechModel(TTSModel):
    """
    Model class for ElevenLabs Text to Speech model.
    """

    @property
    def _invoke_error_mapping(self) -> Dict[Type[InvokeError], list]:
        """
        Map ElevenLabs API exceptions to Dify plugin error types
        """
        import requests.exceptions

        return {
            InvokeBadRequestError: [ValueError, requests.exceptions.HTTPError, json.JSONDecodeError],
            InvokeConnectionError: [requests.exceptions.ConnectionError, requests.exceptions.RequestException],
            InvokeServerUnavailableError: [requests.exceptions.Timeout],
            InvokeAuthorizationError: [PermissionError],
        }

    def _invoke(
        self,
        model: str,
        tenant_id: str,
        credentials: dict,
        content_text: str,
        voice: str,
        user: Optional[str] = None,
    ) -> bytes | Generator[bytes, None, None]:
        """
        _invoke text2speech model

        :param model: model name
        :param tenant_id: user tenant id
        :param credentials: model credentials
        :param content_text: text content to be translated
        :param voice: model timbre
        :param user: unique user id
        :return: text translated to audio file or stream of audio chunks
        """
        # Get API key from credentials
        if not isinstance(credentials, dict):
            raise CredentialsValidateFailedError("Credentials must be a dictionary")
            
        api_key = credentials.get('api_key')
        if not api_key:
            raise CredentialsValidateFailedError("API key is required")
        
        # Initialize ElevenLabs client
        client = ElevenLabs(api_key=api_key)
        
        try:
            # Get available voices
            available_voices = client.voices.get_all().voices
            
            # Find the requested voice or use the first available voice
            voice_id = None
            for v in available_voices:
                if v.name.lower() == voice.lower():
                    voice_id = v.voice_id
                    break
            
            if not voice_id and available_voices:
                # Use the first available voice if requested voice not found
                voice_id = available_voices[0].voice_id
            
            if not voice_id:
                raise CredentialsValidateFailedError("No voices available")
            
            # Use model parameter or default to eleven_turbo_v2
            model_id = model or "eleven_turbo_v2"
            
            # Convert input text to ensure proper encoding for non-ASCII characters
            if content_text and not isinstance(content_text, str):
                content_text = str(content_text)
                
            # Generate audio stream
            response = client.text_to_speech.convert(
                voice_id=voice_id,
                text=content_text,
                model_id=model_id,
                output_format="mp3_44100_128",
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                    style=0.0,
                    use_speaker_boost=True
                )
            )
            
            # Determine if we need to return a stream or full bytes based on response type
            if hasattr(response, '__iter__') and not isinstance(response, bytes):
                # Return generator that yields chunks
                def stream_generator():
                    for chunk in response:
                        if chunk:
                            yield chunk
                
                return stream_generator()
            else:
                # For non-iterable responses, return as bytes
                return response
            
        except Exception as ex:
            # Map the exception to the appropriate error type
            error_message = f"Text-to-speech generation failed: {str(ex)}"
            
            if isinstance(ex, json.JSONDecodeError):
                logging.error(f"JSON decode error: {str(ex)}")
                raise InvokeBadRequestError(f"Failed to process response: {str(ex)}")
            elif "401" in str(ex) or "403" in str(ex):
                raise InvokeAuthorizationError(error_message)
            elif "429" in str(ex):
                raise InvokeRateLimitError(error_message)
            elif "5" in str(ex) and len(str(ex)) > 0 and str(ex)[0] == "5":
                raise InvokeServerUnavailableError(error_message)
            else:
                logging.error(f"Unexpected error in TTS invoke: {str(ex)}", exc_info=True)
                raise InvokeBadRequestError(error_message)

    def validate_credentials(
        self, model: str, credentials: dict, user: Optional[str] = None
    ) -> None:
        """
        validate credentials text2speech model

        :param model: model name
        :param credentials: model credentials
        :param user: unique user id
        :return: text translated to audio file
        """
        try:
            api_key = credentials.get('api_key')
            if not api_key:
                raise CredentialsValidateFailedError("API key is required")
            
            # Initialize ElevenLabs client
            client = ElevenLabs(api_key=api_key)
            
            # Test by fetching voices
            client.voices.get_all()
            
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))
