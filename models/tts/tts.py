from collections.abc import Generator
from typing import Optional, Dict, Type, Union, Any
import logging
from dataclasses import dataclass
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
import requests.exceptions

logger = logging.getLogger(__name__)

@dataclass
class TTSResponse:
    """Data class for TTS response"""
    audio_data: Union[bytes, Generator[bytes, None, None]]

class DifyElevenlabsProviderText2SpeechModel(TTSModel):
    """Model class for ElevenLabs Text to Speech model."""

    DEFAULT_VOICE_SETTINGS = VoiceSettings(
        stability=0.5,
        similarity_boost=0.75,
        style=0.0,
        use_speaker_boost=True
    )

    @property
    def _invoke_error_mapping(self) -> Dict[Type[InvokeError], list[Type[Exception]]]:
        """Map ElevenLabs API exceptions to Dify plugin error types."""
        return {
            InvokeBadRequestError: [ValueError, requests.exceptions.HTTPError, json.JSONDecodeError],
            InvokeConnectionError: [requests.exceptions.ConnectionError, requests.exceptions.RequestException],
            InvokeServerUnavailableError: [requests.exceptions.Timeout],
            InvokeAuthorizationError: [PermissionError],
        }

    def _validate_credentials(self, credentials: Any) -> str:
        """Validate and extract API key from credentials.

        Args:
            credentials: The credentials dictionary containing the API key.

        Returns:
            str: The validated API key.

        Raises:
            CredentialsValidateFailedError: If credentials are invalid.
        """
        if not isinstance(credentials, dict):
            raise CredentialsValidateFailedError("Credentials must be a dictionary")

        api_key = credentials.get('api_key')
        if not api_key:
            raise CredentialsValidateFailedError("API key is required")

        return api_key

    def _invoke(
        self,
        model: str,
        tenant_id: str,
        credentials: dict,
        content_text: str,
        voice: str,
        user: Optional[str] = None,
    ) -> Union[bytes, Generator[bytes, None, None]]:
        """Invoke text2speech model.

        Args:
            model: Model name
            tenant_id: User tenant id
            credentials: Model credentials
            content_text: Text content to be translated
            voice: voice_id of elevenlabs
            user: Unique user id

        Returns:
            Union[bytes, Generator[bytes, None, None]]: Audio data or stream

        Raises:
            Various InvokeError types based on the error encountered
        """
        api_key = self._validate_credentials(credentials)
        client = ElevenLabs(api_key=api_key)
        
        try:
            response = client.text_to_speech.convert(
                voice_id=voice,
                text=content_text,
                model_id=model,
                output_format="mp3_44100_128",
                voice_settings=self.DEFAULT_VOICE_SETTINGS
            )

            if hasattr(response, '__iter__') and not isinstance(response, bytes):
                return (chunk for chunk in response if chunk)
            
            return response
            
        except Exception as ex:
            error_message = f"Text-to-speech generation failed: {str(ex)}"
            logger.error(error_message, exc_info=True)
            
            if isinstance(ex, json.JSONDecodeError):
                raise InvokeBadRequestError(f"Failed to process response: {str(ex)}")
            elif any(code in str(ex) for code in ('401', '403')):
                raise InvokeAuthorizationError(error_message)
            elif '429' in str(ex):
                raise InvokeRateLimitError(error_message)
            elif str(ex).startswith('5'):
                raise InvokeServerUnavailableError(error_message)
            else:
                raise InvokeBadRequestError(error_message)

    def validate_credentials(
        self, model: str, credentials: dict, user: Optional[str] = None
    ) -> None:
        """Validate credentials for text2speech model.

        Args:
            model: Model name
            credentials: Model credentials
            user: Unique user id

        Raises:
            CredentialsValidateFailedError: If credentials validation fails
        """
        try:
            api_key = self._validate_credentials(credentials)
            client = ElevenLabs(api_key=api_key)
            client.voices.get_default_settings()
            
        except Exception as ex:
            logger.error(f"Credentials validation failed: {str(ex)}", exc_info=True)
            raise CredentialsValidateFailedError(str(ex))
