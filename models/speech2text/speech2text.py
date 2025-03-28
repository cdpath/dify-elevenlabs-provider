from typing import Optional, Dict, Type, Any
from io import BytesIO
import logging
import json

from dify_plugin import Speech2TextModel
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeBadRequestError,
    InvokeAuthorizationError,
    InvokeConnectionError, 
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError
)
from elevenlabs.client import ElevenLabs
import requests.exceptions

logger = logging.getLogger(__name__)

class DifyElevenlabsProviderSpeech2TextModel(Speech2TextModel):
    """Model class for ElevenLabs Speech to Text model."""

    DEFAULT_TRANSCRIPTION_SETTINGS = {
        "tag_audio_events": True,
        "language_code": "eng",
        "diarize": True,
    }

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
        credentials: dict,
        audio_file: bytes,
        user: Optional[str] = None,
    ) -> str:
        """Speech to text invoke.

        Args:
            model: Model name
            credentials: Model credentials
            audio_file: Audio file bytes
            user: Unique user id

        Returns:
            str: Transcribed text

        Raises:
            Various InvokeError types based on the error encountered
        """
        api_key = self._validate_credentials(credentials)
            
        try:
            client = ElevenLabs(api_key=api_key)
            
            audio_data = BytesIO(audio_file) if not hasattr(audio_file, 'read') else audio_file
            
            transcription = client.speech_to_text.convert(
                file=audio_data,
                model_id=model,
                **self.DEFAULT_TRANSCRIPTION_SETTINGS
            )
            
            return transcription.text
                
        except Exception as ex:
            error_message = f"Speech-to-text transcription failed: {str(ex)}"
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
        """Validate speech to text credentials.

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
            client.generate("Hello Dify")
        except Exception as ex:
            logger.error(f"Credentials validation failed: {str(ex)}", exc_info=True)
            raise CredentialsValidateFailedError(str(ex))