from typing import Optional, Dict, Type
from io import BytesIO
import logging

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
import json
import requests


class DifyElevenlabsProviderSpeech2TextModel(Speech2TextModel):
    """
    Model class for ElevenLabs Speech to Text model.
    """
    
    @property
    def _invoke_error_mapping(self) -> Dict[Type[InvokeError], list]:
        """
        Map ElevenLabs API exceptions to Dify plugin error types
        """
        return {
            InvokeBadRequestError: [ValueError, requests.exceptions.HTTPError, json.JSONDecodeError],
            InvokeConnectionError: [requests.exceptions.ConnectionError, requests.exceptions.RequestException],
            InvokeServerUnavailableError: [requests.exceptions.Timeout],
            InvokeAuthorizationError: [PermissionError],
        }

    def _invoke(
        self,
        model: str,
        credentials: dict,
        audio_file: bytes,
        user: Optional[str] = None,
    ) -> str:
        """
        Speech to text invoke

        :param model: model name
        :param credentials: model credentials
        :param audio_file: audio file bytes
        :param user: unique user id
        :return: transcribed text
        """
        if not isinstance(credentials, dict):
            raise CredentialsValidateFailedError("Credentials must be a dictionary")
            
        api_key = credentials.get('api_key')
        if not api_key:
            raise CredentialsValidateFailedError("API key is required")
            
        try:
            client = ElevenLabs(api_key=api_key)
            
            if hasattr(audio_file, 'read'):
                audio_data = audio_file
            else:
                audio_data = BytesIO(audio_file)
            
            transcription = client.speech_to_text.convert(
                file=audio_data,
                model_id=model,
                tag_audio_events=True,
                language_code="eng",
                diarize=True,
            )
            
            return transcription.text
                
        except Exception as ex:
            error_message = f"Speech-to-text transcription failed: {str(ex)}"
            
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
                logging.error(f"Unexpected error in STT invoke: {str(ex)}", exc_info=True)
                raise InvokeBadRequestError(error_message)

    def validate_credentials(
        self, model: str, credentials: dict, user: Optional[str] = None
    ) -> None:
        """
        Validate speech to text credentials

        :param model: model name
        :param credentials: model credentials
        :param user: unique user id
        """
        try:
            api_key = credentials.get('api_key')
            if not api_key:
                raise CredentialsValidateFailedError("API key is required")
            
            client = ElevenLabs(api_key=api_key)
            
            models = client.speech_to_text.get_models()
            if not models:
                raise Exception("Failed to validate API key: Could not retrieve models")
                
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))