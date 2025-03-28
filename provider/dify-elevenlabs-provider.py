import logging
from collections.abc import Mapping

from dify_plugin import ModelProvider
from dify_plugin.entities.model import ModelType
from dify_plugin.errors.model import CredentialsValidateFailedError
from elevenlabs.client import ElevenLabs

logger = logging.getLogger(__name__)


class DifyElevenlabsProviderModelProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: Mapping) -> None:
        """
        Validate provider credentials
        if validate failed, raise exception

        :param credentials: provider credentials, credentials form defined in `provider_credential_schema`.
        """
        try:
            api_key = credentials.get('api_key')
            if not api_key:
                raise CredentialsValidateFailedError("API key is required")
                
            client = ElevenLabs(api_key=api_key)
            client.voices.get_default_settings()
            
        except CredentialsValidateFailedError as ex:
            raise ex
        except Exception as ex:
            logger.exception(
                f"{self.get_provider_schema().provider} credentials validate failed"
            )
            raise CredentialsValidateFailedError(f"Invalid API key: {str(ex)}")
