"""Integration tests for Mistral model helpers."""

from __future__ import annotations

import pytest
import utils.llm.providers.mistral as mistral_module  # type: ignore[import]
from utils.llm.lab_registry import LABS  # type: ignore[import]
from utils.llm.model_registry import Model  # type: ignore[import]
from utils.tests.integration.helpers import (
    assert_capital_of_france,  # type: ignore[import]
)

# We don't have a Mistral model in the model registry, so we need to create it manually.
MISTRAL_MODEL: Model = Model(
    id="mistral-large-latest",
    full_name="mistral-large-latest",
    token_limit=256_000,
    provider_cls=mistral_module.MistralProvider,
    lab=LABS["Mistral"],
)


@pytest.mark.integration
def test_mistral_provider_get_response_live_call():
    """It invokes the live Mistral API and returns text."""
    from utils.llm.model_registry import configure_api_keys  # type: ignore[import]

    from gcp.secret_manager import get_secret  # type: ignore[import]
    from helpers.constants import MISTRAL_API_KEY_SECRET_NAME  # type: ignore[import]

    # Configure API keys from GCP
    configure_api_keys(from_gcp=True)
    api_key = get_secret(MISTRAL_API_KEY_SECRET_NAME)
    provider = mistral_module.MistralProvider(api_key=api_key)
    assert_capital_of_france(
        lambda prompt: provider.get_response(
            MISTRAL_MODEL,
            prompt,
            temperature=0,
            wait_time=1,
            max_tokens=256,
        )
    )
