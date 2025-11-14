"""Integration tests for OpenAI model helpers."""

from __future__ import annotations

import pytest
import utils.llm.providers.openai as openai_module  # type: ignore[import]
from utils.llm.model_registry import MODELS, Model  # type: ignore[import]
from utils.tests.integration.helpers import (
    assert_capital_of_france,  # type: ignore[import]
)

OPENAI_MODEL: Model | None = next(
    (model for model in MODELS if model.id == "gpt-5-2025-08-07"), None
)
assert OPENAI_MODEL is not None


@pytest.mark.integration
def test_openai_provider_get_response_live_call():
    """It invokes the live OpenAI API and returns text."""
    from utils.llm.model_registry import configure_api_keys  # type: ignore[import]

    from gcp.secret_manager import get_secret  # type: ignore[import]
    from helpers.constants import OPENAI_API_KEY_SECRET_NAME  # type: ignore[import]

    # Configure API keys from GCP
    configure_api_keys(from_gcp=True)
    api_key = get_secret(OPENAI_API_KEY_SECRET_NAME)
    provider = openai_module.OpenAIProvider(api_key=api_key)
    assert_capital_of_france(
        lambda prompt: provider.get_response(
            OPENAI_MODEL,
            prompt,
            temperature=0,
            max_tokens=256,
            wait_time=1,
        )
    )
