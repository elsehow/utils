"""Integration tests for xAI model helpers."""

from __future__ import annotations

import pytest
import utils.llm.providers.xai as xai_module  # type: ignore[import]
from utils.llm.model_registry import MODELS, Model  # type: ignore[import]
from utils.tests.integration.helpers import (
    assert_capital_of_france,  # type: ignore[import]
)

XAI_MODEL: Model | None = next((model for model in MODELS if model.id == "grok-4-0709"), None)
assert XAI_MODEL is not None


@pytest.mark.integration
def test_xai_provider_get_response_live_call():
    """It invokes the live xAI API and returns text."""
    from utils.llm.model_registry import configure_api_keys  # type: ignore[import]

    from gcp.secret_manager import get_secret  # type: ignore[import]
    from helpers.constants import XAI_API_KEY_SECRET_NAME  # type: ignore[import]

    # Configure API keys from GCP
    configure_api_keys(from_gcp=True)
    api_key = get_secret(XAI_API_KEY_SECRET_NAME)
    provider = xai_module.XAIProvider(api_key=api_key)
    assert_capital_of_france(
        lambda prompt: provider.get_response(
            XAI_MODEL,
            prompt,
            temperature=0,
            wait_time=1,
        )
    )
