"""Integration tests for Anthropic model helpers."""

from __future__ import annotations

import pytest
import utils.llm.providers.anthropic as anthropic_module  # type: ignore[import]
from utils.llm.model_registry import MODELS, Model  # type: ignore[import]
from utils.tests.integration.helpers import (
    assert_capital_of_france,  # type: ignore[import]
)

ANTHROPIC_MODEL: Model | None = next(
    (model for model in MODELS if model.id == "claude-3-7-sonnet-20250219"), None
)
assert ANTHROPIC_MODEL is not None


@pytest.mark.integration
def test_anthropic_provider_get_response_live_call():
    """It invokes the live Anthropic API and returns text."""
    from utils.llm.model_registry import configure_api_keys  # type: ignore[import]

    from gcp.secret_manager import get_secret  # type: ignore[import]
    from helpers.constants import ANTHROPIC_API_KEY_SECRET_NAME  # type: ignore[import]

    # Configure API keys from GCP
    configure_api_keys(from_gcp=True)
    api_key = get_secret(ANTHROPIC_API_KEY_SECRET_NAME)
    provider = anthropic_module.AnthropicProvider(api_key=api_key)
    assert_capital_of_france(
        lambda prompt: provider.get_response(
            ANTHROPIC_MODEL,
            prompt,
            temperature=0,
            max_tokens=16,
            wait_time=1,
        )
    )
