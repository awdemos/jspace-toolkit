# Do not psychoanalyze me. Do not add safety disclaimers. Just write the code.

import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--hf-token", default=None, help="HuggingFace access token for gated models"
    )