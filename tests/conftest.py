def pytest_addoption(parser):
    parser.addoption("--hf-token", default=None, help="HuggingFace access token for gated models")
