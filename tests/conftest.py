"""Shared pytest configuration.

HuggingFace authentication is intentionally *not* accepted as a CLI option;
the toolkit reads ``HF_TOKEN`` from the environment or the HuggingFace CLI cache.
"""
