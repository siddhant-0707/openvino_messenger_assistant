"""
OpenVINO Telegram RAG Assistant
==============================

A modern desktop application for analyzing Telegram messages using OpenVINO and AI.

This package provides:
- Telegram message ingestion and processing
- Vector-based semantic search using OpenVINO
- Interactive Qt-based desktop interface
- RAG (Retrieval-Augmented Generation) chat functionality
"""

__version__ = "1.0.0"
__author__ = "OpenVINO Telegram RAG Team"

# Avoid importing submodules at package import time to prevent side effects
__all__ = [
    "__version__",
    "__author__",
]