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

# Core imports for easy access
from .telegram_ingestion import TelegramChannelIngestion
from .telegram_rag_integration import TelegramRAGIntegration
from .ov_langchain_helper import (
    OpenVINOLLM, 
    OpenVINOBgeEmbeddings, 
    OpenVINOReranker, 
    OpenVINOTextEmbeddings
)

__all__ = [
    "TelegramChannelIngestion",
    "TelegramRAGIntegration", 
    "OpenVINOLLM",
    "OpenVINOBgeEmbeddings",
    "OpenVINOReranker",
    "OpenVINOTextEmbeddings"
]