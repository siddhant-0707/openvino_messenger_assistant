#!/usr/bin/env python3
"""
Setup script for Telegram RAG System
===================================

Installation:
    pip install -e .

For development:
    pip install -e ".[dev]"
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
README = Path("README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="openvino-messenger-assistant",
    version="1.0.0",
    description="Advanced Telegram message analysis using OpenVINO and RAG",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Siddhant Chauhan",
    author_email="chauhanjoy10@example.com",
    url="https://github.com/siddhant-0707/openvino_messenger_assistant",
    project_urls={
        "Documentation": "https://github.com/siddhant-0707/openvino_messenger_assistant#readme",
        "Source": "https://github.com/siddhant-0707/openvino_messenger_assistant",
        "Tracker": "https://github.com/siddhant-0707/openvino_messenger_assistant/issues",
    },
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.800"
        ],
        "gpu": [
            "openvino-dev[tensorflow,pytorch]>=2024.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "telegram-rag-qt=src.telegram_rag_qt:main",
            "telegram-rag-gradio=examples.telegram_rag_gradio:main",
            "telegram-rag=run_qt_app:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Communications :: Chat",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="telegram, rag, openvino, ai, nlp, chatbot, langchain, qt",
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml", "*.json"],
    },
    zip_safe=False,
) 