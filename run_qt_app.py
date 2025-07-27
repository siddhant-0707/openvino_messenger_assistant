#!/usr/bin/env python3
"""
Telegram RAG System - Qt Application Launcher
============================================

Simple launcher script for the Qt for Python interface.
"""

import sys
import os

def main():
    """Launch the Qt application"""
    try:
        # Import and run the Qt application
        from telegram_rag_qt import main as qt_main
        qt_main()
    except ImportError as e:
        print(f"Error importing Qt application: {e}")
        print("Please ensure PySide6 is installed:")
        print("pip install PySide6")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 