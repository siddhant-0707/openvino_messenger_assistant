#!/usr/bin/env python3
"""
Telegram RAG System - Qt Application Launcher
============================================

Simple launcher script for the Qt for Python interface.
"""

import sys
import os
from pathlib import Path

base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "src"))

def main():
    """Launch the Qt application"""
    try:
        # Prefer package import so PyInstaller bundles the module
        from src.telegram_rag_qt import main as qt_main
        qt_main()
    except ImportError as e:
        # Fallback to source layout (development mode)
        try:
            project_root = Path(__file__).parent
            src_path = project_root / "src"
            sys.path.insert(0, str(src_path))
            from telegram_rag_qt import main as qt_main  # type: ignore
            qt_main()
            return
        except Exception:
            print(f"Error importing Qt application: {e}")
        print("Please ensure PySide6 is installed:")
        print("pip install PySide6")
        print("Also ensure all source files are in the src/ directory")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 