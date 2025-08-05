# Project Structure

This document describes the organized structure of the OpenVINO Telegram RAG Assistant project.

## Directory Layout

```
openvino_messenger_assistant/
├── src/                          # Core source code
│   ├── __init__.py              # Package initialization
│   ├── telegram_rag_qt.py       # Qt desktop interface
│   ├── telegram_ingestion.py    # Telegram API integration
│   ├── telegram_rag_integration.py # RAG system integration
│   ├── ov_langchain_helper.py   # OpenVINO LangChain helpers
│   ├── llm_config.py           # Model configurations
│   ├── article_processor.py     # Text processing utilities
│   ├── genai_helper.py         # GenAI utilities
│   ├── cmd_helper.py           # Command line utilities
│   ├── pip_helper.py           # Package management
│   ├── notebook_utils.py       # Jupyter notebook utilities
│   └── gradio_helper.py        # Gradio interface utilities
│
├── examples/                     # Example implementations
│   ├── __init__.py              # Examples package init
│   ├── telegram_rag_gradio.py   # Gradio web interface
│   └── telegram_rag_example.ipynb # Jupyter notebook demo
│
├── docs/                        # Documentation
│   ├── images/                  # Documentation images
│   ├── ARCHITECTURE.md          # System architecture
│   ├── PROJECT_STRUCTURE.md     # This file
│   ├── telegram_rag_medium_article.md # Medium article
│   └── release_guide.md         # Release documentation
│
├── tests/                       # Test suite (future)
│   └── __init__.py              # Tests package init
│
├── data/                        # Data storage (gitignored)
│   ├── telegram_data/           # Downloaded Telegram messages
│   └── telegram_vector_store/   # Vector database storage
│
├── .models/                     # Model storage (gitignored)
│   └── (OpenVINO models)        # Downloaded and converted models
│
├── run_qt_app.py               # Main application launcher
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Modern Python packaging
├── setup.py                   # Legacy packaging support
├── MANIFEST.in                # Package manifest
├── env.example               # Environment template
├── README.md                 # Main documentation
├── CHANGELOG.md              # Version history
├── LICENSE                   # License information
└── .gitignore               # Git ignore patterns
```

## Key Design Principles

### 1. **Separation of Concerns**
- **`src/`**: Core business logic and implementations
- **`examples/`**: Demonstrations and alternative interfaces
- **`docs/`**: All documentation and guides
- **`tests/`**: Test suite (prepared for future expansion)

### 2. **Data Organization**
- **`data/`**: All runtime data consolidated in one location
- **`.models/`**: Model storage separate from source code
- Both directories are gitignored to keep repository clean

### 3. **Entry Points**
- **`run_qt_app.py`**: Single main entry point for the application
- Clear separation between launcher and implementation

### 4. **Professional Structure**
- Follows Python packaging best practices
- Compatible with PyPI distribution
- Clear module boundaries and imports

## Import Patterns

### From Main Application
```python
# Main launcher automatically handles path setup
python run_qt_app.py
```

### From Examples
```python
# Examples set up their own path imports
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
```

### From Tests (Future)
```python
# Tests will import from src package
import pytest
from src.telegram_ingestion import TelegramChannelIngestion
```

## Benefits of This Structure

1. **Maintainability**: Clear separation of different types of code
2. **Scalability**: Easy to add new examples, tests, or documentation
3. **Professional**: Follows industry standards for Python projects
4. **Distribution Ready**: Structure supports PyPI packaging
5. **Development Friendly**: Clear where to find and add different types of files

## Migration Notes

All file paths in configuration and documentation have been updated to reflect the new structure. The main application entry point remains the same (`run_qt_app.py`) for backward compatibility.