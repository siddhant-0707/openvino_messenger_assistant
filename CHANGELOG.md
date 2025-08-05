# Changelog

All notable changes to the Telegram RAG System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-08-05

### Added
- **Qt for Python Desktop Interface**: Professional desktop application built with PySide6
- **Modern Chat Interface**: Chat-style Q&A with conversation history and message bubbles
- **Real-time Streaming Responses**: Dynamic message sizing with typing indicators
- **TextEmbeddingPipeline Integration**: Native OpenVINO GenAI TextEmbeddingPipeline support
- **Dynamic Model Selection**: Fetch available LLM models from OpenVINO Hugging Face collection
- **GPU Diagnostics**: Comprehensive GPU information and optimization recommendations
- **Enhanced Model Management**: Visual status indicators and background operations
- **Dark/Light Theme Support**: Professional theming with smooth animations
- **Export Chat History**: Save conversations as text or JSON
- **Organized Directory Structure**: Models in `.models/`, data in `.data/`
- **Background Processing**: Non-blocking operations with progress tracking
- **Advanced Device Management**: Descriptive GPU names and automatic fallbacks

### Changed
- **Unified Requirements**: Consolidated multiple requirements files into one
- **Model Download Strategy**: Direct download from OpenVINO Hugging Face repositories
- **LangChain API**: Updated from deprecated `get_relevant_documents` to `invoke`
- **Enhanced Documentation**: Comprehensive README with installation and usage guides
- **Improved Error Handling**: Better GPU memory management and error recovery

### Fixed
- **GPU Memory Issues**: Resolved `CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST` errors
- **Model Loading**: Proper CPU fallback for GPU failures
- **Button State Management**: Fixed button not re-enabling after generation
- **CSS Warnings**: Removed unsupported `box-shadow` properties
- **Parameter Validation**: Fixed channel list/string parameter mismatches

### Technical Improvements
- **Modern Qt6 Widgets**: Professional desktop UI components
- **Signal/Slot Architecture**: Clean component communication
- **Multi-threading**: Background operations with worker threads
- **Smooth Animations**: Property-based animations for scrolling and transitions
- **Memory Optimization**: Efficient text streaming and layout updates
- **Cross-platform Support**: Windows, macOS, and Linux compatibility

## [1.0.0] - 2024-XX-XX

### Added
- **Initial Release**: Gradio-based web interface
- **Telegram Integration**: Download and process Telegram channel messages
- **RAG System**: Retrieval Augmented Generation with OpenVINO models
- **Vector Store**: FAISS-based similarity search
- **Model Support**: Multiple LLM, embedding, and reranking models
- **Jupyter Notebooks**: Interactive exploration and development tools

### Features
- Basic Telegram message downloading
- Message processing into searchable vector store
- Question answering with context retrieval
- Web-based Gradio interface
- OpenVINO model optimization
- LangChain integration for RAG pipeline

---

## Release Process

### For Developers

1. **Update Version Numbers**:
   - `setup.py`: Update version
   - `CHANGELOG.md`: Add new version entry
   - `README.md`: Update any version references

2. **Test Release**:
   ```bash
   python -m build
   python -m twine check dist/*
   ```

3. **Create Release**:
   - Tag version: `git tag v1.0.0`
   - Push tags: `git push --tags`
   - Create GitHub release with changelog

### For Users

See [Installation Guide](README.md#installation) for setup instructions. 