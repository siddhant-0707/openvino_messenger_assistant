# OpenVINO Messenger AI-Assistant for a AI PC

OpenVINO Messenger AI-Assistant is a cross-platform desktop application developed as part of [Google Summer of Code 2024](https://summerofcode.withgoogle.com/programs/2025) under the [OpenVINO Toolkit](https://github.com/openvinotoolkit). It transforms Telegram channel messages into a searchable, conversational knowledge base that runs locally for privacy and performance. It combines efficient OpenVINO-optimized models with LangChain-based RAG and a polished Qt desktop UI (plus a Gradio web option) to deliver real-time, streaming answers grounded in your data.

To use the application, check out the [Github Releases](https://github.com/siddhant-0707/openvino_messenger_assistant/releases) and run the `TelegramRAG` executable on your system. For a detailed walkthrough, check out the [Medium article](https://medium.com/openvino-toolkit/draft-work-in-progress-30b29ed4f8b2). For instructions on how to build the project from source, check out the [README](https://github.com/siddhant-0707/openvino_messenger_assistant/blob/main/README.md).

## Project Goals

1. Build a private, local-first RAG system for Telegram data
2. Automate OpenVINO model discovery, download, and device selection (NPU/GPU/CPU/AUTO)
3. Enable fast ingestion, indexing (FAISS), and semantic querying with reranking
4. Provide real-time streaming responses with markdown rendering and diagnostics
5. Offer professional UX via Qt desktop and a simple web interface via Gradio

## Key Achievements

1. OpenVINO GenAI integrations for LLM, text embeddings, and reranking
2. Telegram ingestion with channel discovery, time filters, session persistence
3. Robust document processing pipeline: chunking, embeddings, FAISS vector store
4. Qt for Python desktop app with streaming chat, model panels, and diagnostics
5. Dynamic model discovery from curated [Hugging Face OpenVINO collections](https://huggingface.co/collections/OpenVINO/llm-6687aaa2abca3bbcec71a9bd)
6. Gradio web UI parity for quick local testing and demos
7. Packaged desktop binaries and a clean Python package layout
8. Pre-built Binaries: Available for Windows x64 and Linux.

## Current State

OV Messenger AI-Assistant is fully functional and can be used to ingest data from Telegram channels and ask questions about your content. Executables are published for Windows and Linux on [GitHub Releases](https://github.com/siddhant-0707/openvino_messenger_assistant/releases).

## Key Components

### Processing

1. Embeddings via `OpenVINOTextEmbeddings` and `OpenVINOTextReranker`
2. Vector store creation and updates using FAISS under `data/telegram_vector_store`
3. Retrieval + RAG combine using LangChain chains; model-specific prompt support
4. Chunking with `RecursiveCharacterTextSplitter` and metadata preservation

### Telegram Ingestion

1. Telethon-based client with session persistence (`telegram_session.session`)
2. Channel discovery, time-window filtering, and per-channel limits
3. JSON export to `data/telegram_data/*.json` with message/article metadata
4. Async helpers for GUI flows: send/verify code, start/stop lifecycle

### User Interface

1. Qt desktop app with streaming responses, markdown rendering, and model panels
2. Device management (NPU/GPU/CPU/AUTO), GPU diagnostics, progress and error reporting
3. Gradio web UI for lightweight local demos with similar RAG capabilities

## Future Work

1. Ingestion from new data sources beyond Telegram
2. Multi-modal support (images, documents) and advanced analytics
3. Test suite expansion and performance benchmarks across devices

## Challenges and Learnings

1. GPU memory variability requires fallbacks (CPU/AUTO) and smaller models
2. Streaming UX in Qt demands careful threading, backpressure, and error handling
3. Reliable Telegram login and channel discovery flows need resilient session logic
4. Dynamic model discovery with friendly names simplifies UX but requires curation
5. Persisting and updating FAISS safely across sessions improves reliability

## Project Resources

1. README: [`README.md`](../README.md)
2. Architecture: [`docs/ARCHITECTURE.md`](./ARCHITECTURE.md)
3. Project structure: [`docs/PROJECT_STRUCTURE.md`](./PROJECT_STRUCTURE.md)
4. Examples: Gradio app and notebook under `examples/`
5. Images: screenshots and diagrams under `docs/images/`
6. [Medium Blog](https://medium.com/openvino-toolkit/draft-work-in-progress-30b29ed4f8b2)

### Presentations

1. [Initial Presentation (29 May 2025)](https://docs.google.com/presentation/d/1kPT8PFMnTzs3TO69iwWzSaH0orwcK8pQ12CkBt1xaFA/edit?usp=sharing)
1. [Midterm Evaluation (10 July 2025)](https://docs.google.com/presentation/d/12LWTb3AjAa99ZzWgXKCduKWss3YW6Mthk6ILgQ3Jjn0/edit?usp=sharing)
1. [Final Evaluation (28 August 2025)](https://docs.google.com/presentation/d/1Y-rhZ1oC_FNOF_ymDJf1UsIxbaFnlqYFeyhPQM4aGkA/edit?usp=sharing)

### Proposal

- [Submitted Proposal](https://summerofcode.withgoogle.com/media/user/c1a22178e8fd/proposal/gAAAAABotM9SoWEAte2QRxYbta9HfXib89VKjYgF0mhxhBSiv6MP0oI9bNs7BT92h2kMtsFaw3XHH1lzNvD9uA8w8LrtYzp7q0nYjqaG25qUp8zHOrYkb6k=.pdf)

## Acknowledgments

I would like to thank my mentors [Dmitriy Pastushenkov](https://github.com/DimaPastushenkov) and [Ethan Yang](https://github.com/openvino-dev-samples) for their support and guidance throughout the project. I would also like to thank the OpenVINO community for their help and feedback.
