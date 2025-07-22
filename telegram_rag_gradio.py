import os
import asyncio
import gradio as gr
import nest_asyncio
from dotenv import load_dotenv
from telegram_ingestion import TelegramChannelIngestion
from telegram_rag_integration import TelegramRAGIntegration
from pathlib import Path
import openvino as ov
from openvino_tokenizers import convert_tokenizer
from transformers import AutoTokenizer, AutoModel
from ov_langchain_helper import OpenVINOLLM, OpenVINOBgeEmbeddings, OpenVINOReranker
from langchain.retrievers import ContextualCompressionRetriever
import json
from datetime import datetime
from llm_config import (
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_RERANK_MODELS,
    SUPPORTED_LLM_MODELS,
)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Setup directories
telegram_data_dir = Path("telegram_data")
telegram_data_dir.mkdir(exist_ok=True)

vector_store_path = Path("telegram_vector_store")
vector_store_path.mkdir(exist_ok=True)

# Default model selections
DEFAULT_LANGUAGE = "English"
DEFAULT_LLM_MODEL = "qwen2.5-3b-instruct"
DEFAULT_EMBEDDING_MODEL = "bge-small-en-v1.5"
DEFAULT_RERANK_MODEL = "bge-reranker-v2-m3"
DEFAULT_DEVICE = "AUTO"

# Get model configurations
embedding_model_configuration = SUPPORTED_EMBEDDING_MODELS[DEFAULT_LANGUAGE][DEFAULT_EMBEDDING_MODEL]
rerank_model_configuration = SUPPORTED_RERANK_MODELS[DEFAULT_RERANK_MODEL]
llm_model_configuration = SUPPORTED_LLM_MODELS[DEFAULT_LANGUAGE][DEFAULT_LLM_MODEL]

# Model paths
embedding_model_dir = Path(DEFAULT_EMBEDDING_MODEL)
rerank_model_dir = Path(DEFAULT_RERANK_MODEL)
llm_base_dir = Path(DEFAULT_LLM_MODEL)
llm_model_dir = llm_base_dir / "INT4_compressed_weights"

# Initialize models
embedding = None
reranker = None
llm = None

def initialize_models(device="AUTO"):
    """Initialize all models with the specified device"""
    global embedding, reranker, llm
    
    # Initialize embedding model
    if embedding_model_dir.exists():
        embedding = OpenVINOBgeEmbeddings(
            model_path=str(embedding_model_dir),
            model_kwargs={"device_name": device},
            encode_kwargs={
                "mean_pooling": embedding_model_configuration["mean_pooling"],
                "normalize_embeddings": embedding_model_configuration["normalize_embeddings"],
                "batch_size": 4,
            }
        )
        print(f"Embedding model loaded from {embedding_model_dir}")
    else:
        print(f"Embedding model not found at {embedding_model_dir}")
    
    # Initialize reranker model
    if rerank_model_dir.exists():
        reranker = OpenVINOReranker(
            model_path=str(rerank_model_dir),
            model_kwargs={"device_name": device},
            top_n=5,
        )
        print(f"Reranking model loaded from {rerank_model_dir}")
    else:
        print(f"Reranking model not found at {rerank_model_dir}")
    
    # Initialize LLM
    if llm_model_dir.exists():
        llm = OpenVINOLLM.from_model_path(
            model_path=str(llm_model_dir),
            device=device,
        )
        
        # Set default parameters from model configuration
        llm.config.max_new_tokens = 1024
        llm.config.temperature = 0.7
        llm.config.top_p = 0.9
        llm.config.top_k = 50
        llm.config.repetition_penalty = 1.1
        llm.config.do_sample = True
        print(f"LLM loaded from {llm_model_dir}")
    else:
        print(f"LLM not found at {llm_model_dir}")

# Initialize RAG system
def initialize_rag():
    """Initialize the RAG system with the current models"""
    global rag
    if embedding is not None:
        rag = TelegramRAGIntegration(
            embedding_model=embedding,
            vector_store_path=str(vector_store_path),
            chunk_size=500,
            chunk_overlap=50
        )
        print("RAG system initialized with embedding model")
    else:
        print("Cannot initialize RAG system: embedding model not loaded")

# Initialize models with default device
initialize_models(DEFAULT_DEVICE)
initialize_rag()

def format_message(msg):
    """Format a message for display"""
    date = datetime.fromisoformat(msg["date"]).strftime("%Y-%m-%d %H:%M:%S")
    output = f"""
Channel: {msg['channel']}
Date: {date}
Views: {msg.get('views', 'N/A')}
Forwards: {msg.get('forwards', 'N/A')}
Message: {msg['text'][:200]}...
{'...' if len(msg['text']) > 200 else ''}
"""
    
    return output

async def download_messages_async(channels_str: str, limit: int, hours: int) -> tuple[str, str]:
    """Download messages from specified channels (async version)"""
    channels = [c.strip() for c in channels_str.split(",") if c.strip()]
    if not channels:
        return "Please provide at least one channel name", ""
    
    try:
        api_id = os.getenv("TELEGRAM_API_ID")
        api_hash = os.getenv("TELEGRAM_API_HASH")
        
        if not api_id or not api_hash:
            return "Telegram API credentials not found! Please check your .env file.", ""
        
        ingestion = TelegramChannelIngestion(
            api_id=api_id,
            api_hash=api_hash,
            storage_dir=str(telegram_data_dir)
        )
        
        await ingestion.start()
        try:
            messages = await ingestion.process_channels(
                channels,
                limit_per_channel=limit,
                since_hours=hours
            )
            
            # Format messages for display
            formatted_messages = "\n\n".join(format_message(msg) for msg in messages[:5])  # Show first 5 messages
            if len(messages) > 5:
                formatted_messages += f"\n\n... and {len(messages) - 5} more messages"
                
            return f"Successfully downloaded {len(messages)} messages from {len(channels)} channels", formatted_messages
        finally:
            await ingestion.stop()
    except Exception as e:
        import traceback
        return f"Error downloading messages: {str(e)}\n{traceback.format_exc()}", ""

def download_messages(channels_str: str, limit: int, hours: int) -> tuple[str, str]:
    """Download messages from specified channels (sync wrapper)"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(download_messages_async(channels_str, limit, hours))

def process_messages() -> str:
    """Process downloaded messages into vector store"""
    try:
        if not telegram_data_dir.exists() or not any(telegram_data_dir.iterdir()):
            return "No message data found. Please download messages first."
        
        rag.process_telegram_data_dir(data_dir=str(telegram_data_dir))
        return "Successfully processed messages into vector store"
    except Exception as e:
        import traceback
        return f"Error processing messages: {str(e)}\n{traceback.format_exc()}"

def query_messages(query: str, channel: str, num_results: int) -> str:
    """Query the vector store for relevant messages"""
    try:
        if not vector_store_path.exists() or not any(vector_store_path.iterdir()):
            return "Vector store not found. Please process messages first."
        
        filter_dict = {"channel": channel} if channel and channel.strip() else None
        results = rag.query_messages(query, k=num_results, filter_dict=filter_dict)
        
        output = []
        for i, doc in enumerate(results, 1):
            output.append(f"Result {i}:")
            output.append(f"Channel: {doc.metadata.get('channel', 'Unknown')}")
            output.append(f"Date: {doc.metadata.get('date', 'Unknown')}")
            
            # Show content snippet
            content = doc.page_content
            if len(content) > 300:
                content = content[:300] + "..."
            output.append(f"Content: {content}")
            output.append("")
            
        return "\n".join(output) if output else "No results found"
    except Exception as e:
        import traceback
        return f"Error querying messages: {str(e)}\n{traceback.format_exc()}"

def answer_question(
    question: str,
    channel: str,
    temperature: float,
    num_context: int,
    show_retrieved: bool = False,
    repetition_penalty: float = 1.1
) -> str:
    """Answer questions about Telegram messages using RAG"""
    try:
        if not vector_store_path.exists() or not any(vector_store_path.iterdir()):
            return "Vector store not found. Please process messages first."
        
        if llm is None:
            return "LLM not initialized. Please check model paths."
            
        filter_dict = {"channel": channel} if channel and channel.strip() else None
        
        # Update LLM configuration
        llm.config.temperature = temperature
        llm.config.top_p = 0.9
        llm.config.top_k = 50
        llm.config.repetition_penalty = repetition_penalty
        
        # Get answer from RAG
        result = rag.answer_question(
            question=question,
            llm=llm,
            k=num_context,
            filter_dict=filter_dict,
            show_retrieved=show_retrieved,
            reranker=reranker
        )
        
        # Format the response
        if show_retrieved and isinstance(result, dict) and "context_docs" in result:
            context_docs = []
            for i, doc in enumerate(result["context_docs"], 1):
                context_docs.append(f"Document {i}:")
                context_docs.append(f"Channel: {doc.metadata.get('channel', 'Unknown')}")
                context_docs.append(f"Date: {doc.metadata.get('date', 'Unknown')}")
                
                # Show content snippet
                content = doc.page_content
                if len(content) > 200:
                    content = content[:200] + "..."
                context_docs.append(f"Content: {content}")
                context_docs.append("")
                
            context_str = "\n".join(context_docs)
            return f"{result['answer']}\n\n--- Retrieved Context ---\n{context_str}"
        else:
            return result if isinstance(result, str) else result.get("answer", "No answer generated")
    except Exception as e:
        import traceback
        return f"Error answering question: {str(e)}\n{traceback.format_exc()}"

def answer_question_stream(
    question: str,
    channel: str,
    temperature: float,
    num_context: int,
    show_retrieved: bool = False,
    repetition_penalty: float = 1.1,
    progress=gr.Progress()
):
    """Answer questions about Telegram messages using RAG with streaming output"""
    try:
        if not vector_store_path.exists() or not any(vector_store_path.iterdir()):
            return "Vector store not found. Please process messages first."
        
        if llm is None:
            return "LLM not initialized. Please check model paths."
            
        filter_dict = {"channel": channel} if channel and channel.strip() else None
        
        # Update LLM configuration
        llm.config.temperature = temperature
        llm.config.top_p = 0.9
        llm.config.top_k = 50
        llm.config.repetition_penalty = repetition_penalty
        
        progress(0, desc="Retrieving relevant documents...")
        
        # Set up streaming for the LLM
        # First retrieve context documents from the vector store
        retriever = rag.vectorstore.as_retriever(
            search_kwargs={"k": num_context * 2 if reranker else num_context, "filter": filter_dict},
            search_type="similarity"
        )
        
        # Add reranking if available
        if reranker:
            from langchain.retrievers import ContextualCompressionRetriever
            reranker.top_n = num_context
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=retriever
            )
        
        # Get context documents
        context_docs = retriever.get_relevant_documents(question)
        
        progress(0.2, desc="Preparing prompt...")
        
        # Try to get model-specific prompt template
        rag_prompt_template = None
        try:
            from llm_config import SUPPORTED_LLM_MODELS
            for language, models in SUPPORTED_LLM_MODELS.items():
                for model_id, config in models.items():
                    if hasattr(llm, "model_name") and model_id in str(llm.model_name) and "rag_prompt_template" in config:
                        rag_prompt_template = config["rag_prompt_template"]
                        break
                if rag_prompt_template:
                    break
        except (ImportError, AttributeError):
            pass
        
        # Use default prompt if model-specific one is not available
        if not rag_prompt_template:
            rag_prompt_template = """
            You are a helpful assistant that answers questions based on Telegram messages and news articles.
            
            CONTEXT:
            {context}
            
            QUESTION:
            {input}
            
            INSTRUCTIONS:
            1. Answer the question directly and concisely based on the context provided.
            2. If the context contains relevant information, use it to provide a detailed answer.
            3. If the context doesn't contain enough information to fully answer the question, provide whatever partial information is available.
            4. Focus on extracting key facts and insights from the context.
            5. Don't be overly cautious - if there's information in the context that's relevant, use it confidently.
            
            ANSWER:
            """
        
        from langchain.prompts import PromptTemplate
        prompt = PromptTemplate.from_template(rag_prompt_template)
        
        # Format the context from documents
        context_str = "\n\n".join([doc.page_content for doc in context_docs])
        prompt_text = prompt.format(context=context_str, input=question)
        
        progress(0.4, desc="Generating response...")
        
        # Stream the response using callback updates
        output_text = ""
        
        # Set up a queue for the stream of tokens
        from threading import Thread
        import time
        from queue import Queue
        
        token_queue = Queue()
        stop_event = False
        
        def stream_tokens():
            nonlocal stop_event
            for chunk in llm._stream(prompt_text):
                if stop_event:
                    break
                token_queue.put(chunk.text)
            token_queue.put(None)  # Signal end of stream
        
        # Start the streaming in a background thread
        thread = Thread(target=stream_tokens)
        thread.start()
        
        # Process tokens with batching for UI updates
        buffer = ""
        last_update_time = time.time()
        update_interval = 0.1  # Update UI every 0.1 seconds
        
        progress_val = 0.4
        progress_step = 0.1
        next_progress = 0.5
        
        while True:
            token = token_queue.get()
            if token is None:
                break
                
            buffer += token
            current_time = time.time()
            
            # Update UI if enough time has passed or buffer contains special characters
            if (current_time - last_update_time > update_interval or 
                any(c in buffer for c in ["\n", ".", "!", "?"])):
                output_text += buffer
                yield output_text
                buffer = ""
                last_update_time = current_time
                
                # Update progress
                if progress_val < next_progress:
                    progress_val = min(0.9, progress_val + progress_step)
                    progress(progress_val, desc="Generating response...")
        
        # Add any remaining buffered content
        if buffer:
            output_text += buffer
            yield output_text
        
        # Handle context display if requested
        if show_retrieved and context_docs:
            progress(0.95, desc="Adding context information...")
            context_info = []
            context_info.append("\n\n--- Retrieved Context ---")
            for i, doc in enumerate(context_docs, 1):
                context_info.append(f"Document {i}:")
                context_info.append(f"Channel: {doc.metadata.get('channel', 'Unknown')}")
                context_info.append(f"Date: {doc.metadata.get('date', 'Unknown')}")
                
                # Show content snippet
                content = doc.page_content
                if len(content) > 200:
                    content = content[:200] + "..."
                context_info.append(f"Content: {content}")
                context_info.append("")
            
            context_display = "\n".join(context_info)
            output_text += context_display
            yield output_text
        
        progress(1.0, desc="Complete!")
        return output_text

    except Exception as e:
        import traceback
        return f"Error answering question: {str(e)}\n{traceback.format_exc()}"

def download_and_convert_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    """This function is deprecated and will be removed"""
    print("Warning: Using deprecated function. Models should be converted using optimum-cli.")
    return DEFAULT_EMBEDDING_MODEL

# Initialize Gradio interface
with gr.Blocks(title="Telegram RAG System") as demo:
    gr.Markdown("# Telegram RAG System with OpenVINO")
    
    with gr.Tab("Models & Configuration"):
        gr.Markdown("## Model Configuration")
        
        gr.Markdown("""
        ℹ️ **Note:** Models need to be downloaded and converted using OpenVINO before they can be used. 
        Please run the Jupyter notebook first to convert the models you want to use. 
        Once converted, models should be placed in directories named after the model ID in the same folder as this script.
        """)
        
        with gr.Row():
            with gr.Column():
                # Language selection
                language_dropdown = gr.Dropdown(
                    choices=list(SUPPORTED_LLM_MODELS.keys()),
                    value=DEFAULT_LANGUAGE,
                    label="Language"
                )
                
                # LLM model selection
                llm_model_dropdown = gr.Dropdown(
                    choices=list(SUPPORTED_LLM_MODELS[DEFAULT_LANGUAGE].keys()),
                    value=DEFAULT_LLM_MODEL,
                    label="LLM Model"
                )
            
            with gr.Column():
                # Embedding model selection
                embedding_model_dropdown = gr.Dropdown(
                    choices=list(SUPPORTED_EMBEDDING_MODELS[DEFAULT_LANGUAGE].keys()),
                    value=DEFAULT_EMBEDDING_MODEL,
                    label="Embedding Model"
                )
                
                # Reranker model selection
                reranker_model_dropdown = gr.Dropdown(
                    choices=list(SUPPORTED_RERANK_MODELS.keys()),
                    value=DEFAULT_RERANK_MODEL,
                    label="Reranker Model"
                )
        
        # Device selection
        device_dropdown = gr.Dropdown(
            choices=["CPU", "GPU", "AUTO"],
            value=DEFAULT_DEVICE,
            label="Device for Models"
        )
        
        reload_btn = gr.Button("Reload Models with Selected Configuration")
        model_status = gr.Textbox(label="Model Status", value="Models loaded with default settings")
        
        # Display current model information
        model_info = gr.Markdown(f"""
        ### Current Models:
        - LLM: {DEFAULT_LLM_MODEL} (INT4)
        - Embedding: {DEFAULT_EMBEDDING_MODEL}
        - Reranker: {DEFAULT_RERANK_MODEL}
        - Device: {DEFAULT_DEVICE}
        """)
        
        # Add a model availability checker
        gr.Markdown("## Check Model Availability")
        check_models_btn = gr.Button("Check Which Models Are Already Converted")
        models_availability = gr.Markdown("Click the button to check model availability...")
        
        def check_model_availability():
            """Check which models are available on disk"""
            # Check LLMs
            available_llms = []
            unavailable_llms = []
            for language, models in SUPPORTED_LLM_MODELS.items():
                for model_id in models.keys():
                    model_path = Path(model_id) / "INT4_compressed_weights"
                    if model_path.exists():
                        available_llms.append(f"✅ {language}/{model_id}")
                    else:
                        unavailable_llms.append(f"❌ {language}/{model_id}")
            
            # Check embeddings
            available_embeddings = []
            unavailable_embeddings = []
            for language, models in SUPPORTED_EMBEDDING_MODELS.items():
                for model_id in models.keys():
                    model_path = Path(model_id)
                    if model_path.exists():
                        available_embeddings.append(f"✅ {language}/{model_id}")
                    else:
                        unavailable_embeddings.append(f"❌ {language}/{model_id}")
            
            # Check rerankers
            available_rerankers = []
            unavailable_rerankers = []
            for model_id in SUPPORTED_RERANK_MODELS.keys():
                model_path = Path(model_id)
                if model_path.exists():
                    available_rerankers.append(f"✅ {model_id}")
                else:
                    unavailable_rerankers.append(f"❌ {model_id}")
            
            # Generate report
            report = "### Model Availability Report\n\n"
            
            report += "#### Available LLMs:\n"
            if available_llms:
                report += "\n".join(available_llms) + "\n\n"
            else:
                report += "_No LLM models found_\n\n"
            
            report += "#### Available Embedding Models:\n"
            if available_embeddings:
                report += "\n".join(available_embeddings) + "\n\n"
            else:
                report += "_No embedding models found_\n\n"
            
            report += "#### Available Reranker Models:\n"
            if available_rerankers:
                report += "\n".join(available_rerankers) + "\n\n"
            else:
                report += "_No reranker models found_\n\n"
            
            report += "\n\n_Note: To use models that are not yet available, run the notebook to convert them first._"
            
            return report
        
        check_models_btn.click(
            fn=check_model_availability,
            inputs=[],
            outputs=[models_availability]
        )
        
        def update_llm_choices(language):
            """Update LLM model choices based on selected language"""
            return gr.Dropdown(choices=list(SUPPORTED_LLM_MODELS[language].keys()))
        
        def update_embedding_choices(language):
            """Update embedding model choices based on selected language"""
            return gr.Dropdown(choices=list(SUPPORTED_EMBEDDING_MODELS[language].keys()))
        
        def reload_models(language, llm_model, embedding_model, reranker_model, device):
            """Reload models with the selected configuration"""
            global embedding, reranker, llm, embedding_model_dir, rerank_model_dir, llm_base_dir, llm_model_dir
            
            try:
                # Update model paths
                embedding_model_dir = Path(embedding_model)
                rerank_model_dir = Path(reranker_model)
                llm_base_dir = Path(llm_model)
                llm_model_dir = llm_base_dir / "INT4_compressed_weights"
                
                # Check if models exist on disk
                missing_models = []
                if not embedding_model_dir.exists():
                    missing_models.append(f"Embedding model: {embedding_model_dir}")
                if not rerank_model_dir.exists():
                    missing_models.append(f"Reranker model: {rerank_model_dir}")
                if not llm_model_dir.exists():
                    missing_models.append(f"LLM model: {llm_model_dir}")
                
                # Get model configurations
                embedding_model_configuration = SUPPORTED_EMBEDDING_MODELS[language][embedding_model]
                rerank_model_configuration = SUPPORTED_RERANK_MODELS[reranker_model]
                llm_model_configuration = SUPPORTED_LLM_MODELS[language][llm_model]
                
                # Initialize models
                initialize_models(device)
                initialize_rag()
                
                # Prepare status messages
                model_statuses = []
                if embedding is not None:
                    model_statuses.append("✅ Embedding model loaded successfully")
                else:
                    model_statuses.append("❌ Embedding model failed to load")
                    
                if reranker is not None:
                    model_statuses.append("✅ Reranker model loaded successfully")
                else:
                    model_statuses.append("❌ Reranker model failed to load")
                    
                if llm is not None:
                    model_statuses.append("✅ LLM loaded successfully")
                else:
                    model_statuses.append("❌ LLM failed to load")
                
                # Update model info display
                model_info_text = f"""
                ### Current Models:
                - LLM: {llm_model} (INT4)
                - Embedding: {embedding_model}
                - Reranker: {reranker_model}
                - Device: {device}
                
                ### Status:
                {chr(10).join(model_statuses)}
                """
                
                # Add warning if models are missing
                status_message = f"Models reloaded successfully on {device} device"
                if missing_models:
                    warning_message = "Warning: The following models were not found on disk:\n" + "\n".join(missing_models)
                    status_message = f"{status_message}\n\n{warning_message}\n\nSome functionality may be limited."
                
                return status_message, model_info_text
            except Exception as e:
                import traceback
                error_msg = f"Error reloading models: {str(e)}\n{traceback.format_exc()}"
                return error_msg, model_info.value
        
        # Set up event handlers for dropdowns
        language_dropdown.change(
            fn=update_llm_choices,
            inputs=[language_dropdown],
            outputs=[llm_model_dropdown]
        )
        
        language_dropdown.change(
            fn=update_embedding_choices,
            inputs=[language_dropdown],
            outputs=[embedding_model_dropdown]
        )
        
        # Add model parameter preview
        model_params_preview = gr.Markdown("Select models to see parameters...")
        
        def show_model_parameters(language, llm_model, embedding_model, reranker_model):
            """Display parameters of the selected models"""
            try:
                preview = "### Model Parameters\n\n"
                
                # LLM parameters
                llm_config = SUPPORTED_LLM_MODELS[language][llm_model]
                preview += "#### LLM Parameters:\n"
                preview += f"- Model ID: `{llm_config.get('model_id', 'N/A')}`\n"
                preview += f"- Remote Code: `{llm_config.get('remote_code', False)}`\n"
                
                # Embedding parameters
                embedding_config = SUPPORTED_EMBEDDING_MODELS[language][embedding_model]
                preview += "\n#### Embedding Parameters:\n"
                preview += f"- Model ID: `{embedding_config.get('model_id', 'N/A')}`\n"
                preview += f"- Mean Pooling: `{embedding_config.get('mean_pooling', False)}`\n"
                preview += f"- Normalize Embeddings: `{embedding_config.get('normalize_embeddings', True)}`\n"
                
                # Reranker parameters
                reranker_config = SUPPORTED_RERANK_MODELS[reranker_model]
                preview += "\n#### Reranker Parameters:\n"
                preview += f"- Model ID: `{reranker_config.get('model_id', 'N/A')}`\n"
                
                return preview
            except Exception as e:
                return f"Error retrieving model parameters: {str(e)}"
        
        # Update model parameters when selections change
        for dropdown in [language_dropdown, llm_model_dropdown, embedding_model_dropdown, reranker_model_dropdown]:
            dropdown.change(
                fn=show_model_parameters,
                inputs=[language_dropdown, llm_model_dropdown, embedding_model_dropdown, reranker_model_dropdown],
                outputs=[model_params_preview]
            )
        
        reload_btn.click(
            fn=reload_models,
            inputs=[
                language_dropdown, 
                llm_model_dropdown, 
                embedding_model_dropdown, 
                reranker_model_dropdown, 
                device_dropdown
            ],
            outputs=[model_status, model_info]
        )
    
    with gr.Tab("Download Messages"):
        gr.Markdown("## Download Messages from Telegram Channels")
        channels_input = gr.Textbox(
            label="Channel Names (comma-separated)",
            placeholder="Enter channel names without @ symbol (e.g., guardian, bloomberg)",
            value="guardian,bloomberg"
        )
        limit_input = gr.Slider(
            minimum=1,
            maximum=1000,
            value=100,
            step=1,
            label="Messages per Channel"
        )
        hours_input = gr.Slider(
            minimum=1,
            maximum=168,
            value=24,
            step=1,
            label="Hours to Look Back"
        )
        download_btn = gr.Button("Download Messages")
        download_status = gr.Textbox(label="Download Status")
        download_preview = gr.Textbox(label="Message Preview", lines=10)
        
    with gr.Tab("Process Messages"):
        gr.Markdown("## Process Downloaded Messages")
        process_btn = gr.Button("Process Messages")
        process_output = gr.Textbox(label="Processing Status")
        
    with gr.Tab("Query Messages"):
        gr.Markdown("## Query Processed Messages")
        query_input = gr.Textbox(
            label="Search Query",
            placeholder="Enter your search query"
        )
        channel_filter = gr.Textbox(
            label="Filter by Channel (Optional)",
            placeholder="Enter channel name to filter results"
        )
        num_results = gr.Slider(
            minimum=1,
            maximum=20,
            value=5,
            step=1,
            label="Number of Results"
        )
        query_btn = gr.Button("Search")
        query_output = gr.Textbox(label="Search Results", lines=15)
    
    with gr.Tab("Question Answering"):
        gr.Markdown("## Ask Questions About Messages")
        gr.Markdown("Answers are streamed in real-time as they are generated, so you can see the response as it's being written.")
        question_input = gr.Textbox(
            label="Question",
            placeholder="Ask a question about the Telegram messages",
            lines=2
        )
        qa_channel_filter = gr.Textbox(
            label="Filter by Channel (Optional)",
            placeholder="Enter channel name to filter results"
        )
        
        with gr.Row():
            with gr.Column():
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature (controls creativity)",
                    info="Lower values (0.1-0.4) give more factual responses. Higher values (0.7-1.0) give more creative responses."
                )
                context_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=5,
                    step=1,
                    label="Number of Messages for Context",
                    info="More context can improve answer quality but may slow down response time."
                )
            with gr.Column():
                show_retrieved_checkbox = gr.Checkbox(
                    label="Show Retrieved Context",
                    value=False,
                    info="Display the messages used to generate the answer"
                )
                repetition_penalty_slider = gr.Slider(
                    minimum=1.0,
                    maximum=1.5,
                    value=1.1,
                    step=0.05,
                    label="Repetition Penalty",
                    info="Higher values reduce repetition in the response"
                )
        
        qa_btn = gr.Button("Get Answer")
        qa_output = gr.Textbox(label="Answer", lines=15)
    
    # Set up event handlers
    download_btn.click(
        fn=download_messages,
        inputs=[channels_input, limit_input, hours_input],
        outputs=[download_status, download_preview]
    )
    
    process_btn.click(
        fn=process_messages,
        inputs=[],
        outputs=process_output
    )
    
    query_btn.click(
        fn=query_messages,
        inputs=[query_input, channel_filter, num_results],
        outputs=query_output
    )
    
    qa_btn.click(
        fn=answer_question_stream,
        inputs=[
            question_input,
            qa_channel_filter,
            temperature_slider,
            context_slider,
            show_retrieved_checkbox,
            repetition_penalty_slider
        ],
        outputs=qa_output
    )

# Update the main function
if __name__ == "__main__":
    import sys
    
    # Check if models exist
    missing_models = []
    if not embedding_model_dir.exists():
        missing_models.append(f"Embedding model: {embedding_model_dir}")
    if not rerank_model_dir.exists():
        missing_models.append(f"Reranker model: {rerank_model_dir}")
    if not llm_model_dir.exists():
        missing_models.append(f"LLM model: {llm_model_dir}")
    
    if missing_models:
        print("WARNING: The following models are missing:")
        for model in missing_models:
            print(f"  - {model}")
        print("\nPlease run the notebook to convert these models first.")
        print("You can still run the app, but some functionality may be limited.")
    
    # Check for Telegram API credentials
    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    
    if not api_id or not api_hash:
        print("WARNING: Telegram API credentials not found in .env file.")
        print("You will not be able to download messages from Telegram.")
        print("Please create a .env file with your TELEGRAM_API_ID and TELEGRAM_API_HASH.")
    
    # Launch the app
    try:
        demo.queue().launch(share=False)
    except Exception as e:
        print(f"Error launching Gradio app: {e}")
        print("Trying with share=True...")
        demo.queue().launch(share=True) 