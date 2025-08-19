import os
import asyncio
import gradio as gr
import nest_asyncio
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from telegram_ingestion import TelegramChannelIngestion
from telegram_rag_integration import TelegramRAGIntegration
import openvino as ov
from openvino_tokenizers import convert_tokenizer
from transformers import AutoTokenizer, AutoModel
from ov_langchain_helper import OpenVINOLLM, OpenVINOBgeEmbeddings, OpenVINOReranker, OpenVINOTextEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA
import json
from datetime import datetime

# Import NPU models integration
from npu_models import (
    get_npu_models, is_npu_device, download_npu_model, 
    add_npu_models_to_config, get_npu_model_path, is_npu_compatible_model
)

from llm_config import (
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_RERANK_MODELS,
    SUPPORTED_LLM_MODELS,
)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Global variables for RAG system
embedding = None
reranker = None
llm = None
retriever = None
rag_chain = None

# Setup directories - organized structure for data and models
models_dir = Path(".models")
models_dir.mkdir(exist_ok=True)

# Use the new consolidated data directory structure
project_root = Path(__file__).parent.parent
data_dir = project_root / "data"
data_dir.mkdir(exist_ok=True)

telegram_data_dir = data_dir / "telegram_data"
telegram_data_dir.mkdir(exist_ok=True)

vector_store_path = data_dir / "telegram_vector_store"
vector_store_path.mkdir(exist_ok=True)

# Default model selections
DEFAULT_LANGUAGE = "English"
DEFAULT_LLM_MODEL = "DeepSeek-R1-Distill-Qwen-1.5B"  # Use exact name from llm_config.py
DEFAULT_EMBEDDING_MODEL = "bge-small-en-v1.5"
DEFAULT_RERANK_MODEL = "bge-reranker-v2-m3"
DEFAULT_DEVICE = "AUTO"
DEFAULT_EMBEDDING_TYPE = "text_embedding_pipeline"  # "text_embedding_pipeline", "openvino_genai" or "legacy"

# Get model configurations
embedding_model_configuration = SUPPORTED_EMBEDDING_MODELS[DEFAULT_LANGUAGE][DEFAULT_EMBEDDING_MODEL]
rerank_model_configuration = SUPPORTED_RERANK_MODELS[DEFAULT_RERANK_MODEL]
llm_model_configuration = SUPPORTED_LLM_MODELS[DEFAULT_LANGUAGE][DEFAULT_LLM_MODEL]

# Use OpenVINO preconverted model paths from Hugging Face
def get_ov_model_path(model_name, precision="int4"):
    """Get the local path for OpenVINO preconverted models"""
    model_path_name = model_name.lower()
    return models_dir / f"{model_path_name}_{precision}_ov"

def download_ov_model_if_needed(model_name, precision="int4", model_type="llm"):
    """Download OpenVINO preconverted model from Hugging Face if not already present"""
    import huggingface_hub as hf_hub
    
    # Convert model name to lowercase for file paths (consistent with Hugging Face repos)
    model_path_name = model_name.lower()
    model_dir = models_dir / f"{model_path_name}_{precision}_ov"
    
    if model_dir.exists() and (model_dir / "openvino_model.xml").exists():
        print(f"✅ OpenVINO {precision.upper()} {model_name} already exists at {model_dir}")
        return model_dir
    
    # Construct OpenVINO model hub ID based on model type
    if model_type == "llm":
        # LLM models follow the pattern: OpenVINO/{model_name}-{precision}-ov
        ov_model_hub_id = f"OpenVINO/{model_path_name}-{precision}-ov"
    elif model_type == "embedding":
        # Embedding models: OpenVINO/bge-base-en-v1.5-int8-ov
        if model_name == "bge-small-en-v1.5":
            ov_model_hub_id = "OpenVINO/bge-base-en-v1.5-int8-ov"
        else:
            ov_model_hub_id = f"OpenVINO/{model_path_name}-{precision}-ov"
    elif model_type == "rerank":
        # Reranking models: OpenVINO/bge-reranker-base-int8-ov  
        if model_name == "bge-reranker-v2-m3":
            ov_model_hub_id = "OpenVINO/bge-reranker-base-int8-ov"
        else:
            ov_model_hub_id = f"OpenVINO/{model_path_name}-{precision}-ov"
    else:
        ov_model_hub_id = f"OpenVINO/{model_path_name}-{precision}-ov"
    
    try:
        print(f"📥 Downloading OpenVINO {precision.upper()} {model_name} from {ov_model_hub_id}...")
        
        # Download the model to our models directory
        local_path = hf_hub.snapshot_download(
            repo_id=ov_model_hub_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        
        if (model_dir / "openvino_model.xml").exists():
            print(f"✅ OpenVINO {precision.upper()} {model_name} downloaded successfully to {model_dir}")
            return model_dir
        else:
            print(f"❌ Downloaded model at {model_dir} is missing openvino_model.xml")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading OpenVINO {precision.upper()} {model_name}: {e}")
        return None

# Model paths - now using OpenVINO preconverted models
embedding_model_dir = download_ov_model_if_needed(DEFAULT_EMBEDDING_MODEL, "int8", "embedding") or Path(DEFAULT_EMBEDDING_MODEL)
rerank_model_dir = download_ov_model_if_needed(DEFAULT_RERANK_MODEL, "int8", "rerank") or Path(DEFAULT_RERANK_MODEL)
llm_model_dir = download_ov_model_if_needed(DEFAULT_LLM_MODEL, "int4", "llm") or get_ov_model_path(DEFAULT_LLM_MODEL, "int4")

# Initialize models
embedding = None
reranker = None
llm = None

def get_available_devices():
    """Get available OpenVINO devices with descriptive GPU names"""
    import openvino as ov
    
    try:
        core = ov.Core()
        available_devices = core.available_devices
        
        # Print diagnostic info
        print(f"Available OpenVINO devices detected: {available_devices}")
        
        # Create detailed device options with descriptive names
        device_options = ["CPU", "AUTO"]
        
        # Check specifically for NPU
        has_npu = "NPU" in available_devices or "VPUX" in available_devices
        if has_npu:
            device_options.append("NPU (Neural Compute)")
            print("NPU device detected and added to options list")
        
        gpu_count = 0
        intel_gpu_found = False
        for device in available_devices:
            if device.startswith("GPU"):
                gpu_count += 1
                try:
                    # Try to get device properties for more info
                    device_name = core.get_property(device, "FULL_DEVICE_NAME")
                    print(f"GPU device found: {device_name}")
                    
                    # Create descriptive names based on known patterns
                    if "Intel" in device_name or "UHD" in device_name or "Iris" in device_name or "Arc" in device_name:
                        descriptive_name = f"GPU.{gpu_count-1} (Intel Graphics)"
                        intel_gpu_found = True
                    elif "NVIDIA" in device_name or "GeForce" in device_name or "RTX" in device_name:
                        descriptive_name = f"GPU.{gpu_count-1} (NVIDIA)"
                    elif "AMD" in device_name or "Radeon" in device_name:
                        descriptive_name = f"GPU.{gpu_count-1} (AMD)"
                    else:
                        # Extract meaningful parts of the device name
                        short_name = device_name.split()[0:2]  # First two words
                        descriptive_name = f"GPU.{gpu_count-1} ({' '.join(short_name)})"
                    
                    device_options.append(descriptive_name)
                    
                except Exception as e:
                    print(f"Error getting GPU properties: {e}")
                    # Fallback if we can't get device properties
                    if gpu_count == 1:
                        device_options.append(f"GPU.0 (Primary)")
                    else:
                        device_options.append(f"GPU.{gpu_count-1} (Secondary)")
        
        # Only add Intel GPU option if we found an Intel GPU
        # This helps avoid duplicate generic GPU entries
        if intel_gpu_found:
            # Don't add a generic GPU option when we already have Intel GPU listed
            pass
        elif gpu_count == 1:
            device_options.append("GPU (Generic)")
        elif gpu_count > 1:
            device_options.append("GPU (Auto-select)")
        
        print(f"Final device options: {device_options}")
        return device_options
    
    except Exception as e:
        print(f"Error getting devices: {e}")
        return ["CPU", "AUTO", "GPU"]

def get_optimized_device_config(device="AUTO"):
    """Get optimized device configuration for OpenVINO models"""
    import openvino.properties as props
    import openvino.properties.hint as hints
    import openvino.properties.streams as streams
    
    config = {
        hints.performance_mode(): hints.PerformanceMode.LATENCY,
        streams.num(): "1",
        props.cache_dir(): "",
    }
    
    if "GPU" in device:
        # GPU-specific optimizations
        config.update({
            # Reduce memory usage
            "GPU_ENABLE_SDPA_OPTIMIZATION": "NO",
            # Limit parallel execution
            hints.execution_mode(): hints.ExecutionMode.ACCURACY,
            # Conservative memory management
            "GPU_MEMORY_OPTIMIZATION": "YES",
        })
    
    return config

def parse_device_name(device_selection):
    """Parse the user-friendly device name to get the actual OpenVINO device ID"""
    # Print diagnostic info
    print(f"Parsing device name: {device_selection}")
    
    if device_selection in ["CPU", "AUTO"]:
        return device_selection
    elif device_selection.startswith("GPU."):
        # Extract GPU number from "GPU.0 (Intel Graphics)" -> "GPU.0"
        gpu_id = device_selection.split(" ")[0]  # "GPU.0"
        return gpu_id
    elif "NPU" in device_selection:
        # Return the standard NPU device name (not the display name)
        print("NPU device selected, using 'NPU' as device ID")
        return "NPU"
    elif "GPU" in device_selection:
        # Generic GPU selection
        return "GPU"
    else:
        return device_selection

def initialize_models(device="AUTO", embedding_type="text_embedding_pipeline"):
    """Initialize all models with the specified device and optimized configuration"""
    global embedding, reranker, llm
    
    # Check if we're using NPU
    using_npu = is_npu_device(device)
    if using_npu:
        print("🔍 Using NPU-optimized models where available")
        # Update model list with NPU models
        add_npu_models_to_config()
    
    # Parse the device selection to get actual OpenVINO device ID
    actual_device = parse_device_name(device)
    
    # For NPU devices, set batch size to 1 for best compatibility
    batch_size = 1 if using_npu else (2 if "GPU" in actual_device else 4)
    
    try:
        # Initialize embedding model with the selected type
        if embedding_model_dir and embedding_model_dir.exists():
            try:
                if embedding_type == "text_embedding_pipeline":
                    # Use the TextEmbeddingPipeline approach from OpenVINO GenAI RAG samples
                    print(f"🔄 Loading TextEmbeddingPipeline...")
                    embedding = OpenVINOTextEmbeddings(
                        model_path=str(embedding_model_dir),
                        device=actual_device,
                        batch_size=2 if "GPU" in actual_device else 4,  # Reduce batch size for GPU
                        show_progress=False,
                    )
                    print(f"✅ TextEmbeddingPipeline loaded from {embedding_model_dir} on {device} ({actual_device})")
                    
                elif embedding_type == "openvino_genai":
                    # Use the legacy BGE implementation for now
                    print(f"🔄 Loading OpenVINO BGE embedding model...")
                    embedding = OpenVINOBgeEmbeddings(
                        model_path=str(embedding_model_dir),
                        device=actual_device,
                        model_kwargs={"device_name": actual_device},
                        encode_kwargs={
                            "mean_pooling": embedding_model_configuration["mean_pooling"],
                            "normalize_embeddings": embedding_model_configuration["normalize_embeddings"],
                            "batch_size": 2 if "GPU" in actual_device else 4,  # Reduce batch size for GPU
                        }
                    )
                    print(f"✅ OpenVINO BGE embedding model loaded from {embedding_model_dir} on {device} ({actual_device})")
                else:
                    # Use the legacy implementation
                    print(f"🔄 Loading legacy OpenVINO embedding model...")
                    embedding_config = {"device_name": actual_device}
                    
                    embedding = OpenVINOBgeEmbeddings(
                        model_path=str(embedding_model_dir),
                        model_kwargs=embedding_config,
                        encode_kwargs={
                            "mean_pooling": embedding_model_configuration["mean_pooling"],
                            "normalize_embeddings": embedding_model_configuration["normalize_embeddings"],
                            "batch_size": 2 if "GPU" in actual_device else 4,  # Reduce batch size for GPU
                        }
                    )
                    print(f"✅ Legacy embedding model loaded from {embedding_model_dir} on {device} ({actual_device})")
                    
            except Exception as e:
                print(f"❌ Error loading embedding model on {device} ({actual_device}): {e}")
                if actual_device != "CPU":
                    print("🔄 Falling back to CPU for embedding model...")
                    try:
                        if embedding_type == "text_embedding_pipeline":
                            embedding = OpenVINOTextEmbeddings(
                                model_path=str(embedding_model_dir),
                                device="CPU",
                                batch_size=4,
                            )
                        elif embedding_type == "openvino_genai":
                            embedding = OpenVINOBgeEmbeddings(
                                model_path=str(embedding_model_dir),
                                model_kwargs={"device_name": "CPU"},
                                encode_kwargs={
                                    "mean_pooling": embedding_model_configuration["mean_pooling"],
                                    "normalize_embeddings": embedding_model_configuration["normalize_embeddings"],
                                    "batch_size": 4,
                                }
                            )
                        else:
                            embedding = OpenVINOBgeEmbeddings(
                                model_path=str(embedding_model_dir),
                                model_kwargs={"device_name": "CPU"},
                                encode_kwargs={
                                    "mean_pooling": embedding_model_configuration["mean_pooling"],
                                    "normalize_embeddings": embedding_model_configuration["normalize_embeddings"],
                                    "batch_size": 4,
                                }
                            )
                        print(f"✅ Embedding model loaded on CPU (fallback)")
                    except Exception as fallback_error:
                        print(f"❌ CPU fallback also failed: {fallback_error}")
                        embedding = None
                else:
                    embedding = None
        else:
            print(f"❌ Embedding model not found at {embedding_model_dir}")
            embedding = None
        
        # Initialize reranker model with simplified config
        if rerank_model_dir and rerank_model_dir.exists():
            try:
                # Simple device config for reranker models
                reranker_config = {"device_name": actual_device}
                
                reranker = OpenVINOReranker(
                    model_path=str(rerank_model_dir),
                    model_kwargs=reranker_config,
                    top_n=3 if "GPU" in actual_device else 5,  # Reduce top_n for GPU to save memory
                )
                print(f"✅ Reranking model loaded from {rerank_model_dir} on {device} ({actual_device})")
            except Exception as e:
                print(f"❌ Error loading reranking model on {device} ({actual_device}): {e}")
                if actual_device != "CPU":
                    print("🔄 Falling back to CPU for reranking model...")
                    reranker = OpenVINOReranker(
                        model_path=str(rerank_model_dir),
                        model_kwargs={"device_name": "CPU"},
                        top_n=5,
                    )
                    print(f"✅ Reranking model loaded on CPU (fallback)")
                else:
                    reranker = None
        else:
            print(f"❌ Reranking model not found at {rerank_model_dir}")
            reranker = None
        
        # Initialize LLM with enhanced config only for supported models
        if llm_model_dir and llm_model_dir.exists():
            try:
                # Enhanced config for LLM models (these support more properties)
                llm_kwargs = {}
                if "GPU" in actual_device:
                    # Only add GPU-specific properties that are known to work
                    import openvino.properties as props
                    import openvino.properties.hint as hints
                    import openvino.properties.streams as streams
                    
                    llm_kwargs = {
                        props.hint.performance_mode(): hints.PerformanceMode.LATENCY,
                        streams.num(): "1",
                        "GPU_ENABLE_SDPA_OPTIMIZATION": "NO",
                    }
                
                llm = OpenVINOLLM.from_model_path(
                    model_path=str(llm_model_dir),
                    device=actual_device,
                    **llm_kwargs
                )
                
                # Set conservative parameters for GPU to avoid memory issues
                if "GPU" in actual_device:
                    llm.config.max_new_tokens = 512  # Reduced for GPU
                    llm.config.temperature = 0.7
                    llm.config.top_p = 0.9
                    llm.config.top_k = 40  # Reduced for GPU
                    llm.config.repetition_penalty = 1.1
                    llm.config.do_sample = True
                else:
                    # Standard parameters for CPU
                    llm.config.max_new_tokens = 1024
                    llm.config.temperature = 0.7
                    llm.config.top_p = 0.9
                    llm.config.top_k = 50
                    llm.config.repetition_penalty = 1.1
                    llm.config.do_sample = True
                
                print(f"✅ LLM loaded from {llm_model_dir} on {device} ({actual_device})")
            except Exception as e:
                print(f"❌ Error loading LLM on {device} ({actual_device}): {e}")
                if actual_device != "CPU":
                    print("🔄 Falling back to CPU for LLM...")
                    llm = OpenVINOLLM.from_model_path(
                        model_path=str(llm_model_dir),
                        device="CPU",
                    )
                    llm.config.max_new_tokens = 1024
                    llm.config.temperature = 0.7
                    llm.config.top_p = 0.9
                    llm.config.top_k = 50
                    llm.config.repetition_penalty = 1.1
                    llm.config.do_sample = True
                    print(f"✅ LLM loaded on CPU (fallback)")
                else:
                    llm = None
        else:
            print(f"❌ LLM not found at {llm_model_dir}")
            llm = None
    
    except Exception as e:
        print(f"❌ Critical error during model initialization: {e}")
        print("🔄 Falling back to CPU for all models...")
        # Fallback to CPU for all models
        if actual_device != "CPU":
            initialize_models("CPU", embedding_type)

def initialize_rag():
    """Initialize RAG system"""
    global retriever, rag_chain
    
    if embedding is None:
        print("❌ Embedding model not available, cannot initialize RAG")
        return False
    
    try:
        # Load existing vector store from new location
        if vector_store_path.exists() and any(vector_store_path.iterdir()):
            vector_store = FAISS.load_local(str(vector_store_path), embedding, allow_dangerous_deserialization=True)
            print(f"📚 Loaded existing vector store from {vector_store_path}")
        else:
            print(f"📚 No existing vector store found at {vector_store_path}")
            return False
        
        # Set up base retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        
        # Wrap with reranker if available
        if reranker is not None:
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=retriever
            )
            print("🔄 RAG system initialized with reranking")
        else:
            print("🔍 RAG system initialized with basic retrieval")
        
        # Set up RAG chain
        if llm is not None:
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            print("💬 RAG system initialized with LLM")
        else:
            print("❌ LLM not available, RAG chain not initialized")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return False

# Initialize models (with new directory structure)
print("📁 Setting up organized directory structure:")
print(f"  - Models: {models_dir.absolute()}")
print(f"  - Data: {data_dir.absolute()}")
print(f"  - Telegram Data: {telegram_data_dir.absolute()}")
print(f"  - Vector Store: {vector_store_path.absolute()}")

# Download/locate models with consistent paths
print("\n🔄 Initializing models...")
embedding_model_dir = download_ov_model_if_needed(DEFAULT_EMBEDDING_MODEL, "int8", "embedding")
if not embedding_model_dir:
    embedding_model_dir = get_ov_model_path(DEFAULT_EMBEDDING_MODEL, "int8")
    print(f"📁 Embedding model path: {embedding_model_dir}")

rerank_model_dir = download_ov_model_if_needed(DEFAULT_RERANK_MODEL, "int8", "rerank")
if not rerank_model_dir:
    rerank_model_dir = get_ov_model_path(DEFAULT_RERANK_MODEL, "int8")
    print(f"📁 Reranker model path: {rerank_model_dir}")

llm_model_dir = download_ov_model_if_needed(DEFAULT_LLM_MODEL, "int4", "llm")
if not llm_model_dir:
    llm_model_dir = get_ov_model_path(DEFAULT_LLM_MODEL, "int4")
    print(f"📁 LLM model path: {llm_model_dir}")

# Initialize models with default device and TextEmbeddingPipeline
print(f"\n⚙️ Loading models on {DEFAULT_DEVICE} with {DEFAULT_EMBEDDING_TYPE}...")
initialize_models(DEFAULT_DEVICE, DEFAULT_EMBEDDING_TYPE)

# Initialize RAG system if possible
print("\n🔗 Setting up RAG system...")
if initialize_rag():
    print("✅ RAG system ready")
else:
    print("⚠️ RAG system not ready - will initialize when vector store is available")

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

async def get_user_channels_async() -> tuple[str, List[Dict]]:
    """Get list of user's subscribed channels (async version)"""
    try:
        api_id = os.getenv("TELEGRAM_API_ID")
        api_hash = os.getenv("TELEGRAM_API_HASH")
        
        if not api_id or not api_hash:
            return "Telegram API credentials not found! Please check your .env file.", []
        
        ingestion = TelegramChannelIngestion(
            api_id=api_id,
            api_hash=api_hash,
            storage_dir=str(telegram_data_dir)
        )
        
        await ingestion.start()
        try:
            channels = await ingestion.get_user_channels()
            
            # Sort channels by type and name
            channels.sort(key=lambda x: (x["type"], x["name"].lower()))
            
            status_msg = f"Found {len(channels)} channels and supergroups"
            return status_msg, channels
            
        finally:
            await ingestion.stop()
            
    except Exception as e:
        import traceback
        error_msg = f"Error fetching channels: {str(e)}\n{traceback.format_exc()}"
        return error_msg, []

def get_user_channels() -> tuple[str, List[Dict]]:
    """Get list of user's subscribed channels (sync wrapper)"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(get_user_channels_async())

def process_messages() -> str:
    """Process downloaded messages into vector store"""
    try:
        if not telegram_data_dir.exists() or not any(telegram_data_dir.iterdir()):
            return "No message data found. Please download messages first."
        
        # Create RAG integration instance with current embedding model
        if embedding is None:
            return "Embedding model not loaded. Please check model configuration."
            
        rag_integration = TelegramRAGIntegration(
            embedding_model=embedding,
            vector_store_path=str(vector_store_path),
            telegram_data_path=str(telegram_data_dir)
        )
        
        rag_integration.process_telegram_data_dir()
        
        # Reinitialize RAG system with new vector store
        initialize_rag()
        
        return "Successfully processed messages into vector store"
    except Exception as e:
        import traceback
        return f"Error processing messages: {str(e)}\n{traceback.format_exc()}"

def query_messages(query: str, channel: str, num_results: int) -> str:
    """Query the vector store for relevant messages"""
    try:
        if not vector_store_path.exists() or not any(vector_store_path.iterdir()):
            return "Vector store not found. Please process messages first."
        
        if retriever is None:
            return "RAG system not initialized. Please check if vector store exists and models are loaded."
        
        # Get relevant documents using the retriever
        try:
            results = retriever.invoke(query)
            # Limit results to requested number
            results = results[:num_results]
        except Exception as e:
            return f"Error retrieving documents: {str(e)}"
        
        if not results:
            return "No results found for your query."
        
        output = []
        for i, doc in enumerate(results, 1):
            # Filter by channel if specified
            if channel and channel.strip():
                doc_channel = doc.metadata.get('channel', '').lower()
                if channel.lower() not in doc_channel:
                    continue
            
            output.append(f"Result {i}:")
            output.append(f"Channel: {doc.metadata.get('channel', 'Unknown')}")
            output.append(f"Date: {doc.metadata.get('date', 'Unknown')}")
            
            # Show content snippet
            content = doc.page_content
            if len(content) > 300:
                content = content[:300] + "..."
            output.append(f"Content: {content}")
            output.append("")
            
        return "\n".join(output) if output else f"No results found for channel '{channel}'"
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
        
        if rag_chain is None:
            return "RAG system not initialized. Please check if models are loaded."
            
        # Update LLM configuration if LLM is available
        if llm is not None:
            llm.config.temperature = temperature
            llm.config.repetition_penalty = repetition_penalty
        
        # Get answer from RAG chain
        try:
            result = rag_chain({"query": question})
            answer = result.get("result", "No answer generated")
            source_docs = result.get("source_documents", [])
            
            # Filter by channel if specified
            if channel and channel.strip() and source_docs:
                filtered_docs = []
                for doc in source_docs:
                    doc_channel = doc.metadata.get('channel', '').lower()
                    if channel.lower() in doc_channel:
                        filtered_docs.append(doc)
                source_docs = filtered_docs[:num_context]
            else:
                source_docs = source_docs[:num_context]
            
            # Format the response
            if show_retrieved and source_docs:
                context_docs = []
                context_docs.append("\n--- Retrieved Context ---")
                for i, doc in enumerate(source_docs, 1):
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
                return f"{answer}\n{context_str}"
            else:
                return answer
                
        except Exception as e:
            return f"Error during RAG query: {str(e)}"
            
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
        if retriever is None:
            return "RAG system not initialized. Please check if vector store exists and models are loaded."
        
        # Get context documents using the retriever
        context_docs = retriever.invoke(question)
        
        # Filter by channel if specified
        if channel and channel.strip():
            filtered_docs = []
            for doc in context_docs:
                doc_channel = doc.metadata.get('channel', '').lower()
                if channel.lower() in doc_channel:
                    filtered_docs.append(doc)
            context_docs = filtered_docs[:num_context]
        else:
            context_docs = context_docs[:num_context]
        
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

def get_available_openvino_llm_models(device=DEFAULT_DEVICE):
    """Fetch available LLM models from OpenVINO Hugging Face collection"""
    import huggingface_hub as hf_hub
    
    # If using NPU, prioritize NPU-optimized models
    if is_npu_device(device):
        # Add NPU models to configuration
        add_npu_models_to_config()
        
        # Get NPU-optimized models
        npu_models = get_npu_models("llm")
        if npu_models:
            print(f"Found {len(npu_models)} NPU-optimized models")
            # Return dictionary of repo_id: model_info
            npu_model_dict = {model["repo_id"]: model for model in npu_models}
            print(f"NPU model dict: {list(npu_model_dict.keys())}")
            return npu_model_dict
    
    # If not NPU or no NPU models found, use regular collection
    try:
        # Get models from the OpenVINO LLM collection
        collection_models = hf_hub.get_collection("OpenVINO/llm-6687aaa2abca3bbcec71a9bd")
        
        available_models = {}
        
        for item in collection_models.items:
            if hasattr(item, 'item_id'):
                repo_name = item.item_id.replace("OpenVINO/", "")
                
                # Skip if doesn't end with -ov
                if not repo_name.endswith("-ov"):
                    continue
                
                # Remove -ov suffix
                model_id = repo_name[:-3]
                
                # Extract precision from the end
                precision = None
                if model_id.endswith("-int4"):
                    precision = "int4"
                    model_name = model_id[:-5]  # Remove -int4
                elif model_id.endswith("-int8"):
                    precision = "int8"
                    model_name = model_id[:-5]  # Remove -int8
                elif model_id.endswith("-fp16"):
                    precision = "fp16"
                    model_name = model_id[:-5]  # Remove -fp16
                else:
                    continue  # Skip if precision not recognized
                
                # Convert model name to standard format
                # Handle special cases in model naming
                model_name = model_name.lower()
                
                # Map common model naming variations
                name_mappings = {
                    "tinyllama-1.1b-chat-v1.0": "tinyllama-1.1b-chat-v1.0",
                    "qwen2.5-3b-instruct": "qwen2.5-3b-instruct",
                    "qwen2.5-7b-instruct": "qwen2.5-7b-instruct", 
                    "qwen2.5-14b-instruct": "qwen2.5-14b-instruct",
                    "qwen2.5-1.5b-instruct": "qwen2.5-1.5b-instruct",
                    "qwen2.5-0.5b-instruct": "qwen2.5-0.5b-instruct",
                    "phi-3.5-mini-instruct": "phi-3.5-mini-instruct",
                    "phi-4-mini-instruct": "phi-4-mini-instruct",
                    "gemma-2-9b-it": "gemma-2-9b-it",
                    "mistral-7b-instruct-v0.1": "mistral-7b-instruct-v0.1",
                    "neural-chat-7b-v3-3": "neural-chat-7b-v3-3",
                    "deepseek-r1-distill-qwen-1.5b": "deepseek-r1-distill-qwen-1.5b",
                    "deepseek-r1-distill-qwen-7b": "deepseek-r1-distill-qwen-7b",
                }
                
                # Use mapping if available, otherwise use the extracted name
                final_model_name = name_mappings.get(model_name, model_name)
                
                if final_model_name not in available_models:
                    available_models[final_model_name] = []
                
                if precision not in available_models[final_model_name]:
                    available_models[final_model_name].append(precision)
        
        # Sort models alphabetically and sort precisions by preference
        sorted_models = {}
        for model_name in sorted(available_models.keys()):
            precision_order = ["int4", "int8", "fp16"]
            sorted_precisions = [p for p in precision_order if p in available_models[model_name]]
            sorted_models[model_name] = sorted_precisions
            
        return sorted_models
    
    except Exception as e:
        print(f"Error fetching OpenVINO models: {e}")
        # Fallback to a curated list of known working models based on the collection
        return {
            "deepseek-r1-distill-qwen-1.5b": ["int4", "int8", "fp16"],
            "deepseek-r1-distill-qwen-7b": ["int4", "int8", "fp16"],
            "deepseek-r1-distill-qwen-14b": ["int4", "int8", "fp16"],
            "gemma-2-9b-it": ["int4", "int8", "fp16"],
            "mistral-7b-instruct-v0.1": ["int4", "int8", "fp16"],
            "neural-chat-7b-v3-3": ["int4", "int8", "fp16"],
            "phi-3.5-mini-instruct": ["int4", "int8", "fp16"],
            "phi-4-mini-instruct": ["int4", "int8", "fp16"],
            "qwen2.5-0.5b-instruct": ["int4", "int8", "fp16"],
            "qwen2.5-1.5b-instruct": ["int4", "int8", "fp16"],
            "qwen2.5-3b-instruct": ["int4", "int8", "fp16"],
            "qwen2.5-7b-instruct": ["int4", "int8", "fp16"],
            "qwen2.5-14b-instruct": ["int4", "int8", "fp16"],
            "tinyllama-1.1b-chat-v1.0": ["int4", "int8", "fp16"],
        }

def get_model_display_name(model_id):
    """Convert model ID to a user-friendly display name"""
    # Check if this is an NPU-optimized model from our list
    npu_models = get_npu_models("llm")
    for model in npu_models:
        if model["repo_id"] == model_id:
            return model["display_name"]
    
    # For models in our configuration
    for lang in SUPPORTED_LLM_MODELS:
        if model_id in SUPPORTED_LLM_MODELS[lang] and "display_name" in SUPPORTED_LLM_MODELS[lang][model_id]:
            return SUPPORTED_LLM_MODELS[lang][model_id]["display_name"]
    
    # Convert to title case and replace hyphens with spaces
    display_name = model_id.replace("-", " ").title()
    
    # Handle specific model naming patterns
    replacements = {
        "Qwen2.5": "Qwen 2.5",
        "Qwen2": "Qwen 2",
        "Qwen3": "Qwen 3",
        "Phi 3.5": "Phi-3.5",
        "Phi 4": "Phi-4",
        "Gemma 2": "Gemma-2",
        "Tinyllama": "TinyLlama",
        "Mistral 7b": "Mistral-7B",
        "Neural Chat": "Neural-Chat",
        "Deepseek R1": "DeepSeek-R1",
        "V1.0": "v1.0",
        "V0.1": "v0.1",
        "It": "Instruct",
        "3b": "3B",
        "7b": "7B",
        "9b": "9B",
        "14b": "14B",
        "1.5b": "1.5B",
        "0.5b": "0.5B",
    }
    
    for old, new in replacements.items():
        display_name = display_name.replace(old, new)
    
    return display_name

# Initialize Gradio interface
with gr.Blocks(title="Telegram RAG System") as demo:
    gr.Markdown("# Telegram RAG System with OpenVINO")
    
    with gr.Tab("Models & Configuration"):
        gr.Markdown("## Model Configuration")
        
        gr.Markdown("""
        ℹ️ **TextEmbeddingPipeline Integration:** This application now uses the native `openvino_genai.TextEmbeddingPipeline` from the [official OpenVINO GenAI RAG samples](https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/python/rag). 
        
        **Key Features:**
        - **Native TextEmbeddingPipeline**: Direct integration with `openvino_genai.TextEmbeddingPipeline(model_dir, device)`
        - **Official RAG Sample Approach**: Based on the exact implementation from OpenVINO GenAI RAG samples
        - **Multiple Implementation Options**: Choose between TextEmbeddingPipeline, OpenVINO GenAI, or Legacy approaches
        
        **Usage Example from Official Samples:**
        ```python
        import openvino_genai
        pipeline = openvino_genai.TextEmbeddingPipeline(model_dir, "CPU")
        embeddings = pipeline.embed_documents(["document1", "document2"])
        ```
        
        LLM models are downloaded from the [OpenVINO LLM collection](https://huggingface.co/collections/OpenVINO/llm-6687aaa2abca3bbcec71a9bd), 
        while embedding and reranking models are downloaded from individual [OpenVINO repositories](https://huggingface.co/OpenVINO).
        All models are cached locally in the `.models` directory.
        """)
        
        # Fetch available models
        with gr.Row():
            refresh_models_btn = gr.Button("🔄 Refresh Available Models", variant="secondary", size="sm")
        
        # Get initial model list (will be updated when device changes)
        initial_device = device_dropdown.value if "device_dropdown" in locals() else DEFAULT_DEVICE
        available_llm_models = get_available_openvino_llm_models(device=initial_device)
        
        gr.Markdown("""
        ### 🖥️ NPU Support
        
        When using Neural Processing Unit (NPU), the application will automatically show 
        NPU-optimized models from the [OpenVINO NPU collection](https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu-686e7f0bf7bc184bd71f8ba0).
        These models are specially quantized and optimized for Intel NPU hardware.
        
        If you have an NPU, select "NPU (Neural Compute)" from the device dropdown to see NPU-optimized models.
        """)
        
        with gr.Row():
            with gr.Column():
                # Language selection (keeping for embedding/rerank models)
                language_dropdown = gr.Dropdown(
                    choices=list(SUPPORTED_LLM_MODELS.keys()),
                    value=DEFAULT_LANGUAGE,
                    label="Language (for embedding/rerank models)"
                )
                
                # Create two separate model selection dropdowns - one for regular models, one for NPU
                # Regular LLM model selection
                regular_llm_choices = [(get_model_display_name(model_id), model_id) for model_id in available_llm_models.keys()]
                llm_model_dropdown = gr.Dropdown(
                    choices=regular_llm_choices,
                    value=DEFAULT_LLM_MODEL if DEFAULT_LLM_MODEL in available_llm_models else (regular_llm_choices[0][1] if regular_llm_choices else None),
                    label="LLM Model",
                    allow_custom_value=False,
                    visible=True,
                    elem_id="regular_llm_dropdown"
                )
                
                # Create a visible notification for NPU mode
                npu_status = gr.Markdown("**[NPU MODE ACTIVATED - SELECT AN NPU MODEL BELOW]**", visible=False)
                
                # NPU-specific model selection (initially hidden)
                npu_models = get_npu_models("llm")
                npu_llm_choices = [(model["display_name"], model["repo_id"]) for model in npu_models]
                print(f"Initializing NPU dropdown with {len(npu_llm_choices)} models: {[c[0] for c in npu_llm_choices]}")
                
                npu_model_dropdown = gr.Dropdown(
                    choices=npu_llm_choices,
                    value=npu_llm_choices[0][1] if npu_llm_choices else None,
                    label="⚡ NPU-Optimized Models ⚡",
                    allow_custom_value=False,
                    visible=False,
                    elem_id="npu_llm_dropdown"
                )
                
                # Precision selection for LLM
                llm_precision_dropdown = gr.Dropdown(
                    choices=available_llm_models.get(DEFAULT_LLM_MODEL, ["int4"]),
                    value="int4",
                    label="LLM Precision",
                    info="Higher precision = better quality, larger size"
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
            choices=get_available_devices(),
            value=DEFAULT_DEVICE,
            label="Device for Models",
            info="Select 'NPU (Neural Compute)' if you have an Intel NPU for specialized models"
        )
        
        # Embedding implementation selection
        embedding_type_dropdown = gr.Dropdown(
            choices=[
                ("TextEmbeddingPipeline (Latest)", "text_embedding_pipeline"),
                ("OpenVINO GenAI", "openvino_genai"),
                ("Legacy OpenVINO", "legacy")
            ],
            value=DEFAULT_EMBEDDING_TYPE,
            label="Embedding Implementation",
            info="TextEmbeddingPipeline uses the official OpenVINO GenAI RAG samples approach"
        )
        
        reload_btn = gr.Button("Reload Models with Selected Configuration")
        model_status = gr.Textbox(label="Model Status", value="Models loaded with default settings")
        
        # Display current model information
        model_info = gr.Markdown(f"""
        ### Current Models:
        - LLM: {get_model_display_name(DEFAULT_LLM_MODEL)} (INT4) - {'✅' if llm_model_dir and llm_model_dir.exists() else '⬬ Will download from HF'}
        - Embedding: {DEFAULT_EMBEDDING_MODEL} (INT8) - {'✅' if embedding_model_dir and embedding_model_dir.exists() else '⬬ Will download from HF'}
        - Reranker: {DEFAULT_RERANK_MODEL} (INT8) - {'✅' if rerank_model_dir and rerank_model_dir.exists() else '⬬ Will download from HF'}
        - Device: {DEFAULT_DEVICE}
        
        💡 Models are automatically downloaded from [OpenVINO's Hugging Face repositories](https://huggingface.co/OpenVINO)
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
                    # Check for OpenVINO preconverted models first
                    ov_model_path = get_ov_model_path(model_id, "int4")
                    legacy_model_path = Path(model_id) / "INT4_compressed_weights"
                    
                    if (ov_model_path.exists() and (ov_model_path / "openvino_model.xml").exists()) or \
                       (legacy_model_path.exists() and (legacy_model_path / "openvino_model.xml").exists()):
                        available_llms.append(f"✅ {language}/{model_id}")
                    else:
                        unavailable_llms.append(f"❌ {language}/{model_id}")
            
            # Check embeddings
            available_embeddings = []
            unavailable_embeddings = []
            for language, models in SUPPORTED_EMBEDDING_MODELS.items():
                for model_id in models.keys():
                    # Check for OpenVINO preconverted models first
                    ov_model_path = get_ov_model_path(model_id, "int8")
                    legacy_model_path = Path(model_id)
                    
                    if (ov_model_path.exists() and (ov_model_path / "openvino_model.xml").exists()) or \
                       (legacy_model_path.exists() and (legacy_model_path / "openvino_model.xml").exists()):
                        available_embeddings.append(f"✅ {language}/{model_id}")
                    else:
                        unavailable_embeddings.append(f"❌ {language}/{model_id}")
            
            # Check rerankers
            available_rerankers = []
            unavailable_rerankers = []
            for model_id in SUPPORTED_RERANK_MODELS.keys():
                # Check for OpenVINO preconverted models first
                ov_model_path = get_ov_model_path(model_id, "int8")
                legacy_model_path = Path(model_id)
                
                if (ov_model_path.exists() and (ov_model_path / "openvino_model.xml").exists()) or \
                   (legacy_model_path.exists() and (legacy_model_path / "openvino_model.xml").exists()):
                    available_rerankers.append(f"✅ {model_id}")
                else:
                    unavailable_rerankers.append(f"❌ {model_id}")
            
            # Format the results
            result_md = "## Model Availability Status\n\n"
            
            if available_llms:
                result_md += "### ✅ Available LLM Models:\n"
                result_md += "\n".join(available_llms) + "\n\n"
            
            if unavailable_llms:
                result_md += "### ❌ Unavailable LLM Models:\n"
                result_md += "\n".join(unavailable_llms) + "\n\n"
            
            if available_embeddings:
                result_md += "### ✅ Available Embedding Models:\n"
                result_md += "\n".join(available_embeddings) + "\n\n"
            
            if unavailable_embeddings:
                result_md += "### ❌ Unavailable Embedding Models:\n"
                result_md += "\n".join(unavailable_embeddings) + "\n\n"
            
            if available_rerankers:
                result_md += "### ✅ Available Reranking Models:\n"
                result_md += "\n".join(available_rerankers) + "\n\n"
            
            if unavailable_rerankers:
                result_md += "### ❌ Unavailable Reranking Models:\n"
                result_md += "\n".join(unavailable_rerankers) + "\n\n"
            
            result_md += "\n💡 **Note**: Unavailable models will be automatically downloaded from OpenVINO's Hugging Face collection when selected."
            
            return result_md
        
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
        
        def refresh_available_models():
            """Refresh the list of available LLM models from OpenVINO collection"""
            try:
                # Get the current device selection to determine which models to show
                current_device = device_dropdown.value
                print(f"Refreshing models for device: {current_device}")
                
                # DIRECT APPROACH: Force show NPU models if device contains "NPU" string
                if "NPU" in current_device:
                    print("NPU device detected - using direct NPU model list")
                    
                    # Directly get NPU models from npu_models.py
                    from src.npu_models import get_npu_models
                    npu_models = get_npu_models("llm")
                    
                    # Create model choices directly from NPU model list
                    llm_choices = [(model["display_name"], model["repo_id"]) for model in npu_models]
                    
                    print(f"DIRECT NPU MODELS: Found {len(llm_choices)} models")
                    for name, repo_id in llm_choices:
                        print(f" - {name}: {repo_id}")
                    
                    status_msg = f"✅ NPU-optimized models loaded ({len(llm_choices)} models)"
                    return gr.Dropdown(choices=llm_choices), status_msg
                
                # Regular flow for non-NPU devices
                available_models = get_available_openvino_llm_models(device=current_device)
                llm_choices = [(get_model_display_name(model_id), model_id) for model_id in available_models.keys()]
                
                return gr.Dropdown(choices=llm_choices), "✅ Models refreshed successfully"
            except Exception as e:
                import traceback
                print(f"Error refreshing models: {e}\n{traceback.format_exc()}")
                return gr.Dropdown(), f"❌ Error refreshing models: {str(e)}"
        
        def update_precision_choices(selected_llm_model):
            """Update available precision choices based on selected LLM model"""
            available_models = get_available_openvino_llm_models()
            if selected_llm_model in available_models:
                precisions = available_models[selected_llm_model]
                # Sort precisions by preference: int4, int8, fp16
                precision_order = ["int4", "int8", "fp16"]
                sorted_precisions = [p for p in precision_order if p in precisions]
                return gr.Dropdown(choices=sorted_precisions, value=sorted_precisions[0] if sorted_precisions else "int4")
            return gr.Dropdown(choices=["int4"], value="int4")
        
        def reload_models(language, llm_model, embedding_model, reranker_model, llm_precision, device, embedding_type):
            """Reload models with the selected configuration"""
            global embedding, reranker, llm, embedding_model_dir, rerank_model_dir, llm_model_dir
            
            try:
                status_msg = f"🔄 Loading models on {device}...\n"
                
                # Check if this is an NPU device and the model is NPU-optimized
                if is_npu_device(device) and is_npu_compatible_model(llm_model):
                    status_msg += f"🛠️ Using NPU-optimized model: {llm_model}\n"
                    
                    # Find the NPU model info
                    npu_models = get_npu_models("llm")
                    npu_model_info = None
                    
                    for model in npu_models:
                        if model["repo_id"] == llm_model:
                            npu_model_info = model
                            break
                    
                    if npu_model_info:
                        # Download NPU-optimized model
                        llm_model_dir = get_npu_model_path(npu_model_info, models_dir)
                        if llm_model_dir is None:
                            status_msg += f"❌ Could not download NPU model for {llm_model}. Falling back.\n"
                            # Fall back to standard model
                            llm_model_dir = download_ov_model_if_needed(llm_model, llm_precision, "llm") or get_ov_model_path(llm_model, llm_precision)
                    else:
                        # Fall back to standard model
                        llm_model_dir = download_ov_model_if_needed(llm_model, llm_precision, "llm") or get_ov_model_path(llm_model, llm_precision)
                else:
                    # Standard model flow
                    # Update model paths using OpenVINO preconverted models
                    llm_model_dir = download_ov_model_if_needed(llm_model, llm_precision, "llm") or get_ov_model_path(llm_model, llm_precision)
                
                # Always use standard flow for embedding and reranking models (no NPU versions yet)
                embedding_model_dir = download_ov_model_if_needed(embedding_model, "int8", "embedding") or Path(embedding_model)
                rerank_model_dir = download_ov_model_if_needed(reranker_model, "int8", "rerank") or Path(reranker_model)
                
                status_msg += f"📁 Model paths updated\n"
                
                # Initialize models with optimized configuration
                initialize_models(device, embedding_type)
                
                # Update model info display
                embedding_impl_display = {
                    "text_embedding_pipeline": "TextEmbeddingPipeline",
                    "openvino_genai": "OpenVINO GenAI",
                    "legacy": "Legacy OpenVINO"
                }.get(embedding_type, embedding_type)
                
                model_info_text = f"""
                ### Current Models:
                - LLM: {get_model_display_name(llm_model)} ({llm_precision.upper()}) - {'✅' if llm and llm_model_dir and llm_model_dir.exists() else '❌'}
                - Embedding: {embedding_model} (INT8, {embedding_impl_display}) - {'✅' if embedding and embedding_model_dir and embedding_model_dir.exists() else '❌'}
                - Reranker: {reranker_model} (INT8) - {'✅' if reranker and rerank_model_dir and rerank_model_dir.exists() else '❌'}
                - Device: {device}
                
                💡 Using {embedding_impl_display} for text embeddings
                """
                
                # Check if all models loaded successfully
                models_loaded = 0
                if embedding: models_loaded += 1
                if reranker: models_loaded += 1
                if llm: models_loaded += 1
                
                if models_loaded == 3:
                    status_msg += "✅ All models loaded successfully!"
                elif models_loaded > 0:
                    status_msg += f"⚠️ {models_loaded}/3 models loaded successfully. Some may have fallen back to CPU."
                else:
                    status_msg += "❌ Failed to load models. Check GPU memory and try CPU device."
                
                return model_info_text, status_msg
                
            except Exception as e:
                error_msg = f"❌ Error reloading models: {str(e)}\n"
                error_msg += "💡 Try using CPU device or smaller models"
                return "❌ Model loading failed", error_msg
        
        # Set up event handlers for dropdowns
        # Function to toggle between regular and NPU model dropdowns
        def toggle_model_dropdowns(device):
            """Toggle visibility of regular and NPU model dropdowns based on device selection"""
            print(f"Toggle function called with device: '{device}'")
            
            # Explicit debug output
            print(f"Device type: {type(device)}")
            print(f"Device contains 'NPU': {'NPU' in str(device)}")
            print(f"Device equals 'NPU': {device == 'NPU'}")
            print(f"Device equals 'NPU (Neural Compute)': {device == 'NPU (Neural Compute)'}")
            
            # Explicit and robust check for NPU
            is_npu_device = False
            if isinstance(device, str) and ("NPU" in device.upper() or "NEURAL" in device.upper()):
                is_npu_device = True
                print("NPU DEVICE DETECTED! Showing NPU dropdown.")
                
            if is_npu_device:
                # Force update model list immediately
                npu_models = get_npu_models("llm")
                print(f"Found {len(npu_models)} NPU models:")
                for m in npu_models:
                    print(f" - {m['display_name']}")
                
                # Show NPU dropdown and status, hide regular dropdown
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
            else:
                print("Non-NPU device selected - showing regular models dropdown")
                # Show regular dropdown, hide NPU dropdown and status
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        
        device_dropdown.change(
            fn=toggle_model_dropdowns,
            inputs=[device_dropdown],
            outputs=[llm_model_dropdown, npu_model_dropdown, npu_status]
        )
        
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

        llm_model_dropdown.change(
            fn=update_precision_choices,
            inputs=[llm_model_dropdown],
            outputs=[llm_precision_dropdown]
        )
        
        refresh_models_btn.click(
            fn=refresh_available_models,
            inputs=[],
            outputs=[llm_model_dropdown, model_status]
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
        
        # Helper function to handle model selection based on which dropdown is visible
        def get_selected_model(regular_model, npu_model, device):
            """Select the appropriate model based on device and visibility"""
            if "NPU" in device:
                print(f"Using NPU model: {npu_model}")
                return npu_model
            else:
                print(f"Using regular model: {regular_model}")
                return regular_model
                
        reload_btn.click(
            fn=lambda lang, reg_model, npu_model, emb_model, rerank_model, precision, device, emb_type: 
                reload_models(
                    lang, 
                    get_selected_model(reg_model, npu_model, device), 
                    emb_model, 
                    rerank_model, 
                    precision, 
                    device, 
                    emb_type
                ),
            inputs=[
                language_dropdown, 
                llm_model_dropdown, 
                npu_model_dropdown,  # Add the NPU model dropdown
                embedding_model_dropdown, 
                reranker_model_dropdown, 
                llm_precision_dropdown, 
                device_dropdown,
                embedding_type_dropdown
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

    # Add GPU diagnostics section
    gr.Markdown("## GPU Diagnostics")
    gpu_info_btn = gr.Button("🔍 Check GPU Memory & Devices", variant="secondary")
    gpu_info_display = gr.Markdown("Click to check GPU information...")
    
    def check_gpu_info():
        """Check GPU devices and memory information"""
        import openvino as ov
        
        try:
            core = ov.Core()
            available_devices = core.available_devices
            
            info_lines = ["### GPU Information\n"]
            
            # Check for GPU devices
            gpu_devices = [d for d in available_devices if d.startswith("GPU")]
            if not gpu_devices:
                info_lines.append("❌ **No GPU devices detected**")
                info_lines.append("- Only CPU inference available")
            else:
                info_lines.append(f"✅ **Found {len(gpu_devices)} GPU device(s):**")
                
                for i, gpu in enumerate(gpu_devices):
                    try:
                        # Get detailed device information
                        device_name = core.get_property(gpu, "FULL_DEVICE_NAME")
                        
                        # Identify GPU type
                        if "Intel" in device_name or "UHD" in device_name or "Iris" in device_name:
                            gpu_type = "🔵 Intel Integrated Graphics"
                            recommendation = "Good for light workloads, embedding models"
                        elif "NVIDIA" in device_name or "GeForce" in device_name or "RTX" in device_name:
                            gpu_type = "🟢 NVIDIA Discrete GPU"
                            recommendation = "Best for LLM inference, high performance"
                        elif "AMD" in device_name or "Radeon" in device_name:
                            gpu_type = "🔴 AMD GPU" 
                            recommendation = "Good performance, may need specific drivers"
                        else:
                            gpu_type = "⚪ Unknown GPU Type"
                            recommendation = "Performance may vary"
                        
                        info_lines.append(f"\n**{gpu} - {gpu_type}**")
                        info_lines.append(f"  - Full Name: `{device_name}`")
                        info_lines.append(f"  - Selection: `GPU.{i} ({gpu_type.split()[1]} {gpu_type.split()[2]})`")
                        info_lines.append(f"  - Recommendation: {recommendation}")
                        
                    except Exception as e:
                        info_lines.append(f"\n**{gpu}**")
                        info_lines.append(f"  - Could not get detailed info: {e}")
            
            # Your specific GPU configuration
            info_lines.append("\n### Your Hardware Configuration:")
            info_lines.append("Based on your system info:")
            info_lines.append("- **GPU.0**: Likely Intel Raptor Lake-S UHD Graphics (Integrated)")
            info_lines.append("- **GPU.1**: Likely NVIDIA GeForce RTX 4070 Max-Q (Discrete)")
            
            # Device selection recommendations
            info_lines.append("\n### Device Selection Recommendations:")
            info_lines.append("🎯 **For Best Performance:**")
            info_lines.append("  - **LLM**: Use `GPU.1 (NVIDIA)` or `AUTO`")
            info_lines.append("  - **Embedding**: Use `CPU` or `GPU.0 (Intel Graphics)`") 
            info_lines.append("  - **Reranking**: Use `CPU` or `GPU.0 (Intel Graphics)`")
            
            info_lines.append("\n⚡ **For Memory Safety:**")
            info_lines.append("  - **All Models**: Use `CPU` (most reliable)")
            info_lines.append("  - **Mixed**: LLM on GPU, others on CPU")
            
            # Memory optimization tips
            info_lines.append("\n### Memory Optimization Tips:")
            if gpu_devices:
                info_lines.append("🎯 **For RTX 4070 Max-Q (8GB VRAM):**")
                info_lines.append("  - ✅ Use INT4 models (smaller memory footprint)")
                info_lines.append("  - ✅ DeepSeek-R1-Distill-Qwen-1.5B (current choice is optimal)")
                info_lines.append("  - ⚠️ Avoid models larger than 3B parameters")
                info_lines.append("  - 🔧 Use max_new_tokens ≤ 512 for GPU inference")
                
                info_lines.append("\n🔧 **Troubleshooting GPU Errors:**")
                info_lines.append("  - Try `CPU` device if GPU fails")
                info_lines.append("  - Use `AUTO` to let OpenVINO choose optimal device")
                info_lines.append("  - Restart application if memory error occurs")
                info_lines.append("  - Close other GPU applications (games, video editing)")
            else:
                info_lines.append("💡 Install GPU drivers to enable GPU acceleration")
            
            # Available devices summary
            info_lines.append(f"\n### All Available OpenVINO Devices:")
            for device in available_devices:
                if device.startswith("GPU"):
                    try:
                        name = core.get_property(device, "FULL_DEVICE_NAME")
                        info_lines.append(f"  - `{device}`: {name}")
                    except:
                        info_lines.append(f"  - `{device}`: (Details unavailable)")
                else:
                    info_lines.append(f"  - `{device}`")
            
            return "\n".join(info_lines)
            
        except Exception as e:
            return f"❌ Error checking GPU info: {str(e)}"
    
    gpu_info_btn.click(
        fn=check_gpu_info,
        inputs=[],
        outputs=[gpu_info_display]
    )

# Update the main function
if __name__ == "__main__":
    import sys
    
    print("\n🚀 Starting Telegram RAG Application with organized directory structure...")
    
    # Check if models exist in new location
    missing_models = []
    if not embedding_model_dir or not embedding_model_dir.exists():
        missing_models.append(f"Embedding model: {DEFAULT_EMBEDDING_MODEL}")
    if not rerank_model_dir or not rerank_model_dir.exists():
        missing_models.append(f"Reranker model: {DEFAULT_RERANK_MODEL}")
    if not llm_model_dir or not llm_model_dir.exists():
        missing_models.append(f"LLM model: {DEFAULT_LLM_MODEL}")
    
    if missing_models:
        print("INFO: The following models will be downloaded automatically when needed:")
        for model in missing_models:
            print(f"  - {model}")
        print(f"\nModels will be downloaded to: {models_dir.absolute()}")
        print("You can still run the application - models download on first use.")
    
    # Initialize models with default device
    initialize_models(DEFAULT_DEVICE)
    
    # Initialize RAG system if possible
    if initialize_rag():
        print("✅ RAG system ready")
    else:
        print("⚠️ RAG system not ready - will initialize when vector store is available")
    
    print(f"\n📁 Directory Structure:")
    print(f"  - Models: {models_dir.absolute()}")
    print(f"  - Data: {data_dir.absolute()}")
    print(f"  - Telegram Data: {telegram_data_dir.absolute()}")
    print(f"  - Vector Store: {vector_store_path.absolute()}")
    
    # Create Gradio interface
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    ) 