#!/usr/bin/env python3
"""
NPU-optimized model integration for OpenVINO Messenger Assistant.

This module provides functionality for detecting, downloading, and using 
OpenVINO's NPU-optimized models from Hugging Face.
"""

from pathlib import Path
import os
import json
from typing import Dict, List, Optional, Any

# NPU-optimized models from OpenVINO on HuggingFace
NPU_OPTIMIZED_MODELS = {
    "llm": [
        {
            "name": "Phi-3-mini-4k-instruct-int4-cw-ov",
            "repo_id": "OpenVINO/Phi-3-mini-4k-instruct-int4-cw-ov",
            "display_name": "Phi-3-mini (NPU)",
            "description": "Phi-3 mini model (4k) optimized for NPU",
            "max_new_tokens": 1024,
            "context_window": 4096,
            "type": "instruct",
            "temperature": 0.7
        },
        {
            "name": "Phi-3.5-mini-instruct-int4-cw-ov",
            "repo_id": "OpenVINO/Phi-3.5-mini-instruct-int4-cw-ov",
            "display_name": "Phi-3.5-mini (NPU)",
            "description": "Phi-3.5 mini model optimized for NPU",
            "max_new_tokens": 1024,
            "context_window": 4096,
            "type": "instruct",
            "temperature": 0.7
        },
        {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov",
            "repo_id": "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov",
            "display_name": "DeepSeek-R1-Qwen-1.5B (NPU)",
            "description": "DeepSeek R1 distilled from Qwen 1.5B optimized for NPU",
            "max_new_tokens": 1024,
            "context_window": 4096,
            "type": "chat",
            "temperature": 0.7
        },
        {
            "name": "DeepSeek-R1-Distill-Qwen-7B-int4-cw-ov",
            "repo_id": "OpenVINO/DeepSeek-R1-Distill-Qwen-7B-int4-cw-ov",
            "display_name": "DeepSeek-R1-Qwen-7B (NPU)",
            "description": "DeepSeek R1 distilled from Qwen 7B optimized for NPU",
            "max_new_tokens": 1024,
            "context_window": 4096, 
            "type": "chat",
            "temperature": 0.7
        },
        {
            "name": "Mistral-7B-Instruct-v0.3-int4-cw-ov",
            "repo_id": "OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov",
            "display_name": "Mistral-7B-v0.3 (NPU)",
            "description": "Mistral 7B Instruct v0.3 optimized for NPU",
            "max_new_tokens": 1024,
            "context_window": 4096,
            "type": "instruct",
            "temperature": 0.7
        },
        {
            "name": "Qwen3-8B-int4-cw-ov",
            "repo_id": "OpenVINO/Qwen3-8B-int4-cw-ov",
            "display_name": "Qwen3-8B (NPU)",
            "description": "Qwen3 8B optimized for NPU",
            "max_new_tokens": 1024,
            "context_window": 4096,
            "type": "chat",
            "temperature": 0.7
        }
    ],
    # Currently embedding and reranking models don't have specialized NPU variants
    "embedding": [],
    "rerank": []
}

def get_npu_models(model_type="llm"):
    """
    Get list of available NPU-optimized models for the specified type.
    
    Args:
        model_type: Type of model ("llm", "embedding", or "rerank")
        
    Returns:
        List of NPU-optimized models for the specified type
    """
    return NPU_OPTIMIZED_MODELS.get(model_type, [])

def is_npu_device(device):
    """Check if the specified device is an NPU"""
    return device == "NPU" or "NPU" in device

def download_npu_model(model_info, models_dir=None):
    """
    Download NPU-optimized model from HuggingFace
    
    Args:
        model_info: Dictionary containing model information
        models_dir: Directory to store models (default: .models)
        
    Returns:
        Path to downloaded model directory
    """
    try:
        import huggingface_hub as hf_hub
        
        # Use default models directory if not specified
        if models_dir is None:
            models_dir = Path(".models")
        elif isinstance(models_dir, str):
            models_dir = Path(models_dir)
            
        # Ensure models directory exists
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model directory name
        model_name = model_info["name"]
        model_dir = models_dir / model_name
        
        # Check if model already exists
        if model_dir.exists() and (model_dir / "openvino_model.xml").exists():
            print(f"✅ NPU-optimized {model_name} already exists at {model_dir}")
            return model_dir
            
        # Download model from HuggingFace
        repo_id = model_info["repo_id"]
        print(f"📥 Downloading NPU-optimized {model_name} from {repo_id}...")
        
        local_path = hf_hub.snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        
        if (model_dir / "openvino_model.xml").exists():
            print(f"✅ NPU-optimized {model_name} downloaded successfully to {model_dir}")
            return model_dir
        else:
            print(f"❌ Downloaded model at {model_dir} is missing openvino_model.xml")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading NPU-optimized {model_info['name']}: {e}")
        return None

def add_npu_models_to_config():
    """
    Adds NPU-optimized models to the LLM config
    """
    try:
        from src.llm_config import SUPPORTED_LLM_MODELS
        
        # Add NPU models to English language models
        for model in get_npu_models("llm"):
            SUPPORTED_LLM_MODELS["English"][model["repo_id"]] = {
                "display_name": model["display_name"],
                "description": model["description"],
                "max_new_tokens": model["max_new_tokens"],
                "context_window": model["context_window"],
                "type": model["type"],
                "temperature": model["temperature"],
                "npu_optimized": True  # Flag to identify NPU-optimized models
            }
            
        return True
    except Exception as e:
        print(f"Error adding NPU models to config: {e}")
        return False

def get_npu_model_path(model_info, models_dir=None):
    """
    Get path to NPU-optimized model, downloading if needed
    
    Args:
        model_info: Dictionary containing model information
        models_dir: Directory containing models (default: .models)
        
    Returns:
        Path to model directory
    """
    # Use default models directory if not specified
    if models_dir is None:
        models_dir = Path(".models")
    elif isinstance(models_dir, str):
        models_dir = Path(models_dir)
        
    model_name = model_info["name"]
    model_dir = models_dir / model_name
    
    # Check if model exists
    if model_dir.exists() and (model_dir / "openvino_model.xml").exists():
        return model_dir
        
    # Download if not exists
    return download_npu_model(model_info, models_dir)

def is_npu_compatible_model(model_name):
    """
    Check if a model name matches an NPU-optimized model
    
    Args:
        model_name: Model name or repo ID to check
        
    Returns:
        True if model is NPU-optimized, False otherwise
    """
    for model_type in NPU_OPTIMIZED_MODELS:
        for model in NPU_OPTIMIZED_MODELS[model_type]:
            if model["name"] == model_name or model["repo_id"] == model_name:
                return True
    return False
