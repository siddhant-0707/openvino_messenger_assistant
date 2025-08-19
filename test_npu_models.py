#!/usr/bin/env python3
"""
Test script for diagnosing NPU model filtering issues
"""

import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from npu_models import (
    get_npu_models, is_npu_device, add_npu_models_to_config, is_npu_compatible_model
)

def test_npu_device_detection():
    """Test the NPU device detection function"""
    print("\n--- Testing NPU Device Detection ---")
    test_devices = ["CPU", "GPU", "AUTO", "NPU", "NPU (Neural Compute)", "GPU.0"]
    
    for device in test_devices:
        result = is_npu_device(device)
        print(f"Device '{device}' is NPU: {result}")
        
    # This should print True for "NPU" and "NPU (Neural Compute)"

def test_npu_model_listing():
    """Test the NPU model listing function"""
    print("\n--- Testing NPU Model Listing ---")
    npu_models = get_npu_models("llm")
    
    print(f"Found {len(npu_models)} NPU-optimized models:")
    for i, model in enumerate(npu_models):
        print(f"{i+1}. {model['display_name']} - {model['repo_id']}")
        
    # This should print all NPU models defined in npu_models.py

def test_add_models_to_config():
    """Test adding NPU models to the config"""
    print("\n--- Testing Adding NPU Models to Config ---")
    success = add_npu_models_to_config()
    
    if success:
        print("✅ Successfully added NPU models to config")
    else:
        print("❌ Failed to add NPU models to config")
        
    # Try importing the config now
    try:
        from src.llm_config import SUPPORTED_LLM_MODELS
        
        if "English" in SUPPORTED_LLM_MODELS:
            # Check if any NPU models are in the config
            npu_models = get_npu_models("llm")
            npu_repo_ids = [model["repo_id"] for model in npu_models]
            
            found_models = []
            for model_id in npu_repo_ids:
                if model_id in SUPPORTED_LLM_MODELS["English"]:
                    found_models.append(model_id)
            
            if found_models:
                print(f"✅ Found {len(found_models)}/{len(npu_repo_ids)} NPU models in config:")
                for model_id in found_models:
                    print(f" - {model_id}")
            else:
                print("❌ No NPU models found in config")
        else:
            print("❌ 'English' language not found in SUPPORTED_LLM_MODELS")
    except Exception as e:
        print(f"❌ Error importing llm_config: {e}")

def test_model_compatibility():
    """Test the model compatibility check function"""
    print("\n--- Testing NPU Model Compatibility Check ---")
    
    test_models = [
        "OpenVINO/Phi-3-mini-4k-instruct-int4-cw-ov",
        "Phi-3-mini-4k-instruct-int4-cw-ov",
        "Qwen/Qwen3-8B",
        "DeepSeek-R1-Distill-Qwen-7B-int4-cw-ov",
        "some-random-model"
    ]
    
    for model in test_models:
        result = is_npu_compatible_model(model)
        print(f"Model '{model}' is NPU compatible: {result}")
        
    # This should print True for actual NPU models and False for others

def simulate_device_change():
    """Simulate what happens when device changes to NPU"""
    print("\n--- Simulating Device Change to NPU ---")
    
    # This simulates what happens in the UI when device is changed to NPU
    device = "NPU (Neural Compute)"
    print(f"Device changed to: {device}")
    
    if is_npu_device(device):
        print("✅ Device correctly detected as NPU")
        
        # Add NPU models to config
        add_npu_models_to_config()
        
        # Get NPU models
        npu_models = get_npu_models("llm")
        model_ids = [model["repo_id"] for model in npu_models]
        
        print(f"Found {len(model_ids)} NPU model IDs:")
        for model_id in model_ids:
            print(f" - {model_id}")
    else:
        print("❌ Device not detected as NPU")

if __name__ == "__main__":
    print("NPU Model Integration Test Script")
    print("================================")
    
    test_npu_device_detection()
    test_npu_model_listing()
    test_add_models_to_config()
    test_model_compatibility()
    simulate_device_change()
    
    print("\nTest completed. Check the output above to diagnose NPU model filtering issues.")
