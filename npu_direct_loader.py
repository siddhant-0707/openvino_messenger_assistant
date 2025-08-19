#!/usr/bin/env python3
"""
Direct NPU model loader for OpenVINO Messenger Assistant.
This script provides a direct way to test NPU model loading.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def list_npu_models():
    """List all available NPU models"""
    from src.npu_models import get_npu_models
    
    print("\n=== NPU-OPTIMIZED MODELS ===\n")
    npu_models = get_npu_models("llm")
    
    if not npu_models:
        print("No NPU-optimized models found!")
        return
    
    print(f"Found {len(npu_models)} NPU-optimized models:\n")
    
    for i, model in enumerate(npu_models):
        print(f"{i+1}. {model['display_name']}")
        print(f"   Repo ID: {model['repo_id']}")
        print(f"   Description: {model['description']}")
        print(f"   Type: {model['type']}")
        print()

def download_npu_model(model_index=0):
    """Download a specific NPU model by index"""
    from src.npu_models import get_npu_models, download_npu_model
    
    npu_models = get_npu_models("llm")
    
    if not npu_models:
        print("No NPU-optimized models found!")
        return
    
    if model_index >= len(npu_models):
        print(f"Invalid model index! Please choose between 0 and {len(npu_models)-1}")
        return
    
    model_info = npu_models[model_index]
    print(f"Downloading NPU model: {model_info['display_name']} ({model_info['repo_id']})")
    
    model_dir = download_npu_model(model_info)
    
    if model_dir:
        print(f"✅ Model downloaded successfully to {model_dir}")
    else:
        print("❌ Failed to download model")

def direct_test_with_model(model_index=0):
    """Direct test of loading an NPU model"""
    from src.npu_models import get_npu_models, download_npu_model
    
    try:
        import openvino as ov
        
        print("\n=== TESTING NPU MODEL LOADING ===\n")
        
        # Check if NPU is available
        core = ov.Core()
        available_devices = core.available_devices
        
        print(f"Available OpenVINO devices: {available_devices}")
        
        if "NPU" not in available_devices:
            print("❌ NPU device not found! Available devices: ", available_devices)
            return
        
        # Get NPU models
        npu_models = get_npu_models("llm")
        
        if not npu_models or model_index >= len(npu_models):
            print("No valid NPU model available!")
            return
        
        model_info = npu_models[model_index]
        print(f"Selected model: {model_info['display_name']} ({model_info['repo_id']})")
        
        # Download model if needed
        model_dir = download_npu_model(model_info)
        
        if not model_dir:
            print("❌ Failed to get model directory")
            return
        
        model_path = model_dir / "openvino_model.xml"
        
        if not model_path.exists():
            print(f"❌ Model file not found at {model_path}")
            return
        
        print(f"Loading model from {model_path} on NPU...")
        
        try:
            # Try to load the model on NPU
            model = core.read_model(model_path)
            compiled_model = core.compile_model(model, "NPU")
            
            print("✅ Model loaded successfully on NPU!")
            print(f"Model inputs: {compiled_model.inputs}")
            print(f"Model outputs: {compiled_model.outputs}")
            
        except Exception as e:
            print(f"❌ Error loading model on NPU: {e}")
            print("Trying on CPU instead...")
            
            try:
                compiled_model = core.compile_model(model, "CPU")
                print("✅ Model loaded successfully on CPU")
            except Exception as e2:
                print(f"❌ Error loading model on CPU: {e2}")
                
    except Exception as e:
        import traceback
        print(f"Error in direct test: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    print("NPU Direct Model Loader")
    print("======================")
    print("This script lets you directly list and test NPU-optimized models.")
    
    list_npu_models()
    
    choice = input("\nDo you want to download and test an NPU model? (y/n): ")
    
    if choice.lower() == 'y':
        try:
            model_index = int(input("Enter model index (0 for first model): "))
            direct_test_with_model(model_index)
        except ValueError:
            print("Invalid input! Using default model (index 0)")
            direct_test_with_model(0)
    else:
        print("Exiting without testing model loading.")
