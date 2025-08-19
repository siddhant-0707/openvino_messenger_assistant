#!/usr/bin/env python3
"""
Direct NPU selection script to bypass UI issues
"""

import sys
from pathlib import Path
import os

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
        return []
    
    print(f"Found {len(npu_models)} NPU-optimized models:\n")
    
    for i, model in enumerate(npu_models):
        print(f"{i+1}. {model['display_name']}")
        print(f"   Repo ID: {model['repo_id']}")
        print(f"   Description: {model['description']}")
        print()
    
    return npu_models

def test_device_detection():
    """Test OpenVINO device detection"""
    try:
        import openvino as ov
        
        print("\n=== OPENVINO DEVICE DETECTION ===\n")
        
        core = ov.Core()
        available_devices = core.available_devices
        
        print(f"Available devices: {available_devices}")
        
        # Check if NPU is available
        if "NPU" in available_devices:
            print("✅ NPU device is available!")
            return True
        else:
            print("❌ NPU device is NOT available in OpenVINO!")
            return False
            
    except Exception as e:
        print(f"Error detecting devices: {e}")
        return False

def download_and_test_npu_model(model_index=0):
    """Download and test an NPU model directly"""
    from src.npu_models import get_npu_models, download_npu_model
    
    npu_models = get_npu_models("llm")
    if not npu_models or model_index >= len(npu_models):
        print("Invalid model selection!")
        return False
    
    model_info = npu_models[model_index]
    print(f"Selected model: {model_info['display_name']} ({model_info['repo_id']})")
    
    # Download model
    model_dir = download_npu_model(model_info)
    if not model_dir:
        print("Failed to download/find model!")
        return False
    
    print(f"Model directory: {model_dir}")
    
    # Test loading the model
    try:
        import openvino as ov
        
        core = ov.Core()
        model_path = model_dir / "openvino_model.xml"
        
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            return False
        
        print(f"Loading model from {model_path}...")
        
        model = core.read_model(model_path)
        
        print("Testing on NPU...")
        try:
            compiled_model = core.compile_model(model, "NPU")
            print("✅ Successfully loaded model on NPU!")
            return True
        except Exception as e:
            print(f"❌ Failed to load on NPU: {e}")
            
            print("Testing on CPU instead...")
            try:
                compiled_model = core.compile_model(model, "CPU")
                print("✅ Successfully loaded model on CPU!")
                return True
            except Exception as e2:
                print(f"❌ Failed to load on CPU: {e2}")
                return False
                
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def main():
    """Main function"""
    print("Direct NPU Model Selector")
    print("=======================")
    
    # Test device detection
    has_npu = test_device_detection()
    
    # List available models
    npu_models = list_npu_models()
    
    if not npu_models:
        print("No NPU models available! Exiting.")
        return
    
    # Get user selection
    try:
        choice = int(input("\nEnter model number to download and test (or 0 to exit): "))
        if choice == 0:
            print("Exiting without downloading.")
            return
            
        # Adjust for 0-based index
        model_index = choice - 1
        
        if model_index < 0 or model_index >= len(npu_models):
            print(f"Invalid selection! Please choose between 1 and {len(npu_models)}")
            return
            
        # Download and test the model
        success = download_and_test_npu_model(model_index)
        
        if success:
            print("\n=== SUCCESS ===")
            print("The NPU model was downloaded and loaded successfully.")
            print("\nYou can now use this model in the application by:")
            print("1. Selecting 'NPU (Neural Compute)' from the device dropdown")
            print(f"2. Selecting '{npu_models[model_index]['display_name']}' from the NPU models dropdown")
            print("3. Clicking 'Reload Models'")
        else:
            print("\n=== FAILURE ===")
            print("There was a problem loading the NPU model.")
            print("Please check the error messages above for details.")
        
    except ValueError:
        print("Invalid input! Please enter a number.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
