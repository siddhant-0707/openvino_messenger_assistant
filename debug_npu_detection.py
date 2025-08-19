#!/usr/bin/env python3
"""
Script to debug NPU device detection issues
"""

import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def debug_device_detection():
    """Debug the device detection process"""
    try:
        import openvino as ov
        from src.npu_models import is_npu_device
        
        print("OpenVINO Device Detection Debug")
        print("==============================")
        
        # Get available devices
        core = ov.Core()
        available_devices = core.available_devices
        
        print(f"\nRaw OpenVINO devices: {available_devices}")
        
        # Check if NPU is in the raw device list
        raw_npu_found = "NPU" in available_devices or "VPUX" in available_devices
        print(f"Raw NPU device found: {raw_npu_found}")
        
        # Create the device list like the application does
        device_options = ["CPU", "AUTO"]
        
        # Add NPU if found
        if raw_npu_found:
            device_options.append("NPU (Neural Compute)")
            
        # Add GPUs
        for device in available_devices:
            if device.startswith("GPU"):
                try:
                    device_name = core.get_property(device, "FULL_DEVICE_NAME")
                    device_options.append(f"{device} ({device_name})")
                except:
                    device_options.append(device)
        
        print(f"\nDevice options that would appear in dropdown: {device_options}")
        
        # Test NPU detection for each device option
        print("\nTesting NPU detection for each device option:")
        for device in device_options:
            is_npu = is_npu_device(device)
            print(f"  Device '{device}' is detected as NPU: {is_npu}")
        
        # Test the specific "NPU (Neural Compute)" string
        test_string = "NPU (Neural Compute)"
        print(f"\nSpecifically testing '{test_string}':")
        print(f"  Contains 'NPU': {'NPU' in test_string}")
        print(f"  Equals 'NPU': {test_string == 'NPU'}")
        print(f"  is_npu_device(): {is_npu_device(test_string)}")
        
        # Test toggle function logic
        print("\nSimulating device dropdown change:")
        for device in device_options:
            is_npu = "NPU" in device
            print(f"  Device '{device}' would trigger NPU mode: {is_npu}")
            
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    debug_device_detection()
