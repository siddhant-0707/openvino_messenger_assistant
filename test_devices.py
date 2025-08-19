#!/usr/bin/env python3
"""
Test script to diagnose device detection in OpenVINO
"""

import openvino as ov
import sys

def print_device_info():
    """Print detailed OpenVINO device information"""
    try:
        # Initialize OpenVINO Core
        print("Initializing OpenVINO Core...")
        core = ov.Core()
        
        # Get available devices
        print("\n--- AVAILABLE DEVICES ---")
        available_devices = core.available_devices
        print(f"Devices: {available_devices}")
        
        # Test specific device detection
        print("\n--- CHECKING FOR SPECIFIC DEVICES ---")
        has_cpu = "CPU" in available_devices
        has_gpu = any(device.startswith("GPU") for device in available_devices)
        has_npu = "NPU" in available_devices or "VPUX" in available_devices
        has_auto = "AUTO" in available_devices
        
        print(f"CPU detected: {has_cpu}")
        print(f"GPU detected: {has_gpu}")
        print(f"NPU detected: {has_npu}")
        print(f"AUTO detected: {has_auto}")
        
        # Get detailed device properties
        print("\n--- DEVICE PROPERTIES ---")
        for device in available_devices:
            print(f"\nDevice: {device}")
            try:
                # Get device properties
                properties = []
                
                try:
                    full_name = core.get_property(device, "FULL_DEVICE_NAME")
                    properties.append(("FULL_DEVICE_NAME", full_name))
                except:
                    pass
                
                try:
                    # These properties may not be available for all devices
                    optional_properties = [
                        "OPTIMIZATION_CAPABILITIES", 
                        "AVAILABLE_DEVICES",
                        "RANGE_FOR_STREAMS", 
                        "RANGE_FOR_ASYNC_INFER_REQUESTS",
                        "DEVICE_ARCHITECTURE"
                    ]
                    
                    for prop in optional_properties:
                        try:
                            value = core.get_property(device, prop)
                            properties.append((prop, value))
                        except:
                            # Skip properties that aren't available
                            pass
                except:
                    pass
                    
                # Print all properties we were able to retrieve
                if properties:
                    for prop, value in properties:
                        print(f"  {prop}: {value}")
                else:
                    print("  No additional properties available")
                
            except Exception as e:
                print(f"  Error getting properties: {e}")
        
        # Check OpenVINO version
        print("\n--- OPENVINO VERSION INFO ---")
        print(f"OpenVINO version: {ov.__version__}")
        
        print("\n--- SYSTEM INFO ---")
        import platform
        print(f"System: {platform.system()}")
        print(f"Platform: {platform.platform()}")
        print(f"Python version: {sys.version}")
        
    except Exception as e:
        print(f"Error during device detection: {e}")

if __name__ == "__main__":
    print("OpenVINO Device Detection Diagnostic Tool")
    print("========================================")
    print_device_info()
    print("\nTest completed. Check the output above for device information.")
