#!/usr/bin/env python3
"""
Interactive script to verify NPU device selection logic
"""

import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_npu_detection_interactive():
    """Interactive test of NPU device detection"""
    from src.npu_models import is_npu_device
    
    print("NPU Device Detection Test")
    print("=======================")
    print("This script will help diagnose NPU detection issues.")
    print("Enter the exact device name as it appears in the dropdown.")
    print("Enter 'q' to quit.\n")
    
    while True:
        # Get user input
        device = input("\nEnter device name to test (exactly as shown in UI): ")
        
        # Check for exit
        if device.lower() == 'q':
            break
        
        # Test the string
        print(f"\nTesting: '{device}'")
        print(f"Type: {type(device)}")
        print(f"Length: {len(device)}")
        print(f"Contains 'NPU': {'NPU' in device}")
        print(f"Contains 'Neural': {'Neural' in device}")
        
        # Test with our function
        detected = is_npu_device(device)
        print(f"is_npu_device(): {detected}")
        
        # Test with our updated robust check
        robust_check = False
        if isinstance(device, str) and ("NPU" in device.upper() or "NEURAL" in device.upper()):
            robust_check = True
        print(f"Robust check: {robust_check}")
        
        # Test character by character
        print("\nCharacter analysis:")
        for i, char in enumerate(device):
            print(f"  Pos {i}: '{char}' (ASCII: {ord(char)})")
        
        # Recommendation
        if not detected and not robust_check:
            print("\nRecommendation: There might be hidden/special characters in the device name.")
            print("Try selecting a different device or modifying the detection logic.")
        elif detected != robust_check:
            print("\nRecommendation: The detection functions disagree - this suggests a bug.")
        else:
            print("\nRecommendation: Device detection seems to be working correctly.")

if __name__ == "__main__":
    test_npu_detection_interactive()
