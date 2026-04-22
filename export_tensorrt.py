#!/usr/bin/env python3
"""
Export YOLO26s to TensorRT Engine format
NOTE: TensorRT engines are device-specific!
- Run on RTX 3050 for testing
- Re-export on Jetson Orin Nano for deployment
"""

from ultralytics import YOLO
import torch
import os

def main():
    print("="*60)
    print("YOLO26s TensorRT Export")
    print("="*60)
    
    # Verify CUDA
    print(f"\n✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
    else:
        print("❌ CUDA not available! TensorRT export requires GPU.")
        return
    
    # Check if model exists
    model_path = "models/yolo26s.pt"
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        print("Please ensure yolo26s.pt is in the models/ directory")
        return
    
    print(f"\n✓ Loading model: {model_path}")
    model = YOLO(model_path)
    
    print("\n⏳ Exporting to TensorRT engine (this may take 5-10 minutes)...")
    print("   - Format: TensorRT")
    print("   - Precision: FP16")
    print("   - Image size: 640x640")
    print("   - Workspace: 4GB")
    
    try:
        model.export(
            format="engine",
            imgsz=640,
            half=True,  # FP16 optimization for Jetson
            device=0,
            workspace=4,  # 4GB workspace
            verbose=True
        )
        
        output_path = model_path.replace('.pt', '.engine')
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS! TensorRT engine created:")
        print(f"   {output_path}")
        print(f"{'='*60}")
        print("\nNext steps for testing on RTX 3050:")
        print("1. Update config.py:")
        print(f"   YOLO_MODEL_PATH = '{output_path}'")
        print("2. Run: python main.py")
        print("\nExpected FPS improvement: 2-3x faster! 🚀")
        print("\n⚠️  IMPORTANT: When deploying to Jetson Orin Nano:")
        print("   Re-export this engine ON THE JETSON (engines are device-specific)")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        print("\nTroubleshooting:")
        print("- Ensure JetPack SDK is installed")
        print("- Check TensorRT: dpkg -l | grep tensorrt")
        print("- Try reducing workspace: workspace=2")

if __name__ == "__main__":
    main()
