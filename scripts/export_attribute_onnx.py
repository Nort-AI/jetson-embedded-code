"""
One-time script to export the Backbone_nFC attribute model from PyTorch to ONNX.
Run once on a machine that has torch installed:

    python scripts/export_attribute_onnx.py

Produces: assets/models/attribute_model.onnx
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.utils_body import Backbone_nFC, load_network

MODEL_PATH = os.path.join("assets", "models", "net_last.pth")
OUT_PATH   = os.path.join("assets", "models", "attribute_model.onnx")

# 1. Instantiate and load weights
model = Backbone_nFC(class_num=30)
model = load_network(model, MODEL_PATH)
model.eval()

# 2. Create dummy input matching inference preprocessing (288 × 144)
dummy = torch.randn(1, 3, 288, 144)

# 3. Export
torch.onnx.export(
    model,
    dummy,
    OUT_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)

print(f"✓ Exported attribute model to {OUT_PATH}")
print(f"  Input shape : (batch, 3, 288, 144)")
print(f"  Output shape: (batch, 30)")
