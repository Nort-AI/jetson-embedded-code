"""
One-time script to export the OSNet-AIN x1.0 Re-ID model from PyTorch to ONNX.
Run once on a machine that has torchreid installed:

    python scripts/export_osnet_onnx.py

Produces: assets/models/osnet_ain_x1_0.onnx
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchreid

OUT_PATH = os.path.join("assets", "models", "osnet_ain_x1_0.onnx")

# 1. Build the model via torchreid (downloads weights automatically)
model = torchreid.models.build_model(
    name="osnet_ain_x1_0",
    num_classes=1000,  # Doesn't matter for feature extraction
    pretrained=True,
)
model.eval()

# The torchreid FeatureExtractor preprocesses crops to (256, 128) and
# normalizes with ImageNet mean/std. The model's forward pass returns
# a 512-d feature vector.

# 2. Create dummy input matching the standard Re-ID crop size
dummy = torch.randn(1, 3, 256, 128)

# 3. Export
torch.onnx.export(
    model,
    dummy,
    OUT_PATH,
    input_names=["input"],
    output_names=["features"],
    dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
    opset_version=17,
)

print(f"✓ Exported OSNet-AIN x1.0 to {OUT_PATH}")
print(f"  Input shape : (batch, 3, 256, 128)")
print(f"  Output shape: (batch, 512)")
