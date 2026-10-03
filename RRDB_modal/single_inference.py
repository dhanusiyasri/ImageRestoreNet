import os
import numpy as np
import torch
from PIL import Image

from models.rrdb_lite import RestorationNet


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# GET INPUT IMAGE FROM USER
# ============================================================

input_image = input(
    "\nEnter the path of the noisy JPEG/JPG image: "
).strip()

if not os.path.exists(input_image):
    raise FileNotFoundError(
        f"File not found: {input_image}"
    )


# ============================================================
# GET OUTPUT NAME
# ============================================================

output_image = input(
    "Enter output filename [restored_output.png]: "
).strip()

if output_image == "":
    output_image = "restored_output.png"


# ============================================================
# MODEL WEIGHTS
# ============================================================

weights = "weights/model_best.pt"

if not os.path.exists(weights):
    raise FileNotFoundError(
        f"Model weights not found: {weights}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = RestorationNet(
    in_channels=1,
    channels=64,
    growth=32,
    num_blocks=8,
    scale=2,
    res_scale=0.2,
    use_noise_head=True
)


checkpoint = torch.load(
    weights,
    map_location=device
)


# Handle checkpoint formats
if isinstance(checkpoint, dict):

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


# Remove DataParallel prefix
state_dict = {
    key.replace("module.", ""): value
    for key, value in state_dict.items()
}


model.load_state_dict(
    state_dict,
    strict=True
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# READ JPEG IMAGE
# ============================================================

print("\nReading input image...")

image = Image.open(input_image)

print("Original image size:", image.size)
print("Original mode:", image.mode)


# ============================================================
# CONVERT TO GRAYSCALE
# ============================================================

image = image.convert("L")


# ============================================================
# CHECK / RESIZE INPUT
# ============================================================

# The model can technically accept arbitrary even H,W,
# but for your trained 2x model, use 128x128 input.

if image.size != (128, 128):

    print(
        "Resizing input to 128 × 128..."
    )

    image = image.resize(
        (128, 128),
        Image.Resampling.BICUBIC
    )


# ============================================================
# CONVERT IMAGE TO NUMPY
# ============================================================

image = np.array(
    image,
    dtype=np.float32
)


# JPEG pixel range:
# 0 → 255
#
# Convert to:
# 0 → 1

image = image / 255.0


print("Input shape:", image.shape)
print("Input min:", image.min())
print("Input max:", image.max())


# ============================================================
# NUMPY → PYTORCH
# ============================================================

# H,W
# ↓
# 1,1,H,W

input_tensor = torch.from_numpy(image)

input_tensor = (
    input_tensor
    .unsqueeze(0)
    .unsqueeze(0)
    .to(device)
)


# ============================================================
# INFERENCE
# ============================================================

print("\nRunning restoration...")

with torch.no_grad():

    output = model(input_tensor)


# ============================================================
# OUTPUT
# ============================================================

output = output.squeeze().cpu().numpy()

print("\nRestored image shape:", output.shape)
print("Output min:", output.min())
print("Output max:", output.max())


# ============================================================
# NORMALIZE OUTPUT FOR IMAGE SAVING
# ============================================================

output = np.clip(
    output,
    0.0,
    1.0
)

output = (
    output * 255
).astype(np.uint8)


# ============================================================
# SAVE RESTORED IMAGE
# ============================================================

Image.fromarray(output).save(
    output_image
)


print("\n========================================")
print("       RESTORATION COMPLETE")
print("========================================")

print("Input :", input_image)
print("Output:", output_image)
print("Size  :", output.shape)