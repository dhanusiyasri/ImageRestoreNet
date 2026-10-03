# SemiconductorAI backend

FastAPI service that serves your trained PyTorch restoration model.

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Your model

`weights/model.pt` is already in place — it's your `best_model.pt`
checkpoint (epoch 92, val_psnr 19.43, val_ssim 0.716).

Its format is a training-state dict: `{"epoch", "model_state",
"val_psnr", "val_ssim"}`. `app/model.py` reads the weights out of
`model_state` and loads them into `WaferRestorationNet`, an
architecture reconstructed to match every tensor shape in your
checkpoint: grayscale (1-channel) in/out, a 3-stage conv+BN encoder
with two stride-2 downsamples, 3 Residual Dense Blocks + a
Squeeze-Excite block at the bottleneck, a 2-stage transpose-conv
decoder with skip connections, and a PixelShuffle head that does a
final 2x super-resolution upscale.

**This was inferred from shapes alone** — it will load your exact
weights without errors, but two implementation details couldn't be
recovered from shapes: the encoder's downsampling method (assumed
strided conv, not maxpool) and the activation function (assumed
ReLU). If restored images look wrong, check `WaferRestorationNet` in
`app/model.py` against your actual training script for those two
things first.

Because of the two downsampling stages, input images are padded to a
multiple of 4 before inference and cropped back out afterward — see
`ModelWrapper.restore()`. Output comes back as a 2x-upscaled
grayscale image (that's the super-resolution step).

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

The API is now at `http://localhost:8000`. Interactive docs at
`http://localhost:8000/docs`.

## Endpoints

| Method | Path                     | Purpose                                   |
|--------|--------------------------|--------------------------------------------|
| GET    | `/api/health`            | Model/service status                       |
| GET    | `/api/model/status`      | Model name, version, device, batch id      |
| GET    | `/api/dashboard/stats`   | Live counters for the Dashboard page       |
| GET    | `/api/dashboard/volume`  | 24h inspection volume series               |
| GET    | `/api/metrics`           | Evaluation metrics (accuracy/precision/…)  |
| POST   | `/api/restore`           | Upload an image, get the restored version + PSNR/SSIM |

## Wiring in real telemetry / eval numbers

`app/state.py` currently keeps counters in memory and seeds a
plausible volume chart so the UI has something to show immediately.
Replace its methods with reads from your real inspection log /
evaluation run output whenever you're ready.
