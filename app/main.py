"""FastAPI service + Gradio UI for the dumpsite detector.

    CHECKPOINT_PATH=outputs/resnet50_fpn_best.pth MODEL_NAME=resnet50_fpn \
        uvicorn app.main:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000/demo for the upload UI, or POST an image
to http://localhost:8000/predict for the raw JSON API.

The model is loaded lazily on first request (not at import time), so the
app can start and serve /health even before a checkpoint exists — useful
for container health checks during deployment.
"""

import io
import os
from pathlib import Path
from typing import List, Optional

import gradio as gr
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageDraw

from src.model import get_model_by_name

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "outputs/resnet50_fpn_best.pth")
MODEL_NAME = os.environ.get("MODEL_NAME", "resnet50_fpn")
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.5"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_TO_LABEL = {0: "no_dumpsite", 1: "dumpsite"}

app = FastAPI(
    title="Illegal Dumpsite Detection API",
    description="Upload an aerial/satellite image and get back dumpsite detections.",
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not Path(CHECKPOINT_PATH).exists():
            raise RuntimeError(
                f"No checkpoint at {CHECKPOINT_PATH}. Train one with train.py first, "
                "or set the CHECKPOINT_PATH env var to point at an existing .pth file."
            )
        model = get_model_by_name(MODEL_NAME).to(DEVICE)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        model.eval()
        _model = model
    return _model


def run_inference(image: Image.Image, score_threshold: float = SCORE_THRESHOLD) -> List[dict]:
    array = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.tensor(array).permute(2, 0, 1).to(DEVICE)

    model = get_model()
    with torch.no_grad():
        output = model([tensor])[0]

    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    keep = scores >= score_threshold

    return [
        {
            "box": boxes[i].tolist(),
            "score": float(scores[i]),
            "label": TARGET_TO_LABEL.get(int(labels[i]), str(labels[i])),
        }
        for i in np.nonzero(keep)[0]
    ]


def draw_detections(image: Image.Image, detections: List[dict]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for det in detections:
        x_min, y_min, x_max, y_max = det["box"]
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.text((x_min, max(y_min - 12, 0)), f"{det['label']} {det['score']:.2f}", fill="red")
    return annotated


@app.get("/health")
def health():
    return {"status": "ok", "model_name": MODEL_NAME, "checkpoint_path": CHECKPOINT_PATH, "device": DEVICE}


@app.post("/predict")
async def predict(file: UploadFile = File(...), score_threshold: Optional[float] = None):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image = Image.open(io.BytesIO(await file.read()))
        image.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image.")

    threshold = score_threshold if score_threshold is not None else SCORE_THRESHOLD
    try:
        detections = run_inference(image, threshold)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {"filename": file.filename, "detections": detections}


def _gradio_predict(image: np.ndarray):
    pil_image = Image.fromarray(image)
    try:
        detections = run_inference(pil_image)
    except RuntimeError as e:
        return pil_image, str(e)

    annotated = draw_detections(pil_image, detections)
    summary = "\n".join(f"{d['label']} @ {d['score']:.2f}" for d in detections) or "No detections above threshold."
    return annotated, summary


demo = gr.Interface(
    fn=_gradio_predict,
    inputs=gr.Image(type="numpy", label="Aerial / satellite image"),
    outputs=[gr.Image(type="pil", label="Detections"), gr.Textbox(label="Summary")],
    title="Illegal Dumpsite Detection",
    description="Upload an aerial or satellite image to detect illegal waste dumpsites.",
)

app = gr.mount_gradio_app(app, demo, path="/demo")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
