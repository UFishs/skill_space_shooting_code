from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, Field
from PIL import Image
import uvicorn

from value_model_codebase.models.value_model import ValueFlowModel, ValueModelConfig
from value_model_codebase.utils.image_transforms import build_image_transform


class PredictJsonRequest(BaseModel):
    """JSON endpoint payload.

    images must be three base64-encoded RGB image strings in the order:
    cam_high, cam_left_wrist, cam_right_wrist.
    state is the raw action chunk from your converter: flattened [160] or nested [16, 10].
    """

    images: List[str] = Field(..., min_length=3, max_length=3)
    state: List[Any]
    ode_steps: Optional[int] = None


class ModelServer:
    def __init__(self, config_path: str, checkpoint_path: str, device: Optional[str] = None):
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        model_config = ckpt.get("model_config") or self.cfg["model"]
        model_config = dict(model_config)
        if model_config.get("target_state_dim") is None:
            model_config["target_state_dim"] = int(self.cfg["model"].get("target_state_dim", 10) or 10)
        self.model_cfg = ValueModelConfig(**model_config)
        self.model = ValueFlowModel(self.model_cfg).to(self.device)
        state_dict = ckpt.get("model", ckpt)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        data_cfg = self.cfg.get("data", {})
        self.image_size = int(data_cfg.get("image_size", 224))
        self.chunk_size = int(data_cfg.get("chunk_size", self.model_cfg.chunk_size))
        self.action_dim = int(data_cfg.get("action_dim", self.model_cfg.action_dim))
        self.default_ode_steps = int(self.cfg.get("train", {}).get("ode_steps", 16))
        self.transform = build_image_transform(self.image_size)

    def _pil_from_bytes(self, data: bytes) -> Image.Image:
        return Image.open(io.BytesIO(data)).convert("RGB")

    def _pil_from_base64(self, text: str) -> Image.Image:
        if "," in text and text.strip().startswith("data:"):
            text = text.split(",", 1)[1]
        return self._pil_from_bytes(base64.b64decode(text))

    def preprocess(self, images: List[Image.Image], state: Any) -> Dict[str, torch.Tensor]:
        if len(images) != 3:
            raise ValueError(f"Expected exactly 3 images, got {len(images)}")
        img_t = torch.stack([self.transform(img) for img in images], dim=0).unsqueeze(0)

        arr = np.asarray(state, dtype=np.float32)
        expected = self.chunk_size * self.action_dim
        if arr.size != expected:
            raise ValueError(
                f"Expected raw state/action chunk with {expected} values "
                f"or shape [{self.chunk_size}, {self.action_dim}], got shape {arr.shape} with {arr.size} values."
            )
        actions = torch.from_numpy(arr.reshape(self.chunk_size, self.action_dim)).unsqueeze(0)
        return {
            "images": img_t.to(self.device),
            "actions": actions.to(self.device),
        }

    @torch.no_grad()
    def predict(self, images: List[Image.Image], state: Any, ode_steps: Optional[int] = None) -> Dict[str, Any]:
        batch = self.preprocess(images, state)
        out = self.model.predict(
            images=batch["images"],
            actions=batch["actions"],
            ode_steps=int(ode_steps or self.default_ode_steps),
        )
        target = out["future_state"].detach().cpu().float().view(-1).tolist()
        value_logits = out["value"].detach().cpu().float().view(-1)
        # value = torch.sigmoid(value_logits).tolist()
        value = value_logits.tolist()
        value_probability = out["value_probability"].detach().cpu().float().view(-1).tolist()
        return {
            "target": target,
            "future_state": target,
            "value": value,
            "value_probability": value_probability,
        }


def create_app(config_path: str, checkpoint_path: str, device: Optional[str] = None) -> FastAPI:
    server = ModelServer(config_path=config_path, checkpoint_path=checkpoint_path, device=device)
    app = FastAPI(title="Value Model Server", version="0.1.0")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "device": str(server.device),
            "chunk_size": server.chunk_size,
            "action_dim": server.action_dim,
            "image_size": server.image_size,
        }

    @app.post("/predict")
    async def predict_multipart(
        cam_high: UploadFile = File(...),
        cam_left_wrist: UploadFile = File(...),
        cam_right_wrist: UploadFile = File(...),
        state: str = Form(...),
        ode_steps: Optional[int] = Form(None),
    ) -> Dict[str, Any]:
        images = [
            server._pil_from_bytes(await cam_high.read()),
            server._pil_from_bytes(await cam_left_wrist.read()),
            server._pil_from_bytes(await cam_right_wrist.read()),
        ]
        parsed_state = json.loads(state)
        return server.predict(images=images, state=parsed_state, ode_steps=ode_steps)

    @app.post("/predict_json")
    async def predict_json(req: PredictJsonRequest) -> Dict[str, Any]:
        images = [server._pil_from_base64(x) for x in req.images]
        return server.predict(images=images, state=req.state, ode_steps=req.ode_steps)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    app = create_app(config_path=args.config, checkpoint_path=args.checkpoint, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
