import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForKeypointMatching

from ..utils.base_model import BaseModel


class EfficientLoFTR(BaseModel):
    default_conf = {
        "model_name": "zju-community/efficientloftr",
        "threshold": 0.2,
    }
    required_inputs = ["image0", "image1"]

    def _init(self, conf):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(conf["model_name"])
        self.net = (
            AutoModelForKeypointMatching.from_pretrained(conf["model_name"])
            .to(self.device)
            .eval()
        )

    @staticmethod
    def _to_pil(image: torch.Tensor) -> Image.Image:
        if image.ndim == 4:
            image = image[0]
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        image = image.detach().cpu().clamp(0, 1)
        array = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(array)

    def _forward(self, data):
        image0 = self._to_pil(data["image0"])
        image1 = self._to_pil(data["image1"])

        inputs = self.processor([image0, image1], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.net(**inputs)
        processed = self.processor.post_process_keypoint_matching(
            outputs,
            [[(image0.height, image0.width), (image1.height, image1.width)]],
            threshold=self.conf["threshold"],
        )[0]

        return {
            "keypoints0": processed["keypoints0"].to(self.device),
            "keypoints1": processed["keypoints1"].to(self.device),
            "scores": processed["matching_scores"].to(self.device),
        }
