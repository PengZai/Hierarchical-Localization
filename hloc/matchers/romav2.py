import torch

from ..utils.base_model import BaseModel


class RoMaV2Matcher(BaseModel):
    default_conf = {
        "setting": "precise",
        "sample_num": 10000,
        "compile": False,
    }
    required_inputs = ["image0", "image1"]

    def _init(self, conf):
        from romav2 import RoMaV2

        self.net = RoMaV2(RoMaV2.Cfg(compile=conf["compile"]))
        self.net.apply_setting(conf["setting"])

    def _forward(self, data):
        image0 = data["image0"]
        image1 = data["image1"]
        if image0.shape[1] == 1:
            image0 = image0.repeat(1, 3, 1, 1)
        if image1.shape[1] == 1:
            image1 = image1.repeat(1, 3, 1, 1)

        h0, w0 = image0.shape[-2:]
        h1, w1 = image1.shape[-2:]

        preds = self.net.match(image0, image1)
        matches, overlaps, _, _ = self.net.sample(preds, self.conf["sample_num"])
        kpts0, kpts1 = self.net.to_pixel_coordinates(matches, h0, w0, h1, w1)

        return {
            "keypoints0": kpts0,
            "keypoints1": kpts1,
            "scores": overlaps,
        }
