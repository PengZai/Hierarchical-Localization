import torch
import torch.nn.functional as F

from ..utils.base_model import BaseModel


class RoMa(BaseModel):
    default_conf = {
        "weights": "outdoor",
        "coarse_res": 560,
        "upsample_res": 864,
        "sample_thresh": 0.05,
        "sample_num": 10000,
        "symmetric": True,
        "use_custom_corr": False,
        "upsample_preds": False,
        "with_padding": False,
        "do_compile": False,
    }
    required_inputs = ["image0", "image1"]

    def _init(self, conf):
        from romatch import roma_indoor, roma_outdoor

        torch.set_float32_matmul_precision("highest")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        factory = roma_outdoor if conf["weights"] == "outdoor" else roma_indoor
        self.net = factory(
            device=device,
            coarse_res=conf["coarse_res"],
            upsample_res=conf["upsample_res"],
            symmetric=conf["symmetric"],
            use_custom_corr=conf["use_custom_corr"],
            upsample_preds=conf["upsample_preds"],
            with_padding=conf["with_padding"],
            do_compile=conf["do_compile"],
        )
        self.net.sample_thresh = conf["sample_thresh"]

    def _forward(self, data):
        image0 = data["image0"]
        image1 = data["image1"]

        if image0.shape[1] == 1:
            image0 = image0.repeat(1, 3, 1, 1)
        if image1.shape[1] == 1:
            image1 = image1.repeat(1, 3, 1, 1)

        h0, w0 = image0.shape[-2:]
        h1, w1 = image1.shape[-2:]
        match_h, match_w = max(h0, h1), max(w0, w1)
        match_h = ((match_h + 13) // 14) * 14
        match_w = ((match_w + 13) // 14) * 14

        if (h0, w0) != (match_h, match_w):
            image0 = F.interpolate(
                image0, size=(match_h, match_w), mode="bilinear", align_corners=False
            )
        if (h1, w1) != (match_h, match_w):
            image1 = F.interpolate(
                image1, size=(match_h, match_w), mode="bilinear", align_corners=False
            )

        matches, certainty = self.net.match(image0, image1, batched=True)
        matches, scores = self.net.sample(
            matches, certainty, num=self.conf["sample_num"]
        )
        kpts0, kpts1 = self.net.to_pixel_coordinates(
            matches, match_h, match_w, match_h, match_w
        )

        if match_w != w0:
            kpts0[..., 0] *= w0 / match_w
        if match_h != h0:
            kpts0[..., 1] *= h0 / match_h
        if match_w != w1:
            kpts1[..., 0] *= w1 / match_w
        if match_h != h1:
            kpts1[..., 1] *= h1 / match_h

        return {
            "keypoints0": kpts0,
            "keypoints1": kpts1,
            "scores": scores,
        }
