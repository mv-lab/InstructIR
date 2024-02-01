# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import numpy as np
import yaml
import torch
from cog import BasePredictor, Input, Path

from utils import *
from models import instructir
from text.models import LanguageModel, LMHead

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        LM_MODEL = "models/lm_instructir-7d.pt"
        MODEL_NAME = "models/im_instructir-7d.pt"
        device = torch.device("cpu")

        with open(os.path.join("configs/eval5d.yml"), "r") as f:
            config = yaml.safe_load(f)

        cfg = dict2namespace(config)

        torch.backends.cudnn.deterministic = True
        self.model = instructir.create_model(
            input_channels=cfg.model.in_ch,
            width=cfg.model.width,
            enc_blks=cfg.model.enc_blks,
            middle_blk_num=cfg.model.middle_blk_num,
            dec_blks=cfg.model.dec_blks,
            txtdim=cfg.model.textdim,
        )

        self.model = self.model.to(device)
        print("IMAGE MODEL CKPT:", MODEL_NAME)
        self.model.load_state_dict(
            torch.load(MODEL_NAME, map_location="cpu"), strict=True
        )

        # Initialize the LanguageModel class
        LMODEL = cfg.llm.model
        self.language_model = LanguageModel(model=LMODEL)
        self.lm_head = LMHead(
            embedding_dim=cfg.llm.model_dim,
            hidden_dim=cfg.llm.embd_dim,
            num_classes=cfg.llm.nclasses,
        )
        self.lm_head = self.lm_head  # .to(device)

        print("LMHEAD MODEL CKPT:", LM_MODEL)
        self.lm_head.load_state_dict(
            torch.load(LM_MODEL, map_location="cpu"), strict=True
        )
        print("Loaded weights!")

    def predict(
        self,
        image: Path = Input(description="Input image."),
        prompt: str = Input(description="Input prompt."),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        seed_everything(SEED=seed)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        image = load_img(str(image))
        out_image = process_img(
            image, prompt, self.language_model, self.model, self.lm_head
        )

        out_path = "/tmp/out.png"
        saveImage(out_path, out_image)

        return Path(out_path)


def process_img(image, prompt, language_model, model, lm_head):
    """
    Given an image and a prompt, we run InstructIR to restore the image following the human prompt.
    image: RGB image as numpy array normalized to [0,1]
    prompt: plain python string,

    returns the restored image as numpy array.
    """

    # Convert the image to tensor
    y = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0)

    # Get the text embedding (and predicted degradation class)
    lm_embd = language_model(prompt)
    lm_embd = lm_embd  # .to(device)
    text_embd, deg_pred = lm_head(lm_embd)

    # Forward pass: Paper Figure 2
    x_hat = model(y, text_embd)

    # convert the restored image <x_hat> into a np array
    restored_img = x_hat[0].permute(1, 2, 0).cpu().detach().numpy()
    restored_img = np.clip(restored_img, 0.0, 1.0)
    return restored_img
