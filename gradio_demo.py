import argparse

import gradio as gr
from PIL import Image
import os
import torch
import numpy as np
import yaml

from gradio_imageslider import ImageSlider

## local code
from models import instructir
from text.models import LanguageModel, LMHead


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


CONFIG     = "configs/eval5d.yml"
LM_MODEL   = "models/lm_instructir-7d.pt"
MODEL_NAME = "models/im_instructir-7d.pt"

# parse config file
with open(os.path.join(CONFIG), "r") as f:
    config = yaml.safe_load(f)

cfg = dict2namespace(config)

device = torch.device("cuda")
model = instructir.create_model(input_channels =cfg.model.in_ch, width=cfg.model.width, enc_blks = cfg.model.enc_blks, 
                            middle_blk_num = cfg.model.middle_blk_num, dec_blks = cfg.model.dec_blks, txtdim=cfg.model.textdim)
model = model.to(device)
print ("IMAGE MODEL CKPT:", MODEL_NAME)
model.load_state_dict(torch.load(MODEL_NAME, map_location="cpu"), strict=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
LMODEL = cfg.llm.model
language_model = LanguageModel(model=LMODEL)
lm_head = LMHead(embedding_dim=cfg.llm.model_dim, hidden_dim=cfg.llm.embd_dim, num_classes=cfg.llm.nclasses)
lm_head = lm_head.to(device)

print("LMHEAD MODEL CKPT:", LM_MODEL)
lm_head.load_state_dict(torch.load(LM_MODEL, map_location="cpu"), strict=True)


def load_img (filename, norm=True,):
    img = np.array(Image.open(filename).convert("RGB"))
    if norm:
        img = img / 255.
        img = img.astype(np.float32)
    return img


def process_img (image, prompt):
    img = np.array(image)
    img = img / 255.
    img = img.astype(np.float32)
    y = torch.tensor(img).permute(2,0,1).unsqueeze(0).to(device)

    lm_embd = language_model(prompt)
    lm_embd = lm_embd.to(device)
    text_embd, deg_pred = lm_head (lm_embd)

    x_hat = model(y, text_embd)

    restored_img = x_hat.squeeze().permute(1,2,0).clamp_(0, 1).cpu().detach().numpy()
    restored_img = np.clip(restored_img, 0. , 1.)

    restored_img = (restored_img * 255.0).round().astype(np.uint8)  # float32 to uint8
    return (image, Image.fromarray(restored_img))


title = "Demo: InstructIR - Restoring images with user prompts"
description = ''' 
**This demo expects an image with some degradations (blur, noise, low-light, haze, ...) and a prompt requesting what should be done.**
'''
# **Demo notebook can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Swin2SR/Perform_image_super_resolution_with_Swin2SR.ipynb).

article = "<p style='text-align: center'><a href='https://arxiv.org/abs/TODO' target='_blank'>High-Quality Image Restoration Following Human Instructions</a></p>"

examples = [['images/gradio_demo_images/city.jpg', "Remove the haze"], ['images/gradio_demo_images/frog.png', "Clear up the speckles."], ['images/gradio_demo_images/bear.png', "Make the rain disappear!"],
            ["images/gradio_demo_images/car.png", "Remove the noise, please."], ["images/gradio_demo_images/car.png", "Remove the rain, please."]]

gr.Interface(
    fn=process_img,
    inputs=[
            gr.Image(type="pil", label="Input"),
            gr.Text(label="Prompt")
    ],
    outputs=[ImageSlider(position=0.5, type="pil", label="SideBySide")], #gr.Image(type="pil", label="Ouput"),  #
    title=title,
    description=description,
    article=article,
    examples=examples,
).launch(share=True)

# with gr.Blocks() as demo:
#     with gr.Row(equal_height=True):
#         with gr.Column(scale=1):
#             input = gr.Image(type="pil", label="Input")
#         with gr.Column(scale=1):
#             prompt = gr.Text(label="Prompt")
#             process_btn = gr.Button("Process")
#     with gr.Row(equal_height=True):
#         output = gr.Image(type="pil", label="Ouput")
#         slider = ImageSlider(position=0.5, type="pil", label="SideBySide")
#     process_btn.click(fn=process_img, inputs=[input, prompt], outputs=[output, slider])
# demo.launch(share=True)