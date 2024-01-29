import argparse

import gradio as gr
from PIL import Image
import os
import torch
import numpy as np
import yaml
from huggingface_hub import hf_hub_download
#from gradio_imageslider import ImageSlider

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


hf_hub_download(repo_id="marcosv/InstructIR", filename="im_instructir-7d.pt", local_dir="./")
hf_hub_download(repo_id="marcosv/InstructIR", filename="lm_instructir-7d.pt", local_dir="./")

CONFIG     = "configs/eval5d.yml"
LM_MODEL   = "lm_instructir-7d.pt"
MODEL_NAME = "im_instructir-7d.pt"

# parse config file
with open(os.path.join(CONFIG), "r") as f:
    config = yaml.safe_load(f)

cfg = dict2namespace(config)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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

    with torch.no_grad():
        text_embd, deg_pred = lm_head (lm_embd)
        x_hat = model(y, text_embd)

    restored_img = x_hat.squeeze().permute(1,2,0).clamp_(0, 1).cpu().detach().numpy()
    restored_img = np.clip(restored_img, 0. , 1.)

    restored_img = (restored_img * 255.0).round().astype(np.uint8)  # float32 to uint8
    return Image.fromarray(restored_img) #(image, Image.fromarray(restored_img))



title = "InstructIR ✏️🖼️ 🤗"
description = ''' ## [High-Quality Image Restoration Following Human Instructions](https://github.com/mv-lab/InstructIR)

[Marcos V. Conde](https://scholar.google.com/citations?user=NtB1kjYAAAAJ&hl=en), [Gregor Geigle](https://scholar.google.com/citations?user=uIlyqRwAAAAJ&hl=en), [Radu Timofte](https://scholar.google.com/citations?user=u3MwH5kAAAAJ&hl=en)

Computer Vision Lab, University of Wuerzburg | Sony PlayStation, FTG

### TL;DR: quickstart
***InstructIR takes as input an image and a human-written instruction for how to improve that image.*** 
The (single) neural model performs all-in-one image restoration. InstructIR achieves state-of-the-art results on several restoration tasks including image denoising, deraining, deblurring, dehazing, and (low-light) image enhancement. 
**🚀 You can start with the [demo tutorial.](https://github.com/mv-lab/InstructIR/blob/main/demo.ipynb)** Check [our github](https://github.com/mv-lab/InstructIR) for more information

<details>
<summary> <b> Abstract</b> (click me to read)</summary>
<p>
Image restoration is a fundamental problem that involves recovering a high-quality clean image from its degraded observation. All-In-One image restoration models can effectively restore images from various types and levels of degradation using degradation-specific information as prompts to guide the restoration model. In this work, we present the first approach that uses human-written instructions to guide the image restoration model. Given natural language prompts, our model can recover high-quality images from their degraded counterparts, considering multiple degradation types. Our method, InstructIR, achieves state-of-the-art results on several restoration tasks including image denoising, deraining, deblurring, dehazing, and (low-light) image enhancement. InstructIR improves +1dB over previous all-in-one restoration methods. Moreover, our dataset and results represent a novel benchmark for new research on text-guided image restoration and enhancement.
</p>
</details>

> **Disclaimer:** please remember this is not a product, thus, you will notice some limitations.
**This demo expects an image with some degradations (blur, noise, rain, low-light, haze) and a prompt requesting what should be done.**
Due to the GPU memory limitations, the app might crash if you feed a high-resolution image (2K, 4K). <br>
The model was trained using mostly synthetic data, thus it might not work great on real-world complex images. 
However, it works surprisingly well on real-world foggy and low-light images.
You can also try general image enhancement prompts (e.g., "retouch this image", "enhance the colors") and see how it improves the colors.

<br>
'''


article = "<p style='text-align: center'><a href='https://github.com/mv-lab/InstructIR' target='_blank'>High-Quality Image Restoration Following Human Instructions</a></p>"

#### Image,Prompts examples
examples = [['images/rain-020.png', "I love this photo, could you remove the raindrops? please keep the content intact"],
            ['images/gradio_demo_images/city.jpg', "I took this photo during a foggy day, can you improve it?"], 
            ['images/gradio_demo_images/frog.png', "can you remove the tiny dots in the image? it is very unpleasant"], 
            ["images/lol_748.png", "my image is too dark, I cannot see anything, can you fix it?"], 
            ["images/gopro.png", "I took this photo while I was running, can you stabilize the image? it is too blurry"],
            ["images/a0010.jpg", "please I want this image for my photo album, can you edit it as a photographer"],
            ["images/real_fog.png", "How can I remove the fog and mist from this photo?"]]

css = """
    .image-frame img, .image-container img {
        width: auto;
        height: auto;
        max-width: none;
    }
"""

demo = gr.Interface(
    fn=process_img,
    inputs=[
            gr.Image(type="pil", label="Input"),
            gr.Text(label="Prompt")
    ],
    outputs=[gr.Image(type="pil", label="Ouput")],
    title=title,
    description=description,
    article=article,
    examples=examples,
    css=css,
)

if __name__ == "__main__":
    demo.launch()