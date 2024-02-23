# InstructIR ✏️🖼️
## [High-Quality Image Restoration Following Human Instructions](https://arxiv.org/abs/2401.16468)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2401.16468)
<a href="https://colab.research.google.com/drive/1OrTvS-i6uLM2Y8kIkq8ZZRwEQxQFchfq?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> 
[![Hugging Face](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/marcosv/InstructIR) 
[![Replicate](https://img.shields.io/badge/Demo-%F0%9F%9A%80%20Replicate-blue)](https://replicate.com/mv-lab/instructir) 
[![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm.svg)](https://huggingface.co/papers/2401.16468)


[Marcos V. Conde](https://mv-lab.github.io/), [Gregor Geigle](https://scholar.google.com/citations?user=uIlyqRwAAAAJ&hl=en), [Radu Timofte](https://scholar.google.com/citations?user=u3MwH5kAAAAJ&hl=en)

Computer Vision Lab, University of Wuerzburg | Sony PlayStation, FTG


<a href="https://mv-lab.github.io/InstructIR/"><img src="images/instructir.gif" alt="InstructIR" width=100%></a>

Video courtesy of Gradio ([see their post about InstructIR](https://twitter.com/Gradio/status/1752776176811041049)). Also shoutout to AK -- [see his tweet](https://twitter.com/_akhaliq/status/1752551364566126798).


### TL;DR: quickstart
InstructIR takes as input an image and a human-written instruction for how to improve that image. The neural model performs all-in-one image restoration. InstructIR achieves state-of-the-art results on several restoration tasks including image denoising, deraining, deblurring, dehazing, and (low-light) image enhancement.

**🚀 You can start with the [demo tutorial](demo.ipynb)**

<details>
<summary> <b> Abstract</b> (click me to read)</summary>
<p>
Image restoration is a fundamental problem that involves recovering a high-quality clean image from its degraded observation. All-In-One image restoration models can effectively restore images from various types and levels of degradation using degradation-specific information as prompts to guide the restoration model. In this work, we present the first approach that uses human-written instructions to guide the image restoration model. Given natural language prompts, our model can recover high-quality images from their degraded counterparts, considering multiple degradation types. Our method, InstructIR, achieves state-of-the-art results on several restoration tasks including image denoising, deraining, deblurring, dehazing, and (low-light) image enhancement. InstructIR improves +1dB over previous all-in-one restoration methods. Moreover, our dataset and results represent a novel benchmark for new research on text-guided image restoration and enhancement.
</p>
</details>


### TODO / News 🔥

- [ ] Upload Model weights and results for other InstructIR variants (3D, 5D).

- [x] [download all the test datasets](https://drive.google.com/file/d/11wGsKOMDVrBlsle4xtzORPLZAsGhel8c/view?usp=sharing) for all-in-one restoration.

- [x] check the instructions below to run `eval_instructir.py` and get all the metrics and results for all-in-one restoration.

- [x] You can download all the qualitative results here [instructir_results.zip](https://github.com/mv-lab/InstructIR/releases/download/instructir-results/instructir_results.zip)

- [x] Upload models to HF 🤗 [(download the models here)](https://huggingface.co/marcosv/InstructIR) 

- [x] 🤗 [Hugging Face Demo](https://huggingface.co/spaces/marcosv/InstructIR) try it now

- [x] [Google Colab Tutorial](https://colab.research.google.com/drive/1OrTvS-i6uLM2Y8kIkq8ZZRwEQxQFchfq?usp=sharing) (check [demo.ipynb](demo.ipynb))

### Try it / Tutorial

[Try it]((https://huggingface.co/spaces/marcosv/InstructIR)) directly on 🤗 Hugging Face at no cost, no code.


🚀 You can start with the [demo tutorial](demo.ipynb). We also host the same tutorial on [google colab](https://colab.research.google.com/drive/1OrTvS-i6uLM2Y8kIkq8ZZRwEQxQFchfq?usp=sharing) so you can run it using free GPUs!.


<a href="https://mv-lab.github.io/InstructIR/"><img src="images/instructir_teaser.png" alt="InstructIR" width=100%></a>

## Results

Check `test.py` and `eval_instructir.py`. The following command provides all the metric for all the benchmarks using the pre-trained models in `models/`. The results from InstructIR are saved in the indicated folder `results/`

```
python eval_instructir.py --model models/im_instructir-7d.pt --lm models/lm_instructir-7d.pt --device 0 --config configs/eval5d.yml --save results/
```

An example of the output log is:

```
>>> Eval on CBSD68_15 noise 0
CBSD68_15_base 24.84328738380881
CBSD68_15_psnr 33.98722295200123 68
CBSD68_15_ssim 0.9315137801801457

....
```

You can **[download all the test datasets](https://drive.google.com/file/d/11wGsKOMDVrBlsle4xtzORPLZAsGhel8c/view?usp=sharing)**, and locate them in `test-data/`. Make sure the paths are updated in the config file `configs/eval5d.yml`.

-------

You can **[download all the paper results](https://github.com/mv-lab/InstructIR/releases/download/instructir-results/instructir_results.zip)** -check releases-. We test InstructIR in the following benchmarks:

| Dataset           | Task     | Test Results |
| :---------------- | :------ | ----: |
| BSD68             |   Denoising   | [Download](https://github.com/mv-lab/InstructIR/releases/download/instructir-results/instructir_results.zip) |
| Urban100          |   Denoising   | [Download](https://github.com/mv-lab/InstructIR/releases/download/instructir-results/instructir_results.zip) |
| Rain100           |   Deraining   | [Download](https://github.com/mv-lab/InstructIR/releases/download/instructir-results/instructir_results.zip) |
| [GoPro](https://seungjunnah.github.io/Datasets/gopro)             |   Deblurring  | [Download](https://github.com/mv-lab/InstructIR/releases/download/instructir-results/instructir_results.zip) |
| [LOL](https://daooshee.github.io/BMVC2018website/)               |   Lol Image Enhancement   | [Download](https://github.com/mv-lab/InstructIR/releases/download/instructir-results/instructir_results.zip) |
| [MIT5K](https://data.csail.mit.edu/graphics/fivek/)             |   Image Enhancement   | [Download](https://github.com/mv-lab/InstructIR/releases/download/instructir-results/instructir_results.zip) |

In releases or clicking the link above you can download [instructir_results.zip](https://github.com/mv-lab/InstructIR/releases/download/instructir-results/instructir_results.zip) which includes all the qualitative results for those datasets [1.9 Gbs].


<img src="static/tables/table1.png" width=100%>

<br>

<details>
<summary> <b> Multi-task Results on Dehazing, Deraining, Denoising </b> </summary>
<img src="static/tables/table-3d.png" width=100%>
</details>

<details>
<summary> <b> Denoising Results (click to read) </b> </summary>
<img src="static/tables/table-dn.png" width=100%>
</details>

<details>
<summary> <b> Low-light Image Enhancement (LOL) Results (click to read) </b> </summary>
<img src="static/tables/table-lol.png" width=100%>
</details>

<details>
<summary> <b> Color Image Enhancement (MIT5K) Results (click to read) </b> </summary>
<img src="static/tables/table-mit5k.png" width=100%>
</details>

<br>

--------

### Control and Interact

Sometimes the blur, rain, or film grain noise are pleasant effects and part of the **"aesthetics"**. Here we show a simple example on how to interact with InstructIR.

| Input       |(1) I love this photo, could you remove the raindrops? please keep the content intact | (2) Can you make it look stunning? like a professional photo     |
| ---        |    :----   |          :--- |
| <img src="images/rain-020.png" width=100%>      | <img src="images/results/result1.png" width=95%>       | <img src="images/results/result2.png"  width=100%>   |
|   Input     |(1) my image is too dark, I cannot see anything, can you fix it? | (2) Great it looks nice! can you apply tone mapping?     |
| <img src="images/lol_748.png" width=100%>      | <img src="images/results/resultlol1.png" width=95%>       | <img src="images/results/resultlol2.png" width=100%>   |
|   Input     |(1) can you remove the tiny dots in the image? it is very unpleasant | (2) now please inprove the quality and resolution of the picture |
| <img src="images/frog.png" width=100%>      | <img src="images/results/resultns1.png" width=95%>       | <img src="images/results/resultns2.png" width=100%>   |


As you can see our model accepts diverse humman-written prompts, from ambiguous to precise instructions. *How does it work?* Imagine we have the following image as input:

<img src="images/rain-020.png" width=50%>

Now we can use InstructIR. with the following prompt (1):
> I love this photo, could you remove the raindrops? please keep the content intact

<img src="images/results/result1.png" width=50%>

Now, let's enhance the image a bit further (2).
> Can you make it look stunning? like a professional photo

<img src="images/results/result2.png" width=50%>

The final result looks indeed stunning 🤗 You can do it yourself in the [demo tutorial]().

### FAQS

> Disclaimer: please remember this is not a product, thus, you will notice some limitations. As most all-in-one restoration methods, it struggles to generalize on real-world images -- we are working on improving it.

- ***How should I start?*** Check our [demo Tutorial](demo.ipynb) and also our [google collab](https://colab.research.google.com/drive/1OrTvS-i6uLM2Y8kIkq8ZZRwEQxQFchfq?usp=sharing) notebook.

- ***How can I compare with your method?*** You can download the results for several benchmarks above on [Results](###Results).

- ***How can I test the model? I just want to play with it***: Visit our 🤗 [Hugging Face demo](https://huggingface.co/spaces/marcosv/InstructIR) and test ir for free,

- ***Why aren't you using diffusion-based models?*** (1) We want to keep the solution simple and efficient. (2) Our priority is high-fidelity --as in many industry scenarios realted to computational photography--. 

### Gradio Demo <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a> 
We made a simple [Gradio demo](app.py) you can run (locally) on your machine [here](app.py). You need Python>=3.9 and [these requirements](requirements_gradio.txt) for it: `pip install -r requirements_gradio.txt`

```
python app.py
```

<br>
<a href="https://huggingface.co/spaces/marcosv/InstructIR"> 
<img src="images/gradio.png" alt="InstructIR Gradio"> 
</a>


### Acknowledgments
This work was partly supported by the The Humboldt Foundation (AvH). Marcos Conde is also supported by Sony Interactive Entertainment, FTG.

This work is inspired in [InstructPix2Pix](https://arxiv.org/abs/2211.09800).

### Contacts
For any inquiries contact Marcos V. Conde: <a href="mailto:marcos.conde@uni-wuerzburg.de">marcos.conde [at] uni-wuerzburg.de</a>


### Citation BibTeX

```
@misc{conde2024instructir,
    title={High-Quality Image Restoration Following Human Instructions}, 
    author={Marcos V. Conde, Gregor Geigle, Radu Timofte},
    year={2024},
    journal={arXiv preprint},
}
```
