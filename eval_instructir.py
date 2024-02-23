import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms.functional as TF

import json
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import yaml
import random
import gc

from utils import *
from models import instructir

from text.models import LanguageModel, LMHead

from test import test_model


def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True 


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  type=str, default='configs/eval5d.yml', help='Path to config file')
    parser.add_argument('--model',   type=str, default="models/im_instructir-7d.pt", help='Path to the image model weights')
    parser.add_argument('--lm',      type=str, default="models/lm_instructir-7d.pt", help='Path to the language model weights')
    parser.add_argument('--promptify', type=str, default="simple_augment")
    parser.add_argument('--device',  type=int, default=0, help="GPU device")
    parser.add_argument('--debug',   action='store_true', help="Debug mode")
    parser.add_argument('--save',    type=str, default='results/', help="Path to save the resultant images")
    args = parser.parse_args()

    SEED=42
    seed_everything(SEED=SEED)
    torch.backends.cudnn.deterministic = True

    GPU        = args.device
    DEBUG      = args.debug
    MODEL_NAME = args.model
    CONFIG     = args.config
    LM_MODEL   = args.lm
    SAVE_PATH  = args.save

    print ('CUDA GPU available: ', torch.cuda.is_available())

    torch.cuda.set_device(f'cuda:{GPU}')
    device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else "cpu")
    print ('CUDA visible devices: ' + str(torch.cuda.device_count()))
    print ('CUDA current device: ', torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))

    # parse config file
    with open(os.path.join(CONFIG), "r") as f:
        config = yaml.safe_load(f)

    cfg = dict2namespace(config)


    print (20*"****")
    print ("EVALUATION")
    print (MODEL_NAME, LM_MODEL, device, DEBUG, CONFIG, args.promptify)
    print (20*"****")

    ################### TESTING DATASET

    TESTSETS      = []
    dn_testsets   = []
    rain_testsets = []

    # Denoising
    try:
        for testset in cfg.test.dn_datasets:
            for sigma in cfg.test.dn_sigmas:
                noisy_testpath = os.path.join(cfg.test.dn_datapath, testset+ f"_{sigma}")
                clean_testpath = os.path.join(cfg.test.dn_datapath, testset)
                #print (clean_testpath, noisy_testpath)
                dn_testsets.append([clean_testpath, noisy_testpath])
    except:
        dn_testsets = []

    # RAIN
    try:
        for noisy_testpath, clean_testpath in zip(cfg.test.rain_inputs, cfg.test.rain_targets):
            rain_testsets.append([clean_testpath, noisy_testpath])
    except:
        rain_testsets = []

    # HAZE
    try:
        haze_testsets = [[cfg.test.haze_targets, cfg.test.haze_inputs]]
    except:
        haze_testsets = []

    # BLUR
    try:
        blur_testsets = [[cfg.test.gopro_targets, cfg.test.gopro_inputs]]
    except:
        blur_testsets = []

    # LOL
    try:
        lol_testsets = [[cfg.test.lol_targets, cfg.test.lol_inputs]]
    except:
        lol_testsets = []

    # MIT5K
    try:
        mit_testsets = [[cfg.test.mit_targets, cfg.test.mit_inputs]]
    except:
        mit_testsets = []

    TESTSETS += dn_testsets
    TESTSETS += rain_testsets
    TESTSETS += haze_testsets
    TESTSETS += blur_testsets
    TESTSETS += lol_testsets
    TESTSETS += mit_testsets
    
    # print ("Tests:", TESTSETS)
    print ("TOTAL TESTSET:", len(TESTSETS))
    print (20 * "----")
    

    ################### RESTORATION MODEL

    print ("Creating InstructIR")
    model = instructir.create_model(input_channels =cfg.model.in_ch, width=cfg.model.width, enc_blks = cfg.model.enc_blks, 
                                middle_blk_num = cfg.model.middle_blk_num, dec_blks = cfg.model.dec_blks, txtdim=cfg.model.textdim)
    
    ################### LOAD IMAGE MODEL

    assert MODEL_NAME, "Model weights required for evaluation"
    
    print ("IMAGE MODEL CKPT:", MODEL_NAME)
    model.load_state_dict(torch.load(MODEL_NAME), strict=True)

    model = model.to(device)

    nparams   = count_params (model)
    print ("Loaded weights!", nparams / 1e6)

    ################### LANGUAGE MODEL

    try: 
        PROMPT_DB  = cfg.llm.text_db
    except:
        PROMPT_DB  = None

    if cfg.model.use_text:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Initialize the LanguageModel class
        LMODEL = cfg.llm.model
        language_model = LanguageModel(model=LMODEL)
        lm_head = LMHead(embedding_dim=cfg.llm.model_dim, hidden_dim=cfg.llm.embd_dim, num_classes=cfg.llm.nclasses)
        lm_head = lm_head.to(device)
        lm_nparams   = count_params (lm_head)

        print ("LMHEAD MODEL CKPT:", LM_MODEL)
        lm_head.load_state_dict(torch.load(LM_MODEL), strict=True)
        print ("Loaded weights!")

    else:
        LMODEL = None
        language_model = None
        lm_head = None
        lm_nparams = 0

    print (20 * "----")

    ################### TESTING !!

    from datasets import RefDegImage, augment_prompt, create_testsets

    if args.promptify == "simple_augment":
        promptify = augment_prompt
    elif args.promptify == "chatgpt":
        prompts = json.load(open(cfg.llm.text_db))
        def promptify(deg):

            return np.random.choice(prompts[deg])
    else:
        def promptify(deg):
            return args.promptify
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    test_datasets = create_testsets(TESTSETS, debug=True)

    test_model (model, language_model, lm_head, test_datasets, device, promptify, savepath=SAVE_PATH)
    