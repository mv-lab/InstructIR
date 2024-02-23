import os
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from metrics import pt_psnr, calculate_ssim, calculate_psnr
from pytorch_msssim import ssim
from utils import save_rgb


def test_model (model, language_model, lm_head, testsets, device, promptify, savepath="results/"):

    model.eval()
    if language_model:
        language_model.eval()
        lm_head.eval()

    DEG_ACC = []
    derain_datasets = ['Rain100L', 'Rain100H', 'Test100', 'Test1200', 'Test2800']

    with torch.no_grad():

        for testset in testsets:

            if savepath:
                dt_results_path = os.path.join(savepath, testset.name)
                if not os.path.exists(dt_results_path):
                    os.mkdir(dt_results_path)
                    
            print (">>> Eval on", testset.name, testset.degradation, testset.deg_class)

            testset_name = testset.name
            test_dataloader = DataLoader(testset, batch_size=1, num_workers=4, drop_last=True, shuffle=False)
            psnr_dataset = []
            ssim_dataset = []
            psnr_noisy   = []
            use_y_channel= False

            if testset.name in derain_datasets:
                use_y_channel = True
                psnr_y_dataset = []
                ssim_y_dataset = []

            for idx, batch in enumerate(test_dataloader):

                x = batch[0].to(device) # HQ image
                y = batch[1].to(device) # LQ image
                f = batch[2][0]         # filename
                t = [promptify(testset.degradation) for _ in range(x.shape[0])]

                if language_model:
                    if idx < 5:
                        # print the input prompt for debugging
                        print("\tInput prompt:", t)

                    lm_embd = language_model(t)
                    lm_embd = lm_embd.to(device)
                    text_embd, deg_pred = lm_head (lm_embd)

                    x_hat = model(y, text_embd)

                psnr_restore = torch.mean(pt_psnr(x, x_hat))
                psnr_dataset.append(psnr_restore.item())
                ssim_restore = ssim(x, x_hat, data_range=1., size_average=True)
                ssim_dataset.append(ssim_restore.item())
                psnr_base    = torch.mean(pt_psnr(x, y))
                psnr_noisy.append(psnr_base.item())

                if use_y_channel:
                    _x_hat = np.clip(x_hat[0].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)
                    _x     = np.clip(x[0].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)
                    _x_hat = (_x_hat*255).astype(np.uint8)
                    _x     = (_x*255).astype(np.uint8)
                    
                    psnr_y = calculate_psnr(_x, _x_hat, crop_border=0, input_order='HWC', test_y_channel=True)
                    ssim_y = calculate_ssim(_x, _x_hat, crop_border=0, input_order='HWC', test_y_channel=True)
                    psnr_y_dataset.append(psnr_y)
                    ssim_y_dataset.append(ssim_y)
                
                ## SAVE RESULTS
                if savepath:
                    restored_img = np.clip(x_hat[0].permute(1,2,0).cpu().detach().numpy(), 0, 1).astype(np.float32)
                    img_name = f.split("/")[-1]
                    save_rgb (restored_img, os.path.join(dt_results_path, img_name))
                    

            print(f"{testset_name}_base", np.mean(psnr_noisy), "Total images:", len(psnr_dataset)) 
            print(f"{testset_name}_psnr", np.mean(psnr_dataset))
            print(f"{testset_name}_ssim", np.mean(ssim_dataset))
            if use_y_channel:
                print(f"{testset_name}_psnr-Y", np.mean(psnr_y_dataset), len(psnr_y_dataset))
                print(f"{testset_name}_ssim-Y", np.mean(ssim_y_dataset))
            
            print (); print (25 * "***")

            del test_dataloader,psnr_dataset, psnr_noisy; gc.collect()
            

        # END OF FUNCTION