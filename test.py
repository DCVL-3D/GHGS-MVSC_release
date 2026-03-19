from __future__ import print_function, division

import argparse
import logging
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from lib.loss import l1_loss, l2_loss, ssim, psnr
from lib.human_loader import HumanDataset
from lib.network import GHGSModel
from torch.utils.data import DataLoader
from config.config import Config as config
import lpips
import torch
import warnings
import json
import time
warnings.filterwarnings("ignore", category=UserWarning)
from lib.encoder.models.vggt import VGGT
from lib.utils import set_requires_grad

def init_vggt():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    return model


class HumanRender:
    def __init__(self, cfg_file, phase):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size

        self.model = GHGSModel(self.cfg)
        self.dataset = HumanDataset(self.cfg.dataset, phase=phase)
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=4,
                                 pin_memory=True)

        self.model.cuda()
        if self.cfg.restore_ckpt:
            print(self.cfg.restore_ckpt) 
            self.load_ckpt(self.cfg.restore_ckpt)
        
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))
        self.vggt = init_vggt()
        self.vggt.cuda()
        self.model.eval()
        self.vggt.eval()

    def infer_static(self):
        
        all_lpips=[]
        sample_names=[]
        psnr_list, ssim_list, lpips_list = [], [], []
        for data in tqdm(self.loader):
            data = self.fetch_data(data)
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    feat = self.vggt(data['ref']["image"].squeeze(0))
                
                img_pred, mask_pred, Lgeo  = self.model(data,feat)

                img_gt = data['target']['image'].cuda()
                img_gt = img_gt[:,:,3:-3,3:-3]

                psnr_value = psnr(img_pred, img_gt).mean().double()
                psnr_list.append(psnr_value.item())
                ssim_value = ssim(img_pred, img_gt).mean().double()
                ssim_list.append(ssim_value.item())
                lpips_value = self.loss_fn_vgg(img_pred, img_gt).mean().double()
                lpips_list.append(lpips_value.item())

                all_lpips.append(lpips_value)
                sample_names.append(data['target']['sample_name'][0])

                tmp_novel = img_pred[0].detach()
                tmp_novel *= 255
                tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                tmp_img_name = '%s/%s.jpg' % (
                    self.cfg.test_out_path, data['target']['sample_name'][0])


        val_psnr = np.round(np.mean(np.array(psnr_list)), 4)
        val_ssim = np.round(np.mean(np.array(ssim_list)), 4)
        val_lpips = np.round(np.mean(np.array(lpips_list)), 4)
        print(f"psnr: {val_psnr}")
        print(f"ssim: {val_ssim}")
        print(f"lpips: {val_lpips}")
        metrics = {
            "psnr": val_psnr,
            "ssim": val_ssim,
            "lpips": val_lpips
        }
        with open(os.path.join(self.cfg.test_out_path, 'result.json'), 'w', encoding="utf-8") as f:
            json.dump(metrics, f)

    def fetch_data(self, data):
        for view in ['ref']:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda()
        return data

    def load_ckpt(self, load_path):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=True)
        logging.info(f"Parameter loading done")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_out_path', type=str, required= True)
    arg = parser.parse_args()

    cfg = config()
    cfg_for_train = os.path.join('./config', 'config_thu.yaml')
    cfg.load(cfg_for_train)
    cfg = cfg.get_cfg()

    cfg.defrost()
    cfg.test_out_path = arg.test_out_path
    Path(cfg.test_out_path).mkdir(exist_ok=True, parents=True)

    cfg.freeze()

    render = HumanRender(cfg, phase='test')
    render.infer_static()
