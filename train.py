from __future__ import print_function, division

import logging

import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm 
from datetime import datetime

from lib.human_loader import HumanDataset
from lib.network import HumanModel
from config.config import Config as config
from lib.train_recoder import Logger, file_backup
from lib.loss import l1_loss, l2_loss, psnr
from argparse import ArgumentParser
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
from lib.utils import set_requires_grad
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import lpips
from PIL import Image
from vggt.models.vggt_dino import VGGT
from pytorch_msssim import ssim

def init_vggt():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    return model

class Trainer:
    def __init__(self, cfg_file):
        self.cfg = cfg_file

        self.model = HumanModel(self.cfg)
        self.train_set = HumanDataset(self.cfg.dataset, phase='train')
        num_workers = 0 if args.debug else self.cfg.batch_size * 2
        self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                       num_workers=num_workers, pin_memory=True)  #
        self.train_iterator = iter(self.train_loader)
        self.val_set = HumanDataset(self.cfg.dataset, phase='val')
        num_workers = 0 if args.debug else 2
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=num_workers,
                                     pin_memory=True)
        self.len_val = int(len(self.val_loader))  # real length of val set
        self.val_iterator = iter(self.val_loader)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.cfg.lr, self.cfg.stage_1_num_steps+self.cfg.depth_refine_num_steps+ 100,
                                                       pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

        self.logger = Logger(self.scheduler, cfg.record)
        self.total_steps = 0

        ## Model init
        self.model.cuda()

        ## VGGT init
        self.vggt = init_vggt()
        self.vggt.cuda()

        self.max_psnr = 0

        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

    def train(self):
        self.model.train()
        self.vggt.eval()
        progress_bar = tqdm(range(self.total_steps, self.cfg.stage_1_num_steps),
                            ncols=100,
                            colour='cyan')
        for iter in progress_bar:
            self.optimizer.zero_grad()
            step = iter 
            data = self.fetch_data(phase='train')
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    feat = self.vggt(data['ref']["image"].squeeze(0))
            
            img_pred, mask_pred, Lgeo  = self.model(data,feat)
                        
            img_gt = data['target']['image'].cuda()
            img_gt = img_gt[:,:,3:-3,3:-3]
            mask_gt = data['target']['mask'].cuda()

            Ll1 = l1_loss(img_pred, img_gt)
            Lssim = 1 - ssim(img_pred, img_gt, data_range=1.0, size_average=True)
            ## optional ##
            # lp = self.loss_fn_vgg(img_pred, img_gt).mean()
            
            loss = 0.8 * Ll1 + 0.2 * Lssim + Lgeo

            metrics = {
                'l1': Ll1.item(),
                'ssim': Lssim.item(),
                'Lgeo': Lgeo.item()
            }
            psnr_value = psnr(img_pred, img_gt).mean().double()
            self.logger.push(metrics)
            progress_bar.set_postfix(loss=loss.item(),
                                     L1 = Ll1.item(),
                                     PS = f"{psnr_value:.2f}",
                                     LC = Lgeo.item(),
                                    )

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                self.logger.writer.add_scalar(f'lr', self.optimizer.param_groups[0]['lr'], self.total_steps)
                if (psnr_value > self.max_psnr):
                    self.save_ckpt(save_path=Path('%s/%s.pth' % (cfg.record.ckpt_path,'Max_PSNR')), show_log=False)
                    self.max_psnr = psnr_value
                if  (iter>=180000):
                    self.save_ckpt(save_path=Path('%s/%.2f_%d.pth' % (cfg.record.ckpt_path, psnr_value , iter)), show_log=False)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()
            if self.total_steps and self.total_steps % self.cfg.record.eval_freq == 0:
                self.model.eval()
                self.run_eval()
                self.model.train()
            self.total_steps += 1

        print("FINISHED TRAINING")
        self.logger.close()
        self.save_ckpt(save_path=Path('%s/%s_final.pth' % (cfg.record.ckpt_path, cfg.name)))
        self.total_steps = 0


    def run_eval(self):
        logging.info(f"Doing validation ...")
        torch.cuda.empty_cache()
        psnr_list = []
        show_idx = np.random.choice(list(range(self.len_val)), 1)

        for idx in range(self.len_val):
            data = self.fetch_data(phase='val')
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    feat = self.vggt(data['ref']["image"].squeeze(0))
                    
                
                img_pred, mask_pred, _ = self.model(data, feat)
                img_gt = data['target']['image'][:,:,3:-3,3:-3].cuda()

                psnr_value = psnr(img_pred, img_gt).mean().double()
                psnr_list.append(psnr_value.item())
                tmp_novel = img_pred[0].detach()
                tmp_novel *= 255
                tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                
                tmp_img_name = '%s/%s_%s.jpg' % (
                    cfg.record.show_path, self.total_steps, idx)
                cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))

        val_psnr = np.round(np.mean(np.array(psnr_list)), 4)
        logging.info(f"Validation Metrics ({self.total_steps}): psnr {val_psnr}")
        self.logger.write_dict({'val_psnr': val_psnr}, write_step=self.total_steps)
        torch.cuda.empty_cache()

    def fetch_data(self, phase):
        if phase == 'train':
            try:
                data = next(self.train_iterator)
            except:
                self.train_iterator = iter(self.train_loader)
                data = next(self.train_iterator)
        elif phase == 'val':
            try:
                data = next(self.val_iterator)
            except:
                self.val_iterator = iter(self.val_loader)
                data = next(self.val_iterator)
        for view in ['ref']:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda()
        return data

    def load_ckpt(self, load_path, load_optimizer=True, strict=True):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=strict)
        logging.info(f"Parameter loading done")
        if load_optimizer:
            self.total_steps = ckpt['total_steps'] + 1
            self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            logging.info(f"Optimizer loading done")

    def save_ckpt(self, save_path, show_log=True):
        if show_log:
            logging.info(f"Save checkpoint to {save_path} ...")
        torch.save({
            'total_steps': self.total_steps,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    cfg = config()
    cfg.load("config/config_thu.yaml")
    cfg = cfg.get_cfg()
    cfg.defrost()

    dt = datetime.today()
    cfg.exp_name = '%s_%s%s' % (cfg.name, str(dt.month).zfill(2), str(dt.day).zfill(2))
    cfg.record.ckpt_path = "experiments/%s/ckpt" % cfg.exp_name
    cfg.record.show_path = "experiments/%s/show" % cfg.exp_name
    cfg.record.logs_path = "experiments/%s/logs" % cfg.exp_name
    cfg.record.file_path = "experiments/%s/file" % cfg.exp_name
    cfg.record.vis_path = "experiments/%s/vis_train" % cfg.exp_name
    cfg.freeze()

    for path in [cfg.record.ckpt_path, cfg.record.show_path, cfg.record.logs_path, cfg.record.file_path, cfg.record.vis_path]:
        Path(path).mkdir(exist_ok=True, parents=True)

    file_backup(cfg.record.file_path, cfg, train_script=os.path.basename(__file__))

    torch.manual_seed(3407)
    np.random.seed(3407)

    trainer = Trainer(cfg)
    trainer.train()

