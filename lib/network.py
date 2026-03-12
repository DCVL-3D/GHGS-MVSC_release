import torch
from torch import nn
import torch.nn.functional as F
from gaussian_renderer import renderThuman
import torch, numpy as np, trimesh

from vggt.heads.dpt_head import DPTHead
from vggt.heads.dpt_head_clus import DPTHead_clus

from pytorch3d.loss import chamfer_distance

from lib.modules.SC_3D_Atten import SC_3D_Atten

def depth2pts(depth, extrinsic, intrinsic):
    # depth H W extrinsic 3x4 intrinsic 3x3 pts map H W 3
    depth = depth.squeeze(0)
    extrinsic = extrinsic.squeeze(0)
    intrinsic = intrinsic.squeeze(0)
    rot = extrinsic[:3, :3]
    trans = extrinsic[:3, 3:]
    S, S = depth.shape

    y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device),
                          torch.linspace(0.5, S-0.5, S, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # H W 3

    pts_2d[..., 2] = 1.0 / (depth + 1e-8)
    # pts_2d[..., 2] = depth
    pts_2d[..., 0] -= intrinsic[0, 2]
    pts_2d[..., 1] -= intrinsic[1, 2]
    pts_2d_xy = pts_2d[..., :2] * pts_2d[..., 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[0, 0]
    pts_2d[..., 1] /= intrinsic[1, 1]
    pts_2d = pts_2d.reshape(-1, 3).T
    pts = rot.T @ pts_2d - rot.T @ trans
    return pts.T.view(S, S, 3)

def depth2pts_token(depth, extrinsic, intrinsic):
    # depth H W extrinsic 3x4 intrinsic 3x3 pts map H W 3
    depth = depth.squeeze(0)
    extrinsic = extrinsic.squeeze(0)
    intrinsic = intrinsic.squeeze(0)
    rot = extrinsic[:3, :3]
    trans = extrinsic[:3, 3:]
    S, S = depth.shape
    scale = S/512
    y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device),
                            torch.linspace(0.5, S-0.5, S, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # H W 3

    pts_2d[..., 2] = 1.0 / (depth + 1e-8)
    pts_2d[..., 0] -= intrinsic[0, 2] * scale
    pts_2d[..., 1] -= intrinsic[1, 2] * scale
    pts_2d_xy = pts_2d[..., :2] * pts_2d[..., 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[0, 0]* scale
    pts_2d[..., 1] /= intrinsic[1, 1]* scale
    pts_2d = pts_2d.reshape(-1, 3).T
    pts = rot.T @ pts_2d - rot.T @ trans
    return pts.T.view(S, S, 3)

def to_cuda(data):
    for k in data:
        if k in ['scale', 'transl', 'y_transl', 'human_height', 'v_scale','Rh','Th']:
            if isinstance(data[k],torch.Tensor):
                data[k] = data[k].cuda()
            else:
                data[k] = torch.tensor(data[k]).cuda()
    return data

class HumanModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.ref_view_num = cfg.dataset.valid_ref_view_num

        self.pos_head = DPTHead(dim_in=2 * 1024, output_dim=2, activation="linear", conf_activation="expp1")
        self.blk = SC_3D_Atten(dim = 1024 * 2, k = 64)
        self.indim = 32
        self.decoder = DPTHead_clus(dim_in=2 * 1024, output_dim=self.indim, activation="linear", conf_activation="expp1")
        
        self.offset_header = nn.Sequential(
            nn.Linear(self.indim, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 3),
        )
        self.scale_header = nn.Sequential(
            nn.Linear(self.indim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softplus(100),
        )
        self.rot_header = nn.Sequential(
            nn.Linear(self.indim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

        self.opacity_header = nn.Sequential(
            nn.Linear(self.indim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def predict_gaussian(self, data,feat):
        data['target'] = to_cuda(data['target'])
        data['ref'] = to_cuda(data['ref'])
        bs = data['ref']['image'].shape[0]
        
        n_channels, H, W = data['ref']['image'].shape[2:]
        mask = data['ref']['mask'].to(torch.bool)
        extr = data['ref']['extr']
        intr = data['ref']['intr']
        
        view_num = 4

        if isinstance(data['target']['smpl'],torch.Tensor):
            smpl = data['target']['smpl'].cuda()
        else:
            smpl = torch.from_numpy(data['target']['smpl'])[None].cuda()

        pos_out, _ = self.pos_head(feat[0], images=data['ref']["image"], patch_start_idx=feat[1])

        token_depth = F.interpolate(pos_out.squeeze(-1), size = (37, 37), mode = "bilinear")
        token_pos = []
        for v in range(view_num ):
            tmp = depth2pts_token(token_depth[:,v], extr[:,v], intr[:,v])
            token_pos.append(tmp.unsqueeze(0))
        token_pos = torch.cat(token_pos, dim = 0)

        ## depth 2 point
        pos_out = pos_out[:,:,3:-3,3:-3,:].squeeze(-1)
        pts = []
        for v in range(view_num ):
            tmp = depth2pts(pos_out[:,v], extr[:,v], intr[:,v]) 
            pts.append(tmp.unsqueeze(0))
        pts = torch.cat(pts, dim = 0) ##

        V, N, C, D = view_num ,1369, 2048, 1024 
        ## 이제 여기 token fusing ##
        feat_layer = []
        for i in [4, 11, 17, 23]:
            tmp= self.blk(feats = feat[0][i][:,:,5:,:].reshape(1, V*N, C),
                            pos3d = token_pos.reshape(1, V*N, 3),
                            feat_sem  = feat[2].reshape(1,V*N, D),
                            token_mask = None)
            feat_layer.append(tmp.reshape(1, V, N, 2048))

        feature_out = self.decoder(feat_layer, images=data['ref']["image"], patch_start_idx=0)

        image = data['ref']['image'].reshape(-1, n_channels, H, W)
        image = image[:,:,3:-3, 3:-3]
        feature_out = feature_out[:,:,3:-3,3:-3].permute(0,2,3,1) 
        rgb = data['ref']['image'].permute(0,1,3,4,2)
        
        mask_i = mask[0].permute(0, 2, 3, 1).squeeze(-1)
        color_out = rgb[:,:,3:-3,3:-3,:].squeeze(0)

        pos = pts[mask_i].unsqueeze(0) 
        init_feature = feature_out[mask_i].unsqueeze(0) 
    
        rot = self.rot_header(init_feature) 
        rot = F.normalize(rot, p=2, dim=-1) 
        scale= self.scale_header(init_feature)
        opacity= self.opacity_header(init_feature)
        offset = self.offset_header(init_feature)
        
        rgb = color_out[mask_i].unsqueeze(0)

        loss_cham, _ = chamfer_distance(smpl, pos)
        pos = offset + pos

        return pos, rot, scale, opacity, rgb, loss_cham, pos_out

    def forward(self, data,feat):
        
        pcd, rot, scale, opacity, rgb, loss_cham, pos_out = self.predict_gaussian(data, feat)
        render_ = renderThuman(data, 0 ,pcd[0], rgb[0],rot[0], scale[0], opacity[0],  bg_color=self.cfg.dataset.bg_color)    
        
        img_pred = render_[0].unsqueeze(0)
        depth_pred = render_[1].unsqueeze(0)
        mask_pred = render_[2].unsqueeze(0)

        return img_pred, mask_pred, loss_cham