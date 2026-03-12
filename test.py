import torch
import lib.utils as utils
import cv2
import os
import numpy as np
import torch
import torch, numpy as np, trimesh
def visualize_pcd(i,coarse_pcd_i: torch.Tensor,
                      save_path: str = "coarse_pcd.glb",
                      rgba=(0, 0, 0, 255)):
    """
    coarse_pcd_i : (B, N, 3) float32 PyTorch tensor
    save_path    : .glb 파일 이름
    rgba         : (R,G,B,A) 0–255 단일 색
    """
    # 1) 텐서 → (N,3) numpy
    points = coarse_pcd_i.squeeze(0).detach().cpu().numpy()

    # 2) 색 배열 만들기
    color_u8 = np.array(rgba, dtype=np.uint8)
    colors   = np.tile(color_u8, (points.shape[0], 1))   # (N,4)

    # 3) trimesh PointCloud → GLB 내보내기
    pc = trimesh.points.PointCloud(vertices=points, colors=colors)
    save_path = f"test/{i}_" + save_path
    pc.export(save_path)        # 확장자가 .glb 이면 glTF-binary 로 저장

    print(f"Saved GLB: {save_path}  (PowerPoint ▸ 삽입 ▸ 3-D 모델)")
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

    # pts_2d[..., 2] = 1.0 / (depth + 1e-8)
    ## DC 이렇게 써보자 ,,,SMPL 을 depth 로 저장해서 이렇게 해야될듯 ##
    pts_2d[..., 2] = depth
    pts_2d[..., 0] -= intrinsic[0, 2]
    pts_2d[..., 1] -= intrinsic[1, 2]
    pts_2d_xy = pts_2d[..., :2] * pts_2d[..., 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[0, 0]
    pts_2d[..., 1] /= intrinsic[1, 1]
    pts_2d = pts_2d.reshape(-1, 3).T
    pts = rot.T @ pts_2d - rot.T @ trans
    return pts.T.view(S, S, 3)

data_root = "/data/KJG/dataset/ZJU-Mocap/train"
human_name = "CoreView_313"
cam_name = "Camera_B1"
pose_name = f"{1:06d}"


depth_path = os.path.join("/data2/KJG/dataset/ZJU-DC", "%s/%s/%s.png") # human_idx:str // view:str // pose_num:int 
cams_path = os.path.join(data_root,'%s/cameras/%s.npy')
mask_path = os.path.join(data_root, '%s/mask/%s/%s.png')
extr_list = []
intr_list = []
depth_list = []
mask_list = []

for cam_name in ["Camera_B1", "Camera_B6", "Camera_B12", "Camera_B18"]:

    cam_param_name = cams_path % (human_name, cam_name)
    depth_name = depth_path % (human_name, cam_name, pose_name)
    mask_name = mask_path % (human_name, cam_name, pose_name)    
    ## Cam load
    cam = np.load(cam_param_name, allow_pickle=True).item()  # {"K","R","T"}
    K = cam["K"].astype(np.float32)
    R = cam["R"].astype(np.float32)
    T = cam["T"].astype(np.float32)  # 위에서 TO_METERS=True였으면 이미 m 단위
    extr = np.concatenate([R, T.reshape(3,1)], axis=-1).astype(np.float32)
    intr = K

    ## Depth load 
    if os.path.exists(depth_name):
        depth = utils.read_depth(depth_name)
        depth = cv2.resize(depth, (512, 512), interpolation = cv2.INTER_NEAREST)
        mask = utils.read_img(mask_name)
        mask = cv2.resize(mask, (512,512), interpolation=cv2.INTER_NEAREST)

    extr_list.append(torch.from_numpy(extr).unsqueeze(0))
    intr_list.append(torch.from_numpy(intr).unsqueeze(0))
    depth_list.append(torch.from_numpy(depth).unsqueeze(0))
    mask_list.append(torch.from_numpy(mask).unsqueeze(0))

extr_torch = torch.cat(extr_list, dim = 0)
intr_torch = torch.cat(intr_list, dim = 0)
depth_torch = torch.cat(depth_list, dim = 0)
mask_torch = torch.cat(mask_list, dim = 0)

pos_out = depth_torch.unsqueeze(0)
extr= extr_torch.unsqueeze(0)
intr = intr_torch.unsqueeze(0)

pts = []
for v in range(4):
    tmp = depth2pts(pos_out[:,v], extr[:,v], intr[:,v]) #[512, 512, 3], 
    pts.append(tmp.unsqueeze(0))
pts = torch.cat(pts, dim = 0) ##
pos =pts[mask_torch.bool()].unsqueeze(0) # B, NP, 3

visualize_pcd(2,pos)


