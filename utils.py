import random
import os
from pathlib import Path

import numpy as np
import torch
import cv2 as cv



# embed view direction into text
def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis

    # res[(phis < front)] = 0
    res[(phis >= (2 * np.pi - front / 2)) & (phis < front / 2)] = 0

    # res[(phis >= front) & (phis < np.pi)] = 1
    res[(phis >= front / 2) & (phis < (np.pi - front / 2))] = 1

    # res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi - front / 2)) & (phis < (np.pi + front / 2))] = 2

    # res[(phis >= (np.pi + front))] = 3
    res[(phis >= (np.pi + front / 2)) & (phis < (2 * np.pi - front / 2))] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def tensor2numpy(tensor:torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().numpy()    
    tensor = (tensor * 255).astype(np.uint8)
    return tensor

def numpy2image(array:np.ndarray) -> np.ndarray:
    array = (array * 255).astype(np.uint8)
    return array

def make_path(path: Path) -> Path:
    path.mkdir(exist_ok=True,parents=True)
    return path

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()  # not R but R^-1
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def near_far_from_sphere(rays_o, rays_d):
    a = torch.sum(rays_d**2, dim=-1, keepdim=True)
    b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    mid = 0.5 * (-b) / a
    near = mid - 1.0
    far = mid + 1.0
    return near, far

def read_intrinsic_inv(conf):
    # copy from dataset.py
    data_dir = conf.get_string('data_dir')
    render_cameras_name = conf.get_string('render_cameras_name')
    camera_dict = np.load(os.path.join(data_dir, render_cameras_name))

    # take camera of first image in dataset
     # world_mat is a projection matrix from world to image
    world_mat = camera_dict['world_mat_0'].astype(np.float32)

    # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
    scale_mat = camera_dict['scale_mat_0'].astype(np.float32)
    P = world_mat @ scale_mat
    P = P[:3, :4]
    intrinsics, pose = load_K_Rt_from_P(None, P)
    return torch.inverse(torch.from_numpy(intrinsics).float())

'''
Project
'''
def gen_random_ray_at_pose(theta, phi, radius, H, W, intrincis_inv, resolution_level=1, half=True):
    l = resolution_level
    tx = torch.linspace(0, W - 1, W // l)
    ty = torch.linspace(0, H - 1, H // l)
    pixels_x, pixels_y = torch.meshgrid(tx, ty)
    
    # device = intrincis_inv.device # device in LatentPaintTrainer
    # pixels_x = pixels_x.to(device)
    # pixels_y = pixels_y.to(device)
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3        
    p = torch.matmul(intrincis_inv[:3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
    '''
    trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
    pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
    pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
    pose_0 = np.linalg.inv(pose_0)
    pose_1 = np.linalg.inv(pose_1)
    rot_0 = pose_0[:3, :3]
    rot_1 = pose_1[:3, :3]
    rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
    key_times = [0, 1]
    slerp = Slerp(key_times, rots)
    rot = slerp(ratio)
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose = pose.astype(np.float32)
    pose[:3, :3] = rot.as_matrix()
    pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
    pose = np.linalg.inv(pose)
    rot = torch.from_numpy(pose[:3, :3]).cuda()
    trans = torch.from_numpy(pose[:3, 3]).cuda()
    
    # Randomly generate pose(translate + rotation)
    trans = torch.rand(3, 1)
    rot_vec = torch.randn(3)
    rot_vec = rot_vec / torch.norm(rot_vec) 
    rot = Rot.from_rotvec(rot_vec).as_matrix()
    '''

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    if half:
        trans = torch.tensor([x, y, z], dtype=torch.float16)
    else:
        trans = torch.tensor([x, y, z], dtype=torch.float32)

    # Calculate camera's forward, right, and up vectors
    # forward: (-x, -y, -z)
    if half:
        forward = -torch.tensor([x, y, z], dtype=torch.float16)
    else:
        forward = -torch.tensro([x, y, z], dtype=torch.float32)
    forward /= torch.norm(forward)
    
    # up: (0, 0, radius) - (x, y, z)
    if half:
        up = torch.cross(torch.tensor([-x, -y, radius-z], dtype=torch.float16), forward)
    else:
        up = torch.cross(torch.tensor([-x, -y, radius-z], dtype=torch.float32), forward)
    up /= torch.norm(up)
    
    right = torch.cross(forward, up)
    right /= torch.norm(right)
    
    # Construct rotation matrix
    rot = torch.stack((right, up, forward), dim=1)

    # trans = trans.to(device)
    # rot = rot.to(device)
    rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
    rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
    return rays_o.transpose(0, 1), rays_v.transpose(0, 1)