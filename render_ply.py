import trimesh
import numpy as np
import os

def draw_ply(ply_path):
    mesh = trimesh.load_mesh(ply_path)

    # 定義相機參數（視角）
    resolution = (800, 600)  # 渲染解析度
    fov_degrees = (60, 60)
    camera = trimesh.scene.Camera(resolution=resolution, fov=fov_degrees)
    camera_pose = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
    camera_pose[:3, 3] = [0, 0, 3]  # 放置相機位置

    # 設置相機視角
    scene = trimesh.Scene(mesh)
    scene.set_camera(camera)

    scene.save_image(resolution=resolution)

if __name__ == "__main__":
    draw_ply(os.path.join("./experiments/new_exp/meshes/00000000_vertex_color.ply"))