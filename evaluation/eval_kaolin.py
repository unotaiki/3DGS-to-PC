from sklearn.metrics import f1_score
import torch
import numpy as np
from plyfile import PlyData
import sys
import os
import kaolin

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def load_ply(file_path):
    """Loads a PLY file and returns the points as a numpy array."""
    plydata = PlyData.read(file_path)
    points = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
    return points


def eval_chamfer_distance(pcd, pcd_name, gt_file='../eval_ply/gt/points3d.ply'):

    gt = load_ply(gt_file)

    pcd = pcd.poitns

    # Initialize the Chamfer distance function
    chamfer_distance = kaolin.metrics.pointcloud.chamfer_distance(gt, pcd, w1=1.0, w2=1.0, squared=True) # if squared=False, use euclidean distance
    print(f"Chamfer distance of {pcd_name}: {chamfer_distance}")

    RADIOUS = 0.10
    f1_score = kaolin.metrics.pointcloud.f_score(pcd, gt, radius=RADIOUS, eps=1e-08)
    print(f"F1_score = {f1_score}")




def main():
    """Main function to evaluate the Chamfer distance between two point clouds."""
    # File paths for the two point clouds
    file1 = 'data/points3d.ply'
    file2 = 'data/gt-raster-pc_clean.ply'

    # Check if the files exist
    if not os.path.exists(file1):
        print(f"Error: File not found at {file1}")
        return
    if not os.path.exists(file2):
        print(f"Error: File not found at {file2}")
        return

    # Load the point clouds
    points1 = load_ply(file1)
    points2 = load_ply(file2)

    # Convert to PyTorch tensors
    pcd1 = torch.from_numpy(points1).float().cuda()[None, ...]
    pcd2 = torch.from_numpy(points2).float().cuda()[None, ...]

    # Initialize the Chamfer distance function
    chamfer_distance = kaolin.metrics.pointcloud.chamfer_distance(pcd1, pcd2, w1=1.0, w2=1.0, squared=True) # if squared=False, use euclidean distance
    print(f"Chamfer distance between {os.path.basename(file1)} and {os.path.basename(file2)}: {chamfer_distance}")

    RADIOUS = 0.15
    f1_score = kaolin.metrics.pointcloud.f_score(pcd1, pcd2, radius=RADIOUS, eps=1e-08)
    print(f"F1_score = {f1_score}")

if __name__ == '__main__':
    main()