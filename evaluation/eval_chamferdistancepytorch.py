import torch
import numpy as np
from plyfile import PlyData
import sys
import os

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from submodules.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from submodules.ChamferDistancePytorch import fscore

def load_ply(file_path):
    """Loads a PLY file and returns the points as a numpy array."""
    plydata = PlyData.read(file_path)
    points = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
    return points

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
    chamfer_dist = dist_chamfer_3D.chamfer_3DDist()

    # Calculate the Chamfer distance
    dist1, dist2, _, _ = chamfer_dist(pcd1, pcd2)
    loss = torch.mean(dist1) + torch.mean(dist2)

    print(f"Chamfer distance between {os.path.basename(file1)} and {os.path.basename(file2)}: {loss.item()}")

    f1_score, precision, recall = fscore.fscore(dist1, dist2)

    print(f"F1_score = {f1_score}")

if __name__ == '__main__':
    main()