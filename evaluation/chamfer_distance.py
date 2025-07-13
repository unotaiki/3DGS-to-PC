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

def chamfer_distance(pcd1, pcd2, squared=True):
    """
    Computes the Chamfer Distance between two point clouds.

    The distance is computed as the sum of two directional distances:
    - The average distance from each point in p1 to its nearest neighbor in p2.
    - The average distance from each point in p2 to its nearest neighbor in p1.

    Args:
        p1 (torch.Tensor): The first point cloud, shape [N, D].
                           N the number of points, D the dimension.
        p2 (torch.Tensor): The second point cloud, shape [M, D].
                           M the number of points, D the dimension.

    Returns:
        torch.Tensor: The Chamfer Distance for each item in the batch, shape [B].
    """    

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


