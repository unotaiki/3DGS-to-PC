import torch
import numpy as np
from plyfile import PlyData
import sys
import os
import kaolin
import json
import argparse

def load_ply(file_path):
    """Loads a PLY file and returns the points as a numpy array."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}", file=sys.stderr)
        return None
    try:
        plydata = PlyData.read(file_path)
        points = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        return points
    except Exception as e:
        print(f"Error loading PLY file {file_path}: {e}", file=sys.stderr)
        return None

def calculate_f_score_metrics(pred_pcd, gt_pcd, radius, eps=1e-8):
    """
    Calculates F1-score, Precision, and Recall based on MVS benchmark definitions.

    - Precision = |P_rec| / |P| (Fraction of predicted points close to GT)
    - Recall    = |G_hit| / |G| (Fraction of GT points close to prediction)

    Args:
        pred_pcd (torch.Tensor): Predicted point cloud, shape (B, N, 3).
        gt_pcd (torch.Tensor): Ground truth point cloud, shape (B, M, 3).
        radius (float): The distance threshold (d_th).
        eps (float): A small value to prevent division by zero.

    Returns:
        tuple: (f1, precision, recall) as Python floats.
    """
    # sided_distance は二乗距離を返すため、半径も二乗して比較
    radius_squared = radius ** 2

    # 1. Precisionの計算
    # P_rec: 予測点群(P)のうち、正解(GT)から半径以内にある点の数
    dist_p_to_g, _ = kaolin.metrics.pointcloud.sided_distance(pred_pcd, gt_pcd)
    p_rec = torch.sum(dist_p_to_g < radius_squared).item()
    # |P|: 全ての予測点の数
    num_predicted_points = pred_pcd.shape[1]
    
    precision = float(p_rec) / float(num_predicted_points + eps)

    # 2. Recallの計算
    # G_hit: 正解点群(G)のうち、予測(P)から半径以内にある点の数
    dist_g_to_p, _ = kaolin.metrics.pointcloud.sided_distance(gt_pcd, pred_pcd)
    g_hit = torch.sum(dist_g_to_p < radius_squared).item()
    # |G|: 全ての正解点の数
    num_gt_points = gt_pcd.shape[1]

    recall = float(g_hit) / float(num_gt_points + eps)

    # 3. F1-Scoreの計算
    f1 = 2.0 * (precision * recall) / (precision + recall + eps)
    
    return f1, precision, recall

def main(args):
    """
    Main function to evaluate Chamfer distance and F-scores between a ground truth
    point cloud and multiple predicted point clouds.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    gt_points = load_ply(args.gt_file)
    if gt_points is None:
        print(f"Error: Ground truth file not found or could not be loaded from {args.gt_file}. Aborting.", file=sys.stderr)
        return

    gt_pcd_tensor = torch.from_numpy(gt_points).float().to(device)[None, ...]

    all_results = {}
    print(f"Starting evaluation for indices {args.start_idx} to {args.end_idx}...")

    for i in range(args.start_idx, args.end_idx + 1):
        pred_file_path = os.path.join(args.base_dir, str(i), args.pred_filename)
        print(f"\n--- Processing: {pred_file_path} ---")

        pred_points = load_ply(pred_file_path)
        if pred_points is None:
            continue

        pred_pcd_tensor = torch.from_numpy(pred_points).float().to(device)[None, ...]

        current_results = {}

        # 1. Chamfer Distance の計算
        try:
            # kaolin の chamfer_distance は (pred->gt, gt->pred) の平均を取る
            chamfer_dist = kaolin.metrics.pointcloud.chamfer_distance(pred_pcd_tensor, gt_pcd_tensor)
            chamfer_dist_val = chamfer_dist.item()
            current_results['chamfer_distance'] = chamfer_dist_val
            print(f"  Chamfer Distance: {chamfer_dist_val:.6f}")
        except Exception as e:
            print(f"  Error calculating Chamfer Distance: {e}", file=sys.stderr)
            current_results['chamfer_distance'] = 'error'

        # 2. F-score, Precision, Recall の計算 (複数の半径で)
        f1_scores_results = {}
        for radius in args.radii:
            try:
                # ★★★ 新しく実装した関数を呼び出す ★★★
                f1_val, precision_val, recall_val = calculate_f_score_metrics(
                    pred_pcd_tensor, gt_pcd_tensor, radius=radius
                )
                
                f1_scores_results[f'radius_{radius}'] = {
                    'f1_score': f1_val,
                    'precision': precision_val,
                    'recall': recall_val
                }
                print(f"  Metrics for Radius {radius}:")
                print(f"    F1-Score:  {f1_val:.6f}")
                print(f"    Precision: {precision_val:.6f}")
                print(f"    Recall:    {recall_val:.6f}")

            except Exception as e:
                print(f"  Error calculating F-score for radius {radius}: {e}", file=sys.stderr)
                f1_scores_results[f'radius_{radius}'] = 'error'

        current_results['f1_scores'] = f1_scores_results
        all_results[str(i)] = current_results

    try:
        with open(args.output_json, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\nEvaluation complete. All results saved to {args.output_json}")
    except Exception as e:
        print(f"\nError writing results to JSON file: {e}", file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate Chamfer Distance and F-scores for multiple point clouds."
    )
    parser.add_argument(
        '--base_dir', type=str, default='eval_ply',
        help='Base directory containing the numbered evaluation folders.'
    )
    parser.add_argument(
        '--gt_file', type=str, default='eval_ply/gt/points3d.ply',
        help='Path to the ground truth PLY file.'
    )
    parser.add_argument(
        '--pred_filename', type=str, default='gs_pc.ply',
        help='Filename of the predicted point cloud within each numbered folder.'
    )
    parser.add_argument(
        '--start_idx', type=int, default=1,
        help='Starting index for the evaluation folders.'
    )
    parser.add_argument(
        '--end_idx', type=int, default=11,
        help='Ending index for the evaluation folders.'
    )
    parser.add_argument(
        '--radii', type=float, nargs='+', default=[0.05, 0.10, 0.30],
        help='A list of radii to use for F-score calculation.'
    )
    parser.add_argument(
        '--output_json', type=str, default='evaluation_results.json',
        help='File path to save the JSON results.'
    )

    args = parser.parse_args()
    main(args)