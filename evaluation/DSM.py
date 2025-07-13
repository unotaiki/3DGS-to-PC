import open3d as o3d
import numpy as np

def create_dsm(pcd: o3d.geometry.PointCloud, 
               resolution: float, 
               x_range: list = [-10.0, 10.0], 
               y_range: list = [-10.0, 10.0]) -> np.ndarray:
    """
    点群から指定された解像度のDSM（Digital Surface Model）を生成します。

    Args:
        pcd (o3d.geometry.PointCloud): 入力となる点群データ。
        resolution (float): DSMの1ピクセルの辺の長さ（解像度）。
        x_range (list, optional): DSMを生成するX軸の範囲。Defaults to [-10.0, 10.0].
        y_range (list, optional): DSMを生成するY軸の範囲。Defaults to [-10.0, 10.0].

    Returns:
        np.ndarray: 生成されたDSM。点が存在しないピクセルはNaNになります。
    """
    # 点群データをNumpy配列に変換
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        return np.array([])

    x_min, x_max = x_range
    y_min, y_max = y_range

    # DSMのグリッドサイズを計算
    width = int(np.ceil((x_max - x_min) / resolution))
    height = int(np.ceil((y_max - y_min) / resolution))

    # DSMを保持する配列を最小値で初期化
    dsm = np.full((height, width), -np.inf, dtype=np.float64)

    # 各点の座標に対応するグリッドインデックスを計算
    # yが画像の行、xが画像の列に対応するため、インデックスの順序に注意
    ix = ((points[:, 0] - x_min) / resolution).astype(int)
    iy = ((points[:, 1] - y_min) / resolution).astype(int)

    # 有効なインデックス範囲内の点のみを対象にする
    valid_indices = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    ix = ix[valid_indices]
    iy = iy[valid_indices]
    z_values = points[:, 2][valid_indices]

    # 各グリッドセルにZの最大値を効率的に格納
    # np.maximum.atは、(iy, ix)の各ペアに対して、dsmの既存値とz_valuesの値を比較し、大きい方で更新します。
    # これにより、各セルのZの最大値が記録されます。
    np.maximum.at(dsm, (iy, ix), z_values)

    # Z値が更新されなかったピクセル（点が存在しなかった場所）をNaNに置換
    dsm[dsm == -np.inf] = np.nan
    
    # Y軸の方向を一般的な画像座標系に合わせるために上下反転させる
    return np.flipud(dsm)


# --- メイン処理 ---

# 1. サンプル点群データの作成
# 実際には `pcd = o3d.io.read_point_cloud("your_file.pcd")` のようにファイルを読み込みます。
print("1. サンプル点群を生成します...")
points = np.random.rand(50000, 3)
points[:, 0] = points[:, 0] * 30 - 15  # X: -15 ~ 15
points[:, 1] = points[:, 1] * 30 - 15  # Y: -15 ~ 15
points[:, 2] = np.sin(points[:, 0]*0.5) + np.cos(points[:, 1]*0.5) # Z: 波状の表面
# 外れ値も追加
outliers = np.random.rand(100, 3) * 50 - 25
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.vstack((points, outliers)))
print(f"   生成された点群の数: {len(pcd.points)}")
# o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

# 2. XY平面でクロッピング
print("\n2. 点群をXY平面[-10, 10] x [-10, 10]の範囲でクロップします...")
bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=(-10.0, -10.0, -np.inf),
    max_bound=(10.0, 10.0, np.inf)
)
pcd_cropped = pcd.crop(bbox)
print(f"   クロップ後の点群の数: {len(pcd_cropped.points)}")
# o3d.visualization.draw_geometries([pcd_cropped], window_name="Cropped Point Cloud")


# 3. 外れ値のフィルタリング
print("\n3. 外れ値を除去します...")
# statistical_outlier_removal: 近傍点との平均距離を計算し、標準偏差に基づいて外れ値を除去
# nb_neighbors: 考慮する近傍点の数
# std_ratio: 標準偏差の乗数。この値が小さいほど、より多くの点が外れ値と見なされる
pcd_filtered, ind = pcd_cropped.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"   フィルタリング後の点群の数: {len(pcd_filtered.points)}")
# o3d.visualization.draw_geometries([pcd_filtered], window_name="Filtered Point Cloud")


# 4. DSMの生成
print("\n4. Z軸上方向からラスタライズしてDSMを生成します...")
# 解像度（1ピクセルの大きさ）を設定
dsm_resolution = 0.2  # 例えば0.2m x 0.2mの解像度
print(f"   設定解像度: {dsm_resolution} m/pixel")

# DSMを生成
dsm_array = create_dsm(pcd_filtered, dsm_resolution)

print(f"\n✅ DSMが生成されました。")
print(f"   DSMの形状 (height, width): {dsm_array.shape}")
print("   DSMデータ (一部表示):")
# 表示範囲を限定
display_slice = dsm_array[:5, :5]
print(display_slice)

# (オプション) MatplotlibでDSMを可視化
try:
    import matplotlib.pyplot as plt
    print("\n(オプション) MatplotlibでDSMを可視化します...")
    plt.figure(figsize=(8, 8))
    plt.imshow(dsm_array, cmap='viridis', origin='upper', extent=[-10, 10, -10, 10])
    plt.colorbar(label='Elevation (Z)')
    plt.title(f'Digital Surface Model (DSM) - Resolution: {dsm_resolution}m')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()
except ImportError:
    print("\nMatplotlibがインストールされていません。`pip install matplotlib`でインストールするとDSMを可視化できます。")