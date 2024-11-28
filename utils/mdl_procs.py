import open3d as o3d
import numpy as np

def down_sample_cloud(cloud:np.ndarray, mode:str="uniform", voxel_size:float=0.5, value:int=10) -> np.ndarray:
    """
    down sample point cloud data
    :param cloud: shape=[n,2]
    :return:
    """
    assert len(cloud.shape)==2, f"the expected input cloud should be 2d, but {len(cloud.shape)} was gotten."

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(np.hstack([cloud, np.zeros((cloud.shape[0], 1))]))

    if mode=="voxel":
        downpcd_o3d = pcd_o3d.voxel_down_sample(voxel_size=voxel_size)
    elif mode=="uniform":
        downpcd_o3d = pcd_o3d.uniform_down_sample(every_k_points=value)
    else:
        raise ValueError(f"the expected mode is one of ['voxel', 'uniform'], but {mode} was gotten.")

    down_cloud = np.asarray(downpcd_o3d.points)[:, :2]  # Only keep x and y coordinates

    return down_cloud

def pre_downsampling(cloud:np.ndarray,
                     target_num:int=5000,
                     start_voxel_size:float=0.5,
                     isDebug:bool=False) -> (np.ndarray, float):
    """
    pre downsample point cloud data to target point number
    :param cloud:            shape=[n,2], building's point cloud data
    :param target_num:       the target number of pre-downsampled point cloud data
    :param start_voxel_size: the voxel_size used for downsampling in the first iteration.
    :param isDebug:          whether open debug mode and print related info. to the console/terminal.
    :return:
          cloud_ds:             shape=[n,2], the downsampled point cloud data
          used_voxel_size-0.1:  the ultimately used voxel_size for downsampling
    """
    used_voxel_size = start_voxel_size
    cloud_ds = cloud.copy()
    while cloud_ds.shape[0]>target_num:
        cloud_ds = down_sample_cloud(cloud, mode="voxel", voxel_size=used_voxel_size)
        used_voxel_size += 0.1
        if isDebug:
            print(f"[mdl_procs/pre_downsampling()] :: cloud_ds_shape={cloud_ds.shape[0]}, "
                  f"used_voxel_size={used_voxel_size-0.1}")

    return cloud_ds, used_voxel_size-0.1