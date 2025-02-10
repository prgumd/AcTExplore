import os

import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter
import numba as nb
import matplotlib.animation as animation
import xml.etree.ElementTree as ET
import cv2
from scipy.interpolate import interp2d

class Recons():
    def __init__(self, gt_cfg, obj_idx, threshold, resolution):
        ##### Data list #####
        self.x_offset = 0.02
        self.pcd = np.array([0])
        self.o3pcd = o3d.geometry.PointCloud()
        self.trajectory = []
        self.orientation = []
        self.diff = 0

        ##### Metric Variables #####
        self.cfg = gt_cfg
        urdf_path = self.cfg.object.urdf_path[obj_idx]
        urdf_xml_tree = ET.parse(urdf_path)
        obj_path = os.path.join(os.path.dirname(urdf_path),
                                urdf_xml_tree.find("link/visual/geometry/mesh").get("filename"))
        self.obj_gt = o3d.io.read_triangle_mesh(obj_path)
        scale_factors = np.array(urdf_xml_tree.find("link/visual/geometry/mesh").get("scale").split(' ')).astype(
            np.float)
        self.preprocessing(self.cfg, scale_factors)

        self.gt_tree = o3d.geometry.KDTreeFlann(self.pcd_gt)
        self.mat_obj = o3d.visualization.rendering.MaterialRecord()
        self.mat_obj.shader = 'defaultLitSSR'
        self.mat_obj.base_color = [1, 1, 1, 0.2]
        self.mat_obj.base_roughness = 0.0
        self.mat_obj.base_reflectance = 0.0
        self.mat_obj.base_clearcoat = 0.0
        self.mat_obj.thickness = 0.5
        self.mat_obj.transmission = 0.2
        self.mat_obj.absorption_distance = 10
        self.mat_obj.absorption_color = [1, 1, 1]

        self.D_THRESHOLD = 0.005
        self.coverage = 0.0
        self.chamfer = 0.0
        self.last_index_acc = 0
        self.pcd_gt_grid = np.zeros(len(self.pcd_gt.points), dtype=int)

        ##### Voxel Variables #####
        self.threshold = threshold
        self.th_min = np.array([threshold["x"].min, threshold["y"].min, threshold["z"].min])
        self.resolution = resolution
        self.grid_size = [np.ceil((threshold["x"].max - threshold["x"].min) / resolution).astype(np.int),
                          np.ceil((threshold["y"].max - threshold["y"].min) / resolution).astype(np.int),
                          np.ceil((threshold["z"].max - threshold["z"].min) / resolution).astype(np.int)]
        self.grid = np.zeros((self.grid_size[0], self.grid_size[1], self.grid_size[2]))

        ##### Real-time Visualization Variables #####
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.obs_ax = plt.figure().add_subplot(projection='3d')
        self.obs_ax.set_box_aspect([0.6, 0.8, 0.11])
        self.obs_ax.axis('off')
        self.obs_ax.view_init(elev=26, azim=130)
        # self.scatter = self.ax.scatter([], [], [])
        (self.graph,) = self.ax.plot([], [], [], marker='o', markersize=2)
        self.ax.set_xlim(threshold['x'].min, threshold['x'].max)
        self.ax.set_ylim(threshold['y'].min, threshold['y'].max)
        self.ax.set_zlim(threshold['z'].min, threshold['z'].max)
        self.ax.set_xticks(np.around(np.linspace(threshold['x'].min, threshold['x'].max, 2), decimals=2))
        self.ax.set_yticks(np.around(np.linspace(threshold['y'].min, threshold['y'].max, 2), decimals=2))
        self.ax.set_zticks(np.around(np.linspace(threshold['z'].min, threshold['z'].max, 2), decimals=2))
        font = {'family': 'serif',
                'weight': 'normal',
                'size': 13,
                }
        tex_fonts = {
            # Use LaTeX to write all text
            "text.usetex": True,
            "font.family": "serif",
            # Use 10pt font in plots, to match 10pt font in document
            "axes.labelsize": 10,
            "font.size": 10,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid.axis": "y",
            "grid.alpha": 0.3
        }
        plt.rcParams.update(tex_fonts)
        self.ax.set_xlabel('X', fontdict=font)
        self.ax.set_ylabel('Y', fontdict=font)
        self.ax.set_zlabel('Z', fontdict=font)
        for axis in [self.ax.xaxis, self.ax.yaxis, self.ax.zaxis]:
            axis.set_ticklabels(axis.get_ticklabels(), fontdict=font)
        ani = animation.FuncAnimation(self.fig, self.update, frames=360, interval=40, blit=True)
        self.ax.set_aspect("equal")
        plt.show(block=False)
        plt.pause(0.1)
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.graph)
        self.fig.canvas.blit(self.fig.bbox)
        self.is_ready = False
        self.vis_obs = False
        self.local_grid = np.array([0])

    def update(self, i):
        self.ax.view_init(elev=30, azim=i)
        return self.ax.collections

    def insert_data(self, history, pcd):
        if len(pcd) < 1:
            return

        self.trajectory.append(history["pose"][0])
        self.orientation.append(history["pose"][1])

        if len(self.trajectory) >= 1:
            self.is_ready = True
        else:
            self.is_ready = False

        ##### Checking every 'sampling' number of pointclouds are already in the grid or not        
        pcd = self._conver2global(pcd)
        pc_idx = self._convert_grid_index(pcd, self.th_min)
        idx, self.grid = filter_grid(self.grid, pc_idx)
        pcd = pcd[idx.astype(np.bool)]
        self.diff, _ = np.shape(pcd)
        if self.diff == 0:
            return

        if self.pcd.shape[0] < 3:
            self.last_index_acc = 0
            self.pcd = pcd
        else:
            self.last_index_acc = self.pcd.shape[0]
            self.pcd = np.concatenate((self.pcd, pcd), axis=0)

    def update_plt(self):
        ani = animation.FuncAnimation(self.fig, self.update, frames=360, interval=40, blit=True)

    def realtime_visualize(self, obs):
        if self.is_ready:
            if self.vis_obs:
                self.scatter.remove()
                x, y, z = np.indices(self.local_grid.shape).reshape(3, -1)
                values = self.local_grid.flatten()
                threshold = 0.1
                indices = np.where(values > threshold)
                x_filtered = x[indices]
                y_filtered = y[indices]
                z_filtered = z[indices]
                values_filtered = values[indices]
                cmap = 'Reds'
                self.ax.set_xlim(0, 20)
                self.ax.set_ylim(0, 20)
                self.ax.set_zlim(0, 20)
                marker_size = 20
                self.scatter = self.ax.scatter(x_filtered, y_filtered, z_filtered, c=values_filtered, cmap=cmap,
                                               s=marker_size)
                plt.pause(0.001)
            else:
                try:
                    data = self.pcd.T
                    self.graph.set_data(data[0][::2], data[1][::2])
                    self.graph.set_3d_properties(data[2][::2])
                    self.ax.draw_artist(self.graph)
                    self.fig.canvas.blit(self.fig.bbox)
                    self.fig.canvas.flush_events()
                except:
                    pass
                image_blurred = cv2.blur(obs, ksize=(15, 15))
                # Define the X, Y, and Z dimensions based on the image shape
                self.obs_ax.clear()
                self.obs_ax.axis("off")
                x = np.linspace(0, obs.shape[1], obs.shape[1])
                y = np.linspace(0, obs.shape[0], obs.shape[0])
                x_interp = np.linspace(0, obs.shape[1], 2 * obs.shape[1])
                y_interp = np.linspace(0, obs.shape[0], 2 * obs.shape[0])
                X_interp, Y_interp = np.meshgrid(x_interp, y_interp)
                Z = image_blurred  # Use the red channel as the height
                Z_interp = interp2d(range(image_blurred.shape[1]), range(image_blurred.shape[0]), Z)(x_interp, y_interp)
                if not np.max(Z_interp) > 1e-10:
                    Z_interp -= 0.002
                # Plot the 3D surface
                surf = self.obs_ax.plot_surface(X_interp, Y_interp, Z_interp+0.002, cmap='plasma', alpha=.8)
                X, Y = np.meshgrid(x, y)
                self.obs_ax.contourf(X, Y, obs, zdir='z', offset=0, cmap='gray', alpha=.97, antialiased=True)
                plt.show(block=False)
    def visualize_result(self, mesh_gen=False):
        self.o3pcd.points = o3d.utility.Vector3dVector(self.pcd)
        self.o3pcd.paint_uniform_color([0, 1, 0])
        # np.save(path, self.pcd)
        if mesh_gen:
            self.o3pcd.estimate_normals()
            self.o3pcd.orient_normals_consistent_tangent_plane(k=15)
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.o3pcd, depth=8)
            mesh = mesh.filter_smooth_simple(100)
            mesh.translate([0, 0.2, 0])
            o3d.visualization.draw_geometries([mesh])
        else:
            geoms = [
                {'name': 'cube', 'geometry': self.obj_gt, 'material': self.mat_obj},
                {'name': 'pcd', 'geometry': self.o3pcd},
            ]
            o3d.visualization.draw(geoms, show_skybox=False)

    def reset(self, obj_idx, threshold):
        ##### Reset History
        self.trajectory.clear()
        self.orientation.clear()
        self.pcd = np.array([0])
        self.o3pcd.clear()

        ##### Reset GT Object
        urdf_path = self.cfg.object.urdf_path[obj_idx]
        urdf_xml_tree = ET.parse(urdf_path)
        obj_path = os.path.join(os.path.dirname(urdf_path),
                                urdf_xml_tree.find("link/visual/geometry/mesh").get("filename"))
        self.obj_gt = o3d.io.read_triangle_mesh(obj_path)
        scale_factors = np.array(urdf_xml_tree.find("link/visual/geometry/mesh").get("scale").split(' ')).astype(
            np.float)
        self.preprocessing(self.cfg, scale_factors)
        self.gt_tree = o3d.geometry.KDTreeFlann(self.pcd_gt)

        ##### Reset Global Grid
        self.threshold = threshold
        self.th_min = np.array([threshold["x"].min, threshold["y"].min, threshold["z"].min])
        self.grid_size = [np.ceil((threshold["x"].max - threshold["x"].min) / self.resolution).astype(np.int),
                          np.ceil((threshold["y"].max - threshold["y"].min) / self.resolution).astype(np.int),
                          np.ceil((threshold["z"].max - threshold["z"].min) / self.resolution).astype(np.int)]
        self.grid = np.zeros((self.grid_size[0], self.grid_size[1], self.grid_size[2]))
        self.pcd_gt.paint_uniform_color([0.5, 0.5, 0.5])
        self.coverage = 0.0
        self.chamfer = 0.0
        self.last_index_acc = 0
        self.pcd_gt_grid = np.zeros(len(self.pcd_gt.points), dtype=int)

        # Reset Plots
        self.ax.set_xlim(threshold['x'].min, threshold['x'].max)
        self.ax.set_ylim(threshold['y'].min, threshold['y'].max)
        self.ax.set_zlim(threshold['z'].min, threshold['z'].max)
        self.ax.set_xticks(np.around(np.linspace(threshold['x'].min, threshold['x'].max, 2), decimals=2))
        self.ax.set_yticks(np.around(np.linspace(threshold['y'].min, threshold['y'].max, 2), decimals=2))
        self.ax.set_zticks(np.around(np.linspace(threshold['z'].min, threshold['z'].max, 2), decimals=2))

    def _convert_grid_index(self, point, origin, order=1):
        return np.floor((point - origin) / (order * self.resolution)).astype(np.int)

    def _reverse2point(self, index, origin):
        return (self.resolution * index + origin).astype(np.float64)

    def _conver2global(self, pcd):
        rot = R.from_quat(self.orientation[-1]).inv().as_matrix()
        return np.matmul(pcd, rot) + self.trajectory[-1]

    def convert_local_grid(self, pose, ori, size):
        order = 4
        sigma = 1.0

        local_grid = np.zeros((size, size, size), dtype=float)
        offset = np.array([size / 2, size / 2, size / 2]).astype(np.int)
        rot = R.from_quat(ori).as_matrix()
        trans = -np.matmul(rot, pose)

        index = self._convert_grid_index(pose, self.th_min)
        range_max = index + order * size / 2
        range_min = index - order * size / 2
        limit = np.shape(self.grid)
        zero = np.zeros(3).astype(np.int)
        range_max = np.clip(range_max, zero, limit).astype(np.int)
        range_min = np.clip(range_min, zero, limit).astype(np.int)
        global_grid = self.grid[range_min[0]:range_max[0], range_min[1]:range_max[1], range_min[2]:range_max[2]]
        mask = global_grid > 0
        indices = np.argwhere(mask) + range_min
        if len(indices) < 1:
            return local_grid

        global_point = self._reverse2point(indices, self.th_min)
        local_point = np.matmul(global_point - pose, rot)

        local_index = self._convert_grid_index(local_point, np.zeros(3), order=order) + offset

        grid_idx = np.where(np.all((local_index < size) & (local_index >= 0), axis=1))
        local_index = local_index[grid_idx]
        local_grid[local_index[:, 0], local_index[:, 1], local_index[:, 2]] = 1.0

        local_grid = gaussian_filter(local_grid, sigma=sigma)
        if self.vis_obs:
            self.local_grid = gaussian_filter(local_grid, sigma=sigma)

        return local_grid

    def preprocessing(self, env, scale):
        base_rot = R.from_quat(self.cfg.pose.base_orientation).as_matrix()
        # scale -= 0.02
        self.obj_gt.vertices = o3d.utility.Vector3dVector(scale * np.asarray(self.obj_gt.vertices))
        self.obj_gt.translate(env.pose.base_position).rotate(
            base_rot)  # this rotation is not necessarily based on the same origin as in pybullet rotation
        self.pcd_gt = self.obj_gt.sample_points_uniformly(number_of_points=1000000)

    def compute_coverage(self, accumulate=True, visualize=True):
        if self.pcd.shape[0] < 3:
            return

        if not accumulate:
            self.last_index_acc = 0
        for pc in self.pcd[self.last_index_acc:-1]:
            [_, idx, _] = self.gt_tree.search_radius_vector_3d(pc, self.D_THRESHOLD)
            self.pcd_gt_grid[idx] = 1
            if visualize:
                pass

        self.coverage = (np.float32(np.sum(self.pcd_gt_grid) / len(self.pcd_gt.points))) * 100

    def compute_chamfer_distance(self):
        if self.pcd.shape[0] < 3:
            return

        point1 = np.asarray(self.pcd_gt.points)

        self.o3pcd.points = o3d.utility.Vector3dVector(self.pcd)
        pcd_tree = o3d.geometry.KDTreeFlann(self.o3pcd)
        idx_12 = []
        idx_21 = []

        for pc in self.pcd:
            [_, idx, _] = self.gt_tree.search_knn_vector_3d(pc, 1)
            idx_12.append(idx)

        for pc in point1:
            [_, idx, _] = pcd_tree.search_knn_vector_3d(pc, 1)
            idx_21.append(idx)

        chamfer1 = np.linalg.norm((self.pcd - point1[idx_12].squeeze(axis=1)), ord=2, axis=1).mean()
        chamfer2 = np.linalg.norm((point1 - self.pcd[idx_21].squeeze(axis=1)), ord=2, axis=1).mean()
        self.chamfer = chamfer1 + chamfer2

    def compute_metrics(self, visualize=True):
        if self.pcd.shape[0] < 3:
            return
        point1 = np.asarray(self.pcd_gt.points)
        self.o3pcd.points = o3d.utility.Vector3dVector(self.pcd)
        pcd_tree = o3d.geometry.KDTreeFlann(self.o3pcd)
        idx_12 = []
        idx_21 = []

        for pc in self.pcd:
            [_, idx_cov, _] = self.gt_tree.search_radius_vector_3d(pc, self.D_THRESHOLD)
            [_, idx_cham, _] = self.gt_tree.search_knn_vector_3d(pc, 1)
            self.pcd_gt_grid[idx_cov] = 1
            idx_12.append(idx_cham)
            if visualize:
                np.asarray(self.pcd_gt.colors)[idx_cov, :] = [0, 1, 0]

        for pc in point1:
            [_, idx_cham, _] = pcd_tree.search_knn_vector_3d(pc, 1)
            idx_21.append(idx_cham)

        chamfer1 = np.linalg.norm((self.pcd - point1[idx_12].squeeze(axis=1)), ord=2, axis=1).mean()
        chamfer2 = np.linalg.norm((point1 - self.pcd[idx_21].squeeze(axis=1)), ord=2, axis=1).mean()
        self.chamfer = chamfer1 + chamfer2
        self.coverage = np.float32(np.sum(self.pcd_gt_grid) / len(self.pcd_gt.points)) * 100


@nb.jit(nopython=True)
def filter_grid(grid, pc_idx):
    result = np.zeros(len(pc_idx))
    for i in range(0, len(pc_idx)):
        if grid[pc_idx[i, 0], pc_idx[i, 1], pc_idx[i, 2]] < 1:
            result[i] = True
            grid[pc_idx[i, 0], pc_idx[i, 1], pc_idx[i, 2]] = 1

    return result, grid
