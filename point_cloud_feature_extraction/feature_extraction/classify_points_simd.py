import open3d as o3d
import numpy as np
import heapq
import math
import time, datetime

# RAD_NEIGHBORS = 0.12
# MIN_NEIGHBORS = 5 # used for outlier removal
# MAX_NEIGHBORS = 50
DOWN_SAMPLE_VOXEL_SIZE = 0.01
NUM_NEIGHBORS = 32
CORNER_ALHPA = 0.75
TAU = 1.5

class PCDPointClassifier:
    def __init__(self, pcd):
        self.pcd = pcd
        self.kdtree = o3d.geometry.KDTreeFlann(pcd)

        points = np.asarray(self.pcd.points)
        self.neighbors = np.zeros((points.shape[0], NUM_NEIGHBORS), dtype=np.int32)
        for i in range(0, points.shape[0]):
            [k, idx, _] = self.kdtree.search_knn_vector_3d(points[i], NUM_NEIGHBORS + 1)
            self.neighbors[i] = idx[1:]

        start_time = time.time()
        self.calc_prerequisites()
        end_time = time.time() - start_time
        print('Elapsed Time: {}'.format(datetime.timedelta(seconds=end_time)))

    def calc_prerequisites(self):
        points = np.asarray(self.pcd.points)
        neighbors = points[self.neighbors[:,]]

        centroids = np.mean(points[self.neighbors], axis=1)

        local_neighbors = neighbors - centroids.reshape((len(points), 1, 3))
        local_points = points - centroids

        point_centroid_distances = np.zeros(points.shape[0])
        avg_neighbor_dist = np.zeros(points.shape[0])
        corr_mats = np.zeros((points.shape[0], 3, 3))
        for i in np.arange(points.shape[0]):
            corr_mats[i] = np.matmul(local_neighbors[i].T, local_neighbors[i])
            corr_mats[i] /= NUM_NEIGHBORS

            point_centroid_distances[i] = np.sqrt(np.vdot(local_points[i], local_points[i]))
            avg_neighbor_dist[i] = np.sqrt(np.vdot(local_neighbors[i], local_neighbors[i]))
        

        eigvals, eigvecs = np.linalg.eig(corr_mats)
        for i in np.arange(points.shape[0]):
            args = np.argsort(eigvals[i])
            eigvals[i] = eigvals[i][args]
            eigvecs[i] = eigvecs[i][args]

        normals = eigvecs[:,0]

        crease = np.maximum(
            eigvals[:,1] - eigvals[:,0],
            np.abs(eigvals[:,2] - (eigvals[:,0] + eigvals[:,1]))
        )
        crease /= eigvals[:,2]
        vecs = np.hsplit(eigvecs, 3)[2].reshape((crease.shape[0], 3))

        self.w_curvature = 2 * point_centroid_distances / (avg_neighbor_dist ** 2)
        self.v_crease = np.multiply(vecs, crease.reshape((crease.shape[0], 1)))
        self.w_corner = np.divide(eigvals[:,2] - eigvals[:,0], eigvals[:,2])

        self.w_edge = np.zeros((points.shape[0], NUM_NEIGHBORS))
        for i in np.arange(points.shape[0]):
            n_indices = self.neighbors[i]
            edges = points[n_indices] - points[i]
            edges_mags = np.sqrt(np.einsum('ij,ij->i', edges, edges)) # dot product
            edges_norm = np.divide(edges, edges_mags.reshape(NUM_NEIGHBORS, 1))

            self.w_edge[i] = CORNER_ALHPA * (self.w_curvature[i] + self.w_curvature[self.neighbors[i]])
            self.w_edge[i] += (2 * edges_mags) / (avg_neighbor_dist[i] + avg_neighbor_dist[n_indices])

            point_min = np.minimum(
                np.absolute(np.einsum('i,ji->j', self.v_crease[i], edges_norm)), # dot product
                self.w_corner[i]
            )

            self.w_edge[i] += (1 - CORNER_ALHPA) * (
                point_min + np.minimum(
                    np.absolute(np.einsum('ij,ij->i', self.v_crease[n_indices], edges_norm)), # dot product
                    self.w_corner[n_indices]
                )
            )
         
    def colorize(self):
        points = np.asarray(self.pcd.points)
        colors = np.asarray(self.pcd.colors)
        self.pcd.paint_uniform_color([0,0,0.2])
        
        for i in range(0, points.shape[0]):
            c = min(self.w_edge[i]) - 0.6
            colors[i][0] = c
   
    def visualize(self, draw_pcd=True):
        toDraw = []

        if draw_pcd:
            toDraw.append(self.pcd)
        
        o3d.visualization.draw_geometries(toDraw)

if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud("./TestData/fragment.ply")
    pcd.transform([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    # uniform_down_pcd = pcd.uniform_down_sample(every_k_points=5)
    # voxel_down_pcd = uniform_down_pcd.voxel_down_sample(voxel_size=DOWN_SAMPLE_VOXEL_SIZE)
    # cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points = math.floor(MAX_NEIGHBORS / 2), radius=RAD_NEIGHBORS)
    # cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors = math.floor(MAX_NEIGHBORS / 2), std_ratio=2.0)
    # down_pcd = voxel_down_pcd.select_down_sample(ind)
    # classifier = PCDPointClassifier(down_pcd)
    # classifier.build_feature_lines()

    down_pcd = pcd.voxel_down_sample(voxel_size=DOWN_SAMPLE_VOXEL_SIZE)
    print(len(down_pcd.points))
    classifier = PCDPointClassifier(down_pcd)
    classifier.colorize()
    classifier.visualize()