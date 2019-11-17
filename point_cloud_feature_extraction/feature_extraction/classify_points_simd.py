import ds
import open3d as o3d
import numpy as np
import heapq
import math
import time, datetime

# RAD_NEIGHBORS = 0.12
# MIN_NEIGHBORS = 5 # used for outlier removal
# MAX_NEIGHBORS = 50
DOWN_SAMPLE_VOXEL_SIZE = 0.0045
NUM_NEIGHBORS = 12
CORNER_ALHPA = 0.2
TAU_MIN = 1.0
TAU_MAX = 1.5

class PCDPointClassifier:
    def __init__(self, pcd : o3d.geometry.PointCloud):
        self.pcd = pcd
        self.line_set = o3d.geometry.LineSet()
        self.line_set.points = self.pcd.points
        self.kdtree = o3d.geometry.KDTreeFlann(pcd)

        points = np.asarray(self.pcd.points)
        self.neighbors = np.zeros((points.shape[0], NUM_NEIGHBORS), dtype=np.int32)
        for i in range(0, points.shape[0]):
            [k, idx, _] = self.kdtree.search_knn_vector_3d(points[i], NUM_NEIGHBORS + 1)
            self.neighbors[i] = idx[1:]

        start_time = time.time()
        self.calc_prerequisites()
        elapsed_time = time.time() - start_time
        print('Elapsed Time [calc_prerequisites]: {}'.format(datetime.timedelta(seconds=elapsed_time)))

        start_time = time.time()
        self.build_debug_lineset()
        elapsed_time = time.time() - start_time
        print('Elapsed Time [build_debug_lineset]: {}'.format(datetime.timedelta(seconds=elapsed_time)))

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

    def build_debug_lineset(self):
        points = np.asarray(self.pcd.points)
        neighbors = points[self.neighbors[:,]]
        prio_queue = [];
        disjoint =  ds.disjoint_set(points.shape[0])
        graph = ds.graph()

        for p in np.arange(points.shape[0]):
            n_indices = self.neighbors[p]
            for n in np.arange(n_indices.shape[0]):
                weight = self.w_edge[p][n]
                if weight > TAU_MIN and weight < TAU_MAX:
                    prio_queue.append((weight, p, n_indices[n]))
        
        heapq.heapify(prio_queue)

        count = 0
        while len(prio_queue) > 0:
            count += 1
            if count % 100 == 0:
                print(len(prio_queue))

            edge = heapq.heappop(prio_queue)
            if disjoint.union(edge[1], edge[2]) or graph.path_is_longer(edge[1], edge[2], 200):
                graph.add_edge(edge[1], edge[2])
        
        lines = []
        num_items = len(graph.edges)
        count = 0
        for p in graph.edges:

            count += 1
            if count % 1000 == 0:
                print("{}/{}".format(count, num_items))

            for q in graph.edges[p]:
                if p > q:
                    lines.append((p, q))
        
        self.line_set.lines = o3d.utility.Vector2iVector(lines)
         
    def colorize(self):
        points = np.asarray(self.pcd.points)
        colors = np.asarray(self.pcd.colors)
        self.pcd.paint_uniform_color([0,0,0.2])
        
        for i in range(0, points.shape[0]):
            c = min(self.w_edge[i]) - 0.6
            colors[i][0] = c
   
    def visualize(self, draw_pcd=True, draw_lines=True):
        toDraw = []

        if draw_pcd:
            toDraw.append(self.pcd)

        if draw_lines:
            toDraw.append(self.line_set)
        
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
    radii = o3d.utility.DoubleVector([DOWN_SAMPLE_VOXEL_SIZE * 1.5] * 2)
    tris = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(down_pcd, radii)
    o3d.visualization.draw_geometries([down_pcd, tris])
    # print(len(down_pcd.points))
    # classifier = PCDPointClassifier(down_pcd)
    # classifier.colorize()
    # classifier.visualize()