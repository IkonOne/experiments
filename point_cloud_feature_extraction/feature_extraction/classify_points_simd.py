import ds
import open3d as o3d
import numpy as np
import heapq
import math
import time, datetime

# RAD_NEIGHBORS = 0.12
# MIN_NEIGHBORS = 5 # used for outlier removal
# MAX_NEIGHBORS = 50

DOWN_SAMPLE_VOXEL_SIZE = 0.04
NUM_NEIGHBORS = 16
CORNER_ALHPA = 0.8
TAU_MIN = 0.0
TAU_MAX = 2

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

        # pg. 3, eq. 2
        centroids = np.mean(neighbors, axis=1)

        # pg. 3, eq. 3
        corr_mats = np.zeros((points.shape[0], 3, 3))
        local_neighbors_centroids = neighbors - centroids.reshape((len(points), 1, 3))
        for i in np.arange(points.shape[0]):
            for n in np.arange(neighbors[i].shape[0]):
                corr_mats[i] += np.outer(local_neighbors_centroids[i][n], local_neighbors_centroids[i][n])
            corr_mats[i] /= NUM_NEIGHBORS

        eigvals, eigvecs = np.linalg.eig(corr_mats)
        for i in np.arange(points.shape[0]):
            args = np.argsort(eigvals[i])
            eigvals[i] = eigvals[i][args]
            eigvecs[i] = eigvecs[i][args]
        
        self.eigvecs = eigvecs

        # pg. 4, eq. 1,4,5
        local_points = points - centroids # p - c_i
        point_centroid_distances = np.zeros(points.shape[0]) # d
        avg_neighbor_dist = np.zeros(points.shape[0]) # mu_i
        for i in np.arange(points.shape[0]):
            point_centroid_distances[i] += abs(np.vdot(local_points[i], eigvecs[i, 0]))

            for n in np.arange(neighbors[i].shape[0]):
                avg_neighbor_dist[i] += np.linalg.norm(points[i] - neighbors[i][n])
            avg_neighbor_dist[i] /= NUM_NEIGHBORS
        
        self.w_curvature = 2 * point_centroid_distances / np.power(avg_neighbor_dist, 2)
        max_curvature = max(self.w_curvature)
        self.w_curvature = 1 - (self.w_curvature / max_curvature)

        # pg. 4, eq. 6
        crease = np.maximum(
            eigvals[:,1] - eigvals[:,0],
            np.abs(eigvals[:,2] - (eigvals[:,0] + eigvals[:,1]))
        )
        crease /= eigvals[:,2]
        self.v_crease = np.multiply(crease.reshape((crease.shape[0], 1)), eigvecs[:,2])

        # pg. 4, eq. 9
        self.w_corner = np.divide(eigvals[:,2] - eigvals[:,0], eigvals[:,2])

        # pg. 5, eq. w_e
        self.w_edge = np.zeros((points.shape[0], NUM_NEIGHBORS))
        for i in np.arange(points.shape[0]):
            edges = points[i] - points[self.neighbors[i]]
            edges_mags = np.sqrt(np.einsum('ij,ij->i', edges, edges)) # dot product
            edges_norm = np.divide(edges, edges_mags.reshape(NUM_NEIGHBORS, 1))

            point_min = np.minimum(
                np.absolute(np.einsum('i,ji->j', self.v_crease[i], edges_norm)), # dot product
                self.w_corner[i]
            )

            neighbors_min = np.minimum(
                np.absolute(np.einsum('ij,ij->i', self.v_crease[self.neighbors[i]], edges_norm)),
                self.w_corner[self.neighbors[i]]
            )

            self.w_edge[i] = CORNER_ALHPA * (self.w_curvature[i] + self.w_curvature[self.neighbors[i]])
            self.w_edge[i] += (1 - CORNER_ALHPA) * (point_min + neighbors_min)
            self.w_edge[i] += (2 * edges_mags) / (avg_neighbor_dist[i] + avg_neighbor_dist[self.neighbors[i]])

    def build_debug_lineset(self):
        points = np.asarray(self.pcd.points)
        neighbors = points[self.neighbors[:,]]
        prio_queue = [];
        disjoint =  ds.disjoint_set(points.shape[0])
        graph = ds.graph(points.shape[0])

        for p in np.arange(points.shape[0]):
            n_indices = self.neighbors[p]
            for n in np.arange(n_indices.shape[0]):
                weight = self.w_edge[p][n]
                if weight > TAU_MIN and weight < TAU_MAX:
                    prio_queue.append((weight, p, n_indices[n]))
        
        heapq.heapify(prio_queue)
        print("len(prio_queue) = {}".format(len(prio_queue)))

        count = 0
        while len(prio_queue) > 0:

            edge = heapq.heappop(prio_queue)
            if disjoint.union(edge[1], edge[2]) or graph.path_is_longer(edge[1], edge[2], 200):
                graph.add_edge(edge[1], edge[2])
        
        lines = []
        num_items = len(graph.edges)
        for p in range(0, len(graph.edges)):
            if graph.edges[p] is None:
                continue
            for q in graph.edges[p]:
                if p > q:
                    lines.append((p, q))
        
        self.line_set.lines = o3d.utility.Vector2iVector(lines)
         
    def colorize(self):
        points = np.asarray(self.pcd.points)
        self.pcd.paint_uniform_color([0,0,0])
        colors = np.asarray(self.pcd.colors)

        for i in range(0, points.shape[0]):
            # c = 0 if np.sqrt(np.vdot(self.v_crease[i], self.v_crease[i])) > TAU_MAX else 1
            # vec = np.vdot(self.v_crease[i], self.eigvecs[i,2]) * self.eigvecs[i,2]
            # c = 1 - min(np.sqrt(np.vdot(vec, vec)), self.w_corner[i])
            # c = 1 - self.v_crease[i]
            # c = 1 - self.w_corner[i]
            # c = 1 - np.min(self.w_edge[i])
            c = 1 - np.min(self.w_edge[i])
            colors[i] = [c, 0, 0]
  
    def visualize(self, draw_pcd=True, draw_lines=True, add_geoms=[]):
        toDraw = add_geoms

        if draw_pcd:
            toDraw.append(self.pcd)

        if draw_lines and len(self.line_set.lines) > 0:
            toDraw.append(self.line_set)
        
        o3d.visualization.draw_geometries(toDraw)

if __name__ == '__main__':
    # pcd = o3d.io.read_point_cloud("./TestData/fragment.ply")
    # pcd.transform([
    #     [1, 0, 0, 0],
    #     [0, -1, 0, 0],
    #     [0, 0, -1, 0],
    #     [0, 0, 0, 1]
    # ])

    box : o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_box()
    pcd = box.sample_points_uniformly(20000)

    inner_box = o3d.geometry.TriangleMesh.create_box(0.95, 0.95, 0.95)
    inner_box.translate([0.025] * 3)

    down_pcd = pcd.voxel_down_sample(voxel_size=DOWN_SAMPLE_VOXEL_SIZE)
    print(len(down_pcd.points))
    classifier = PCDPointClassifier(down_pcd)
    classifier.colorize()
    classifier.visualize(draw_lines=True, add_geoms=[inner_box])