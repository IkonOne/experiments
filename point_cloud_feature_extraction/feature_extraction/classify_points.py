import open3d as o3d
import numpy as np
import heapq
import math

RAD_NEIGHBORS = 0.12
MIN_NEIGHBORS = 5 # used for outlier removal
MAX_NEIGHBORS = 50
DOWN_SAMPLE_VOXEL_SIZE = 0.02
# NUM_NEIGHBORS = 16
CORNER_ALHPA = 0.55
TAU = 1.5

class PCDPointClassifier:
    def __init__(self, pcd):
        self.pcd = pcd
        self.kdtree = o3d.geometry.KDTreeFlann(pcd)

        points = np.asarray(self.pcd.points)
        self.neighbors = []
        self.avg_neighbor_dists = []
        self.centroids = []
        self.corr_mats = []
        self.eig = []
        self.curvatures = []
        self.betas = []
        self.weights = []
        self.max_curvature = -1

        print('Processing point cloud with [{}] points.'.format(points.shape[0]))
        print('Calculating prerequisites.')
        for i in range(0, points.shape[0]):
            if i % 1000 == 0:
                print('{} / {}'.format(i, points.shape[0]))

            # [k, idx, _] = self.kdtree.search_knn_vector_3d(points[i], NUM_NEIGHBORS + 1)
            [k, idx, _] = self.kdtree.search_hybrid_vector_3d(points[i], RAD_NEIGHBORS, MAX_NEIGHBORS) 
            self.neighbors.append(idx[1:])

            self.centroids.append(self.calc_centroid_of_point(i))
            self.corr_mats.append(self.calc_corr_mat_of_point(i))
            self.eig.append(self.calc_eigens_of_point(i))
            (curvature, beta, avg_neighbor_dist) = self.calc_curvature_and_beta_of_point(i)
            self.curvatures.append(curvature)
            self.betas.append(beta)
            self.avg_neighbor_dists.append(avg_neighbor_dist)
        
        print('Calculating max curvature.')
        self.max_curvature = max(self.curvatures)
        # for i in range(0, points.shape[0]):
        #     if self.curvatures[i] > self.max_curvature:
        #         self.max_curvature = self.curvatures[i]

        print('Calculating weights.')
        for i in range(0, points.shape[0]):
            if i % 1000 == 0:
                print('{} / {}'.format(i, points.shape[0]))
            self.weights.append(self.pre_calc_weights_of_point(i))

    def calc_centroid_of_point(self, i):
        points = self.pcd.points
        num_neighbors = len(self.neighbors[i])
        p = points[i]
        centroid = np.zeros(3)
        for n in range(0, num_neighbors):
            centroid += points[self.neighbors[i][n]]
        return centroid / num_neighbors
    
    def calc_corr_mat_of_point(self, i):
        points = self.pcd.points
        num_neighbors = len(self.neighbors[i])
        c = self.centroids[i]
        corr_mat = np.zeros((3, 3))
        for n in range(0, num_neighbors):
            local = points[self.neighbors[i][n]] - c
            corr_mat += np.outer(local, local)
        return corr_mat / num_neighbors
    
    def calc_eigens_of_point(self, i):
        (vals, vecs) = np.linalg.eig(self.corr_mats[i])
        idx = np.argsort(vals)
        return {
            'vals': vals[idx],
            'vecs': vecs[idx]
        }
    
    def calc_curvature_and_beta_of_point(self, i):
        points = self.pcd.points
        num_neighbors = len(self.neighbors[i])

        p = self.pcd.points[i]
        c = self.centroids[i]
        n = self.eig[i]['vecs'][0]
        p_local = p - c
        d = np.sqrt(np.dot(p_local, p_local))

        # calculate curvature
        avg_neighbor_dist = 0
        for n in range(0, num_neighbors):
            local = points[self.neighbors[i][n]] - p
            avg_neighbor_dist += np.sqrt(np.dot(local, local))
        avg_neighbor_dist /= num_neighbors
        curvature = 2 * d / (avg_neighbor_dist ** 2)

        # calculate beta
        e1 = self.eig[i]['vecs'][1]
        e2 = self.eig[i]['vecs'][2]
        xy_offset = [
            np.dot(p_local, e1),
            np.dot(p_local, e2)
        ]

        xy_thetas = []
        for n in range(0, num_neighbors):
            n_local = points[self.neighbors[i][n]] - c
            xy_n = [
                np.dot(n_local, e1),
                np.dot(n_local, e2)
            ]
            xy_thetas.append(math.atan2(xy_n[1], xy_n[0]))
        
        xy_idx = np.argsort(xy_thetas)
        beta = -1
        for n in range(1, num_neighbors):
            b = xy_thetas[xy_idx[n - 1]] - xy_thetas[xy_idx[n]]
            if b > beta:
                beta = b
        
        return (curvature, beta, avg_neighbor_dist)
    
    def pre_calc_weights_of_point(self, i):
        eig_vals = self.eig[i]['vals']
        eig_vecs = self.eig[i]['vecs']

        w_curvature = 1 - self.curvatures[i] / self.max_curvature

        v_crease = max(
            eig_vals[1] - eig_vals[0],
            abs(eig_vals[2] - (eig_vals[0] + eig_vals[1]))
        )
        v_crease /= eig_vals[2]
        v_crease *= eig_vecs[2]

        v_border1 = abs(eig_vals[2] - 2 * eig_vals[1])
        v_border1 /= eig_vals[2]
        v_border1 *= eig_vecs[2]

        w_border2 = 1 - self.betas[i] / (2 * np.pi)

        w_corner = (eig_vals[2] - eig_vals[0]) / eig_vals[2]

        return {
            'w_curvature': w_curvature,
            'v_crease': v_crease,
            'v_border1': v_border1,
            'w_border2': w_border2,
            'w_corner': w_corner
        }

    def calc_weight_crease_of_point(self, i):
        eig_vals = self.eig[i]['vals']
        eig_vecs = self.eig[i]['vecs']

    
    def build_feature_lines(self):
        points = self.pcd.points

        edge_prio_queue  = []

        points = np.asarray(self.pcd.points)

        print('-- Calculating edge weights')
        for i in range(0, len(points)):
            p = points[i]
            nn = self.neighbors[i]
            self.weights[i]['w_edge'] = []
            weights = self.weights[i]
            w_edge = 0

            for n in range(0, len(nn)):
                neighbor = points[nn[n]]
                edge = neighbor - p

                w_edge = CORNER_ALHPA * (self.weights[i]['w_curvature'] + self.weights[n]['w_curvature'])
                w_edge += (1 - CORNER_ALHPA) * min(
                    abs(np.dot(self.weights[i]['v_crease'], edge)),
                    self.weights[i]['w_corner']
                )
                w_edge += min(
                    abs(np.dot(self.weights[i]['v_crease'], edge)),
                    self.weights[i]['w_corner']
                )
                w_edge += (2 * np.sqrt(np.dot(edge, edge))) / (self.avg_neighbor_dists[i] + self.avg_neighbor_dists[n])

                self.weights[i]['w_edge'].append(w_edge)

                if w_edge < TAU:
                    heapq.heappush(edge_prio_queue, (w_edge, i, nn[n]))
           
    def colorize(self):
        points = np.asarray(self.pcd.points)
        colors = np.asarray(self.pcd.colors)
        self.pcd.paint_uniform_color([0,0,0.2])
        
        for i in range(0, points.shape[0]):
            c = 1.0 - min(self.weights[i]['w_edge'])
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
    uniform_down_pcd = pcd.uniform_down_sample(every_k_points=5)
    voxel_down_pcd = uniform_down_pcd.voxel_down_sample(voxel_size=DOWN_SAMPLE_VOXEL_SIZE)
    cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points = math.floor(MAX_NEIGHBORS / 2), radius=RAD_NEIGHBORS)
    # cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors = math.floor(MAX_NEIGHBORS / 2), std_ratio=2.0)
    down_pcd = voxel_down_pcd.select_down_sample(ind)
    classifier = PCDPointClassifier(down_pcd)
    classifier.build_feature_lines()
    classifier.colorize()
    classifier.visualize()