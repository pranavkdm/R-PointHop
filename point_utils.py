import numpy as np
import torch
import threading
from sklearn.neighbors import NearestNeighbors as kNN
from sklearn.decomposition import PCA

def calc_distances(tmp, pts):
    '''
    :param tmp:(B, k, 3)/(B, 3)
    :param pts:(B, N, 3)
    :return:(B, N, k)/(B, N)
    '''
    if len(tmp.shape) == 2:
        tmp = np.expand_dims(tmp, axis=1)
    tmp_trans = np.transpose(tmp, [0,2,1])
    xy = np.matmul(pts, tmp_trans)
    pts_square = (pts**2).sum(axis=2, keepdims=True)
    tmp_square_trans = (tmp_trans**2).sum(axis=1, keepdims=True)
    return np.squeeze(pts_square + tmp_square_trans - 2 * xy)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = np.tile(np.arange(B).reshape(view_shape),repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def furthest_point_sample(pts, K):
    """
    Input:
        pts: pointcloud data, [B, N, C]
        K: number of samples
    Return:
        (B, K, 3)
    """
    # pts = np.expand_dims(pts, axis=0)
    B, N, C = pts.shape
    centroids = np.zeros((B, K), dtype=int)
    distance = np.ones((B, N), dtype=int) * 1e10
    # np.random.seed(0)
    farthest = np.random.randint(0, N, (B,))
    # farthest = np.array([0,0])
    batch_indices = np.arange(B)
    for i in range(K):
        centroids[:, i] = farthest
        centroid = pts[batch_indices, farthest, :].reshape(B, 1, 3)
        dist = np.sum((pts - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, axis=-1)

    return index_points(pts, centroids)

def furthest_point_sample_2(pts, local_kernels, local_mean, K):
    """
    Input:
        pts: pointcloud data, [B, N, C]
        K: number of samples
    Return:
        (B, K, 3)
    """
    # pts = np.expand_dims(pts, axis=0)
    B, N, C = pts.shape
    centroids = np.zeros((B, K), dtype=int)
    distance = np.ones((B, N), dtype=int) * 1e10
    # np.random.seed(0)
    farthest = np.random.randint(0, N, (B,))
    # farthest = np.array([0,0])
    batch_indices = np.arange(B)
    for i in range(K):
        centroids[:, i] = farthest
        centroid = pts[batch_indices, farthest, :].reshape(B, 1, 3)
        dist = np.sum((pts - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, axis=-1)

    # kernels = local_kernels.reshape(local_kernels.shape[0],local_kernels.shape[1],-1)
    kernels = index_points(local_kernels, centroids)
    # kernels = kernels.reshape(kernels.shape[0],kernels.shape[1],3,3)

    # print(centroids)

    return index_points(pts, centroids), kernels, index_points(local_mean, centroids)

def knn_query(new_pts, pts, n_sample, idx):
    '''
    new_pts:(B, K, 3)
    pts:(B, N, 3)
    n_sample:int
    :return: nn_idx (B, n_sample, K)
    '''
    distance_matrix = calc_distances(new_pts, pts)
    nn_idx = np.argpartition(distance_matrix, (0, n_sample), axis=1)[:, :n_sample, :]
    idx.append(nn_idx)


def knn(new_xyz, point_data, n_sample):
    idx1 = []
    idx2 = []
    idx3 = []
    idx4 = []
    idx5 = []
    idx6 = []
    idx7 = []
    idx8 = []
    threads = []
    batch_size = point_data.shape[0]//8
    t1 = threading.Thread(target=knn_query, args=(new_xyz[:batch_size], point_data[:batch_size], n_sample, idx1))
    threads.append(t1)
    t2 = threading.Thread(target=knn_query, args=(new_xyz[batch_size:2*batch_size], point_data[batch_size:2*batch_size], n_sample, idx2))
    threads.append(t2)
    t3 = threading.Thread(target=knn_query, args=(new_xyz[2*batch_size:3*batch_size], point_data[2*batch_size:3*batch_size], n_sample, idx3))
    threads.append(t3)
    t4 = threading.Thread(target=knn_query, args=(new_xyz[3*batch_size:4*batch_size], point_data[3*batch_size:4*batch_size], n_sample, idx4))
    threads.append(t4)
    t5 = threading.Thread(target=knn_query, args=(new_xyz[4*batch_size:5*batch_size], point_data[4*batch_size:5*batch_size], n_sample, idx5))
    threads.append(t5)
    t6 = threading.Thread(target=knn_query, args=(new_xyz[5*batch_size:6*batch_size], point_data[5*batch_size:6*batch_size], n_sample, idx6))
    threads.append(t6)
    t7 = threading.Thread(target=knn_query, args=(new_xyz[6*batch_size:7*batch_size], point_data[6*batch_size:7*batch_size], n_sample, idx7))
    threads.append(t7)
    t8 = threading.Thread(target=knn_query, args=(new_xyz[7*batch_size:], point_data[7*batch_size:], n_sample, idx8))
    threads.append(t8)

    for t in threads:
        t.setDaemon(False)
        t.start()
    for t in threads:
        if t.isAlive():
            t.join()
    idx = idx1 + idx2 + idx3 + idx4 + idx5 + idx6 + idx7 + idx8
    idx_tmp = np.concatenate(idx, axis=0)

    return idx_tmp

def calc_feature(pc_temp, pc_bin, pc_gather):
    value = np.multiply(pc_temp, pc_bin)
    value = np.sum(value, axis=2, keepdims=True)
    num = np.sum(pc_bin, axis=2, keepdims=True)
    final = np.squeeze(value/num, axis=(2,))
    pc_gather.append(final)


def calc_feature_single(pc_temp, pc_bin):
    value = np.multiply(pc_temp, pc_bin)
    value = np.sum(value, axis=2, keepdims=True)
    num = np.sum(pc_bin, axis=2, keepdims=True)
    final = np.squeeze(value/num, axis=(2,))
    return final


def gather_fea(nn_idx, point_data, fea, local_kernels, local_mean):
    """
    nn_idx:(B, n_sample, K)
    pts:(B, N, dim)
    :return: pc_n(B, K, dim_fea)
    """
    num_newpts = nn_idx.shape[2]
    pts_fea = np.concatenate([point_data, fea], axis=-1)
    num_dim = pts_fea.shape[2]

    pts_fea_expand = index_points(pts_fea, nn_idx)
    pts_fea_expand = pts_fea_expand.transpose(0, 2, 1, 3)  # (B, K, n_sample, dim)
    pc_temp = pts_fea_expand[..., 3:]

    pc_n_t = pts_fea_expand[..., :3]
    pc_n = np.zeros(pc_n_t.shape)
    for i in range(pc_n_t.shape[0]):
        for j in range(pc_n_t.shape[1]):
            pc_n[i,j] = (pc_n_t[i,j] -local_mean[i,j])@ local_kernels[i,j].T
    
    pc_n_center = np.expand_dims(pc_n[:, :, 0, :], axis=2)
    pc_n_uncentered = pc_n - pc_n_center

    pc_idx = []
    pc_idx.append(pc_n_uncentered[:, :, :, 0] >= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 0] <= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 1] >= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 1] <= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 2] >= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 2] <= 0)

    pc_bin = []
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[2] * pc_idx[4])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[2] * pc_idx[5])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[3] * pc_idx[4])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[3] * pc_idx[5])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[2] * pc_idx[4])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[2] * pc_idx[5])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[3] * pc_idx[4])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[3] * pc_idx[5])*1.0, axis=3))

    pc_gather1 = []
    pc_gather2 = []
    pc_gather3 = []
    pc_gather4 = []
    pc_gather5 = []
    pc_gather6 = []
    pc_gather7 = []
    pc_gather8 = []
    threads = []
    t1 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[0], pc_gather1))
    threads.append(t1)
    t2 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[1], pc_gather2))
    threads.append(t2)
    t3 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[2], pc_gather3))
    threads.append(t3)
    t4 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[3], pc_gather4))
    threads.append(t4)
    t5 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[4], pc_gather5))
    threads.append(t5)
    t6 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[5], pc_gather6))
    threads.append(t6)
    t7 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[6], pc_gather7))
    threads.append(t7)
    t8 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[7], pc_gather8))
    threads.append(t8)
    for t in threads:
        t.setDaemon(False)
        t.start()
    for t in threads:
        if t.isAlive():
            t.join()
    pc_gather = pc_gather1 + pc_gather2 + pc_gather3 + pc_gather4 + pc_gather5 + pc_gather6 + pc_gather7 + pc_gather8

    # pc_gather = []
    # for i in range(len(pc_bin)):
    #     pc_gather.append(calc_feature_single(pc_temp, pc_bin[i]))

    pc_fea = np.concatenate(pc_gather, axis=2)

    # print(pc_fea[:,3,:])

    return pc_fea

def gather_fea_hop_1(point_data, fea):
    """
    nn_idx:(B, n_sample, K)
    pts:(B, N, dim)
    :return: pc_n(B, K, dim_fea)
    """
    pts_fea_expand = np.concatenate([point_data, fea], axis=-1)

    pc_n = pts_fea_expand[..., :3]
    # print(pc_n.shape)
    pc_temp = pts_fea_expand[..., 3:]

    pc_n_center = np.expand_dims(pc_n[:, :, 0, :], axis=2)
    pc_n_uncentered = pc_n - pc_n_center

    pc_idx = []
    pc_idx.append(pc_n_uncentered[:, :, :, 0] >= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 0] <= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 1] >= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 1] <= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 2] >= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 2] <= 0)

    pc_bin = []
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[2] * pc_idx[4])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[2] * pc_idx[5])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[3] * pc_idx[4])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[3] * pc_idx[5])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[2] * pc_idx[4])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[2] * pc_idx[5])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[3] * pc_idx[4])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[3] * pc_idx[5])*1.0, axis=3))

    pc_gather1 = []
    pc_gather2 = []
    pc_gather3 = []
    pc_gather4 = []
    pc_gather5 = []
    pc_gather6 = []
    pc_gather7 = []
    pc_gather8 = []
    threads = []
    t1 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[0], pc_gather1))
    threads.append(t1)
    t2 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[1], pc_gather2))
    threads.append(t2)
    t3 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[2], pc_gather3))
    threads.append(t3)
    t4 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[3], pc_gather4))
    threads.append(t4)
    t5 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[4], pc_gather5))
    threads.append(t5)
    t6 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[5], pc_gather6))
    threads.append(t6)
    t7 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[6], pc_gather7))
    threads.append(t7)
    t8 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[7], pc_gather8))
    threads.append(t8)
    for t in threads:
        t.setDaemon(False)
        t.start()
    for t in threads:
        if t.isAlive():
            t.join()
    pc_gather = pc_gather1 + pc_gather2 + pc_gather3 + pc_gather4 + pc_gather5 + pc_gather6 + pc_gather7 + pc_gather8

    # pc_gather = []
    # for i in range(len(pc_bin)):
    #     pc_gather.append(calc_feature_single(pc_temp, pc_bin[i]))

    pc_fea = np.concatenate(pc_gather, axis=2)

    # print(pc_fea[:,1,:])

    return pc_fea


def gather_global_fea(feature, xyz, part=5):
    '''

    :param feature: (B, n_point, dim)
    :param xyz: (B, n_point, 3)
    :param part:int
    :return: (B, dim*part)
    '''

    pts_square = (xyz**2).sum(axis=2, keepdims=False)
    dis = np.sqrt(pts_square)  # (B, n_point)
    total_fea = []
    for i in range(part):
        idx = (dis >= (i/float(part))) * (dis <= ((i+1)/float(part)))*1.0
        part_fea = (feature*np.expand_dims(idx, axis=2)).max(axis=1, keepdims=False)
        total_fea.append(part_fea)
    return np.concatenate(total_fea, axis=1)