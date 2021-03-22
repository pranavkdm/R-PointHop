import numpy as np
from sklearn.decomposition import PCA
from numpy import linalg as LA
import point_utils
import threading
from multiprocessing import Process, Value, Array, Manager, Pool
import h5py
from os import getpid
import sklearn 
import time

def sample_knn(point_data, n_newpoint, n_sample):
    point_num = point_data.shape[1]
    if n_newpoint == point_num:
        new_xyz = point_data
    else:
        new_xyz = point_utils.furthest_point_sample(point_data, n_newpoint)
    idx = point_utils.knn(new_xyz, point_data, n_sample)
    return new_xyz, idx

def sample_knn_2(point_data, n_newpoint, n_sample, local_kernels, local_mean):
    point_num = point_data.shape[1]
    if n_newpoint == point_num:
        new_xyz = point_data
    else:
        new_xyz, local_kernels, local_mean = point_utils.furthest_point_sample_2(point_data, local_kernels, local_mean, n_newpoint)
    idx = point_utils.knn(new_xyz, point_data, n_sample)

    return new_xyz, local_kernels, local_mean, idx


def tree_multi(local_kernels, local_mean, Train, Bias, point_data, data, grouped_feature, idx, pre_energy, threshold, params, j, out):

    if grouped_feature is None:
        grouped_feature = data
    grouped_feature = point_utils.gather_fea(idx, point_data, grouped_feature, local_kernels, local_mean)
    s1 = grouped_feature.shape[0]
    s2 = grouped_feature.shape[1]
    grouped_feature = grouped_feature.reshape(s1 * s2, -1)

    if Train is True:
        kernels, mean, energy = find_kernels_pca(grouped_feature)
        bias = LA.norm(grouped_feature, axis=1)
        bias = np.max(bias)
        if pre_energy is not None:
            energy = energy * pre_energy
        num_node = np.sum(energy > threshold)
        params = {}
        params['bias'] = bias
        params['kernel'] = kernels
        params['pca_mean'] = mean
        params['energy'] = energy
        params['num_node'] = num_node
    else:
        kernels = params['kernel']
        mean = params['pca_mean']
        bias = params['bias']

    if Bias is True:
        grouped_feature = grouped_feature + bias

    transformed = np.matmul(grouped_feature, np.transpose(kernels))

    if Bias is True:
        e = np.zeros((1, kernels.shape[0]))
        e[0, 0] = 1
        transformed -= bias * e

    transformed = transformed.reshape(s1, s2, -1)

    output = []
    for i in range(transformed.shape[-1]):
        output.append(transformed[:, :, i].reshape(s1, s2, 1))
    out.append([[params], [output], [j], [getpid()]])


def remove_mean(features, axis):
    '''
    Remove the dataset mean.
    :param features [num_samples,...]
    :param axis the axis to compute mean
    
    '''
    feature_mean = np.mean(features, axis=axis, keepdims=True)
    feature_remove_mean = features-feature_mean
    return feature_remove_mean, feature_mean


def remove_zero_patch(samples):
    std_var = (np.std(samples, axis=1)).reshape(-1, 1)
    ind_bool = (std_var == 0)
    ind = np.where(ind_bool==True)[0]
    samples_new = np.delete(samples, ind, 0)
    return samples_new


def find_kernels_pca(sample_patches):
    '''
    Do the PCA based on the provided samples.
    If num_kernels is not set, will use energy_percent.
    If neither is set, will preserve all kernels.
    :param samples: [num_samples, feature_dimension]
    :param num_kernels: num kernels to be preserved
    :param energy_percent: the percent of energy to be preserved
    :return: kernels, sample_mean
    '''
    # Remove patch mean
    sample_patches_centered, dc = remove_mean(sample_patches, axis=1)
    sample_patches_centered = remove_zero_patch(sample_patches_centered)
    # Remove feature mean (Set E(X)=0 for each dimension)
    training_data, feature_expectation = remove_mean(sample_patches_centered, axis=0)

    pca = PCA(n_components=training_data.shape[1], svd_solver='full', whiten=True)
    pca.fit(training_data)

    num_channels = sample_patches.shape[-1]
    largest_ev = [np.var(dc*np.sqrt(num_channels))]
    dc_kernel = 1/np.sqrt(num_channels)*np.ones((1, num_channels))/np.sqrt(largest_ev)

    kernels = pca.components_[:, :]
    mean = pca.mean_
    kernels = np.concatenate((dc_kernel, kernels), axis=0)[:kernels.shape[0], :]

    energy = np.concatenate((largest_ev, pca.explained_variance_[:kernels.shape[0]-1]), axis=0) \
             / (np.sum(pca.explained_variance_[:kernels.shape[0]-1]) + largest_ev)
    return kernels, mean, energy

def tree(Train, Bias, point_data, data, grouped_feature, idx, pre_energy, threshold, params, size):
    
    if grouped_feature is None:
        grouped_feature = data

    local_data = np.zeros((data.shape[0],data.shape[1],size,3))
    local_kernels = np.zeros((data.shape[0],data.shape[1],3,3))
    local_mean = np.zeros((data.shape[0],data.shape[1],3,))
    pca = PCA(n_components=3)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            pca.fit(data[i,idx[i,:,j],:])
            kernels = pca.components_

            p = (data[i,idx[i,:,j],:]-data[i,idx[i,:,j],:][0])@kernels[0,:].T
            q = (data[i,idx[i,:,j],:]-data[i,idx[i,:,j],:][0])@kernels[1,:].T
            r = (data[i,idx[i,:,j],:]-data[i,idx[i,:,j],:][0])@kernels[2,:].T

            median = np.median(p)
            sort = np.argsort(p)
            left = np.sum(median-p[sort[:int(size/2)]])
            right = np.sum(p[sort[int(size/2):]]-median)
            if right >= left:
                kernels[0,:] = -kernels[0,:]

            median = np.median(q)
            sort = np.argsort(q)
            left = np.sum(median-q[sort[:int(size/2)]])
            right = np.sum(q[sort[int(size/2):]]-median)
            if right >= left:
                kernels[1,:] = -kernels[1,:]

            median = np.median(r)
            sort = np.argsort(r)
            left = np.sum(median-r[sort[:int(size/2)]])
            right = np.sum(r[sort[int(size/2):]]-median)
            if right >= left:
                kernels[2,:] = -kernels[2,:]

            local_data[i,j] = (data[i,idx[i,:,j],:]-data[i,idx[i,:,j],:][0])@kernels.T
            local_kernels[i,j] = kernels
            local_mean[i,j] = data[i,idx[i,:,j],:][0]

    grouped_feature = point_utils.gather_fea_hop_1(local_data, local_data)

    s1 = grouped_feature.shape[0]
    s2 = grouped_feature.shape[1]
    grouped_feature = grouped_feature.reshape(s1 * s2, -1)
    
    if Train is True:
        kernels, mean, energy = find_kernels_pca(grouped_feature)
        print(energy)
        bias = LA.norm(grouped_feature, axis=1)
        bias = np.max(bias)
        if pre_energy is not None:
            energy = energy * pre_energy
        num_node = np.sum(energy > threshold)
        params = {}
        params['bias'] = bias
        params['kernel'] = kernels
        params['pca_mean'] = mean
        params['energy'] = energy
        params['num_node'] = num_node
    else:
        kernels = params['kernel']
        mean = params['pca_mean']
        bias = params['bias']

    if Bias is True:
        grouped_feature = grouped_feature + bias

    transformed = np.matmul(grouped_feature, np.transpose(kernels))

    if Bias is True:
        e = np.zeros((1, kernels.shape[0]))
        e[0, 0] = 1
        transformed -= bias * e

    transformed = transformed.reshape(s1, s2, -1)
    output = []
    for i in range(transformed.shape[-1]):
        output.append(transformed[:, :, i].reshape(s1, s2, 1))

    return params, output, local_kernels, local_mean

def mySort(out):
    r = []
    idx = {}
    ppid = {}
    if len(out) == 0:
        return {}, out
    tt = np.zeros((len(out)))
    for i in range(len(out)):
        ppid[out[i][2][0]] = i
        tt[i] = out[i][2][0]
    t = np.min(tt)
    for i in range(len(out)):
        tmp = out[i]#.get()
        r.append(tmp)
        idx[i] = ppid[t]
        t+=1
    return idx, r



def pointhop_train(Train, data, n_newpoint, n_sample, threshold):
    '''
    Train based on the provided samples.
    :param train_data: [num_samples, num_point, feature_dimension]
    :param n_newpoint: point numbers used in every stage
    :param n_sample: k nearest neighbors
    :param layer_num: num kernels to be preserved
    :param energy_percent: the percent of energy to be preserved
    :return: idx, new_idx, final stage feature, feature, pca_params
    '''
    manager=Manager()
    point_data = data
    Bias = [False, True, True, True]
    info = {}
    pca_params = {}
    leaf_node = []
    leaf_node_energy = []

    for i in range(len(n_newpoint)):
        to=time.time()
        print("------",i)
        if i == 0:
            new_xyz, idx = sample_knn(point_data, n_newpoint[i], n_sample[i])
        else:
            new_xyz, local_kernels, local_mean, idx = sample_knn_2(point_data, n_newpoint[i], n_sample[i], local_kernels, local_mean)
            print(local_kernels.shape)
        print("------done  ",time.time()-to)
        if i == 0:
            print(i)
            pre_energy = 1
            params, output, local_kernels, local_mean = tree(Train, Bias[i], point_data, data, None, idx, pre_energy, threshold, None, n_sample[0])
            pca_params['Layer_{:d}_pca_params'.format(i)] = params
            num_node = params['num_node']
            energy = params['energy']
            info['Layer_{:d}_feature'.format(i)] = output[:num_node]
            info['Layer_{:d}_energy'.format(i)] = energy
            info['Layer_{:d}_num_node'.format(i)] = num_node
            # if num_node != len(output):
            #     for m in range(num_node, len(output), 1):
            #         leaf_node.append(output[m])
            #         leaf_node_energy.append(energy[m])

        elif i == 1:
            output = info['Layer_{:d}_feature'.format(i - 1)]
            pre_energy = info['Layer_{:d}_energy'.format(i - 1)]
            num_node = info['Layer_{:d}_num_node'.format(i - 1)]
            s1 = 0
            out = []#manager.list([])
            threads = []
            for j in range(num_node):
                t = threading.Thread(target=tree_multi, args=(local_kernels, local_mean, Train, Bias[i], point_data, data, output[j], idx,
                                                                   pre_energy[j], threshold, None, j, out))
                threads.append(t)
                t.start()
            #for t in threads:
                t.join()
           # print(out)
            idxt, out = mySort(out)
            print(idxt, len(out))
            for jj in range(num_node):
                #print(i, j)
                j = idxt[jj]
                params = out[j][0][0]
                output_t = out[j][1][0]
                print("l1-----",jj,j, out[j][2][0])
                # params, output_t = tree(Train, Bias[i], point_data, data, output[j], idx, pre_energy[j], threshold, None)
                pca_params['Layer_{:d}_{:d}_pca_params'.format(i, j)] = params
                num_node_t = params['num_node']
                energy = params['energy']
                info['Layer_{:d}_{:d}_feature'.format(i, j)] = output_t[:num_node_t]
                info['Layer_{:d}_{:d}_energy'.format(i, j)] = energy
                info['Layer_{:d}_{:d}_num_node'.format(i, j)] = num_node_t
                s1 = s1 + num_node_t
                # if num_node_t != len(output_t):
                #     for m in range(num_node_t, len(output_t), 1):
                #         leaf_node.append(output_t[m])
                #         leaf_node_energy.append(energy[m])

        elif i == 2:
            num_node = info['Layer_{:d}_num_node'.format(i - 2)]
            for j in range(num_node):
                output = info['Layer_{:d}_{:d}_feature'.format(i - 1, j)]
                pre_energy = info['Layer_{:d}_{:d}_energy'.format(i - 1, j)]
                num_node_t = info['Layer_{:d}_{:d}_num_node'.format(i - 1, j)]

                out = []#manager.list([])
                threads = []
                for k in range(num_node_t):
                    t = threading.Thread(target=tree_multi, args=(local_kernels, local_mean, Train, Bias[i], point_data, data, output[k], idx,
                                                              pre_energy[k], threshold, None, k, out))
                    threads.append(t)
                    t.start()
                #for t in threads:
                    t.join()

                idxt, out = mySort(out)

                print(idxt, len(out))
                for kk in range(num_node_t):
                    k = idxt[kk]
                    #print(i, j, k)
                    params = out[k][0][0]
                    output_t = out[k][1][0]
                    print("l2-----", kk,k,out[k][2][0])
                    # params, output_t = tree(Train, Bias[i], point_data, data, output[k], idx, pre_energy[k], threshold, None)
                    pca_params['Layer_{:d}_{:d}_{:d}_pca_params'.format(i, j, k)] = params
                    num_node_tt = params['num_node']
                    energy = params['energy']
                    info['Layer_{:d}_{:d}_{:d}_feature'.format(i, j, k)] = output_t[:num_node_tt]
                    info['Layer_{:d}_{:d}_{:d}_energy'.format(i, j, k)] = energy
                    info['Layer_{:d}_{:d}_{:d}_num_node'.format(i, j, k)] = num_node_tt
                    # if num_node_tt != len(output_t):
                    #     for m in range(num_node_tt, len(output_t), 1):
                    #         leaf_node.append(output_t[m])
                    #         leaf_node_energy.append(energy[m])

        elif i == 3:
            num_node = info['Layer_{:d}_num_node'.format(i - 3)]
            for j in range(num_node):
                num_node_t = info['Layer_{:d}_{:d}_num_node'.format(i - 2, j)]
                for k in range(num_node_t):
                    output = info['Layer_{:d}_{:d}_{:d}_feature'.format(i - 1, j, k)]
                    pre_energy = info['Layer_{:d}_{:d}_{:d}_energy'.format(i - 1, j, k)]
                    num_node_tt = info['Layer_{:d}_{:d}_{:d}_num_node'.format(i - 1, j, k)]
                    out = []#manager.list([])
                    threads = []
                    for t in range(num_node_tt):
                        t = threading.Thread(target=tree_multi, args=(local_kernels, local_mean, Train, Bias[i], point_data, data, output[t], idx,pre_energy[t], threshold, None, t, out))
                        threads.append(t)
                        t.start()
                    #for t in threads:
                        t.join()

                    idxt, out = mySort(out)
                    print(idxt, len(out))
                    for tt in range(num_node_tt):
                        t = idxt[tt]
                        #print(i, j, k, t)
                        params = out[t][0][0]
                        output_t = out[t][1][0]
                        print("l3-----", tt,t,out[t][2][0])

                        # params, output_t = tree(Train, Bias[i], point_data, data, output[t], idx, pre_energy[t],
                        #                       threshold, None)
                        pca_params['Layer_{:d}_{:d}_{:d}_{:d}_pca_params'.format(i, j, k, t)] = params
                        num_node_ttt = params['num_node']
                        energy = params['energy']
                        info['Layer_{:d}_{:d}_{:d}_{:d}_feature'.format(i, j, k, t)] = output_t[:num_node_ttt]
                        info['Layer_{:d}_{:d}_{:d}_{:d}_energy'.format(i, j, k, t)] = energy
                        info['Layer_{:d}_{:d}_{:d}_{:d}_num_node'.format(i, j, k, t)] = num_node_ttt
                        for m in range(len(output_t)):
                            leaf_node.append(output_t[m])
                            leaf_node_energy.append(energy[m])
        point_data = new_xyz
            
    return pca_params


def pointhop_pred(Train, data, pca_params, n_newpoint, n_sample):
    '''
    Test based on the provided samples.
    :param test_data: [num_samples, num_point, feature_dimension]
    :param pca_params: pca kernel and mean
    :param n_newpoint: point numbers used in every stage
    :param n_sample: k nearest neighbors
    :param layer_num: num kernels to be preserved
    :param idx_save: knn index
    :param new_xyz_save: down sample index
    :return: final stage feature, feature, pca_params
    '''
    manager=Manager()
    point_data = data
    Bias = [False, True, True, True]
    info_test = {}
    leaf_node = []

    for i in range(len(n_newpoint)):
        # print("------",i)
        if i == 0:
            new_xyz, idx = sample_knn(point_data, n_newpoint[i], n_sample[i])
        else:
            new_xyz, local_kernels, local_mean, idx = sample_knn_2(point_data, n_newpoint[i], n_sample[i], local_kernels, local_mean)
        # print("-----done")
        if i == 0:
            #print(i)
            params = pca_params['Layer_{:d}_pca_params'.format(i)]
            num_node = params['num_node']
            params_t, output, local_kernels, local_mean = tree(Train, Bias[i], point_data, data, None, idx, None, None, params, n_sample[0])
            info_test['Layer_{:d}_feature'.format(i)] = output[:num_node]
            info_test['Layer_{:d}_num_node'.format(i)] = num_node
            # if num_node != len(output):
            #     for m in range(num_node, len(output), 1):
            #         leaf_node.append(output[m])

        elif i == 1:
            output = info_test['Layer_{:d}_feature'.format(i - 1)]
            num_node = info_test['Layer_{:d}_num_node'.format(i - 1)]
            out = []#manager.list([])
            threads = []
            for j in range(num_node):
                t = threading.Thread(target=tree_multi, args=(local_kernels, local_mean, Train, Bias[i], point_data, data, output[j], idx,None, None, pca_params['Layer_{:d}_{:d}_pca_params'.format(i, j)], j, out))
                threads.append(t)
                t.start()
            #for t in threads:
                t.join()

            idxt, out = mySort(out)
            # print(idxt, len(out))
            for jj in range(num_node):
                j = idxt[jj]
                #print(i, j)
                # params = out[j][0][0]
                output_t = out[j][1][0]
                # print("l1-----",jj,j,out[j][2][0])

                params = pca_params['Layer_{:d}_{:d}_pca_params'.format(i, j)]
                num_node_t = params['num_node']

                # params, output_t = tree(Train, Bias[i], point_data, data, output[j], idx, None, None, params)
                info_test['Layer_{:d}_{:d}_feature'.format(i, j)] = output_t[:num_node_t]
                info_test['Layer_{:d}_{:d}_num_node'.format(i, j)] = num_node_t
                # if num_node_t != len(output_t):
                #     for m in range(num_node_t, len(output_t), 1):
                #         leaf_node.append(output_t[m])

        elif i == 2:
            num_node = info_test['Layer_{:d}_num_node'.format(i - 2)]
            for j in range(num_node):
                output = info_test['Layer_{:d}_{:d}_feature'.format(i - 1, j)]
                num_node_t = info_test['Layer_{:d}_{:d}_num_node'.format(i - 1, j)]

                out = []#manager.list([])
                threads = []
                for k in range(num_node_t):
                    t = threading.Thread(target=tree_multi, args=(local_kernels, local_mean, Train, Bias[i], point_data, data, output[k], idx,
                                                              None, None, pca_params['Layer_{:d}_{:d}_{:d}_pca_params'.format(i, j, k)], k, out))
                    threads.append(t)
                    t.start()
                #for t in threads:
                    t.join()

                idxt, out = mySort(out)
                # print(idxt, len(out))
                for kk in range(num_node_t):
                    k = idxt[kk]
                    #print(i, j, k)
                    params = pca_params['Layer_{:d}_{:d}_{:d}_pca_params'.format(i, j, k)]
                    num_node_tt = params['num_node']
                    output_t = out[k][1][0]
                    # print("l2-----",kk,k,out[k][2][0])

                    # params, output_t = tree(Train, Bias[i], point_data, data, output[k], idx, None, None, params)
                    info_test['Layer_{:d}_{:d}_{:d}_feature'.format(i, j, k)] = output_t[:num_node_tt]
                    info_test['Layer_{:d}_{:d}_{:d}_num_node'.format(i, j, k)] = num_node_tt
                    # if num_node_tt != len(output_t):
                    #     for m in range(num_node_tt, len(output_t), 1):
                    #         leaf_node.append(output_t[m])

        elif i == 3:
            num_node = info_test['Layer_{:d}_num_node'.format(i - 3)]
            for j in range(num_node):
                num_node_t = info_test['Layer_{:d}_{:d}_num_node'.format(i - 2, j)]
                for k in range(num_node_t):
                    output = info_test['Layer_{:d}_{:d}_{:d}_feature'.format(i - 1, j, k)]
                    num_node_tt = info_test['Layer_{:d}_{:d}_{:d}_num_node'.format(i - 1, j, k)]

                    out = []#manager.list([])
                    threads = []
                    for t in range(num_node_tt):
                        t =threading.Thread(target=tree_multi, args=(local_kernels, local_mean, Train, Bias[i], point_data, data, output[t], idx,
                                                                  None, None, pca_params['Layer_{:d}_{:d}_{:d}_{:d}_pca_params'.format(i, j, k, t)], t, out))
                        threads.append(t)
        
                        t.start()
                    # for t in threads:
                        t.join()

                    idxt, out = mySort(out)
                    # print(idxt, len(out))
                    for tt in range(num_node_tt):
                        t = idxt[tt]
                        #print(i, j, k, t)
                        params = pca_params['Layer_{:d}_{:d}_{:d}_{:d}_pca_params'.format(i, j, k, t)]
                        num_node_ttt = params['num_node']
                        output_t = out[t][1][0]
                        # print("l3-----",tt,t,out[t][2][0])

                        # params, output_t = tree(Train, Bias[i], point_data, data, output[t], idx, None, None, params)

                        info_test['Layer_{:d}_{:d}_{:d}_{:d}_feature'.format(i, j, k, t)] = output_t[:num_node_ttt]
                        info_test['Layer_{:d}_{:d}_{:d}_{:d}_num_node'.format(i, j, k, t)] = num_node_ttt
                        for m in range(len(output_t)):
                            leaf_node.append(output_t[m])
        point_data = new_xyz

    return leaf_node, new_xyz
