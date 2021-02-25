import os
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

def load_dir(data_dir, name='train_files.txt'):
    with open(os.path.join(data_dir,name),'r') as f:
        lines = f.readlines()
    return [os.path.join(data_dir, line.rstrip().split('/')[-1]) for line in lines]

def data_load(num_point=None, data_dir=None, train=True):
    if not os.path.exists(data_dir):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('rm %s' % (zipfile))

    if train:
        data_pth = load_dir(data_dir, name='train_files.txt')
    else:
        data_pth = load_dir(data_dir, name='test_files.txt')

    point_list = []
    label_list = []
    for pth in data_pth:
        data_file = h5py.File(pth, 'r')
        point = data_file['data'][:]
        label = data_file['label'][:]
        point_list.append(point)
        label_list.append(label)
    data = np.concatenate(point_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    if not num_point:
        return data[:, :, :], label
    else:
        return data[:, :num_point, :], label
