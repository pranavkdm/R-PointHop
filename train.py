import argparse
import pickle
import os
import modelnet40
import pointhop
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--initial_point', type=int, default=1024, 
                    help='Number of points to be used')
parser.add_argument('--model_dir', default='./model', 
                    help='Model directory [default: model]')
parser.add_argument('--num_point', default=[1024, 768, 512, 384], 
                    help='Point number after down sampling')
parser.add_argument('--num_sample', default=[64, 32, 48, 48], 
                    help='kNN query number')
parser.add_argument('--threshold', default=0.001, 
                    help='energy threshold for channel-wise Saab transform')
FLAGS = parser.parse_args()

initial_point = FLAGS.initial_point
num_point = FLAGS.num_point
num_sample = FLAGS.num_sample
threshold = FLAGS.threshold
MODEL_DIR = FLAGS.model_dir

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

def main():

    train_data, train_label = modelnet40.data_load(num_point=initial_point, 
                                data_dir=os.path.join(BASE_DIR, 
                                'modelnet40_ply_hdf5_2048'), train=True)

    model = pointhop.pointhop_train(True, train_data, n_newpoint=num_point,
                            n_sample=num_sample, threshold=threshold)

    with open(os.path.join(MODEL_DIR, 'R-PointHop.pkl'), 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    main()
