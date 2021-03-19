# R-PointHop: A Green, Accurate and Unsupervised Point Cloud Registration Method

R-PointHop is an unsupervised learning based method for registration of 3D point cloud objcets. This is an official implementation of R-PointHop by [Pranav Kadam](https://github.com/pranavkdm), [Min Zhang](https://github.com/minzhang-1), Shan Liu and C.-C. Jay Kuo. This work was carried out at Media Communications Lab (MCL), University of Southern California, USA.

[[arXiv](https://arxiv.org/abs/2103.08129)]

## Introduction

R-PointHop is an unsupervised learning method for registration of two point cloud objects. It derives point features from training data statistics in a hierarchical feedforward manner without end-to-end optimization. The features are used to find point correspondences which in turn lead to estimating the 3D transformation. More technical details can be found in our paper. 

In this repository, we release the code for training R-PointHop method and evaluating on a given pair of point cloud objects.

## Packages

The code has been developed and tested in Python 3.6. The following packages need to be installed.

```
h5py
numpy
scipy
sklearn
open3d
```

## Training

Train the model on all 40 classes of ModelNet40 dataset

```
python train.py --first_20 False
```

Train the model on first 20 classes of ModelNet40 dataset

```
python train.py --first_20 True
```

User can specify other parameters like number of points in each hop, neighborhood size and energy threshold, else default parameters will be used.

## Registration 

```
python test.py --source ./data/source_0.ply --target ./data/target_0.ply
```

A set of sample source and target point cloud objects are present in the [data](https://github.com/pranavkdm/R-PointHop/tree/main/data) folder which can be used for testing. Replace source_0 and target_0 with your choice of souce and target.

## Citation

If you find our work useful in your research, please consider citing:

```
@article{kadam2021rpointhop,
      title={R-PointHop: A Green, Accurate and Unsupervised Point Cloud Registration Method}, 
      author={Pranav Kadam and Min Zhang and Shan Liu and C.-C. Jay Kuo},
      year={2021},
      eprint={2103.08129},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
