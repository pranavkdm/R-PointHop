import numpy as np
from scipy.spatial.transform import Rotation

def add_gaussian_noise(pc, sigma=0.01, clip=0.05):
    """
    Add gaussian noise of given mean and standard deviation to point cloud
    :param pc: B X N X 3 array, original batch of point clouds
    :param sigma: mean of noise
    :param clip: standard deviation of noise
    :return: point cloud with noise
    """
    jittered_data = np.clip(sigma * np.random.randn(*pc.shape), -1 * clip, clip)
    jittered_data += pc

    return jittered_data

def matrix2euler(matrix, degree, seq='zyx'):
    """
    Convert rotation matrix to corresponding euler angles
    :param matrix: 3x3 rotation matrix
    :param degree: degree or radians
    :param seq: order of angles
    :return: three euler angles
    """
    r = Rotation.from_matrix(matrix)
    angles = r.as_euler(seq, degrees=degree)

    return angles

def apply_transformation(data, angle_x, angle_y, angle_z, translation):
    """
    Apply spatial transformation to point cloud
    :param data: input point cloud 
    :param angle_x: rotation along X axis
    :param angle_y: rotation along Y axis
    :param angle_z: rotation along Z axis
    :param translation: vector of translation along XYZ
    :return: transformed point cloud
    """
    rotation = Rotation.from_euler('zyx', [angle_z, angle_y, angle_x])
    data_rotated = rotation.apply(data, inverse=False) + translation.T
    data_transformed = data_rotated + translation.T

    return data_transformed

def apply_inverse_transformation(data, angle_x, angle_y, angle_z, translation):
    """
    Apply inverse spatial transformation to point cloud
    :param data: input point cloud 
    :param angle_x: rotation along X axis
    :param angle_y: rotation along Y axis
    :param angle_z: rotation along Z axis
    :param translation: vector of translation along XYZ
    :return: transformed point cloud
    """
    rotation = Rotation.from_euler('zyx', [angle_z, angle_y, angle_x])
    data_inverse_translated = data - translation.T
    data_inverse_transformed = rotation.apply(data_inverse_translated, inverse=True)

    return data_inverse_transformed