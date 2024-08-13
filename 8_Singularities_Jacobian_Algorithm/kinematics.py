import numpy as np                            # numpy is for math (it's hyper fast)
from scipy.spatial.transform import Rotation  # we are going to use this for deriving euler angles from rot mat
from laura_interface import *                





def get_dh_param_table(q, ee_module_length, mode=MODE_2D):
    '''
    Returns a table containing the Denavit-Hartenberg parameters,
    with the row i having the following pattern: [theta_i, alpha_i, r_i, d_i].
    In contrast to the lecture, i starts at zero (because informatics..)
    The values of this table depend on the robot's current configuration q.

    (!) This function incorporates the length of the end effector module.

    :param q:                       the robot's current configuration (in our case, a list a joint angles in rad)
    :param ee_module_length:        the length of the attached end effector module
    :return:                        (n x 4) np.ndarray - a table with Denavit Hartenberg parameters
    '''

    # lengths of laura's links [millimeters]
    LINK_LENGTH_0       = 60.5 if mode == MODE_2D else 89.5
    LINK_LENGTH_1       = 72.5
    LINK_LENGTH_2       = 72.5
    LINK_LENGTH_3       = 21.0 + ee_module_length

    return np.array([[q[0] + np.pi / 2,  -np.pi / 2,              0,  LINK_LENGTH_0],
                     [q[1] - np.pi / 2,           0,  LINK_LENGTH_1,              0],
                     [            q[2],           0,  LINK_LENGTH_2,              0],
                     [            q[3],   np.pi / 2,  LINK_LENGTH_3,              0]])
# ---



def get_dh_matrix(theta:float, alpha:float, r:float, d:float) -> np.ndarray:
    '''
    Returns a (4x4) Denavit-Hartenbert transformation matrix based
    on the DH parameters (classic DH convention, not Craig).
    '''
    return np.array([[np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), r * np.cos(theta)],
                     [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), r * np.sin(theta)],
                     [            0,                  np.sin(alpha),                  np.cos(alpha),                 d], 
                     [            0,                              0,                              0,                 1]])
# ---



def get_dh_matrices(dh_parameter_table):
    '''
    Returns a list of Denavit-Hartenberg matrices. The i-th matrix realizes a transformation between frame i and (i+1), 
    with zero being the robot's first joint frame (because of informatics, i starts at zero, not one).
    These matrices are derived from a table containing the Denavit-Hartenberg parameters.

    :param dh_parameter_table:    (n x 4) np.ndarray - table with Denavit Hartenberg parameters
    :return:                      List of n homogeneous transformations, defining the robot's frames relative to their respective predecessor frame
    '''
    dh_matrices = []
    
    for row in dh_parameter_table:
        dh_matrices.append(get_dh_matrix(*row))

    return dh_matrices
# ---



def map_frames_to_base_frame(dh_matrices):
    '''
    Returns a list of homogeneous transformation matrices 
    that represent the robot's joint frames and its end effector frame 
    relative to its first joint frame.

    Remember: The DH parameters and matrices encode the transitions(!) between frames.
    Therefore, they do not represent the frames themselves. Thus, the base frame is not part of the DH parameter table!

    :param dh_matrices:   list of n homogeneous transformations, encoding transitions between DH frames.
    :return:              list of (n + 1) homogeneous transformations, encoding the robot's joint frames and the end effector frame, relative to the base frame
    '''
    
    tmp_transform = np.eye(4)
    transforms    = [tmp_transform]

    # Generate transformations from base frame to i-th robot frame
    for mat in dh_matrices:
        tmp_transform = tmp_transform @ mat
        transforms.append(tmp_transform)
    
    # The list transforms includes the robot's frames relative to the base frame: 
    # Thus, this list looks like this: [T_0_0, T_0_1, T_0_2, ..., T_0_ee]
    return transforms    
# ---



def get_ee_frame_for_dh_param_table(dh_parameter_table):
    '''
    Returns the robot's end effector frame relative to the base frame, based on a given Denavit-Hartenberg parameter table. 
    This table depends on the robot's current configuration q = [theta_1, ..., theta_n].
    
    :param dh_parameter_table:    (n x 4) np.ndarray - table with Denavit Hartenberg parameters
    :return:                      homogeneous transformations, encoding the robot's end effector frame, relative to the base frame
    '''    

    dh_matrices    = get_dh_matrices(dh_parameter_table)    
    frames         = map_frames_to_base_frame(dh_matrices)

    return frames[-1]
# ---


def get_6D_vec_from_ht(T):
    '''
    Extracts the pose encoded in a homogeneous transformation and returns it as a 6D vector.

    :param T:      (4x4) homogeneous transformation matrix
    :return:       6D pose [x, y, z, roll, pitch, yaw]
    '''
    x, y, z          = T[:3, 3]
    roll, pitch, yaw = Rotation.from_matrix(T[:3, :3]).as_euler('XYZ', degrees=False)    
    return np.array([x, y, z, roll, pitch, yaw])
# ----


def extract_planar_pose_from_frame(T):
    '''
    Extracts the robot's end effector pose for 2D Mode from a homogeneous transformation, encoding the end effector frame.
    In this mode, the end effector moves within a plane. Its pose therefore has three dimensions (x, y, theta).

    :param T:       end effector frame encoeded as a homogeneous transformation relative to the base frame 
    :return:        3D vector (x, y, z), representing the planar end effector pose relative to the base frame in 2D Mode
    '''
    pose_6d = get_6D_vec_from_ht(T)
    return np.array([pose_6d[0], pose_6d[2], pose_6d[4]])
# ----


def get_ee_pose(q, ee_module_length, mode=MODE_2D, as_vector=True):
    '''
    Returns the robot's end effector pose.

    :param dh_param_table:       Denavit-Hartenberg parameter table. This table depends on the current robot configuration q = [theta_1, ..., theta_n].
    
    :return:                     In MODE_2D: A 3D Vector (x, y, theta), encoding the position and orientation of the end effector on the plane
                                 In MODE_3D: A homogeneous transformation, representing the robot's end effector frame, relative to the base frame
    '''
    
    # compose Denavit-Hartenberg parameter table, based on the robot's current configuration q
    dh_param_table = get_dh_param_table(q, ee_module_length, mode)

    # compute the robot's end effector frame relative to the base frame
    T_0_ee = get_ee_frame_for_dh_param_table(dh_param_table)
    
    if mode == MODE_2D:
        return extract_planar_pose_from_frame(T_0_ee)
    else:    
        if as_vector:
            return get_6D_vec_from_ht(T_0_ee)
        else:
            return T_0_ee
# ---



def get_jacobian(q:np.ndarray, mode:int=MODE_2D, ee_module_length=0) -> np.ndarray:
    '''
    Computes the Jacobian matrix for LAURA in 2D mode. Thus, the end effector pose has three dimensions [x, y, theta].

    :param q:       configuration vector
    :param mode     LAURA's operating mode (either MODE_2D or MODE_3D)
    :return:        Jacobian matrix whose entries J_ij determine the partial derivatie 
                    of the i-th end effector pose dimension with respect to the j-th joint angle
                    In 2D mode, the jacobian matrix has shape (3x4) and in 3D mode (6x4).
    '''

    # create the DH parameter table for the current configuration
    param_table = get_dh_param_table(q, ee_module_length)

    # create a list of DH matrices, one for each row of the parameter table.
    # each such matrix encodes a DH frame that is represented relative to its predecessor frame.
    dh_matrices = get_dh_matrices(param_table)

    # map the DH frames to the base frame.
    # The list T_0_ is of the following shape: [T_0_0, T_0_1, ..., T_0_n, T_0_ee], for i in [0, .., n],
    # where n is the number of jonts. Here, the transformation T_0_0 denotes the base frame relative 
    # to the base frame (i.e, the identity matrix), the transformation T_0_n denotes the last joint frame relative 
    # to the base frame, and T_0_ee denotes the end effector frame relative to the bae frame.
    T_0_        = map_frames_to_base_frame(dh_matrices) 
    n_frames    = len(T_0_)
    n_joints    = n_frames - 1 

    J     = np.zeros((6, n_joints))                      # initialize jacobian with zeros
    d_0_n = T_0_[-1][:3, 3]                              # translation to end effector wrt. frame 0
    
    for i in range(1, n_frames):                         # go through columns of Jacobian (i starts at 1)
        z_0_im1    = T_0_[i-1][:3, 2]                    # extract z-axis from T_0_(i-1)
        d_0_im1    = T_0_[i-1][:3, 3]                    # extract translation vec from T_0_(i-1)

        J[:3, i-1] = np.cross(z_0_im1, d_0_n - d_0_im1)  # fill translation part of Jacobian for joint i
        J[3:, i-1] = z_0_im1                             # fill rotation part of Jacobian for joint i
        


    if mode == MODE_3D:
        return J
    
    else:

        # extract relevant rows from the (6x4) 3D Jacobian to obtain the (3x4) 2D Jacobian
        return np.array([J[0, :],   # partial     x / partial q_[0:n]
                         J[2, :],   # partial     z / partial q_[0:n] (robot's z in 3D is it's y in 2D)
                         J[4, :]])  # partial pitch / partial q_[0:n]
# ---