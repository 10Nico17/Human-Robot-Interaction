import numpy as np
from scipy.spatial.transform import Rotation  # we are going to use this for deriving euler angles from rot mat


def get_dh_matrix(theta, alpha, r, d):
    '''
    Returns a (4x4) Denavit-Hartenbert transformation matrix based
    on the DH parameters (classic DH convention, not Craig).
    '''
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), r * np.cos(theta)],
        [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), r * np.sin(theta)],
        [            0,                  np.sin(alpha),                  np.cos(alpha),                 d], 
        [            0,                              0,                              0,                 1]
    ])
# ---


def turn_dh_params_into_dh_matrices(dh_param_table):
    '''
    Returns a list of Denavit-Hartenberg matrices
    (i.e. the elements of this list are 2D numpy arrays).
    The i-th matrix realizes a transformation between 
    frame i and (i+1) - because i stars at zero, not at one.
    These matrices are derived from a table containing
    the Denavit-Hartenberg parameters.
    '''
    dh_matrices = []
    
    for row in dh_param_table:
        dh_matrices.append(get_dh_matrix(*row))
    
    return dh_matrices
# ---


def get_base_to_frame_transforms(dh_matrices):
    '''
    Returns a list of homogeneous transformations form base frame (first frame with index 0) 
    to frame with index i. Thus, the first entry of this list is transformation from 
    base frame to base frame which is identity matrix, and the last entry of this list is 
    the transformation from base frame to end effector frame.
    '''
    tmp_transform = np.eye(4)
    transforms    = [tmp_transform]

    for mat in dh_matrices:
        tmp_transform = tmp_transform.dot(mat)
        transforms.append(tmp_transform)
    
    return transforms
# ---


def get_frames(dh_param_table):
    '''
    Returns a list of all frames (as specified in the DH table) that are defined relative to the base frame.
    '''
    # The matrix FRAME has been defined for your convinience.
    # It represents the position and orientation of the 
    # end effector relative to the end effector frame.
    # The first column vector of FRAME contains the origin of the end 
    # effector (hence [0,0,0]). The following column vectors contain 
    # the x, y, and z base vectors of the end effector frame 
    # relative to the end effector frame (hence [1,0,0], [0,1,0], and [0,0,1]). 
    # All column vectors have been appended with a one 
    # so that we can use them with homogeneous transformations.
    # We will now transform the pose of the end effector (represented by FRAME)
    # from the end effector frame to the base frame.

    FRAME = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 1, 1],
    ])

    # second, derive the DH matrices from the DH parameter table
    dh_matrices    = turn_dh_params_into_dh_matrices(dh_param_table)    
    
    # get transformatios from base frame to each frame
    transforms     = get_base_to_frame_transforms(dh_matrices)
    
    # compute frames relative to base frame
    frames         = [trans.dot(FRAME) for trans in transforms]

    return frames
# ---


def extract_R(homogeneous_transform):
    '''
    Returns the (3x3) rotation matrix extracted from a (4x4) homogeneous transformation matrix.
    '''
    return homogeneous_transform[:3,:3]
# ---


def extract_d(homogeneous_transform):
    '''
    Returns the (3x1) translation vector extracted from a (4x4) homogeneous transformation matrix.
    '''
    return homogeneous_transform[:3, 3].reshape(3,1)
# ---


def get_jacobian(transforms):
    '''
    Returns the Jacobian matrix based on a given list of homogeneous transformations 
    from base frame (first frame with index 0) to frame with index i. 
    Thus, the first entry of this list is transformation from base frame to base 
    frame which is identity matrix, and the last entry of this list is the 
    transformation from base frame to end effector frame.
    '''
    n_ee_dim = 6
    n_joints = len(transforms) - 1

    jacobian = np.zeros((n_ee_dim, n_joints))
    
    for j in range(n_joints):    
        translation_part = extract_R(transforms[j]).dot(np.array([[0, 0, 1]]).T)
        translation_part = np.cross(translation_part.T, (extract_d(transforms[-1]) - extract_d(transforms[j])).T)        
        rotation_part    = extract_R(transforms[j]).dot(np.array([[0, 0, 1]]).T).flatten()
        
        jacobian[:3, j]  = translation_part
        jacobian[3:, j]  = rotation_part
    
    return jacobian
# ---


def get_J(dh_params):
    return get_jacobian(get_base_to_frame_transforms(turn_dh_params_into_dh_matrices(dh_params)))


def get_rot_from_frame(frame_ee):
    '''
    Returns the orientation of the end effector as a rotation matrix relative to the base frame.
    
    frame_ee is a matrix whose first column vector contans the origin 
    of the end effector's reference frame relative to the base frame. 
    The following three column vectors contain the x, y, and z axes of the end effector frame 
    relative to the base frame. 
    
    You do not need to do anything here.
    '''
    
    # Extract origin and basis vectors from end effector frame.
    # These vectors are defined in the base frame.
    ee_origin = frame_ee[:3, 0]
    ee_x_axis = frame_ee[:3, 1] - ee_origin
    ee_y_axis = frame_ee[:3, 2] - ee_origin
    ee_z_axis = frame_ee[:3, 3] - ee_origin

    # The rotation matrix is R the end effector frame's basis vectors defined
    # in the base frame. But the origins of the ee frame needs to be 
    # the origin of the base frame. That's why we subtracted the origin above.
    R = np.zeros((3,3))
    R[:,0] = ee_x_axis
    R[:,1] = ee_y_axis
    R[:,2] = ee_z_axis

    return R
# ---


def forward_kinematics(dh_param_table):
    '''
    Returns the position and orientation of the end effector in base frame.
    Orientation is represented in XYZ Euler angles.
    '''
    frames   = get_frames(dh_param_table)
    frame_ee = frames[-1]

    # get position of end effector from its frame
    x, y, z  = frame_ee[:3, 0]
    
    # get orientatino of end effector as Euler angles
    R                = Rotation.from_matrix(get_rot_from_frame(frame_ee))
    roll, pitch, yaw =  R.as_euler('xyz')
    
    return np.array([x, y, z, roll, pitch, yaw])
# ---


def get_step(target_pose, dh_param_table):
    '''
    Returns a step in configuration space that realizes a movement of the end effector towards 
    the desired end effector pose in operational space.
    '''
    
    # Compute the step operational space (end effector space)
    # that minimizes the difference between the current end effector pose
    # and the target pose
    delta_x        = target_pose - forward_kinematics(dh_param_table)

    # Derive the DH matrices from the DH parameter table
    dh_matrices    = turn_dh_params_into_dh_matrices(dh_param_table)    
    
    # Derive list of homogeneous transformations from base frame (which has index 0) 
    # to frame i, based on the DH matrices
    transforms     = get_base_to_frame_transforms(dh_matrices)
    
    # Compute the Jacobian matrix based on the homogeneous transformations
    # from base frame to frame i
    J              = get_jacobian(transforms)

    # Compute the pseudoinverse of the Jacobian matrix
    J_inv          = np.linalg.pinv(J)

    # Compute the step in configuration space delta_q
    # that realizes the desired step in end effector space
    return J_inv.dot(delta_x)
# ---

