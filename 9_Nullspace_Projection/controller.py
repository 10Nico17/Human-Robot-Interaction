import numpy as np
from laura_interface import *
import dynamics as dmx
import kinematics as ktx


'''
ATTENTION!
----------

READ ME!
READ ME!
READ ME!

When we specify the position and orientation of LAURA's end effector,
there is only a single (unique) configuraiton that achieves this! 
Thus, the nullspace is empty!

Therefore, we ignore the orientation of LAURA's end effector and only consider its position.
Now we have a robot with 4 joints controlling 3 end effector dimensions.
Thus, there are multiple configurations that realize a desired end effector position.
Hence, the nullspace is not empty!
'''



def compute_delta_q(x_des:np.ndarray, q:np.ndarray, gamma:float, ee_module_length:float, avoid_joint_limits:bool=True) -> np.ndarray:
    '''
    Computes a step delta_q in configuration space which reduces an error in operational space,
    while avoiding reaching joint limits. Here, we only consider the position of the end effector and ignore
    its orientation.

    :param x_des:               desired end effector pose
    :param q:                   configuration
    :param gamma:               step size
    :param ee_module_length:    length of the end effector module, starting from the screw at link 3 [millimeters]
    :param avoid_joint_limits:  uses nullspace projection to avoid joint limits, if set to True. Does not, else.
    :return:                    step in configuration space
    '''

    # use forward kinematics to compute the 6D pose of the end effector
    x = ktx.get_ee_pose(q, ee_module_length, MODE_3D)

    # compute jacobian matrix and extract (3x4) sub-matrix 
    # that is only concerned with the position of the end effector
    J = ktx.get_jacobian(q, MODE_3D, ee_module_length)[:3]
    
    # compute pseudo-inverse of jacobian
    J_pinv = np.linalg.pinv(J)
    
    # compute error in end effector space and extract 3D sub-vector
    # that is only concerned with the position of the end effector
    delta_x = (x_des - x)[:3]

    # compute step in configuration space to reduce error
    error_rejection = gamma * J_pinv @ delta_x

    
    if avoid_joint_limits:    
        
        # compute nullspace projection
        P = (np.eye(4) - J_pinv @ J)
        
        # compute derivative of objective function H with respect to configuration q
        dH_dq = -q
        
        # compute nullspace vector by projecting dH_dq into the nullspace of J
        q0 = P @ dH_dq
        
        # compute step in configuration space
        delta_q = error_rejection + q0
    
    else:

        # compute setp in configuration space
        delta_q  = error_rejection
    # -- if 
    
    return delta_q
# ---