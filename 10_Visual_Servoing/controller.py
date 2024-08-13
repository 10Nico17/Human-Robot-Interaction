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



def compute_delta_q(delta_x:np.ndarray, q:np.ndarray, gamma:float, ee_module_length:float) -> np.ndarray:
    '''
    Computes a step delta_q in configuration space which realizes a step in operational space.

    :param delta_x:             step in end effector space
    :param q:                   configuration
    :param gamma:               step size
    :param ee_module_length:    length of the end effector module, starting from the screw at link 3 [millimeters]
    :param avoid_joint_limits:  uses nullspace projection to avoid joint limits, if set to True. Does not, else.
    :return:                    step in configuration space
    '''

    # compute jacobian matrix 
    J = ktx.get_jacobian(q, MODE_3D, ee_module_length)
    
    # compute pseudo-inverse of jacobian
    J_pinv = np.linalg.pinv(J)
    
    # compute step in configuration space to reduce error
    delta_q =  gamma * J_pinv @ delta_x

    return delta_q
# ---