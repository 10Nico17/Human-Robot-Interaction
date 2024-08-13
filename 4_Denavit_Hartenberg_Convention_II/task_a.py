'''
-----------------------------
Denavit-Hartenberg Convention
-----------------------------


TASK A - 2D Forward Kinematics with Denavit-Hartenberg
------------------------------------------------------

What this task is about:
    Your task is to implement the forward kinematics function for the LAURA robot in 2D mode,
    based on Denavit-Hartenberg parameters.

What you need to do in the real world:
    * Take out the four base plate pieces and put them together.
    * Attach LAURA to the base blate in 2D mode. 
    * Connect LAURA to a power supply and to your computer via USB
    * Attach the roller to the last motor (the roller should point downwards and touch the plate to provide structural support).
    * Attach the Pin module to the end effector.

What you need to implement:
    * Have a look at the file kinematics.py and implement the missing code there.
    * Also implement the missing code in this file.

What you should see if your code is correct:
    The pin will indicate the end effector's current poition on the base plate's coordinate system.
    Your code is correct when the pin and your code indicate the same position 
    (expect minor errors due to manufacturing-based inaccuracies). 
    Make sure you compare your results with the 2D-Mode scale on the base plate (because there are two of these)!
'''



from laura_interface import *
import kinematics as kin
import numpy as np
import time






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
    LINK_LENGTH_3       = 21

    return np.array([[q[0] + np.pi / 2,  -np.pi / 2,                               0,  LINK_LENGTH_0],
                     [q[1] - np.pi / 2,           0,                   LINK_LENGTH_1,              0],
                     [            q[2],           0,                   LINK_LENGTH_2,              0],
                     [            q[3],   np.pi / 2,  LINK_LENGTH_3+ee_module_length,              0]])
# ---



def get_ee_pose(q, ee_module_length, mode=MODE_2D):
    '''
    Returns the robot's end effector pose.

    :param dh_param_table:       Denavit-Hartenberg parameter table. This table depends on the current robot configuration q = [theta_1, ..., theta_n].
    
    :return:                     In MODE_2D: A 3D Vector (x, y, theta), encoding the position and orientation of the end effector on the plane
                                 In MODE_3D: A homogeneous transformation, representing the robot's end effector frame, relative to the base frame
    '''
    
    # compose Denavit-Hartenberg parameter table, based on the robot's current configuration q
    dh_param_table = get_dh_param_table(q, ee_module_length, mode)

    # compute the robot's end effector frame relative to the base frame
    T_0_ee = kin.get_ee_frame_for_dh_param_table(dh_param_table)
    
    if mode == MODE_2D:
        return kin.extract_planar_pose_from_frame(T_0_ee)
    else:    
        return T_0_ee
# ---




def main_task_a():
    '''
    YOU DO NOT NEED TO DO ANYTHING HERE.
    JUST READ THE CODE AND TRY TO UNDERSTAND!
    '''

    PIN_MODULE_LENGTH = 8.5 # [millimeters]

    # create a serial (USB) connection to the LAURA robot
    # you need to change the name of the usb device. You can use the dynamixel wizard to find out this name.
    laura = LAURA(usb_device_name='/dev/tty.usbmodem21401', mode=MODE_2D)

    # disable motors so you can manually move them 
    laura.disable()


    while True:
        
        # update sensor readings (this updates the laura.q vector)
        laura.read_sensors()
        
        # compute the end effector pose
        pose = get_ee_pose(q                = laura.q, 
                           ee_module_length = PIN_MODULE_LENGTH,
                           mode             = MODE_2D)
        
        
        pose[:2] = pose[:2] / 10. # convert translation to centimeters for better comparison with ground plate scale
        print('end effector pose:\n', np.round(pose, 1), '\n')
    # --- while

    # shut down laura
    laura.disable()

# --- def



if __name__ == "__main__":
    main_task_a()

