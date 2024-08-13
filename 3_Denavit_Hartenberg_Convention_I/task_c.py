'''
------------------
Forward Kinematics
------------------


TASK C - 2D Forward Kinematics with Homogeneous Transformations
---------------------------------------------------------------

Your task is to implement the forward kinematics function for the LAURA robot in 2D mode.
In this mode, the end effector has three dimensions (two for position and one for orientation).
However, we want to represent the pose of the end effector as a reference frame, encoded in a homogeneous transformation.

Use homogeneous transformations to map between frames!
There is one base frame, one frame for the end effector, and one frame for each joint 
(except joint 0, because it is not used in 2D mode).

* Take out the four base plate pieces and put them together.
* Attach LAURA to the base blate in 2D mode. 
* Attach the roller to the last motor (the roller should point downwards and touch the plate to provide structural support).
* Attach the Pin module to the end effector.

The pin will indicate the end effector's current poition on the base plate's coordinate system.
Your code is correct when the pin and your code indicate the same position 
(expect minor errors due to manufacturing-based inaccuracies). 

Make sure you compare your results with the 2D-Mode scale on the base plate (because there are two of these)!
'''



from laura_interface import *
from laura_utils import *
import numpy as np
import time



# create a serial (USB) connection to the LAURA robot
# you need to change the name of the usb device. You can use the dynamixel wizard to find out this name.
laura = LAURA(usb_device_name='/dev/tty.usbmodem21401', mode=MODE_2D)

# disable motors so you can manually move them 
laura.disable()




def get_ee_pose(q):
    '''<IMPLEMENT YOUR CODE BELOW>'''
    
    # lengths of laura's links [centimeters]
    LINK_LENGTH_0     = 6.05
    LINK_LENGTH_1     = 7.25
    LINK_LENGTH_2     = 7.25
    LINK_LENGTH_3     = 2.1
    PIN_MODULE_LENGTH = 0.85 

    # frame of joint 1 expressed in base frame 0
    T_0_1 = np.array([[np.cos(q[1] + np.pi/2), -np.sin(q[1] + np.pi/2), 0],
                      [np.sin(q[1] + np.pi/2),  np.cos(q[1] + np.pi/2), LINK_LENGTH_0],
                      [0, 0, 1]])

    # frame of joint 2 expressed in frame of joint 1
    T_1_2 = np.array([[np.cos(q[2]), -np.sin(q[2]), LINK_LENGTH_1],
                      [np.sin(q[2]),  np.cos(q[2]), 0],
                      [0, 0, 1]])

    # frame of joint 3 expressed in frame of joint 2
    T_2_3 = np.array([[np.cos(q[3]), -np.sin(q[3]), LINK_LENGTH_2],
                      [np.sin(q[3]),  np.cos(q[3]), 0],
                      [0, 0, 1]])

    # end effector frame expressed in frame of joint 3
    T_3_ee = np.array([[np.cos(-np.pi/2), -np.sin(-np.pi/2), LINK_LENGTH_3 + PIN_MODULE_LENGTH],
                       [np.sin(-np.pi/2),  np.cos(-np.pi/2), 0],
                       [0, 0, 1]])
    
    # end effector frame expressed in base frame
    T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee

    return T_0_ee
    # ---

# ---







loop_duration = 100 # in seconds
start_time    = time.time()

# repeat the following code for <loop_duration> seconds 
while time.time() - start_time < loop_duration:
    '''<YOU DO NOT NEED TO DO ANYTHING HERE>'''
    
    laura.read_sensors()                # update sensor readings (this updates the laura.q vector)
    pose    = get_ee_pose(laura.q)      # compute the end effector pose
    rounded = np.round(pose, 1)         # round to one decimal precision

    print('end effector pose:\n', rounded, '\n')

# shut down laura
laura.disable()