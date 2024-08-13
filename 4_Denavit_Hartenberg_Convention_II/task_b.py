'''
-----------------------------
Denavit-Hartenberg Convention
-----------------------------


TASK B - 3D Forward Kinematics with Denavit-Hartenberg
------------------------------------------------------

What this task is about:
    Your task is to implement the forward kinematics function for the LAURA robot in 3D mode based on Denavit-Hartenberg parameters. 
    You will evaluate your results by comparing this end effector pose with the pose of the April Module.

What you need to do in the real world:
    * Take out the four base plate pieces and put them together.
    * Attach LAURA to the base blate in 3D mode. 
    * Connect LAURA to a power supply and to your computer via USB
    * Attach the April Modul to the end of the robot arm
    * Connect the camera and make sure the camera is able to see the base of the robot and the April Module
    
What you need to implement:
    * Have a look at the file april_module.py and implement the missing code there.

What you should see if your code is correct:
    You will two reference frames:
    the first is detected by the camera via the April Modul
    and the second is computed by the forward kinematics with DH parameters.
    Both frames should be very similar (expect minor error due to camera distortions and manufacturing-based inaccuracies).
'''



from laura_interface import *
from april_detection import *
import kinematics as kin
import april_module as am
import task_a
import numpy as np
import time






def main_task_b():
    '''
    YOU DO NOT NEED TO DO ANYTHING HERE.
    JUST READ THE CODE AND TRY TO UNDERSTAND!
    '''

    APRIL_MODULE_LENGTH = 20.5 # millimeters
    BASE_FRAME_ID       = 0

    # homogeneous transformation that represents LAURA's base frame 
    # relative to its april tag marker frame [translation in millimeters]
    T_L_0 = np.array([[1.,  0., 0.,  30.],
                      [0.,  1., 0., -35.],
                      [0.,  0., 1., -29.],
                      [0.,  0., 0.,   1.],])


    # create a serial (USB) connection to the LAURA robot
    laura = LAURA(usb_device_name='/dev/tty.usbmodem21401', mode=MODE_3D, auto_enable=False)

    # create an april object for simple april tag detection
    april = AprilDetection(cam_index=0)


    while True:

        # initialize
        laura.read_sensors()                     # update sensor readings (this updates the laura.q vector)
        Ts = april.detect()                      # detect april tags (this function resurns a dictionary {id:T})
        
        visualized_frames = []                   # List of frames (defined relative to the camera frame) that we are going to visualize.
       
        # compute the end effector pose (relative to the camera frame) based on the detected april markers on the April Module
        april_T_C_ee = am.get_april_ee_pose(Ts)  
        if april_T_C_ee is not None: visualized_frames.append(april_T_C_ee)

        # compute the end effector pose (relative to the base frame) based on the forward kinematic function, according to denavit hartenberg
        dh_T_0_ee = task_a.get_ee_pose(q = laura.q, ee_module_length = APRIL_MODULE_LENGTH, mode = MODE_3D)
        print('end effector pose (relative to baes frame):\n', np.round(dh_T_0_ee, 1), '\n')


        # The following code maps the end effector frame from base frame to camera frame.
        if BASE_FRAME_ID in Ts:
            
            # extract the april tag maker frame on LAURA's base
            T_C_L = Ts[BASE_FRAME_ID]
            visualized_frames.append(T_C_L)

            # compute LAURA's base frame (it lies directly under by transforming the static offset
            T_C_0 = T_C_L @ T_L_0

            # map the end effector pose from base frame to camera frame
            dh_T_C_ee = T_C_0 @ dh_T_0_ee
            visualized_frames.append(dh_T_C_ee)
        # --- if
        

        # only show frames in visualized_frames
        april.show(show_markers=False, additional_frames=visualized_frames)

    # --- while

# --- def



if __name__ == "__main__":
    main_task_b()


