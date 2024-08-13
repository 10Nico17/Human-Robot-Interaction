'''
--------------------
NULLSPACE PROJECTION
--------------------


TASK B - Visual Servoing on Robot
---------------------------------

What this task is about:
    You will now implement visual servoing on the LAURA robot.
    Pretty cool, huh?!

What you need to do in the real world:
    * Take out the four base plate pieces and put them together.
    * Attach LAURA to the base blate in 3D mode. 
    * Connect LAURA to a power supply and to your computer via USB
    * Attach the camera as end effector module
    * Attach the camera to your computer
    * Take out the blob

What you need to implement:
    * Implement the missing code in this file.

What you should see if your code is correct:
    * LAURA should try to look at the blob while maintaining a distance of 30 centimeters
'''

import time
import numpy as np
from cubic_interpolation import *
from visual_servoing import *
from laura_interface import *
import numpy as np
import kinematics as ktx
import controller as ctrl
import signal
import math




'''
---------------------------------------------------------------------------
The folloing code ensures a proper shut-down of this script,
when the user presses CTRL-C on the keyboard. 
In this case, a callback function will be called, which tells the while loop to stop by setting a flag.
'''

# this will be our flag to indicate a CTRL-C signal was caught
shutdown_flag = False

# callback function, triggered by CTRL-C
def ctrl_c_callback(signum, frame):
    global shutdown_flag
    shutdown_flag = True
# ---

# Register the callback function for the SIGINT signal (CTRL-C)
signal.signal(signal.SIGINT, ctrl_c_callback)

'''
---------------------------------------------------------------------------
'''

def angle_difference(goal_angles, source_angles):
    """
    Compute the signed smallest difference between pairs of source and goal angles.
    """
    # Compute the sine and cosine of the differences
    sin_diff = np.sin(goal_angles - source_angles)
    cos_diff = np.cos(goal_angles - source_angles)

    # Use arctan2 to find the signed angle difference
    angle_diffs = np.arctan2(sin_diff, cos_diff)

    return angle_diffs
# ---


def main():

    # global flag for shutting down the control loop
    global shutdown_flag

 
    # Initialize LAURA
    # ----------------

    # specify the length of end effector module [millimeters] 
    EE_MODULE_LENGTH = 0.
                     
    # setup LAURA - you need to change the usb device name!
    laura = LAURA(usb_device_name='/dev/tty.usbmodem11401', baudrate=1000000, mode=MODE_3D)

     # tune the gains of the dynamixels' on-board C-Space PID controller
    laura.set_gains(p=900, i=200, d=500)

    # step size: scalar value for adjusting the size of the error rejection step
    gamma = 0.01

    # LAURA should start at the neutral configuration
    q_initial = np.array([-0.00306796, -0.81147584,  1.3330293 ,  1.11367005])
    x_initial = ktx.get_ee_pose(q_initial, EE_MODULE_LENGTH, MODE_3D)
    laura.move_to_configuration(q_initial, 2)


    # Initialize Visual Servoing
    # --------------------------
    
    # connect to webcam and initialize blob detection via OpenCV
    cap, detector = get_webcam_blob_detector(cam_index=0)

    r     = 17.5   # radius of real circle [millimeters]
    f     = 1650   # camera's focal length 
    d_des = 300    # desired distance between real circle and webcam [millimeters]



    

    # Control Loop
    # ------------

    while (not shutdown_flag):

        # update the joint positions laura.q and joint velocity values laura.dq
        laura.read_sensors()

        # compute end effector pose (only considert position)
        x = ktx.get_ee_pose(laura.q, EE_MODULE_LENGTH, MODE_3D)


        # capture and resize camera image
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        if not ret: print("Failed to grab frame.")

        # detect blobs (i.e. circles) in camera image
        keypoints = detector.detect(frame)

        # have we detected at least one blob?
        if len(keypoints) > 0:
            
            # extract image features from first blob
            blob = keypoints[0]
            curr_img_features = np.array([blob.pt[0], blob.pt[1], blob.size])

            # compute the step in end effector space based on visual servoing
            delta_cam = get_delta_cam(des_img_features  = get_des_img_features(frame, f, r, d_des), 
                                      curr_img_features = curr_img_features, 
                                      J_img             = get_image_jacobian(curr_img_features, f, r),
                                      step              = 1)
        else:
            delta_cam = np.zeros(3)
        # -- if 


        # compute delta_x from delta_cam:
        # x in camera frame is x in base frame 
        # y in camera frame is z in base frame
        # z in camera frame is y in base frame and
        
        # we now compose the step in end effector space
        delta_x = np.zeros(6)

        # keep the initial end effector orientation
        delta_x[3:] = 0.2 * angle_difference(x_initial[3:], x[3:])

        # keep the ee x-position at zero
        delta_x[0] = (0. - x[0])

        # adjust the ee y-position according to the camera's step in z
        delta_x[1] = 0.2 * delta_cam[2]

        # adjust the ee z-position according to the camera's step in y
        delta_x[2] = -0.2 * delta_cam[1] 

        # prints for debugging
        print('delta_cam', delta_cam)
        print('delta_x', np.round(delta_x, 2))

        # every end effector dimension has a separate step size
        gamma = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.2])

        # compute control step in configuration space
        delta_q = ctrl.compute_delta_q(delta_x, laura.q, gamma=1., ee_module_length=EE_MODULE_LENGTH)


        q_des = laura.q + delta_q
        
        # make sure the first joint always points forward to make our lives easier
        q_des[0] = 0.

        # tell LAURA to go to a certain configuration by calling a DXL-based PID controller
        laura.set_configuration(q_des)

    # --- while



    # Shut-Down Procedure
    # -------------------

    # properly shutdown LAURA to prevent errors
    laura.disable()


# --- main



if __name__ == "__main__":
    main()



