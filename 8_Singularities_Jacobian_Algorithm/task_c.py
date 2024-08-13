'''
------------------
INVERSE-KINEMATICS
------------------


TASK C - LUCY'S IN THE SKY!
---------------------------

What this task is about:
    Draw awesome pictures in the air with the LED module and capture long-exposed images!
    Pretty cool, huh?!
    
What you need to do in the real world:
    * Take out the four base plate pieces and put them together.
    * Attach LAURA to the base blate in 3D mode. 
    * Connect LAURA to a power supply and to your computer via USB
    * Connect the Camera to the Camera stand
    * Connect the Camera to you system via USB
    * Attach the LED module
    
What you need to implement:
    * All the relevant code is given to you
    * You only need to adjust / add keyframes

What you should see if your code is correct:
    * Amazing long-exposed images of LAURA drawing pictures. 
'''


import time              
import numpy as np       
from cubic_interpolation import *
from laura_interface import *
from dynamixel_port import *
import numpy as np
import dynamics as dmx
import kinematics as ktx
from long_exposure import *
import signal







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






def main():


    # this flag indicateds the termination of the control loop
    global shutdown_flag

    # for computing the end effector pose, we need to know the length [in millimeters] 
    LED_MODULE_LENGTH = 30

    # setup LAURA - you need to change the usb device name!
    laura = LAURA(usb_device_name='/dev/tty.usbmodem21401', mode=MODE_3D)
    
    # tune the gains of the dynamixels' on-board C-Space PID controller
    laura.set_gains(p=900, i=100, d=500)

    laura.set_configuration(laura.q)

    # create object for taking long-exposure images
    # You probalby need to change the camera ID to find your webcam
    cam = LongExposure(cam_id=0)



    # Specify the keypoints in operational space (i.e., in end effector space). 
    # Since LAURA is in 3D mode, the end effector pose has six dimensions [x, y, z, roll, pitch, yaw].
    # Be creative!
    '''
    ATTENTION!
    ----------
    1.) Make sure, LAURA is in a non-singular configuration!
    2.) Make sure, LAURA is properly calibrated!
    '''
    l = 40
    kf1 = ktx.get_ee_pose(laura.q, LED_MODULE_LENGTH, mode=MODE_3D)
    
    # N
    kf2 = kf1 + np.array([0, 0,  l, 0., 0., 0.])
    kf3 = kf1 + np.array([l, 0, 0., 0., 0., 0.])
    kf4 = kf1 + np.array([l, 0,  l, 0., 0., 0.])
    
    # Dach
    kf5 = kf1 + np.array([l/2., 0, l + l/2., 0., 0., 0.])
    kf6 = kf1 + np.array([   0, 0,        l, 0., 0., 0.])

    # Z
    kf7 = kf1 + np.array([   l,        0., l, 0., 0., 0.])
    kf8 = kf1 + np.array([   0,        0, 0., 0., 0., 0.])
    kf9 = kf1 + np.array([   l,        0, 0., 0., 0., 0.])
    
    # assemble the keyframes above in a list
    keyframes = np.array([kf1, kf2, kf3, kf4, kf5, kf6, kf7, kf8, kf9])

    # it should take LAURA three seconds to transition between consecutive keyframes
    durations = [3.] * (len(keyframes) - 1)

    # create a cubic interpolation, based on the keyframes and time durations
    cspline = CubicInterpolation(keyframes, durations)

    # these will store the actual and the desired end effector trajectory
    actual_ee_trajectory  = []
    desired_ee_trajectory = []

    # step size: scalar value for adjusting the size of the error rejection step
    gamma = 1
    
    start_time = time.time()
    while (not shutdown_flag) and (time.time() - start_time < cspline.total_duration):

        # take picture for creating long-exposure image
        cam.capture_image()
        
        # show current state of the long exposure image
        cam.show()

        # determine the control time this loop has been running
        time_in_loop     = time.time() - start_time

        # update the joint positions laura.q and joint velocity values laura.dq
        laura.read_sensors()
        
        # compute the desired end effector pose and end effector velocity, based on c-spline interpolation.
        x_des, _, _ = cspline.get_values_for_time(time_in_loop)

        # use forward kinematics to compute the actual end effector pose
        x = ktx.get_ee_pose(laura.q, LED_MODULE_LENGTH, mode=MODE_3D)

        # compute the error in the end effector pose
        delta_x = x_des - x

        # store the desired and the actual ee pose for plotting
        desired_ee_trajectory.append(x_des)
        actual_ee_trajectory.append(x)

        # compute jacobian matrix
        J = ktx.get_jacobian(laura.q, ee_module_length=LED_MODULE_LENGTH, mode=MODE_3D)

        # compute step in configuration space via inverse kinematics
        delta_q = gamma * np.linalg.pinv(J) @ delta_x

        # compute desired configuration by taking a step in configuration space
        q_des = laura.q + delta_q

        # tell LAURA to go to a certain configuration by calling a DXL-based PID controller
        laura.set_configuration(q_des)

    # --- while

    # properly shutdown LAURA to prevent errors
    laura.disable()

    # save long-exposure image to file
    cam.save('long-exposure.jpg')

# --- main



if __name__ == "__main__":
    main()



