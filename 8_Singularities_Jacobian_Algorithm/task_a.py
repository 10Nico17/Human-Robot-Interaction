'''
------------------
INVERSE-KINEMATICS
------------------


TASK A - Operational Space Control
----------------------------------

What this task is about:
    For this task, you have to implement the jacobian algorithm.
    It genereates the robot's jacobian matrix, based on the denivat hartenberg parameter table.
    Thus, if you want to do forward and inverse kinematics in the future 
    with any kind of robot, all you need is the DH table. 
    Pretty awesome!
        
What you need to do in the real world:
    * Take out the four base plate pieces and put them together.
    * Attach LAURA to the base blate in 3D mode. 
    * Connect LAURA to a power supply and to your computer via USB
    * ATTENTION: Do NOT use LAURA in zero configuratoin in this task! Bad things will happen!
    
What you need to implement:
    * Implement the missing code in kinematics.py
    * Implement the code in this file.
    * Things will likely go wrong in the beginning, so better keep a second terminal open for quick recalibration
    * You may tune the gains of the dynamixels' on-board PID controller (see below)

What you should see if your code is correct:
    * Laura's end effector should move on a triangle within the xy-plane
    * After LAURA finished moving, a plot should appear, comparing LAURA's desired and actual end effector trajectory. 
    * There will always be minor oscillations, but try to minimize them as much as possible.
'''

import time
import numpy as np
from cubic_interpolation import *
from laura_interface import *
from dynamixel_port import *
import numpy as np
import dynamics as dmx
import kinematics as ktx
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



def plot_ee_trajectory(desired, actual):
    '''
    Plots the desired and the actual end effector trajectories in the xy plane.

    :param desired:   list of desired end effector trajectories : [[x_t, y_t, theta_t] for t in times]
    :param actual:    list of actual end effector trajectories  : [[x_t, y_t, theta_t] for t in times]
    '''
    desired, actual = np.array(desired).T, np.array(actual).T

    plt.figure()
    plt.title('End effector trajectory')
    plt.plot(desired[0], desired[1], '--', label='desired')
    plt.plot(actual[0], actual[1], label='actual')
    plt.gca().set_aspect('equal', 'box')
    plt.xlabel('x position [mm]')
    plt.ylabel('y position [mm]')
    plt.legend()
    plt.show()
# ---




def main():

    # this flag indicateds the termination of the control loop
    global shutdown_flag

    # for computing the end effector pose, we need to know the length [in millimeters] 
    LED_MODULE_LENGTH = 30

    # setup LAURA - you need to change the usb device name!
    laura = LAURA(usb_device_name='/dev/tty.usbmodem21401', mode=MODE_3D)
    
    # tune the gains of the dynamixels' on-board C-Space PID controller
    laura.set_gains(p=900, i=200, d=500)

    # Specify the keypoints in operational space (i.e., in end effector space). 
    # Since LAURA is in 3D mode, the end effector pose has six dimensions [x, y, z, roll, pitch, yaw].
    # The keypoints prescribe a triangle. The first keypoint and the last keypoint are LAURA's current end effector pose.
    '''
    ATTENTION!
    ----------
    1.) Make sure, LAURA is in a non-singular configuration!
    2.) Make sure, LAURA is properly calibrated!
    '''
    kf1 = ktx.get_ee_pose(laura.q, LED_MODULE_LENGTH, mode=MODE_3D)
    kf2 = kf1 + np.array([ 40., 70., 0., 0., 0., 0.])
    kf3 = kf1 + np.array([-40., 70., 0., 0., 0., 0.])
    kf4 = kf1

    # assemble the keyframes above in a list
    keyframes = np.array([kf1, kf2, kf3, kf4])

    # it should take LAURA three seconds to transition between consecutive keyframes
    durations = [2., 2., 2]

    # create a cubic interpolation, based on the keyframes and time durations
    cspline = CubicInterpolation(keyframes, durations)

    # these will store the actual and the desired end effector trajectory
    actual_ee_trajectory  = []
    desired_ee_trajectory = []

    # step size: scalar value for adjusting the size of the error rejection step
    gamma = 1
    
    start_time = time.time()
    while (not shutdown_flag) and (time.time() - start_time < cspline.total_duration):

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

    plot_ee_trajectory(desired_ee_trajectory, actual_ee_trajectory)
# --- main



if __name__ == "__main__":
    main()



