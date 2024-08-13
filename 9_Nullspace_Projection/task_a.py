'''
--------------------
NULLSPACE PROJECTION
--------------------


TASK A - Avoiding Joint Limits
------------------------------

What this task is about:
    Controlling the robot's end effector using the jacobian matrix can lead to problematic behaviors,
    such as reaching joint limits or singular configurations.
    To avoid such problems, we can use nullspace projection. I.e., we achieve secondary objectives
    while moving the joints but without moving the end effector.
    In this task, you will use nullspace projection to prevent reaching the joint limits.

What you need to do in the real world:
    * Take out the four base plate pieces and put them together.
    * Attach LAURA to the base blate in 3D mode. 
    * Connect LAURA to a power supply and to your computer via USB
    * Attach no end effector module

What you need to implement:
    * Implement the missing code in control.py
    * Implement the missing code in this file.
    * Things will likely go wrong in the beginning, so better keep a second terminal open for quick recalibration

What you should see if your code is correct:
    * LAURA's end effector moves on a sraight line
    * LAURA should reach its joint limits when the parameter avoid_joint_limits is set to False
    * LAURA should avoid its joint limits when the parameter acoid_joint_limits is set to True
'''

import time
import numpy as np
from cubic_interpolation import *
from laura_interface import *
from dynamixel_port import *
import numpy as np
import kinematics as ktx
import controller as ctrl
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

    # global flag for shutting down the control loop
    global shutdown_flag

 
    # Initialize LAURA
    # ----------------

    # specify the length of end effector module [millimeters] 
    EE_MODULE_LENGTH = 0.
                     
    # setup LAURA - you need to change the usb device name!
    laura = LAURA(usb_device_name='/dev/tty.usbmodem21401', baudrate=1000000, mode=MODE_3D)

     # tune the gains of the dynamixels' on-board C-Space PID controller
    laura.set_gains(p=900, i=200, d=500)

    # LAURA should start at the neutral configuration
    laura.move_to_neutral(2)


    # Initialize C-Spline Interpolation
    # ---------------------------------

    # update the joint positions laura.q and joint velocity values laura.dq
    laura.read_sensors()

    # specify keyframes in operational space
    kf1 = ktx.get_ee_pose(laura.q, EE_MODULE_LENGTH, mode=MODE_3D)
    kf2 = kf1 + np.array([-150.,-70., -130., 0., 0., 0.])
    kf3 = kf1

    # assemble the keyframes above in a list
    keyframes = np.array([kf1, kf2, kf3])

    # it should take LAURA two seconds to transition between consecutive keyframes
    durations = [3., 3.]

    # create a cubic interpolation, based on the keyframes and time durations
    cspline = CubicInterpolation(keyframes, durations)



    # Initialize Control Loop
    # -----------------------

    # these will store the actual and the desired end effector trajectory
    actual_ee_trajectory  = []
    desired_ee_trajectory = []

    # step size: scalar value for adjusting the size of the error rejection step
    gamma = 1

    # remember the start time for computing the time inside the control loop
    start_time = time.time()



    # Control Loop
    # ------------

    while (not shutdown_flag) and (time.time() - start_time < cspline.total_duration):

        # update the joint positions laura.q and joint velocity values laura.dq
        laura.read_sensors()
        
        # compute the desired end effector pose and end effector velocity, based on c-spline interpolation.
        x_des, _, _ = cspline.get_values_for_time(time.time() - start_time)

        # compute control step in configuration space
        '''<CHANGE THE PARAMETER avoid_joint_limits TO OBSERVE BEHAVIORAL DIFFERENCES>'''
        delta_q = ctrl.compute_delta_q(x_des, laura.q, gamma, EE_MODULE_LENGTH, avoid_joint_limits=True)

        # store the desired and the actual ee pose for plotting
        desired_ee_trajectory.append(x_des)
        actual_ee_trajectory.append(ktx.get_ee_pose(laura.q, EE_MODULE_LENGTH, mode=MODE_3D))

        # tell LAURA to go to a certain configuration by calling a DXL-based PID controller
        laura.set_configuration(laura.q + delta_q)

    # --- while



    # Shut-Down Procedure
    # -------------------

    # properly shutdown LAURA to prevent errors
    laura.disable()

    # launch graphical comparison between desired and actual trajectory
    plot_ee_trajectory(desired_ee_trajectory, actual_ee_trajectory)

# --- main



if __name__ == "__main__":
    main()



