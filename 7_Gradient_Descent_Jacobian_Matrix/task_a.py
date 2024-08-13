'''
------------------
INVERSE-KINEMATICS
------------------


TASK A - Operational Space Control
-------------------------------------

What this task is about:
    Previously, we controlled the robot in configuration space. Thus, we interpolated between joint angles.
    As a result, the end effector moved on arcs and not on a straight line. 
    We now want to control the position of the end effector directly, and not the position of the joints. 
    More specifically, we will interpolate between keyframes in operational space, so that the end effector moves on a straight line.
    For this task, you will determine torques, based on the transpose of the jacobian matrix.

What you need to do in the real world:
    * Take out the four base plate pieces and put them together.
    * Attach LAURA to the base blate in 2D mode. 
    * Connect LAURA to a power supply and to your computer via USB
    * Attach Pen April Modul to the end of the robot arm
    * ATTENTION: Do NOT use LAURA in zero configuratoin in this task! Bad things will happen!
    
What you need to implement:
    * Implement the missing code in kinematics.py
    * Implement the code in this file.
    * Things will likely go wrong in the beginning, so better keep a second terminal open for quick recalibration

What you should see if your code is correct:
    * Laura's end effector should move on a straight line to a desired end effector position.
    * After LAURA finished moving, a plot should appear, comparing LAURA's desired and actual end effector trajectory. 
    * There will always be minor oscillations, but try to minimize them as much as possible.
'''

from curses import baudrate
import time                              # For accessing the current time
import numpy as np                       # Fast math library
from cubic_interpolation import *        # Custom python package for cubic spline interpolation
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





def main_task_c():

    # make sure we know about the flag's status inside this function.
    # their values are changed by a callback functions outside this function.
    global shutdown_flag

    # for computing the end effector pose, we need to know the length [in millimeters] 
    # of the module attached to the end of LAURA's arm
    PEN_MODULE_LENGTH = 15.

    # setup LAURA - you need to change the usb device name!
    laura = LAURA(usb_device_name='/dev/tty.usbmodem21401', baudrate=1000000, mode=MODE_2D)

    # we operate LAURA by defining the torques in its joints. 
    # under the hood, this is realized through adjustments of the electical current in the motors.
    laura.set_operating_mode(CURRENT_CONTROL_MODE)
    laura.set_torque(np.zeros(4))

    # At the beginning, the desired configuration is identical to LAURA's current configuraion
    # and the desired velocity is zero. Thus, laura should try to stay still at the onset of the control loop.
    laura.read_sensors()

    # Previously, we specified keypoints in configuraion space (i.e., in joint angle space).
    # Now, we specify the keypoints in operational space (i.e., in end effector space). 
    # Since LAURA is in 2D mode, the end effector pose has three dimensions [x, y, theta].
    # The keypoints should prescribe a line from the end effector's current pose to a desired pose, and back again.
    # Thus, the first keypoint and the last keypoint are LAURA's current end effector pose.
    # You need to choose the other keypoints manually.
    # To find out LAURA's current configuration, run the script laura-teaching/examples/get_config.py
    kf1 = ktx.get_ee_pose(laura.q, PEN_MODULE_LENGTH, mode=MODE_2D)
    kf2 = kf1 + np.array([-70., 0., 0.])
    kf3 = kf1

    # assemble the keyframes above in a list
    keyframes = np.array([kf1, kf2, kf3])

    # it takes LAURA 3 seconds to transition between consecutive keyframes
    durations = [2., 2.]

    # create a cubic interpolation, based on the keyframes and time durations
    cin = CubicInterpolation(keyframes, durations)

    # these will store the actual and the desired end effector trajectory
    actual_ee_trajectory  = []
    desired_ee_trajectory = []

    '''
    ATTENTION!
    The values of the kp and kv work well on my (i.e., Steffen's) robot,
    but that does not necessarily mean that these also work well on yours!
    You probably need to further tune these gains. 
    Keep in mind: it will never be perfect, given our equipment...

    MORE ATTENTION!
    Since we now control the end effector, both kp and kv have three dimensions,
    one for each end effector pose dimension [x, y, theta].
    '''
    # specify the propotional and derivative gains for the PD controller
    kp = np.array([1., 1., 0.01*0]) * 0.0002
    kv = np.array([0.001, 0.001, 0.00001 * 0]) * 0.004
    
    # always remember the end effector pose of the previous loop iteration
    # for numerically computing the end effector velocity
    old_x = ktx.get_ee_pose(laura.q, PEN_MODULE_LENGTH)

    start_time = time.time()
    while (not shutdown_flag) and (time.time() - start_time < cin.total_duration):

        # determine the control time this loop has been running
        time_in_loop     = time.time() - start_time

        # update the joint positions laura.q and joint velocity values laura.dq
        laura.read_sensors()
        
        # compute the torques acting on the joints because of friction. 
        b = dmx.predict_friction(laura.dq)        

        # compute the desired end effector pose and end effector velocity, 
        # based on c-spline interpolation, for the current point in time.
        x_des, dx_des, _ = cin.get_values_for_time(time_in_loop)

        # use forward kinematics to compute the actual end effector pose
        x = ktx.get_ee_pose(laura.q, PEN_MODULE_LENGTH)

        # numerically determine the current end effector velocity.
        # for this, compute the difference between the current end effector pose
        # and the end effector pose of the previous loop. 
        # finally, remember the current pose for computing the velocity in the next loop.
        dx = x - old_x
        old_x = x

        # store the desired and the actual ee pose for plotting
        desired_ee_trajectory.append(x_des)
        actual_ee_trajectory.append(x)

        # compute jacobian matrix
        J = ktx.get_jacobian(laura.q, ee_module_length=PEN_MODULE_LENGTH)

        # compute force exerted at the end effector
        F = kp * (x_des - x) + kv * (dx_des - dx)

        # compute torque based on the force via inverse kinematics
        tau = J.T @ F

        # scale torque for the different joints
        #tau = tau * np.array([0., 1., 1., 0.1])

        # apply joint torques as specified above
        laura.set_torque(tau)

    # --- while

    # properly shutdown LAURA to prevent errors
    laura.disable()

    plot_ee_trajectory(desired_ee_trajectory, actual_ee_trajectory)
# --- main



if __name__ == "__main__":
    main_task_c()



