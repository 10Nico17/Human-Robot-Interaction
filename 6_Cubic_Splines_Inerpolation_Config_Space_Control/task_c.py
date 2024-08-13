'''
--------------------------
CUBIC SPLINE INTERPOLATION
--------------------------


TASK C - 2D Fun with C-Splines
------------------------------

What this task is about:
    In this task, LAURA will draw pictures.
    You will combine Cubic Spline Interpolation with model-based feedback control.
    More specifically, you will create a trajectory in configurational space,
    when the LAURA robot is in 2D mode by specifying keyframes. 

What you need to do in the real world:
    * Use the DynamixelWizard and switch the baudrates of all motors to 1000000 (if you have not done so before)
    * Take out the four base plate pieces and put them together.
    * Attach LAURA to the base blate in 2D mode. 
    * Connect LAURA to a power supply and to your computer via USB
    * Securely attach the Pen Module and insert the pen so that it touches the ground (without the cap).

What you need to keep in mind:
    * Your code will not(!) run perfectly, the first time you run it. 
      If the robot goes crazy, you may have to unplug it from the power supply.
      In this case, you need to recalibrate it. Therefore, it makes sense to keep
      a teminal open which is located in the laura-teaching/example directory.

What you need to implement:
    * Implement the code in this file

What you should see if your code is correct:
    * Laura's will try to draw the four edges of a square. These corners will be curved, as we use c-space control
    * You will plot both the desired and the actual end effector trajectory. These will not be identical.
    * You will see oscillations, which is normal. Try to keep them as small as possible.
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
    plt.xlabel('x position [mm]')
    plt.ylabel('y position [mm]')
    plt.legend()
    plt.show()
# ---





def main_task_c():

    # make sure we know about the flag's status inside this function.
    # their values are changed by a callback functions outside this function.
    global shutdown_flag

    PEN_MODULE_LENGTH = 15 # millimeters

    # setup LAURA - you need to change the usb device name!
    laura = LAURA(usb_device_name='/dev/tty.usbmodem2101', baudrate=1000000, mode=MODE_3D)

    # we operate LAURA by defining the torques in its joints. 
    # under the hood, this is realized through adjustments of the electical current in the motors.
    laura.set_operating_mode(CURRENT_CONTROL_MODE)

    # At the beginning, the desired configuration is identical to LAURA's current configuraion
    # and the desired velocity is zero. Thus, laura should try to stay still at the onset of the control loop.
    laura.read_sensors()

    # Specify keypoint in configurational space. These keypoints should prescribe a scquare on the ground plate.
    # The first keypoint and the last keypoint are LAURA's current configuration.
    # You need to choose the other keypoints manually.
    # To find out LAURA's current configuration, run the script laura-teaching/examples/get_config.py
    kf1 = laura.q
    kf2 = [ 1.53551477,  1.0860584 , -0.159534  , -0.06289321]
    kf3 = [ 1.51403904,  0.82681564, -1.62755362, -0.57217483]
    kf4 = [ 1.60300992, -0.61972824,  0.76699039,  0.85135934]
    kf5 = laura.q

    # assemble the keyframes above in a list
    keyframes = np.array([kf1, kf2, kf3, kf4, kf5])

    # it takes LAURA 3 seconds to transition between consecutive keyframes
    durations = [4, 4, 4, 4]

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
    '''
    # specify the propotional and derivative gains for the PD controller
    kp = np.array([1, 3., 10., 1.])
    kv = np.array([0.001, 0.003, 0.01, 0.001])
    

    start_time = time.time()
    while (not shutdown_flag) and (time.time() - start_time < cin.total_duration):

        # determine the control time this loop has been running
        time_in_loop     = time.time() - start_time

        # update the joint positions laura.q and joint velocity values laura.dq
        laura.read_sensors()
        
        # compute the torques acting on the joints because of friction. 
        b = dmx.predict_friction(laura.dq)        

        # compute the desired joint positions and joint velocities, 
        # based on c-spline interpolation, for the current point in time.
        q_des, dq_des, _ = cin.get_values_for_time(time_in_loop)

        # use forward kinematics to store the desired and the actual end effector pose
        desired_ee_pose = ktx.get_ee_pose(q_des, PEN_MODULE_LENGTH)
        actual_ee_pose  = ktx.get_ee_pose(laura.q, PEN_MODULE_LENGTH)

        # store the desired and the actual ee pose for plotting
        desired_ee_trajectory.append(desired_ee_pose)
        actual_ee_trajectory.append(actual_ee_pose)

        # model-based feedback PD controller in configurational space
        tau = kp * (q_des - laura.q) + kv * (dq_des - laura.dq) -b 

        # apply joint torques as specified above
        laura.set_torque(tau)

    # --- while

    # properly shutdown LAURA to prevent errors
    laura.disable()

    plot_ee_trajectory(desired_ee_trajectory, actual_ee_trajectory)
# --- main



if __name__ == "__main__":
    main_task_c()



