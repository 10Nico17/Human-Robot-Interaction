'''
--------------------------
CUBIC SPLINE INTERPOLATION
--------------------------


TASK B - Configurational Space Control in 3D Mode
-------------------------------------------------

What this task is about:
    In this task, you will combine Cubic Spline Interpolation with model-based feedback control.
    More specifically, you will create a trajectory in configurational space,
    when the LAURA robot is in 3D mode by specifying keyframes.

What you need to do in the real world:
    * Use the DynamixelWizard and switch the baudrates of all motors to 1000000 (if you have not done so before)
    * Take out the four base plate pieces and put them together.
    * Attach LAURA to the base blate in 3D mode. 
    * Connect LAURA to a power supply and to your computer via USB
    * Calibrate LAURA (run the laura-teaching/examples/calibrate.py script)

What you need to keep in mind:
    * Your code will not(!) run perfectly, the first time you run it. 
      If the robot goes crazy, you may have to unplug it from the power supply.
      In this case, you need to recalibrate it. Therefore, it makes sense to keep
      a teminal open which is located in the laura-teaching/example directory.

What you need to implement:
    * Implement the code in this file

What you should see if your code is correct:
    * The robot should move smoothly from its initial configuration to a second configuration and
      then back again to its initial configuration. 
    * You will plot both the desired and the actual trajectory in configuration space. These will not be identical.
    * There will always be oscillations. Nonetheless, try to keep these as small as possible.
'''



import time
import numpy as np
from cubic_interpolation import *
from laura_interface import *
from dynamixel_port import *
import numpy as np
import dynamics as dmx
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






def plot_q_trajectory(desired, actual):
    '''
    Plots the desired and the actual trajectory in configuration space.

    :param desired:   list of desired trajectory : [[q0_t,..qn_t] for t in times]
    :param actual:    list of actual end effector trajectories  : [[q0_t,..qn_t] for t in times]
    '''

    # select nice colors
    cmap1 = plt.get_cmap('Pastel1')
    cmap2 = plt.get_cmap('Set1')

    desired, actual = np.array(desired).T, np.array(actual).T
    n_dof = desired.shape[0]
    ts    = np.arange(desired.shape[1])

    plt.figure()
    plt.title('Trajectory in configuration space')
    [plt.plot(ts, desired[d], '--', label='desired q'+str(d), color=cmap1(d)) for d in range(n_dof)]
    [plt.plot(ts, actual[d], label='actual q'+str(d), color=cmap2(d)) for d in range(n_dof)]
    plt.xlabel('Time [ctrl loop step]')
    plt.ylabel('Joint position [rad]')
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width * 0.85, box.height])

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
# ---





def main_task_b():

    # make sure we know about the flag's status inside this function.
    # their values are changed by a callback functions outside this function.
    global shutdown_flag

    # setup LAURA - you need to change the usb device name!
    laura = LAURA(usb_device_name='/dev/tty.usbmodem2101', baudrate=1000000, mode=MODE_3D)

    # we operate LAURA by defining the torques in its joints. 
    # under the hood, this is realized through adjustments of the electical current in the motors.
    laura.set_operating_mode(CURRENT_CONTROL_MODE)

    # At the beginning, the desired configuration is identical to LAURA's current configuraion
    # and the desired velocity is zero. Thus, laura should try to stay still at the onset of the control loop.
    laura.read_sensors()

    # Specify keypoint in configurational space. The first keyframe and the last keyframe are LAURA's current configuration.
    # You need to choose the second keyframe manually. 
    # To find out LAURA's current configuration, run the script laura-teaching/examples/get_config.py
    kf1 = laura.q
    kf2 = [1.16735938, 0.32213597, 0.32673791, 0.65194183]
    kf3 = laura.q

    # assemble the keyframes above in a list
    keyframes = np.array([kf1, kf2, kf3])

    # it takes LAURA 3 seconds to transition between consecutive keyframes
    durations = [4, 4]

    # create a cubic interpolation, based on the keyframes and time durations
    cin = CubicInterpolation(keyframes, durations)

    # these will to store the actual and the desired trajectory in configuration space
    actual_q_trajectory  = []
    desired_q_trajectory = []

    '''
    ATTENTION!
    The values of the kp and kv work well on my (i.e., Steffen's) robot,
    but that does not necessarily mean that these also work well on yours!
    You probably need to further tune these gains. 
    Keep in mind: it will never be perfect, given our equipment...
    '''
    # specify the propotional and derivative gains for the PD controller
    kp = np.array([0.5, 3., 10., 1.])
    kv = np.array([0.001, 0.003, 0.01, 0.001])
    

    start_time = time.time()
    while (not shutdown_flag) and (time.time() - start_time < cin.total_duration):

        # determine the control time this loop has been running
        time_in_loop     = time.time() - start_time

        # update the joint positions laura.q and joint velocity values laura.dq
        laura.read_sensors()
        
        # compute the torques acting on the joints because of friction. 
        b = dmx.predict_friction(laura.dq)        

        # compute the torques acting on the joints because of gravity 
        G = dmx.predict_gravity(laura.q, ee_mass=17.5, ee_com_radius=20)

        # compute the desired joint positions and joint velocities, 
        # based on c-spline interpolation, for the current point in time.
        q_des, dq_des, _ = cin.get_values_for_time(time_in_loop)

       # store the desired and the actual configuration for plotting
        desired_q_trajectory.append(q_des)
        actual_q_trajectory.append(laura.q)

        # model-based feedback PD controller in configurational space
        tau = kp * (q_des - laura.q) + kv * (dq_des - laura.dq) -b -G

        # apply joint torques as specified above
        laura.set_torque(tau)

    # --- while

    plot_q_trajectory(desired_q_trajectory, actual_q_trajectory)

    # properly shutdown LAURA to prevent errors
    laura.disable()

# --- main



if __name__ == "__main__":
    main_task_b()



