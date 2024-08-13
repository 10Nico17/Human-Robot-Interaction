'''
--------------------
NULLSPACE PROJECTION
--------------------


TASK B - Automated Grasping
---------------------------

What this task is about:
    * Automatically detect the pose of the cube relative to LAURA's base frame
    * Automatically compute a trajectory to pick up the cube and place it somewhere else
    * Follow the trajectory using inverse kinematics
    * Avoid reaching joint limits using nullspace projection

What you need to do in the real world:
    * Take out the four base plate pieces and put them together.
    * Attach LAURA to the base blate in 3D mode. 
    * Connect LAURA to a power supply and to your computer via USB
    * Attach the Gripper module
    * Change the baudrate of the gripper motor to 1000000 via the DynamixelWizard
    * Make sure the ID of the gripper motor is set to 4 via the DynamixelWizard
    * Attach the camera to the camera stand and place it so that it looks over LAURA's shoulder
    * Connect the camera to your system
    * Take out the cube object and place it somewhere in front of LAURA within its workspace

What you need to implement:
    * implement the missing code in object_detection.py
    * Implement the missing code in this file.
    * Things will likely go wrong in the beginning, so better keep a second terminal open for quick recalibration

What you should see if your code is correct:
    * Laura's end effector should move on a straight line to a desired end effector position.
    * After LAURA finished moving, a plot should appear, comparing LAURA's desired and actual end effector trajectory. 
    * There will always be minor oscillations, but try to minimize them as much as possible.
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
import object_detection as obd
import keyboard




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

'''
The following callback function is executed when the DOWN key is pressed.
It will initiate the object detection and trajectory generation processes.
'''

# global variables
laura               = None
x                   = None
gripper_position    = None
grasping            = False
grasping_start_time = None
grasping_trajectory = None



def initialize_grasping(event):
    global x, gripper_position
    global grasping_trajectory, grasping_start_time, grasping

    print('Initialize grasping process...')

    # motor values of gripper motor (id 4) for open and closed state
    gripper_pos_open   = 2200.
    gripper_pos_closed = 500.

    object_goal_position = np.array([50, 100, 0])

    # detect object
    print('Detect object ...')
    object_position = obd.detect_cube()

    # skip if object was not detected
    if object_position is None:
        grasping = False
        print('ERROR: Could not detect LAURA and cube!')
        return
    else:
        print('Object found at position:', object_position)
    
    # tell LAURA to start grasping the cube
    grasping = True

    # remember the time we started the grasping process
    grasping_start_time  = time.time()


    # We compute create a trajectory in the combined space of end effector poses and gripper motor positions.
    # Therefore, we operate in a 7D space: [x, y, z, roll, pitch, yaw, gripper_motor]
    print('Initialize grasping trajectory...')

    lift = 70 # [millimeters]
    kf_start                = np.array([x[0], x[1], x[2], x[3], x[4], x[5], gripper_position])
    
    '''<Interpret these keyframes and fill-in the rest!>'''
    kf_above_object_open    = np.array([object_position[0], object_position[1], object_position[2]+lift, np.pi, 0, 0, gripper_pos_open])
    kf_at_object_open       = np.array([object_position[0], object_position[1], object_position[2],      np.pi, 0, 0, gripper_pos_open])
    kf_at_object_closed     = np.array([object_position[0], object_position[1], object_position[2],      np.pi, 0, 0, gripper_pos_closed])
    kf_above_object_closed  = np.array([object_position[0], object_position[1], object_position[2]+lift, np.pi, 0, 0, gripper_pos_closed])

    kf_above_goal_closed    = np.array([object_goal_position[0], object_goal_position[1], object_goal_position[2]+lift,  np.pi, 0, 0, gripper_pos_closed])
    kf_at_goal_closed       = np.array([object_goal_position[0], object_goal_position[1], object_goal_position[2],       np.pi, 0, 0, gripper_pos_closed])
    kf_at_goal_open         = np.array([object_goal_position[0], object_goal_position[1], object_goal_position[2],       np.pi, 0, 0, gripper_pos_open])
    kf_above_goal_open      = np.array([object_goal_position[0], object_goal_position[1], object_goal_position[2]+lift,  np.pi, 0, 0, gripper_pos_open])

    kf_end                  = kf_start   


    keyframes = np.array([kf_start, kf_above_object_open, kf_at_object_open, kf_at_object_closed, kf_above_object_closed, kf_above_goal_closed, kf_at_goal_closed, kf_at_goal_open, kf_above_goal_open, kf_end])
    durations = np.array([         2,                    1,                 1,                   1,                      2,                    2,                 1,               1,                  2])

    # compute the two trajectories - the first for approaching the object and the second for moving to the final pose
    grasping_trajectory = CubicInterpolation(keyframes, durations)

    print('Start grasping...')
# ---


# Register callback function for catching down key
keyboard.on_press_key("down", initialize_grasping)

'''
---------------------------------------------------------------------------
'''





def main():

    # global flag for shutting down the control loop
    global shutdown_flag

    # global variables for automatic trajectory generation
    global x, gripper_position
    global grasping_trajectory, grasping_start_time, grasping

 

    # Initialize LAURA
    # ----------------
                     
    # setup LAURA - you need to change the usb device name!
    laura = LAURA(usb_device_name='/dev/tty.usbmodem21401', baudrate=1000000, mode=MODE_3D)

     # tune the gains of the dynamixels' on-board C-Space PID controller
    laura.set_gains(p=900, i=200, d=500)

    # step size: scalar value for adjusting the size of the error rejection step
    gamma = 1



    # INITIALIZE GRIPPER
    # ------------------

    # specify the length of the closed gripper [millimeters] 
    GRIPPER_MODULE_LENGTH = 92.

    # specify ID of gripper's dynamixel servomotor
    GRIPPER_MOTOR_ID = 4

    # operate the gripper module in current-based position control mode
    laura.dxl.set_operating_mode(GRIPPER_MOTOR_ID, CURRENT_POSITION_CONTROL_MODE)

    # set gripper's maximum current rather low to achieve high compliance
    laura.dxl.set_goal_current(GRIPPER_MOTOR_ID, 1000)

    # enable gripper
    laura.dxl.set_torque_enabled(GRIPPER_MOTOR_ID, True)



    # Control Loop
    # ------------

    while (not shutdown_flag):

        # update the joint positions laura.q and joint velocity values laura.dq
        laura.read_sensors()
        
        # read out the current grupper position
        gripper_position = laura.dxl.get_pos([GRIPPER_MOTOR_ID], multi_turn=True)[0]
 
        # compute end effector pose based on the current configuration
        x = ktx.get_ee_pose(laura.q, GRIPPER_MODULE_LENGTH, mode=MODE_3D)

        if grasping:

            # determine the time since starting the grasping process
            time_since_start = time.time() - grasping_start_time            

            # get desired end effector pose based on c-spline interpolation
            des, _, _ = grasping_trajectory.get_values_for_time(time_since_start)
            
            # extract desired end effector pose and gripper position
            x_des         = des[:-1]
            gripper_des   = des[-1] 

            # stop grasping if time is up
            if time_since_start > grasping_trajectory.total_duration:
                grasping = False
        else:

            # tell LAURA to stay as it is
            x_des       = x
            gripper_des = int(gripper_position)
        # -- if


        # compute control step in configuration space
        delta_q = ctrl.compute_delta_q(x_des, laura.q, gamma, GRIPPER_MODULE_LENGTH, avoid_joint_limits=True)

        # tell LAURA to go to a certain configuration by calling a DXL-based PID controller
        laura.set_configuration(laura.q + delta_q)

        # set gripper to desired position
        laura.dxl.set_goal_pos([GRIPPER_MOTOR_ID], gripper_des)

    # --- while



    # Shut-Down Procedure
    # -------------------

    # properly shutdown LAURA to prevent errors
    laura.disable()

# --- main



if __name__ == "__main__":
    main()



