'''
-------------------
MODEL-BASED CONTROL
-------------------


TASK C - Model-Based Feedback Control
-------------------------------------

What this task is about:
    Feedback control allows the robot to minimize errors. This is necessary, because our dynamics models are not perfect. And they never will be.
    Thus, there will always be an error. Furthermore, by moving the desired configuration, this error compensation also allows us to let the robot move
    in a desired way.

What you need to do in the real world:
    * Take out the four base plate pieces and put them together.
    * Attach LAURA to the base blate in 3D mode. 
    * Connect LAURA to a power supply and to your computer via USB
    * Attach the April Modul to the end of the robot arm
    
What you need to implement:
    * Implement the code in this file.

What you should see if your code is correct:
    You will two reference frames:
    the first is detected by the camera via the April Modul
    and the second is computed by the forward kinematics with DH parameters.
    Both frames should be very similar (expect minor error due to camera distortions and manufacturing-based inaccuracies).
'''



from laura_interface import *
from dynamixel_port import *
import numpy as np
import dynamics as dnx
import signal
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
The following callback functions are executed when the UP key or the DOWN key is pressed.
It will change the desired configuration of the robot. 
Thus, this allows us to control the robot with the keyboard.
'''

# this will be our desired robot configuration
q_des = None

def up_callback(event):    q_des[1] = q_des[1] + np.pi/10.
def down_callback(event):  q_des[1] = q_des[1] - np.pi/10.    

# Register callback function for catching space bar hit
keyboard.on_press_key("up", up_callback)
keyboard.on_press_key("down", down_callback)

'''
---------------------------------------------------------------------------
'''







def main_task_c():

    # make sure we know about the flag's status and the desired configuration inside this function.
    # their values are changed by a callback functions outside this function.
    global shutdown_flag, q_des

    # setup LAURA - you need to change the usb device name!
    laura = LAURA(usb_device_name='/dev/tty.usbmodem1101', mode=MODE_3D)

    # at the beginning, LAURA moves to the zero configuration
    laura.move_to_zero(duration=2)

    # we now operate LAURA by defining the torques in its joints. 
    # under the hood, this is realized through adjustments of the electical current in the motors.
    laura.set_operating_mode(CURRENT_CONTROL_MODE)

    # at the beginning, the desired configuration is identical to LAURA's current configuraion
    laura.read_sensors()
    q_des = laura.q

    '''
    ATTENTION!
    LAURA's gains are very(!) small. Start with values around 1.0 for kp and around 0.001 for kv,
    then gradually increase or decrease them, depending on the robot's behavior.
    '''
    # specify the propotional and derivative gains for the PD controller
    kp = np.array([0.5, 0.5, 0.5, 0.5]) 
    kv = np.array([0.003, 0.003, 0.003, 0.003])
    
    
    while not shutdown_flag:

        # this function call updates the joint position values laura.q 
        # and also the joint velocity values laura.dq
        laura.read_sensors()
        
        # compute the torques acting on the joints because of gravity 
        G = dnx.predict_gravity(laura.q, ee_mass=17.5, ee_com_radius=20)

        # compute the torques acting on the joints because of friction. 
        b = dnx.predict_friction(laura.dq)        

        # model-based feedback PD controller: reduces a configurational error 
        # (i.e., the difference between the desired and actual configuration)
        # while at the same time, compensates for friction and gravity
        tau = kp * (q_des - laura.q) - kv * laura.dq -b -G

        # apply joint torques as specified above
        laura.set_torque(tau)

        print('q_des', q_des - laura.q)
    # --- while


    # properly shutdown LAURA to prevent errors
    laura.disable()

# --- main



if __name__ == "__main__":
    main_task_c()


