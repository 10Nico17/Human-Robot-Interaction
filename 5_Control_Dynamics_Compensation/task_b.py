'''
-------------------
MODEL-BASED CONTROL
-------------------


TASK B - Feedforward Controller : Gravity Compensation
------------------------------------------------------

What this task is about:
    There are many dynamics acting on the joints. 
    One important aspect of dynamics is gravity. The robot's links would fall towards the ground if we do not compensate for gravity.
    We can model the effect of gravity in a simplified way, by calculating the links' center of masses and calculating their effect on 
    the joints due to gravity. With this model, we can cancel out gravity.

What you need to do in the real world:
    * Take out the four base plate pieces and put them together.
    * Attach LAURA to the base blate in 3D mode. 
    * Connect LAURA to a power supply and to your computer via USB
    * Attach the April Modul to the end of the robot arm
    
What you need to implement:
    * Have a look at the file dynamics.py and implement the function 'predict_gravity()'

What you should see if your code is correct:
    * LAURA's joints will not fall to the ground any longer. The robot acts alsmost as if there was no gravity.
'''




from laura_interface import *
from dynamixel_port import *
import dynamics as dnx
import numpy as np
import signal




'''
---------------------------------------------------------------------------
The folloing code ensures a proper shut-down of this script, when the user presses CTRL-C on the keyboard. 
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






def main_task_b():

    # make sure we know about the flag's status inside this function.
    # its values is changed by a callback function outside this function.
    global shutdown_flag
  
    # setup LAURA - you need to change the usb device name!
    laura = LAURA(usb_device_name='/dev/tty.usbmodem2101', mode=MODE_3D, auto_enable=False)

    # we now operate LAURA by defining the torques in its joints. 
    # under the hood, this is realized through adjustments of the electical current in the motors.
    laura.set_operating_mode(CURRENT_CONTROL_MODE)
    

    while not shutdown_flag:
    
        # this function call updates the joint position values laura.q 
        # and also the joint velocity values laura.dq
        laura.read_sensors()
        
        # compute the torques acting on the joints because of gravity 
        G = dnx.predict_gravity(laura.q, ee_mass=17.5, ee_com_radius=20)

        # compute the torques acting on the joints because of friction. 
        b = dnx.predict_friction(laura.dq)        

        # compensate for friction and gravity
        tau = -b -G

        # apply joint torques as specified above
        laura.set_torque(tau)

    # --- while
    

    # properly shutdown LAURA to prevent errors
    laura.disable()

# --- main






if __name__ == "__main__":
    main_task_b()


