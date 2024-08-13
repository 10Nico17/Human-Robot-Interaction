'''
-------------------
MODEL-BASED CONTROL
-------------------


TASK A - Feedforward Controller : Friction Compensation
-------------------------------------------------------

What this task is about:
    There are many dynamics acting on the joints. One of these is friction. It is very difficult, if not impossible, to model.
    But if we can come up with a fairly good prediction of how friction acts on the joints, then we can apply a torque
    so that friction is cancelled out. Ideally, we will have a robot that behaves as if there was no friction at all. But this is close to impossible.
    Nonetheless, it makes sense to compensate for friction as good as we can.

What you need to do in the real world:
    * Connect LAURA to a power supply and to your computer via USB
    
What you need to implement:
    * Have a look at the file dynamics.py and implement the function 'predict_friction()'

What you should see if your code is correct:
    You will hopefully be able to recognize that LAURA's joints activly support you, when you manually rotate them.
    Thus, it will seem as if there is less friction in the joint's gear boxes.
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




def main_task_a():

    # make sure we know about the flag's status inside this function.
    # its values is changed by a callback function outside this function.
    global shutdown_flag
  
    # setup LAURA - you need to change the usb device name!
    laura = LAURA(usb_device_name='/dev/tty.usbmodem101', mode=MODE_3D, auto_enable=False)

    # we now operate LAURA by defining the torques in its joints. 
    # under the hood, this is realized through adjustments of the electical current in the motors.
    laura.set_operating_mode(CURRENT_CONTROL_MODE)


    while not shutdown_flag:
        
        # this function call updates the joint position values laura.q 
        # and also the joint velocity values laura.dq
        laura.read_sensors()

        # compute the torques acting on the joints because of friction. 
        b = dnx.predict_friction(laura.dq)

        # compensate for friction in the joints
        tau = -b

        # apply joint torques as specified above
        laura.set_torque(tau)

    # --- while


    # properly shutdown LAURA to prevent errors
    laura.disable()

# --- main



if __name__ == "__main__":
    main_task_a()


