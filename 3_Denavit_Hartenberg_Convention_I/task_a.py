'''
------------------
Forward Kinematics
------------------


TASK A - Getting started
------------------------


The aim of this exercise is to get the the laura-interface running on your system.
For this, you need to have installed everything shown in the video (i.e., dynamixel-port, linear-interpolation, numpy, matplotlib, seaborn).

This simple code example demonstrates how to connect to the LAURA robot via a USB serial connection,
and how to read-out the current joint configuration, both in radians and in motor values [between 0 and 4095]

Your task is to make sure that this code is running and you can see the robot's configuration for 10 seconds.
Read the code and try to understand everything.
'''


from laura_interface import *
from laura_utils import *
import time



# create a serial (USB) connection to the LAURA robot
# you need to change the name of the usb device. You can use the dynamixel wizard to find out this name.
laura = LAURA(usb_device_name='/dev/tty.usbmodem21401', mode=MODE_2D)

# disable motors so you can manually move them 
laura.disable()


loop_duration = 10 # in seconds
start_time    = time.time()

# repeat the following code for <loop_duration> seconds 
while time.time() - start_time < loop_duration:

    # update sensor readings (this updates the laura.q vector)
    laura.read_sensors()

    # by default, the configuration is represented in radians
    print('Configuration [radians]: ', laura.q)

    # to convert the configuration into motor values, we can use a function from the laura_utils library:
    print('Configuration [motor val]', radians_to_motor_values(laura.q), '\n')

# shut down laura
laura.disable()