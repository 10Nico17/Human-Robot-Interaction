'''
---------------------------
Homogeneous Transformations
---------------------------

TASK A - Getting started
------------------------


The aim of this exercise is to get the the april tag detection running on your system.

The only thing requested of you is that you are able to run this code see detected reference frames.
Please read the code carefully and try to understand everything. 

If this code exceeds your understanding of the python programming language, you are in sereous trouble and you should contact your class supervisor (Steffen).
'''



# First things first: Import the custom python API for april tag detection, called AprilDetection
# If this following line of code causes an error, you probably did not correctly install the april-detection python package.
from april_detection import *


# Second, we need to create an object from the AprilDetection class. 
# This object serves two purposes: first, connecting to the USB camera and receiving camera images, and second, detecting april tags.
# During the creation process of this object (i.e., inside the constructor method), a connection to the camera will be established.
# Potentially, there could be several cameras connected to your system (e.g. a USB camera and your notebook's internal camera).
# These cameras have different indices, starting with index 0. 
# You need to try out various indices (i.e., change the value of cam_index) to find your camera of choice.
april = AprilDetection(cam_index=0)


while True:

    # This function call takes a new camera image and detects april tags in it.
    # The function returns a list of april tag indices ids:[int] and a list of homogeneous transformations, relative to the camera frame Ts:[np.ndarray].
    ids, Ts = april.detect()

    # The following function call displays the camera image and the detected reference frames
    april.show()

    # you can terminate this loop with [Ctrl-C]
# ---










