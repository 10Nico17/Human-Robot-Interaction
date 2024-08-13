'''
---------------------------------------------
Fun with Frames - Homogeneous Transformations
---------------------------------------------

TASK D - Computing LAURA's base frame
-------------------------------------

Background:
    We can use homogenous transformations to easily calculate a pose, based on some known reference frame.

What you should learn here:
    You will apply the insights from the previous exercises to something that is actually somewhat useful...
    More specifically, you will compute LAURA's base frame, based on the pose of its april tag.
    Hopefully you will find out that this is actually quite easy and pretty cool.

What you need to prepare in the physical world (in reality):
    Take LAURA out of the box and place it on the table.
    Connect the USB camera module with your system. Make sure the camera is able to see LAURA's april tag.

What you should see if you did everything right:
    A reference frame at the base of LAURA, positioned right under joint 0, on the same level as the surface of the table.
'''


import numpy as np
from april_detection import *

april = AprilDetection(cam_index=0)




# Task D.1:
# Define a homogeneous transformation that represents LAURA's base [0] frame, relative to its april tag frame [L].
# The base frame lies 30 mm to the right (positive x), 35 to the back (negative y), and 29 mm to the bottom (negative z) of the april tag.
# The base frame has the same orientation as the april tag frame.
#
# Remember the general shape of a homogeneous transformation:
# [R | t]
# [_____]
# [0   1]

T_L0 = np.array([[1.,  0., 0.,  30.],
                 [0.,  1., 0., -35.],
                 [0.,  0., 1., -29.],
                 [0.,  0., 0.,  1.],])



while True:

    # Returns a list of april tag indices ids:[int] and a list of homogeneous transformations, relative to the camera frame Ts:[np.ndarray].
    ids, Ts = april.detect()

    # execute the rest of the code only if at least one april tag was detected
    if len(ids) == 1:

        # Task D.2
        # Extract LAURA's april tag frame [L] from the list of homogeneous transformations Ts
        # Remember that LAURA's april tag frame [L] is represented relative to the camera frame [C].
        T_CL = Ts[0]

        # Task D.3
        # Map the base frame [0] to the camera frame [C]. 
        # Before, LAURA's base frame [0] was represented relative to its april tag frame [L].
        # After this mapping, frame [0] will be represented relative to the camera frame [C]
        T_C0 = T_CL @ T_L0
        
        # Display the camera image and LAURA's base frame [0]
        april.show(show_markers=False, additional_frames=[T_C0])
    
    else:
        # Display only the camera image without any frame if no april tag was detected.
        april.show(show_markers=False, additional_frames=[])
# ---




