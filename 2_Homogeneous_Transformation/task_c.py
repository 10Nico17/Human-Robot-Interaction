'''
---------------------------------------------
Fun with Frames - Homogeneous Transformations
---------------------------------------------

TASK C - Relative representation of reference frames
----------------------------------------------------

Background:
    Any position and any orientation is alwyas defined relative to some reference frame.
    Therefore, also any pose (combination of position and orientation) is always defined relative to some reference frame.

Basic idea:
    So far, we always represented a pose in 3D space as a 6D vector: [x, y, z, roll, pitch, yaw].
    But now, we do not do this anymore.

    Instead, we represent a pose as a reference frame. 
    More specifically, the orientation of the pose is represented by the three axes of the frame (the base vectors)
    and the position of the pose is represented by the frame's point of origin. 
    Both the frame's axes and its point of origin are defined relative to some other reference frame.
    Got it? If not, read again or ask!

Groundbreaking idea:
    We represent a pose as a homogeneous transformation. Mind-blow!
    Because this type of matrix encodes both translation and rotation.

What you should learn here:
    The aim of this exercise is first, to show you how to represent a reference frame relative to another frame.
    And second, to show you how to map a frame, so it is represented relative to a different frame than before.

What you need to prepare in the physical world (in reality):
    Take frame [A] out of the box and palce it on your desk.
    Connect the USB camera module with your system. Make sure the camera is able to see the frame.

What you should see if you did everything right:
    A reference frame at your april tag and a second reference frame (which you will have constructed yourself).
    This second frame sould be consistently translated and rotated relative to the april tag's frame.
'''


import numpy as np
from april_detection import *

april = AprilDetection(cam_index=0)



# Task C.1:
# Define a homogeneous transformation that realizes a rotation of 90 degrees about the z-axis 
# and a translation of 50 millimeters along the x-axis and a translation of 70 mm along the y-axis
# This new homogenous transformation is defined relative to frame [A].
#
# Remember the general shape of a homogeneous transformation:
# [R | t]
# [_____]
# [0   1]

T_AB = np.array([[0., -1., 0., 50.],
                 [1.,  0., 0., 70.],
                 [0.,  0., 1.,  0.],
                 [0.,  0., 0.,  1.],])



while True:

    # Returns a list of april tag indices ids:[int] and a list of homogeneous transformations, relative to the camera frame Ts:[np.ndarray].
    ids, Ts = april.detect()

    # execute the rest of the code only if at least one april tag was detected
    if len(ids) == 1:

        # Task C.2
        # Extract the frame [A] from the list of homogeneous transformations Ts
        # Remember that frame [A] is defined relative to the camera frame [C]
        T_CA = Ts[0]

        # Task C.3
        # Map frame [B] to the camera frame [C]. 
        # Before, frame [B] was represented as a homogeneous transformation reative to frame [A].
        # After this mapping, frame [B] will be represented relative to the camera frame [C]
        T_CB = T_CA @ T_AB
        
        # Display the camera image, the detected april tags, and the new frame T_CB
        april.show(additional_frames=[T_CB])
    
    else:
        # Display only the camera image without any frame if no april tag was detected.
        april.show(additional_frames=[])
# ---




