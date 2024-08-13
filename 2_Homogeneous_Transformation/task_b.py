'''
---------------------------------------------
Fun with Frames - Homogeneous Transformations
---------------------------------------------

TASK B - Relative representation of vectors
-------------------------------------------

Background:
    Any vector is defined / represented relative to some reference frame. 
    Usually, a vector is illustrated as an arrow. 
    The start of this arrow is located at the origin of the reference frame the arrow is defined in.
    The end of this arrow is located at the point encoded by the values of the vector. Again, this point is expressed relative to a reference frame.

What you should learn here:
    The aim of this exercise is to show you that vectors are defined relative to reference frames.
    And in case these frames move themselves, also the vectors defined in it move.

What you need to prepare in the physical world (in reality):
    Take frame [A] out of the box and palce it on your desk.
    Connect the USB camera module with your system. Make sure the camera is able to see the frame.

What you should see if you did everything right:
    A reference frame at your april tag and a red arrow representing a vector.
    If you move the reference frame, the arrow should move accordingly.
'''



import numpy as np
from april_detection import *

april = AprilDetection(cam_index=0)



# Task B.1
# Create a vector with the values (30, 70, 0). 
# Since we want to use this vector with homogeneous transformations, it should have an additional augmented one.
vec = np.array([30, 70, 0])



while True:

    # Returns a list of april tag indices ids:[int] and a list of homogeneous transformations, relative to the camera frame Ts:[np.ndarray].
    ids, Ts = april.detect()

    # execute the rest of the code only if at least one april tag was detected
    if len(ids) == 1:

        # Task B.2
        # Extract the frame [A] from the list of homogeneous transformations Ts
        # Remember that frame [A] is defined relative to the camera frame [C]
        T_CA = Ts[0]

        # Display the camera image, the detected april tag frames, and the vector relative to frame [A]
        april.show(additional_vectors=[[vec, T_CA]])
    
    else:
        # Display only the camera image without any frame if no april tag was detected.
        april.show(additional_frames=[])
# ---




