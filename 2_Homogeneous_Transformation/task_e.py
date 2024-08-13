'''
---------------------------------------------
Fun with Frames - Homogeneous Transformations
---------------------------------------------

TASK E - Mapping vectors between frames
---------------------------------------

Background:
    Any vector is defined / represented relative to some reference frame. 
    Usually, a vector is illustrated as an arrow. 
    The start of this arrow is located at the origin of the reference frame the arrow is defined in.
    The end of this arrow is located at the point encoded by the values of the vector. Again, this point is expressed relative to a reference frame.

The goal:
    We want to change the reference frame, relative to which the vector is represented.
    In other words, we want to change the start of the arrow. However, the end of the arrow should, in the end, point at the identical location. 
    To accomplish this, the values of the vector need to change in a specific way.

Groundbreaking idea:
    We use a homogeneous transformation to map a vector from one frame to another frame.

Explanation:
    A homogeneous transformation can represent a reference frame B relative to another reference frame A (as you hopefully learned in a previous task).
    Thus, the position and orientation of B are defined relative to frame A.
    
    Okay, here comes the magic: 
    If a vector is defined relative to B, we can use the homogeneous transformation T_AB (which expresses frame B relative to A)
    to map the vector that was defined in frame B so that it is now defined in frame A. 
    I hope you closely followed the lecuture! Because otherwise, this proably makes no sense at all.

What you should learn here:
    Homogeneous transformations cannot only be used to express frames relative to other frames,
    but they can also be used to map vectors (or even frames) between frames. 

What you need to prepare in the physical world (in reality):
    Take both frame [A] and [B] out of the box and place them on the desk.
    Connect the USB camera module with your system. Make sure the camera is able to see both frames.

What you should see if you did everything right:
    Two reference frames, one for frame [A] and one for frame [B].
    There should be a vector defined in [A] and another vector defined in [B]
    Both vectors should point at the same spot (because they are the same, just expressed in different frames! - Magic!).
'''



# Every April Tag geometry has its own ID
# For the frames [A] and [B], these are their IDs
ID_FRAME_A = 20
ID_FRAME_B = 23

import numpy as np
from april_detection import *

april = AprilDetection(cam_index=0)



# Task E.1
# Define a vector [30, 40, 0] that is defined in frame [B].
# We want to use this vector for homogeneous transformations, so it needs to be augmented with an additional one.
v_B = np.array([30., 40., 0., 1.])


while True:

    # Returns a list of april tag indices ids:[int] and a list of homogeneous transformations, relative to the camera frame Ts:[np.ndarray].
    ids, Ts = april.detect()

    # execute the rest of the code only if at least two april tags were detected
    if len(ids) > 1:

        # Task E.2:
        # Assign the correct homogeneous transformation from the list Ts to T_CA and T_CB.
        # For this, you probably want to take a look at their corresponding IDs (see above).

        if ids[0] == ID_FRAME_A:
            T_CA = Ts[0]
            T_CB = Ts[1]
        else:
            T_CA = Ts[1]
            T_CB = Ts[0]

        # Task E.3:
        # Obtain the homogeneous transformation T_AB that represent frame [B] relative to frame [A]
        T_AB = np.linalg.pinv(T_CA) @ T_CB

        # Task E.4:
        # Map the vector from frame B to frame A. 
        v_A = T_AB @ v_B
        
        # Display the camera image, the detected april tag frames, 
        # and the vectors v_A and v_B, relative to the frames they are defined in
        april.show(additional_vectors=[[v_A, T_CA], [v_B, T_CB]])
    
    else:
        # Display only the camera image without any frame if no april tag was detected.
        april.show(additional_frames=[])
# ---