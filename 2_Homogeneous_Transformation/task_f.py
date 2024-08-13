'''
---------------------------------------------
Fun with Frames - Homogeneous Transformations
---------------------------------------------

TASK F - Mapping everything to the base frame, like a real roboticist
---------------------------------------------------------------------


The goal:
    You will apply the insights from the previous task on a real-world application.

    1.) Compute LAURA's base frame from its april tag frame 
    2.) Compute the object frames relative to their april tag frames
    3.) Compute the poses of the objects relative to LAURA's base frame    
    This is how roboticists, engineers and scientists around the world are actually doing this kind of stuff.   

What you should learn here:
    Homogeneous transformations are awesome and incredibly useful!
    Say it three times out loud! And never forget!

What you need to prepare in the physical world (in reality):
    Take the two objects (the cylinder and the cube) out of the box and place them on the desk with their april tags facing upwards.
    Take LAURA out of the box and place it on the desk.
    Connect the USB camera module with your system. Make sure the camera is able to see both objects and LAURA.

What you should see if you did everything right:
    Three reference frames: one at the base of the cube, one at the base of the cylinder, one at LAURA's base.
    There should be two vectors: one pointing at the cube and one pointing at the cylinder. Both vectors are defined in LAURA's base frame
'''



# Every April Tag geometry has its own ID
ID_FRAME_LAURA    = 0
ID_FRAME_CUBE     = 7
ID_FRAME_CYLINDER = 10

import numpy as np
from april_detection import *

april = AprilDetection(cam_index=0)




# Task F1:
# Fill out the following dictionary.
# A dictionary is a data type that let's you access data with a so-called key.
# Here, the key is a string (for example 'laura'). 
# Each key maps to a specific value. In our case, each key maps to a homogeneous transformation matrix.
# Google python dictionaries if you have doubts or questions. Or just ask... 

# Homogenous transformations for computing the base relative to the corresponding april tag
T0 = {
    # LAURA's base lies directly under its zero joint and on the level of the table's surface
    ID_FRAME_LAURA    : np.array([[1., 0., 0.,  30.],
                                  [0., 1., 0., -35.],
                                  [0., 0., 1., -29.],
                                  [0., 0., 0.,   1.],]),
    
    # the box's bases lies at its center and on the level of the table's surface. The box is 24 mm heigh.
    ID_FRAME_CUBE     : np.array([[1., 0., 0.,   0.],
                                  [0., 1., 0.,   0.],
                                  [0., 0., 1., -24.],
                                  [0., 0., 0.,   1.],]),

    # the zylinder's bases lies at its center and on the level of the table's surface. The cylinder is 30 mm heigh.
    ID_FRAME_CYLINDER : np.array([[1., 0., 0.,   0.],
                                  [0., 1., 0.,   0.],
                                  [0., 0., 1., -30.],
                                  [0., 0., 0.,   1.],]),
}


# helper function for returning the index of an element inside a list
def get_index(element, list): return np.argmax(list == element)


while True:

    # Returns a list of april tag indices ids:[int] and a list of homogeneous transformations, relative to the camera frame Ts:[np.ndarray].
    ids, Ts = april.detect()
    
    # execute the rest of the code only if three april tags have been detected
    if len(ids) == 3:

        # tries out the code below, but does not break up if it fails 
        try:

            # Task F.2:
            # Assign the correct homogeneous transformation from the list Ts to T_CBox, T_CCylinder, and T_CLaura.
            # These frames specify the homogeneous transformations of the corresponding april tags relative to the camera frame [C]
            # Tipp: use the helper function get_index()
            T_CBox      = Ts[get_index(ID_FRAME_CUBE,     ids)]
            T_CCylinder = Ts[get_index(ID_FRAME_CYLINDER, ids)]
            T_CLaura    = Ts[get_index(ID_FRAME_LAURA,    ids)]

            # Task F.3:
            # Compute the base frame for the cube, for the cylinder, and for laura.
            # For this, use the homogeneous transformations in the dictionary 
            T_C0Box      = T_CBox      @  T0[ID_FRAME_CUBE]
            T_C0Cylinder = T_CCylinder @  T0[ID_FRAME_CYLINDER]
            T_C0Laura    = T_CLaura    @  T0[ID_FRAME_LAURA]

            # Task E.4:
            # Map the frames of the two objects (box and cylinder) so that they are defined relative to LAURA's base frame
            T_00Box      = np.linalg.pinv(T_C0Laura) @ T_C0Box
            T_00Cylinder = np.linalg.pinv(T_C0Laura) @ T_C0Cylinder

            # Task E.5:
            # Extract the two vectors v_00Box and v_00Cylinder.  
            # These vectors are defined relative to LAURA's base frame 
            # and point to the base of the box and to the center of the cylinder, respectively.
            v_0Box      = T_00Box[:3, 3]
            v_0Cylinder = T_00Cylinder[:3, 3]
            
            # Display the camera image, the detected april tag frames, 
            # and the vectors v_A and v_B, relative to the frames they are defined in
            #april.show(show_markers=False, additional_frames=[T_00Box, T_00Cylinder], additional_vectors=[[v_0Box, T_C0Laura], [v_0Cylinder, T_C0Laura]])
            april.show(show_markers=False, additional_vectors=[[v_0Box, T_C0Laura], [v_0Cylinder, T_C0Laura]])
        except:
             pass
    else:
            # Display only the camera image without any frame if no april tag was detected.
            april.show(show_markers=False)
# ---