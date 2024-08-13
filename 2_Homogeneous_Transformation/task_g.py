'''
---------------------------------------------
Fun with Frames - Homogeneous Transformations
---------------------------------------------

TASK G - Lucy's in the sky with diamonds
----------------------------------------

Awesome, you did it! You mastered homogeneous transormations! Congratulations! Wooohoo!

As a reward, you now may draw amazing 3D pictures by using what you've learned.
Have a look at the code below and update the data matrix (which encodes the image).
'''


import numpy as np
from april_detection import *



april = AprilDetection()


while april.is_running:
    
    ids, Ts = april.detect()

    if len(Ts) > 0:

        # Define image:
        # Every column vector is a point.
        # Consecutive points are connected by a line.
        # All vectors in this data matrix are augmented with an additional one (last row)
        l = 50 # in cm
        image = np.array([np.array([0, 0, 1, 1, 0.5, 0, 1, 0, 1]) * l,
                          np.array([0, 1, 0, 1, 1.5, 1, 1, 0, 0]) * l,
                          np.array([0, 0, 0, 0,   0, 0, 0, 0, 0]) * l,
                          np.array([1, 1, 1, 1,   1, 1, 1, 1, 1])])


        # project points to camera frame
        T_CI = Ts[0] @ image
    
        april.show(additional_points=T_CI, show_markers=False)

    
    else:
        april.show(show_markers=False)


# --- while
