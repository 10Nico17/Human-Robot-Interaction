import numpy as np
from april_detection import  *


# initialize object for easy april tag detection
april = AprilDetection(cam_index=0)


# Every April Tag marker has its own ID
LAURA    = 0
CUBE     = 7
CYLINDER = 10


# Homogenous transformations representing the base frame [0] relative to the marker frame [M]
T_M_0 = {

    # LAURA's base lies directly under its zero joint and on the level of the table's surface
    LAURA    : np.array([[1., 0., 0.,  30.],
                         [0., 1., 0., -35.],
                         [0., 0., 1., -29.],
                         [0., 0., 0.,   1.],]),
    
    # the box's bases lies at its center. The box is 24 mm heigh.
    CUBE     : np.array([[1., 0., 0.,   0.],
                         [0., 1., 0.,   0.],
                         [0., 0., 1., -12.],
                         [0., 0., 0.,   1.],]),

    # the zylinder's bases lies at its center. The cylinder is 30 mm heigh.
    CYLINDER : np.array([[1., 0., 0.,   0.],
                         [0., 1., 0.,   0.],
                         [0., 0., 1., -15.],
                         [0., 0., 0.,   1.],]),
}



def detect_cube() -> np.ndarray:
    '''
    Returns the 3D position of the cube, relative to LAURA's base frame, based on april tag markers. 
    Returns None, if the object's marker or LAURA's marker were not detected.
    '''

    # list of object IDs that we want need to detect and transform
    objects = [LAURA, CUBE]

    # dict of homogeneous transformations, representing april marker frames relative to the camera frame
    T_C_M = april.detect()
        
    # break up if cube or laura were not detected
    if np.any([ID not in T_C_M for ID in objects]): return None

    # compute base frames relative to the camera frame for each object
    T_C_0 = {}
    for ID in objects: T_C_0[ID] = T_C_M[ID] @ T_M_0[ID]

    # compute the cube's base frame relative to LAURA's base frame
    T_0LAURA_CUBE = np.linalg.pinv(T_C_0[LAURA]) @ T_C_0[CUBE]

    # extract position
    cube_position = T_0LAURA_CUBE[:3, 3]

    return cube_position
# ---
