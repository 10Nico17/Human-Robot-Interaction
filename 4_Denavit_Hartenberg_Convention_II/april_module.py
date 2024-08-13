'''
APRIL MODULE

The April Module possesses four different april tag markers. 
This ensures that its pose can be determined from various camera perspectives.
However, these four markers are not located at the center of the April Module, but on its faces.

Here, you have to implement a function that infers the pose of the April Module,
based on the detected april tag markers.
'''


import numpy as np
from april_detection import *


# april tag marker IDs on the April Module
APRIL_LEFT    = 12
APRIL_FRONT   = 13
APRIL_TOP     = 15
APRIL_RIGHT   = 14

MODULE_LENGTH = 30.5 # millimeters
D_CENTER      = MODULE_LENGTH / 2. # distance from april tag to the center of the april module


# These homogeneous transformations encode the static translational and rotational offset 
# between the poses of the april markers and the pose of the April Module.
April_Ts = {

    APRIL_TOP   :   np.array([[ 0., 1., 0., 0.],
                              [-1., 0., 0., 0.],
                              [ 0., 0., 1., -D_CENTER],
                              [ 0., 0., 0., 1.]]),

    APRIL_FRONT :   np.array([[0., 1., 0., 0.],
                              [0., 0., 1., 0.],
                              [1., 0., 0., -D_CENTER],
                              [0., 0., 0., 1.]]),

    APRIL_LEFT  :   np.array([[-1., 0., 0., 0.],
                              [0., 0., 1., 0.],
                              [0., 1., 0., -D_CENTER],
                              [0., 0., 0., 1.]]),

    APRIL_RIGHT :   np.array([[1.,  0., 0., 0.],
                              [0.,  0., 1., 0.],
                              [0., -1., 0., -D_CENTER],
                              [0.,  0., 0., 1.]]),
}






def get_april_ee_pose(Ts):
    '''
    Returns the pose of the April Module, based on the detected april tag frames.
    If multiple april tag markers of the April Module were detected, it returns the average.

    :param Ts:      Dict {(id, T)}, where id is the ID of the april tag and T its respective frame, encoded in a homogeneous transformation
    :return:        The pose of the April Module, represented as a homogeneous transformation.
                    If no april tag of the April Module was detected, this function should return None
    '''

    # filter out frames that belong to the april module and apply their respective offset transformation
    april_Ts = [T @ April_Ts[id] for (id, T) in Ts.items() if id in April_Ts ]

    if len(april_Ts) > 0:
        # compute average
        mean_T   = np.mean(april_Ts, axis=0)
    else:
        # return None if no marker from the april module was detected
        mean_T   = None

    return mean_T
# ---




def main_april():
    '''
    YOU DO NOT NEED TO DO ANYTHING HERE.
    JUST READ THE CODE AND TRY TO UNDERSTAND!
    '''

    april = AprilDetection(cam_index=0)


    while True:

        # Returns a dictionary {id:int, T:np.ndarray} of detected April tag IDs 
        # and the corresponding poses encoded as homogeneous transformations
        Ts = april.detect()
            

        # execute the rest of the code only if at least one april tag was detected
        if len(Ts) >= 1:

            ee_pose = get_april_ee_pose(Ts)

            if ee_pose is not None:
                april.show(show_markers=False, additional_frames=[ee_pose])
            else:
                april.show(show_markers=False, additional_frames=[])
        
        else:
            # Display only the camera image without any frame if no april tag was detected.
            april.show(additional_frames=[])

    # --- while

# --- def





if __name__ == "__main__":
    main_april()


