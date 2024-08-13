import cv2
import numpy as np



def get_webcam_blob_detector(cam_index=0):
    '''
    Connects to Webcam and initialize blob detection
    
    :param cam_index:      Index of the camera connected to the operating system
    :return:               VideoCapture Object, SimpleBlobDetector Object
    '''

    # connect to webcam 
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920/2)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080/2)

    # check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # check if camera provides images
    ret, frame = cap.read()
    if not ret:
        raise IOError("Failed to grab a frame.")  

    # specify blob detection parameters
    params = cv2.SimpleBlobDetector_Params()
    params.blobColor           = 0
    params.filterByColor       = True
    params.minArea             = 500
    params.maxArea             = 100000
    params.filterByArea        = True
    params.filterByCircularity = True
#    params.minThreshold        = 100

    # setup blob detection, depending on the installed version of open-cv
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params) 

    return cap, detector
# ---


def estimate_focal_length(r:float, blob_r:float, d:float) -> float:
    '''
    Returns the focal length.

    :param r:         radius of real circle
    :param blob_r:    radius of detected circle in camera image
    :param d:         known distance between real blob and camera
    
    :return:          focal length
    '''
    return blob_r * d / r
# ---


def estimate_distance(blob_r:float, r:float, f:float) -> float:
    '''
    Returns the distance between the real circle and the webcam.
    
    :param blob_r:    radius of detected circle in camera image
    :param r:         radius of real circle
    :param d:         known distance between real blob and camera
    '''
    return f * r / blob_r
# ---


def get_des_img_features(frame:object, f:float, r:float, d_des:float=30) -> np.ndarray:
    '''
    Returns the desired image features [blob_center_x, blob_center_y, blob_radius], 
    based on a desired distance between the real blob and the webcam.
    
    :param frame:  OpenCV frame object
    :param f:      camera's focal length
    :param r:      radius of real circle
    :param d_des:  distance (in cm) between real blob and webcam
    '''
    desired_x = frame.shape[1] / 2
    desired_y = frame.shape[0] / 2
    desired_r = f * r / d_des

    return np.array([desired_x, desired_y, desired_r])
# ---


def get_image_jacobian(curr_img_features:np.ndarray, f:float, r:float) -> np.ndarray:
    '''
    Returns the image jacobian.

    :param curr_img_features:   current image features [x_img, y_img, r_img]
    :param f:                   camera's focal length
    :param r:                   radius of real circle
    
    :return:                    (3x3) Jacobian matrix
    '''

    x_img, y_img, r_img = curr_img_features
    z = estimate_distance(r_img, r, f)

    return np.array([[-f / z,      0,  x_img / z],
                     [     0, -f / z,  y_img / z],
                     [     0,      0,  r_img / z]])
# ---


def get_delta_cam(des_img_features:np.ndarray, curr_img_features:np.ndarray, J_img:np.ndarray, step:float) -> np.ndarray:
    '''
    Returns a step in camera frame that realizes a step towards the desired image features.

    :param des_img_features:    desired image features
    :param curr_img_features    current image features
    :param J_img:               image Jacobian
    :oaram step:                step size
    '''
    error_img_features  = des_img_features - curr_img_features
    return step * np.linalg.pinv(J_img) @ error_img_features
# ---