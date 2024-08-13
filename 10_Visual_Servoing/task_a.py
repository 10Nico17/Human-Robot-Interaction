'''
---------------
VISUAL SERVOING
---------------


TASK A - Piece-by-piece implementation and debugging
----------------------------------------------------

What this task is about:
    Visual servoing is a method in which visual features guide the movement of the robot.
    More specifically, we define a forward function that determines the features in the image,
    based on the end effector pose. To invert this function, we linearize it and obtain the image jacobian.
    In this file, you will first implement visual servoing and make sure that everything
    runs smoothly, before moving on to the next task in which will implement visual servoing on the LAURA robot.
    
What you need to do in the real world:
    * Connect the camera to your computer
    * Take out the Blob

What you need to implement:
    * Implement the missing code in visual_servoing.py
    * Update the code in this file to run the different tasks

What you should see if your code is correct:
    * If all goes well, you will see instructions of how to move the blob,
      so that it ends up at the center of the camera image and at a certain distance.
'''


import cv2
import numpy as np
from visual_servoing import *






def main():
 
    # connect to webcam and initialize blob detection via OpenCV
    cap, detector = get_webcam_blob_detector(cam_index=0)

    r     = 17.5   # radius of real circle [millimeters]
    f     = 1650   # camera's focal length 
    d_des = 300    # desired distance between real circle and webcam [millimeters]


    # main control loop
    while True:

        # exit loop on ESC key-hit
        c = cv2.waitKey(1)
        if c == 27:
            break

        # capture and resize camera image
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        if not ret:
            print("Failed to grab frame.")

        # detect blobs (i.e. circles) in camera image
        keypoints = detector.detect(frame)

        
        # have we detected at least one blob?
        if len(keypoints) > 0:
            
            # extract image features from first blob
            blob = keypoints[0]
            curr_img_features = np.array([blob.pt[0], blob.pt[1], blob.size])


            '''<Change TASK to execute different pieces of code!>'''
            TASK = 2

            if TASK == 1:
                '''
                ESTIMATE FOCAL LENGTH            

                First, implement the function estimage_focal_length(). 
                Then, hold the real circle <d_des> centimeters away from the camera 
                and run this program to estimate the focal length.
                Finally, set the variable f to your estimated focal length value!
                '''
                print('estimated focal length:', estimate_focal_length(r, curr_img_features[2], d_des))
            

            elif TASK == 2:
                '''
                ESTIMATE CIRCLE DISTANCE TO CAMERA

                First, solve task 1.
                Second, implement the function estimage_distance(). 
                Third, hold the real circle in front of the camera while
                running this program and test weather the estimated distance is correct.
                '''
                print('estimated distance [mm]:', estimate_distance(curr_img_features[2], r, f))
            

            elif TASK == 3:
                '''
                COMPUTE DESIRED IMAGE FEATURES

                First, solve tasks 1 and 2.
                Second, implement the function get_des_img_features().
                Third, run this program and test, wheather the desired image features make sense.
                '''
                print('desired image features:', get_des_img_features(frame, f, r, d_des))
            

            elif TASK == 4:
                '''
                COMPUTE CAMERA MOVEMENT

                First, solve tasks 1 to 3. 
                Second, implement the functions get_image_jacobian() and get_delta_cam().
                Third, run this program and test, wheather the desired image features make sense.
                '''

                delta_cam = get_delta_cam(des_img_features  = get_des_img_features(frame, f, r, d_des), 
                                          curr_img_features = curr_img_features, 
                                          J_img             = get_image_jacobian(curr_img_features, f, r),
                                          step              = 1.)

                print('camera step:', delta_cam)
            # -- /if

        # -- /if


        
        # draw detected blobs as red circles whose size corresponds to the size of the blob
        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # display camera image with detected blobs as red circles
        cv2.imshow("Keypoints", im_with_keypoints)

    # -- /while


    # clean-up OpenCV
    cap.release()
    cv2.destroyAllWindows()

# --- main









if __name__ == "__main__":
    main()