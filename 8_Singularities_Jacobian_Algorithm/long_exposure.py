import cv2
import numpy as np



class LongExposure(object):
    '''
    Provides easy-to-use functionalities for capturing long-exposure umages with the webcam.
    '''

    def __init__(self, cam_id=0):
        '''
        Initializes long exposure image capture 
        while connecting to the webcam.
        '''

        # Initialize the webcam
        self.video_capture = cv2.VideoCapture(cam_id)    
        self.video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)     
        
        # Check if the webcam is opened correctly
        if not self.video_capture.isOpened():
            raise IOError("Cannot open webcam")

        # Initialize an image to store the maximum values for color images
        self.max_blend = None
    # ---


    def capture_image(self):
        '''
        Add new image to the long exposure image.
        Additional images are blent-in by taking the maximum values.
        '''

        # capture camera image from webam
        ret, frame = self.video_capture.read()

        if not ret:
            raise IOError("Cannot capture camera image!")

        # Initialize max_blend with the first frame
        if self.max_blend is None:
            self.max_blend = frame
        else:
            # Blend using maximum value for each color channel
            self.max_blend = np.maximum(self.max_blend, frame)        
    # ---


    def show(self):
        '''
        Display current long exposure image.
        '''
        if self.max_blend is not None:
            cv2.imshow('Long Exposure Effect', self.max_blend)        
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pass
    # ---


    def save(self, file_name):
        '''
        Save the long exposure image to file.
        '''
        # Check if max_blend is not empty
        if self.max_blend is not None:
            # Save the result
            cv2.imwrite(file_name, self.max_blend)
    # ---


    def __del__(self):
        '''
        Properly shut down long time exposure module
        '''
        cv2.destroyAllWindows()
        self.video_capture.release()
    # ----


# --- class

