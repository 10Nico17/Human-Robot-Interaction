import numpy as np                            # numpy is for math (it's hyper fast)
from scipy.spatial.transform import Rotation  # we are going to use this for deriving euler angles from rot mat




class Kinematics(object):

    def __init__(self, dh_param_func:object, q_min:np.ndarray, q_max:np.ndarray):
        '''
        Initializes a kinamtic object based on a robot kinematic structure given by a DH parameter table
        and by minimum and maximum joint angle values.

        :param dh_param_func       pointer to a function that provides a DH parameter table for a given robot configuraiton q
        :param q_min:              minimum joint angle values
        :param q_max:              maximum joint angle values
        '''
        self.dh_param_func = dh_param_func
        self.q_min         = q_min
        self.q_max         = q_max
    # ---


    def sample_configuration(self, check_self_collision=False) -> np.ndarray:
        '''
        Returns a random robot configuration within the joint limits.
        It can also make sure that the robot does not collide with itself.
        '''
        if check_self_collision:
            # sample until valid configuration was found
            while True:
                q_rnd = self.q_min + np.random.random(self.q_min.shape) * (self.q_max - self.q_min)
                if not self.self_collision(q_rnd):
                    return q_rnd
            # --while
        else:
            return self.q_min + np.random.random(self.q_min.shape) * (self.q_max - self.q_min)
        # --if
    #---


    def get_HT(self, x:float, y:float, theta:float) -> np.ndarray:
        '''
        Returns a (4x4) homogenous transformation that realizes a rotation with angle theta about the z-axis
        and a translation along x and y.
        '''
        return np.array([[np.cos(theta), -np.sin(theta),  0,  x],
                         [np.sin(theta),  np.cos(theta),  0,  y],
                         [            0,              0,  1,  0],
                         [            0,              0,  0,  1]])
    # ---


    def get_dh_matrix(self, theta:float, alpha:float, r:float, d:float) -> np.ndarray:
        '''
        Returns a (4x4) Denavit-Hartenbert transformation matrix based
        on the DH parameters (classic DH convention, not Craig).
        '''
        return np.array([[np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), r * np.cos(theta)],
                         [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), r * np.sin(theta)],
                         [            0,                  np.sin(alpha),                  np.cos(alpha),                 d], 
                         [            0,                              0,                              0,                 1]])
    # ---


    def turn_dh_params_into_dh_matrices(self, dh_param_table:np.ndarray) -> list:
        '''
        Returns a list of Denavit-Hartenberg matrices.
        These matrices are derived from a table containing
        the Denavit-Hartenberg parameters.
        The i-th matrix realizes a transformation between 
        frame i and (i+1), with zero being the robot's first joint frame.
        (Because of informatics, i starts at zero, not one)
        
        <Make your code work for an arbitrary number of rows in the DH parameter table!>
        '''
        dh_matrices = []
        
        for row in dh_param_table:
            dh_matrices.append(self.get_dh_matrix(*row))
        
        return dh_matrices
    # ---


    def map_frames_to_base_frame(self, dh_matrices:list) -> list:
        '''
        Returns a list of homogeneous transformation matrices 
        that represent the robot's joint frames and its end effector frame 
        relative to its first joint frame.

        Remember: The base frame is not part of the DH parameter table!
        '''
        
        # The DH matrices specify the relationship between neighboring robot frames.
        # Thus, the first DH matrix represents the second joint frame relative to the first joint frame.
        #
        # However, to be able to plot the robot in the base frame, we also need to represents 
        # the very first joint frame relative to the base frame. We assume that when 
        # the robot is in zero configuration, the base frame and the first joint frame are identical.
        # Therefore, the first transformation in this function's return list is the identity matrix.
        tmp_transform = np.eye(4)
        transforms    = [tmp_transform]

        # Generate transformations from base frame to i-th robot frame
        for mat in dh_matrices:
            tmp_transform = tmp_transform @ mat
            transforms.append(tmp_transform)
        
        return transforms    
    # ---


    def get_frames_for_dh_param_table(self, dh_param_table:np.ndarray) -> list:
        '''
        Returns the robot's joint frames and its end effector frame relative to 
        the base frame, based on the Denavit-Hartenberg parameter table. 
        This table depends on the robot's current configuration q = [theta_1, ..., theta_n].
        
        :param dh_param_table:    table with denavit-hartenberg parameters
        '''    
        dh_matrices    = self.turn_dh_params_into_dh_matrices(dh_param_table)    
        
        # 3) Generate transformations from the base frame to each robot frame
        frames         = self.map_frames_to_base_frame(dh_matrices)

        return frames    
    # ---


    def get_frames_for_configuration(self, q:np.ndarray) -> list:
        '''
        Returns the robot's joint frames and its end effector frame relative to 
        the base frame, based on the robot's current configuration.

        :param q:       robot's current configuration q = [theta_1, ..., theta_n]
        '''
        return self.get_frames_for_dh_param_table(self.dh_param_func(q))
    # ---


    def get_6D_vec_from_ht(self, T:np.ndarray) -> np.ndarray:
        '''
        Extracts the pose encoded in a homogeneous transformation and returns it as a vector.

        :param T:       (4x4) homogeneous transformation matrix
        :return:        6D pose [x, y, z, roll, pitch, yaw]
        '''
        x, y, z          = T[:3, 3]
        roll, pitch, yaw = Rotation.from_matrix(T[:3, :3]).as_euler('xyz', degrees=False)    
        return np.array([x, y, z, roll, pitch, yaw])
    # ----


    def forward_kinematics(self, q:np.ndarray) -> np.ndarray:
        '''
        Returns the robot's end effector pose based on its current configuration.

        :param q:       robot's current configuration q = [theta_1, ..., theta_n]       
        :return:        6D end effector pose [x, y, z, roll, pitch, yaw]
        '''

        frames = self.get_frames_for_configuration(q)
        return self.get_6D_vec_from_ht(frames[-1])
    # ---


    def get_jacobian(self, q:np.ndarray) -> np.ndarray:
        '''
        Returns the Jacobian matrix based on the robot's current configuration.

        :param q:       robot's current configuration q = [theta_1, ..., theta_n]       
        :return:        Jacobian matrix
        '''
        transforms = self.get_frames_for_configuration(q)    # get homog. transf from base frame to each joint frame
        J     = np.zeros((6, len(transforms) - 1))                 # initialize jacobian with zeros
        d_0_n = transforms[-1][:3, 3]                              # translation to end effector wrt. frame 0
        
        for i in range(1, len(transforms)):                        # go through columns of Jacobian (i starts at 1)
            z_0_im1    = transforms[i-1][:3, 2]                    # extract z-axis from T_0_(i-1)
            d_0_im1    = transforms[i-1][:3, 3]                    # extract translation vec from T_0_(i-1)

            J[:3, i-1] = np.cross(z_0_im1, d_0_n - d_0_im1)        # fill translation part of Jacobian for joint i
            J[3:, i-1] = z_0_im1                                   # fill rotation part of Jacobian for joint i
        
        return J
    # ---


    def perform_step(self, q:np.ndarray, target_pose:np.ndarray, step_size:float = 0.1) -> np.ndarray:
        '''
        Returns a step in configuration space that realizes a movement of the end effector towards 
        the desired end effector pose in operational space.
        
        :param q:              robot's current 6D end effector pose [x, y, z, roll, pitch, yaw]
        :param target_pose:    desired 6D end effector pose [x, y, z, roll, pitch, yaw]
        :param step_size:      factor (leq 1) multiplied with step in end effector space
        
        :return:               updated configuration after having performed a step
                            towards the desired end effector pose
        '''

        # Compute the error between the desired end effector pose
        # and the robot's current end effector pose
        e              = target_pose - self.forward_kinematics(q)
        
        # Compute the Jacobian matrix based on the homogeneous transformations
        # from base frame to frame i, following the jacobian algorithm
        J              = self.get_jacobian(q)

        # Compute the pseudoinverse of the Jacobian matrix
        J_inv          = np.linalg.pinv(J)

        # Compute the robot configuration after following a step
        # towareds the desired end effector pose
        return q + step_size * J_inv @ e
    # ---


    def self_collision(self, q:np.ndarray, n_samples:float=10, d_thresh:float=0.2) -> bool:
        '''
        Checks whether a configuration results in a self-collision.
        It uniformly samples points along each of the robot's links and ckecks weather
        these points are too close to a point on the other links.

        :param q:            robot configuration
        :param n_samples:    number of sample points per link
        :param d_thresh:     minimal allowed distance between sampled points.

        :return:             True if configuration results in self-collision, otherwise False
        '''
        transforms = self.get_frames_for_configuration(q)
        
        # go through all robot links (vector from origin of frame i to origin of frame i+1)
        for i in range(len(transforms)-1):

            t_0_i    = transforms[i][:3, 3]                  # vector pointing from frame 0 to frame i
            t_0_ip1  = transforms[i+1][:3, 3]                # vector pointing from frame 0 to frame i+1
            t_i_ip1  = t_0_ip1 - t_0_i                       # vector pointing from frame i to frame i+1

            # sample points along link (between origin of frame i and origin of frame i+1)
            for f in np.linspace(0., 0.9, n_samples):
                p_i = t_0_i + f * t_i_ip1
                
                # go through all other links of the robot
                for j in range(len(transforms)-1):

                    # do not check for collision between a link and itself
                    if i == j:
                        continue

                    t_0_j    = transforms[j][:3, 3]          # vector pointing from frame 0 to frame j
                    t_0_jp1  = transforms[j+1][:3, 3]        # vector pointing from frame 0 to frame j+1
                    t_j_jp1  = t_0_jp1 - t_0_j               # vector pointing from frame j to frame j+1

                    # sample points along other link (between origin of frame j and origin of frame j+1)
                    for f in np.linspace(0., 0.9, n_samples):
                        p_j = t_0_j + f * t_j_jp1
                        
                        # are points too close?
                        if np.linalg.norm(p_i - p_j) < d_thresh:
                            return True   
        return False
    # ---


    def point_inside_box(self, p:np.ndarray, box:np.ndarray) -> bool:
        '''
        Helper function that tests wether a point p lies inside a box.
        
        :param p:     point [x, y]
        :param box:   [x, y, width, height]
        :return:      True if point lies inside box, otherwise False.
        '''
        
        if p[0] < box[0]:
            return False
        if p[1] < box[1]:
            return False
        if p[0] > box[0] + box[2]:
            return False
        if p[1] > box[1] + box[3]:
            return False
        return True
    # ---        


    def robot_inside_box(self, box:np.ndarray, q:np.ndarray, n_samples:int=10) -> bool:
        '''
        Tests weather the robot that is in a given configuration collides with a box.
        To test this, this function samples points on the robot's links and tests if any
        of these points lies within the box.

        :param box:          a box [x, y, width, height]
        :param q:            robot configuraion q = [theta_1, ..., theta_n]
        :param n_samples:    number of sample points per link
        
        return               True if configuration results in a collision with the box, otherwise False.
        '''
        transforms = self.get_frames_for_configuration(q)
        
        for i in range(len(transforms)-1):
            t_0_i    = transforms[i][:3, 3]            # vector pointing from frame 0 to frame i
            t_0_ip1  = transforms[i+1][:3, 3]          # vector pointing from frame 0 to frame i+1
            t_i_ip1  = t_0_ip1 - t_0_i                 # vector pointing from frame i to frame i+1

            # sample points along link (between origin of frame i and origin of frame i+1)
            for f in np.linspace(0., 1., n_samples):
                p = t_0_i + f * t_i_ip1
                if self.point_inside_box(p, box):
                    return True
        
        # check end effector
        r      = 0.5
        ee_pos = transforms[-1][:2, 3] + 0.5 * transforms[-1][:2, 0]
        ee_box = np.copy(1. * box)
        ee_box[0] = 1. * ee_box[0] - r
        ee_box[1] = 1. * ee_box[1] - r
        ee_box[2] = 1. * ee_box[2] + 2 * r
        ee_box[3] = 1. * ee_box[3] + 2 * r
        #print(box, ee_box)
        if self.point_inside_box(ee_pos, ee_box):
            return True

        return False
    # ---


    def robot_inside_obstacles(self, boxes:np.ndarray, q:np.ndarray, n_samples:int=10) -> bool:
        '''
        Tests weather the robot that is in a given configuration collides with any box in a list of boxes.
        To test this, this function samples points on the robot's links and tests if any
        of these points lies within any of the boxes.

        :param boxes:        a list of boxes [[x_i, y_i, width_i, height_i]]
        :param q:            robot configuraion q = [theta_1, ..., theta_n]
        :param n_samples:    number of sample points per link

        return               True if configuration results in a collision with any box, otherwise False.
        '''
        for box in boxes:
            if self.robot_inside_box(box, q, n_samples):
                return True

        return False
    # ---


# --- class