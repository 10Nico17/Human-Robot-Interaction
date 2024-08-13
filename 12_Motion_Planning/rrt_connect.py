import numpy as np


class Tree(object):
    '''
    This simple tree structure constitutes the foundation of our RRT-Connect implementation below.
    Such tree always possesses a root node which is the starting point of the search.
    Every node, except for the root node, has a parent node pointing to it. 

    This tree structure allows:
    1) insertion of new nodes
    2) finding the closest node in the tree to a specific node, both either in end effector space or in configuration space
    3) finding a path from the root node of the tree to a specific node in the tree
    '''

    def __init__(self, root:object):
        '''
        Initializes tree data structure with a single root node as its starting point.
        :param root:       root node object
        '''
        self.root    = root
        self.X       = np.array([root.x])               # op-space matrix for fast distance computation
        self.Q       = np.array([root.q])               # c-space matrix for fast distance computation
        self.x_nodes = {root.x.tobytes() : root}        # hash map for fast look-up (on average O(1) operations)
        self.q_nodes = {root.q.tobytes() : root}        # hash map for fast look-up (on average O(1) operations)
    # ---
    

    def insert_node(self, node:object) -> None:
        '''
        Inserts node that consists of an end effector pose x and a respective
        configuration q in the tree structure.

        :param node:       object of new node that is to be inserted in the tree
        '''
        self.X   = np.vstack([self.X, node.x])
        self.Q   = np.vstack([self.Q, node.q])
        self.x_nodes[node.x.tobytes()] = node
        self.q_nodes[node.q.tobytes()] = node
    # --
    

    def get_closest(self, node:object, ee_space:bool=True) -> object:
        '''
        Returns the node object that is closest to node.
        This distance is either measured in ee space or in configuration space.

        :param node:       node for which we want to find the closest pose in the tree
        :param ee_space:   distance is measured in ee space if set to True, in C-space if set to False
        :return:           node object whose ee pose is closest to the pose of node
        '''
        if ee_space:
            dists     = np.linalg.norm(node.x - self.X, axis=1)
            x_closest = self.X[np.argmin(dists)]
            return self.x_nodes[x_closest.tobytes()] 
        else:
            dists     = np.linalg.norm(node.q - self.Q, axis=1)
            q_closest = self.Q[np.argmin(dists)]
            return self.q_nodes[q_closest.tobytes()]
    # ---
    

    def get_path(self, goal_node:object) -> list:
        '''
        Returns a path to a goal node, starting from the root node.
        This node needs to be already inserted in the tree.

        :param goal node:  already inserted node that constitutes the end of the path
        :return:           list of nodes forming a path through the tree. 
                           The first node is the root node and last node is the queried node.
        '''
        
        '''<IMPLEMENT YOUR CODE HERE!>'''
        
        return '''<IMPLEMENT YOUR CODE HERE!>'''
    # ---
        
# --- Tree



class Node(object):
    '''
    In our version of the RRT-Connect algorithm, a node is defined by
    1) a configuration q = [theta_1, theta_2, ...], 
    2) the respective end effector pose x = [x, y, z, roll, pitch, yaw] (relative to the robot's first joint frame), and
    3) a parent node. Each node has a parent node, except for the tree's root node which is the starting point of the search.
    '''
    
    def __init__(self, x:np.ndarray, q:np.ndarray, parent:object=None):
        self.x      = x
        self.q      = q
        self.parent = parent
    # ---
# --- Node



class RRT_Connect(object):
    '''
    The RRT-Connect algorithm allows collision free motion planning.
    It is a sampling-based search algorithm that finds a non-optimal path
    in the robot's configuration space to a desired end effector pose.
    '''

    def __init__(self, q_start:np.ndarray, x_goal:np.ndarray, kinematics:object, obstacles:np.ndarray, goal_bias:float=0.1, ee_thresh:float=0.1, q_thresh:float=0.2):
        '''
        Initialize a RRT-Connect. It consts of a start node, a goal node, and a tree sturcture.
        Furthermore, it knows about the robot kinematics (based on DH parameters) and about obstacles (defined in ee space).

        :param q_start:          initial state of the robot in configuration space
        :param x_goal:           goal pose in end effector space
        :param kinematics:       object that provides the functions forward_kinematics(q) -> x, and perform_step(q, target_pose) -> q_new
        :param obstacles:        list of boxes that need to be avoided
        :param goal_bias:        probability of sampling the goal node
        :param ee_thres:         we interpret two poses in ee space as identical if their distance is below this threshold
        :param q_thres:          we interpret two configurations as identical if their distance is below this threshold
        '''
        self.kinematics    = kinematics
        self.obstacles     = obstacles
        self.goal_bias     = goal_bias
        self.ee_thresh     = ee_thresh
        self.q_thresh      = q_thresh

        self.start_node    = Node(kinematics.forward_kinematics(q_start), q_start)
        self.goal_node     = Node(x_goal, None)
        self.tree          = Tree(root=self.start_node)
    # ---


    def create_random_node(self) -> object:
        '''
        Craetes a new random node within the workspace of the robot
        by sampling joint angles within the joint limits. This node is not
        yet inserted to the tree structure.

        :return:    new random node
        '''
        q_rnd   = self.kinematics.sample_configuration()
        x_rnd   = self.kinematics.forward_kinematics(q_rnd)
        return Node(x_rnd, q_rnd)
    # ---


    def configuration_is_valid(self, q:np.ndarray, check_self_collision:bool=True):
        '''
        Checks weather a given configuration is valid. 
        Thus, it makes sure that in this configuration, the robot does
        not collide with itself and it does not collide with any obstacle.

        :param q:                        robot configuraion q = [theta_1, ..., theta_n]
        :param check_self_collision:     checks if robot collides with itself if set to True
        '''
        if check_self_collision and self.kinematics.self_collision(q):
            return False
        if self.kinematics.robot_inside_obstacles(self.obstacles, q):
            return False
        return True
    # ---


    def perform_search(self, check_self_collision:bool=True, n_steps:int=100, n_conn_steps:int=100, step_size:float=0.1):
        '''
        Starts the search.

        :param check_self_collision:     makes sure the robot does not collide with itself, if set to True
        :param n_steps:                  stops the whole search if the goal pose was not found after this many search steps
        :param n_conn_steps:             stops the connect step if the sampled node was not reached after this many iterations
        :param step_size:                travelling distance in configuratoin space during the connect step
        
        :return:                         the path (list of nodes) from the root node to the goal node if the goal pose (in ee space) was found. 
                                         Otherwise it returns None.
        '''
        for i in range(n_steps):
            goal_reached = self.perform_step(check_self_collision, n_conn_steps, step_size)

            if goal_reached:
                break
        
        if goal_reached:
            print('goal reached')
            return self.tree.get_path(self.goal_node)
        else:
            return None
    # ---


    def perform_step(self, check_self_collision=True, n_conn_steps:int=100, step_size:float=0.1):
        '''
        Performs one search step. 

        :param check_self_collision:     makes sure the robot does not collide with itself, if set to True
        :param n_steps:                  stops the whole search if the goal pose was not found after this many search steps
        :param n_conn_steps:             stops the connect step if the sampled node was not reached after this many iterations
        :param step_size:                travelling distance in configuratoin space during the connect step

        :return:                         True if the goal pose (in ee space) was found, otherwise False
        '''

        # With probability goal_bias, choose the goal node.
        if np.random.random() < self.goal_bias:
            sampled_node = self.goal_node
            ee_space     = True
        
        # With probability (1-goal_bias), sample a random node inside the robot's workspace.
        else:
            sampled_node = self.create_random_node()
            ee_space     = False
        # -- if
    
        # Find the node in the tree that is closest to the sampled node.
        # If the sampled node is the goal node, the distance is measured in 
        # end effector space, otherwise it is measured in configuration space
        closest_node = self.tree.get_closest(sampled_node, ee_space=ee_space)

        # In configuration space, move from the closest node in the tree towards the sampled node 
        # until either an invalid configuration was hit or the sampled node was reached
        # Along the way, create and insert new nodes to the tree.
        last_connected_node = self.connect(closest_node, sampled_node, ee_space, check_self_collision, n_conn_steps, step_size)
        if last_connected_node is None:
            return False

        # Did we find the goal node?
        if np.linalg.norm(self.goal_node.x - last_connected_node.x) < self.ee_thresh:
            self.goal_node = last_connected_node
            return True
        else:
            return False
    # ---


    def connect(self, start_node, end_node, ee_space:bool, check_self_collision=True, n_conn_steps:int=100, step_size:float=0.1):
        '''
        Tries to connect a start node with an end node while moving through the configuration space. 
        During this process, it creates and inserts new nodes to the tree.
        It stops, if either an invalid configuration was hit, the end node was reached, or the maximum number of steps were taken.
        If the end node is the goal node (i.e., the desired end effector pose; in this case ee_space=True),
        the Jacobian is used to derive the joint movement that results in a step towards the goal node's end effector pose.

        :param start_node:               node in the tree from which we start the connect proces
        :param end_node:                 new node that is not part of the tree towards which we try to connect
        :param ee_space:                 if True, uses Jacobian while connecting in ee space, otherwise connects in configuration space.
        :param check_self_collision:     makes sure the robot does not collide with itself, if set to True
        :param n_conn_steps:             maximum number of steps to connect to the end node
        :param step_size:                distance in configuration space per connect step
        
        :return:                         node that as added last during the connect process if at least one was added, otherwise None
        '''
        curr_q    = np.copy(start_node.q)
        curr_x    = np.copy(start_node.x)
        curr_node = start_node
        
        # compute unit vector towards desired configuration in configuration space
        if ee_space == False:
            q_diff    = (end_node.q - start_node.q) / np.linalg.norm(end_node.q - start_node.q)
        

        added_at_least_one_node = False
        for i in range(n_conn_steps):
            
            # make a test step towards goal node
            if ee_space:
                # in end effector space, we use the jacobian to move towards the target pose
                q_test = self.kinematics.perform_step(q=curr_q, target_pose=end_node.x, step_size=step_size)

            else:
                # in configuration space, we simply move towards the desired configuration
                q_test = curr_q + step_size * q_diff
            # -- if


            # stop moving if the test step results in an invalid configuration
            if not self.configuration_is_valid(q_test, check_self_collision):
                break

            
            # if test step is valid, take this step
            # and compute the new end effector pose
            curr_q    = q_test
            curr_x    = self.kinematics.forward_kinematics(curr_q)

            # create and insert new node
            curr_node = Node(curr_x, curr_q, curr_node)
            self.tree.insert_node(curr_node)

            # remember that we added at least one node
            added_at_least_one_node = True


            # stop connecting if we reached the goal node
            if ee_space:
                if np.linalg.norm(curr_x - end_node.x) < self.ee_thresh:
                    break
            else:
                if np.linalg.norm(curr_q - end_node.q) < self.q_thresh:
                    break
        # -- for

        # return the last added node, if one exists
        if added_at_least_one_node:
            return curr_node
        else:
            return None
    # ---


# --- class