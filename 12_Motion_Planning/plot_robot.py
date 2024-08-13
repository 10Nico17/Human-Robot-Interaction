import numpy as np
import matplotlib.pyplot as plt     # matplotlib is for plotting stuff
import seaborn as sns               # seaborn is for nicer looking plots
sns.set()                           # activate seaborn



def get_T_for_6D_vec(x:np.ndarray) -> np.ndarray:
    '''
    Returns a (4x4) homogenous transformation that realizes a rotation with angle theta about the z-axis
    and a translation along x and y.
    '''
    x, y, _, _, _, theta = x
    return np.array([[np.cos(theta), -np.sin(theta),  0,  x],
                     [np.sin(theta),  np.cos(theta),  0,  y],
                     [            0,              0,  1,  0],
                     [            0,              0,  0,  1]])
# ---



def plot_joint(T:np.ndarray) -> None:
    '''
    Plots a robot joint as a circle. The joint frame is represented as a homogeneous transformation
    that is defined relative to the world frame.
    '''
    CIRCLE_RADIUS = 0.1
    CIRCLE_COLOR  = 'k'

    # the joint location is given by the translation vector from the homogeneous transformation.
    circle = plt.Circle((T[0, 2],  T[1, 2]), CIRCLE_RADIUS, color=CIRCLE_COLOR, zorder=0)
    plt.gca().add_patch(circle)
 # ---



def plot_link(T_A:np.ndarray, T_B:np.ndarray) -> None:
    '''
    Plots a line between two joints. The two joint frames are represented as homogeneous transformations
    that are defined relative to the world frame.
    '''
    LINK_THICKNESS = 2

    # the joint locations are given by the translation vectors from their respective homogeneous transformation.
    plt.plot([T_A[0, 2], T_B[0, 2]], [T_A[1, 2], T_B[1, 2]], color='k', linewidth=LINK_THICKNESS, zorder=1)
# ---


def plot_ee(T:np.ndarray, color='k') -> None:
    '''
    Plots the end effector of a robot as three lines. The end effector frame is represented as a homogeneous transformation
    that is defined relative to the world frame. 
    '''
    EE_SIZE      = 0.5
    EE_THICKNESS = 2

    origin = T[:2, 2]
    xaxis  = (T[:2, 0])
    yaxis  = (T[:2, 1])

    # A--B
    # |
    # C--D
    point_A = origin + yaxis * EE_SIZE / 2.
    point_B = point_A + xaxis * EE_SIZE
    point_C = origin - yaxis * EE_SIZE / 2.
    point_D = point_C + xaxis * EE_SIZE

    plt.plot([point_A[0], point_B[0]],[point_A[1], point_B[1]], color=color, linewidth=EE_THICKNESS, zorder=1)
    plt.plot([point_A[0], point_C[0]],[point_A[1], point_C[1]], color=color, linewidth=EE_THICKNESS, zorder=1)
    plt.plot([point_C[0], point_D[0]],[point_C[1], point_D[1]], color=color, linewidth=EE_THICKNESS, zorder=1)
 # ---


def plot_frame(T, frame_name, rel_frame_name='', xlim=[-1,8], ylim=[-1,4], new_fig=True, do_show=True) -> None:
    '''
    Plots a reference frame that is defined by a homogeneous transformation matrix T.
    '''    
    if new_fig:
        plt.figure(dpi=150)
        plt.title('Reference Frame: [' + rel_frame_name + ']')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal')
        plt.xlim(xlim)
        plt.ylim(ylim)

        # plot axes of frame that T is defined in 
        plt.arrow(0, 0, 1, 0, head_width=0.2, facecolor='r', edgecolor='r', length_includes_head=True)
        plt.arrow(0, 0, 0, 1, head_width=0.2, facecolor='g', edgecolor='g', length_includes_head=True)
        plt.text(-0.4, -0.3, '[' + rel_frame_name + ']')

        
    # plot frame defined in T
    plt.arrow(T[0, 2], T[1, 2], T[0, 0], T[1, 0], head_width=0.2, facecolor='r', edgecolor='r', length_includes_head=True, zorder=2)
    plt.arrow(T[0, 2], T[1, 2], T[0, 1], T[1, 1], head_width=0.2, facecolor='g', edgecolor='g', length_includes_head=True, zorder=2)
    plt.text(T[0, 2]-0.4, T[1, 2]-0.3, '[' + frame_name + ']')

        
    if do_show:
        plt.show()
# ---


def plot_robot(frames:list, target_pose:np.ndarray=None, do_show=True) -> None:
    '''
    Plots a robot for a given set of reference frames that are represented as homogeneous transformations.

    :param frames:         list of homogeneous transformations based on the configuration of a robot. 
                           The first frame in this list is the base frame, the second frame is the first joint frame, 
                           and the last frame is the end effector frame.
                           All frames are defined in the world frame.
    :param target_pose:    Target end effector pose, represented as homogeneous transformation relative to the base frame.
    '''

    plt.figure(dpi=150)
    plt.title('Reference Frame: [w]')
    plt.xlabel('x')
    plt.ylabel('y')

    # plot axes of world frame
    plt.arrow(0, 0, 1, 0, head_width=0.2, facecolor='r', edgecolor='r', length_includes_head=True)
    plt.arrow(0, 0, 0, 1, head_width=0.2, facecolor='g', edgecolor='g', length_includes_head=True)
    plt.text(-0.4, -0.3, '[w]')

    # plot joints
    for (i, T) in enumerate(frames):
        if i != len(frames) - 1:
            plot_joint(T)

    # plot links
    for i in range(len(frames) - 1):
        T_A = frames[i]
        T_B = frames[i+1]
        plot_link(T_A, T_B)

    # plot end effector
    plot_ee(frames[-1])

    # plot frames
    for (i, T) in enumerate(frames):
        if i != len(frames) - 1:
            plot_frame(T, frame_name=str(i), rel_frame_name='w', new_fig=False, do_show=False)
        else:
            plot_frame(T, frame_name='ee', rel_frame_name='w', new_fig=False, do_show=False)

    # plot target pose as red end effector
    if target_pose is not None:
        plot_ee(target_pose, color='r')


    plt.gca().set_aspect("equal")
    
    if do_show:
        plt.show()
# ---


def extract_2d(T_4x4:np.ndarray) -> np.ndarray:
    T_3x3 = np.eye(3)
    T_3x3[:2, :2] = T_4x4[:2, :2]
    T_3x3[:2, 2]  = T_4x4[:2, 3]
    return T_3x3
# ---


def plot_robot_dh(frames:list, target_poses:np.ndarray=None, interp_xs:np.ndarray=None, interp_ys:np.ndarray=None, do_show=True, reference_frame_name='0') -> None:
    '''
    Plots a robot for a given set of reference frames that are represented as homogeneous transformations defined relative to the base frame.

    :param frames:         list of homogeneous transformations based on the configuration of a robot. 
                           The first frame in this list is the first joint frame, the second frame is the second joint frame, 
                           and the last frame is the end effector frame.
                           All frames are defined in the robot's base frame.
    :param target_pose:    Target end effector pose, represented as homogeneous transformation relative to the base frame.

    '''

    plt.figure(dpi=150)
    plt.title('Reference Frame: [' + str(reference_frame_name) + ']')
    plt.xlabel('x')
    plt.ylabel('y')

    # plot axes of world frame
    #plt.arrow(0, 0, 1, 0, head_width=0.2, facecolor='r', edgecolor='r', length_includes_head=True)
    #plt.arrow(0, 0, 0, 1, head_width=0.2, facecolor='g', edgecolor='g', length_includes_head=True)
    #plt.text(-0.4, -0.3, '[' + str(reference_frame_name) + ']')

    # plot joints
    for (i, T) in enumerate(frames):
        if i != len(frames) - 1:
            plot_joint(extract_2d(T))

    # plot links
    for i in range(len(frames) - 1):
        T_A = frames[i]
        T_B = frames[i+1]
        plot_link(extract_2d(T_A), extract_2d(T_B))

    # plot frames
    for (i, T) in enumerate(frames):
        if i != len(frames) - 1:
            plot_frame(extract_2d(T), frame_name=str(i), rel_frame_name='w', new_fig=False, do_show=False)
        else:
            plot_frame(extract_2d(T), frame_name='ee', rel_frame_name='w', new_fig=False, do_show=False)

    # plot trajectory
    if interp_xs is not None:
        plt.plot(interp_xs, interp_ys, '--', color='r', alpha=0.5)

    # plot target poses as red end effector
    if target_poses is not None:
        for pose in target_poses:
            #pass
            plot_ee(extract_2d(get_T_for_6D_vec(pose)), color='r')

    # plot end effector
    plot_ee(extract_2d(frames[-1]), color='k')

    #plt.gca().set_xlim(left=-1)
    #plt.gca().set_ylim(bottom=-1)
    plt.gca().set_aspect("equal")
    
    if do_show:
        plt.show()
# ---


def plot_frame_dh(T, frame_name, rel_frame_name='', xlim=[-1,8], ylim=[-1,4], new_fig=True, do_show=True) -> None:
    '''
    Plots a reference frame that is defined by a homogeneous transformation matrix T.
    '''    

    T = extract_2d(T)

    if new_fig:
        plt.figure(dpi=150)
        plt.title('Reference Frame: [' + rel_frame_name + ']')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal')
        plt.xlim(xlim)
        plt.ylim(ylim)

        # plot axes of frame that T is defined in 
        plt.arrow(0, 0, 1, 0, head_width=0.2, facecolor='r', edgecolor='r', length_includes_head=True)
        plt.arrow(0, 0, 0, 1, head_width=0.2, facecolor='g', edgecolor='g', length_includes_head=True)
        plt.text(-0.4, -0.3, '[' + rel_frame_name + ']')

        
    # plot frame defined in T
    plt.arrow(T[0, 2], T[1, 2], T[0, 0], T[1, 0], head_width=0.2, facecolor='r', edgecolor='r', length_includes_head=True, zorder=2)
    plt.arrow(T[0, 2], T[1, 2], T[0, 1], T[1, 1], head_width=0.2, facecolor='g', edgecolor='g', length_includes_head=True, zorder=2)
    plt.text(T[0, 2]-0.4, T[1, 2]-0.3, '[' + frame_name + ']')

        
    if do_show:
        plt.show()
# ---
