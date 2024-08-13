import numpy as np
import matplotlib.pyplot as plt     # matplotlib is for plotting stuff
import seaborn as sns               # seaborn is for nicer looking plots
sns.set()                           # activate seaborn



def plot_joint(joint_pos):
    # Plots a circle, representing a joint at a given position.
    # This position is given by (joint_pos[0], joint_pos[1]) in base frame.
    CIRCLE_RADIUS = 0.02
    CIRCLE_COLOR  = 'k'

    circle = plt.Circle((joint_pos[0], joint_pos[1]), CIRCLE_RADIUS, color=CIRCLE_COLOR)
    plt.gca().add_patch(circle)
 # ---


def plot_link(jointA_pos, jointB_pos):
    # Plots a line between two joints, given by their position (joint_pos[0], joint_pos[1]) in base frame.
    LINK_THICKNESS = 4
    plt.plot([jointA_pos[0], jointB_pos[0]], [jointA_pos[1], jointB_pos[1]], color='k', linewidth=LINK_THICKNESS)
# ---


def plot_ee(frame, color='k'):
    # Plots the end effector, specified by its frame.
    # A frame is a matrix with three column vectors.
    # The first column vector represents the origin of the frame, defined in base frame.
    # The second column vector represents the x-axis of the frame
    # and the third column vector its y-axis. Both axis are represented in base frame.
    EE_SIZE      = 0.05
    EE_THICKNESS = 4
 
    origin = frame[:, 0]
    xaxis  = (frame[:, 1] - origin)
    yaxis  = (frame[:, 2] - origin)

    # A--B
    # [
    # C--D
    point_A = origin + yaxis * EE_SIZE / 2
    point_B = point_A + xaxis * EE_SIZE
    point_C = origin - yaxis * EE_SIZE / 2
    point_D = point_C + xaxis * EE_SIZE

    plt.plot([point_A[0], point_B[0]],[point_A[1], point_B[1]], color=color, linewidth=EE_THICKNESS)
    plt.plot([point_A[0], point_C[0]],[point_A[1], point_C[1]], color=color, linewidth=EE_THICKNESS)
    plt.plot([point_C[0], point_D[0]],[point_C[1], point_D[1]], color=color, linewidth=EE_THICKNESS)
 # ---


def plot_frame(frame):
    # Plots a reference frame.
    # A frame is a matrix with three column vectors.
    # The first column vector represents the origin of the frame, defined in base frame.
    # The second column vector represents the x-axis of the frame
    # and the third column vector its y-axis. Both axis are represented in base frame.
    AXIS_LENGTH    = 0.1
    AXIS_THICKNESS = 2

    origin = frame[:, 0]
    xaxis  = origin + (frame[:, 1] - origin) * AXIS_LENGTH
    yaxis  = origin + (frame[:, 2] - origin) * AXIS_LENGTH

    plt.plot([origin[0], xaxis[0]], [origin[1], xaxis[1]], color='r', linewidth=AXIS_THICKNESS)
    plt.plot([origin[0], yaxis[0]], [origin[1], yaxis[1]], color='g', linewidth=AXIS_THICKNESS)
# ---


def plot_robot_for_frames_old(frame_w, frame_0, frame_1, frame_2, frame_3, frame_ee, target_pose=None):

    # extract frame origins for better readability
    joint1_pos = frame_1[:, 0]
    joint2_pos = frame_2[:, 0]
    joint3_pos = frame_3[:, 0]
    ee_pos     = frame_ee[:, 0]


    plt.figure(figsize=(10,5))

    if target_pose is not None:
        if target_pose.ndim == 1:
            plot_target_pose(target_pose)
        else:
            for i in range(target_pose.shape[1]):
                plot_target_pose(target_pose[i])

    # plot joints as circles
    plot_joint(joint1_pos)
    plot_joint(joint2_pos)
    plot_joint(joint3_pos)

    # plot links as lines between circles
    plot_link(joint1_pos, joint2_pos)
    plot_link(joint2_pos, joint3_pos)
    plot_link(joint3_pos, ee_pos)

    # plot end effector as three lines, representing a parallel jaw gripper
    plot_ee(frame_ee)

    # plot reference frames
    plot_frame(frame_w)
    plot_frame(frame_0)
    plot_frame(frame_1)
    plot_frame(frame_2)
    plot_frame(frame_3)
    plot_frame(frame_ee)

    plt.xlim(-0.7, 1.2)
    plt.ylim(-0.1, 0.8)

    plt.gca().set_aspect("equal")
    plt.show()
# ---


def plot_robot_for_frames(frames, first_joint_idx=0, target_poses=None, xlim=(-0.7, 1.2), ylim=(-0.1, 0.8), interp_xs=None, interp_ys=None, plot_env=False, init_fig=True, do_show=True, get_env_border=None):

    origins        = [frame[:, 0] for frame in frames]
    frame_is_joint = np.logical_and(np.arange(len(frames)) >= first_joint_idx, np.arange(len(frames)) < len(frames)-1)

    if init_fig: plt.figure(figsize=(10,5))
    
    if plot_env:
        xs = np.linspace(xlim[0], xlim[1], 1000)
        ys = get_env_border(xs)
        plt.plot(xs, ys, color='blue', linewidth=2)
    # ---

    # plot interpolation between target poses
    if interp_xs is not None:
        plt.plot(interp_xs, interp_ys, '--', color='red')

    # plot target poses
    if target_poses is not None:
        if target_poses.ndim ==1:
            target_poses = np.array([target_poses])

        for target_pose in target_poses:
            theta = target_pose[5]
            target_pose_frame = np.array([
                [target_pose[0], target_pose[0]+np.cos(theta), target_pose[0]-np.sin(theta)],
                [target_pose[1], target_pose[1]+np.sin(theta), target_pose[1]+np.cos(theta)],
                [            0,                            0,                             0],
            ])

            plot_ee(target_pose_frame, color='#FF0000')
        # --
    # --


    # plot circle if frame belongs to joint
    for i, frame in enumerate(frames):
        if frame_is_joint[i]:
            plot_joint(origins[i])

    # plot links as lines between circles
    for i in range(first_joint_idx, len(frames)-1):
        plot_link(origins[i], origins[i+1])

    # plot end effector as three lines, representing a parallel jaw gripper
    plot_ee(frames[-1])

    # plot reference frames
    for frame in frames:
        plot_frame(frame)

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.gca().set_aspect("equal")
    if do_show: plt.show()
# ---



#def get_env_border(x: float) -> float:
#    '''
#    The robot cannot move through the environment. 
#    This function defines the environment.
#
#    :param x:     horizontal position on border
#    :return:      vertical position on border that corresponds to x
#    '''
#
#    return 0.1 * np.sin(x) - 0.1
# ---


def inside_env(x: float, y: float) -> bool:
    '''
    Decides whether a point is inside the environment.

    :param x:    horizontal coordinate of point
    :param y:    vertical coordinate of point
    :return:     True if point (x, y) lies inside the environment, else False.
    '''

    return y >= get_env_border(y)
# ---