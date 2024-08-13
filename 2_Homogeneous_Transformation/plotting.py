import numpy as np
import matplotlib.pyplot as plt





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
    plt.arrow(T[0, 2], T[1, 2], T[0, 0], T[1, 0], head_width=0.2, facecolor='r', edgecolor='r', length_includes_head=True)
    plt.arrow(T[0, 2], T[1, 2], T[0, 1], T[1, 1], head_width=0.2, facecolor='g', edgecolor='g', length_includes_head=True)
    plt.text(T[0, 2]-0.4, T[1, 2]-0.3, '[' + frame_name + ']')

        
    if do_show:
        plt.show()
# ---


def plot_vector(a:np.ndarray, rel_frame_name='', xlim=[-1,4], ylim=[-1,4], color='k', new_fig=True, do_show=True) -> None:
    '''
    Plots 2D vector starting from origin of reference frame
    '''
    if new_fig:
        plt.figure(dpi=100)
        plt.title('Reference Frame: [' + rel_frame_name + ']')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal')

        # plot axes of frame that a is defined in 
        plt.arrow(0, 0, 1, 0, head_width=0.2, facecolor='r', edgecolor='r', length_includes_head=True)
        plt.arrow(0, 0, 0, 1, head_width=0.2, facecolor='g', edgecolor='g', length_includes_head=True)
        plt.text(-0.4, -0.3, '[' + rel_frame_name + ']')


    plt.arrow(0, 0, a[0], a[1], head_width=0.2, facecolor=color, edgecolor=color, length_includes_head=True)
    
    if do_show:
        plt.show()
# ---


def plot_image(M:np.ndarray, rel_frame_name='', xlim=[-1,3], ylim=[-1,3], color='k', new_fig=True, do_show=True) -> None:
    '''
    Plots an image by drawing lines between neighboring data points in a data matrix.
    
    :param M: data matrix with its 2D column vectors being the data points.
    '''    
    if new_fig:
        plt.figure(dpi=100)
        plt.title('Reference Frame: [' + rel_frame_name + ']')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal')

        # plot axes of frame that the data points are defined in 
        plt.arrow(0, 0, 1, 0, head_width=0.2, facecolor='r', edgecolor='r', length_includes_head=True)
        plt.arrow(0, 0, 0, 1, head_width=0.2, facecolor='g', edgecolor='g', length_includes_head=True)
        plt.text(-0.4, -0.3, '[' + rel_frame_name + ']')


    # plot lines
    for i in range(M.shape[1]-1):
        plt.plot([M[0, i], M[0, i+1]], [M[1, i], M[1, i+1]], color=color)

    if do_show:
        plt.show()            
# ---