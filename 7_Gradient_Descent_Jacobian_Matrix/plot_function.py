import numpy as np                  # fast math library
import matplotlib.pyplot as plt     # matplotlib is for plotting stuff
import seaborn as sns               # seaborn is for nicer looking plots
sns.set()                           # activate seaborn



def plot_1D_function(f, xlim, curr_x=None, df_dx=None, yd=None, n_samples=1000, gradient_plotting_factor=1):
    '''
    Plots a function within a certain interval, 
    the current location of x, the desired y value,
    and the gradient
    '''
    
    plt.figure(dpi=100)

    # plot sampled points on function
    xs = np.linspace(xlim[0], xlim[1], n_samples)
    ys = f(xs)

    # plot function    
    plt.plot(xs, ys, label='f(x)')
    
    # plot current position
    if curr_x is not None:
        plt.plot([curr_x], [f(curr_x)], 'o', color='r', markersize=10, label='current solution')
    
        # plot desired y-value
        if yd is not None:
            plt.plot([xlim[0], xlim[1]], [yd, yd], color='orange', label='desired y')

            # plot inverse gradient
            if df_dx is not None:
                inv_gradient = 1. / df_dx(curr_x)
                plt.plot([curr_x, curr_x + inv_gradient*gradient_plotting_factor], [f(curr_x), f(curr_x)], color='r', zorder=2, label='inverse gradient')
        else:
            # plot gradient
            if df_dx is not None:
                plt.plot([curr_x, curr_x + df_dx(curr_x)*gradient_plotting_factor], [f(curr_x), f(curr_x)], color='r', zorder=2, label='gradient')

        
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(xlim)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
# ---



def plot_2D_function(f, xlim=[-np.pi, np.pi], ylim=[-np.pi, np.pi], curr_x=None, df_dx=None, yd=None, n_samples=100, cmap='viridis', gradient_plotting_factor=1, new_fig=True, do_show=True, ax=None):

    if new_fig:
        plt.figure(dpi=100)

    xs   = np.linspace(xlim[0], xlim[1], n_samples)
    ys   = np.linspace(ylim[0], ylim[1], n_samples)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)
    

    plt.imshow(Z, cmap=cmap, origin='lower')
    
    if new_fig:
        plt.colorbar()
    
    plt.grid(False)

    if curr_x is not None:

        # plot current position
        im_pos = np.zeros(2)
        im_pos[0] =  1. * n_samples * (-xlim[0] + curr_x[0]) / (xlim[1] - xlim[0])
        im_pos[1] =  1. * n_samples * (-ylim[0] + curr_x[1]) / (ylim[1] - ylim[0])

        circle = plt.Circle(im_pos , 2)
        plt.gca().add_patch(circle)

        # plot gradient
        if df_dx is not None:
            p_end = curr_x + df_dx(curr_x[0], curr_x[1]) * gradient_plotting_factor
            p_end[0] =  1. * n_samples * (-xlim[0] + p_end[0]) / (xlim[1] - xlim[0])
            p_end[1] =  1. * n_samples * (-ylim[0] + p_end[1]) / (ylim[1] - ylim[0])
            
            plt.plot([im_pos[0], p_end[0]], [im_pos[1], p_end[1]])

    if do_show:
        plt.show()
# ---

