import numpy as np
import matplotlib.pyplot as plt     # matplotlib is for plotting stuff
import seaborn as sns               # seaborn is for nicer looking plots
sns.set()                           # activate seaborn



def get_cspline_coeffs(t_f: float, x_0: np.ndarray, x_f: np.ndarray, dx_0: np.ndarray, dx_f: np.ndarray) -> np.ndarray:
    '''
    Computes the coefficients of a cubic spline, for a given set of boundary conditions.
    The cubic spline is defined at   x(t) =  a_0  +  a_1*t  +  a_2 * t^2  +  a_3 *t^3
    
    :param t_f:    duration in seconds
    :param x_0:    pose at starting point
    :param x_f:    pose at final point
    :param dx_0:   velocity at starting point
    :param dx_f:   velocity at final point
    :return:       np.array containing coefficients [a_0, a_1, a_2, a_3
    '''
    a_0 = x_0
    a_1 = dx_0
    a_2 = (3. / t_f**2) * (x_f - x_0)  -  (2. / t_f) * dx_0  -  (1. / t_f) * dx_f
    a_3 = -(2. / t_f**3) * (x_f - x_0) + (1. / t_f**2) * (dx_0 + dx_f)

    return np.array([a_0, a_1, a_2, a_3])
# ---



def eval_cspline_pos(ts: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    '''
    Evaluates a cubic spline, defined by its coefficients [a0, a1, a2, a3] at time points ts. 
    
    :param ts:       time points for evaluation
    :param coeffs:   coefficients [a_0, a_1, a_2, a_3]
    :return:         positions: x_t = a_0 + a_1 * t + a_2 * t^2 + a_3 * t^3,  for t in ts
    '''
    return coeffs[0] + coeffs[1] * ts + coeffs[2] * ts**2 + coeffs[3] * ts**3
# ---



def eval_cspline_vel(ts: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    '''
    Evaluates the derivative (i.e., velocity) of a cubic spline, defined by 
    its coefficients [a0, a1, a2, a3] at time points ts. 
    
    :param ts:       time points for evaluation
    :param coeffs:   coefficients [a_0, a_1, a_2, a_3]
    :return:         velocities: dx_t = a_1 + 2 * a_2 * t + 3 * a_3 * t^2,  for t in ts
    '''
    return coeffs[1] + 2. * coeffs[2] * ts + 3 * coeffs[3] * ts**2
# ---



def eval_cspline_acc(ts: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    '''
    Evaluates the second derivative (i.e., acceleration) of a 
    cubic spline, defined by its coefficients [a0, a1, a2, a3] at time points ts. 
    
    :param ts:       time points for evaluation
    :param coeffs:   coefficients [a_0, a_1, a_2, a_3]
    :return:         accelerations: ddx_t = 2 * a_2 + 6 * a_3 * t,  for t in ts
    '''
    return 2. * coeffs[2] + 6 * coeffs[3] * ts
# ---



def plot_cspline(t_f: float, coeffs: np.ndarray, new_fig=True, do_show=True, n_pts=1000) -> None:
    '''
    Plots a cspline (position), its derivative (velocity), and its 2nd order derivative (acceleration).
    
    :param t_f:      duration
    :param coeffs:   coefficients [a_0, a_1, a_2, a_3]
    :param new_fig:  if True, creates a new figure
    :param do_show:  if True, forces end of plot
    :param n_pts:    number of uniformly sampled poins on spline for plotting
    '''
    ts  = np.linspace(0, t_f, n_pts)
    
    pos = eval_cspline_pos(ts, coeffs)
    vel = eval_cspline_vel(ts, coeffs)
    acc = eval_cspline_acc(ts, coeffs)  
    
    if new_fig: plt.figure(figsize=(20,4))
    
    # plot position x
    plt.subplot(1, 3, 1)
    plt.plot(ts, pos)
    plt.xlabel('time $t$')
    plt.ylabel('position $x$')
    
    # plot velocity dot{x}
    plt.subplot(1, 3, 2)
    plt.plot(ts, vel)
    plt.xlabel('time $t$')
    plt.ylabel('velocity $\dot{x}$')
    
    # plot acceleration ddot{x}
    plt.subplot(1, 3, 3)
    plt.plot(ts, acc)
    plt.xlabel('time $t$')
    plt.ylabel('acceleration $\ddot{x}$')
    
    if do_show: plt.show()
# ---



def plot_ndim_cspline(t_f: float, coeffs: np.ndarray) -> None:
    '''
    Plots cspline for each pose dimension. 
    (Each dimension of the end effector has its own cspline, but all of them have
    the same starting time and duration)
    
    :param tf:      duration in seconds
    :param coeffs:  coefficients [[a_0, a_1, a_2, a_3], [a_0, a_1, a_2, a_3], ..] for each pose dimension
    '''
    ndim = coeffs.shape[0]
    for i in range(ndim):
        plot_cspline(0, t_f, coeffs[i], new_fig=(i==0), do_show=(i==ndim-1))
# ---



def get_ndim_cspline_coeffs(t_f: float, xs_0: np.ndarray, xs_f: np.ndarray, dxs_0: np.ndarray, dxs_f: np.ndarray) -> np.ndarray:
    '''
    Generates multiple csplines by determining their coefficients - one for each pose dimension. 
    All of these csplines have the same starting time and duration.

    :param t_f:   duration in seconds
    :xs_0:        pose at starting time
    :xs_f:        pose at final time
    :dxs_0:       velocity at starting time
    :dxs_f:       velocity at final time
    :return:      coefficients [[a_0, a_1, a_2, a_3], [a_0, a_1, a_2, a_3], ..] for each pose dimension
    '''
    coeffs = np.zeros((xs_0.shape[0], 4))
    
    for i, (x_0, x_f, dx_0, dx_f) in enumerate(zip(xs_0, xs_f, dxs_0, dxs_f)):
        coeffs[i] = get_cspline_coeffs(t_f, x_0, x_f, dx_0, dx_f)
    
    return coeffs
# ---



def eval_ndim_cspline_pos_vel_acc(ts: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    '''
    Computes position and velocity for a set of csplines at multiple time points.
    
    :param ts:      time points
    :param coeffs:  coefficients [[a_0, a_1, a_2, a_3], [a_0, a_1, a_2, a_3], ..] for each pose dimension
    '''
    pos = np.zeros((coeffs.shape[0], ts.shape[0]))
    vel = np.zeros((coeffs.shape[0], ts.shape[0]))
    acc = np.zeros((coeffs.shape[0], ts.shape[0]))
    
    for i in range(coeffs.shape[0]):
        pos[i] = eval_cspline_pos(ts, coeffs[i])
        vel[i] = eval_cspline_vel(ts, coeffs[i])
        acc[i] = eval_cspline_acc(ts, coeffs[i])
    
    return pos, vel, acc
# ---



def create_multi_point_cspline_interpolation(poses: np.ndarray, durations: np.ndarray) -> np.ndarray:
    '''
    Creates cubic spline interpolation between a set of poses (n-dim points).
    
    :param: poses:      list of n-dim poses.
    :param: durations:  list of interpolation durations
    :return:            list of list of coefficients: (m_transitions x n_dim x 4)
    '''
    coeffs = []
    
    # determine number of pose dimensions
    ndim  = poses[0].shape[0]
    
    # determine avg velocities of splines through linearization
    avg_v = [(poses[i+1] - poses[i]) / durations[i] for i in range(len(poses)-1)]

    # create csplines
    for i in range(len(poses)-1):
            
        # extract start and end pose
        x_0 = poses[i]
        x_f = poses[i+1]
        
        # determine start velocities
        dx_0 = np.zeros(ndim)

        # start velocity is average of linear velocity of this spline and 
        # linear velocity of prev spline, if the signs of these velocities are the same
        if i > 0:
            for d in range(ndim):
                if np.sign(avg_v[i-1][d]) == np.sign(avg_v[i][d]):
                    dx_0[d] = (avg_v[i-1][d] + avg_v[i][d]) / 2.
                
        # determine end velocities
        dx_f = np.zeros(ndim)

        # end velocity is average of linear velocity of this spline and 
        # linear velocity of next spline, if the signs of these velocities are the same
        if i < len(poses)-2:
            for d in range(ndim):
                if np.sign(avg_v[i][d]) == np.sign(avg_v[i+1][d]):
                    dx_f[d] = (avg_v[i][d] + avg_v[i+1][d]) / 2.
        
        coeffs.append(get_ndim_cspline_coeffs(durations[i], x_0, x_f, dx_0, dx_f))
    
    return coeffs
# ---



def eval_multi_point_spline_interpolation_helper(t:float, poses: np.ndarray, durations: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    '''
    Interpolates a pose and its velocity at time t, based on a sequence of multi-variate csplines.
    
    :param t:          time point
    :param poses:      poses between which this function interpolates
    :param durations:  interpolation durations
    :param coeffs:     list of list of coefficients: (m_transitions x n_dim x 4)
    :return:           interpolated pose and velocity at time t
    '''
    # derive start times and end times based on durations
    ts_f  = np.cumsum(durations)
    ts_0  = np.insert(ts_f, 0, 0)[:-1]

    # edge cases: time t is outside of time interval covered by spline interpolation
    if t < ts_0[0]:
        return poses[0]
    elif t > ts_f[-1]:
        return poses[-1]

    # determine index of active spline based on time t
    curr_idx = np.argmax(t <= ts_f)
    
    
    return eval_ndim_cspline_pos_vel_acc(np.array([t-ts_0[curr_idx]]), coeffs[curr_idx])
# ---



def eval_multi_point_spline_interpolation(ts: np.ndarray, poses: np.ndarray, durations: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    '''
    Interpolates poses and velocities at multiple time points, based on a sequence of multi-variate csplines.
    
    :param ts:         list of time point
    :param poses:      poses between which this function interpolates
    :param durations:  interpolation durations
    :param coeffs:     list of list of coefficients: (m_transitions x n_dim x 4)
    :return:           list interpolated poses and list of interpolated velocities
    '''
    interp_poses = [eval_multi_point_spline_interpolation_helper(t, poses, durations, coeffs) for t in ts]
    pos = np.array(interp_poses)[:, 0]
    vel = np.array(interp_poses)[:, 1]
    acc = np.array(interp_poses)[:, 2]

    return pos, vel, acc
# ---    



def plot_multi_point_cspline_interpolation(poses, durations, coeffs, n_pts=100) -> None:
    '''
    Plots a sequence of multi-variate csplines.
    
    :param poses:      poses between which this function interpolates
    :param durations:  interpolation durations
    :param coeffs:     list of list of coefficients: (m_transitions x n_dim x 4)
    :param n_pts:      number of uniformly sampled plotting points on cspline
    '''
    ts            = np.linspace(0, np.sum(durations), n_pts)
    pos, vel, acc = eval_multi_point_spline_interpolation(ts, poses, durations, coeffs)
    time_points  = np.insert(np.cumsum(durations), 0, 0)
    
    plt.figure(figsize=(15,4))
    
    # plot poses as dots
    for i, pose in enumerate(poses):
        for elem in pose:
            plt.plot(time_points[i], elem, 'o', color='red')
        
    # plot cspline interpolation separately 
    # for the different pose dimensions
    xs = np.linspace(0, np.sum(durations), n_pts)
    for dim in range(pos.shape[1]):
        plt.plot(xs, pos[:, dim])
# ---