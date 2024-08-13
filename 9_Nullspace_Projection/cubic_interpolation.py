import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


'''
This file contains a Python class that provdes easy functionality for smoothly interpolating between keyframes.
Please investigate the /examples/ directory for a minimal working examples of how to use this class.

In case you have any suggestions, doubts or questions, please send an email to Steffen Puhlmann (spuhlmann@bht-berlin.de),
or even better: post your question on Moodle.
'''



class CubicInterpolation(object):

    '''
    Defining a function that specifies every motor value for every point in time can be cumbersome.
    An easier alternative is to specify only a few selected motor positions (keyframes) and the time 
    it takes to transition between these keyframes. The motor values along this transition are computed automatically.
    '''


    def __init__(self, keyframes:np.ndarray, durations:np.ndarray) -> None:
        '''
        Creates a linear interpolation between keyframes. 
        The keyframes are multidimensional vectors, stored either as lists or as numpy arrays in <keyframes>.
        There must be at least two keyframes.
        The time [in seconds] it takes to transition from a keyframe to its successor keyframe is stored in <durations>.
        The number of time durations must be exactly the number of keyframes minus one.
        All keyframes must have the same number of dimensions!

        :param keyframes:   list of keyframes. A keyframe is a vector. Alle keyframes must have the same number of dimensions.
        :param durations:   list of scalar values representing the time [in seconds] it takes to transition between neighboring keyframes.
        
        :return:            Returns initialized LinearInterpolation object
        '''

        # check for errors in parameters
        if len(keyframes) <= 1:
            print('ERROR: Cubic interpolation requires at least two keyframes!')
            quit()

        elif len(durations) != len(keyframes) - 1:
            print('ERROR: The number of time durations needs to be exactly the number of keyframes minus one!')
            quit()

        tmp_n_dim = len(keyframes[0])
        for k in keyframes:
            if len(k) != tmp_n_dim:
                print('ERROR: All keyframes need to have the same number of dimensions!')
                quit()
        # ---

        self.keyframes      = np.array(keyframes, dtype=np.float32)
        self.durations      = np.array(durations, dtype=np.float32)

        self.prepare()
    # ---


    def prepare(self) -> None:
        '''
        This methods needs to be called each time the keyframes or the durations have changed.
        It initiates quick-access variables based on current set of keyframes and time durations.
        More importantly, it computes the cubic-spline coefficients and the velocities at keyframe transitions.
        '''


        '''
        Define some informative quick-access variables for improved code-readability
        '''
        self.keyframe_times = np.insert(np.cumsum(self.durations), 0, 0)   # starting time of each keyframe (the first starts at 0.0) [seconds]
        self.n_dim          = self.keyframes.shape[1]                      # number of keyframe dimensions (identical for each keyframe)
        self.n_keyframes    = self.keyframes.shape[0]                      # total number of keyframes
        self.total_duration = np.sum(self.durations)                       # total duration of the whole keyframe interpolation [seconds]


        '''
        Compute the coefficients for the cubic spline interpolations.
        If we have n keyframes with m dimensions each, then we will have (n-1) * m many cubic splines.
        Each of these cubic splines is defined by four coefficients [a0, a1, a2, a3].
        We obtain these four parameters by taking the boundary conditions into account:
        starting time t0=0, duration tf, value at start x0, value at end xf, velocity at start dx0, and velocity at end dxf.
        '''

        # compute average velocities interpolations between neighboring keyframes (via linearization)
        # This matrix is of shape: ((n-1), m)
        self.avg_vel = self.get_avg_vel()

        # compute the velocities at the cspline transitions points
        # This matrix is of shape: ((n-1), m)
        self.start_vels, self.end_vels = self.get_transition_velocities()

        # compute cspline coefficients. 
        # This matrix is of shape: ((n-1), m, 4)
        self.coeffs = self.get_cspline_coeffs(self.durations, self.keyframes[:-1], self.keyframes[1:], self.start_vels, self.end_vels)
    # ---


    def get_avg_vel(self) -> np.ndarray:
        '''
        Computes the average velocities of keyframe interpolations via linearization (thus, to compute the average velocities, we assume a linear interpolation).
        This velocity is different for every neighboring pair of keyframes and for every keyframe dimension.
        Thus, if we have n keyframe vectors with m dimensions each, then we obtain (n-1) velocity vectors, also with m dimensions each.
        '''
        return np.array([(self.keyframes[i+1] - self.keyframes[i]) / self.durations[i] for i in range(self.n_keyframes-1)], dtype=np.float32)
    # ---


    def get_transition_velocities(self) -> np.ndarray:
        '''
        Computes the velocities at c-spline transition points.
        When we have n keyframes, then we have (n-1) c-splines, and thus, (n-2) c-spline transitions
        If the average velocities of consecutive c-splines have the same sign, the velocity at the 
        transition point is the average of these average velocities. Otherwise, the transition point velocity is zero.
        '''

        # If we have n keyframes, then we have (n-1) keyframe transitions.
        # Since every keyframe has m dimensions, we have ((n-1), m) start velocities and ((n-1), m) end velocities
        start_velocities = np.zeros((self.n_keyframes - 1, self.n_dim), dtype=np.float32)
        end_velocities   = np.zeros((self.n_keyframes - 1, self.n_dim), dtype=np.float32)

        # calculating keyframe transition velocities makes only sense if we have more than two keyframes
        if self.n_keyframes < 3: return
        
        # check whether the velocities of neighboring csplines have itendical sign
        # since we have (n-1) keyframe transitions, we will obtain (n-2) comparisons of average velocities
        identical_sign   = (np.sign(self.avg_vel[1:]) == np.sign(self.avg_vel[:-1]))

        # If the signs of neighboring mean velocities are identical, the cspline transition velocity is their average.
        # Otherwise the cspline transition velocity is zero.
        for n in range(self.n_keyframes-2):
            for m in range(self.n_dim):
                if identical_sign[n, m]:
                    avg_vel                  = (self.avg_vel[n, m] + self.avg_vel[n+1, m]) / 2.
                    end_velocities[n, m]     = avg_vel
                    start_velocities[n+1, m] = avg_vel
        
        return start_velocities, end_velocities
    # ---


    def get_cspline_coeffs(self, t_f: float, x_0: np.ndarray, x_f: np.ndarray, dx_0: np.ndarray, dx_f: np.ndarray) -> np.ndarray:
        '''
        Computes the coefficients of a 1D cubic spline, for a given set of boundary conditions.
        The 1D cubic spline is defined as   x(t) =  a_0  +  a_1*t  +  a_2 * t^2  +  a_3 *t^3
        We assume that every cspline interpolation starts at t_0 = 0. Therefore, t_f corresponds to the duration.

        :param t_f:    duration [seconds].              This matrix is of shape (n-1,)
        :param x_0:    values at starting point.        This matrix is of shape (n-1, m)
        :param x_f:    values at final time point.      This matrix is of shape (n-1, m)
        :param dx_0:   velocities at starting point.    This matrix is of shape (n-1, m)
        :param dx_f:   velocities at final point.       This matrix is of shape (n-1, m)
        
        :return:       np.array containing the cspline coefficients. 
                       This matrix is of shape ((n-1), m, 4),
                       where n is the number of keyframes, n is the number of keyframe dimensions, and 4 is the number of coefficients.
        '''

        coeffs = []
        for i in range(self.n_keyframes-1):
            a_0    = x_0[i]
            a_1    = dx_0[i]
            a_2    = 3. * (x_f[i] - x_0[i]) / t_f[i]**2  - (2. * dx_0[i] + dx_f[i]) / t_f[i]
            a_3    = -(2. / t_f[i]**3) * (x_f[i] - x_0[i]) + (1. / t_f[i]**2) * (dx_0[i] + dx_f[i])
            
            coeffs.append(np.array([a_0, a_1, a_2, a_3]).T)

        return coeffs
    # ---


    def get_cspline_pos(self, t: float, coeffs: np.ndarray) -> np.ndarray:
        '''
        Evaluates a cubic spline, defined by its coefficients [a0, a1, a2, a3] at time point t. 
        
        :param t:        time point for evaluation
        :param coeffs:   coefficients [[a_i0, a_i1, a_i2, a_i3] for in [1 .. m]], where m is the number of keyframe dimensions
        :return:         positions: array [x_i(t)], where x_i(t) = a_i0 + a_i1 * t + a_i2 * t^2 + a_i3 * t^3
        '''
        return coeffs[:, 0] + coeffs[:, 1] * t + coeffs[:, 2] * t**2 + coeffs[:, 3] * t**3
    # ---


    def get_cspline_vel(self, t: float, coeffs: np.ndarray) -> np.ndarray:
        '''
        Evaluates the derivative (i.e., velocity) of a cubic spline, defined by 
        its coefficients [a0, a1, a2, a3] at time point t. 
        
        :param t:        time point for evaluation
        :param coeffs:   coefficients [[a_i0, a_i1, a_i2, a_i3] for in [1 .. m]], where m is the number of keyframe dimensions
        :return:         velocities: array [dx_i(t)], where dx_i(t) = a_i1 + 2 * a_i2 * t + 3 * a_i3 * t^2
        '''
        return coeffs[:, 1] + 2. * coeffs[:, 2] * t + 3 * coeffs[:, 3] * t**2
    # ---


    def get_cspline_acc(self, t: float, coeffs: np.ndarray) -> np.ndarray:
        '''
        Evaluates the second derivative (i.e., acceleration) of a 
        cubic spline, defined by its coefficients [a0, a1, a2, a3] at time point t. 
        
        :param ts:       time points for evaluation
        :param coeffs:   coefficients [[a_i0, a_i1, a_i2, a_i3] for in [1 .. m]], where m is the number of keyframe dimensions
        :return:         accelerations: array [ddx_i(t)], where ddx_i(t) = 2 * a_i2 + 6 * a_i3 * t
        '''
        return 2. * coeffs[:, 2] + 6. * coeffs[:, 3] * t
    # ---


    def get_values_for_time(self, t:float) -> list:
        '''
        Returns a cubic interpolation between the closest keyframe before a specified time point and the closest keyframe after this time point.
        More specifically, thus function returns the position, the velocity, and the acceleration. 

        :param t:           time point [seconds]
        :return:            interpolated values: positions, velocities, and accelerations for each keyframe dimension
        '''

        # catch boundary cases
        if t <= 0.: return self.keyframes[0], np.zeros(self.n_dim), np.zeros(self.n_dim)
        if t >= self.total_duration: return self.keyframes[-1], np.zeros(self.n_dim), np.zeros(self.n_dim)

        # compute time difference between t and each keyframe
        time_diffs  = t - self.keyframe_times

        # extract ids from all keyframes that happen before t
        earlier_kfs = np.arange(self.n_keyframes)[time_diffs >= 0]

        # extract ID of keyframe that is closest to t and happens before t
        closest_idx   = earlier_kfs[-1]

        # get starting time of the closest past keyframe
        t_start     = self.keyframe_times[closest_idx]

        # consider coefficients of the cspline transition between the closest past keyframe and its successor
        closest_coeffs = self.coeffs[closest_idx]

        # compute the time passed since the closest keyframe in the past
        time_in_interpolation = t - t_start

        # compute positions, velocities, and accelerations
        pos = self.get_cspline_pos(time_in_interpolation, closest_coeffs)
        vel = self.get_cspline_vel(time_in_interpolation, closest_coeffs)
        acc = self.get_cspline_acc(time_in_interpolation, closest_coeffs)

        return pos, vel, acc
    # ---


    def appfront(self, keyframe:np.ndarray, duration:float) -> None:
        '''
        Inserts a new keyframe at the beginning of the already existing keyframe sequence.

        :param keyframe:  keyframe (vector) which needs to have the same dimensionality as the already existing keyframes.
        :param duration:  the time it takes to transition from this new keyframe to its successor.
        '''
        flipped_kfs    = np.flip(self.keyframes, axis=0)
        self.keyframes = np.flip(np.append(flipped_kfs, [keyframe], axis=0), axis=0)
        self.durations = np.insert(self.durations, 0, duration)
        self._update()
    # ---


    def append(self, keyframe:np.ndarray, duration) -> None:
        '''
        Inserts a new keyframe at the end of the already existing keyframe sequence.

        :param keyframe:  keyframe (vector) which needs to have the same dimensionality as the already existing keyframes.
        :param duration:  the time it takes to transition from this new keyframe to its successor.
        '''
        self.keyframes = np.append(self.keyframes, [keyframe], axis=0)
        self.durations = np.append(self.durations, duration)
        self._update()
    # ---

    
    def delete(self, kf_index:int) -> None:
        '''
        Deletes a keyframe at a given index. Also deletes the corresponding duration.
        In case the last keyframe is deleted, the duration between this last keyframe and its predecessor will be deleted.
        Otherwise, the time between this keyframe and its successor will be deleted.

        :param kf_index:   Index of keyframe that is to be deleted
        '''
        
        # check for errors in parameters
        if (kf_index < 0) or (kf_index > self.n_keyframes - 1):
            print('ERROR: Could not delete keyframe with index ' + str(kf_index) + '. Index out of bounds.')
            return

        # in the special case where we delete the last keyframe, we delete the last time duration (i.e., between predecessor and this one)
        if kf_index == self.n_keyframes-1:
            self.keyframes = self.keyframes[:-1]
            self.durations = self.durations[:-1]
            self._update()
        
        # otherwise, we delete the keyframe and the duration between this keyframe and its successor
        else:
            self.keyframes = np.delete(self.keyframes, kf_index, axis=0)
            self.durations = np.delete(self.durations, kf_index)
            self._update()
    # ---


    def plot(self, plot_linear=True) -> None:
        '''
        Plots the multidimensional keyframes at their respective time points, 
        and the cubic interpolations between neighboring keyframes.
        '''
        
        # select nice colors
        cmap1 = plt.get_cmap('Pastel1')
        cmap2 = plt.get_cmap('Set1')
        
        # check for errors in parameters
        if len(self.keyframes) <= 1: print('Interpolation requires at least two keyframes! Provided number of keyframes: ' + str(self.n_keyframes))
        
        
        # Create a figure with three subplots (3 rows, 1 column)
        fig, axs = plt.subplots(3, 1, figsize=(6, 8))
       
        # linearly sample time points
        ts  = np.linspace(-1., self.total_duration+1., 1000)
        
        # compute positions, velocities and accelerations for sample time points
        pos, vel, acc = [], [], []
        for t in ts:
            p, v, a = self.get_values_for_time(t)
            pos.append(p)
            vel.append(v)
            acc.append(a)

        pos, vel, acc = np.array(pos), np.array(vel), np.array(acc)


        # plot positions
        for d in range(self.n_dim):  
            axs[0].scatter(self.keyframe_times, self.keyframes[:, d], color=cmap1(d))
            axs[0].plot(ts, pos[:, d], color=cmap2(d)) 
        axs[0].set_title('C-Spline Positions')
        axs[0].set_xlabel('Time [s]')
        axs[0].set_ylabel('Position')

        # plot velocities
        for d in range(self.n_dim):  
            axs[1].plot(ts, vel[:, d], color=cmap2(d)) 
        axs[1].set_title('C-Spline Velocities')
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Velocity')
    
        for d in range(self.n_dim):  
            axs[2].plot(ts, acc[:, d], color=cmap2(d)) 
        axs[2].set_title('C-Spline Accelerations')
        axs[2].set_xlabel('Time [s]')
        axs[2].set_ylabel('Acceleration')

        plt.tight_layout()  # adjusting layout to prevent overlapping
        plt.show()          # display plot
    # ---

# --- class