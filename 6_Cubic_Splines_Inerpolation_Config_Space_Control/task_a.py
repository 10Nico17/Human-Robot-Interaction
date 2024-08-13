'''
--------------------------
CUBIC SPLINE INTERPOLATION
--------------------------


TASK A - IMPLEMENTING C-SPLINES
-------------------------------

What this task is about:
    In order to prevent sudden jumps, overshoots, and oscillations, we want the robot to smoothly transition between keyframes. 
    For this, we dot only need to determine the positions at the keyframes, but also the velocities. 
    We can fit a cubic polynomial between two consecutive keyframes. Having four unknowns and four boundary
    conditions (start/end positions and start/end velocities), we can analytically solve for their coefficients.
    In this task, you will implement cubic spline interpolation. Only if this works well, you may move on to the next task.

What you need to do in the real world:
    * Nothing. This task is only concerned with software.
    
What you need to implement:
    * Have a look at the file cubic_interpolation and implement your code there.
    * Read and understand the code in this script. You will need it for task b.

What you should see if your code is correct:
    * This script demonstrates how to create a cubic spline interpolation. In this case, we have three keyframes
      whith four dimensions each. You can also specify (in the list <durations>) the time it takes to transition 
      between consecutive keyframes.
    * You should see three plots: the first showing the keyframes (dots) and the interpolated values (solid lines),
      the second showing the velocities (these should look like parabulas), 
      and the third showing the accelerations (which should look like lines).
'''



import time                              # For accessing the current time
import numpy as np                       # Fast math library
from cubic_interpolation import *        # Custom python package for cubic spline interpolation


# Here, we define four different keyframes.
# All keyframes must have the same number of dimensions. In this case, all four keyframes have three dimensions.
kf_1 = [0, 0, 0, 0]
kf_2 = [1, 2, 3, 3]
kf_3 = [1, 3, 3, 2]

# We now put three keyframes in a list, starting with kf_1 and ending with kf_3
keyframes = [kf_1, kf_2, kf_3]

# Here, we specify the time (in seconds) it takes to transition between consecutive keyframes.
# Since there are three keyframes in the list above, there can only be two (n_keyframes-1) time durations.
durations = [1, 2]

# Here, we create an object <cin> of class CubicInterpolation. 
# This object provides all relevant functionality for cubic interpolation between the keyframes.
# The constuctor takes as arguments a list of keyframes and a list of durations. 
cin = CubicInterpolation(keyframes, durations)



# Here is how we get interpolated values for some specific point in time.
# Cubic splines determine positions, velocities, and accelerations.
pos, vel, acc = cin.get_values_for_time(1.5)

print('position:    ', np.round(pos, 2))
print('velocity:    ', np.round(vel, 2))
print('acceleration:', np.round(acc, 2))


# Plot the cspline interpolation between the three keyframes.
# This will plot the positions, velocities and accelerations.
cin.plot()

