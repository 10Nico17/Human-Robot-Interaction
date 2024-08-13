import numpy as np


'''
---------------
DYNAMICS MODELS
---------------

In here, you will implement two simple dynamics models: a friction model and a gravity model.
You will later use these models to compensate for the actual friction and gravity acting on the LAURA robot in reality.
'''




def predict_friction(dq):
    '''
    Predicts the torques acting on the joints due friction, for example inside their gearboxes.
    We assume a proportional relationship between joint velocity and friction.

    :param dq:      joint velocities : np.ndarray
    :return:        frictional torques : np.ndarray
    '''

    '''
    ATTENTION!
    Friction coefficients are usually very(!) small. 
    Start with values around 0.001 and than gradually increase or decrease them, depending on the robot's behavior.
    '''
    coeffs = np.array([0.0007, 0.001, 0.005, 0.0007])
    return -coeffs * dq
# ---



def predict_gravity(q, ee_mass, ee_com_radius):
    '''
    Predicts the torques acting on the joints due to gravity. Gravity acts on all parts of the robot.
    However, we simplify the robot's links to center of masses (COM) which are located at a specific distance
    to their respecitve predecessor joints. Depending on its location within the robot's kinematic chain, a COM can act
    simultaneously on multiple joints.

    :param q:               configuration : np.ndarray
    :param ee_mass:         mass at the end effector [grams] : float
    :param ee_com_radius:   distance between the end effector's COM and the screw at the end of link 3 : float
    :return                 gravitational torques : np.ndarray
    '''

    # helper function for improved code readability
    def c(*qs, offset=np.pi/2.): return np.cos(np.sum(qs) + offset)

    # unpack q for improved code readability
    _, q1, q2, q3 = q

    # gravity constant
    g = 9.81 # N / kg

    # Robot's link masses [kilograms]
    MASS_LINK_1 = 36.4  / 1000.
    MASS_LINK_2 = 38.4  / 1000.
    MASS_LINK_3 = (22.7 + ee_mass) / 1000.

    # Robot's link lengths [meters]
    LINK_LENGTH_0       = 89.5 / 1000.
    LINK_LENGTH_1       = 72.5 / 1000.
    LINK_LENGTH_2       = 72.5 / 1000.
    LINK_LENGTH_3       = 21   / 1000.

    # distance between joint and center of mass [millimeters]
    COM_RADIUS_LINK_1 = 50 / 1000.
    COM_RADIUS_LINK_2 = 50 / 1000.
    COM_RADIUS_LINK_3 = (20 + ee_com_radius) / 1000.

    # torque exerted on each joint due to link 1
    lever_1 = np.array([0, 
                        c(q1) * COM_RADIUS_LINK_1,
                        0,
                        0])
                      
    # torque exerted on each joint due to link 2
    lever_2 = np.array([0, 
                        (c(q1) * LINK_LENGTH_1 + c(q1, q2) * COM_RADIUS_LINK_2),
                        (                        c(q1, q2) * COM_RADIUS_LINK_2),
                        0])

    # torque exerted on each joint due to link 2
    lever_3 = np.array([0, 
                        (c(q1) * LINK_LENGTH_1 + c(q1, q2) * LINK_LENGTH_2 + c(q1, q2, q3) * COM_RADIUS_LINK_3),
                        (                        c(q1, q2) * LINK_LENGTH_2 + c(q1, q2, q3) * COM_RADIUS_LINK_3),
                        (                                                    c(q1, q2, q3) * COM_RADIUS_LINK_3)])

    tau_1 = -lever_1 * g * MASS_LINK_1
    tau_2 = -lever_2 * g * MASS_LINK_2
    tau_3 = -lever_3 * g * MASS_LINK_3

    return tau_1 + tau_2 + tau_3
# ---