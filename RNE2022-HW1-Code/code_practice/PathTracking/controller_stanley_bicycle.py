import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerStanleyBicycle(Controller):
    def __init__(self, kp=0.5):
        self.path = None
        self.kp = kp

    # State: [x, y, yaw, delta, v, l]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State 
        x, y, yaw, delta, v, l = info["x"], info["y"], info["yaw"], info["delta"], info["v"], info["l"]

        # Search Front Wheel Target
        front_x = x + l*np.cos(np.deg2rad(yaw))
        front_y = y + l*np.sin(np.deg2rad(yaw))
        vf = v / np.cos(np.deg2rad(delta))
        min_idx, min_dist = utils.search_nearest(self.path, (front_x,front_y))
        target = self.path[min_idx]

        # TODO: Stanley Control for Bicycle Kinematic Model
        next_delta = 0        
        yaw = utils.angle_norm(yaw)
        theta_p = utils.angle_norm(self.path[min_idx, 2])
        theta_e = utils.angle_norm(theta_p - yaw)
        e = np.dot([x - self.path[min_idx, 0], y - self.path[min_idx, 1]], [np.cos(np.deg2rad(theta_p + 90)), np.sin(np.deg2rad(theta_p + 90))])
        delta = np.rad2deg(np.arctan(-1*self.kp*e/vf)) + theta_e

        next_delta = delta        
        return next_delta, target
