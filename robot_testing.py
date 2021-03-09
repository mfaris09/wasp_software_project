import unittest
import numpy as np
from simple_robot import SimpleRobot

class testing_robot(unittest.TestCase):
    robot = SimpleRobot(0., 0., 0.,)
    def test_set_robot_position(self):
        #testing if the robot position is updated properly
        for i in range(-5, 5, 2):
            self.robot.set_robot_position(i, i, i*np.pi)
            self.assertEquals(self.robot.pos_x, i)
            self.assertEquals(self.robot.pos_y, i)
            self.assertEquals(self.robot.phi, i*np.pi)
    
    def test_init(self):
        #testing if the rotation does not add more components
        self.assertEquals(len(self.robot.init_robot_body), 4)
        self.assertEquals(len(self.robot.obstacle_map), self.robot.n_direction)