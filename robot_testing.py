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
        
    def test_move(self):
        #testing if the robot moves in x-axis
        self.robot.set_robot_position(0., 0., 0.)
        for i in range(5):
            self.robot.move(1.0, 0., 0.1)
            self.assertGreater(self.robot.pos_x, 0., 'robot does not move positively in x-axis')
        self.robot.set_robot_position(0., 0., 0.)
        for i in range(5):
            self.robot.move(-1.0, 0., 0.1)
            self.assertLess(self.robot.pos_x, 0., 'robot does not move negatively in x-axis')
        #testing if the robot moves in y-axis
        self.robot.set_robot_position(0., 0., np.pi/2)
        for i in range(5):
            self.robot.move(1.0, 0., 0.1)
            self.assertGreater(self.robot.pos_y, 0., 'robot does not move positively in y-axis')
        self.robot.set_robot_position(0., 0., np.pi/2)
        for i in range(5):
            self.robot.move(-1.0, 0., 0.1)
            self.assertLess(self.robot.pos_y, 0., 'robot does not move negatively in y-axis')
        #testing if the robot keeps its position during rotataion
        self.robot.set_robot_position(0., 0., 0.)
        for i in range(-12,12):
            self.robot.move(0.0, i*np.pi/6, 0.1)
            self.assertEqual(self.robot.pos_x, 0., 'robot moves in x axis')
            self.assertEqual(self.robot.pos_y, 0., 'robot moves in y axis')