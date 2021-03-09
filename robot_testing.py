import unittest
import numpy as np
from simple_robot import SimpleRobot, SimpleRobotEnv

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
            
    def test_get_parts(self):
        #test if get_parts return the correct number of robot's part
        robot_body    = self.robot.get_robot_body()
        robot_sensors = self.robot.get_robot_sensors()
        self.assertEqual(len(robot_body), 4, 'quantity of robot body incorrect')
        self.assertEqual(len(robot_sensors), self.robot.n_direction, 'quantity of robot sensors incorrect')
        
class testing_environment(unittest.TestCase):
    env = SimpleRobotEnv()
    def test_init(self):
        self.assertIsNone(self.env.fig, 'fig is not None')
        self.assertIsNone(self.env.ax,  'ax is not None')
        self.assertGreater(self.env.xlim[1], self.env.xlim[0], 'x-axis is not define properly, xmax should > xmin')
        self.assertGreater(self.env.ylim[1], self.env.ylim[0], 'y-axis is not define properly, ymax should > ymin')
        self.assertGreater(self.env.dt, 0., 'timestep must be greater than zero')
        self.assertGreater(len(self.env.obstacles), 0, 'there must be obstacles created to be avoided by the robot')
        
    def test_render(self):
        self.env.render(hold=False)
        self.assertIsNotNone(self.env.fig, 'fig is None')
        self.assertIsNotNone(self.env.ax,  'ax is None')
        
    def test_get_random_position(self):
        #test if the random position is correct
        self.assertGreater(len(self.env.target_list), 0, 'no target position is available')
        for i in range(100):
            pos_x, pos_y, phi = self.env.get_random_position()
            self.assertIn([pos_x, pos_y], self.env.target_list, 'position is not in target list')