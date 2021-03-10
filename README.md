# Reinforcement Learning for Simple Robot and Vehicle

This project contains Reinforcement Learning (RL) simulations for 
either a simple differential-driven robot avoiding obstacles or a 
simple vehicle aims to park in Cartesian x-y coordinate. 
An example of DQN training is also built to demo the simulations.

This project is dedicated to fulfill Versioning and Testing assignments 
of WASP Software Engineering and Cloud Computing. 

## Installation

Clone or download the entire package:
```
git clone https://github.com/mfaris0910/wasp_software_project.git
cd wasp_software_project
```

Then, install the required libraries (e.g. numpy, shapely, keras). Or run it in a virtual environment or container (e.g. Docker).
```
pip install -r requirements.txt
```


## Main usage

You can go to the main file where you will find two different environments to run: 
1) Robot, or 
2) Vehicle. 

This can be done by uncomment #env  = SimpleRobotEnv() to 
choose Robot or #env  = SimpleVehicle() for Vehicle and comment the 
other one. Once you do this you can run the main and a figure will 
appear to show the simulations.
```
python main.py
```

## Testing

This repository is also equipped with testing functions for the robot 
and vehicle, respectively. 
For robot, you can run: 
```
python robot_testing.py
```
while for vehicle:
```
python vehicle_testing.py
```
These tests are done automatically 
and built based on unittest library. 

## Support
Please either contact Ahmad Terra (terra@kth.se) or Muhammad Faris 
(farism@chalmers.se)
