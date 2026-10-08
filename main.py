from models.kinematic.unicycle import (
    Unicycle, 
    BetterUnicycle, 
    EvenBetterUnicycle
)
from models.kinematic.ddr import (
    DifferentialDriveRobot,
    SmoothDDR
)
from models.kinematic.car import (
    SimpleCar,
    ContinuousSteeringCar,
    SteeringAccelCar
)


from models.controllers.keyboard import KeyboardController
from utils.plots import AnimParams, AnimTrajectory
import numpy as np


def main():
    models = [SimpleCar(), SteeringAccelCar()]      
    ctrl = KeyboardController(2, amp=np.array([2.0, 0.4]))
    AnimTrajectory(
        models, ctrl, AnimParams(follow=True, view=15),
        u_maxs=[np.array([2.0, 0.6]), np.array([2.0, 0.6])],   # limit s per model
    ).show()


if __name__ == "__main__":
    main()