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
    model = SteeringAccelCar()
    ctrl = KeyboardController(model.n_control, amp=np.array([2.0, 0.4]))
    AnimTrajectory(model, ctrl, AnimParams()).show()


if __name__ == "__main__":
    main()