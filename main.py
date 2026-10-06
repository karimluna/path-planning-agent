from models.kinematic.unicycle import Unicycle
from models.kinematic.ddr import DifferentialDriveRobot
from models.kinematic.car import SimpleCar


from models.controllers.keyboard import KeyboardController
from utils.plots import AnimParams, AnimTrajectory


def main():
    model = SimpleCar()
    ctrl = KeyboardController(model.n_control, du=0.25)
    AnimTrajectory(model, ctrl, AnimParams()).show()


if __name__ == "__main__":
    main()