from models.controllers.keyboard import KeyboardController
from utils.plots import AnimParams, AnimTrajectory
from utils.config_space import run
import numpy as np


def main():
    model = ...
    ctrl = KeyboardController(2, amp=np.array([2.0, 0.4]))
    AnimTrajectory(
        model, ctrl, AnimParams(follow=True, view=15),
        u_maxs=[np.array([2.0, 0.6]), np.array([2.0, 0.6])],   # limit s per model
    ).show()


if __name__ == "__main__":
    run("ddr")