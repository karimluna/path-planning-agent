from models.kinematic.unicycle import (
    Unicycle, 
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

import numpy as np


FAMILIES = {
    "unicycle": dict(
        models=lambda: [Unicycle(), EvenBetterUnicycle()],
        amp=[2.0, 1.0],
        u_maxs=[[3.0, 2.0], [3.0, 2.0]],
        view=10,
    ),
    "ddr": dict(
        models=lambda: [DifferentialDriveRobot(), SmoothDDR()],
        amp=[10.0, 10.0],
        u_maxs=[[10.0, 10.0], [10.0, 10.0]],
        view=10,
    ),
    "car": dict(
        models= lambda: [SimpleCar(), ContinuousSteeringCar(), SteeringAccelCar()],
        amp=[2.0, 0.4],
        u_maxs=[[2.0, 0.6], [2.0, 1.0], [2.0, 2.0]],
        view=10.0
    )
}

def run(family: str, only: int | None = None):
    from models.controllers.keyboard import KeyboardController
    from utils.plots import AnimParams, AnimTrajectory
    f = FAMILIES[family]
    models, u_maxs = f["models"](), [np.array(u) for u in f["u_maxs"]]
    if only is not None:
        models, u_maxs = [models[only]], [u_maxs[only]]
    ctrl = KeyboardController(2, amp=np.array(f["amp"]))
    AnimTrajectory(models, ctrl, AnimParams(follow=True, view=f["view"]),u_maxs=u_maxs).show()

