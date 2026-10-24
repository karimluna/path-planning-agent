import numpy as np
from models.controllers.pid import PID
from models.controllers.tracking import PointTracker, circle, static_point
from models.any.car import (
    SimpleCar,
    ContinuousSteeringCar,
    SteeringAccelCar,
    SkiddingCar, 
    )
from models.any.ddr import (
    DifferentialDriveRobot,
    SmoothDDR
)
from models.any.unicycle import (
    Unicycle, 
    BetterUnicycle,
    EvenBetterUnicycle
)
from utils.plots import AnimParams, AnimTrajectory


def main():
    # model = SimpleCar()
    model = SkiddingCar()
    ref = static_point([100, 50])
    ctrl = PointTracker(model, ref, PID(kp=1.5, kd=0.7), d=0.5, v_max=3.0)
    AnimTrajectory(
        model, ctrl,
        AnimParams(follow=True, view=20),
        # For nonholonomic constraint systems use something like u2 ∈ [0.5, 1.0) 
        # except for SkiddingCar, it uses a single control steering angle ∼0.6, study more 
        u_maxs=[np.array([0.6])], 
        ref=ref,
    ).show()


if __name__ == "__main__":
    main()

    