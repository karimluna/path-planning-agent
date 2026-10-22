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
    model = EvenBetterUnicycle()
    ref = static_point([-20, 10])
    ctrl = PointTracker(model, ref, PID(kp=2.0, kd=0.5), d=0.8, v_max=3.0)
    AnimTrajectory(
        model, ctrl,
        AnimParams(follow=True, view=20),
        # For nonholonomic constraint systems use u2 ∈ [0.5, 1.0) 
        # except for SkiddingCar, study more 
        u_maxs=[np.array([10., 10.])], ref=ref,
    ).show()


if __name__ == "__main__":
    main()

    