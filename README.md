# Simulate, control, and analyze dynamic models

An analytic system to see how a single mobile robot (e.g., a car, DDR, or unicycle) discovers a trajectory based on feedback.


The goal is to make these agents autonomous but we start with a simple Two-Degree of Freedom (2DOF) controller. 

In control theory, a system has two degrees of freedom when you can tune its response to setpoint changes (reference tracking) completely independently from its response to external disturbances and modeling errors. Since feedforward path shapes the tracking results without altering the closed loop poles, it can be tuned without risking stability governed by the PD. This mechanism is pretty similar to residual reinforcement learning, thus in the future this method will be used.

Examples

