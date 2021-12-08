import numpy as np

from saferl.aerospace.models.cwhspacecraft.platforms.oriented import CWHSpacecraftOriented2d
from saferl.environment.models.platforms import AgentController

controller = {
    'class': AgentController,
    'actuators': [
        {
            'name': 'thrust',
        },
        {
            'name': 'reaction_wheel'
        }
    ]
}

platform = CWHSpacecraftOriented2d(name='test', controller=controller, n=0)

turn_time = 135
thrust_time = 5

platform.step(None, 2, (0, 1))
platform.step(None, turn_time, (0, 0))
platform.step(None, 2, (0, -1))

platform.step(None, thrust_time, (1, 0))

expected_theta = 2*np.deg2rad(1)*0.5*(2**2) + np.deg2rad(2) * (turn_time)

thrust_dist = 1/12 * 1/2 * thrust_time**2
x_err = abs(platform.x - thrust_dist*np.cos(platform.state.theta))
y_err = abs(platform.y - thrust_dist*np.sin(platform.state.theta))
theta_err = abs(platform.theta - expected_theta) % (2*np.pi)
assert x_err < 0.01, f"x value error = {x_err}"
assert y_err < 0.01, f"y value error = {y_err}"
assert theta_err < np.deg2rad(0.2), f"theta value error = {theta_err}"


print(platform.state.vector)
print("done")
