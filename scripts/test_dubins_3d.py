from saferl.aerospace.models.dubins import Dubins3dPlatform
import math

config = {
    'controller': {
        'actuators': [
            {
                'name': 'ailerons',
                'space': 'continuous',
                'points': 5,
            },
            {
                'name': 'elevator',
                'space': 'continuous',
                'points': 5,
            },
            {
                'name': 'throttle',
                'space': 'continuous',
                'points': 5,
            },
        ],
    },
}

agent = Dubins3dPlatform(controller='agent', config=config)
platform = Dubins3dPlatform()

agent.reset(v=100, roll=5 * (math.pi / 180))
platform.reset()

print(5 * (math.pi / 180))
continuous_action_ex = (None, 5 * (math.pi / 180), None)
continuous_action_ex2 = (None, None, None)
discrete_action_ex = (2, 2, 4)
discrete_action_ex2 = (None, None, 4)

state0 = agent.state
agent.step(1, continuous_action_ex2)
state1 = agent.state
agent.step(1, continuous_action_ex2)
state2 = agent.state
agent.step(1, continuous_action_ex2)
state3 = agent.state

1+1
