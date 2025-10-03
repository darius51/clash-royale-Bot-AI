import numpy as np
from gymnasium import spaces
from state.representation import GameState


class ClashEnv:
    def __init__(self, controller, detectors):
        self.ctrl = controller
        self.det = detectors
        self.observation_space = spaces.Box(low=0, high=10, shape=(3,), dtype=np.int32)
        self.action_space = spaces.Discrete(5) # 0..3: joacă slot, 4: NOOP


    def reset(self, seed=None):
        # TODO: intră în arenă, pregătește meci
        s = GameState(5, None, [0,1,2,3])
        return np.array(s.to_vector(), dtype=np.int32), {}


    def step(self, action):
        # TODO: aplică acțiune via controller
        reward = 0.0
        terminated = False
        truncated = False
        obs = np.array([5, -1, 4], dtype=np.int32)
        info = {}
        return obs, reward, terminated, truncated, info