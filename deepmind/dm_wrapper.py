#deepmind/dm_wrapper.py 
import numpy as np
from dm_control import suite

class GeneralizedDMControl:
    def __init__(self, domain_name, task_name, seed=42):
        self.domain_name = domain_name
        self.task_name = task_name
        self.random_state = np.random.RandomState(seed)
        self.env = suite.load(domain_name, task_name, task_kwargs={'random': seed})
        self.physics = self.env.physics
        self.original_masses = self.physics.model.body_mass.copy()
        self.original_friction = self.physics.model.geom_friction.copy()

    def set_physics(self, mass_scale=1.0, friction_scale=1.0):
        self.physics.model.body_mass[:] = self.original_masses * mass_scale
        self.physics.model.geom_friction[:] = self.original_friction * friction_scale

    def flatten_observation(self, time_step):
        obs = time_step.observation
        return np.concatenate([np.atleast_1d(v) for v in obs.values()])

    def generate_trajectory(self, n_steps=100, mass_scale=1.0, friction_scale=1.0):
        self.set_physics(mass_scale, friction_scale)
        time_step = self.env.reset()
        traj = [self.flatten_observation(time_step)]
        action_spec = self.env.action_spec()
        
        for _ in range(n_steps):
            action = self.random_state.uniform(
                action_spec.minimum, action_spec.maximum, size=action_spec.shape
            )
            time_step = self.env.step(action)
            traj.append(self.flatten_observation(time_step))
            
        return np.stack(traj)