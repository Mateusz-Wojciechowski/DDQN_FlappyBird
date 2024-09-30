import numpy as np
import random
from collections import deque
import torch


class AgentExperience:
    def __init__(self, batch_size, max_buffer_size=500000):
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size

        self.actions = deque(maxlen=max_buffer_size)
        self.states = deque(maxlen=max_buffer_size)
        self.rewards = deque(maxlen=max_buffer_size)
        self.next_states = deque(maxlen=max_buffer_size)

    def sample_data(self):
        indices = random.sample(range(len(self.states)), self.batch_size)

        return (torch.stack([self.actions[i] for i in indices]),
                torch.stack([self.states[i] for i in indices]),
                torch.tensor([self.rewards[i] for i in indices], dtype=torch.float32),
                torch.stack([self.next_states[i] for i in indices]))

    def append_lists(self, action, state, reward, next_state):
        self.actions.append(torch.tensor(action, dtype=torch.long))
        self.states.append(state)
        self.rewards.append(torch.tensor(reward, dtype=torch.float32))
        self.next_states.append(next_state)

    def clear_lists(self):
        self.actions.clear()
        self.states.clear()
        self.rewards.clear()
        self.next_states.clear()
