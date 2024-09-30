from DDQN.AgentExperience import AgentExperience
from DDQN.QNet import QNet
import random
import torch
from DDQN.training_constants import BATCH_SIZE



class Agent:
    def __init__(self, n_epochs, gamma, n_actions, input_dims, batch_size, epsilon=0.1,
                 target_update_threshold=1000, epsilon_end=0.0001, decay_steps=3000000):
        self.agent_experience = AgentExperience(batch_size)
        self.online_net = QNet(n_actions, input_dims)
        self.target_net = QNet(n_actions, input_dims)
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.target_update_threshold = target_update_threshold
        self.steps_since_update = 0


    def choose_action(self, state):
        if random.random() < self.epsilon:
            # choosing random action
            action = random.randint(0, self.n_actions - 1)
        else:
            # choosing action according to the current policy
            action = torch.argmax(self.online_net(state))
            action = action.item()

        return action

    def train_agent(self, optim, loss_fn):
        if len(self.agent_experience.actions) < BATCH_SIZE:
            return

        actions, states, rewards, next_states = self.agent_experience.sample_data()

        for _ in range(self.n_epochs):
            states_for_training = states
            actions_for_training = actions.view(-1, 1).long()
            rewards_for_training = rewards.unsqueeze(1)
            next_states_for_training = next_states

            online_net_output = self.online_net(states_for_training)
            next_actions = torch.argmax(self.online_net(next_states_for_training), dim=1).unsqueeze(1)
            target_net_output = self.target_net(next_states_for_training).gather(1, next_actions)

            online_net_output = torch.gather(online_net_output, 1, actions_for_training)

            online_net_loss = loss_fn((rewards_for_training + self.gamma * target_net_output).detach(),
                                      online_net_output)

            optim.zero_grad()
            online_net_loss.backward()
            optim.step()

            self.steps_since_update += 1
            if self.steps_since_update == self.target_update_threshold:
                self.steps_since_update = 0
                # synchronizing target net with online net
                self.target_net.load_state_dict(self.online_net.state_dict())

