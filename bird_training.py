import pygame

from BirdEnv import BirdEnv
from bird_training_constants import ACTION_SPACE, OBSERVATION_SPACE
from DDQN.training_constants import N_EPOCHS, GAMMA, BATCH_SIZE, EPSILON, NUM_GAMES, LEARNING_RATE, FULL_EXPLORATION_STEPS, START_EPSILON, DECAY_STEPS, FINAL_EPSILON
from DDQN.Agent import Agent
import torch
from Flappy_Bird.game_constants import BIRD_INITIAL_LOCATION
from Flappy_Bird.FlappyBird import FlappyBird
import torch.nn as nn
import os


def initialize_env(online_net_path=None, target_net_path=None):
    flappy_bird = FlappyBird()
    env = BirdEnv("Flappy_Bird/ground_fb.png", "Flappy_Bird/background_fb.png", flappy_bird)
    action_space = ACTION_SPACE
    observation_space = OBSERVATION_SPACE
    agent = Agent(N_EPOCHS, GAMMA, action_space, observation_space, BATCH_SIZE, EPSILON)

    online_net = agent.online_net
    target_net = agent.target_net

    if online_net_path:
        online_net.load_state_dict(torch.load(online_net_path))
        print("Online net state loaded")

    if target_net_path:
        target_net.load_state_dict(torch.load(target_net_path))
        print("Target net state loaded")

    return env, agent


# the function responsible for decaying epsilon
def exponential_epsilon_decay(initial_epsilon, final_epsilon, decay_steps, step):
    decay_rate = (final_epsilon / initial_epsilon) ** (1.0 / decay_steps)
    return initial_epsilon * (decay_rate ** step)


""" 
The agent begins the training process with epsilon set to 1, meaning it chooses random actions and concentrates only on exploring the env.
After gathering FULL_EXPLORATION_STEPS of experience, where each experience corresponds to a single game frame epsilon is set to START_EPSILON and is decayed
over DECAY_STEPS steps until it reaches FINAL_EPSILON.
"""


def interact_with_env(opt_online, update_threshold=4):
    pygame.init()
    games_played = 0
    total_frames = 0
    decay_epsilon = False
    decay_step = 0

    for i in range(1, NUM_GAMES + 1):
        clock, last_pipe_time = env.init_game()
        x_pipe_bottomleft, _ = env.pipe_group.sprites()[0].rect.bottomleft

        current_state = [env.bird.rect.y, env.bird.vertical_speed, abs(x_pipe_bottomleft - BIRD_INITIAL_LOCATION[0]),
                         env.pipe_group.sprites()[1].rect.top, env.pipe_group.sprites()[0].rect.bottom]
        current_state = torch.tensor(current_state).float()

        done = False
        ground_scroll = 0
        pipe_id = 0
        frames = 0
        episode_reward = 0
        games_played += 1

        while not done:
            frames += 1
            total_frames += 1

            action = agent.choose_action(current_state)
            previous_state = current_state
            current_state, reward, done, last_pipe_time, ground_scroll, pipe_id = env.step(clock, last_pipe_time,
                                                                                           ground_scroll, pipe_id, action)
            current_state = torch.tensor(current_state).float()
            agent.agent_experience.append_lists(action, previous_state, reward, current_state)
            episode_reward += reward

            if total_frames == FULL_EXPLORATION_STEPS:
                decay_epsilon = True
                agent.epsilon = START_EPSILON
                decay_step = 1

            if decay_epsilon and total_frames < DECAY_STEPS + FULL_EXPLORATION_STEPS:
                agent.epsilon = exponential_epsilon_decay(START_EPSILON, FINAL_EPSILON, DECAY_STEPS, decay_step)
                decay_step += 1

            # check if the agent should be trained
            if total_frames >= FULL_EXPLORATION_STEPS and frames % update_threshold == 0:
                agent.train_agent(opt_online, loss_fn)


        torch.save(agent.online_net.state_dict(), os.path.join(save_dir_online, f"game_{i}_online_net__score_{round(episode_reward, 3)}"))
        torch.save(agent.target_net.state_dict(), os.path.join(save_dir_target, f"game_{i}_target_net__score_{round(episode_reward, 3)}"))
        print(f"Game: {i}, Episode's total reward: {round(episode_reward, 3)}")
        env.reset()

    pygame.quit()


save_dir_online = "online_net_states_v2/"
save_dir_target = "target_net_states_v2/"
env, agent = initialize_env()
loss_fn = nn.HuberLoss()
opt_online = torch.optim.Adam(params=agent.online_net.parameters(), lr=LEARNING_RATE)
interact_with_env(opt_online)

