# DDQN FlappyBird



https://github.com/user-attachments/assets/c9870a12-3afb-422b-bad6-6ff378632a45



This repository is the implementation of Double Deep Q Network Algorithm presented in the following paper: https://arxiv.org/abs/1509.06461
Using pygame I have also implemented a simple FlappyBird game from scratch on which the performance of the algorithm is tested.
The FlappyBird code as well as the custom enviroment is not present in this repository

## FlappyBird enviroment 
The bird is able to perform two actions, it chooses whether to jump or do nothing in a particular game frame
Its state is described using a 5 element vector: y coord, vertical speed, x distance to nearest pipe, x speed of the pipe, height of the upper pipe and height of the lower pipe
The reward system is presented in bird_training_constants.py

## DDQN Training approach
Initially epsilon is set to 1, which results in full env exploration. After 20000 frames of experience gathered it is changed to 0.1 and then decayed over 3000000 until it reaches 0.0001. 
This ensures that the experience buffer is partially filled with all kinds of experiences.
Traing starts the moment we start decaying the epsilon, the online network is updated every 4 frames (steps), the online net is synchronized after 1000 online net updates in other words 4000 steps.
After each episodes states of both nets are saved.
All the hyperparameters have been choosen experimentally and ones presented are those I achieved highest results with.

## DDQN performance
The training performace, specifically average reward across 10 preciding episodes are presented below.
I was able to achieve a few optimal policies, following which the agent is able to infinitely play the game.

![average_reward_plot](https://github.com/user-attachments/assets/2bc8126e-41d0-49d4-8886-fde6c362afae)
