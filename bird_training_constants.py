from Flappy_Bird.game_constants import BIRD_INITIAL_LOCATION

ACTION_SPACE = 2  # jump or no jump
OBSERVATION_SPACE = 5  # y coord, vertical speed, x distance to nearest pipe, x speed of the pipe, height of the upper pipe, height of the lower pipe
FRAME_REWARD = 0.002
COLLISION_REWARD = -1
PIPE_PASSED_REWARD = 0.2
BIRD_INITIAL_Y = BIRD_INITIAL_LOCATION[1]
BIRD_INITIAL_VERTICAL_SPEED = 0
WRONG_HEIGHT_REWARD = -0.002



