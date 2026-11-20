import numpy as np

SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
PIPE_WIDTH = 52
BIRD_HEIGHT = 24
BIRD_X = 57

def extract_features(observation):
    bird_x = BIRD_X / SCREEN_WIDTH
    pipe_width = PIPE_WIDTH / SCREEN_WIDTH

    for i in range(0, 9, 3):
        pipe_x = observation[i]
        pipe_top = observation[i + 1]
        pipe_bottom = observation[i + 2]

        if pipe_x + pipe_width > bird_x:
            break

    bird_y = observation[9] + (BIRD_HEIGHT / 2) / SCREEN_HEIGHT
    bird_velocity = observation[10]

    gap_centre = (pipe_top + pipe_bottom) / 2
    distance_y = bird_y - gap_centre

    return (pipe_x, distance_y, bird_velocity)

def discretise(value, minimum, maximum, num_bins):
    boundaries = np.linspace(minimum, maximum, num_bins + 1)[1:-1]
    return int(np.digitize(value, boundaries))

def get_state(observation, use_velocity=True):
    pipe_x, distance_y, bird_velocity = extract_features(observation)
    x_bin = discretise(pipe_x, 0, 1, 20)
    y_bin = discretise(distance_y, -1, 1, 20)
    v_bin = discretise(bird_velocity, -1, 1, 10)

    if use_velocity == True:
        return (x_bin, y_bin, v_bin)
    else:
        return (x_bin, y_bin)
    