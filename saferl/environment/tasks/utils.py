import numpy as np


def draw_from_rand_bounds_dict(rand_dict):
    draw_dict = {}
    # loop over dict keys
    for key, val in rand_dict.items():
        if type(val) == dict:
            draw = draw_from_rand_bounds_dict(val)
        else:
            if type(val) == list:
                draw = np.random.uniform(val[0], val[1])
            else:
                draw = val

        draw_dict[key] = draw

    return draw_dict
