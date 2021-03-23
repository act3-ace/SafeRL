import numpy as np
import io

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

def numpy_to_matlab_txt(mat, name=None, output_stream=None):
    ret_str = False
    if output_stream is None:
        output_stream = io.StringIO
        ret_str = True

    if name:
        output_stream.write('{} = '.format(name))

    output_stream.write('[\n')
    np.savetxt(output_stream, mat, delimiter=',', newline=';\n')
    output_stream.write('];\n')

    if ret_str:
        return output_stream.getvalue()
    else: return output_stream    