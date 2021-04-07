import numpy as np
import io

from saferl.environment.models import BaseGeometry, RelativeGeometry, geo_from_config


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
    else:
        return output_stream


def setup_env_objs_from_config(config):
    env_objs = {}
    agent = None

    agent_name = config["agent"]

    for obj_config in config["env_objs"]:
        name = obj_config["name"]
        cls = obj_config["class"]
        if issubclass(cls, BaseGeometry) or issubclass(cls, RelativeGeometry):
            if issubclass(cls, RelativeGeometry):
                ref_name = obj_config["config"]["ref"]
                obj_config["config"]["ref"] = env_objs[ref_name]
            obj = geo_from_config(cls, config=obj_config["config"])
        else:
            obj = cls(config=obj_config["config"])
        env_objs[name] = obj
        if name == agent_name:
            agent = obj

    return agent, env_objs


def parse_env_config(config_yaml, lookup):
    pass
