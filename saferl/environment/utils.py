import io
import os

import yaml

import numpy as np

from saferl.environment.models import BaseGeometry, RelativeGeometry, geo_from_config


PATH_CHAR = '#'


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
    config_path = os.path.abspath(config_yaml)

    with open(config_path, 'r') as f:
        config = yaml.load(f)
    env_str = config["env"]
    env_config = config["env_config"]
    env = lookup[env_str]
    env_config = process_yaml_items(env_config, lookup)
    return env, env_config


def process_yaml_items(target_dict, lookup):
    for k, v in target_dict.items():
        if k == "class":
            target_dict[k] = lookup[v]
        else:
            if isinstance(v, str) and len(v) < 0 and v[0] == PATH_CHAR:
                # Value is path to yaml config
                path_str = v[1:]
                path = os.path.abspath(path_str)
                with open(path, 'r') as f:
                    v = yaml.load(f)
                target_dict[k] = process_yaml_items(v, lookup)
            elif isinstance(v, dict):
                target_dict[k] = process_yaml_items(v, lookup)
            elif isinstance(v, list):
                result = []
                for i in v:
                    if isinstance(i, dict):
                        result.append(process_yaml_items(i, lookup))
                    else:
                        result.append(i)
                target_dict[k] = result
    return target_dict
