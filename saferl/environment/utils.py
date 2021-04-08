import io
import os

import yaml

import numpy as np

from saferl.environment.models import BaseGeometry, RelativeGeometry, geo_from_config


PATH_CHAR = '!'


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
    config_dir = os.path.dirname(config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f)
    env_str = config["env"]
    env_config = config["env_config"]
    env = lookup[env_str]
    env_config = process_yaml_items(env_config, lookup, working_dir=config_dir)
    return env, env_config


def process_yaml_items(target, lookup, working_dir):
    if isinstance(target, dict):
        for k, v in target.items():
            target[k] = process_yaml_items(v, lookup, working_dir=working_dir)
    elif isinstance(target, str):
        if target[0] == PATH_CHAR:
            # Value is path to yaml config
            path_str = target[1:]
            path = os.path.abspath(os.path.join(working_dir, path_str))
            with open(path, 'r') as f:
                contents = yaml.load(f)
            target = process_yaml_items(contents, lookup, working_dir=os.path.dirname(path))
        elif target in lookup.keys():
            target = lookup[target]
    elif isinstance(target, list):
        target = [process_yaml_items(i, lookup, working_dir=working_dir) for i in target]

    return target
