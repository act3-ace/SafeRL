import io
import os

import yaml

import numpy as np

from saferl.environment.models.geometry import BaseGeometry, RelativeGeometry


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


def setup_initializers_from_config(config, env_objs, default_init):
    initializers = []
    reg_objs = []
    i_list = config["initializers"]
    for i in i_list:
        i_class = i["class"]
        i_config = i["config"]
        env_obj = env_objs[i_config["env_obj"]]
        initializer = i_class(env_obj=env_obj)
        initializers.append(initializer)
        reg_objs.append(env_obj.name)

    # Give default initializer to objs with no defined initializer
    no_init = list(set(env_objs.keys()) - set(reg_objs))
    for name in no_init:
        initializers.append(default_init(env_obj=env_objs[name]))

    return initializers


def setup_env_objs_from_config(config, default_initializer):
    env_objs = {}
    agent = None
    initializers = []

    agent_name = config["agent"]

    for obj_config in config["env_objs"]:
        # Get config values
        name = obj_config["name"]
        cls = obj_config["class"]
        cfg = obj_config["config"]

        # Assign ref property to existing env_obj for Geometry objects
        if issubclass(cls, BaseGeometry) or issubclass(cls, RelativeGeometry):
            if issubclass(cls, RelativeGeometry):
                ref_name = cfg["ref"]
                cfg["ref"] = env_objs[ref_name]

        # Instantiate object
        obj = cls(**cfg)
        env_objs[name] = obj
        if name == agent_name:
            agent = obj

        # Create object initializer
        i = cfg[ini]

    return agent, env_objs


class YAMLParser:

    COMMAND_CHAR = '!'

    def __init__(self, yaml_file, lookup):
        self.commands = {
            "file": self.file_command
        }
        self.yaml_path = os.path.abspath(yaml_file)
        self.working_dir = os.path.dirname(self.yaml_path)
        self.lookup = lookup

    def parse_env(self):
        with open(self.yaml_path, 'r') as f:
            config = yaml.load(f)
        assert "env" in config.keys(), "environment config missing required field: env"
        assert "env_config" in config.keys(), "environment config missing required field: env_config"
        env_str = config["env"]
        env_config = config["env_config"]
        env = self.lookup[env_str]
        env_config = self.process_yaml_items(env_config)
        return env, env_config

    def process_yaml_items(self, target):
        if isinstance(target, dict):
            for k, v in target.items():
                target[k] = self.process_yaml_items(v)
        elif isinstance(target, str):
            target = self.process_str(target)
        elif isinstance(target, list):
            # Remove redundant dimensions
            # if len(target) == 1 and isinstance(target[0], list):
            #     target = target[0]
            target = [self.process_yaml_items(i) for i in target]
        return target

    def process_str(self, input_str):
        if input_str[0] == "!":
            command, value = input_str[1:].split(":", 1)
            value = self.commands[command](value)
        elif input_str in self.lookup.keys():
            value = self.lookup[input_str]
        else:
            value = input_str
        return value

    def file_command(self, value):
        path = os.path.abspath(os.path.join(self.working_dir, value))
        old_working_dir = self.working_dir
        self.working_dir = os.path.dirname(path)
        with open(path, 'r') as f:
            contents = yaml.load(f)
        target = self.process_yaml_items(contents)
        self.working_dir = old_working_dir
        return target
