import io
import os

import yaml
import jsonlines
import numpy as np
import json

from saferl.environment.models.geometry import BaseGeometry, RelativeGeometry, geo_from_config


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


def log_to_jsonlines(contents, output_dir, jsonline_filename):
    """
    A helper function to handle writing to a file in JSONlines format.

    Parameters
    ----------
    contents : dict
        The JSON-friendly contents to be appended to the file
    output_dir : str
        The path to the parent directory containing the JSONlines file.
    jsonline_filename : str
        The name of the JSONlines formatted file to append given contents to.
    """
    os.makedirs(output_dir, exist_ok=True)
    with jsonlines.open(output_dir + jsonline_filename, mode='a') as writer:
        writer.write(contents)


def jsonify(map):
    """
    A function to convert non-JSON serializable objects (numpy arrays and data types) within a dictionary to JSON
    friendly data types.

    Parameters
    ----------
    map : dict
        The dictionary which may or may not contain non-JSON serializable values

    Returns
    -------
    map : dict
        The same dictionary passed in from parameters, but with converted values
    """

    for key in map.keys():
        # iterate through dictionary, converting objects as needed
        suspicious_object = map[key]
        is_json_ready = is_jsonable(suspicious_object)

        if is_json_ready is True:
            # move along sir
            continue
        elif is_json_ready == TypeError:
            if type(suspicious_object) is dict:
                # recurse if we find sub-dictionaries
                map[key] = jsonify(suspicious_object)
            if type(suspicious_object) is np.ndarray:
                # handle numpy array conversion
                map[key] = suspicious_object.tolist()
            elif type(suspicious_object) is np.bool_:
                # handle numpy bool conversion
                map[key] = bool(suspicious_object)
            elif type(suspicious_object) is np.int64:
                # handle int64 conversion
                map[key] = int(suspicious_object)

        elif is_json_ready == OverflowError:
            raise OverflowError
        elif is_json_ready == ValueError:
            raise ValueError

    return map


def is_jsonable(object):
    """
    A helper function to determine whether or not an object is JSON serializable.

    Parameters
    ----------
    object
        The object in question

    Returns
    -------
    bool or Error
        True if object is JSON serializable, otherwise the specific error encountered
    """
    try:
        json.dumps(object)
        return True
    except TypeError:
        return TypeError
    except OverflowError:
        return OverflowError
    except ValueError:
        return ValueError
