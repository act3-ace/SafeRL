import io
import os
import copy
import inspect

import yaml
import jsonlines
import numpy as np
import json


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


def initializer_from_config(ref_obj, config, default_initializer):
    init_config = config["init"] if "init" in config.keys() else None
    if init_config is not None and "initializer" in init_config.keys():
        initializer = init_config["initializer"]
    else:
        initializer = default_initializer
    return initializer(ref_obj, init_config)


def get_ref_objs(env_objs, config):
    if "ref" in config.keys():
        config["ref"] = env_objs[config["ref"]]
    return config


def setup_env_objs_from_config(config, default_initializer):
    safe_config = copy.deepcopy(config)
    env_objs = {}
    agent = None
    initializers = []

    agent_name = safe_config["agent"]

    for obj_config in safe_config["env_objs"]:
        # Get config values
        name = obj_config["name"]
        cls = obj_config["class"]
        cfg = obj_config["config"]

        # Populate ref obj in config if it exists
        cfg = get_ref_objs(env_objs, cfg)

        # Instantiate object
        if type(cls) is str:
            print()
        obj = cls(**{k: v for k, v in cfg.items() if k != "init"})
        env_objs[name] = obj
        if name == agent_name:
            agent = obj

        # Create object initializer
        initializers.append(initializer_from_config(obj, cfg, default_initializer))

    return agent, env_objs, initializers


def build_lookup(pkg, parent, checked_modules=None):
    checked_modules = set() if checked_modules is None else checked_modules
    modules = inspect.getmembers(pkg, inspect.ismodule)
    modules = [m for m in modules if str(m[1]) not in checked_modules and parent in str(m[1])]
    checked_modules = checked_modules.union(set([str(m[1]) for m in modules]))
    classes = inspect.getmembers(pkg, inspect.isclass)
    classes = [c for c in classes if parent in str(c[1])]
    local_lookup = {pkg.__name__ + "." + k: v for k, v in classes if "saferl" in str(v)}
    for m in modules:
        m_lookup, checked_modules = build_lookup(m[1], parent=parent, checked_modules=checked_modules)
        local_lookup = {**local_lookup, **m_lookup}
    return local_lookup, checked_modules


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
