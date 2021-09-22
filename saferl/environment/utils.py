import abc
import io
import os
import copy
import inspect
import ast
import yaml
import jsonlines
import numpy as np
import json
from ray import tune
import saferl


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
    for k, v in config.items():
        if isinstance(v, dict):
            config[k] = get_ref_objs(env_objs, v)
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
        cfg['name'] = name
        cfg = get_ref_objs(env_objs, cfg)

        # Instantiate object
        obj = cls(**{k: v for k, v in cfg.items() if k != "init"})
        env_objs[name] = obj
        if name == agent_name:
            agent = obj

        # Create object initializer
        initializers.append(initializer_from_config(obj, cfg, default_initializer))

    return agent, env_objs, initializers


def build_lookup(pkg=saferl):
    return _build_lookup(pkg=pkg, parent=pkg.__name__)[0]


def _build_lookup(pkg, parent, checked_modules=None):
    checked_modules = set() if checked_modules is None else checked_modules
    modules = inspect.getmembers(pkg, inspect.ismodule)
    modules = [m for m in modules if m[1].__name__ not in checked_modules and parent in m[1].__name__]
    checked_modules = checked_modules.union(set([m[1].__name__ for m in modules]))
    classes = inspect.getmembers(pkg, inspect.isclass)
    classes = [c for c in classes if parent in c[1].__module__]
    local_lookup = {pkg.__name__ + "." + k: v for k, v in classes}
    for m in modules:
        m_lookup, checked_modules = _build_lookup(m[1], parent=parent, checked_modules=checked_modules)
        local_lookup = {**local_lookup, **m_lookup}
    return local_lookup, checked_modules


def dict_merge(dict_a, dict_b, recursive=True):
    '''
    Merges dictionaries dict_a, dict_b
    key collisions take value from dict_b
    recursive flag allows dict values to be merged recursively
    '''
    dict_merged = copy.deepcopy(dict_a)

    for k, v_b in dict_b.items():
        if recursive and k in dict_a:
            v_a = dict_a[k]
            if isinstance(v_a, dict) and isinstance(v_b, dict):
                dict_merged[k] = dict_merge(v_a, v_b, recursive=recursive)
            else:
                dict_merged[k] = copy.deepcopy(v_b)
        else:
            dict_merged[k] = copy.deepcopy(v_b)
    return dict_merged


class YAMLParser:

    COMMAND_CHAR = '!'

    def __init__(self, yaml_file, lookup):
        self.commands = {
            "file": self.file_command,
            "tune_search_space": self.tune_search_space_command,
        }
        self.yaml_path = os.path.abspath(yaml_file)
        self.working_dir = os.path.dirname(self.yaml_path)
        self.lookup = lookup

    def parse_env(self):
        with open(self.yaml_path, 'r') as f:
            config = yaml.load(f)
        config = self.process_yaml_items(config)
        return config

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

    def tune_search_space_command(self, value):
        method_value_arg = value.split('.', 1)[1]
        left_paren = method_value_arg.find('(')
        right_paren = method_value_arg.find(')')
        method = method_value_arg[0:left_paren]
        argument_str = method_value_arg[left_paren+1:right_paren]
        arg_values = ast.literal_eval(argument_str)
        if argument_str[0] == '[':
                return getattr(tune, method)(arg_values)
        else:
                return getattr(tune, method)(*arg_values)


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


class PostProcessor:
    @abc.abstractmethod
    def __call__(self, input_array):
        """
        Subclasses should implement this method to apply post-processing to processor's return values.

        Parameters
        ----------
        input_array : numpy.ndarray
            The value passed to a given post processor for modification.

        Returns
        -------
        input_array
            The modified (processed) input value
        """
        raise NotImplementedError


class Normalize(PostProcessor):
    def __init__(self, mu=0, sigma=1):
        # ensure mu and sigma compatible types
        acceptable_types = [float, int, list, np.ndarray]
        assert type(mu) in acceptable_types, \
            "Expected kwarg \'mu\' to be type int, float, list, or numpy.ndarray, but received {}" \
            .format(type(mu))
        assert type(sigma) in acceptable_types, \
            "Expected kwarg \'sigma\' to be type int, float, list, or numpy.ndarray, but received {}" \
            .format(type(sigma))

        # convert lists to numpy arrays
        if type(mu) == list:
            mu = np.array(mu)
        if type(sigma) == list:
            sigma = np.array(sigma)

        # store values
        self.mu = mu
        self.sigma = sigma

    def __call__(self, input_array):
        # ensure input_array is numpy array
        assert type(input_array) == np.ndarray, \
            "Expected \'input_array\' to be type numpy.ndarray, but instead received {}.".format(type(input_array))

        # check that dims line up for mu and sigma (or that they're scalars)
        if type(self.mu) == np.ndarray:
            assert input_array.shape == self.mu.shape, \
                "Incompatible shapes for \'input_array\' and \'mu\': {} vs {}".format(input_array, self.mu)
        if type(self.sigma) == np.ndarray:
            assert input_array.shape == self.sigma.shape, \
                "Incompatible shapes for \'input_array\' and \'sigma\': {} vs {}".format(input_array, self.sigma)

        # apply normalization
        input_array = np.subtract(input_array, self.mu)
        input_array = np.divide(input_array, self.sigma)

        return input_array


class Clip(PostProcessor):
    def __init__(self, low=-1, high=1):
        # ensure bounds correct types
        assert type(low) in [int, float], \
            "Expected kwarg \'low\' to be type int or float, but instead received {}".format(type(low))
        assert type(high) in [int, float], \
            "Expected kwarg \'high\' to be type int or float, but instead received {}".format(type(high))
        # ensure correct relation
        assert low < high, "Expected value of variable \'low\' to be less than variable \'high\'"

        # store values
        self.low = low
        self.high = high

    def __call__(self, input_array):
        # ensure input_array is numpy array
        assert type(input_array) == np.ndarray, \
            "Expected \'input_array\' to be type numpy.ndarray, but instead received {}.".format(type(input_array))

        # apply clipping in specified range
        input_array = np.clip(input_array, self.low, self.high)

        return input_array
