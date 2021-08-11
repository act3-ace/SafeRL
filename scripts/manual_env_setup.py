import argparse
from saferl.environment.utils import YAMLParser, build_lookup

# create env from config

# use the saferl.environment to set it up

# default config set to docking env
CONFIG = '../configs/docking/docking_default.yaml'

# 1. setup an argeparse block to take in config parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='path to config file', default=CONFIG)
args = parser.parse_args()

# 2. parse yaml file and call appropriate classes

# behavior of YAMLParser changed in master

lookup = build_lookup()
parser = YAMLParser(args.config, lookup)
mod_config = parser.parse_env()

env_class = mod_config['env']
env_config = mod_config['env_config']

# 3 . instantiate the environment class
setup_env = env_class(env_config)

# 4 - a step on the environment

obs, reward, done, info = setup_env.step(None)

print('obs=', obs)
print('reward=', reward)
print('done=', done)
print('info=', info)
