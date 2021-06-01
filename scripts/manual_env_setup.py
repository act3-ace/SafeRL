import saferl.environment
import yaml
import argparse
import cProfile
import pstats

from saferl import lookup
from saferl.environment.utils import setup_env_objs_from_config,YAMLParser
# create env from config

# use the saferl.environment to set it up

#default config set to docking env
#CONFIG = '../configs/docking/docking_default.yaml'
CONFIG = '../configs/docking/docking_default_checkProcessors.yaml'
# 1. setup an argeparse block to take in config parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config',type=str,help='path to config file',default=CONFIG)
args = parser.parse_args()

# 2. parse yaml file and call appropriate classes
parser = YAMLParser(yaml_file=args.config,lookup=lookup)
env , env_config = parser.parse_env()

# 3 . instantiate the environment class
envObj = env(env_config)

def run_1000_times(envObj):
    for _ in range(1000):
        envObj.step((0,0))

# 4 - a step on the environment

obs,reward,done,info = envObj.step((0,0))

print('obs=',obs)
print('reward=',reward)
print('done=',done)
print('info=',info)


#obs,reward,done,info = envObj.step((0,0))

#profile_output_loc = 'env_step_profile.txt'
#cProfile.run("envObj.step((0,0))",profile_output_loc)
#cProfile.run("run_1000_times(envObj)",profile_output_loc)

#results_file = 'env_step_profile_results_V2.txt'
#with open(results_file,'w') as stream:
#    p = pstats.Stats(profile_output_loc,stream=stream)
#    p.strip_dirs().sort_stats(pstats.SortKey.TIME).print_stats()
