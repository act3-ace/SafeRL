#Policy Evaluation and Animation

## Training Output

After training a policy on an environment of your choosing, using the train.py module 
from the `scripts/` directory, a training output directory will be created at the path 
`scripts/output/expr_[date]_[time]`. Inside this output directory, relevant training data 
will be stored including episode logging, config files, and checkpoints of policy parameter values. 
Policy checkpoints may be found inside the experiment directory, which will follow the naming convention
of `[RL Algorithm]_[environment name]_[experiment ID]_[date]_[time]`.

```
scripts/
├─ output/
│  ├─ expr_[date]_[time]
├─ └─ ── [RL Algorithm]_[environment name]_[experiment ID]_[date]_[time]
```

Copy the full path to your experiment directory, or one of the training checkpoints directories
therein.

## Evaluation

Also residing in the `scripts/` directory, eval.py may be used to run post hoc evaluation
rollouts to benchmark your trained policy. From the `scripts/` directory, run the following
command:

`$ python eval.py --dir <full path to experiment dir>`

This will load in the policy and environment with the same configuration from training and run
ten rollout episodes. The logs of these evaluation episodes are stored in a new directory, which
may be found at `path/to/experiment_directory/eval/ckpt_[checkpoint number]`.

There exist a number of command line options that can be used to tailor the eval script's function.
The following is a brief description of each:

### `--ckpt_num : int`
Given the number of a checkpoint directory within the experiment directory, the policy saved at the corresponding
checkpoint will be loaded and evaluated. If not specified, the latest checkpoint's policy is used.

### `--seed : int`
Use this option to specify the seed used for evaluation episodes. If not specified, the eval script will use the
same seed used in training.

### `--explore`
Use this flag on the command line to enable off-policy evaluation. By default, all evaluation episodes are on-policy.

### `--output_dir : str`
Supply the full path to the desired custom location for output logs for evaluation episodes. By default, output logs
are saved to `eval/ckpt_[checkpoint number]` inside the experiment directory.

### `--num_rollouts : int`
Use this option to configure the number of evaluation rollout episodes to execute. If not specified, 10 evaluation
rollouts will be run.

### `--alt_env_config : str`
Supply the full path to an alternative config file for the environment the policy was trained on, to evaluate the policy
in an alternative environment. This is good for testing the generalizability of your policy. By default, the same 
configuration as used in training will be used to initialize the environments for evaluation.

### `--render and --render_config`
These two options pertain to visualization of the evaluation episodes. See the next section for details.

## Animation
For visual intuition, our 2D base environments come with builtin animations.

To use our default renderers, for the 2D Rejoin and 2D Docking tasks, simply add the `--render`
flag to the evaluation script launch command from above:

`$ python eval.py --dir <full path to experiment dir> --render`

If you wish to customize some aspects of the visualization (like the size of the render window, scale of objects, 
padding, etc) pass the full path to a render config file to the `--render_config` option on the command line. The 
render config file needs to be a yaml file which specifies a dictionary for kwargs passed to the constructor of either 
2D environment's Renderer class (DockingRenderer or RejoinRenderer). The kwargs accepted by each of these classes vary
greatly, so please reference the [docking](../../saferl/aerospace/tasks/docking/render.py) and 
[rejoin](../../saferl/aerospace/tasks/rejoin/render.py) renderer classes to see the full list of available options. Some
notable mentions include the inclusion of safety margins, adjustment of animation speed, inclusion of rejoin region, and 
inclusion of trace in the rejoin animation. For the docking animation, the inclusion of velocity arrows, the inclusion 
of thrust arrows, adjustment of ellipses sizes, and inclusion of trace are all configurable parameters.

You may also create your own custom animations by extending the BaseRenderer class and passing
it to the environment in the config.
