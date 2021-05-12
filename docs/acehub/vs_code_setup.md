# ACE Hub VS Code Setup

The following are instructions for setting up the default
VS Code environment hosted on ACE Hub for use with saferl.

1. Follow the instructions [here](acehub_quickstart.md) for logging
into ACE Hub and setting up your User Configuration. 
   
2. Create a new environment using the ```vscode-server``` launch
template, assigning your needed resources to the environment.
   
3. Once the new environment is ready, click the "Open UI" button to
launch the VS Code Editor in your browser.
   
4. Open a new terminal (File > Terminal > New Terminal).

5. Initialize conda by running ```conda init bash``` and restarting
   your terminal.
   
6. Set the ```GIT_ASKPASS``` environment variable with the
following command (VS Code may overwrite ACE Hub's attempt to do
   this by default during startup):
   
```shell
export GIT_ASKPASS="/ace/hub/envfile/GIT_ASKPASS" 
```
   
7. Create a new conda environment with Python 3.7 and pip:

```shell
conda create -y -n <env_name> python=3.7 pip
```

8. Activate the new conda environment:

```shell
conda activate <env_name>
```

9. Clone the have-deepsky repository into the VS Code environment:

```shell
git clone https://git.act3-ace.com/rta/have-deepsky.git
```

10. Perform a local pip install of saferl into the conda environment:

```shell
cd path/to/have-deepsky/
pip --default-timeout=1000 install .
```

or perform an editable install if you will be changing the saferl
package code:

```shell
cd path/to/have-deepsky/
pip --default-timeout=1000 install -e .
```

NOTE: The default timeout argument in pip is a workaround for
the on-prem instance having long read times. This will be the default
behavior of pip in a future release of ACE Hub.
