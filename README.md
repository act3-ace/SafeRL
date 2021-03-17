# Custom Environments
Members:
Tyler "Kode" Brown, 
Kerianne Hobbs, 
Nate Hamilton, 
Umberto Ravaioli, 

## Installation
To install the custom environments, clone with SSH. If you need an SSH key, follow the following steps:  
1. Install xclip:  
`sudo apt install xclip`  
2. In GitLab, in the top right go to your settings.  Then on the left, select SSH Keys.  Follow the tutorial to generate an SSH key.  
3. In your terminal, enter the command:  
`xclip -sel clip < ~/.ssh/id_ed25519.pub`  
This copies your SSH key into your clipboard.  
4. Paste your SSH key into GitLab (from step 2)  
5. In your terminal, enter the command:  
`ssh -T git@git.act3-ace.ai`  

Once you have an SSH key, install the environment:  
1. Install Anaconda (Recommended):  
Follow the installation instructions for Anaconda 3 [here](https://docs.continuum.io/anaconda/install/).  
2. Create a conda Python 3.6 environment, which will help organize the packages used:  
`conda create -n <env_name> python=3.6`  
3. To use Python in this environment, activate it by running:  
`conda activate <env_name>`  
4. Install OpenMPI (Ubuntu/Debian):  
`sudo apt-get update && sudo apt-get install libopenmpi-dev`  
5. In the directory you want to save the environment, run the command:  
`git clone git@git.act3-ace.com:rta/have-deepsky.git`
6. Then run the commands:  
`cd have-deepsky`  
`pip install .`

## Pulling an update from Gitlab
If someone else has updated their files, and you want to pull the most recent changes without erasing your progress:  
1. Navigate to the spacecraft docking folder `cd spacecraftdockingrl`  
2. `git fetch`  
3. `git merge`  

If you want to pull the latest changes and overwrite any changes you made:  
1. Navigate to the spacecraft docking folder `cd spacecraftdockingrl`  
2. `git stash`  
3. `git pull`  

## Running from the Command Line
These files assume that you have Anaconda or equivalent and standard Python packages installed.
1. To run from the command line, open a terminal
2. To run using the standard Anaconda packages, type `conda activate base`. This should make (base) show up before you username.
3. Navigate to the correct folder.
4. Decide what algorithm you'd like to run, e.g. run the vanilla policy gradient RL algorithm `python VPG.py`

## Uploading to GitLab
If you want to add your files to the GitLab, navigate to the spacecraftdockingrl folder in your terminal and run the following commands (Update with your COMMENTS):  
`git add .`  
`git commit -m 'COMMENTS'`  
`git push -u origin master`
