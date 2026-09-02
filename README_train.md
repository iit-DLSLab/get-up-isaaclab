## Installation Train

1. Install Isaac Lab by following the [installation guide](https://github.com/isaac-sim/IsaacLab). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

2. Install git for very large file
```bash
sudo apt install git-lfs
```

3. Clone the repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory)


4. Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/get_up_isaaclab
```



## Run a train/play in IsaacLab

- To train:

```bash
python scripts/rsl_rl/train.py --task=GetUp-Go2-Flat --num_envs=4096
python scripts/rsl_rl/train.py --task=GetUp-Go2-Rough-Blind --num_envs=4096
```


- To test the policy, you can press:
```bash
python scripts/rsl_rl/play.py --task=GetUp-Go2-Flat --num_envs=16
python scripts/rsl_rl/play.py --task=GetUp-Go2-Rough-Blind --num_envs=16
```


- If you have speed problem in training, may be due to cylinder collision. Then add

```bash
--kit_args="--/physics/collisionApproximateCylinders=true"
```


## Use Morphological Symmetries
1. If you want to play with [Morphologycal Symmetries](https://arxiv.org/pdf/2403.17320), install the repo [morphosymm-rl](https://github.com/iit-DLSLab/morphosymm-rl)

2. See the specific README in its own script folder for how to run.



## Run Hyperparameter Search

```bash
echo "import ray; ray.init(); import time; [time.sleep(10) for _ in iter(int, 1)]" | python3 (TERMINAL 1)
```

```bash
python3 ../get_up_isaaclab/exts/get_up_isaaclab/get_up_isaaclab/hyperparameter_tuning/tuner.py --run_mode local --cfg_file ../get_up_isaaclab/exts/get_up_isaaclab/get_up_isaaclab/hyperparameter_tuning/locomotion_aliengo_cfg.py --cfg_class LocomotionAliengoFlatTuner (TERMINAL 2)
```
