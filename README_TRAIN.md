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

5. If you want to play with [Morphologycal Symmetries](https://arxiv.org/pdf/2403.17320), install the repo [morphosymm-rl](https://github.com/iit-DLSLab/morphosymm-rl)


## Run a train/play in IsaacLab

- To train:

```bash
python scripts/rsl_rl/train.py --task=GetUp-Aliengo-Flat --num_envs=4096 --headless
python scripts/rsl_rl/train.py --task=GetUp-Aliengo-Rough-Blind --num_envs=4096 --headless
```

- To train with Symmetries, modify the related [rsl_rl_ppo_cfg.py](https://github.com/iit-DLSLab/get-up-dls-isaaclab/blob/devel/source/get_up_isaaclab/get_up_isaaclab/tasks/getup/agents/rsl_rl_ppo_cfg.py) setting *class_name = PPOSymmDataAugmented*
```bash
python scripts/morphosymm_rl/train_symm.py --task=GetUp-Aliengo-Flat --num_envs=4096 --headless
python scripts/morphosymm_rl/train_symm.py --task=GetUp-Aliengo-Rough-Blind --num_envs=4096 --headless
```


- To test the policy, you can press:
```bash
python scripts/rsl_rl/play.py --task=GetUp-Aliengo-Flat --num_envs=16
python scripts/rsl_rl/play.py --task=GetUp-Aliengo-Rough-Blind --num_envs=16
```


- If you have speed problem in training, may be due to cylinder collision. Then add

```bash
--kit_args="--/physics/collisionApproximateCylinders=true"
```

## Run Hyperparameter Search

```bash
echo "import ray; ray.init(); import time; [time.sleep(10) for _ in iter(int, 1)]" | python3 (TERMINAL 1)
```

```bash
python3 ../get_up_isaaclab/exts/get_up_isaaclab/get_up_isaaclab/hyperparameter_tuning/tuner.py --run_mode local --cfg_file ../get_up_isaaclab/exts/get_up_isaaclab/get_up_isaaclab/hyperparameter_tuning/locomotion_aliengo_cfg.py --cfg_class LocomotionAliengoFlatTuner (TERMINAL 2)
```


## Convert XML to USD
We use model from [gym-quadruped](https://github.com/iit-DLSLab/gym-quadruped).

```bash
./isaaclab.sh -p scripts/tools/convert_mjcf.py   ../get_up_isaaclab/scripts/sim_to_sim_mujoco/gym-quadruped/gym_quadruped/robot_model/aliengo/aliengo.xml   ../aliengo.usd   --import-sites   --make-instanceable
```

Remember to set in the application above, "set as default prim" to the root of the robot. Furthermore, for now, add the following lines in the xml of your robots to make the feet seen as body

```bash
<body name="FL_foot" pos="0 0 -0.25">
    <!-- FL_foot only collision -->
    <geom name="FL" class="collision" size="0.0265" pos="0 0 0" />
</body>
```
