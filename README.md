> [!TIP]
> Acknowledgment：The core simulation environment of this repository is built upon the excellent work of https://github.com/learnsyslab/gym-pybullet-drones. We deeply appreciate their contribution to the open-source community. My modifications mainly focus on the reinforcement learning algorithms and target tracking logic located in the examples/ folder. 

# gym-pybullet-drones

This repository is for displaying reinforcement learning examples of pybullet drones.

<img src="gym_pybullet_drones/assets/episode_4_trajectory.png" alt="rl example_1" width="450"><img src="gym_pybullet_drones/assets/episode_7_trajectory.png" alt="rl example_2" width="450">

## Installation

```sh
git clone https://github.com/NishimiyaXSean/pybullet-rl-examples.git
cd gym-pybullet-drones/
conda create -n drone_rl python=3.10
conda activate drone_rl
pip install -e .
```

## Use

```sh
cd gym-pybullet-drones/examples
python mission_v1.py
python mission_v2.py
---
```

