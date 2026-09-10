# Pushing_task

Basic scene for a pushing object

## Setups

### Setup 1

| Component | Version |
|-----------|---------|
| Isaac Sim | 5.1     |
| Isaac Lab | 2.3.2   |
| cuRobo    | 0.7.7   |

### Setup 2

| Component | Version |
|-----------|---------|
| Isaac Sim | 6.0     |
| Isaac Lab | 3.0     |
| cuRobo    | 0.7.7   |

## Scripts

| Script                          | Setup   |
|----------------------------------|---------|
| `Test_pushing_isaaclab.py`       | Setup 1 |
| `Test_pushing_isaaclab_2env.py`  | Setup 1 |
| `Test_pushing_isaaclab3.py`      | Setup 2 |

## Launch commands

### Setup 1

```bash
cd IsaacLab
./isaaclab.sh -p /home/berith/Documents/Pushing_task/Test_pushing_isaaclab.py
```

```bash
cd IsaacLab
./isaaclab.sh -p /home/berith/Documents/Pushing_task/Test_pushing_isaaclab_2env.py
```

### Setup 2

```bash
cd ~/isaac-lab-3
./isaaclab.sh -p /media/berith/DataDrive/Documents/Pushing_task_Isaaclab3/Test_pushing_isaaclab3.py --viz kit
```
