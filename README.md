# Training Boston Dyanmic Spot To Walk

Insipration: https://github.com/Argo-Robot/quadrupeds_locomotion

- Import dog to MuJuCo - Done
- Create custom gym environment - Done
- Train with stable baseline - Not Done
    - robot dog won't move after a while


### Installation
```bash
pip install -r requirements.txt
```
If not using GPU, remove +cu118 from
```bash
torch==2.3.0+cu118
torchvision==0.18.0+cu118
```

### Joints

| Joint  | Range |
| -----  | ----- |
| fl_hx (q1)  | (-0.785398, 0.785398) |
| fl_hy (q2)  | (-0.898845, 2.29511) |
| fl_kn (q3)  | (-2.7929, -0.254402) |
| fr_hx (q4)  | (-0.785398, 0.785398) |
| fr_hy (q5)  | (-0.898845, 2.24363) |
| fr_kn (q6)  | (-2.7929, -0.255648) |
| hl_hx (q7)  | (-0.785398, 0.785398) |
| hl_hy (q8)  | (-0.898845, 2.29511) |
| hl_kn (q9)  | (-2.7929, -0.247067) |
| hr_hx (q10) | (-0.785398, 0.785398) |
| hr_hy (q11) | (-0.898845, 2.29511) |
| hr_kn (q12) | (-2.7929, -0.248282) |

### Initial Joint Positions

| Joint        | Position    |
|--------------|-------------|
| fl_hx (q1)   | 0    |
| fl_hy (q2)   | 1.04   |
| fl_kn (q3)   | -1.8   |
| fr_hx (q4)   | 0      |
| fr_hy (q5)   | 1.04   |
| fr_kn (q6)   | -1.8   |
| hl_hx (q7)   | 0      |
| hl_hy (q8)   | 1.04   |
| hl_kn (q9)   | -1.8   |
| hr_hx (q10)  | 0      |
| hr_hy (q11)  | 1.04   |
| hr_kn (q12)  | -1.8   |

### Action Space

| Action | Range (Min Δ, Max Δ)          |
|--------|-------------------------------|
| a1     | (-0.785398, 0.785398)         |
| a2     | (-1.938845, 1.25511)          |
| a3     | (-0.9929, 1.545598)           |
| a4     | (-0.785398, 0.785398)         |
| a5     | (-1.938845, 1.20363)          |
| a6     | (-0.9929, 1.544352)           |
| a7     | (-0.785398, 0.785398)         |
| a8     | (-1.938845, 1.25511)          |
| a9     | (-0.9929, 1.552933)           |
| a10    | (-0.785398, 0.785398)         |
| a11    | (-1.938845, 1.25511)          |
| a12    | (-0.9929, 1.551718)           |

### Observation Space
| Variable | Range |
| -----    | ----- |
| x      | (-inf, inf) |
| y      | (-inf, inf) |
| z      | (-inf, inf) |
| roll    | (-180, 180) |
| ptch   | (-180, 180) |
| yaw   | (-180, 180) |
| q1      | (-0.785398, 0.785398) |
| q2      | (-0.898845, 2.29511) |
| q3      | (-2.7929, -0.254402) |
| q4      | (-0.785398, 0.785398) |
| q5      | (-0.898845, 2.24363) |
| q6      | (-2.7929, -0.255648) |
| q7      | (-0.785398, 0.785398) |
| q8      | (-0.898845, 2.29511) |
| q9      | (-2.7929, -0.247067) |
| q10     | (-0.785398, 0.785398) |
| q11     | (-0.898845, 2.29511) |
| q12     | (-2.7929, -0.248282) |
| q1'     | (-inf, inf) |
| q2'     | (-inf, inf) |
| q3'     | (-inf, inf) |
| q4'     | (-inf, inf) |
| q5'     | (-inf, inf) |
| q6'     | (-inf, inf) |
| q7'     | (-inf, inf) |
| q8'     | (-inf, inf) |
| q9'     | (-inf, inf) |
| q10'    | (-inf, inf) |
| q11'    | (-inf, inf) |
| q12'    | (-inf, inf) |
| a1     | (-0.785398, 0.785398)         |
| a2     | (-1.938845, 1.25511)          |
| a3     | (-0.9929, 1.545598)           |
| a4     | (-0.785398, 0.785398)         |
| a5     | (-1.938845, 1.20363)          |
| a6     | (-0.9929, 1.544352)           |
| a7     | (-0.785398, 0.785398)         |
| a8     | (-1.938845, 1.25511)          |
| a9     | (-0.9929, 1.552933)           |
| a10    | (-0.785398, 0.785398)         |
| a11    | (-1.938845, 1.25511)          |
| a12    | (-0.9929, 1.551718)           |
| x      | (-inf, inf) |
| y      | (-inf, inf) |
| z      | (-inf, inf) |

### Reward
| Reward | Range |
| -----  | ----- |
| Position Tracking Reward (R_xyz) | (0, 1)       |
| Pose Similarity Penalty (R_pose) | (0, bounded) |
| Action Rate Penalty (R_action)   | (0, bounded) |
| Stabilization Penalty (R_stable) | (0, bounded) |
| Facing Target Reward (R_facing)  | (-1, 1)      |

Normalization Calculation:

R_pose = sqrt(4(0.785398-0)^2 + 4(2.29511-1.04)^2 + 4(-0.254402-(-1.8))^2) = 4.28

R_action = sqrt((max - min of action range)^2) = 8.75

R_stable = (180)^2 + (180)^2 = 64800

R_facing = +1 / 2

### Experiments
1. Robot move too much too quickly, so scaled R_action by 10. Result was that the robot learned to not move
2. Scaled R_xyz by 10 and R_action by 1. Robot still not moving
3. Change raw_th and pitch_th from 10 to 45. scale R_action 1/3.

https://gymnasium.farama.org/introduction/create_custom_env/#step-function
