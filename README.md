# Training Boston Dyanmic Spot To Walk

Insipration: https://github.com/Argo-Robot/quadrupeds_locomotion

- Import dog to MuJuCo - Done
- Create custom gym environment - Not Done
    - double check if environment is working (reward seems wrong)
- Train with stable baseline - Not Done

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
| roll    | (-pi, pi) |
| pitch   | (-pi, pi) |
| yaw   | (-pi, pi) |
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

https://gymnasium.farama.org/introduction/create_custom_env/#step-function
