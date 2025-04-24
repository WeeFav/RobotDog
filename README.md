# Training Boston Dyanmic Spot To Walk

Insipration: https://github.com/Argo-Robot/quadrupeds_locomotion

- Import dog to MuJuCo - Done
- Create custom gym environment - Not Done
    - determine homing position
    - determine observation space
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

### Action Space
Box

### Observation Space

| Variable | Range |
| -----    | ----- |
| vx      | (,) |
| vy      | (,) |
| vz      | (,) |
| wx      | (,) |
| wy      | (,) |
| wz      | (,) |
| roll    | (,) |
| pitch   | (,) |
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
| q1'     | (,) |
| q2'     | (,) |
| q3'     | (,) |
| q4'     | (,) |
| q5'     | (,) |
| q6'     | (,) |
| q7'     | (,) |
| q8'     | (,) |
| q9'     | (,) |
| q10'    | (,) |
| q11'    | (,) |
| q12'    | (,) |
| a1      | (,) |
| a2      | (,) |
| a3      | (,) |
| a4      | (,) |
| a5      | (,) |
| a6      | (,) |
| a7      | (,) |
| a8      | (,) |
| a9      | (,) |
| a10     | (,) |
| a11     | (,) |
| a12     | (,) |
| vx_ref  | (,) |
| vy_ref  | (,) |
| wz_ref  | (,) |
| z_ref   | (,) |

https://gymnasium.farama.org/introduction/create_custom_env/#step-function
