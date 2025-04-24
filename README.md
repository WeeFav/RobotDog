# Training Boston Dyanmic Spot To Walk

Insipration: https://github.com/Argo-Robot/quadrupeds_locomotion

- Import dog to MuJuCo - Done
- Create custom gym environment - Not Done
    - determine homing position
- Train with stable baseline - Not Done

### Joints

| Joint  | Range |
| -----  | ----- |
| fl_hx  | (-0.785398, 0.785398) |
| fl_hy  | (-0.898845, 2.29511) |
| fl_kn  | (-2.7929, -0.254402) |
| fr_hx  | (-0.785398, 0.785398) |
| fr_hy  | (-0.898845, 2.24363) |
| fr_kn  | (-2.7929, -0.255648) |
| hl_hx  | (-0.785398, 0.785398) |
| hl_hy  | (-0.898845, 2.29511) |
| hl_kn  | (-2.7929, -0.247067) |
| hr_hx  | (-0.785398, 0.785398) |
| hr_hy  | (-0.898845, 2.29511) |
| hr_kn  | (-2.7929, -0.248282) |

### Action Space
Box

https://gymnasium.farama.org/introduction/create_custom_env/#step-function
