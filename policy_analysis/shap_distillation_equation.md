# SHAP-Informed Distillation Equation

- Samples: 30000
- Sampling seed: 7
- Best repulsion formulation: `exp_decay_0.5_thr4.0`
- Overall R^2: `0.6224`
- Action MAE: `0.3873`

```text
Action X = 0.0429 + 0.1437 * goal_dx - 0.0119 * vx + 0.4630 * repulsion_x
Action Y = -0.0620 + 0.1134 * goal_dy - 0.0071 * vy + 0.6509 * repulsion_y
```

Top candidates:
- `exp_decay_0.5_thr4.0` -> R^2 `0.6224`, MAE `0.3873`
- `exp_decay_0.5_thr3.5` -> R^2 `0.6208`, MAE `0.3894`
- `linear_spring_thr4.0` -> R^2 `0.6208`, MAE `0.3889`
- `linear_spring_thr3.5` -> R^2 `0.6193`, MAE `0.3903`
- `exp_decay_0.5_thr3.0` -> R^2 `0.6189`, MAE `0.3911`
