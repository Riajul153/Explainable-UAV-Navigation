# Policy Distillation Summary

- Samples: 25000
- Seed: 7

## Sparse Linear Equations

### action_x

- R^2: `0.7649`

```text
action_x = -0.1436 + 0.1516 * goal_dx + 0.0373 * goal_dy
```

### action_y

- R^2: `0.6433`

```text
action_y = -0.2777 - 0.0416 * goal_dx + 0.1336 * goal_dy - 0.0205 * vx
```

## Decision Tree: action_x

- R^2: `0.9129`

```text
|-- goal_dx <= 0.491
|  |-- goal_dx <= -0.684
|  |  |-- goal_dx <= -2.152
|  |  |  |-- value: [-0.986]
|  |  |-- goal_dx >  -2.152
|  |  |  |-- value: [-0.794]
|  |-- goal_dx >  -0.684
|  |  |-- goal_dy <= -1.680
|  |  |  |-- value: [-0.957]
|  |  |-- goal_dy >  -1.680
|  |  |  |-- value: [-0.046]
|-- goal_dx >  0.491
|  |-- goal_dy <= -4.255
|  |  |-- goal_dx <= 4.955
|  |  |  |-- value: [-0.607]
|  |  |-- goal_dx >  4.955
|  |  |  |-- value: [0.886]
|  |-- goal_dy >  -4.255
|  |  |-- goal_dx <= 1.609
|  |  |  |-- value: [0.581]
|  |  |-- goal_dx >  1.609
|  |  |  |-- value: [0.953]

```

## Decision Tree: action_y

- R^2: `0.9044`

```text
|-- goal_dy <= 0.964
|  |-- goal_dy <= -0.643
|  |  |-- goal_dy <= -1.232
|  |  |  |-- value: [-0.995]
|  |  |-- goal_dy >  -1.232
|  |  |  |-- value: [-0.816]
|  |-- goal_dy >  -0.643
|  |  |-- repulsion_y <= 0.113
|  |  |  |-- value: [-0.835]
|  |  |-- repulsion_y >  0.113
|  |  |  |-- value: [0.023]
|-- goal_dy >  0.964
|  |-- goal_dx <= 2.848
|  |  |-- goal_dx <= 1.847
|  |  |  |-- value: [0.934]
|  |  |-- goal_dx >  1.847
|  |  |  |-- value: [0.443]
|  |-- goal_dx >  2.848
|  |  |-- goal_dy <= 6.481
|  |  |  |-- value: [-0.765]
|  |  |-- goal_dy >  6.481
|  |  |  |-- value: [0.897]

```

