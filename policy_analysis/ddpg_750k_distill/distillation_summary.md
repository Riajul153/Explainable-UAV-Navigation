# Policy Distillation Summary

- Samples: 25000
- Seed: 7

## Sparse Linear Equations

### action_x

- R^2: `0.5400`

```text
action_x = 0.0241 + 0.1204 * goal_dx - 0.0476 * vx + 0.0307 * vy
```

### action_y

- R^2: `0.7041`

```text
action_y = -0.2706 + 0.1340 * goal_dy
```

## Decision Tree: action_x

- R^2: `0.8238`

```text
|-- goal_dx <= -1.173
|  |-- goal_dy <= -5.703
|  |  |-- goal_dx <= -5.721
|  |  |  |-- value: [-0.702]
|  |  |-- goal_dx >  -5.721
|  |  |  |-- value: [0.763]
|  |-- goal_dy >  -5.703
|  |  |-- goal_dx <= -2.987
|  |  |  |-- value: [-0.905]
|  |  |-- goal_dx >  -2.987
|  |  |  |-- value: [-0.422]
|-- goal_dx >  -1.173
|  |-- goal_dy <= -1.680
|  |  |-- goal_dy <= -5.670
|  |  |  |-- value: [0.918]
|  |  |-- goal_dy >  -5.670
|  |  |  |-- value: [-0.399]
|  |-- goal_dy >  -1.680
|  |  |-- goal_dx <= 2.174
|  |  |  |-- value: [0.388]
|  |  |-- goal_dx >  2.174
|  |  |  |-- value: [0.931]

```

## Decision Tree: action_y

- R^2: `0.8469`

```text
|-- goal_dy <= 0.883
|  |-- goal_dy <= -0.751
|  |  |-- goal_dy <= -1.357
|  |  |  |-- value: [-0.998]
|  |  |-- goal_dy >  -1.357
|  |  |  |-- value: [-0.886]
|  |-- goal_dy >  -0.751
|  |  |-- goal_dx <= -1.950
|  |  |  |-- value: [-0.961]
|  |  |-- goal_dx >  -1.950
|  |  |  |-- value: [-0.268]
|-- goal_dy >  0.883
|  |-- repulsion_y <= -0.163
|  |  |-- goal_dx <= 4.007
|  |  |  |-- value: [0.313]
|  |  |-- goal_dx >  4.007
|  |  |  |-- value: [-0.368]
|  |-- repulsion_y >  -0.163
|  |  |-- goal_dy <= 2.218
|  |  |  |-- value: [0.126]
|  |  |-- goal_dy >  2.218
|  |  |  |-- value: [0.793]

```

