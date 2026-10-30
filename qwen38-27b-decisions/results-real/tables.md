## typed-decision-bench `test` (5387 cases)

| Config | Fit | DecisionScore | accuracy |
|---|---|---|---|
| `stock-nvfp4` | raw | 77.18 | 0.735 |
| `stock-nvfp4` | T | 80.28 | 0.735 |
| `stock-nvfp4` | T+B | 83.97 | 0.767 |
| `swift-nvfp4` | raw | 76.63 | 0.745 |
| `swift-nvfp4` | T | 80.58 | 0.745 |
| `swift-nvfp4` | T+B | 84.17 | 0.768 |
| jev-1.13.0 (published) | — | 81.70 | 0.768 |
| jevfish-v4-r4b (published) | — | 78.24 | 0.701 |
| decider-2b-v8 (published) | — | 77.56 | 0.688 |
| s1-scorer-4b-v2b (published) | — | 77.24 | 0.676 |
| openjev-0.1 (published) | — | 75.56 | 0.737 |
| nanojev-0.6b (published) | — | 56.81 | 0.437 |

Paired gaps (score, B − A, 95% CI resampling cases within groups):

| A | B | B − A | 95% CI |
|---|---|---|---|
| `stock-nvfp4 raw` | `swift-nvfp4 raw` | -0.55 | [-0.86, -0.25] |
| `stock-nvfp4 T+B` | `swift-nvfp4 T+B` | +0.19 | [-0.05, +0.44] |
| `stock-nvfp4 raw` | `stock-nvfp4 T` | +3.10 | [+2.79, +3.41] |
| `stock-nvfp4 raw` | `stock-nvfp4 T+B` | +6.79 | [+6.22, +7.36] |
| `swift-nvfp4 raw` | `swift-nvfp4 T` | +3.95 | [+3.61, +4.30] |
| `swift-nvfp4 raw` | `swift-nvfp4 T+B` | +7.54 | [+6.97, +8.13] |
| `openjev-0.1` | `stock-nvfp4 raw` | +1.62 | [+0.98, +2.24] |
| `openjev-0.1` | `stock-nvfp4 T+B` | +8.41 | [+7.70, +9.12] |
| `openjev-0.1` | `swift-nvfp4 raw` | +1.06 | [+0.43, +1.70] |
| `openjev-0.1` | `swift-nvfp4 T+B` | +8.60 | [+7.91, +9.31] |
| `decider-2b-v8` | `stock-nvfp4 raw` | -0.38 | [-1.07, +0.32] |
| `decider-2b-v8` | `stock-nvfp4 T+B` | +6.41 | [+5.84, +6.98] |
| `decider-2b-v8` | `swift-nvfp4 raw` | -0.93 | [-1.66, -0.20] |
| `decider-2b-v8` | `swift-nvfp4 T+B` | +6.61 | [+6.04, +7.17] |
| `jev-1.13.0` | `stock-nvfp4 raw` | -4.52 | [-5.00, -4.07] |
| `jev-1.13.0` | `stock-nvfp4 T+B` | +2.27 | [+1.76, +2.77] |
| `jev-1.13.0` | `swift-nvfp4 raw` | -5.07 | [-5.56, -4.59] |
| `jev-1.13.0` | `swift-nvfp4 T+B` | +2.46 | [+1.97, +2.97] |

## procedural-typed-decisions `test` (874 cases)

| Config | Fit | proper score | accuracy |
|---|---|---|---|
| `stock-nvfp4` | raw | 74.28 | 0.618 |
| `stock-nvfp4` | T | 76.46 | 0.618 |
| `stock-nvfp4` | T+B | 79.64 | 0.689 |
| `swift-nvfp4` | raw | 74.38 | 0.633 |
| `swift-nvfp4` | T | 77.41 | 0.633 |
| `swift-nvfp4` | T+B | 80.30 | 0.707 |

Paired gaps (score, B − A, 95% CI resampling cases within groups):

| A | B | B − A | 95% CI |
|---|---|---|---|
| `stock-nvfp4 raw` | `swift-nvfp4 raw` | +0.10 | [-0.63, +0.84] |
| `stock-nvfp4 T+B` | `swift-nvfp4 T+B` | +0.66 | [+0.04, +1.28] |
| `stock-nvfp4 raw` | `stock-nvfp4 T` | +2.17 | [+1.79, +2.56] |
| `stock-nvfp4 raw` | `stock-nvfp4 T+B` | +5.36 | [+4.55, +6.16] |
| `swift-nvfp4 raw` | `swift-nvfp4 T` | +3.03 | [+2.56, +3.50] |
| `swift-nvfp4 raw` | `swift-nvfp4 T+B` | +5.92 | [+5.02, +6.84] |

## calibrated-typed-decisions `test` (1797 cases)

| Config | Fit | accuracy | log loss | Brier | ECE |
|---|---|---|---|---|---|
| `stock-nvfp4` | raw | 0.823 | 0.602 | 0.274 | 0.095 |
| `stock-nvfp4` | T | 0.823 | 0.457 | 0.247 | 0.010 |
| `stock-nvfp4` | T+B | 0.878 | 0.316 | 0.176 | 0.011 |
| `swift-nvfp4` | raw | 0.838 | 0.668 | 0.266 | 0.099 |
| `swift-nvfp4` | T | 0.838 | 0.438 | 0.236 | 0.011 |
| `swift-nvfp4` | T+B | 0.874 | 0.318 | 0.176 | 0.011 |

Paired gaps (accuracy, B − A, 95% CI resampling cases within groups):

| A | B | B − A | 95% CI |
|---|---|---|---|
| `stock-nvfp4 raw` | `swift-nvfp4 raw` | +0.014 | [+0.006, +0.023] |
| `stock-nvfp4 T+B` | `swift-nvfp4 T+B` | -0.004 | [-0.014, +0.005] |
| `stock-nvfp4 raw` | `stock-nvfp4 T` | +0.000 | [+0.000, +0.000] |
| `stock-nvfp4 raw` | `stock-nvfp4 T+B` | +0.055 | [+0.039, +0.071] |
| `swift-nvfp4 raw` | `swift-nvfp4 T` | +0.000 | [+0.000, +0.000] |
| `swift-nvfp4 raw` | `swift-nvfp4 T+B` | +0.036 | [+0.022, +0.051] |

## calibrated-typed-decisions `heldout` (1000 cases)

| Config | Fit | accuracy | log loss | Brier | ECE |
|---|---|---|---|---|---|
| `stock-nvfp4` | raw | 0.716 | 0.902 | 0.431 | 0.154 |
| `stock-nvfp4` | T | 0.716 | 0.692 | 0.387 | 0.060 |
| `stock-nvfp4` | T+B | 0.716 | 0.719 | 0.395 | 0.086 |
| `swift-nvfp4` | raw | 0.731 | 0.948 | 0.412 | 0.150 |
| `swift-nvfp4` | T | 0.731 | 0.678 | 0.370 | 0.058 |
| `swift-nvfp4` | T+B | 0.731 | 0.703 | 0.375 | 0.079 |

Paired gaps (accuracy, B − A, 95% CI resampling cases within groups):

| A | B | B − A | 95% CI |
|---|---|---|---|
| `stock-nvfp4 raw` | `swift-nvfp4 raw` | +0.015 | [+0.001, +0.029] |
| `stock-nvfp4 T+B` | `swift-nvfp4 T+B` | +0.015 | [+0.001, +0.029] |
| `stock-nvfp4 raw` | `stock-nvfp4 T` | +0.000 | [+0.000, +0.000] |
| `stock-nvfp4 raw` | `stock-nvfp4 T+B` | +0.000 | [+0.000, +0.000] |
| `swift-nvfp4 raw` | `swift-nvfp4 T` | +0.000 | [+0.000, +0.000] |
| `swift-nvfp4 raw` | `swift-nvfp4 T+B` | +0.000 | [+0.000, +0.000] |

