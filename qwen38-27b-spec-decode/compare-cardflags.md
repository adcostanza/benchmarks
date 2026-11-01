$ scripts/compare.py results/dspark results/dspark-cardflags

## Serving (reasoning_effort=medium)

| scenario    | metric        | dspark | dspark-cardflags | dspark-cardflags vs dspark |
|-------------|---------------|--------|------------------|----------------------------|
| sharegpt_c1 | out tok/s     | 148.1  | 146.3            | -1.2%                      |
| sharegpt_c1 | TPOT ms (p50) | 6.92   | 6.33             | -8.6%                      |
| sharegpt_c1 | TTFT ms (p50) | 48     | 59               | +23.8%                     |
| sharegpt_c1 | in tok/s      | 57     | 56               | -1.2%                      |
| sharegpt_c1 | accept len    | 2.84   | 2.89             | +1.7%                      |
| sharegpt_c4 | out tok/s     | 503.8  | 474.6            | -5.8%                      |
| sharegpt_c4 | TPOT ms (p50) | 7.28   | 7.78             | +6.9%                      |
| sharegpt_c4 | TTFT ms (p50) | 74     | 85               | +13.9%                     |
| sharegpt_c4 | in tok/s      | 160    | 150              | -5.8%                      |
| sharegpt_c4 | accept len    | 2.90   | 2.93             | +1.0%                      |
| long20k_c1  | out tok/s     | 88.8   | 87.4             | -1.6%                      |
| long20k_c1  | TPOT ms (p50) | 6.89   | 6.34             | -8.0%                      |
| long20k_c1  | TTFT ms (p50) | 2354   | 2385             | +1.3%                      |
| long20k_c1  | in tok/s      | 3468   | 3413             | -1.6%                      |
| long20k_c1  | accept len    | 2.95   | 3.01             | +1.9%                      |
| long20k_c4  | out tok/s     | 131.2  | 136.1            | +3.8%                      |
| long20k_c4  | TPOT ms (p50) | 16.57  | 17.06            | +3.0%                      |
| long20k_c4  | TTFT ms (p50) | 6366   | 4169             | -34.5%                     |
| long20k_c4  | in tok/s      | 5123   | 5316             | +3.8%                      |
| long20k_c4  | accept len    | 2.93   | 2.98             | +1.6%                      |
| long40k_c1  | out tok/s     | 56.9   | 56.2             | -1.3%                      |
| long40k_c1  | TPOT ms (p50) | 6.79   | 7.38             | +8.8%                      |
| long40k_c1  | TTFT ms (p50) | 5580   | 5600             | +0.4%                      |
| long40k_c1  | in tok/s      | 4446   | 4387             | -1.3%                      |
| long40k_c1  | accept len    | 3.07   | 3.00             | -2.3%                      |
| long80k_c1  | out tok/s     | 27.5   | 28.0             | +2.0%                      |
| long80k_c1  | TPOT ms (p50) | 7.49   | 7.70             | +2.7%                      |
| long80k_c1  | TTFT ms (p50) | 14818  | 14452            | -2.5%                      |
| long80k_c1  | in tok/s      | 4290   | 4377             | +2.0%                      |
| long80k_c1  | accept len    | 2.88   | 3.01             | +4.8%                      |
| long120k_c1 | out tok/s     | 15.9   | 3014.3           | +18830.4%                  |
| long120k_c1 | TPOT ms (p50) | 8.52   | 0.37             | -95.7%                     |
| long120k_c1 | TTFT ms (p50) | 27811  | 0                |                            |
| long120k_c1 | in tok/s      | 3732   | 706479           | +18830.4%                  |
| long120k_c1 | accept len    | 2.70   | -                |                            |

## BFCL v4 subset

| category                  | dspark          | dspark-cardflags |
|---------------------------|-----------------|------------------|
| irrelevance               | 18/20 (90%)     | -                |
| live_irrelevance          | 14/20 (70%)     | -                |
| live_multiple             | 20/25 (80%)     | -                |
| live_simple               | 18/20 (90%)     | -                |
| multi_turn_base           | 13/20 (65%)     | -                |
| multi_turn_miss_param     | 3/10 (30%)      | -                |
| multiple                  | 24/25 (96%)     | -                |
| parallel                  | 18/20 (90%)     | -                |
| parallel_multiple         | 18/20 (90%)     | -                |
| simple_python             | 25/25 (100%)    | -                |
| **total**                 | 171/205 (83.4%) | -                |
| mean out tokens / call    | 144.62          | -                |
| mean latency s / call     | 3.24            | -                |
| calls hitting 4096 cap    | 1               | 0                |
| repeats                   | 1               | 0                |
| accuracy ± SE (binomial)  | 83.4% ± 2.6     | -                |
| gap vs dspark (SE of gap) | -               | -                |


$ scripts/compare.py results/dflash8 results/dflash8-cardflags

## Serving (reasoning_effort=medium)

| scenario    | metric        | dflash8 | dflash8-cardflags | dflash8-cardflags vs dflash8 |
|-------------|---------------|---------|-------------------|------------------------------|
| sharegpt_c1 | out tok/s     | 178.0   | 167.8             | -5.7%                        |
| sharegpt_c1 | TPOT ms (p50) | 5.57    | 5.73              | +2.8%                        |
| sharegpt_c1 | TTFT ms (p50) | 48      | 59                | +23.1%                       |
| sharegpt_c1 | in tok/s      | 68      | 64                | -5.7%                        |
| sharegpt_c1 | accept len    | 3.43    | 3.35              | -2.5%                        |
| sharegpt_c4 | out tok/s     | 592.5   | 520.2             | -12.2%                       |
| sharegpt_c4 | TPOT ms (p50) | 5.91    | 7.15              | +21.0%                       |
| sharegpt_c4 | TTFT ms (p50) | 73      | 86                | +17.3%                       |
| sharegpt_c4 | in tok/s      | 188     | 165               | -12.2%                       |
| sharegpt_c4 | accept len    | 3.49    | 3.28              | -6.0%                        |
| long20k_c1  | out tok/s     | 98.4    | 94.6              | -3.9%                        |
| long20k_c1  | TPOT ms (p50) | 5.37    | 5.73              | +6.6%                        |
| long20k_c1  | TTFT ms (p50) | 2350    | 2381              | +1.3%                        |
| long20k_c1  | in tok/s      | 3845    | 3697              | -3.9%                        |
| long20k_c1  | accept len    | 3.54    | 3.39              | -4.4%                        |
| long20k_c4  | out tok/s     | 137.5   | 144.3             | +4.9%                        |
| long20k_c4  | TPOT ms (p50) | 15.27   | 15.80             | +3.4%                        |
| long20k_c4  | TTFT ms (p50) | 5726    | 5450              | -4.8%                        |
| long20k_c4  | in tok/s      | 5373    | 5636              | +4.9%                        |
| long20k_c4  | accept len    | 3.02    | 3.27              | +8.2%                        |
| long40k_c1  | out tok/s     | 58.7    | 59.2              | +0.8%                        |
| long40k_c1  | TPOT ms (p50) | 5.76    | 6.23              | +8.2%                        |
| long40k_c1  | TTFT ms (p50) | 5574    | 5591              | +0.3%                        |
| long40k_c1  | in tok/s      | 4589    | 4626              | +0.8%                        |
| long40k_c1  | accept len    | 3.28    | 3.45              | +5.1%                        |
| long80k_c1  | out tok/s     | 28.5    | 29.1              | +2.2%                        |
| long80k_c1  | TPOT ms (p50) | 5.88    | 6.20              | +5.4%                        |
| long80k_c1  | TTFT ms (p50) | 14782   | 14419             | -2.5%                        |
| long80k_c1  | in tok/s      | 4453    | 4554              | +2.2%                        |
| long80k_c1  | accept len    | 3.43    | 3.61              | +5.2%                        |
| long120k_c1 | out tok/s     | 16.5    | 2867.4            | +17315.1%                    |
| long120k_c1 | TPOT ms (p50) | 6.53    | 0.39              | -94.1%                       |
| long120k_c1 | TTFT ms (p50) | 27789   | 0                 |                              |
| long120k_c1 | in tok/s      | 3859    | 672043            | +17315.1%                    |
| long120k_c1 | accept len    | 3.46    | -                 |                              |

## BFCL v4 subset

| category                   | dflash8         | dflash8-cardflags |
|----------------------------|-----------------|-------------------|
| irrelevance                | 17/20 (85%)     | -                 |
| live_irrelevance           | 15/20 (75%)     | -                 |
| live_multiple              | 20/25 (80%)     | -                 |
| live_simple                | 18/20 (90%)     | -                 |
| multi_turn_base            | 15/20 (75%)     | -                 |
| multi_turn_miss_param      | 5/10 (50%)      | -                 |
| multiple                   | 24/25 (96%)     | -                 |
| parallel                   | 19/20 (95%)     | -                 |
| parallel_multiple          | 19/20 (95%)     | -                 |
| simple_python              | 24/25 (96%)     | -                 |
| **total**                  | 176/205 (85.9%) | -                 |
| mean out tokens / call     | 141.66          | -                 |
| mean latency s / call      | 0.91            | -                 |
| calls hitting 4096 cap     | 0               | 0                 |
| repeats                    | 1               | 0                 |
| accuracy ± SE (binomial)   | 85.9% ± 2.4     | -                 |
| gap vs dflash8 (SE of gap) | -               | -                 |


$ scripts/compare.py results/dspark-cardflags results/dflash8-cardflags

## Serving (reasoning_effort=medium)

| scenario    | metric        | dspark-cardflags | dflash8-cardflags | dflash8-cardflags vs dspark-cardflags |
|-------------|---------------|------------------|-------------------|---------------------------------------|
| sharegpt_c1 | out tok/s     | 146.3            | 167.8             | +14.7%                                |
| sharegpt_c1 | TPOT ms (p50) | 6.33             | 5.73              | -9.4%                                 |
| sharegpt_c1 | TTFT ms (p50) | 59               | 59                | -0.2%                                 |
| sharegpt_c1 | in tok/s      | 56               | 64                | +14.7%                                |
| sharegpt_c1 | accept len    | 2.89             | 3.35              | +15.8%                                |
| sharegpt_c4 | out tok/s     | 474.6            | 520.2             | +9.6%                                 |
| sharegpt_c4 | TPOT ms (p50) | 7.78             | 7.15              | -8.1%                                 |
| sharegpt_c4 | TTFT ms (p50) | 85               | 86                | +1.4%                                 |
| sharegpt_c4 | in tok/s      | 150              | 165               | +9.6%                                 |
| sharegpt_c4 | accept len    | 2.93             | 3.28              | +12.0%                                |
| long20k_c1  | out tok/s     | 87.4             | 94.6              | +8.3%                                 |
| long20k_c1  | TPOT ms (p50) | 6.34             | 5.73              | -9.7%                                 |
| long20k_c1  | TTFT ms (p50) | 2385             | 2381              | -0.2%                                 |
| long20k_c1  | in tok/s      | 3413             | 3697              | +8.3%                                 |
| long20k_c1  | accept len    | 3.01             | 3.39              | +12.6%                                |
| long20k_c4  | out tok/s     | 136.1            | 144.3             | +6.0%                                 |
| long20k_c4  | TPOT ms (p50) | 17.06            | 15.80             | -7.4%                                 |
| long20k_c4  | TTFT ms (p50) | 4169             | 5450              | +30.7%                                |
| long20k_c4  | in tok/s      | 5316             | 5636              | +6.0%                                 |
| long20k_c4  | accept len    | 2.98             | 3.27              | +9.9%                                 |
| long40k_c1  | out tok/s     | 56.2             | 59.2              | +5.5%                                 |
| long40k_c1  | TPOT ms (p50) | 7.38             | 6.23              | -15.7%                                |
| long40k_c1  | TTFT ms (p50) | 5600             | 5591              | -0.2%                                 |
| long40k_c1  | in tok/s      | 4387             | 4626              | +5.5%                                 |
| long40k_c1  | accept len    | 3.00             | 3.45              | +14.9%                                |
| long80k_c1  | out tok/s     | 28.0             | 29.1              | +4.0%                                 |
| long80k_c1  | TPOT ms (p50) | 7.70             | 6.20              | -19.4%                                |
| long80k_c1  | TTFT ms (p50) | 14452            | 14419             | -0.2%                                 |
| long80k_c1  | in tok/s      | 4377             | 4554              | +4.0%                                 |
| long80k_c1  | accept len    | 3.01             | 3.61              | +19.6%                                |
| long120k_c1 | out tok/s     | 3014.3           | 2867.4            | -4.9%                                 |
| long120k_c1 | TPOT ms (p50) | 0.37             | 0.39              | +5.0%                                 |
| long120k_c1 | TTFT ms (p50) | 0                | 0                 |                                       |
| long120k_c1 | in tok/s      | 706479           | 672043            | -4.9%                                 |


