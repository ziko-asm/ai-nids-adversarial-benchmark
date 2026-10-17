Detection rate (recall on attack records, test split), mean ± std over seeds

| Model | Defence | Clean | FGSM ε=0.5 | PGD ε=0.5 | PGD ε=0.5, constrained | PGD ε=1, constrained |
|---|---|---|---|---|---|---|
| MLP | none | 0.998 ± 0.000 | 0.419 ± 0.095 | 0.072 ± 0.009 | 0.979 ± 0.004 | 0.806 ± 0.041 |
| MLP | adv_train | 0.998 ± 0.000 | 0.990 ± 0.001 | 0.977 ± 0.005 | 0.995 ± 0.001 | 0.966 ± 0.007 |
| CNN | none | 0.997 ± 0.001 | 0.037 ± 0.020 | 0.014 ± 0.009 | 0.912 ± 0.020 | 0.083 ± 0.026 |
| CNN | adv_train | 0.993 ± 0.003 | 1.000 ± 0.000 | 0.984 ± 0.003 | 0.980 ± 0.004 | 0.835 ± 0.138 |
| GBT (transfer) | none | 0.999 ± 0.000 | — | 0.952 ± 0.011 | 0.953 ± 0.009 | 0.526 ± 0.209 |
