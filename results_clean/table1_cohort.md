# Table 1. Cohort characteristics (patient-grouped split)

Generated directly from the curated event table; do not edit by hand.

| Variable                                    | All           | Train         | Validation    | Test          | p (train vs test)   | Test used                                          |
|:--------------------------------------------|:--------------|:--------------|:--------------|:--------------|:--------------------|:---------------------------------------------------|
| **Patients**                                | 736           | 588           | 74            | 74            |                     |                                                    |
| of which id-less recordings (training-only) | 0             | 0             | 0             | 0             |                     |                                                    |
| Age (years), median (IQR)                   | 4.9 (3.4–7.2) | 4.8 (3.4–7.2) | 5.5 (3.3–8.3) | 5.0 (3.5–7.0) | 0.702               | Mann-Whitney U                                     |
| Sex, n (%)                                  |               |               |               |               | 0.519               | chi-square                                         |
| Female                                      | 362           | 290 (49.3)    | 32 (43.2)     | 40 (54.1)     |                     |                                                    |
| Male                                        | 374           | 298 (50.7)    | 42 (56.8)     | 34 (45.9)     |                     |                                                    |
| Disease group (patient-level), n (%)        |               |               |               |               | 1.000               | chi-square                                         |
| Pneumonia                                   | 438           | 350 (59.5)    | 44 (59.5)     | 44 (59.5)     |                     |                                                    |
| Bronchial diseases                          | 179           | 143 (24.3)    | 18 (24.3)     | 18 (24.3)     |                     |                                                    |
| Normal / Other                              | 119           | 95 (16.2)     | 12 (16.2)     | 12 (16.2)     |                     |                                                    |
| **Events**                                  | 19,693        | 15,877        | 1,913         | 1,903         |                     |                                                    |
| Event type, n (%)                           |               |               |               |               | <0.001              | chi-square (sparse cells - interpret with caution) |
| Normal                                      | 16,396        | 13,200 (83.1) | 1,557 (81.4)  | 1,639 (86.1)  |                     |                                                    |
| Fine Crackle                                | 1,692         | 1,382 (8.7)   | 196 (10.2)    | 114 (6.0)     |                     |                                                    |
| Wheeze                                      | 1,243         | 1,001 (6.3)   | 131 (6.8)     | 111 (5.8)     |                     |                                                    |
| Rhonchi                                     | 208           | 172 (1.1)     | 18 (0.9)      | 18 (0.9)      |                     |                                                    |
| Coarse Crackle                              | 112           | 89 (0.6)      | 10 (0.5)      | 13 (0.7)      |                     |                                                    |
| Wheeze+Crackle                              | 42            | 33 (0.2)      | 1 (0.1)       | 8 (0.4)       |                     |                                                    |
| Recording location, n (%)                   |               |               |               |               | 0.607               | chi-square                                         |
| p1                                          | 4,981         | 4,011 (25.3)  | 492 (25.7)    | 478 (25.1)    |                     |                                                    |
| p2                                          | 5,329         | 4,335 (27.3)  | 497 (26.0)    | 497 (26.1)    |                     |                                                    |
| p3                                          | 4,562         | 3,669 (23.1)  | 432 (22.6)    | 461 (24.2)    |                     |                                                    |
| p4                                          | 4,821         | 3,862 (24.3)  | 492 (25.7)    | 467 (24.5)    |                     |                                                    |

Patients are grouped so that every event of one patient sits in a single partition. Recordings whose SPRSound filename carries no patient identifier are training-only and are therefore excluded from the validation and test columns and from all patient-level statistics.
