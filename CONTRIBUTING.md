# Contributing to Wildfire Response Prioritization

Thank you for your interest in contributing to this project!

## Code Architecture & Constraints

When proposing modifications or improvements, please adhere to the following core constraints:

1. **Permitted Input Features Only**:
   Only 12 input features may be used for model inference:
   `X, Y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain`.
2. **Zero Target Leakage**:
   All scalers, encoders, and the high-impact threshold ($P_{80}$) must be fitted exclusively on the training partition.
   Evaluation routines must never access target labels or compute test-set distribution statistics.
3. **Operational Constraint Enforcement**:
   The response selection engine must guarantee:
   - Exactly 25 rows selected.
   - At most 4 observations per $(X, Y)$ coordinate cell.
4. **Reproducibility**:
   All random states must be deterministic and configurable.

## Running Tests

Before submitting changes, ensure that all unit and edge case tests pass:

```bash
# Run pytest suite
pytest test_web_server.py test_web_edge_cases.py -v

# Run cross-validation benchmark
python train_and_evaluate.py --data-path data/forestfires.csv

# Run adversarial stress testing
python stress_test_worst_scenarios.py
```
