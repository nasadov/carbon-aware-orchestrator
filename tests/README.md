# Tests And Validation

The configured test gate is:

```bash
make syntax-check
make lint
make test
```

`test_repo_hygiene.py` contains lightweight smoke checks for repository-relative
paths and water-sweep provenance.

The validator scripts in this directory can also be run directly against
generated experiment outputs:

```bash
python3 tests/placement_constraint_validator.py path/to/placements.csv
python3 tests/validate_experiments_and_report.py --experiments-dir experiments
```
