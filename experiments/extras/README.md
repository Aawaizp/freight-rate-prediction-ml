# Extra experiments (post-final-model)

These 4 scripts were run **after** the final model (tuned HGB, MAE 119.40)
was already selected, to stress-test whether it could still be improved.
None of them found a consistent improvement, so none changed the final
pipeline - they are kept here as a record of what was tried and why it
was rejected, separate from the main `experiments/01-10` scripts that
document how the final model was actually built.

| Script | Idea tested | Result |
|---|---|---|
| `11_rate_per_mile_target.py` | Predict rate-per-mile instead of posted_rate | Worse on average and inconsistent across windows - rejected |
| `12_route_geometry_features.py` | Extra lat/lon-derived features (midpoints, differences, interactions) | No consistent improvement - rejected |
| `13_route_history_features.py` | Historical average rate per route | Made results notably worse (route-level averages are too noisy with only 64 routes) - rejected |
| `14_high_rate_long_haul.py` | Explicit long-haul/high-rate flags | No effect - the model already uses distance to capture this on its own - rejected |

Each script can be run directly, e.g. `python experiments/extras/11_rate_per_mile_target.py`.
