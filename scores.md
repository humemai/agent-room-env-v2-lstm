# Scores

Updated on 20-Apr-2024. I changed hyperparameters, i.e., disabling ddqn and reducing the
warm start size, and the results look different from those of the original paper.

## stochastic objects

### l

```python
capacity = {
    "episodic": 16,
    "episodic_agent": 0,
    "semantic": 16,
    "semantic_map": 0,
    "short": 1,
}
```

The number of room: 16\
The number of static_objects: 4\
The number of independent_objects: 4\
The number of dependent_objects: 4

#### memory-based LSTM agent

| memory mgmt | qa                | explore     | val_mean | val_std | test_mean | test_std | num_params |
| ----------- | ----------------- | ----------- | -------- | ------- | --------- | -------- | ---------- |
| random      | episodic_semantic | random      |          |         | 204.32    | 15.74    |            |
| random      | episodic_semantic | avoid_walls |          |         | 234.86    | 32.87    |            |
| episodic    | episodic_semantic | random      |          |         | 107.66    | 7.12     |            |
| episodic    | episodic_semantic | avoid_walls |          |         | 137.66    | 19.07    |            |
| semantic    | episodic_semantic | random      |          |         | 130.76    | 18.14    |            |
| semantic    | episodic_semantic | avoid_walls |          |         | 135.28    | 48.18    |            |
| RL          | episodic_semantic | avoid_walls | 385.17   | 38.22   | 330.34    | 48.00    | 206,469    |
| neural      | episodic_semantic | RL          | 447.8    | 38.56   | 405.41    | 50.62    | 206,469    |

#### history-based LSTM agent

average observations per room (block): 5.71

| history_block_size | avg_obs_size | explore     | val_mean | val_std | test_mean | test_std | num_params |
| ------------------ | ------------ | ----------- | -------- | ------- | --------- | -------- | ---------- |
| 1                  | 5.71         | random      |          |         | 55.8      | 26.46    |            |
| 1                  | 5.71         | avoid_walls |          |         | 62.0      | 20.24    |            |
| 1                  | 5.71         | RL          | 258.96   | 15.35   | 214.94    | 19.94    | 285,957    |
| 6                  | 34.26        | random      |          |         | 143.4     | 43.34    |            |
| 6                  | 34.26        | avoid_walls |          |         | 238.8     | 52.76    |            |
| 6                  | 34.26        | RL          | 287.16   | 54.24   | 268.7     | 49.05    | 285,957    |
| 12                 | 68.52        | random      |          |         | 207.6     | 54.04    |            |
| 12                 | 68.52        | avoid_walls |          |         | 239.8     | 106.87   |            |
| 12                 | 68.52        | RL          | 388.26   | 146.64  | 382.72    | 148.79   | 285,957    |
| 24                 | 137.04       | random      |          |         | 272.8     | 74.70    |            |
| 24                 | 137.04       | avoid_walls |          |         | 385.8     | 133.21   |            |
| 24                 | 137.04       | RL          | 296.66   | 62.51   | 265.26    | 98.73    | 285,957    |
| 48                 | 274.08       | random      |          |         | 361.0     | 103.62   |            |
| 48                 | 274.08       | avoid_walls |          |         | 474.8     | 59.63    |            |
| 48                 | 274.08       | RL          | 281.4    | 12.09   | 251.76    | 32.63    | 285,957    |
| 100                | 571          | random      |          |         | 417.6     | 143.32   |            |
| 100                | 571          | avoid_walls |          |         | 498.6     | 61.25    |            |
| 100                | 571          | RL          | 448.12   | 87.07   | 437.66    | 95.34    | 285,957    |

## deterministic objects

### l

```python
capacity = {
    "episodic": 16,
    "episodic_agent": 0,
    "semantic": 16,
    "semantic_map": 0,
    "short": 1,
}
```

The number of room: 16\
The number of static_objects: 4\
The number of independent_objects: 4\
The number of dependent_objects: 4

#### memory-based LSTM agent

| memory mgmt | qa                | explore     | val_mean | val_std | test_mean | test_std | num_params |
| ----------- | ----------------- | ----------- | -------- | ------- | --------- | -------- | ---------- |
| random      | episodic_semantic | random      |          |         | 188.98    | 15.01    |            |
| random      | episodic_semantic | avoid_walls |          |         | 196.49    | 12.12    |            |
| episodic    | episodic_semantic | random      |          |         | 104.3     | 7.21     |            |
| episodic    | episodic_semantic | avoid_walls |          |         | 110.38    | 4.94     |            |
| semantic    | episodic_semantic | random      |          |         | 96.58     | 15.83    |            |
| semantic    | episodic_semantic | avoid_walls |          |         | 227.94    | 4.11     |            |
| RL          | episodic_semantic | avoid_walls | 364.46   | 39.27   | 305.52    | 36.78    | 206,339    |
| neural      | episodic_semantic | RL          | 463.74   | 41.52   | 465.38    | 48.02    | 206,469    |

#### history-based LSTM agent

average observations per room (block): 5.71

| history_block_size | avg_obs_size | explore     | val_mean | val_std | test_mean | test_std | num_params |
| ------------------ | ------------ | ----------- | -------- | ------- | --------- | -------- | ---------- |
| 1                  | 5.71         | random      |          |         | 59.2      | 8.70     |            |
| 1                  | 5.71         | avoid_walls |          |         | 57.8      | 6.67     |            |
| 1                  | 5.71         | RL          | 218.54   | 3.15    | 216.76    | 4.84     | 285,957    |
| 6                  | 34.26        | random      |          |         | 119.4     | 14.8     |            |
| 6                  | 34.26        | avoid_walls |          |         | 187.2     | 35.92    |            |
| 6                  | 34.26        | RL          | 424.34   | 10.86   | 421.96    | 10.38    | 285,957    |
| 12                 | 68.52        | random      |          |         | 171.2     | 13.61    |            |
| 12                 | 68.52        | avoid_walls |          |         | 257.0     | 36.93    |            |
| 12                 | 68.52        | RL          | 458.24   | 53.0    | 457.66    | 49.59    | 285,957    |
| 24                 | 137.04       | random      |          |         | 262.6     | 25.46    |            |
| 24                 | 137.04       | avoid_walls |          |         | 369.6     | 40.85    |            |
| 24                 | 137.04       | RL          | 431.32   | 64.62   | 430.16    | 62.61    | 285,957    |
| 48                 | 274.08       | random      |          |         | 358.8     | 31.79    |            |
| 48                 | 274.08       | avoid_walls |          |         | 404.2     | 48.11    |            |
| 48                 | 274.08       | RL          | 205.7    | 13.64   | 197.74    | 13.1     | 285,957    |
| 100                | 571          | random      |          |         | 413.8     | 64.95    |            |
| 100                | 571          | avoid_walls |          |         | 423.6     | 78.17    |            |
| 100                | 571          | RL          | 294.82   | 67.3    | 288.96    | 65.33    | 285,957    |

### xl

THIS DOESN'T WORK AT ALL!! MAYBE GNN WILL HELP
