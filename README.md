![Automated Tests](https://github.com/github/docs/actions/workflows/run_tests.yml/badge.svg) [![PyPI version](https://badge.fury.io/py/shard-computer.svg)](https://badge.fury.io/py/shard-computer)


# shard-computer
Perform accelerated shard hash computation for [Neuroglancer Precomputed shards](https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/sharded.md#sharding-specification).

```python
import shardcomputer

preshift_bits = 0
shard_bits = 11
minishard_bits = 8

label = 12949142

shard_no = shardcomputer.shard_number(label, preshift_bits, shard_bits, minishard_bits)

# let labels be a uint64 numpy array of labels
# e.g. set(['4d2'])
shard_no_set = shardcomputer.unique_shard_numbers(
	labels, preshift_bits, shard_bits, minishard_bits
)

# Returns shard number -> label list dict
# e.g. {'4d2': [12949142]}
shard_no_to_labels = shardcomputer.assign_labels_to_shards(
	label, preshift_bits, shard_bits, minishard_bits
)

```

## Install

```
pip install shard-computer
```

## Credits

Thank you to Austin Appleby for placing MurMurhash3 into the public domain.

