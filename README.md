![Automated Tests](https://github.com/github/docs/actions/workflows/run_tests.yml/badge.svg)


# shard-computer
Perform accelerated shard hash computation for [Neuroglancer Precomputed shards](https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/sharded.md#sharding-specification).

```python
import shardcomputer

preshift_bits = 0
shard_bits = 11
minishard_bits = 8

label = 12949142

shard_no = shardcomputer.shard_number(label, preshift_bits, shard_bits, minishard_bits)

# let arr be a uint64 numpy array of labels
shard_no_set = shardcomputer.unique_shard_numbers(
	label, preshift_bits, shard_bits, minishard_bits
)
```
