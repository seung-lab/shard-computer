from collections import defaultdict

import pytest
import numpy as np
import mmh3
import shardcomputer
from cloudvolume.datasource.precomputed.sharding import ShardingSpecification

def test_MurmurHash3_x86_64():
    arr = np.arange(1000000)

    for label in arr:
        orig = mmh3.hash64(np.uint64(label).tobytes(), x64arch=False, signed=False)[0]
        assert shardcomputer.MurmurHash3_x86_64(label, 0) == orig

    arr = np.random.randint(0, 100000000, size=(100000,))
    for label in arr:
        orig = mmh3.hash64(np.uint64(label).tobytes(), x64arch=False, signed=False)[0]
        assert shardcomputer.MurmurHash3_x86_64(label, 0) == orig

@pytest.mark.parametrize('preshift_bits', [0,1,2])
@pytest.mark.parametrize('shard_bits', [0,8,11])
@pytest.mark.parametrize('minishard_bits', [0,4,7])
def test_shard_number(preshift_bits, shard_bits, minishard_bits):
    spec = ShardingSpecification(
        'neuroglancer_uint64_sharded_v1',
        preshift_bits=preshift_bits, 
        hash='murmurhash3_x86_128', 
        minishard_bits=minishard_bits, 
        shard_bits=shard_bits, 
    )

    for label in range(1000):
        cv_shard_no = spec.compute_shard_location(label).shard_number
        sc_shard_no = shardcomputer.shard_number(label, preshift_bits, shard_bits, minishard_bits)
        assert cv_shard_no == sc_shard_no, label

    arr = np.random.randint(0, 100000000, size=(10000,))
    for label in arr:
        cv_shard_no = spec.compute_shard_location(label).shard_number
        sc_shard_no = shardcomputer.shard_number(label, preshift_bits, shard_bits, minishard_bits)
        assert cv_shard_no == sc_shard_no, label

@pytest.mark.parametrize('preshift_bits', [0,1,2])
@pytest.mark.parametrize('shard_bits', [0,1,11])
@pytest.mark.parametrize('minishard_bits', [0,1,7])
def test_assign_labels_to_shards(preshift_bits, shard_bits, minishard_bits):
    spec = ShardingSpecification(
        'neuroglancer_uint64_sharded_v1',
        preshift_bits=preshift_bits, 
        hash='murmurhash3_x86_128', 
        minishard_bits=minishard_bits, 
        shard_bits=shard_bits, 
    )

    cv_label_map = defaultdict(list)
    arr = np.random.randint(0, 100000000, size=(10000,), dtype=np.uint64)
    for label in arr:
        cv_shard_no = spec.compute_shard_location(label).shard_number
        cv_label_map[cv_shard_no].append(label)

    sc_label_map = shardcomputer.assign_labels_to_shards(arr, preshift_bits, shard_bits, minishard_bits)
    assert sc_label_map == cv_label_map

@pytest.mark.parametrize('preshift_bits', [0,1,2])
@pytest.mark.parametrize('shard_bits', [0,1,11])
@pytest.mark.parametrize('minishard_bits', [0,1,7])
def test_assign_labels_to_shards_and_minishards(preshift_bits, shard_bits, minishard_bits):
    spec = ShardingSpecification(
        'neuroglancer_uint64_sharded_v1',
        preshift_bits=preshift_bits, 
        hash='murmurhash3_x86_128', 
        minishard_bits=minishard_bits, 
        shard_bits=shard_bits, 
    )

    cv_label_map = {}
    arr = np.random.randint(0, 100000000, size=(10000,), dtype=np.uint64)
    for label in arr:
        loc = spec.compute_shard_location(label)
        if loc.shard_number not in cv_label_map:
            cv_label_map[loc.shard_number] = {}

        if loc.minishard_number not in cv_label_map[loc.shard_number]:
            cv_label_map[loc.shard_number][loc.minishard_number] = []

        cv_label_map[loc.shard_number][loc.minishard_number].append(label)

    sc_label_map = shardcomputer.assign_labels_to_shards_and_minishards(arr, preshift_bits, shard_bits, minishard_bits, True)

    for shard_no, minishards in sc_label_map.items():
        for minishard_no, labels in minishards.items():
            labels.sort()

    for shard_no, minishards in cv_label_map.items():
        for minishard_no, labels in minishards.items():
            labels.sort()

    assert sc_label_map == cv_label_map

@pytest.mark.parametrize('preshift_bits', [0,1,2])
@pytest.mark.parametrize('shard_bits', [0,8,11])
@pytest.mark.parametrize('minishard_bits', [0,4,7])
def test_unique_shard_numbers(preshift_bits, shard_bits, minishard_bits):
    spec = ShardingSpecification(
        'neuroglancer_uint64_sharded_v1',
        preshift_bits=preshift_bits, 
        hash='murmurhash3_x86_128', 
        minishard_bits=minishard_bits, 
        shard_bits=shard_bits, 
    )

    cv_bag = set()
    for label in range(1000):
        cv_shard_no = spec.compute_shard_location(label).shard_number
        cv_bag.add(cv_shard_no)

    arr = np.arange(1000, dtype=np.uint64)
    sc_bag = shardcomputer.unique_shard_numbers(arr, preshift_bits, shard_bits, minishard_bits)
    assert cv_bag == sc_bag

    cv_bag = set()
    arr = np.random.randint(0, 100000000, size=(10000,), dtype=np.uint64)
    for label in arr:
        cv_shard_no = spec.compute_shard_location(label).shard_number
        cv_bag.add(cv_shard_no)
    
    sc_bag = shardcomputer.unique_shard_numbers(arr, preshift_bits, shard_bits, minishard_bits)
    assert cv_bag == sc_bag
