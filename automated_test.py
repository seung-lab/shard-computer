import pytest
import numpy as np
import mmh3
import shardcomputer

def test_MurmurHash3_x86_64():
	arr = np.arange(1000000)

	for label in arr:
		orig = mmh3.hash64(np.uint64(label).tobytes(), x64arch=False, signed=False)[0]
		assert shardcomputer.MurmurHash3_x86_64(label, 0) == orig

	arr = np.random.randint(0, 100000000, size=(100000,))
	for label in arr:
		orig = mmh3.hash64(np.uint64(label).tobytes(), x64arch=False, signed=False)[0]
		assert shardcomputer.MurmurHash3_x86_64(label, 0) == orig

