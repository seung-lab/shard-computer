#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iomanip>
#include <cstdint>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "MurmurHash3.h"

namespace py = pybind11;

// Some of the python code for computing this hash:
// uint64(mmh3.hash64(uint64(x).tobytes(), x64arch=False)[0]) 

// def compute_shard_location(self, key):
//   chunkid = uint64(key) >> uint64(self.preshift_bits)
//   chunkid = self.hashfn(chunkid)
//   minishard_number = uint64(chunkid & self.minishard_mask)
//   shard_number = uint64((chunkid & self.shard_mask) >> uint64(self.minishard_bits))
//   shard_number = format(shard_number, 'x').zfill(int(np.ceil(self.shard_bits / 4.0)))
//   remainder = chunkid >> uint64(self.minishard_bits + self.shard_bits)

//   return ShardLocation(shard_number, minishard_number, remainder)

uint64_t compute_shard_mask(const uint64_t shard_bits, const uint64_t minishard_bits) {
    uint64_t movement = minishard_bits + shard_bits;
    uint64_t shard_mask = ~((0xffffffffffffffff >> movement) << movement);
    return ((shard_mask >> minishard_bits) << minishard_bits);
}

py::str shard_number(
	const uint64_t label, 
	const uint64_t preshift_bits, 
	const uint64_t shard_bits, 
	const uint64_t minishard_bits
) {
	const int zfill = (shard_bits + 3) / 4;
	
	uint64_t shard_mask = compute_shard_mask(shard_bits, minishard_bits);

	std::stringstream strm;
	uint64_t chunk_id = label >> preshift_bits;
	chunk_id = MurmurHash3_x86_64(chunk_id, /*seed=*/0);
	chunk_id = (chunk_id & shard_mask) >> minishard_bits;
	strm << std::setfill('0') << std::setw(zfill) << std::hex << chunk_id;
	return strm.str();
}

auto assign_labels_to_shards(
	const py::array_t<uint64_t, py::array::c_style> labels, 
	const uint64_t preshift_bits, 
	const uint64_t shard_bits, 
	const uint64_t minishard_bits
) {
	const uint64_t size = labels.size();
	
	auto labels_view = labels.unchecked<1>();
	uint64_t shard_mask = compute_shard_mask(shard_bits, minishard_bits);

	std::unordered_map<std::string, std::vector<uint64_t>> all_labels;
	const int zfill = (shard_bits + 3) / 4;
	std::stringstream strm;

	for (uint64_t i = 0; i < size; i++) {
		uint64_t chunk_id = labels_view(i) >> preshift_bits;
		chunk_id = MurmurHash3_x86_64(chunk_id, /*seed=*/0);
		chunk_id = (chunk_id & shard_mask) >> minishard_bits;
		strm.str("");
		strm.clear();
		strm << std::setfill('0') << std::setw(zfill) << std::hex << chunk_id;
		all_labels[strm.str()].push_back(labels_view(i));
	}

	return all_labels;
}

py::set unique_shard_numbers(
	const py::array_t<uint64_t, py::array::c_style> labels, 
	const uint64_t preshift_bits, 
	const uint64_t shard_bits, 
	const uint64_t minishard_bits
) {
	const uint64_t size = labels.size();
	
	auto labels_view = labels.unchecked<1>();
	uint64_t shard_mask = compute_shard_mask(shard_bits, minishard_bits);

	py::set hashes;
	const int zfill = (shard_bits + 3) / 4;
	std::stringstream strm;

	for (uint64_t i = 0; i < size; i++) {
		uint64_t chunk_id = labels_view(i) >> preshift_bits;
		chunk_id = MurmurHash3_x86_64(chunk_id, /*seed=*/0);
		chunk_id = (chunk_id & shard_mask) >> minishard_bits;
		strm.str("");
		strm.clear();
		strm << std::setfill('0') << std::setw(zfill) << std::hex << chunk_id;
		hashes.add(strm.str());
	}

	return hashes;
}

PYBIND11_MODULE(shardcomputer, m) {
    m.doc() = "Module for computing Neuroglancer Precomputed shard hashes rapidly."; // optional module docstring

    m.def("shard_number", &shard_number, 
    	"Compute the shard file hash from a label. Returns str.");

    m.def("unique_shard_numbers", &unique_shard_numbers, 
    	"Compute the set of unique shard file hashes from a numpy array of labels. Returns set.");

	m.def("assign_labels_to_shards", &assign_labels_to_shards, 
    	"From an array of labels, create a dictionary of shardnumber -> list of labels.");

	m.def("MurmurHash3_x86_64", &MurmurHash3_x86_64, 
    	"Compute the MurmurHash3_x86_64 of a uint64.");

}