import os.path
import setuptools
from pybind11.setup_helpers import Pybind11Extension

def read(fname):
  with open(os.path.join(os.path.dirname(__file__), fname), 'rt') as f:
    return f.read()

ext_modules = [
    Pybind11Extension(
        "shardcomputer",
        [ "MurmurHash3.cpp", "shardcomputer.cpp" ] ,  # Sort source files for reproducibility
        cxx_std=14,
    ),
]

setuptools.setup(
  name="shard-computer",
  version="1.2.1",
  setup_requires=[
    'numpy', 
    'pybind11',
  ],
  python_requires=">=3.8.0,<4.0.0",
  author="William Silversmith",
  author_email="ws9@princeton.edu",
  packages=setuptools.find_packages(),
  package_data={
    'shardcomputer': [
      'LICENSE',
    ],
  },
  description="A library for rapid computation of Neuroglancer Precomputed shard hashes.",
  long_description=read('README.md'),
  long_description_content_type="text/markdown",
  license = "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
  keywords = "neuroglancer sharding igneous cloud-volume MurmurHash3",
  url = "https://github.com/seung-lab/shard-computer/",
  classifiers=[
    "Intended Audience :: Developers",
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows :: Windows 10",
    "Topic :: Utilities",
  ],
  ext_modules=ext_modules,
)
