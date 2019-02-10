# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages


with open("qhoptim/version.py") as infile:
    exec(infile.read())


setup(
    name="qhoptim",
    version=version,
    description="Quasi-hyperbolic optimization algorithms from Facebook AI Research.",
    author="Facebook AI Research",
    url="https://github.com/facebookresearch/qhoptim",
    license="MIT",
    packages=find_packages(exclude=["test_*"]),
    data_files=[("source_docs/qhoptim", ["LICENSE", "README.rst", "CODE_OF_CONDUCT.rst", "CONTRIBUTING.rst"])],
    zip_safe=True,
)
