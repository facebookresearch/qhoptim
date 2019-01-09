#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e

cd "${0%/*}"
make clean html
rm -rf ../docs/
cp -r _build/html/ ../docs
