#!/usr/bin/env sh

#./build/tools/caffe_fp16 train --solver=examples/mnist/lenet_solver.prototxt
./build/tools/caffe train -solver=examples/mnist/lenet_solver.prototxt -gpu=0
