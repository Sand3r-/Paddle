# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest
import paddle.fluid as fluid
import paddle.fluid.core as core


def fully_connected_naive(input, weights, bias_data):
    input2d = input.reshape(input.shape[0], -1)
    result = np.dot(input2d, weights) + bias_data
    return result


def fully_connected_3d_naive(input, weights, bias_data):
    input2d = input.reshape(input.shape[0] * input.shape[1], -1)
    result = np.dot(input2d, weights) + bias_data
    return result


class Matrix4Generate:
    def __init__(self, mb, ic, oc, h, w):
        self.input = np.random.random((mb, ic * h * w)).astype("float32")
        self.weights = np.random.random((ic * h * w, oc)).astype("float32")
        self.bias = np.random.random(oc).astype("float32")


class Matrix3Generate:
    def __init__(self, mb, ic, oc, w, dt="float32"):
        if True:
            self.input = np.array([[[1, 2], [4, 8], [4, 2]]]).astype(dt)
            print(self.input)
            self.weights = np.array([[
                1.0,
                3.0,
            ], [4.0, 5.0]]).astype("float32")
            print(self.weights)
            self.bias = np.array([0.0, 0.0]).astype("float32")
            print(self.bias)
        else:
            self.input = np.random.random((mb, ic, w)).astype(dt)
            self.weights = np.random.random((w, oc)).astype("float32")
            self.bias = np.random.random(oc).astype("float32")


class TestFCMKLDNNOp(OpTest):
    def create_data(self):
        self.matrix = Matrix4Generate(1, 10, 15, 3, 3)

    def setUp(self):
        self.op_type = "fc"
        self._cpu_only = True
        self.use_mkldnn = True
        self.create_data()
        self.inputs = {
            'Input': self.matrix.input,
            'W': self.matrix.weights,
            'Bias': self.matrix.bias
        }

        if len(self.matrix.input.shape) == 3:
            self.outputs = {
                'Out': fully_connected_3d_naive(
                    self.matrix.input, self.matrix.weights, self.matrix.bias)
            }
            self.attrs = {
                'use_mkldnn': self.use_mkldnn,
                'in_num_col_dims': 2,
                'scale_weights': [1.0],
                'scale_in': 1.0
            }
        else:
            self.outputs = {
                'Out': fully_connected_naive(
                    self.matrix.input, self.matrix.weights, self.matrix.bias)
            }
            self.attrs = {'use_mkldnn': self.use_mkldnn}

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_output(check_dygraph=False)

    def test_check_grad_normal(self):
        pass

    def test_check_grad_no_weight(self):
        pass


class TestMat4FCMKLDNNOp(TestFCMKLDNNOp):
    def create_data(self):
        self.matrix = Matrix4Generate(2, 15, 48, 2, 2)


class TestMat3FCMKLDNNOp(TestFCMKLDNNOp):
    def create_data(self):
        batch_size = 1
        input_channels = 3
        output_channels = 2
        width = 2
        self.matrix = Matrix3Generate(batch_size, input_channels,
                                      output_channels, width)


class TestMat3Int8FCMKLDNNOp(TestFCMKLDNNOp):
    def create_data(self):
        batch_size = 1
        input_channels = 3
        output_channels = 2
        width = 2
        self.matrix = Matrix3Generate(batch_size, input_channels,
                                      output_channels, width)

    def test_check_output(self):
        self.output = fully_connected_3d_naive(
            self.matrix.input, self.matrix.weights, self.matrix.bias)
        self.attrs = {
            'use_mkldnn': self.use_mkldnn,
            'in_num_col_dims': 2,
            'scale_weights': [1.0],
            'scale_in': 1.0
        }
        self.fetch_list = ['fc_output']
        ground_truth = {
            "input": self.matrix.input,
            "quant_output": self.matrix.input.astype(np.int8),
            "filter": self.matrix.weights,
            "bias": self.matrix.bias,
            "fc_output": self.output,
        }
        program = fluid.Program()
        with fluid.program_guard(program):
            block = program.global_block()
            for name in ground_truth:
                block.create_var(
                    name=name, dtype="float32", shape=ground_truth[name].shape)
            quantize_op = block.append_op(
                type="quantize",
                inputs={"Input": block.var('input'), },
                outputs={"Output": block.var('quant_output')},
                attrs={
                    'use_mkldnn': self.use_mkldnn,
                    'is_negative_input': True,
                    'Scale': 1.0
                })
            fc_op = block.append_op(
                type="fc",
                inputs={
                    'Input': block.var('quant_output'),
                    'W': block.var('filter'),
                    'Bias': block.var('bias')
                },
                outputs={"Out": block.var('fc_output')},
                attrs={
                    'use_mkldnn': self.use_mkldnn,
                    'in_num_col_dims': 2,
                    'scale_weights': [1.0],
                    'scale_in': 1.0
                })
            place = core.CPUPlace()
            exe = fluid.Executor(place)
            out = exe.run(program,
                          feed={
                              name: ground_truth[name]
                              for name in ["input", "filter", "bias"]
                          },
                          fetch_list=self.fetch_list)

            for id, name in enumerate(self.fetch_list):
                output_string = str(name) + ' ' + str(self.output)
                self.assertTrue(
                    np.allclose(
                        ground_truth[name], out[id], atol=1e-4),
                    output_string)


if __name__ == "__main__":
    unittest.main()
