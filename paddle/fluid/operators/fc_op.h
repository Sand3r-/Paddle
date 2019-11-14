/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/fc.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

inline void FCOutputSize(const framework::DDim& in_dims,
                         const framework::DDim& w_dims,
                         std::vector<int64_t>& out_dims,  // NOLINT
                         int in_num_col_dims, bool weight_pass) {
  auto in_mat_dims = framework::flatten_to_2d(in_dims, in_num_col_dims);
  if (weight_pass) {
    PADDLE_ENFORCE_EQ(
        in_mat_dims[1], w_dims[0] - 4,
        "Fully Connected input and weigth size do not match. %s, %s");
  } else {
    PADDLE_ENFORCE_EQ(
        in_mat_dims[1], w_dims[0],
        "Fully Connected input and weigth size do not match. %s, %s");
  }

  out_dims.reserve(static_cast<size_t>(in_num_col_dims + 1));
  for (int i = 0; i < in_num_col_dims; ++i) {
    out_dims.push_back(in_dims[i]);
  }
  if (weight_pass) {
    out_dims.push_back(w_dims[1] - 4);
  } else {
    out_dims.push_back(w_dims[1]);
  }
}

template <typename DeviceContext, typename T>
class FCOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<framework::LoDTensor>("Input");
    auto* w = ctx.Input<Tensor>("W");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* output = ctx.Output<framework::LoDTensor>("Out");
    int in_num_col_dims = ctx.Attr<int>("in_num_col_dims");
    bool with_relu =
        (ctx.Attr<std::string>("activation_type") == "relu") ? true : false;

    auto w_dims = w->dims();
    bool weight_pass = ctx.Attr<bool>("padding_weights");

    std::vector<int64_t> output_dims;
    FCOutputSize(input->dims(), w_dims, output_dims, in_num_col_dims,
                 weight_pass);
    output->Resize(framework::make_ddim(output_dims));
    output->set_lod(input->lod());

    auto out_dims = output->dims();
    int M = 0;
    if (weight_pass) {
      M = framework::product(out_dims) / (w_dims[1] - 4);
    } else {
      M = framework::product(out_dims) / w_dims[1];
    }

    const T* input_data = input->data<T>();
    const T* w_data = w->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    math::FCFunctor<DeviceContext, T> fc;
    if (weight_pass) {
      fc(dev_ctx, M, w_dims[1] - 4, w_dims[0] - 4, input_data, w_data,
         output_data, bias ? bias->data<T>() : NULL, with_relu, weight_pass);
    } else {
      fc(dev_ctx, M, w_dims[1], w_dims[0], input_data, w_data, output_data,
         bias ? bias->data<T>() : NULL, with_relu);
    }
  }
};

}  // namespace operators
}  // namespace paddle
