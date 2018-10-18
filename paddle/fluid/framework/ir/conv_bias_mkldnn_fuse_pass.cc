// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/ir/conv_bias_mkldnn_fuse_pass.h"
#include <functional>
#include <string>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> ConvBiasFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init(name_scope_, graph.get());

  auto* scope = param_scope();
  PADDLE_ENFORCE(scope);

  GraphPatternDetector gpd;
  auto* conv_input =
      gpd.mutable_pattern()
          ->NewNode(patterns::PDNodeName(name_scope_, "conv_input"))
          ->AsInput()
          ->assert_is_op_input("conv2d", "Input");
  patterns::ConvBias conv_bias_pattern(gpd.mutable_pattern(), name_scope_);
  conv_bias_pattern(conv_input);
  int found_conv_bias_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle ConvBias fuse";
    GET_IR_NODE_FROM_SUBGRAPH(conv_weight, conv_weight,
                              conv_bias_pattern);                      // Filter
    GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out, conv_bias_pattern);  // tmp
    GET_IR_NODE_FROM_SUBGRAPH(conv, conv, conv_bias_pattern);  // CONV op
    // bias
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_bias, eltwise_bias, conv_bias_pattern);
    // output
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_out, eltwise_out, conv_bias_pattern);
    // elementwise_add op
    GET_IR_NODE_FROM_SUBGRAPH(eltwise, eltwise, conv_bias_pattern);

    PADDLE_ENFORCE(subgraph.count(conv_input));

    auto* eltwise_bias_tensor =
        scope->FindVar(eltwise_bias->Name())->GetMutable<LoDTensor>();

    auto input_names = conv->Op()->InputNames();
    bool has_bias = std::find(input_names.begin(), input_names.end(), "Bias") !=
                    input_names.end();
    if (has_bias && conv->Op()->Input("Bias").size() > 0) {
      auto conv_bias_names = conv->Op()->Input("Bias");
      // add eltwise bias to existing conv bias
      PADDLE_ENFORCE_EQ(conv_bias_names.size(), 1);
      auto* conv_bias_var = scope->FindVar(conv_bias_names[0]);
      auto* conv_bias_tensor = conv_bias_var->GetMutable<LoDTensor>();
      PADDLE_ENFORCE_EQ(conv_bias_tensor->dims(), eltwise_bias_tensor->dims());

      auto eigen_conv_bias = EigenVector<float>::From(*conv_bias_tensor);
      eigen_conv_bias += EigenVector<float>::From(*eltwise_bias_tensor);
    } else {
      // take eltwise bias as conv bias
      conv->Op()->SetInput("Bias",
                           std::vector<std::string>({eltwise_bias->Name()}));
      IR_NODE_LINK_TO(eltwise_bias, conv);
    }

    conv->Op()->SetOutput("Output",
                          std::vector<std::string>({eltwise_out->Name()}));

    GraphSafeRemoveNodes(graph.get(), {eltwise, conv_out});

    IR_NODE_LINK_TO(conv, eltwise_out);
    found_conv_bias_count++;
  };
  gpd(graph.get(), handler);
  AddStatis(found_conv_bias_count);
  return graph;
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle
REGISTER_PASS(conv_bias_mkldnn_fuse_pass,
              paddle::framework::ir::ConvBiasFusePass);
