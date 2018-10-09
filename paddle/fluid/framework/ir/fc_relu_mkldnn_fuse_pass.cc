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

#include "paddle/fluid/framework/ir/fc_relu_mkldnn_fuse_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> FCReluFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("fc_relu_mkldnn_fuse_pass", graph.get());

  auto* scope = param_scope();
  PADDLE_ENFORCE(scope);

  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode("fc_mkldnn_pass/x")
                ->AsInput()
                ->assert_is_op_input("fc", "Input");
  patterns::FCReLU fc_relu_pattern(gpd.mutable_pattern(),
                                   "fc_relu_mkldnn_fuse_pass");
  fc_relu_pattern(x, true /*with bias*/);

  int found_fc_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Handle FC MKL-DNN pass";
    GET_IR_NODE_FROM_SUBGRAPH(fc, fc, fc_relu_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_output, fc_output, fc_relu_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(relu, relu, fc_relu_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(relu_output, relu_output, fc_relu_pattern);

    OpDesc* desc = fc->Op();
    desc->SetOutput("Out", std::vector<std::string>({relu_output->Name()}));
    desc->SetAttr("fuse_relu", true);
    GraphSafeRemoveNodes(graph.get(), {relu, fc_output});

    PADDLE_ENFORCE(subgraph.count(x));
    IR_NODE_LINK_TO(fc, relu_output);

    found_fc_count++;
  };

  gpd(graph.get(), handler);

  AddStatis(found_fc_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_relu_mkldnn_fuse_pass, paddle::framework::ir::FCReluFusePass);
