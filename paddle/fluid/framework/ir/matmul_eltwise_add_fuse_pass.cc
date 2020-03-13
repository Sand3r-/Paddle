// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/matmul_eltwise_add_fuse_pass.h"
#include <string>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

void MatmulEltwiseAddFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init("matmul_eltwise_add_fuse_pass", graph);
  // From
  //   [matmul]
  //      |
  // (matmul_out)  (residual_var)
  //      \          /
  //      [eltwise_add]
  //           |
  //      (eltwise_out)
  //           |
  //       [next_op]

  // To
  //        [matmul]
  //           |
  //     (residual_var)
  //           |
  //       [next_op]
  auto HasOneOutput = [](Node* x) { return x->outputs.size() == 1UL; };
  GraphPatternDetector detector;
  auto matmul =
      detector.mutable_pattern()->NewNode("matmul")->assert_is_op("matmul");
  auto matmul_out = detector.mutable_pattern()
                        ->NewNode("matmul_out")
                        ->assert_is_op_output("matmul")
                        ->assert_is_op_input("elementwise_add")
                        ->AsIntermediate();
  auto residual_var = detector.mutable_pattern()
                          ->NewNode("residual_var")
                          ->assert_is_op_input("elementwise_add")
                          // variable cannot be an input to any other op
                          ->assert_more(HasOneOutput);
  auto eltwise_add = detector.mutable_pattern()
                         ->NewNode("elementwise_add")
                         ->assert_is_op("elementwise_add");
  auto eltwise_out = detector.mutable_pattern()
                         ->NewNode("eltwise_out")
                         ->assert_is_op_output("elementwise_add")
                         ->AsIntermediate()
                         // eltwise output var should has only one consumer,
                         // otherwise it can't be removed.
                         ->assert_more(HasOneOutput);
  auto next_op = detector.mutable_pattern()->NewNode("next_op")->assert_is_op();

  matmul->LinksTo({matmul_out});
  eltwise_add->LinksFrom({matmul_out, residual_var}).LinksTo({eltwise_out});
  next_op->LinksFrom({eltwise_out});

  GraphPatternDetector::handle_t handler = [&](
      const GraphPatternDetector::subgraph_t& subgraph, Graph* graph) {
    Node* matmul_var = subgraph.at(matmul);
    Node* matmul_out_var = subgraph.at(matmul_out);
    Node* residual_var_var = subgraph.at(residual_var);
    Node* eltwise_add_var = subgraph.at(eltwise_add);
    Node* eltwise_out_var = subgraph.at(eltwise_out);
    Node* next_op_var = subgraph.at(next_op);

    const std::string residual_var_name = residual_var_var->Name();
    const std::string eltwise_out_name = eltwise_out_var->Name();

    // Remove links in graph
    GraphSafeRemoveNodes(graph,
                         {matmul_out_var, eltwise_add_var, eltwise_out_var});

    // Link matmul directly to residual_var
    // and residual var to next_op
    auto matmul_op_desc = matmul_var->Op();
    matmul_op_desc->SetAttr("beta", 1.0f);
    matmul_op_desc->SetOutput("Out", {residual_var_name});
    auto next_op_desc = next_op_var->Op();
    next_op_desc->RenameInput(eltwise_out_name, residual_var_name);

    IR_NODE_LINK_TO(matmul_var, residual_var_var);
    IR_NODE_LINK_TO(residual_var_var, next_op_var);
  };

  detector(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(matmul_eltwise_add_fuse_pass,
              paddle::framework::ir::MatmulEltwiseAddFusePass);
