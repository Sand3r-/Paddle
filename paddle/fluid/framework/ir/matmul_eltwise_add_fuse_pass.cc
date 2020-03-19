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
#include <vector>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

void MatmulEltwiseAddFusePass::ApplyImpl(ir::Graph* graph) const {
  const char* name_scope_ = "matmul_eltwise_add_fuse_pass";
  FusePassBase::Init(name_scope_, graph);

  auto* scope = param_scope();
  PADDLE_ENFORCE(scope);

  // Duplicate stack's output tensor to designate each
  // elementwise add a unique input from stack.
  // This is needed in the fuse pass that follows.
  // Example: transform from
  //         [op]      [stack]
  //          |           |
  //      (op_out)  (stack_0.tmp_0)
  //          \         /   \.
  //          [eltwise_add]  |
  //               |         |
  //          (eltwise_out)  |
  //               |         |
  //             [op2]      /
  //               |       /
  //           (op2_out)  /
  //               |     /
  //         [eltwise_add]
  // ----
  // To
  //                            [stack]
  //                           /     \.
  //                          /       |
  //         [op]            /        |
  //          |             /         |
  //      (op_out)  (stack_0.tmp_0)   |
  //          \         /             /
  //          [eltwise_add]     _____/
  //               |           |
  //          (eltwise_out)    |
  //               |           |
  //             [op2]         |
  //               |      (stack_0.tmp_1)
  //           (op2_out)  /
  //               |     /
  //         [eltwise_add]

  GraphPatternDetector stack_detector;
  auto stack =
      stack_detector.mutable_pattern()->NewNode("stack")->assert_is_op("stack");
  auto stack_out = stack_detector.mutable_pattern()
                       ->NewNode("stack_out")
                       ->assert_is_op_output("stack")
                       ->assert_is_op_input("elementwise_add");

  stack->LinksTo({stack_out});

  GraphPatternDetector::handle_t stack_handler = [&](
      const GraphPatternDetector::subgraph_t& subgraph, Graph* graph) {
    Node* stack_var = subgraph.at(stack);
    Node* stack_out_var = subgraph.at(stack_out);

    size_t num_stack_outputs = stack_out_var->outputs.size();
    std::vector<std::string> out_var_names = stack_var->Op()->Output("Y");
    std::string cur_stack_out_name = out_var_names[0];
    // Iterator from 1, to preserve initial output variable
    for (size_t i = 1; i < num_stack_outputs; i++) {
      VarDesc* proto_var = stack_out_var->Var();
      VarDesc out_var_desc(
          patterns::PDNodeName(name_scope_, stack_out_var->Name()));
      std::string name = out_var_desc.Name();
      // Copy Variable properties
      out_var_desc.SetShape(proto_var->GetShape());
      out_var_desc.SetDataType(proto_var->GetDataType());
      out_var_desc.SetLoDLevel(proto_var->GetLoDLevel());
      // Create node in graph
      Node* new_output = graph->CreateVarNode(&out_var_desc);
      // Create variable in scope
      scope->Var(name)->GetMutable<LoDTensor>();

      // Rename input of eltwise add from stack.tmp_x to a newly created
      // variable name
      stack_out_var->outputs[i]->Op()->RenameInput(cur_stack_out_name, name);

      // Remove the old name from elementwise_add inputs
      auto& eltwise_inputs = stack_out_var->outputs[i]->inputs;
      std::remove_if(eltwise_inputs.begin(), eltwise_inputs.end(),
                     [&cur_stack_out_name](Node* n) {
                       return n->Name() == cur_stack_out_name;
                     });

      // Link nodes in a graph
      IR_NODE_LINK_TO(stack_var, new_output);
      IR_NODE_LINK_TO(new_output, stack_out_var->outputs[i]);

      // Append new variable to stack op's Out
      out_var_names.push_back(name);
    }
    // Remove all links but one from initial stack output variable
    stack_out_var->outputs.resize(1);
    // Update the var name list of Stack's op output names
    stack_var->Op()->SetOutput("Y", out_var_names);
  };

  stack_detector(graph, stack_handler);

  // Following fuse pass replaces old matmul's output
  // with the second input to elementwise_add and
  // remove elementwise_add.
  // Example: transform from
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
  GraphPatternDetector detector;
  auto HasOneOutput = [](Node* x) { return x->outputs.size() == 1UL; };

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
                          // since its contents will be overwritten
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
