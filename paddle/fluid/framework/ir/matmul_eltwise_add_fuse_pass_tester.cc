/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/matmul_eltwise_add_fuse_pass.h"

#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

TEST(MatmulElementwiseAddFusePass, applicable_fuse) {
  Layers layers;
  // Note: Scale is used here as just an example of an
  // op that follows elementwise_add. It's mentioned
  // here, since its input has to be substituted with
  // z output of elementwise_add.
  // Before fuse:
  // (x, y) -> matmul -> (matmul_out)
  // (tmp_0, z) -> elementwise_add -> (eltwise_out)
  // (tmp_1) -> scale -> (scale_out)
  // After fuse:
  // (x, y) -> matmul -> (z)
  // (z) -> scale -> (tmp_2)
  // ----
  // Graph before fuse:
  //    (x)     (y)
  //       \    /
  //       [matmul]
  //          |          (z)
  //      (matmul_out)   /
  //          \         /
  //          [eltwise_add]
  //               |
  //          (eltwise_out)
  //               |
  //            [scale]
  // ----
  // Graph after fuse:
  //    (x)     (y)
  //       \    /
  //       [matmul]
  //          |
  //         (z)
  //          |
  //       [scale]
  auto* x = layers.data("x");
  auto* y = layers.data("y");
  auto* z = layers.data("z");
  auto* matmul_out = layers.matmul(x, y);
  auto* eltwise_out = layers.elementwise_add(matmul_out, z);
  layers.scale(eltwise_out, 1.0f, 0.0f, false);

  std::unique_ptr<Graph> graph(new Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("matmul_eltwise_add_fuse_pass");
  VLOG(3) << DebugString(graph);

  graph->Set("__param_scope__", new Scope());
  graph.reset(pass->Apply(graph.release()));
  int num_matmul_nodes_after = GetNumOpNodes(graph, "matmul");
  int num_eltwise_nodes_after = GetNumOpNodes(graph, "elementwise_add");
  int num_scale_nodes_after = GetNumOpNodes(graph, "scale");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_matmul_nodes_after, 1);
  PADDLE_ENFORCE_EQ(num_eltwise_nodes_after, 0);
  PADDLE_ENFORCE_EQ(num_scale_nodes_after, 1);
}

TEST(MatmulElementwiseAddFusePass, fuse_with_stack_outputs) {
  Layers layers;
  // Again: Scale is used here as just an example of an
  // op that follows elementwise_add. This test checks if
  // stack output has been separated into several outputs
  // when its output is supposed to be used as matmul's
  // out.
  // Before fuse:
  // (x, y) -> matmul -> (matmul_out)
  // (tmp_0, z) -> elementwise_add -> (eltwise_out)
  // (tmp_1) -> scale -> (scale_out)
  // After fuse:
  // (x, y) -> matmul -> (z)
  // (z) -> scale -> (tmp_2)
  // ----
  // Graph before fuse:
  //    (x)     (y)      (z)
  //       \    /         |
  //       [matmul]    [stack]
  //          |           |
  //   (matmul_out)  (stack_0.tmp_0)
  //          \         /   \.
  //          [eltwise_add]  |
  //               |         |
  //          (eltwise_out)  |
  //               |         |
  //            [scale]     /
  //               |       /
  //          (scale_out) /
  //               |     /
  //         [eltwise_add]
  // ----
  // Graph after fuse:
  //    (x)     (y)
  //       \    /
  //       [matmul]  [stack]
  //          |       /  \.
  //          |      /    \.
  //           \    /      |
  //            \  /       |
  //  (stack_0.tmp_0)   (stack_0.tmp_1)
  //             |         /
  //          [scale]     /
  //             |       /
  //        (scale_out) /
  //             |     /
  //       [eltwise_add]
  auto* x = layers.data("x");
  auto* y = layers.data("y");
  auto* z = layers.data("z");
  auto* stack_out = layers.stack({z});
  auto* matmul_out = layers.matmul(x, y);
  auto* eltwise_out = layers.elementwise_add(matmul_out, stack_out);
  auto* scale_out = layers.scale(eltwise_out, 1.0f, 0.0f, false);
  layers.elementwise_add(scale_out, stack_out);

  std::unique_ptr<Graph> graph(new Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("matmul_eltwise_add_fuse_pass");
  VLOG(3) << DebugString(graph);

  graph->Set("__param_scope__", new Scope());
  graph.reset(pass->Apply(graph.release()));
  int num_matmul_nodes_after = GetNumOpNodes(graph, "matmul");
  int num_eltwise_nodes_after = GetNumOpNodes(graph, "elementwise_add");
  int num_scale_nodes_after = GetNumOpNodes(graph, "scale");
  VLOG(3) << DebugString(graph);

  for (auto& node : graph->Nodes()) {
    if (node->IsOp()) {
      if (node->Op()->Type() == "stack") {
        PADDLE_ENFORCE_EQ(node->outputs.size(), 2);
      }
    }
  }

  PADDLE_ENFORCE_EQ(num_matmul_nodes_after, 1);
  PADDLE_ENFORCE_EQ(num_eltwise_nodes_after, 1);
  PADDLE_ENFORCE_EQ(num_scale_nodes_after, 1);
}

TEST(MatmulElementwiseAddFusePass, shared_residual_var) {
  Layers layers;
  // This test checks if fuse isn't performed, when the
  // input that enters elementwise_add and is supposesd to
  // be placed in matmuls output is used as an input
  // in some other operator
  // ----
  // Graph:
  //    (x)     (y)
  //       \    /
  //       [matmul]
  //          |          (shared)
  //      (matmul_out)   /     |
  //          \         /      |
  //          [eltwise_add]   [relu]
  //               |
  //          (eltwise_out)
  //               |
  //            [scale]
  auto* x = layers.data("x");
  auto* y = layers.data("y");
  auto* shared = layers.data("shared");
  auto* matmul_out = layers.matmul(x, y);
  auto* eltwise_out = layers.elementwise_add(matmul_out, matmul_out);
  layers.scale(eltwise_out, 1.0f, 0.0f, false);
  layers.relu(shared);

  std::unique_ptr<Graph> graph(new Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("matmul_eltwise_add_fuse_pass");
  VLOG(3) << DebugString(graph);

  graph->Set("__param_scope__", new Scope());
  graph.reset(pass->Apply(graph.release()));
  int num_matmul_nodes_after = GetNumOpNodes(graph, "matmul");
  int num_eltwise_nodes_after = GetNumOpNodes(graph, "elementwise_add");
  int num_scale_nodes_after = GetNumOpNodes(graph, "scale");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_matmul_nodes_after, 1);
  PADDLE_ENFORCE_EQ(num_eltwise_nodes_after, 1);
  PADDLE_ENFORCE_EQ(num_scale_nodes_after, 1);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(matmul_eltwise_add_fuse_pass);
