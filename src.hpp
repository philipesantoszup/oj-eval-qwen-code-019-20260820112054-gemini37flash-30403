#pragma once
#include "simulator.hpp"

namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  
  Matrix *K_T_hbm = nullptr;
  Matrix *V_block_hbm = nullptr;

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    // current_query is in HBM: shape (i+1, 512)
    // keys[i], values[i] are in HBM: shape (1, 512)

    // Transpose keys[i] in HBM -> shape (512, 1)
    gpu_sim.Transpose(keys[i], Position::kInGpuHbm);

    if (i == 0) {
      K_T_hbm = matrix_memory_allocator.Allocate("K_T_hbm_0");
      gpu_sim.Copy(keys[0], K_T_hbm, Position::kInGpuHbm);

      V_block_hbm = matrix_memory_allocator.Allocate("V_block_hbm_0");
      gpu_sim.Copy(values[0], V_block_hbm, Position::kInGpuHbm);
    } else {
      Matrix *new_k = matrix_memory_allocator.Allocate("K_T_hbm_" + std::to_string(i));
      gpu_sim.Concat(K_T_hbm, keys[i], new_k, 1, Position::kInGpuHbm);
      gpu_sim.ReleaseMatrix(K_T_hbm);
      K_T_hbm = new_k;

      Matrix *new_v = matrix_memory_allocator.Allocate("V_block_hbm_" + std::to_string(i));
      gpu_sim.Concat(V_block_hbm, values[i], new_v, 0, Position::kInGpuHbm);
      gpu_sim.ReleaseMatrix(V_block_hbm);
      V_block_hbm = new_v;
    }

    gpu_sim.ReleaseMatrix(keys[i]);
    gpu_sim.ReleaseMatrix(values[i]);

    // Process Q row by row
    Matrix *Answer_hbm = nullptr;
    for (size_t r = 0; r <= i; ++r) {
      Matrix *q_row = matrix_memory_allocator.Allocate("q_row_" + std::to_string(r));
      gpu_sim.GetRow(current_query, r, q_row, Position::kInGpuHbm);
      gpu_sim.MoveMatrixToSharedMem(q_row);

      // Compute qk_row = q_row * K_T_hbm: shape (1, i+1)
      // via outer product over 512 elements:
      Matrix *qk_row = nullptr;
      for (size_t c = 0; c < 512; ++c) {
        Matrix *q_val = matrix_memory_allocator.Allocate("q_val_" + std::to_string(c));
        gpu_sim.GetColumn(q_row, c, q_val, Position::kInSharedMemory);

        Matrix *k_row_c = matrix_memory_allocator.Allocate("k_row_c_" + std::to_string(c));
        gpu_sim.GetRow(K_T_hbm, c, k_row_c, Position::kInGpuHbm);
        gpu_sim.MoveMatrixToSharedMem(k_row_c);

        Matrix *prod = matrix_memory_allocator.Allocate("qk_prod_" + std::to_string(c));
        gpu_sim.MatMul(q_val, k_row_c, prod);
        gpu_sim.ReleaseMatrix(q_val);
        gpu_sim.ReleaseMatrix(k_row_c);

        if (c == 0) {
          qk_row = prod;
        } else {
          Matrix *new_qk = matrix_memory_allocator.Allocate("qk_accum_" + std::to_string(c));
          gpu_sim.MatAdd(qk_row, prod, new_qk);
          gpu_sim.ReleaseMatrix(qk_row);
          gpu_sim.ReleaseMatrix(prod);
          qk_row = new_qk;
        }
      }
      gpu_sim.ReleaseMatrix(q_row);

      // Softmax on qk_row:
      Matrix *exp_row = matrix_memory_allocator.Allocate("exp_row_" + std::to_string(r));
      gpu_sim.MatExp(qk_row, exp_row);
      gpu_sim.ReleaseMatrix(qk_row);

      Matrix *sum_val = matrix_memory_allocator.Allocate("sum_" + std::to_string(r));
      gpu_sim.Sum(exp_row, sum_val);

      Matrix *softmax_row = matrix_memory_allocator.Allocate("sm_row_" + std::to_string(r));
      gpu_sim.MatDiv(exp_row, sum_val, softmax_row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(sum_val);

      // Compute ans_row = softmax_row * V_block:
      // Outer product over keys c in 0..i:
      Matrix *ans_row = nullptr;
      for (size_t c = 0; c <= i; ++c) {
        Matrix *sm_scalar = matrix_memory_allocator.Allocate("sm_scalar_" + std::to_string(c));
        gpu_sim.GetColumn(softmax_row, c, sm_scalar, Position::kInSharedMemory);

        Matrix *v_row = matrix_memory_allocator.Allocate("v_row_" + std::to_string(c));
        gpu_sim.GetRow(V_block_hbm, c, v_row, Position::kInGpuHbm);
        gpu_sim.MoveMatrixToSharedMem(v_row);

        Matrix *prod = matrix_memory_allocator.Allocate("row_prod_" + std::to_string(c));
        gpu_sim.MatMul(sm_scalar, v_row, prod);
        gpu_sim.ReleaseMatrix(sm_scalar);
        gpu_sim.ReleaseMatrix(v_row);

        if (c == 0) {
          ans_row = prod;
        } else {
          Matrix *new_ans = matrix_memory_allocator.Allocate("row_accum_" + std::to_string(c));
          gpu_sim.MatAdd(ans_row, prod, new_ans);
          gpu_sim.ReleaseMatrix(ans_row);
          gpu_sim.ReleaseMatrix(prod);
          ans_row = new_ans;
        }
      }
      gpu_sim.ReleaseMatrix(softmax_row);

      // Move ans_row to HBM
      gpu_sim.MoveMatrixToGpuHbm(ans_row);

      if (r == 0) {
        Answer_hbm = ans_row;
      } else {
        Matrix *new_answer = matrix_memory_allocator.Allocate("ans_cat_" + std::to_string(r));
        gpu_sim.Concat(Answer_hbm, ans_row, new_answer, 0, Position::kInGpuHbm);
        gpu_sim.ReleaseMatrix(Answer_hbm);
        gpu_sim.ReleaseMatrix(ans_row);
        Answer_hbm = new_answer;
      }
    }
    gpu_sim.ReleaseMatrix(current_query);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*Answer_hbm);
  }

  // Release persistent blocks in HBM at the end
  if (K_T_hbm) gpu_sim.ReleaseMatrix(K_T_hbm);
  if (V_block_hbm) gpu_sim.ReleaseMatrix(V_block_hbm);
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu