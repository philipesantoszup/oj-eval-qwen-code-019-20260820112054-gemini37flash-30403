#pragma once
#include "simulator.hpp"

namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  
  Matrix *K_T = nullptr;
  Matrix *V_block = nullptr;

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    
    // Move Q to shared mem
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Move current key and value to SRAM
    gpu_sim.MoveMatrixToSharedMem(keys[i]);
    gpu_sim.MoveMatrixToSharedMem(values[i]);

    // Transpose keys[i] to (512, 1)
    gpu_sim.Transpose(keys[i], Position::kInSharedMemory);

    if (i == 0) {
      K_T = matrix_memory_allocator.Allocate("K_T_0");
      gpu_sim.Copy(keys[0], K_T, Position::kInSharedMemory);

      V_block = matrix_memory_allocator.Allocate("V_block_0");
      gpu_sim.Copy(values[0], V_block, Position::kInSharedMemory);
    } else {
      Matrix *new_k = matrix_memory_allocator.Allocate("K_T_" + std::to_string(i));
      gpu_sim.Concat(K_T, keys[i], new_k, 1, Position::kInSharedMemory);
      gpu_sim.ReleaseMatrix(K_T);
      K_T = new_k;

      Matrix *new_v = matrix_memory_allocator.Allocate("V_block_" + std::to_string(i));
      gpu_sim.Concat(V_block, values[i], new_v, 0, Position::kInSharedMemory);
      gpu_sim.ReleaseMatrix(V_block);
      V_block = new_v;
    }

    gpu_sim.ReleaseMatrix(keys[i]);
    gpu_sim.ReleaseMatrix(values[i]);

    // QK = Q * K_T via outer product of 512 column-row pairs
    // Q is (i+1, 512), K_T is (512, i+1).
    Matrix *QK = nullptr;
    for (size_t c = 0; c < 512; ++c) {
      Matrix *q_col = matrix_memory_allocator.Allocate("q_col_" + std::to_string(c));
      gpu_sim.GetColumn(current_query, c, q_col, Position::kInSharedMemory);

      Matrix *k_row = matrix_memory_allocator.Allocate("k_row_" + std::to_string(c));
      gpu_sim.GetRow(K_T, c, k_row, Position::kInSharedMemory);

      Matrix *prod = matrix_memory_allocator.Allocate("prod_" + std::to_string(c));
      gpu_sim.MatMul(q_col, k_row, prod);
      gpu_sim.ReleaseMatrix(q_col);
      gpu_sim.ReleaseMatrix(k_row);

      if (c == 0) {
        QK = prod;
      } else {
        Matrix *new_qk = matrix_memory_allocator.Allocate("qk_accum_" + std::to_string(c));
        gpu_sim.MatAdd(QK, prod, new_qk);
        gpu_sim.ReleaseMatrix(QK);
        gpu_sim.ReleaseMatrix(prod);
        QK = new_qk;
      }
    }
    // We are done with current_query (Q)
    gpu_sim.ReleaseMatrix(current_query);

    // Softmax row by row on QK:
    Matrix *Softmax_QK = nullptr;
    for (size_t r = 0; r <= i; ++r) {
      Matrix *row = matrix_memory_allocator.Allocate("row_" + std::to_string(r));
      gpu_sim.GetRow(QK, r, row, Position::kInSharedMemory);

      Matrix *exp_row = matrix_memory_allocator.Allocate("exp_row_" + std::to_string(r));
      gpu_sim.MatExp(row, exp_row);
      gpu_sim.ReleaseMatrix(row);

      Matrix *sum_val = matrix_memory_allocator.Allocate("sum_" + std::to_string(r));
      gpu_sim.Sum(exp_row, sum_val);

      Matrix *softmax_row = matrix_memory_allocator.Allocate("sm_row_" + std::to_string(r));
      gpu_sim.MatDiv(exp_row, sum_val, softmax_row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(sum_val);

      if (r == 0) {
        Softmax_QK = softmax_row;
      } else {
        Matrix *new_sm = matrix_memory_allocator.Allocate("sm_cat_" + std::to_string(r));
        gpu_sim.Concat(Softmax_QK, softmax_row, new_sm, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(Softmax_QK);
        gpu_sim.ReleaseMatrix(softmax_row);
        Softmax_QK = new_sm;
      }
    }
    gpu_sim.ReleaseMatrix(QK);

    // Softmax_QK is (i+1, i+1), V_block is (i+1, 512).
    // Outer product for Answer = Softmax_QK * V_block:
    Matrix *Answer = nullptr;
    for (size_t c = 0; c <= i; ++c) {
      Matrix *s_col = matrix_memory_allocator.Allocate("s_col_" + std::to_string(c));
      gpu_sim.GetColumn(Softmax_QK, c, s_col, Position::kInSharedMemory);

      Matrix *v_row = matrix_memory_allocator.Allocate("v_row_" + std::to_string(c));
      gpu_sim.GetRow(V_block, c, v_row, Position::kInSharedMemory);

      Matrix *prod = matrix_memory_allocator.Allocate("ans_prod_" + std::to_string(c));
      gpu_sim.MatMul(s_col, v_row, prod);
      gpu_sim.ReleaseMatrix(s_col);
      gpu_sim.ReleaseMatrix(v_row);

      if (c == 0) {
        Answer = prod;
      } else {
        Matrix *new_ans = matrix_memory_allocator.Allocate("ans_accum_" + std::to_string(c));
        gpu_sim.MatAdd(Answer, prod, new_ans);
        gpu_sim.ReleaseMatrix(Answer);
        gpu_sim.ReleaseMatrix(prod);
        Answer = new_ans;
      }
    }
    gpu_sim.ReleaseMatrix(Softmax_QK);

    // Move Answer to HBM
    gpu_sim.MoveMatrixToGpuHbm(Answer);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*Answer);
  }

  // Release persistent blocks at the end
  if (K_T) gpu_sim.ReleaseMatrix(K_T);
  if (V_block) gpu_sim.ReleaseMatrix(V_block);
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu