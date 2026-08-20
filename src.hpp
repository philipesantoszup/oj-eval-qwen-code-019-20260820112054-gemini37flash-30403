#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    
    // Move Q to shared mem
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Build K_accum and V_accum in SRAM
    // Or concatenate them
    // keys[0..i] are in HBM or SRAM.
    // For baseline, let's concatenate keys[0..i]
    Matrix *K_block = nullptr;
    Matrix *V_block = nullptr;

    for (size_t k = 0; k <= i; ++k) {
      gpu_sim.MoveMatrixToSharedMem(keys[k]);
      gpu_sim.MoveMatrixToSharedMem(values[k]);
      if (k == 0) {
        K_block = keys[0];
        V_block = values[0];
      } else {
        Matrix *new_k = matrix_memory_allocator.Allocate("K_cat_" + std::to_string(k));
        gpu_sim.Concat(K_block, keys[k], new_k, 0, Position::kInSharedMemory);
        if (k > 1) {
          gpu_sim.ReleaseMatrix(K_block);
        }
        K_block = new_k;

        Matrix *new_v = matrix_memory_allocator.Allocate("V_cat_" + std::to_string(k));
        gpu_sim.Concat(V_block, values[k], new_v, 0, Position::kInSharedMemory);
        if (k > 1) {
          gpu_sim.ReleaseMatrix(V_block);
        }
        V_block = new_v;
      }
    }

    // Now K_block is (i+1, 512), V_block is (i+1, 512)
    // Transpose K_block: (512, i+1)
    // Note: Transpose is in-place!
    // But wait! If K_block is keys[0] (when i==0), transposing keys[0] modifies keys[0] for future rounds!
    // To be safe, let's copy if i == 0 or transpose K_block copy
    Matrix *K_T = matrix_memory_allocator.Allocate("K_T");
    gpu_sim.Copy(K_block, K_T, Position::kInSharedMemory);
    gpu_sim.Transpose(K_T, Position::kInSharedMemory);

    // Q is (i+1, 512), K_T is (512, i+1)
    // QK = Q * K_T -> (i+1, i+1)
    Matrix *QK = matrix_memory_allocator.Allocate("QK");
    gpu_sim.MatMul(current_query, K_T, QK);
    gpu_sim.ReleaseMatrix(K_T);

    // Softmax row by row:
    // For each row r from 0 to i:
    // row = GetRow(QK, r)
    // exp_row = MatExp(row)
    // sum_val = Sum(exp_row)
    // softmax_row = MatDiv(exp_row, sum_val)
    // concat rows vertically to form Softmax_QK
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

    // Softmax_QK is (i+1, i+1), V_block is (i+1, 512)
    // Answer = Softmax_QK * V_block -> (i+1, 512)
    Matrix *Answer = matrix_memory_allocator.Allocate("Answer_" + std::to_string(i));
    gpu_sim.MatMul(Softmax_QK, V_block, Answer);
    gpu_sim.ReleaseMatrix(Softmax_QK);

    if (i > 0) {
      gpu_sim.ReleaseMatrix(K_block);
      gpu_sim.ReleaseMatrix(V_block);
    }
    gpu_sim.ReleaseMatrix(current_query);

    // Move Answer to HBM
    gpu_sim.MoveMatrixToGpuHbm(Answer);

    // Move keys and values back to HBM so next round has them in HBM? Or move them back
    for (size_t k = 0; k <= i; ++k) {
      gpu_sim.MoveMatrixToGpuHbm(keys[k]);
      gpu_sim.MoveMatrixToGpuHbm(values[k]);
    }

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*Answer);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu