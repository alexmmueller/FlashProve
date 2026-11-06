// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// CardFold is the GPU's part of FOLD (blocks/iop/fold.h), the sumcheck that folds the rows of a
// group. It computes the round polynomials for the group's main table m against a coefficient
// vector c, all from one pass over m, then the folded row and, on request, the folded c. (Several
// rounds from one pass over the data: Gruen, "Some Improvements for the PIOP for ZeroCheck",
// ePrint 2024/108, section 5.) CardFold's values are unmasked: FoldProverRounds::round()
// (blocks/iop/fold_rounds.h) adds the hiding polynomial and the mask rows' part.
//
// The caller must ensure the following; nothing here can check it. The rows' cells and random tails
// (CardRows::tail) and the vectors that the constructor's `held` points to are complete for its
// stream before the call: written on that stream or waited for (prover/engine/arena.h explains
// which streams keep order). Their words are canonical. They stay in place, unchanged, until end()
// returns or the CardFold is destroyed. A vector that is too short, freed or not yet written is
// read as whatever is there. Nothing is cleared on release except cuBLAS's work space
// (prover/engine/products.cuh). The unmasked Gram matrix G (below) on the host and the GPU's other
// work spaces are released with their contents (prover/engine/pooled.cuh has the rule).
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "code/code.h"
#include "common/setting.h"  // one-off (not upstream): g_parallel_fold
#include "engine/device.cuh"
#include "engine/pooled.cuh"
#include "engine/encode.cuh" // Cell, cell_at
#include "engine/products.cuh"
#include "engine/table.h"
#include "engine/tiled.cuh"
#include "field/field.h"
#include "field/field_lazy.cuh"
#include "field/mle.h"
#include "iop/tensor.h"

namespace blk {

// What is computed. m and c have 2^k rows of ell entries each. Entry (row b, place p) has the index
// b + 2^k p: variables 0 .. k - 1 are row variables, the others place variables. For a table t,
// t<a>(X) is the line through its rows 2a (at X = 0) and 2a + 1 (at X = 1). A round's polynomial is
// q(X) = sum over a of < c<a>(X) , m<a>(X) >; a challenge gamma then replaces each pair of rows by
// t<a>(gamma), in m and in c.
// F: the base field. E: its cubic extension. Word: 64 bits, one element of F. Group: the oracle
// messages that one proximity test reads; row b of m is the message of its main row b. Cell: one
// coordinate of an entry of m, in one of the formats of `Cell` (prover/engine/encode.cuh).
//
// How. With the Gram matrix G[b][b'] = < row b of m , row b' of c >, 2^k x 2^k elements of E,
//   q(X) = sum_a (1-X)^2 G[2a][2a] + X (1-X) (G[2a][2a+1] + G[2a+1][2a]) + X^2 G[2a+1][2a+1],
// and binding a variable folds G in both indices. So the GPU makes G in one pass over the cells,
// the host computes the rounds from G, and when all challenges are known the GPU reads the cells
// again for the folded row, sum_b eq(b, challenges) row b. The folded coefficient vector is
// the same sum over the rows of c. Neither m nor c is stored partly folded.

// The main rows of one oracle message: a block of neighbouring rows of the group. A row's random
// tail ("encoding randomness" in an error message) is the r random entries that are encoded with
// its message (blocks/code/code.h).
// "Card" or "card" in a name: the GPU.
struct CardRows {
  const void *cells = nullptr; // on the GPU: cell number (row * ell + place) * deg + coordinate
  Cell cell = Cell::WORD;      // how a cell is stored in memory
  unsigned deg = 1;            // 1: entries in F; 3: entries in E
  size_t rows = 0;             // so many neighbouring rows of the group
  u64 first_row = 0;           // the first of them; rows of the group in no CardRows are zero
  const u64 *tail = nullptr;   // on the GPU, the tails: word (row * r + place) * deg + coordinate
};

// ---- what the kernels read ----
// How G is made. c is a Form (blocks/iop/tensor.h): a sum of Tensors, single entries and engine's
// terms. A Tensor is v[j] = scalar * prod_t (bit t of j ? hi[t] : lo[t]), times an entry of a
// written-out factor (Tensor::values; tensor.h's dense factor, `dense` in identifiers) if it has
// one. Variable t picks a bit if one of lo[t], hi[t] is 0, and is open if neither is. The engine is
// CardFold's caller, `EngineGpu` (prover/engine/engine_gpu.cuh). An engine's term is a scaled
// vector on the GPU; a hint states that some Tensors and entries add up to such a vector. c is
// looked at one slab at a time: 2^16 places at most, or the entire row if shorter. Each part of c
// goes one of three ways.
//   PIECES. On its lowest w place variables only, a Tensor is a vector V of 2^w entries ("form" in
//     names): a product form, a slice of its written-out factor, or both. On every aligned piece of
//     2^w places of every row it is a coefficient times such a V. All rows of m are multiplied with
//     V on each piece of the slabs that need it (prover/engine/tiled.cuh), once for all Tensors
//     that share V; then G[b][b'] += coefficient(b', piece) * < row b on the piece , V >. The GPU
//     writes out the coefficients from a short description of the Tensor, its cut (FoldCut).
//   WRITTEN OUT. If at least two V's serve one row of c only, their Tensors are summed into that
//     row of c, which is then multiplied with all rows of m like one V. The vector of a hint or of
//     an engine's term goes the same way. So do the product forms over F on all places of a row
//     that has many of them (CardFold::MANY_TERMS), even if other rows share their V's: their sum
//     is made beforehand for the entire row (k_fold_groups() if their high factors repeat, else
//     k_fold_two_matrices(); conditions in make_gram()) and counts as a held vector (FoldHeld).
//   CELLS. G[b][b'] += coefficient * cell, for single entries, for Tensors with fewer nonzero
//     entries than the smallest V (FoldSetting::least_w), and for all Tensors where a slab is
//     shorter than that.
// A piece is multiplied in chunks (FoldCut::chunk); "the sums" are the chunks' inner products. The
// rows of c that are not zero are numbered by slots, its written-out rows by columns. The GPU works
// on a span at a time: as many whole slabs as FoldSetting::span_bytes and most_log_span allow.
constexpr u8 NO_SHIFT = 255;
constexpr unsigned FOLD_MOST_W = 16; // a V has at most so many variables
constexpr unsigned FOLD_MOST_LOG_SLAB = 16;
static_assert(FOLD_MOST_LOG_SLAB <= FOLD_MOST_W, "a V is at most as long as a slab, and FoldVector::shift has a place for each of its variables");
static_assert(FOLD_MOST_LOG_SLAB <= CardTiled::MOST_LOG_K, "a chunk is at most as long as a slab");
// PIECES: a V of 2^w entries. Variable t < w either carries a pair of factors
// (shift[t] == NO_SHIFT) or is bit shift[t] of an index d into the written-out factor:
// entry i = (product over the former t of factors[2 (factors_at + t) + bit t of i])
//           * (values[values_at + d] if has_values).
struct FoldVector {
  u32 factors_at, w;
  u64 at; // coordinate c of entry i is word at + c 2^w + i of the store
  u64 values_at;
  u8 shift[16];
  bool has_values;
  bool has_factors; // some variable carries factors other than (1, 1)
};
// A product form kept as two tables: entry i = low[i mod 2^w_low] * high[i div 2^w_low].
// Generator row: the row of the code's generator matrix at a position (blocks/code/code.h).
struct FoldSplit {
  u32 factors_at, w, w_low;
  u64 low_at, high_at; // in the tables over E, or in those over F if in_base
  bool in_base;        // all factors are in F (a generator row's are): cheaper tables and products
};
// WRITTEN OUT: weight * (a split form) on the places piece * 2^w .. (piece + 1) * 2^w - 1 of one
// written-out row.
struct FoldTerm {
  u32 split, w;
  u64 piece;
  u32 column; // which written-out row
  E weight;
};
// WRITTEN OUT: scale * (ell elements of E out of a vector on the GPU), on one written-out row.
struct FoldHeld {
  const u64 *vector;
  u32 column;
  E scale;
  u64 step;          // not 0: stored by coordinates: coordinate d of entry j is word d * step + j
  u64 first, stride; // step 0: entry j is the vector's element first + j * stride, three words
};
__device__ __forceinline__ E fold_held_at(const FoldHeld &h, u64 j) {
  if (h.step) return E::from(F{h.vector[j]}, F{h.vector[h.step + j]}, F{h.vector[2 * h.step + j]});
  const u64 *at = h.vector + 3 * (h.first + j * h.stride);
  return E::from(F{at[0]}, F{at[1]}, F{at[2]});
}
// gram[row][slot] += coefficient * x, where x is the element of E at the words
// at + row * row_step .. of the sums or, for CELLS, the row's cell at place `at`.
struct FoldItem {
  u64 at;         // (in the folded coefficient vector: the number of a form)
  u32 row_step, live; // live = 0: nothing is read for the item
  E coefficient;
};

// A Tensor of the ways PIECES and CELLS, as the GPU needs it to write out the coefficients. Outside
// V a variable either picks a bit or is open. The open variables are numbered; coefficient number
// e < 2^open has bit i of e at open variable i. The first `joined` open variables are row variables
// that change neither piece nor V. A written-out factor ("the factor" below) enters in one of two
// ways. many_forms: V includes some of its variables, and there is a V for each value of the
// others. in_coefficient: V includes none, and the factor's entry goes into the coefficient.
constexpr unsigned FOLD_MOST_OPEN = 40;
struct FoldCut {
  E scalar;       // the Tensor's, times the factors of the variables outside V that pick a bit
  u64 index;      // the bits that these pick
  u64 values_at;  // in_coefficient: where the factor's slice begins in FoldCuts::values
  u32 factors_at; // open variable i's pair: FoldCuts::factors[2 (factors_at + i)] and the next
  u32 forms_at;   // its V: FoldCuts::forms[forms_at + (many_forms ? FoldWalk::together : 0)]
  u32 begin_at;   // folded coefficient vector: where pieces of 2^w places begin in Cuts::d_begin
  u8 open, joined; // the numbers of open variables and, among them, of joined ones
  u8 w, chunk;     // V has 2^w entries (0: no V, CELLS), multiplied on chunks of 2^chunk places
  bool joined_in_factor;           // some joined variable indexes the factor
  bool many_forms, in_coefficient; // how the factor enters: see above
  u8 var[FOLD_MOST_OPEN];     // open variable i is this variable of the Tensor
  u8 d_shift[FOLD_MOST_OPEN]; // NO_SHIFT, or this bit of the factor's index (its pair is (1, 1)),
  u8 c_shift[FOLD_MOST_OPEN]; // and this one of the index's open bits pushed together
};
// fold_walk()'s state: index into c; index d into the written-out factor; d's open bits pushed
// together (FoldCut::c_shift); product of the factors.
struct FoldWalk {
  u64 index, d;
  u32 together;
  E x;
};
// Sets open variable i, for from <= i < to, to bit i - from of e.
__device__ __forceinline__ void fold_walk(FoldWalk &s, const FoldCut &c, const E *factors, u64 e, unsigned from, unsigned to) {
  for (unsigned i = from; i < to; i++) {
    const u64 bit = (e >> (i - from)) & 1;
    s.index |= bit << c.var[i];
    if (c.d_shift[i] != NO_SHIFT) s.d |= bit << c.d_shift[i], s.together |= (u32)bit << c.c_shift[i];
    else s.x = lazy::mul(s.x, factors[2 * (c.factors_at + i) + bit]);
  }
}
// Returns the last j < n with first[j] <= i. first is ascending; first[0] = 0.
__device__ __forceinline__ u32 fold_find(const u32 *first, u32 n, u32 i) {
  u32 low = 0, high = n;
  while (high - low > 1) {
    const u32 middle = (low + high) / 2;
    if (first[middle] <= i) low = middle;
    else high = middle;
  }
  return low;
}

// A lazy::AccE (prover/field/field_lazy.cuh) is a sum of products in E that is reduced at the end;
// it takes at most lazy::MAC_ROOM calls of mac().
//
// The threads of a block in every launch below. Several kernels rely on it; none checks it.
constexpr unsigned FOLD_THREADS = 256;

// Writes the V's that `vectors` describes into store, by coordinates (FoldVector::at). Block row y
// (blockIdx.y): vectors[y]. A block makes FOLD_RUNS runs of FOLD_THREADS entries. The entries of a
// run differ in their lowest 8 variables only: a thread computes these variables' product once
// (`low`), thread 0 the higher variables' once for each run (`high`).
constexpr unsigned FOLD_RUNS = 16;
// less: form v begins at word v.at - less of store.
static __global__ void k_fold_vectors(const FoldVector *vectors, const E *factors, const E *values, F *store, u64 less) {
  static_assert(FOLD_THREADS == 256, "the entries of a run differ in 8 variables");
  __shared__ E high;
  const FoldVector v = vectors[blockIdx.y];
  const u64 len = (u64)1 << v.w;
  E low = E::one();
  for (unsigned t = 0; t < 8 && t < v.w; t++)
    if (v.shift[t] == NO_SHIFT) low = lazy::mul(low, factors[2 * (v.factors_at + t) + ((threadIdx.x >> t) & 1)]);
  u64 d_low = 0; // the bits of the written-out factor's index that threadIdx.x sets
  for (unsigned t = 0; t < 8 && t < v.w; t++)
    if (v.shift[t] != NO_SHIFT) d_low |= (u64)((threadIdx.x >> t) & 1) << v.shift[t];
  for (unsigned run = 0; run < FOLD_RUNS; run++) {
    const u64 first = ((u64)blockIdx.x * FOLD_RUNS + run) * FOLD_THREADS, i = first + threadIdx.x;
    if (first >= len) return; // all threads alike
    if (v.has_factors) {
      __syncthreads();
      if (threadIdx.x == 0) {
        E x = E::one();
        for (unsigned t = 8; t < v.w; t++)
          if (v.shift[t] == NO_SHIFT) x = lazy::mul(x, factors[2 * (v.factors_at + t) + ((first >> t) & 1)]);
        high = x;
      }
      __syncthreads();
    }
    if (i >= len) continue;
    E x = v.has_factors ? lazy::mul(high, low) : E::one();
    if (v.has_values) {
      u64 d = d_low;
      for (unsigned t = 8; t < v.w; t++)
        if (v.shift[t] != NO_SHIFT) d |= ((first >> t) & 1) << v.shift[t];
      x = v.has_factors ? lazy::mul(x, values[v.values_at + d]) : values[v.values_at + d];
    }
    for (unsigned c = 0; c < 3; c++) store[v.at - less + c * len + i] = x.c[c];
  }
}
// Writes the two tables of each split form in `splits` (FoldSplit). Block row y: splits[y].
static __global__ void k_fold_splits(const FoldSplit *splits, const E *factors, E *tables, F *base_tables) {
  const FoldSplit v = splits[blockIdx.y];
  const u64 n_low = (u64)1 << v.w_low, n_high = (u64)1 << (v.w - v.w_low);
  const u64 e = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_low + n_high) return;
  const bool high = e >= n_low;
  const u64 index = high ? e - n_low : e;
  const unsigned from = high ? v.w_low : 0, to = high ? v.w : v.w_low;
  E x = E::one();
  for (unsigned t = from; t < to; t++) x = x * factors[2 * (v.factors_at + t) + ((index >> (t - from)) & 1)];
  if (v.in_base) base_tables[(high ? v.high_at : v.low_at) + index] = x.c[0];
  else tables[(high ? v.high_at : v.low_at) + index] = x;
}
// Writes the written-out rows of c on the places first .. first + width - 1 (a span) to
// out[(column * 3 + coordinate) * width + (place - first)], which must be zero before. width is a
// power of two and first a multiple of it. list: the terms that meet these places, sorted by
// column: a thread has one place, sums a column's terms and writes the sum once. Where a low table
// has at least FOLD_THREADS entries, all places of a block meet one entry of the high table, so
// weight * high is computed once for the block (Ready::scale). A form with a shorter low table is
// a short form.
static __global__ void k_fold_write_out(F *out, u64 first, u64 width, const u32 *list, u32 list_n, const FoldTerm *terms, const FoldSplit *splits,
                                        const E *tables, const F *base_tables, const FoldHeld *held, u32 held_n) {
  struct Ready {
    E scale;      // weight * high; for a short form: the weight
    u64 low_at;   // the low table
    u32 low_mask; // its length less 1
    u32 column;
    u32 term;     // in 'terms'
    u8 kind;      // 0: block outside the term's piece; 1, 2: low table over F, E; 3: short form
  };
  __shared__ Ready ready[FOLD_THREADS];
  const u64 p = (u64)blockIdx.x * blockDim.x + threadIdx.x, block_place = first + (u64)blockIdx.x * blockDim.x;
  const u64 place = first + p;
  const bool mine = p < width;
  lazy::AccE sum = lazy::zero_e(); // fewer terms than lazy::MAC_ROOM: `make_gram` checks it
  for (u32 from = 0; from < list_n; from += FOLD_THREADS) {
    const u32 n = list_n - from < FOLD_THREADS ? list_n - from : FOLD_THREADS;
    __syncthreads();
    if (threadIdx.x < n) {
      Ready r;
      r.term = list[from + threadIdx.x];
      const FoldTerm t = terms[r.term];
      const FoldSplit v = splits[t.split];
      r.column = t.column, r.low_at = v.low_at, r.low_mask = (u32)(((u64)1 << v.w_low) - 1), r.scale = t.weight;
      if (((u64)1 << v.w_low) < FOLD_THREADS) r.kind = 3;
      else if ((block_place >> t.w) != t.piece) r.kind = 0;
      else {
        const u64 high = v.high_at + ((block_place & (((u64)1 << t.w) - 1)) >> v.w_low);
        r.kind = v.in_base ? 1 : 2;
        r.scale = v.in_base ? t.weight * base_tables[high] : t.weight * tables[high];
      }
      ready[threadIdx.x] = r;
    }
    __syncthreads();
    for (u32 i = 0; i < n; i++) {
      const Ready &r = ready[i];
      if (mine) {
        if (r.kind == 1) lazy::mac(sum, r.scale, base_tables[r.low_at + (place & r.low_mask)]);
        else if (r.kind == 2) lazy::mac(sum, r.scale, tables[r.low_at + (place & r.low_mask)]);
        else if (r.kind == 3) {
          const FoldTerm t = terms[r.term];
          if ((place >> t.w) == t.piece) {
            const FoldSplit v = splits[t.split];
            const u64 in = place & (((u64)1 << t.w) - 1), low = v.low_at + (in & r.low_mask), high = v.high_at + (in >> v.w_low);
            if (v.in_base) lazy::mac(sum, t.weight, base_tables[low] * base_tables[high]);
            else lazy::mac(sum, t.weight, tables[low] * tables[high]);
          }
        }
      }
      const bool last = from + i + 1 == list_n || (i + 1 < n ? ready[i + 1].column : terms[list[from + i + 1]].column) != r.column;
      if (last) {
        if (mine) {
          const E x = lazy::reduce(sum);
          F *at = out + (u64)r.column * 3 * width + p;
          at[0] = x.c[0], at[width] = x.c[1], at[2 * width] = x.c[2];
        }
        sum = lazy::zero_e();
      }
    }
  }
  if (!mine) return;
  for (u32 i = 0; i < held_n; i++) {
    const FoldHeld h = held[i];
    const E y = h.scale * fold_held_at(h, place);
    F *at = out + (u64)h.column * 3 * width + p;
    at[0] += y.c[0], at[width] += y.c[1], at[2 * width] += y.c[2];
  }
}
// For many product forms over F on all places of one row of c, the sum over i of
// weight_i low_i[j mod 2^w_low] high_i[j div 2^w_low] is a product of two matrices. Writes the two:
// left[(d * n_high + h) * room + i] = coordinate d of weight_i high_i[h], and
// right[l * room + i] = low_i[l]; both are zero for i >= count.
static __global__ void k_fold_two_matrices(F *left, F *right, u64 n_high, u64 n_low, u64 room, const u32 *list, u32 count, const FoldTerm *terms, const FoldSplit *splits,
                                           const F *base_tables) {
  const u64 e = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= (n_high + n_low) * room) return;
  const u64 i = e % room, at = e / room;
  const bool live = i < count;
  const FoldTerm t = terms[list[live ? i : 0]];
  const FoldSplit v = splits[t.split];
  if (at < n_high) {
    const E x = live ? t.weight * base_tables[v.high_at + at] : E::zero();
    for (u64 d = 0; d < 3; d++) left[(d * n_high + at) * room + i] = x.c[d];
  } else {
    right[(at - n_high) * room + i] = live ? base_tables[v.low_at + (at - n_high)] : F::zero();
  }
}
// Sums many product forms over F whose high variables have few distinct factors, as those of a
// code's generator rows do (factor t at a position depends on few bits of the position). This is
// the transpose of the code's fast transform, applied to a vector with few entries and pruned
// accordingly.
// Public source: transposition of a transform: A. Bostan, G. Lecerf, E. Schost, "Tellegen's
// principle into practice", ISSAC 2003.
// Public source: few non-zero inputs or few wanted outputs: FFTW, "Pruned FFTs",
// www.fftw.org/pruned.html.
// Public source: a transform's layers: the computation of the layers (k_fold_layers()) follows
// S. Kim, W. Jung, J. Park, J. H. Ahn, "Accelerating Number Theoretic Transformations for
// Bootstrappable Homomorphic Encryption on GPUs", IISWC 2020, arXiv 2012.01968, sections V, VI.
// Terms that agree in their d highest factors form a group (Layers::d); plan_layers() finds the
// groups from the factors alone. k_fold_groups() sums each group on the low variables only;
// k_fold_layers() then applies one layer for each variable above, which joins the groups that
// agree in the factors above that variable.
// out[(group << log_len) + j] = sum over the group's terms of weight * (the form's entry j), for
// j < 2^log_len. Block row y: group y.
static __global__ void k_fold_groups(E *out, unsigned log_len, const u32 *begin, const u32 *list, const FoldTerm *terms, const FoldSplit *splits, const F *base_tables) {
  const u64 j = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= ((u64)1 << log_len)) return;
  lazy::AccE sum = lazy::zero_e();
  for (u32 i = begin[blockIdx.y]; i < begin[blockIdx.y + 1]; i++) {
    const FoldTerm t = terms[list[i]];
    const FoldSplit v = splits[t.split];
    lazy::mac(sum, t.weight, base_tables[v.low_at + (j & (((u64)1 << v.w_low) - 1))] * base_tables[v.high_at + (j >> v.w_low)]);
  }
  out[((u64)blockIdx.y << log_len) + j] = lazy::reduce(sum);
}
// One layer, of a variable with the factors (1, factor[child]): for j < 2^log_half,
//   out[parent][j] = sum over the parent's children of in[child][j],
//   out[parent][2^log_half + j] = sum over them of factor[child] * in[child][j].
// Memory traffic bounds the kernel, so it applies L layers in one pass over the data: a thread has
// one j < 2^log_leaf and one group L layers above the leaves, and keeps the 2^L sums in registers.
// begin[i], factor[i] belong to the i-th of the L layers, the highest variable's first; a parent's
// children are begin[i][parent] .. begin[i][parent + 1] - 1.
template <int L> struct FoldLayers {
  const u32 *begin[L];
  const F *factor[L];
};
// r[0 .. 2^L - 1]: what the L lowest of the ALL layers make of the leaves below `node`, at j.
template <int L, int ALL> struct FoldSubtree {
  static __device__ __forceinline__ void sum(E *r, u32 node, const E *in, unsigned log_leaf, u64 j, const FoldLayers<ALL> &a) {
    constexpr int half = 1 << (L - 1);
    for (int i = 0; i < 2 * half; i++) r[i] = E::zero();
    for (u32 c = a.begin[ALL - L][node]; c < a.begin[ALL - L][node + 1]; c++) {
      E s[half];
      FoldSubtree<L - 1, ALL>::sum(s, c, in, log_leaf, j, a);
      const F f = a.factor[ALL - L][c];
#pragma unroll
      for (int i = 0; i < half; i++) r[i] += s[i], r[half + i] += s[i] * f;
    }
  }
};
template <int ALL> struct FoldSubtree<0, ALL> {
  static __device__ __forceinline__ void sum(E *r, u32 node, const E *in, unsigned log_leaf, u64 j, const FoldLayers<ALL> &) { r[0] = in[((u64)node << log_leaf) + j]; }
};
// The same for layers in which every group has one child, or two with the factors f and -f, as in
// a code's own transform. The two halves are computed in place, and one product serves both
// children: (s + s', f (s - s')).
template <int L, int ALL> struct FoldButterflies {
  static __device__ __forceinline__ void sum(E *r, u32 node, const E *in, unsigned log_leaf, u64 j, const FoldLayers<ALL> &a) {
    constexpr int half = 1 << (L - 1);
    const u32 c = a.begin[ALL - L][node];
    const bool two = a.begin[ALL - L][node + 1] - c == 2; // (all threads of a block alike)
    FoldButterflies<L - 1, ALL>::sum(r, c, in, log_leaf, j, a);
    if (two) FoldButterflies<L - 1, ALL>::sum(r + half, c + 1, in, log_leaf, j, a);
    const F f = a.factor[ALL - L][c];
#pragma unroll
    for (int i = 0; i < half; i++) {
      const E s = r[i];
      if (two) r[i] = s + r[half + i], r[half + i] = (s - r[half + i]) * f;
      else r[half + i] = s * f;
    }
  }
};
template <int ALL> struct FoldButterflies<0, ALL> {
  static __device__ __forceinline__ void sum(E *r, u32 node, const E *in, unsigned log_leaf, u64 j, const FoldLayers<ALL> &) { r[0] = in[((u64)node << log_leaf) + j]; }
};
// Block row y: group y. With step != 0 (the last pass, one group) the result is stored by
// coordinates (see FoldHeld::step).
template <int L, bool BUTTERFLIES> static __global__ void k_fold_layers(E *out, const E *in, unsigned log_leaf, FoldLayers<L> a, u64 step) {
  const u64 j = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= ((u64)1 << log_leaf)) return;
  E r[1 << L];
  if (BUTTERFLIES) FoldButterflies<L, L>::sum(r, blockIdx.y, in, log_leaf, j, a);
  else FoldSubtree<L, L>::sum(r, blockIdx.y, in, log_leaf, j, a);
#pragma unroll
  for (u64 i = 0; i < (1 << L); i++) {
    const u64 at = (i << log_leaf) + j;
    if (step) {
      F *words = reinterpret_cast<F *>(out);
      for (u64 d = 0; d < 3; d++) words[d * step + at] = r[i].c[d];
    } else {
      out[((u64)blockIdx.y << (log_leaf + L)) + at] = r[i];
    }
  }
}
// A stretch of sums, from word `at` on: for each row, `chunks` elements of E, of which every
// `many` neighbours belong to one piece.
struct FoldStretch {
  u64 at;
  u32 chunks, many;
};
// Adds up each piece's chunks, into the first of them. Block row y: stretch y. One thread for each
// (row, piece, coordinate).
static __global__ void k_fold_pieces(F *sums, const FoldStretch *stretches, u32 n_rows) {
  const FoldStretch s = stretches[blockIdx.y];
  const u64 e = (u64)blockIdx.x * blockDim.x + threadIdx.x, pieces = s.chunks / s.many;
  if (e >= (u64)n_rows * pieces * 3) return;
  F *at = sums + s.at + (e / 3) * s.many * 3 + e % 3; // e / 3 = row * pieces + piece
  F x = at[0];
  for (u32 q = 1; q < s.many; q++) x += at[3 * q];
  at[0] = x;
}
// Returns the sum of a warp's elements, in thread 0 only. All 32 threads must call it.
__device__ __forceinline__ E fold_warp_sum(E sum) {
  for (unsigned h = 16; h > 0; h /= 2)
    sum += E::from(F{__shfl_down_sync(~0u, sum.c[0].v, h)}, F{__shfl_down_sync(~0u, sum.c[1].v, h)}, F{__shfl_down_sync(~0u, sum.c[2].v, h)});
  return sum;
}
// gram[row * slots + slot] += sum over the slot's items of coefficient * (the item's element of E
// among the sums). The slot's items are begin[slot] .. begin[slot + 1] - 1. Block (x, y): slot y;
// each of its warps has a row, and the warp's threads share the slot's items.
static __global__ void k_fold_gather(E *gram, u32 slots, u32 n_rows, const F *sums, const u32 *begin, const FoldItem *items) {
  static_assert(FOLD_THREADS % 32 == 0, "whole warps");
  const u32 slot = blockIdx.y, row = blockIdx.x * (FOLD_THREADS / 32) + threadIdx.x / 32;
  if (row >= n_rows) return; // all threads of a warp alike
  lazy::AccE x = lazy::zero_e(); // fewer items than lazy::MAC_ROOM: `make_gram` checks it
  for (u32 i = begin[slot] + threadIdx.x % 32; i < begin[slot + 1]; i += 32) {
    if (!items[i].live) continue;
    const F *s = sums + items[i].at + (u64)row * items[i].row_step;
    lazy::mac(x, items[i].coefficient, E::from(s[0], s[1], s[2]));
  }
  const E sum = fold_warp_sum(lazy::reduce(x));
  if (threadIdx.x % 32 == 0) gram[(u64)row * slots + slot] += sum;
}
// The same with the rows' own cells (`ProductRow`: prover/engine/products.cuh). Block (x, y): row
// x, slot y. The block's threads share the slot's items: one slot can have tens of thousands, each
// a read at a different place, and rows * slots threads are too few to hide the latency of these
// reads.
static __global__ void k_fold_gather_cells(E *gram, u32 slots, const ProductRow *rows, const u32 *begin, const FoldItem *items) {
  __shared__ E of_warp[FOLD_THREADS / 32];
  const u32 slot = blockIdx.y, from = begin[slot], to = begin[slot + 1];
  if (from == to) return;
  const ProductRow row = rows[blockIdx.x];
  lazy::AccE x = lazy::zero_e();
  for (u32 i = from + threadIdx.x; i < to; i += FOLD_THREADS) lazy::mac(x, items[i].coefficient, cell_at(row.cells, row.cell, row.first + items[i].at * row.stride));
  E sum = fold_warp_sum(lazy::reduce(x));
  if (threadIdx.x % 32 == 0) of_warp[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x) return;
  for (unsigned w = 1; w < FOLD_THREADS / 32; w++) sum += of_warp[w];
  gram[(u64)blockIdx.x * slots + slot] += sum;
}
// *at += x, where other threads may add to the same element at the same time. Each coordinate is
// added atomically by itself.
__device__ __forceinline__ void fold_add(E *at, const E &x) {
  static_assert(sizeof(F) == sizeof(unsigned long long), "a coordinate is one word");
  for (unsigned d = 0; d < 3; d++) {
    unsigned long long *word = reinterpret_cast<unsigned long long *>(&at->c[d]), seen = *word, was;
    do {
      was = seen;
      seen = atomicCAS(word, was, (unsigned long long)(F{(u64)was} + x.c[d]).v);
    } while (seen != was);
  }
}
// What the kernels that expand cuts read.
struct FoldCuts {
  const FoldCut *cuts;
  u32 n;
  const E *factors, *values;
  const u32 *forms;
};
// Writes out all coefficients of all cuts as items for k_fold_gather() and k_fold_gather_cells().
// One thread for each coefficient; a cut's are first[cut] .. first[cut + 1] - 1. An item goes to
// the next free entry of its slot's part of the list (next, cell_next: one counter for each slot),
// so the order within a slot is not fixed; the sums do not depend on it.
// sums_at[slab * n_forms + form]: where that V's sums on that slab begin, or all ones if there are
// none: a coefficient that needs them sets *missed, and the host then ends the process.
static __global__ void k_fold_expand(FoldItem *items, u32 *next, FoldItem *cell_items, u32 *cell_next, FoldCuts a, const u32 *first, const u64 *sums_at, u32 n_forms, const u32 *slot_of,
                                     unsigned k, unsigned log_slab, u32 *missed) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= first[a.n]) return;
  const u32 cut = fold_find(first, a.n, (u32)i);
  const FoldCut &c = a.cuts[cut];
  FoldWalk s{c.index, 0, 0, c.scalar};
  fold_walk(s, c, a.factors, i - first[cut], 0, c.open);
  if (c.in_coefficient) s.x = lazy::mul(s.x, a.values[c.values_at + s.d]);
  const u64 place = s.index >> k;
  const u32 slot = slot_of[s.index & (((u64)1 << k) - 1)];
  if (c.w == 0) {
    cell_items[atomicAdd(&cell_next[slot], 1u)] = FoldItem{place, 0, 1, s.x};
    return;
  }
  const u64 at = sums_at[(place >> log_slab) * n_forms + a.forms[c.forms_at + (c.many_forms ? s.together : 0)]];
  if (at == ~(u64)0) atomicOr(missed, 1u);
  items[atomicAdd(&next[slot], 1u)] = at == ~(u64)0 ? FoldItem{0, 0, 0, E::zero()}
                                                    : FoldItem{at + ((place & (((u64)1 << log_slab) - 1)) >> c.chunk) * 3, (u32)(3u << (log_slab - c.chunk)), 1, s.x}; // (the piece's first chunk)
}
// items[at[i]] = fixed[i]. fixed: the items of the written-out rows, which the host makes.
static __global__ void k_fold_put(FoldItem *items, const FoldItem *fixed, const u32 *at, u32 n) {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) items[at[i]] = fixed[i];
}
// For the folded coefficient vector (end()'s c_folded). The coefficients of a cut that differ only
// in the joined row variables go with one piece and one V, so they add up to one item: the sum of
// eq[row] * coefficient, with the V's number in `at`. One thread for each such sum; a cut's are
// first[cut] .. first[cut + 1] - 1. The item goes to the next free entry of its piece's part of the
// list (next: one counter for each piece of each length; FoldCut::begin_at). Without a V (w = 0)
// the sum is added to out[place] at once. eq: the equality table of the k challenges gamma. Where
// the written-out factor does not depend on the joined variables, the sum is a product over the row
// variables.
static __global__ void k_fold_join(FoldItem *items, u32 *next, E *out, FoldCuts a, const u32 *first, const E *eq, const E *gamma, unsigned k) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= first[a.n]) return;
  const u32 cut = fold_find(first, a.n, (u32)i);
  const FoldCut &c = a.cuts[cut];
  FoldWalk rest{c.index, 0, 0, c.scalar};
  fold_walk(rest, c, a.factors, i - first[cut], c.joined, c.open);
  E sum = E::zero();
  if (!c.joined_in_factor) {
    u64 joined = 0;
    sum = c.in_coefficient ? a.values[c.values_at + rest.d] : E::one();
    for (unsigned i = 0; i < c.joined; i++) {
      const E lo = a.factors[2 * (c.factors_at + i)], hi = a.factors[2 * (c.factors_at + i) + 1];
      sum = lazy::mul(sum, lo + lazy::mul(gamma[c.var[i]], hi - lo));
      joined |= (u64)1 << c.var[i];
    }
    for (unsigned t = 0; t < k; t++)
      if (!((joined >> t) & 1)) sum = lazy::mul(sum, (rest.index >> t) & 1 ? gamma[t] : E::one() - gamma[t]);
  }
  for (u64 e = 0; c.joined_in_factor && e < ((u64)1 << c.joined); e++) {
    FoldWalk s{rest.index, rest.d, 0, E::one()};
    fold_walk(s, c, a.factors, e, 0, c.joined);
    s.x = lazy::mul(s.x, eq[s.index & (((u64)1 << k) - 1)]);
    if (c.in_coefficient) s.x = lazy::mul(s.x, a.values[c.values_at + s.d]);
    sum += s.x;
  }
  if (c.w == 0) return fold_add(out + (rest.index >> k), lazy::mul(rest.x, sum));
  items[atomicAdd(&next[c.begin_at + ((rest.index >> k) >> c.w)], 1u)] = FoldItem{a.forms[c.forms_at + (c.many_forms ? rest.together : 0)], 0, 1, lazy::mul(rest.x, sum)};
}
// out[j] = sum over the rows (over F) of weight[row] * entry j of the row, for j < len.
static __global__ void k_fold_combine(E *out, u64 len, const ProductRow *rows, const E *weight, u32 n_rows) {
  const u64 j = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= len) return;
  lazy::AccE sum = lazy::zero_e(); // at most 65,535 rows
  for (u32 b = 0; b < n_rows; b++) lazy::mac(sum, weight[b], cell_at(rows[b].cells, rows[b].cell, rows[b].first + j * rows[b].stride));
  out[j] = lazy::reduce(sum);
}
// The same for few j: a warp has one j, and its threads share the rows.
static __global__ void k_fold_combine_short(E *out, u64 len, const ProductRow *rows, const E *weight, u32 n_rows) {
  const u64 j = (u64)blockIdx.x * (FOLD_THREADS / 32) + threadIdx.x / 32;
  if (j >= len) return; // all threads of a warp alike
  lazy::AccE sum = lazy::zero_e();
  for (u32 b = threadIdx.x % 32; b < n_rows; b += 32) lazy::mac(sum, weight[b], cell_at(rows[b].cells, rows[b].cell, rows[b].first + j * rows[b].stride));
  const E all = fold_warp_sum(lazy::reduce(sum));
  if (threadIdx.x % 32 == 0) out[j] = all;
}
// The same, for when some rows have cells of three signed bytes (Cell::S24); with rows of
// Cell::WORD only it is the slower of the two. Such a cell is not made into a field element: each
// of its bytes has a separate sum of weight * byte, and the three are joined at the end with the
// bytes' weights. FOLD_COMBINE_ROWS: the rows in shared memory at a time.
constexpr unsigned FOLD_COMBINE_ROWS = 128;
static __global__ void k_fold_combine_narrow(E *out, u64 len, const ProductRow *rows, const E *weight, u32 n_rows) {
  static_assert(FOLD_COMBINE_ROWS <= FOLD_THREADS, "a thread brings in one row's description");
  __shared__ ProductRow row[FOLD_COMBINE_ROWS];
  __shared__ E whole[FOLD_COMBINE_ROWS];
  __shared__ lazy::IPrep limbs[FOLD_COMBINE_ROWS];
  const u64 j = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  // At most 65,535 rows: far within MAC_ROOM, and 65,535 * 128 is within 2^IMAC_ROOM_BITS.
  lazy::AccE sum = lazy::zero_e();
  lazy::IAcc byte0 = lazy::izero(), byte1 = lazy::izero(), byte2 = lazy::izero();
  for (u32 from = 0; from < n_rows; from += FOLD_COMBINE_ROWS) {
    const u32 n = n_rows - from < FOLD_COMBINE_ROWS ? n_rows - from : FOLD_COMBINE_ROWS;
    __syncthreads();
    if (threadIdx.x < n) row[threadIdx.x] = rows[from + threadIdx.x], whole[threadIdx.x] = weight[from + threadIdx.x], limbs[threadIdx.x] = lazy::iprep(weight[from + threadIdx.x]);
    __syncthreads();
    if (j >= len) continue;
    for (u32 i = 0; i < n; i++) {
      const u64 at = row[i].first + j * row[i].stride;
      if (row[i].cell == Cell::S24) {
        const signed char *b = (const signed char *)row[i].cells + 3 * at;
        lazy::imac(byte0, limbs[i], (lazy::i32)__ldg(b)), lazy::imac(byte1, limbs[i], (lazy::i32)__ldg(b + 1)), lazy::imac(byte2, limbs[i], (lazy::i32)__ldg(b + 2));
      } else {
        lazy::mac(sum, whole[i], cell_at(row[i].cells, row[i].cell, at));
      }
    }
  }
  if (j < len) out[j] = lazy::reduce(sum) + lazy::ired(byte0) + lazy::shifted<S24_BITS>(lazy::ired(byte1)) + lazy::shifted<2 * S24_BITS>(lazy::ired(byte2));
}

// The same on the tensor cores (k_fold_combine_bytes()), for the rows with cells of three signed
// bytes; the other rows are added by the field's arithmetic. A weight times a byte's weight
// 2^(S24_BITS i) is an element of E: three coordinates of eight signed bytes. So out[j] is a
// product of two matrices of bytes: 24 rows of weights' bytes, with one column for each
// (row, byte i), times the cells' bytes. The cells are read as they are stored: (row, i) has the
// number 4 row + i, and the fourth byte that is read with a cell meets a weight of 0.
//
// bytes[(8 d + j) * pitch + 4 r + i] = signed byte j of coordinate d of
// weight[narrow[r]] * 2^(S24_BITS i), for r < n and i < 3. All else in `bytes` must be 0 before.
// One thread for each (r, i).
static __global__ void k_fold_weight_bytes(signed char *bytes, u32 pitch, const E *weight, const u32 *narrow, u32 n) {
  const u32 e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= 3 * n) return;
  const E x = lazy::times_2_to(weight[narrow[e / 3]], (int)(S24_BITS * (e % 3)));
  for (unsigned d = 0; d < 3; d++) {
    const u64 digits = lazy::signed_digits(x.c[d]);
    for (unsigned j = 0; j < 8; j++) bytes[(8 * d + j) * pitch + 4 * (e / 3) + e % 3] = (signed char)(unsigned char)(digits >> (8 * j));
  }
}
constexpr unsigned FOLD_BYTES_PLACES = 64;                     // a warp's places at a time
// The distance, in 32-bit sums, from one row of weights' bytes to the next: a warp's writes and
// reads then meet no bank of shared memory twice [Padding, tiled.cuh].
constexpr unsigned FOLD_BYTES_SUMS = FOLD_BYTES_PLACES + 1;
inline size_t fold_bytes_room(u32 pitch, u32 padded) { return (size_t)32 * pitch + (size_t)padded * sizeof(const u8 *) + (FOLD_THREADS / 32) * 24 * FOLD_BYTES_SUMS * sizeof(int); }
// len: a multiple of FOLD_BYTES_PLACES. narrow: the numbers of the n rows with such cells, each
// with stride 1 and cells that begin at a multiple of 4 bytes. padded: n rounded up to a multiple
// of 8. pitch: a multiple of 16, at least 4 * padded. others: the numbers of the other rows. Which
// thread holds what: see tiled_mma() in tiled.cuh.
// Not checked: n is at least 1; bytes is k_fold_weight_bytes()'s, 32 rows of pitch bytes, and
// begins at a multiple of 16; the launch gives fold_bytes_room(pitch, padded) bytes of shared
// memory.
static __global__ void __launch_bounds__(FOLD_THREADS)
    k_fold_combine_bytes(E *out, u64 len, const ProductRow *rows, const E *weight, const u32 *narrow, u32 n, u32 padded, const signed char *bytes, u32 pitch, const u32 *others, u32 n_others) {
  extern __shared__ __align__(16) unsigned char fold_room[];
  u32 *w = reinterpret_cast<u32 *>(fold_room); // the weights' bytes: 32 rows, the last 8 zero
  const u8 **cells = reinterpret_cast<const u8 **>(fold_room + (size_t)32 * pitch);
  int *sums = reinterpret_cast<int *>(fold_room + (size_t)32 * pitch + (size_t)padded * sizeof(const u8 *)) + (threadIdx.x / 32) * 24 * FOLD_BYTES_SUMS; // the warp's own
  for (u32 i = threadIdx.x; i < 32 * pitch / 16; i += FOLD_THREADS) reinterpret_cast<uint4 *>(w)[i] = reinterpret_cast<const uint4 *>(bytes)[i];
  // Beyond the last row the first is read once more; its weights there are 0.
  for (u32 r = threadIdx.x; r < padded; r += FOLD_THREADS) {
    const ProductRow row = rows[narrow[r < n ? r : 0]];
    cells[r] = (const u8 *)row.cells + 3 * row.first;
  }
  __syncthreads();
  const unsigned lane = threadIdx.x % 32, g = lane >> 2, t = lane & 3, words = pitch / 4;
  for (u64 first = ((u64)blockIdx.x * (FOLD_THREADS / 32) + threadIdx.x / 32) * FOLD_BYTES_PLACES; first < len; first += (u64)gridDim.x * (FOLD_THREADS / 32) * FOLD_BYTES_PLACES) {
    int sum[2][4][2][4]; // [32 places][place mod 4][16 rows of weights' bytes][..]
#pragma unroll
    for (unsigned e = 0; e < 64; e++) (&sum[0][0][0][0])[e] = 0;
    for (u32 r = 0; r < padded; r += 8) {
      u32 a[2][4];
#pragma unroll
      for (unsigned m = 0; m < 2; m++) {
        a[m][0] = w[(16 * m + g) * words + r + t], a[m][1] = w[(16 * m + g + 8) * words + r + t];
        a[m][2] = w[(16 * m + g) * words + r + 4 + t], a[m][3] = w[(16 * m + g + 8) * words + r + 4 + t];
      }
#pragma unroll
      for (unsigned half = 0; half < 2; half++) {
        // The thread reads the places 4 g .. 4 g + 3 of the 32, in the rows r + t and r + 4 + t:
        // three 32-bit words each.
        const u32 *low = reinterpret_cast<const u32 *>(cells[r + t] + 3 * (first + 32 * half)) + 3 * g, *high = reinterpret_cast<const u32 *>(cells[r + 4 + t] + 3 * (first + 32 * half)) + 3 * g;
        const u32 l0 = __ldg(low), l1 = __ldg(low + 1), l2 = __ldg(low + 2), h0 = __ldg(high), h1 = __ldg(high + 1), h2 = __ldg(high + 2);
        const u32 b[4][2] = {{l0, h0}, {__byte_perm(l0, l1, 0x6543), __byte_perm(h0, h1, 0x6543)}, {__byte_perm(l1, l2, 0x5432), __byte_perm(h1, h2, 0x5432)}, {__byte_perm(l2, 0, 0x4321), __byte_perm(h2, 0, 0x4321)}};
#pragma unroll
        for (unsigned q = 0; q < 4; q++)
#pragma unroll
          for (unsigned m = 0; m < 2; m++) tiled_mma(sum[half][q][m], a[m], b[q][0], b[q][1]);
      }
    }
    // Column c of product (half, q) is place 32 half + 4 c + q.
#pragma unroll
    for (unsigned half = 0; half < 2; half++)
#pragma unroll
      for (unsigned q = 0; q < 4; q++)
#pragma unroll
        for (unsigned e = 0; e < 4; e++) {
          const unsigned place = 32 * half + 4 * (2 * t + (e & 1)) + q;
          sums[(g + 8 * (e >> 1)) * FOLD_BYTES_SUMS + place] = sum[half][q][0][e];
          if (e < 2) sums[(16 + g) * FOLD_BYTES_SUMS + place] = sum[half][q][1][e];
        }
    __syncwarp();
    for (unsigned place = lane; place < FOLD_BYTES_PLACES; place += 32) {
      E x;
      for (unsigned d = 0; d < 3; d++) {
        long long low = 0, high = 0; // below 2^57 in size
        for (unsigned j = 0; j < 4; j++) low += (long long)sums[(8 * d + j) * FOLD_BYTES_SUMS + place] << (8 * j), high += (long long)sums[(8 * d + 4 + j) * FOLD_BYTES_SUMS + place] << (8 * j);
        x.c[d] = lazy::join_digit(low, 0) + lazy::join_digit(high, 32);
      }
      lazy::AccE rest = lazy::zero_e();
      for (u32 i = 0; i < n_others; i++) {
        const ProductRow row = rows[others[i]];
        lazy::mac(rest, weight[others[i]], cell_at(row.cells, row.cell, row.first + (first + place) * row.stride));
      }
      out[first + place] = x + lazy::reduce(rest);
    }
    __syncwarp();
  }
}

// For k_fold_spread(): out[place] += sum, over the lengths 2^w of pieces, over the items of the
// place's piece, of coefficient * (entry place mod 2^w of form number `at`). The items of piece
// number p of length number l are begin[begin_at[l] + p] .. begin[begin_at[l] + p + 1] - 1. One
// thread for each place.
struct FoldLengths {
  u32 n;
  u32 w[FOLD_MOST_W + 1], begin_at[FOLD_MOST_W + 1];
};
static __global__ void k_fold_spread(E *out, u64 ell, FoldLengths lengths, const u32 *begin, const FoldItem *items, const FoldVector *vectors, const F *store) {
  const u64 place = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (place >= ell) return;
  lazy::AccE x = lazy::zero_e(); // fewer items in all than lazy::MAC_ROOM: `add_cut` checks it
  for (u32 l = 0; l < lengths.n; l++) {
    const unsigned w = lengths.w[l];
    const u64 in = place & (((u64)1 << w) - 1), len = (u64)1 << w;
    const u32 *mine = begin + lengths.begin_at[l] + (place >> w);
    for (u32 i = mine[0]; i < mine[1]; i++) {
      const F *v = store + vectors[items[i].at].at + in;
      lazy::mac(x, items[i].coefficient, E::from(v[0], v[len], v[2 * len]));
    }
  }
  out[place] += lazy::reduce(x);
}
// out[item.at] += eq[item.row_step] * item.coefficient; row_step is here the row of c. Several
// items may name the same place.
static __global__ void k_fold_spread_single(E *out, const FoldItem *items, u32 n, const E *eq) {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) fold_add(out + items[i].at, lazy::mul(eq[items[i].row_step], items[i].coefficient));
}
// out[place] = sum over h in `held` of h.scale * fold_held_at(h, place). Overwrites out. held_n is
// not checked against lazy::MAC_ROOM.
static __global__ void k_fold_spread_held(E *out, u64 ell, const FoldHeld *held, u32 held_n) {
  const u64 place = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (place >= ell) return;
  lazy::AccE x = lazy::zero_e(); // (for a row of c: at most one per hint and term, and one sum)
  for (u32 i = 0; i < held_n; i++) lazy::mac(x, held[i].scale, fold_held_at(held[i], place));
  out[place] = lazy::reduce(x);
}

// CardFold's settings. Slab, V, piece, chunk and span: see "How G is made" above.
struct FoldSetting {
  // BYTES: numbers are cut into signed bytes, which the tensor cores multiply; PLAIN: field
  // arithmetic (products.cuh).
  ProductWay way = ProductWay::BYTES;
  unsigned log_slab = FOLD_MOST_LOG_SLAB; // a slab has min(2^log_slab, 2^16, ell) places
  // A thread block multiplies on a chunk of 2^log_chunk places (tiled.cuh); with a shorter V, of
  // 2^log_short_chunk; with a V shorter still, of the V's length. With few chunk lengths the V's of
  // one slab fill their tiles better.
  unsigned log_chunk = 13, log_short_chunk = 10;
  // A span has as many slabs as keep the rows' bytes, the written-out rows of c and those rows'
  // bytes within span_bytes, and at most 2^most_log_span places, but at least one slab.
  size_t span_bytes = (size_t)1 << 28;
  unsigned most_log_span = 64;
  // Where only the forms' bytes are kept, the forms are written out at most so many words at a
  // time, or one form if it is longer.
  u64 store_words = (u64)1 << 24;
  // A V on fewer low variables is not multiplied separately: its many short pieces would give more
  // 32-bit sums than there are cells.
  unsigned least_w = 10;
  // CardProducts::raw_words (products.cuh) where a row of c is summed as a product of two matrices.
  size_t raw_words = (size_t)1 << 26;
  // A list whose entries take a row of thread blocks each is launched at most so many entries at a
  // time; a grid has at most 65,535 rows of blocks. A lower value is for tests: short lists then
  // take several launches too.
  unsigned grid_rows = 65535;
  bool trace = false; // wait for every step and add up its seconds, for reports
  bool folded_coefficients = true; // false: `end`'s c_folded must be null; nothing is kept for it
};

// Use: the constructor makes G; then k times round() and bind(); then end(). The constructor and
// end() return after the stream's work is done.
// g_parallel_fold (common/setting.h): one-off (not upstream), set from args.parallel_fold
// (inputs.cuh) via -parallel-fold. See bind() below for why.
class CardFold {
public:
  // A row of c with at least this many product forms over F on all places has them summed
  // beforehand for the entire row (further conditions: make_gram()).
  static constexpr size_t MANY_TERMS = 32;

  // held: pointers to elements of E (three words each) in GPU memory, in three groups.
  // 1. For each hint c.held[i]: its vector, or null if the engine does not have it. A hint is used
  //    if it has a vector and picks row variables only; the vector must then have ell * 2^(row
  //    variables that it does not pick) elements. No two used hints may cover the same part of c.
  // 2. For each c.engine_terms[i]: the same. A term is stored nowhere else, so one that cannot be
  //    used ends the process.
  // 3. For each Tensor of c.tensors with is_held(), in order: the table that contains its
  //    written-out factor, not null. The caller must have checked that the table has
  //    Tensor::held_length elements.
  // The comment at the top of this file states what cells, tails, vectors and the stream must be.
  CardFold(const std::vector<CardRows> &parts, unsigned k, const Code &code, const Form &c, const std::vector<const u64 *> &held, cudaStream_t stream,
           const FoldSetting &setting = FoldSetting())
      : k_(k), left_(k), ell_(code.ell), r_(code.r), stream_(stream), setting_(setting) {
    const OnStream on(stream);
    Uploads uploads(stream);
    up_ = &uploads;
    BLK_REQUIRE(k <= 10, "fold: more than 10 row variables");
    BLK_REQUIRE(!parts.empty(), "fold: no rows");
    log_ell_ = ceil_log2(ell_);
    BLK_REQUIRE(((u64)1 << log_ell_) == ell_, "fold: the message length is not a power of two");
    BLK_REQUIRE(c.fits(k_ + log_ell_), "fold: the coefficient vector has not the group's length");
    size_t held_factors = 0;
    for (const Tensor &v : c.tensors) held_factors += v.is_held();
    BLK_REQUIRE(held.size() == c.held.size() + c.engine_terms.size() + held_factors,
                "fold: not one pointer (or null) for each hint, each engine's term and each factor that the engine holds of the coefficient vector");
    BLK_REQUIRE(c.frozen.empty(), "fold: a frozen part in a prover's coefficient vector");
    const u64 all = (u64)1 << k;
    std::vector<ProductRow> rows, tails;
    std::vector<char> taken(all, 0);
    for (const CardRows &p : parts) {
      BLK_REQUIRE(p.deg == 1 || p.deg == E_DEGREE, "fold: entries that are neither in F nor in E");
      BLK_REQUIRE(p.rows >= 1 && p.first_row + p.rows <= all, "fold: rows beyond the group");
      for (size_t b = 0; b < p.rows; b++) {
        BLK_REQUIRE(!taken[p.first_row + b], "fold: a row of the group is given twice");
        taken[p.first_row + b] = 1;
      }
      BLK_REQUIRE(p.cells && (p.tail || r_ == 0), "fold: rows without cells or without their encoding randomness");
      for (size_t b = 0; b < p.rows; b++)
        for (unsigned d = 0; d < p.deg; d++) { // a row over E is three over F; d stands for X^d
          rows.push_back(ProductRow{p.cells, b * ell_ * p.deg + d, p.deg, p.cell});
          narrow_rows_ = narrow_rows_ || p.cell == Cell::S24;
          // k_fold_combine_bytes() takes rows over F of three-byte cells (Cell::S24). It reads
          // 32-bit words, so its rows begin at a multiple of 4 bytes.
          (p.cell == Cell::S24 && p.deg == 1 && ((std::uintptr_t)p.cells + 3 * b * ell_) % 4 == 0 ? narrow_ : others_).push_back((u32)rows.size() - 1);
          tails.push_back(ProductRow{p.tail, b * r_ * p.deg + d, p.deg, Cell::WORD});
          row_of_.push_back((u32)(p.first_row + b));
          power_of_.push_back(d);
        }
    }
    up_->send(tails_, tails.data(), tails.size());
    log_slab_ = std::min(log_ell_, std::min(setting_.log_slab, FOLD_MOST_LOG_SLAB));
    BLK_REQUIRE(setting_.grid_rows >= 1 && setting_.grid_rows <= 65535, "fold: no row of blocks in a launch, or more than a grid has");
    products_ = std::make_unique<CardTiled>(rows, setting_.way, stream_, uploads);
    products_->grid_rows = setting_.grid_rows;
    products_->trace = setting_.trace;
#ifndef BLK_FOLD_PRINT
#define BLK_FOLD_PRINT // one-off (not upstream): enable the fold's own built-in timing printf
#endif
#ifdef BLK_FOLD_PRINT // a build switch: prints where each fold's time goes, from the members below
    const auto t_begin = std::chrono::steady_clock::now();
#endif
    make_gram(c, held);
    uploads.finished(), up_ = nullptr; // (make_gram() has waited for the stream)
#ifdef BLK_FOLD_PRINT
    std::printf("    CardFold k %u ell 2^%u rows %u: %zu tensors %zu entries %zu held | all %.3f: look %.3f (hints %.3f keys %.3f terms %.3f reach %.3f) forms %.3f summing %.3f plan %.3f | slabs+ %.3f (items %.3f) | V %zu items %zu written %zu layers %zu matrices %zu worth %.1f + %.1f\n",
                  k_, log_ell_, products_->rows(), c.tensors.size(), c.entries.size(), c.held.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t_begin).count(), seconds_cut,
                  seconds_cut_part[0], seconds_cut_part[1], seconds_cut_part[2], seconds_cut_part[3], seconds_forms, seconds_summing, seconds_plan, seconds_card, seconds_items, forms_made, items_made,
                  written_rows, rows_summed_by_layers, rows_summed_by_matrices, pieces_worth, written_worth);
#endif
  }

  ~CardFold() { forget_coefficients(); }
  CardFold(const CardFold &) = delete;
  CardFold &operator=(const CardFold &) = delete;

  // < m , c >, the inner product of the two tables as they stand: the sum of G's diagonal.
  E inner_product() const {
    const size_t n = (size_t)1 << left_;
    E sum = E::zero();
    for (size_t a = 0; a < n; a++) sum += gram_[a * n + a];
    return sum;
  }

  // The round's polynomial, unmasked: q(X) = q[0] + q[1] X + q[2] X^2.
  void round(E q[3]) const {
    BLK_REQUIRE(left_ >= 1, "fold: no row variable is left");
    const size_t n = (size_t)1 << left_;
    E g00 = E::zero(), g01 = E::zero(), g10 = E::zero(), g11 = E::zero();
    for (size_t a = 0; a < n; a += 2) {
      g00 += gram_[a * n + a], g01 += gram_[a * n + a + 1];
      g10 += gram_[(a + 1) * n + a], g11 += gram_[(a + 1) * n + a + 1];
    }
    q[0] = g00;
    q[1] = g01 + g10 - g00 - g00;
    q[2] = g00 + g11 - g01 - g10;
  }

  // Binds the lowest row variable that is left to gamma.
  // One-off (not upstream): these two loops over (a, b) are independent across a (each output
  // entry reads only gram_/half, never another a's output), and measurement (stage_clock.h's
  // eqcol_close bucket: its "stream" and "host" times come out equal, meaning no GPU work is
  // queued while this runs) showed this plain host loop, not a GPU kernel, is what "the closing"
  // actually spends its time on for a small fold (CardFold is capped at 2^10 rows). g_parallel_fold
  // (-parallel-fold) splits both loops' outer index across std::thread workers; off by default, so
  // the original sequential code below is unchanged unless asked for.
  void bind(const E &gamma) {
    BLK_REQUIRE(left_ >= 1, "fold: no row variable is left");
    const size_t n = (size_t)1 << left_, h = n / 2;
    std::vector<E> half(n * h), next(h * h);
    if (g_parallel_fold && n >= 64) {
      // Capped low, not hardware_concurrency(): CardFold is at most 2^10 rows by design (this
      // file's header comment), so bind()'s work per call tops out around n/2 * n/2 fold_pair
      // calls (<= ~262k for n=1024) -- too little work to amortize OS thread-creation cost across
      // more than a handful of threads, confirmed by measurement (128 threads measured slower
      // than the unthreaded baseline on a 128-vCPU host).
      const unsigned T = std::max(1u, std::min(8u, (unsigned)n));
      auto split = [&](size_t total, auto body) {
        std::vector<std::thread> pool;
        const size_t chunk = (total + T - 1) / T;
        for (unsigned t = 0; t < T; t++) {
          const size_t a0 = t * chunk, a1 = std::min(total, a0 + chunk);
          if (a0 < a1) pool.emplace_back(body, a0, a1);
        }
        for (std::thread &th : pool) th.join();
      };
      split(n, [&](size_t a0, size_t a1) {
        for (size_t a = a0; a < a1; a++)
          for (size_t b = 0; b < h; b++) half[a * h + b] = fold_pair(gram_[a * n + 2 * b], gram_[a * n + 2 * b + 1], gamma);
      });
      split(h, [&](size_t a0, size_t a1) {
        for (size_t a = a0; a < a1; a++)
          for (size_t b = 0; b < h; b++) next[a * h + b] = fold_pair(half[2 * a * h + b], half[(2 * a + 1) * h + b], gamma);
      });
    } else {
      for (size_t a = 0; a < n; a++)
        for (size_t b = 0; b < h; b++) half[a * h + b] = fold_pair(gram_[a * n + 2 * b], gram_[a * n + 2 * b + 1], gamma);
      for (size_t a = 0; a < h; a++)
        for (size_t b = 0; b < h; b++) next[a * h + b] = fold_pair(half[2 * a * h + b], half[(2 * a + 1) * h + b], gamma);
    }
    gram_.swap(next);
    gamma_.push_back(gamma);
    left_--;
  }

  // After k binds. message: space for ell elements of E on the GPU; it has the folded row when the
  // call returns. tail: gets the rows' random tails folded likewise, one row of r elements of E on
  // the host. c_folded, if not null: like `message`, for the folded coefficient vector.
  void end(E *message, Table &tail, E *c_folded = nullptr) {
    BLK_REQUIRE(left_ == 0, "fold: row variables are left");
    const OnStream on(stream_);
    Uploads uploads(stream_);
    up_ = &uploads;
    std::vector<E> eq((size_t)1 << k_), weight(row_of_.size());
    eq_table(gamma_.data(), k_, eq.data());
    for (size_t i = 0; i < weight.size(); i++) weight[i] = eq[row_of_[i]] * E::theta(power_of_[i]);
    Pooled<E> on_card, folded_tail(r_);
    up_->send(on_card, weight.data(), weight.size());
    if (!combine_by_bytes(message, on_card.get())) {
      (narrow_rows_ ? k_fold_combine_narrow : k_fold_combine)<<<blocks_for(ell_, FOLD_THREADS), FOLD_THREADS, 0, stream_>>>(message, ell_, products_->rows_on_card(), on_card.get(), (u32)weight.size());
      ENGINE_CK(cudaGetLastError());
    }
    tail = Table(E_DEGREE, 1, r_);
    if (r_) {
      k_fold_combine_short<<<blocks_for(r_, FOLD_THREADS / 32), FOLD_THREADS, 0, stream_>>>(folded_tail.get(), r_, tails_.get(), on_card.get(), (u32)weight.size());
      ENGINE_CK(cudaGetLastError());
    }
    ENGINE_CK(cudaStreamSynchronize(stream_));
    static_assert(sizeof(E) == 3 * sizeof(F), "a row over E lies in a Table as its elements do in memory");
    folded_tail.to_host(reinterpret_cast<E *>(tail.a.data()));
#ifdef BLK_FOLD_PRINT
    const auto t_end = std::chrono::steady_clock::now();
#endif
    BLK_REQUIRE(!c_folded || setting_.folded_coefficients, "fold: the folded coefficient vector is asked of a fold that kept nothing for it");
    if (c_folded) write_out_folded(eq, c_folded);
    uploads.finished(), up_ = nullptr; // (the stream has been waited for since the last upload)
#ifdef BLK_FOLD_PRINT
    if (c_folded)
      std::printf("    CardFold end: folded coefficients %.3f s (%zu tensors, %zu entries, %zu held left)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t_end).count(), rest_.tensors.size(),
                  d_entries_.size(), held_.size());
#endif
  }

  // ---- for reports and benchmarks: what the constructor did ----
  // Host wall-clock seconds. seconds_host: until the plan's lists are sent to the GPU;
  // seconds_card: from there until the GPU has finished G. Without FoldSetting::trace the host does
  // not wait after each step, so the GPU's work for a step may fall into a later number.
  double seconds_host = 0, seconds_card = 0;
  // seconds_host = classifying the Tensors + forms and tables to the GPU + entire rows of c summed
  // + the plan of multiplications + uploads.
  double seconds_cut = 0, seconds_forms = 0, seconds_summing = 0, seconds_plan = 0;
  double seconds_items = 0; // in seconds_card: host work on cuts and items while the GPU multiplies
  // Parts of seconds_cut, not all of it: hints; the V's lengths and keys; rows to be written out;
  // which V on which slab.
  double seconds_cut_part[4] = {0, 0, 0, 0};
  // With trace: rows of c written out or summed; G gathered. The other steps' seconds: products().
  double seconds_write_out = 0, seconds_gather = 0;
  const CardTiled &products() const { return *products_; }
  // Multiplications done; the unit is one vector against all rows of m on all ell places.
  double pieces_worth = 0, written_worth = 0;
  // By w. first: tiles of columns with chunks of 2^w places (tiled.cuh). second: pairs (slab, V)
  // with a V of 2^w entries.
  std::map<unsigned, std::pair<size_t, size_t>> steps_made;
  // Items (FoldItem) in all; V's; written-out rows of c.
  size_t items_made = 0, forms_made = 0, written_rows = 0;
  // Rows of c that were summed beforehand: layer by layer; as a product of two matrices.
  size_t rows_summed_by_layers = 0, rows_summed_by_matrices = 0;
  size_t rows_taken_whole = 0; // of these, the rows whose sum was multiplied where it is stored
  // The most launches that a list took (FoldSetting::grid_rows). [0]: the V's, written out all at
  // once. [1]: the split forms. [2]: the stretches (FoldStretch). [3] is 2 if a group of V's was
  // closed because it filled a launch.
  mutable size_t list_launches[4] = {0, 0, 0, 0};
  // The forms were written out in so many groups and only their bytes kept; 0: all at once.
  size_t form_groups = 0;
  // Of factors that the engine holds: tables copied for the V's; entries copied for coefficients.
  size_t held_tables_copied = 0, held_entries_read = 0;

private:
  // Keeps the memory of long host vectors for the same thread's next fold, because fresh memory
  // costs a page fault for every 4 KB. Like the GPU's work spaces it is not wiped: it keeps
  // coefficients of earlier folds' vectors c.
  template <class T> struct Spare {
    static std::vector<std::vector<T>> &all() {
      static thread_local std::vector<std::vector<T>> lists;
      return lists;
    }
    static std::vector<T> take() {
      if (all().empty()) return {};
      std::vector<T> list = std::move(all().back());
      all().pop_back();
      list.clear();
      return list;
    }
    static void give(std::vector<T> &&list) {
      if (list.capacity() && all().size() < 16) all().push_back(std::move(list));
    }
  };
  // What the host needs to know of a Tensor's pairs (lo, hi): one bit for each variable.
  struct Look {
    u64 zero_lo = 0, zero_hi = 0; // lo is 0; hi is 0
    u64 in_base = 0;              // both lie in F
    // Exactly one of lo, hi is 0: the variable picks a bit (the other factor need not be 1).
    u64 picks() const { return zero_lo ^ zero_hi; }
  };
  static Look look_at(const Tensor &v) {
    Look x;
    const E *pair = v.real_pairs();
    const u64 picked = v.pick_mask(), value = v.pick_value();
    for (unsigned t = 0; t < v.vars(); t++) {
      const u64 bit = (u64)1 << t;
      if (picked & bit) {
        (value & bit ? x.zero_lo : x.zero_hi) |= bit, x.in_base |= bit;
        continue;
      }
      if (is_zero(pair[0])) x.zero_lo |= bit;
      if (is_zero(pair[1])) x.zero_hi |= bit;
      if (is_zero(pair[0].c[1]) && is_zero(pair[0].c[2]) && is_zero(pair[1].c[1]) && is_zero(pair[1].c[2])) x.in_base |= bit;
      pair += 2;
    }
    return x;
  }
  // The product of the nonzero factors of the variables in `among`, all of which must pick a bit.
  // (Those in Tensor::pick_mask() have the factor 1.)
  static E picked_factors(const Tensor &v, const Look &x, u64 among) {
    E product = E::one();
    for (u64 rest = among & ~v.pick_mask(); rest; rest &= rest - 1) {
      const unsigned t = (unsigned)__builtin_ctzll(rest);
      product = product * ((x.zero_lo >> t) & 1 ? v.hi(t) : v.lo(t));
    }
    return product;
  }
  // Tensor::dense_vars, the variables that index the written-out factor, as a mask.
  static u64 dense_mask(const Tensor &v) {
    u64 dense = 0;
    for (unsigned d : v.dense_vars) dense |= (u64)1 << d;
    return dense;
  }
  // A form's key: w and the pairs' values for the w variables that begin at `from`, however the
  // Tensor was built. So a V shared by thousands of Tensors is written out and multiplied once.
  // This also merges the generator rows of two positions drawn by the verifier, if their factors on
  // these variables are equal: two draws of one position, and different draws where a shorter
  // codeword is read (a mask row, a shorter tree: at the position modulo the short length,
  // blocks/code/short_position.h). So the memory allocated for V's and split forms, and all that
  // follows from their number, depends on the number of distinct short positions among the draws.
  // Here an allocation depends on a drawn (public) value, and not rarely: among some hundred
  // draws in a codeword of 2^13 symbols, two coincide in nearly every second list. plan_layers()
  // states the condition that this does not meet, and two more such cases.
  static std::string key_of(const Tensor &v, unsigned from, unsigned w) {
    std::string key((const char *)&w, sizeof w);
    for (unsigned t = 0; t < w; t++) {
      const E pair[2] = {v.lo(from + t), v.hi(from + t)};
      key.append((const char *)pair, sizeof pair);
    }
    return key;
  }
  // Returns w, the number of low place variables that make a Tensor's V; 0: it goes cell by cell.
  // k: the number of row variables (0 once they are bound).
  // dense, picks: the variables that index the written-out factor; those that pick a bit.
  // w is the length, at most log_slab, of the run of like variables that begins at the lowest place
  // variable: all index the written-out factor, or none indexes it or picks a bit. Many Tensors
  // share such a V. A run below FoldSetting::least_w is not worth a multiplication: then
  // w = log_slab, or 0 if log_slab or the number of variables that pick no bit is below least_w.
  unsigned width_from(u64 dense, u64 picks, unsigned vars, unsigned k, unsigned log_slab) const {
    const u64 other = (dense >> k) & 1 ? ~dense : dense | picks; // the variables that end the run
    const unsigned run = (unsigned)__builtin_ctzll((other >> k) | ((u64)1 << log_slab));
    if (run >= setting_.least_w) return run;
    return log_slab >= setting_.least_w && (unsigned)__builtin_popcountll(~picks & (((u64)1 << vars) - 1)) >= setting_.least_w ? log_slab : 0;
  }
  // The vectors V of the way PIECES, as they go to the GPU. vector_of: a V's key (Cutter::form())
  // -> its number in `vectors`. store_words: the words of all V's when written out.
  struct Forms {
    std::vector<E> factors;
    // The written-out factors that V's read go to the GPU once each, one after the other, values_n
    // elements in all. values_at: a host vector -> where. tables_at, for the tables that the engine
    // holds, which are copied there: table -> (where, its elements).
    std::map<const std::vector<E> *, u64> values_at;
    std::map<const u64 *, std::pair<u64, u64>> tables_at;
    u64 values_n = 0;
    // The slices of written-out factors whose entries go into coefficients: send_cuts() puts them
    // into one GPU buffer, one after the other.
    struct Slice {
      const E *host;    // on the host; or null, and
      const u64 *table; // on the GPU, in a table that the engine holds: its first word
      u64 entries, at;
    };
    std::map<std::pair<const void *, u64>, u64> slice_at; // (first entry, entries) -> where
    std::vector<Slice> slices;
    u64 slices_n = 0, slices_held = 0; // entries in all; those of them that the engine holds
    std::vector<FoldVector> vectors;
    std::map<std::string, u32> vector_of;
    u64 store_words = 0;
  };
  // A Tensor as coefficient * V, with V on its w lowest place variables. k: see width_from(). With
  // w = 0 there is no V, and the coefficients are the Tensor's entries.
  // The written-out factor's index is d_low (from V's variables) + d_high (from the others). If V
  // includes variables of the factor, there is a V for each d_high; if not, the factor's entry at
  // d_high goes into the coefficient.
  class Cutter {
  public:
    // table: the first word of the table that contains the Tensor's factor, if the engine holds it.
    Cutter(const Tensor &v, unsigned k, unsigned w, Forms &forms, const u64 *table = nullptr) : v_(v), k_(k), w_(w), forms_(forms), key_(key_of(v, k, w)) {
      x_.w = w, x_.has_values = false, x_.has_factors = false, x_.values_at = 0, x_.factors_at = 0, x_.at = 0;
      BLK_REQUIRE(w <= FOLD_MOST_W, "fold: a V on more variables than it has room for");
      for (unsigned t = 0; t < FOLD_MOST_W; t++) x_.shift[t] = NO_SHIFT;
      BLK_REQUIRE(v.dense_vars.empty() || v.values || v.is_held(), "fold: a Tensor has variables joined in a written-out factor, and no such factor");
      BLK_REQUIRE(!v.is_held() || table, "fold: the table of a factor that only the engine holds is not at hand, or is shorter than the Tensor says");
      BLK_REQUIRE(!v.is_held() || (v.dense_vars.size() < 64 && v.first <= v.held_length && ((u64)1 << v.dense_vars.size()) <= v.held_length - v.first),
                  "fold: a factor that the engine holds reaches beyond its table");
      // The kernels skip the pair of a variable that indexes ("is joined in") a written-out factor.
      for (unsigned t : v.dense_vars) BLK_REQUIRE(v.lo(t) == E::one() && v.hi(t) == E::one(), "fold: a variable joined in a written-out factor has factors of its own");
      for (size_t d = 0; d < v.dense_vars.size(); d++) {
        const unsigned t = v.dense_vars[d];
        dense_ |= (u64)1 << t;
        if (t >= k && t < k + w) x_.shift[t - k] = (u8)d, x_.has_values = true;
        else high_mask_ |= (u64)1 << d;
      }
      if (x_.has_values && v.is_held()) {
        auto at = forms.tables_at.find(table);
        if (at == forms.tables_at.end()) {
          at = forms.tables_at.emplace(table, std::make_pair(forms.values_n, v.held_length)).first;
          forms.values_n += v.held_length;
        }
        BLK_REQUIRE(at->second.second == v.held_length, "fold: two Tensors give one table two lengths");
        base_ = at->second.first + v.first;
      } else if (x_.has_values) {
        auto at = forms.values_at.find(v.values.get());
        if (at == forms.values_at.end()) {
          at = forms.values_at.emplace(v.values.get(), forms.values_n).first;
          forms.values_n += v.values->size();
        }
        base_ = at->second + v.first;
      } else if (v.has_dense()) { // the factor's entries go into the coefficients
        static_assert(sizeof(E) == 3 * sizeof(u64), "an element lies in a table as three words");
        const u64 entries = (u64)1 << v.dense_vars.size();
        const Forms::Slice slice{v.is_held() ? nullptr : v.values->data() + v.first, v.is_held() ? table + 3 * v.first : nullptr, entries, forms.slices_n};
        auto at = forms.slice_at.find({v.is_held() ? (const void *)slice.table : (const void *)slice.host, entries});
        if (at == forms.slice_at.end()) {
          at = forms.slice_at.emplace(std::make_pair(v.is_held() ? (const void *)slice.table : (const void *)slice.host, entries), slice.at).first;
          forms.slices.push_back(slice);
          forms.slices_n += entries, forms.slices_held += v.is_held() ? entries : 0;
        }
        slice_at_ = at->second;
      }
    }
    unsigned w() const { return w_; }
    // Calls emit(slab, number of a V) for every V that the Tensor may need on a slab of 2^log_slab
    // places.
    template <class Emit> void reach(unsigned log_slab, Emit emit) {
      if (w_ == 0 || is_zero(v_.scalar)) return;
      std::vector<u64> now = {0}, next;
      for (unsigned t = 0; t < v_.vars(); t++) {
        if (t >= k_ && t < k_ + w_) continue;
        const bool lo_zero = is_zero(v_.lo(t)), hi_zero = is_zero(v_.hi(t));
        if (lo_zero && hi_zero) return;
        // Skipped: a variable that selects neither slab nor V.
        if (t < k_ + log_slab && !(x_.has_values && ((dense_ >> t) & 1))) continue;
        next.clear();
        for (u64 j : now) {
          if (!lo_zero) next.push_back(j);
          if (!hi_zero) next.push_back(j | ((u64)1 << t));
        }
        now.swap(next);
      }
      for (u64 j : now) emit((j >> k_) >> log_slab, form(d_high(j)));
    }
    // Fills c, from which the GPU writes out the Tensor's coefficients (begin_at is left to
    // send_cuts()), and appends the open variables' pairs to `factors` and its V's numbers to
    // `forms`. False: the Tensor is 0, and nothing was appended.
    bool describe(FoldCut &c, unsigned chunk, std::vector<E> &factors, std::vector<u32> &forms) {
      if (is_zero(v_.scalar)) return false;
      c = FoldCut{};
      c.scalar = v_.scalar, c.values_at = slice_at_, c.factors_at = (u32)(factors.size() / 2), c.forms_at = (u32)forms.size();
      c.w = (u8)w_, c.chunk = (u8)chunk, c.many_forms = x_.has_values, c.in_coefficient = v_.has_dense() && !x_.has_values;
      u8 d_of[64];
      for (u8 &d : d_of) d = NO_SHIFT;
      for (size_t i = 0; i < v_.dense_vars.size(); i++)
        if ((high_mask_ >> i) & 1) d_of[v_.dense_vars[i]] = (u8)i;
      std::vector<u64> d_bits; // bit i of `together` below is this bit of the factor's index
      for (int others = 0; others < 2; others++) // the joined variables (see FoldCut) first
        for (unsigned t = 0; t < v_.vars(); t++) {
          if ((t >= k_ && t < k_ + w_) || (t < k_ && !(x_.has_values && d_of[t] != NO_SHIFT)) != (others == 0)) continue;
          const E lo = v_.lo(t), hi = v_.hi(t);
          const bool lo_zero = is_zero(lo), hi_zero = is_zero(hi);
          if (lo_zero && hi_zero) return factors.resize(2 * (size_t)c.factors_at), false;
          if (lo_zero || hi_zero) {
            c.index |= (u64)lo_zero << t, c.scalar = c.scalar * (lo_zero ? hi : lo);
            continue;
          }
          BLK_REQUIRE(c.open < FOLD_MOST_OPEN, "fold: a Tensor with more open variables than a cut has room for");
          c.var[c.open] = (u8)t, c.d_shift[c.open] = d_of[t], c.c_shift[c.open] = 0;
          if (d_of[t] != NO_SHIFT) c.c_shift[c.open] = (u8)d_bits.size(), d_bits.push_back((u64)1 << d_of[t]);
          factors.push_back(lo), factors.push_back(hi);
          c.open++, c.joined += others == 0, c.joined_in_factor = c.joined_in_factor || (others == 0 && d_of[t] != NO_SHIFT);
        }
      BLK_REQUIRE(!x_.has_values || d_bits.size() <= 24, "fold: a Tensor with more than 2^24 vectors V");
      for (u64 together = 0; together < (x_.has_values ? (u64)1 << d_bits.size() : 1); together++) {
        u64 d = 0;
        for (size_t i = 0; i < d_bits.size(); i++)
          if ((together >> i) & 1) d |= d_bits[i];
        forms.push_back(form(d));
      }
      return true;
    }

  private:
    // d_high for the entry `index` of the Tensor.
    u64 d_high(u64 index) const {
      u64 d = 0;
      for (size_t i = 0; i < v_.dense_vars.size(); i++)
        if ((high_mask_ >> i) & 1) d |= ((index >> v_.dense_vars[i]) & 1) << i;
      return d;
    }
    // The number of the V for d_high = d. A V whose key no Tensor has had yet is added to forms_.
    u32 form(u64 d) {
      if (w_ == 0) return 0;
      if (one_known_) return the_one_;
      auto mine = form_of_.find(d);
      if (mine != form_of_.end()) return mine->second;
      std::string whole = key_;
      x_.values_at = base_ + d;
      if (x_.has_values) whole.append((const char *)&x_.values_at, sizeof x_.values_at).append((const char *)x_.shift, sizeof x_.shift);
      auto found = forms_.vector_of.find(whole);
      if (found == forms_.vector_of.end()) {
        x_.factors_at = (u32)(forms_.factors.size() / 2), x_.at = forms_.store_words;
        forms_.store_words += (u64)3 << w_;
        x_.has_factors = false;
        for (unsigned t = 0; t < w_; t++) {
          forms_.factors.push_back(v_.lo(k_ + t)), forms_.factors.push_back(v_.hi(k_ + t));
          x_.has_factors = x_.has_factors || (x_.shift[t] == NO_SHIFT && !(v_.lo(k_ + t) == E::one() && v_.hi(k_ + t) == E::one()));
        }
        found = forms_.vector_of.emplace(whole, (u32)forms_.vectors.size()).first;
        forms_.vectors.push_back(x_);
      }
      if (!x_.has_values) one_known_ = true, the_one_ = found->second;
      else form_of_.emplace(d, found->second);
      return found->second;
    }
    const Tensor &v_;
    unsigned k_, w_;
    Forms &forms_;
    std::string key_;
    FoldVector x_;
    u64 dense_ = 0;     // the variables that index the written-out factor,
    u64 high_mask_ = 0; // the bits of the factor's index that come from variables outside V,
    u64 base_ = 0;      // and where the factor's slice begins on the GPU,
    u64 slice_at_ = 0;  // or, if its entries go into the coefficients, among Forms::slices
    std::unordered_map<u64, u32> form_of_; // by d_high
    bool one_known_ = false;
    u32 the_one_ = 0; // the V of a Tensor that has only one
  };
  // Cuts on the host, and (d_..) their copies on the GPU.
  struct Cuts {
    std::vector<FoldCut> cuts;
    std::vector<E> factors;
    std::vector<u32> forms;
    // Where a cut's coefficients begin among all; the same for its sums over the joined variables.
    std::vector<u32> first = {0}, first_joined = {0};
    // By w > 0: how many of these sums each piece of 2^w places has.
    std::map<unsigned, std::vector<u32>> sums_on;
    std::vector<std::pair<unsigned, u32>> lengths; // (w, where its pieces begin in d_begin)
    Pooled<FoldCut> d_cuts;
    Pooled<E> d_factors, d_values;
    // d_begin: for each w, for each piece and one more: where the piece's sums begin among all.
    Pooled<u32> d_forms, d_first, d_first_joined, d_begin;
    FoldCuts on_card() const { return FoldCuts{d_cuts.get(), (u32)cuts.size(), d_factors.get(), d_values.get(), d_forms.get()}; }
  };
  // Appends a Tensor's cut, if describe() gives one. k: see width_from(). count, cell_count (both
  // null, or neither): one counter for each row of c. The number of the cut's coefficients on a row
  // is added to `count` if the cut has a V, else to cell_count. joined: what the folded coefficient
  // vector needs is counted too (Cuts::sums_on).
  void add_cut(Cuts &all, Cutter &cutter, unsigned k, unsigned chunk, u32 *count, u32 *cell_count, bool joined) const {
    FoldCut c;
    // All bytes of c go to the GPU: its padding must not be whatever was on the host's stack. This
    // clears it, but describe() assigns all of c again, which need not keep it clear.
    std::memset(&c, 0, sizeof c);
    if (!cutter.describe(c, chunk, all.factors, all.forms)) return;
    BLK_REQUIRE(all.first.back() + ((u64)1 << c.open) < lazy::MAC_ROOM, "fold: more coefficients than a sum on the GPU has room for"); // (the gathering kernels and `k_fold_spread` rely on it)
    all.first.push_back(all.first.back() + ((u32)1 << c.open)), all.first_joined.push_back(all.first_joined.back() + ((u32)1 << (c.open - c.joined)));
    std::vector<unsigned> rows, places; // the open variables of either kind
    for (unsigned i = 0; i < c.open; i++) (c.var[i] < k ? rows : places).push_back(c.var[i]);
    auto spread = [](u64 e, const std::vector<unsigned> &to) {
      u64 index = 0;
      for (size_t i = 0; i < to.size(); i++) index |= ((e >> i) & 1) << to[i];
      return index;
    };
    if (count)
      for (u64 e = 0; e < ((u64)1 << rows.size()); e++) (c.w ? count : cell_count)[(c.index | spread(e, rows)) & (((u64)1 << k) - 1)] += (u32)1 << places.size();
    if (joined && c.w) {
      std::vector<u32> &on = all.sums_on[c.w];
      on.resize(ell_ >> c.w, 0);
      for (u64 e = 0; e < ((u64)1 << places.size()); e++) on[((c.index | spread(e, places)) >> k) >> c.w] += (u32)1 << (rows.size() - c.joined);
    }
    all.cuts.push_back(c);
  }
  // Call after the last add_cut(): sets the cuts' begin_at and sends all to the GPU. slices: has
  // the written-out factors' slices that the cuts refer to.
  void send_cuts(Cuts &all, const Forms &slices) const {
    std::vector<u32> begin;
    u32 sums = 0;
    for (const auto &x : all.sums_on) {
      all.lengths.push_back({x.first, (u32)begin.size()});
      for (u32 n : x.second) begin.push_back(sums), sums += n;
      begin.push_back(sums);
    }
    for (FoldCut &c : all.cuts)
      for (const auto &x : all.lengths)
        if (x.first == c.w) c.begin_at = x.second;
    up_->send(all.d_cuts, all.cuts.data(), all.cuts.size());
    up_->send(all.d_factors, all.factors.data(), all.factors.size());
    up_->send(all.d_forms, all.forms.data(), all.forms.size());
    up_->send(all.d_first, all.first.data(), all.first.size());
    up_->send(all.d_first_joined, all.first_joined.data(), all.first_joined.size());
    up_->send(all.d_begin, begin.data(), begin.size());
    all.d_values.resize(slices.slices_n);
    for (const auto &x : slices.slices) {
      if (x.host) up_->send_to(all.d_values.get() + x.at, x.host, x.entries);
      else ENGINE_CK(cudaMemcpyAsync(all.d_values.get() + x.at, x.table, x.entries * sizeof(E), cudaMemcpyDeviceToDevice, stream_));
    }
  }
  // out[place] += the cuts' share of the folded coefficient vector. eq, gamma: on the GPU, 2^k and
  // k elements. vectors, store: the V's that the cuts' forms refer to.
  void spread_cuts(const Cuts &all, const E *eq, const E *gamma, unsigned k, const FoldVector *vectors, const F *store, E *out) const {
    const u32 sums = all.first_joined.back();
    if (!sums) return;
    Pooled<u32> next(all.d_begin.size());
    Pooled<FoldItem> items(sums);
    if (next.size()) ENGINE_CK(cudaMemcpyAsync(next.get(), all.d_begin.get(), next.size() * sizeof(u32), cudaMemcpyDeviceToDevice, stream_));
    k_fold_join<<<blocks_for(sums, FOLD_THREADS), FOLD_THREADS, 0, stream_>>>(items.get(), next.get(), out, all.on_card(), all.d_first_joined.get(), eq, gamma, k);
    ENGINE_CK(cudaGetLastError());
    FoldLengths lengths{};
    for (const auto &x : all.lengths) lengths.w[lengths.n] = x.first, lengths.begin_at[lengths.n++] = x.second; // (w = 1 .. FOLD_MOST_W)
    if (lengths.n) {
      k_fold_spread<<<blocks_for(ell_, FOLD_THREADS), FOLD_THREADS, 0, stream_>>>(out, ell_, lengths, all.d_begin.get(), items.get(), vectors, store);
      ENGINE_CK(cudaGetLastError());
    }
  }
  // Releases what is kept only for write_out_folded().
  void forget_coefficients() { d_factors_.release(), d_values_.release(), d_vectors_.release(), d_entries_.release(), summed_.clear(), rest_ = Form(), held_.clear(), cuts_ = Cuts(); }
  // Sends the forms' descriptions to the GPU; with `whole` the GPU also writes out all forms into
  // d_store, by coordinates. most_w: no form has more variables (not checked).
  void make_forms(const Forms &forms, unsigned most_w, Pooled<E> &d_factors, Pooled<E> &d_values, Pooled<FoldVector> &d_vectors, Pooled<F> &d_store, bool whole = true) const {
    up_->send(d_factors, forms.factors.data(), forms.factors.size());
    d_values.resize(forms.values_n);
    for (const auto &x : forms.values_at) ENGINE_CK(cudaMemcpyAsync(d_values.get() + x.second, x.first->data(), x.first->size() * sizeof(E), cudaMemcpyHostToDevice, stream_));
    for (const auto &x : forms.tables_at) ENGINE_CK(cudaMemcpyAsync(d_values.get() + x.second.first, x.first, x.second.second * sizeof(E), cudaMemcpyDeviceToDevice, stream_));
    // The host vectors are the caller's: they have been read when this function returns.
    if (!forms.values_at.empty()) ENGINE_CK(cudaStreamSynchronize(stream_));
    up_->send(d_vectors, forms.vectors.data(), forms.vectors.size());
    if (whole) make_store(forms.store_words, most_w, d_factors, d_values, d_vectors, d_store);
  }
  // The GPU writes out all forms into d_store (k_fold_vectors()). most_w: as for make_forms().
  void make_store(u64 store_words, unsigned most_w, const Pooled<E> &d_factors, const Pooled<E> &d_values, const Pooled<FoldVector> &d_vectors, Pooled<F> &d_store) const {
    d_store.resize(store_words);
    list_launches[0] = std::max(list_launches[0], (d_vectors.size() + setting_.grid_rows - 1) / setting_.grid_rows);
    for (size_t at = 0; at < d_vectors.size(); at += setting_.grid_rows) {
      const dim3 grid(blocks_for((u64)1 << most_w, FOLD_THREADS * FOLD_RUNS), (unsigned)std::min<size_t>(setting_.grid_rows, d_vectors.size() - at));
      k_fold_vectors<<<grid, FOLD_THREADS, 0, stream_>>>(d_vectors.get() + at, d_factors.get(), d_values.get(), d_store.get(), 0);
      ENGINE_CK(cudaGetLastError());
    }
  }

  // The folded row by k_fold_combine_bytes(); false, with nothing written, if that kernel does not
  // fit the rows or the settings. weight: on the GPU, one for each row over F.
  bool combine_by_bytes(E *message, const E *weight) {
    if (setting_.way != ProductWay::BYTES || narrow_.empty() || ell_ % FOLD_BYTES_PLACES) return false;
    // A pitch of 4 words past a multiple of 32: eight rows' words meet no shared-memory bank twice.
    const u32 n = (u32)narrow_.size(), padded = (n + 7) / 8 * 8, pitch = (4 * padded + 127) / 128 * 128 + 16;
    const size_t room = fold_bytes_room(pitch, padded);
    if (room > (size_t)100 << 10) return false; // (two blocks on a multiprocessor)
    static const bool set = [] {
      ENGINE_CK_ATTRIBUTE(k_fold_combine_bytes, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 << 10);
      return true;
    }();
    (void)set;
    Pooled<u32> d_narrow, d_others;
    Pooled<signed char> bytes((size_t)32 * pitch);
    up_->send(d_narrow, narrow_.data(), narrow_.size());
    up_->send(d_others, others_.data(), others_.size());
    ENGINE_CK(cudaMemsetAsync(bytes.get(), 0, bytes.size(), stream_));
    k_fold_weight_bytes<<<blocks_for(3 * n, FOLD_THREADS), FOLD_THREADS, 0, stream_>>>(bytes.get(), pitch, weight, d_narrow.get(), n);
    ENGINE_CK(cudaGetLastError());
    // 132: the multiprocessors of an H100 SXM5.
    const u64 blocks = std::min<u64>(blocks_for(ell_, FOLD_BYTES_PLACES * (FOLD_THREADS / 32)), 2 * 132 * 4);
    k_fold_combine_bytes<<<(unsigned)blocks, FOLD_THREADS, room, stream_>>>(message, ell_, products_->rows_on_card(), weight, d_narrow.get(), n, padded, bytes.get(), pitch, d_others.get(), (u32)others_.size());
    ENGINE_CK(cudaGetLastError());
    return true;
  }
  // How a list of product forms over F on all places is summed layer by layer (k_fold_groups()). It
  // depends on the forms only, not on their weights: rows of c that have the same forms in the same
  // order share it. A group of depth t: the forms that agree in the factors of their t highest
  // variables; its children are the groups of depth t + 1 within it.
  struct Layers {
    unsigned d = 0;                    // the d highest variables go layer by layer; 0: not worth it
    std::vector<u32> order;            // the list's members, sorted by their groups
    std::vector<u64> nodes;            // by depth: the number of groups
    u64 room = 0;                      // elements of E in each of the two work spaces
    Pooled<u32> begin;                 // where each group of depth d begins in the order
    std::vector<Pooled<u32>> children; // by depth: the first child of each group of depth - 1
    std::vector<Pooled<F>> factors;    // by depth: the groups' factors
    // By depth: every group of depth - 1 has one child, or two with the factors f and -f.
    std::vector<char> butterflies;
  };
  // of: the forms' numbers among the splits. The plan's d stays 0 unless the high factors repeat so
  // much that layers take at most an eighth of the products of summing term by term.
  // What is allocated here from the pool of GPU memory (`Pooled`, pooled.cuh) should not depend on
  // drawn values. The condition aimed at: no allocation of the prover depends on a request or on a
  // sampled value (the values in question are the verifier's positions, which are public). For
  // generator rows the groups follow the bits of the drawn positions: how many draws share a
  // prefix. Therefore:
  // - Every list is sent with as many entries as there are forms, or one more; no depth has more
  //   groups than forms. The rest are zeros, which no kernel reads: a grid has nodes[..] rows of
  //   blocks.
  // - Each of the two work spaces has ell elements times the smallest power of two that fits the
  //   groups of every depth. For generator rows the first few depths, where every prefix occurs,
  //   decide that power.
  // - Of depths that cost the same the deepest is taken, so that the number of lists does not
  //   depend on the draws either. (One more layer costs at most what it saves wherever a depth has
  //   no more groups than forms.)
  // Not met: the stop below at "no more room than 8 rows over E" depends on the true number of
  // groups. Generator rows of a code of rate 1/8 or more never reach it: depth t has at most 2^t
  // groups times the inverse rate. The test against an eighth also depends on the true numbers of
  // groups; it decides between layers and k_fold_two_matrices(), which allocate differently.
  Layers plan_layers(const std::vector<u32> &of, const std::vector<FoldSplit> &splits, const std::vector<E> &split_factors) const {
    Layers plan;
    const unsigned most = std::min(log_ell_, 12u);
    auto pair = [&](size_t i, unsigned depth) { return &split_factors[2 * ((size_t)splits[of[i]].factors_at + log_ell_ - depth)]; }; // of the depth-th highest variable
    unsigned usable = 0; // so many of the highest variables, at most 12, have lo = 1 in all forms
    for (; usable < most; usable++) {
      bool ones = true;
      for (size_t i = 0; i < of.size() && ones; i++) ones = pair(i, usable + 1)[0] == E::one();
      if (!ones) break;
    }
    if (usable == 0) return plan;
    std::vector<u64> factor(of.size() * (usable + 1)); // [member][depth]
    for (size_t i = 0; i < of.size(); i++)
      for (unsigned depth = 1; depth <= usable; depth++) factor[i * (usable + 1) + depth] = pair(i, depth)[1].c[0].v;
    std::vector<u32> &order = plan.order;
    order.resize(of.size());
    for (size_t i = 0; i < of.size(); i++) order[i] = (u32)i;
    std::sort(order.begin(), order.end(), [&](u32 a, u32 b) {
      return std::lexicographical_compare(&factor[(size_t)a * (usable + 1) + 1], &factor[(size_t)(a + 1) * (usable + 1)], &factor[(size_t)b * (usable + 1) + 1], &factor[(size_t)(b + 1) * (usable + 1)]);
    });
    auto factor_of = [&](size_t i, unsigned depth) { return factor[(size_t)order[i] * (usable + 1) + depth]; };
    // node[depth][i]: the number of the group that contains member order[i], among the groups of
    // forms that agree in their `depth` highest factors.
    std::vector<std::vector<u32>> node(usable + 1, std::vector<u32>(order.size(), 0));
    std::vector<u64> &nodes = plan.nodes;
    nodes.assign(usable + 1, 1);
    for (unsigned depth = 1; depth <= usable; depth++) {
      for (size_t i = 1; i < order.size(); i++)
        node[depth][i] = node[depth][i - 1] + (node[depth - 1][i] != node[depth - 1][i - 1] || factor_of(i, depth) != factor_of(i - 1, depth));
      nodes[depth] = node[depth].back() + 1;
    }
    // Products in all: the groups on the low variables, and each layer. The depth with the fewest
    // is taken.
    unsigned d = 0;
    double least = (double)order.size() * (double)ell_, layers = 0;
    for (unsigned depth = 1; depth <= usable; depth++) {
      if (nodes[depth] * (ell_ >> depth) > 8 * ell_) break; // no more room than 8 rows over E
      layers += (double)nodes[depth] * (double)(ell_ >> depth);
      const double cost = layers + (double)order.size() * (double)(ell_ >> depth);
      if (cost <= least) least = cost, d = depth;
    }
    if (d == 0 || least * 8 > (double)order.size() * (double)ell_) return plan;
    plan.d = d;
    plan.room = ell_;
    for (unsigned depth = 1; depth <= d; depth++)
      while (plan.room < nodes[depth] << (log_ell_ - depth)) plan.room *= 2;
    std::vector<u32> begin;
    for (size_t i = 0; i < order.size(); i++)
      if (i == 0 || node[d][i] != node[d][i - 1]) begin.push_back((u32)i);
    begin.push_back((u32)order.size());
    begin.resize(order.size() + 1, 0);
    up_->send(plan.begin, begin.data(), begin.size());
    plan.children.resize(d + 1), plan.factors.resize(d + 1), plan.butterflies.assign(d + 1, 1);
    for (unsigned depth = d; depth >= 1; depth--) {
      std::vector<u32> children; // where the children of each group of depth - 1 begin
      std::vector<F> factors(nodes[depth]);
      for (size_t i = 0; i < order.size(); i++) {
        factors[node[depth][i]] = F{factor_of(i, depth)};
        if (i == 0 || node[depth - 1][i] != node[depth - 1][i - 1]) children.push_back(node[depth][i]);
      }
      children.push_back((u32)nodes[depth]);
      for (size_t p = 0; p + 1 < children.size(); p++) {
        const u32 c = children[p], n = children[p + 1] - c;
        if (n > 2 || (n == 2 && !is_zero(factors[c] + factors[c + 1]))) plan.butterflies[depth] = 0;
      }
      children.resize(order.size() + 1, 0), factors.resize(order.size(), F::zero());
      up_->send(plan.children[depth], children.data(), children.size());
      up_->send(plan.factors[depth], factors.data(), factors.size());
    }
    return plan;
  }
  // Sums the terms of `list` into a new vector, summed_.back(), stored by coordinates (see
  // FoldHeld::step). plan: plan_layers() of the terms' forms, with plan.d > 0 (not checked).
  void sum_by_layers(const Layers &plan, const std::vector<u32> &list, const FoldTerm *d_terms, const FoldSplit *d_splits, const F *d_base_tables) {
    const unsigned d = plan.d;
    std::vector<u32> order(list.size());
    for (size_t i = 0; i < list.size(); i++) order[i] = list[plan.order[i]];
    Pooled<u32> d_order;
    up_->send(d_order, order.data(), order.size());
    Pooled<E> now(plan.room), next(plan.room);
    k_fold_groups<<<dim3(blocks_for(ell_ >> d, FOLD_THREADS), (unsigned)plan.nodes[d]), FOLD_THREADS, 0, stream_>>>(now.get(), log_ell_ - d, plan.begin.get(), d_order.get(), d_terms, d_splits,
                                                                                                              d_base_tables);
    ENGINE_CK(cudaGetLastError());
    summed_.emplace_back(3 * ell_);
    for (unsigned depth = d; depth >= 1;) { // the leaves are the groups of this depth
      // One launch applies layers of one kind: up to four of butterflies, up to three of others.
      const bool butterflies = plan.butterflies[depth];
      unsigned many = 1;
      while (many < (butterflies ? 4u : 3u) && many < depth && (bool)plan.butterflies[depth - many] == butterflies) many++;
      if (butterflies) many == 4 ? some_layers<4, true>(plan, depth, now, next) : many == 3 ? some_layers<3, true>(plan, depth, now, next) : many == 2 ? some_layers<2, true>(plan, depth, now, next) : some_layers<1, true>(plan, depth, now, next);
      else many == 3 ? some_layers<3, false>(plan, depth, now, next) : many == 2 ? some_layers<2, false>(plan, depth, now, next) : some_layers<1, false>(plan, depth, now, next);
      depth -= many;
    }
  }
  // Applies the L layers above the groups of depth `depth`, which are in `now`. The result is in
  // `now` afterwards or, after the last layers (depth == L), in summed_.back().
  template <int L, bool BUTTERFLIES> void some_layers(const Layers &plan, unsigned depth, Pooled<E> &now, Pooled<E> &next) {
    static_assert(sizeof(E) == 3 * sizeof(F), "an element of E is its three coordinates");
    FoldLayers<L> a;
    for (unsigned i = 0; i < L; i++) a.begin[i] = plan.children[depth - L + 1 + i].get(), a.factor[i] = plan.factors[depth - L + 1 + i].get();
    E *out = reinterpret_cast<E *>(summed_.back().get());
    if (depth > L) out = next.get();
    k_fold_layers<L, BUTTERFLIES><<<dim3(blocks_for(ell_ >> depth, FOLD_THREADS), (unsigned)plan.nodes[depth - L]), FOLD_THREADS, 0, stream_>>>(
        out, now.get(), log_ell_ - depth, a, depth == L ? ell_ : 0);
    ENGINE_CK(cudaGetLastError());
    std::swap(now, next);
  }

  // out[place] = sum_b' eq[b'] * c[b'][place]. eq: the equality table of gamma_ (not checked).
  // out: ell elements of E on the GPU, complete when the call returns.
  void write_out_folded(const std::vector<E> &eq, E *out) {
    Form f = rest_;
    f.fix_low(gamma_.data(), k_);
    const unsigned log_slab = log_slab_;
    Forms forms;
    Cuts left; // of the Tensors that were written out: they have no row variables any more
    for (const Tensor &v : f.tensors) {
      const unsigned w = width_from(dense_mask(v), look_at(v).picks(), v.vars(), 0, log_slab);
      Cutter cutter(v, 0, w, forms);
      add_cut(left, cutter, 0, 0, nullptr, nullptr, true);
    }
    Pooled<E> d_factors, d_values, d_eq, d_gamma, d_one;
    Pooled<FoldVector> d_vectors;
    Pooled<F> d_store, d_kept_store;
    const E one = E::one();
    up_->send(d_eq, eq.data(), eq.size());
    up_->send(d_gamma, gamma_.data(), gamma_.size());
    up_->send(d_one, &one, 1);
    make_forms(forms, log_slab, d_factors, d_values, d_vectors, d_store);
    send_cuts(left, forms);
    std::vector<FoldHeld> held;
    for (const auto &h : held_)
      if (!is_zero(eq[h.first])) held.push_back(FoldHeld{h.second.vector, 0, h.second.scale * eq[h.first], h.second.step, h.second.first, h.second.stride});
    Pooled<FoldHeld> d_held;
    up_->send(d_held, held.data(), held.size());
    k_fold_spread_held<<<blocks_for(ell_, FOLD_THREADS), FOLD_THREADS, 0, stream_>>>(out, ell_, d_held.get(), (u32)held.size()); // (this one writes; the others add)
    ENGINE_CK(cudaGetLastError());
    spread_cuts(left, d_one.get(), nullptr, 0, d_vectors.get(), d_store.get(), out);
    // The ways PIECES and CELLS.
    if (d_vectors_.size()) make_store(store_words_, log_slab, d_factors_, d_values_, d_vectors_, d_kept_store);
    spread_cuts(cuts_, d_eq.get(), d_gamma.get(), k_, d_vectors_.get(), d_kept_store.get(), out);
    BLK_REQUIRE(f.entries.empty(), "fold: a single entry among the Tensors that were written out");
    if (d_entries_.size()) {
      k_fold_spread_single<<<blocks_for(d_entries_.size(), FOLD_THREADS), FOLD_THREADS, 0, stream_>>>(out, d_entries_.get(), (u32)d_entries_.size(), d_eq.get());
      ENGINE_CK(cudaGetLastError());
    }
    ENGINE_CK(cudaStreamSynchronize(stream_));
  }

  // Computes the Gram matrix G into gram_. held_vectors: the constructor's `held`. Slab, V, cut,
  // slot, column and the ways PIECES, WRITTEN OUT, CELLS: see "How G is made" above. It also fills
  // what write_out_folded() reads; if FoldSetting::folded_coefficients is false,
  // forget_coefficients() releases that before the return.
  void make_gram(const Form &c, const std::vector<const u64 *> &held_vectors) {
    const auto started = std::chrono::steady_clock::now();
    const u64 all = (u64)1 << k_, row_mask = all - 1;
    const unsigned log_slab = log_slab_;
    const u64 width = (u64)1 << log_slab, slabs = ell_ >> log_slab;
    const u32 n_rows = products_->rows();
    gram_.assign(all * all, E::zero());
    // table_of[i]: the table with Tensor i's written-out factor, if the engine holds it.
    std::vector<const u64 *> table_of(c.tensors.size(), nullptr);
    {
      size_t next = c.held.size() + c.engine_terms.size();
      for (size_t i = 0; i < c.tensors.size(); i++)
        if (c.tensors[i].is_held()) table_of[i] = held_vectors[next++];
    }

    // Slots are given in order of first use.
    std::vector<u32> slot_of(all, ~(u32)0), row_of_slot;
    auto slot = [&](u64 row) {
      if (slot_of[row] == ~(u32)0) slot_of[row] = (u32)row_of_slot.size(), row_of_slot.push_back((u32)row);
      return slot_of[row];
    };

    // Parts of c that are read from a vector on the GPU: hints (c.held) and engine's terms
    // (c.engine_terms). The vector is indexed by all place variables and by the row variables
    // outside the part's choice_mask; each value of those gives a written-out row of c. Nothing
    // checks that a hint's vector is the sum that the hint states.
    std::vector<char> tensor_done(c.tensors.size(), 0), entry_done(c.entries.size(), 0);
    std::vector<u32> column_slot; // the written-out rows of c
    std::map<u32, u32> column_of; // by slot
    auto column = [&](u32 s) {
      auto at = column_of.find(s);
      if (at == column_of.end()) at = column_of.emplace(s, (u32)column_slot.size()).first, column_slot.push_back(s);
      return at->second;
    };
    std::vector<FoldHeld> held;
    // Enters scale * the vector as written-out rows of c.
    auto stands_in = [&](const u64 *vector, const E &scale, u64 choice_mask, u64 choice_value) {
      if (is_zero(scale)) return;
      std::vector<unsigned> open; // the row variables that index the vector
      for (unsigned t = 0; t < k_; t++)
        if (!((choice_mask >> t) & 1)) open.push_back(t);
      for (u64 a = 0; a < ((u64)1 << open.size()); a++) {
        u64 row = choice_value;
        for (size_t t = 0; t < open.size(); t++) row |= ((a >> t) & 1) << open[t];
        held.push_back(FoldHeld{vector, column(slot(row)), scale, 0, a, (u64)1 << open.size()});
        held_.push_back({row, held.back()});
      }
    };
    for (size_t i = 0; i < c.held.size(); i++) {
      const Held &h = c.held[i];
      BLK_REQUIRE(h.from <= h.to && h.to <= c.tensors.size() && h.entries_from <= h.entries_to && h.entries_to <= c.entries.size(), "fold: a hint names parts that the coefficient vector has not");
      BLK_REQUIRE(!(h.choice_value & ~h.choice_mask), "fold: a hint gives a value to a variable that it does not pick");
      if (!held_vectors[i] || (h.choice_mask & ~row_mask)) continue;
      for (size_t t = h.from; t < h.to; t++) {
        BLK_REQUIRE(!tensor_done[t], "fold: two hints name the same Tensor"); // else counted twice
        tensor_done[t] = 1;
      }
      for (size_t e = h.entries_from; e < h.entries_to; e++) {
        BLK_REQUIRE(!entry_done[e], "fold: two hints name the same entry");
        entry_done[e] = 1;
      }
      stands_in(held_vectors[i], h.scale, h.choice_mask, h.choice_value);
    }
    for (size_t i = 0; i < c.engine_terms.size(); i++) {
      const EngineTerm &t = c.engine_terms[i];
      BLK_REQUIRE(held_vectors[c.held.size() + i], "fold: the vector of a term that only the engine knows is not at hand, or has not the length that the term asks for");
      BLK_REQUIRE(!(t.choice_mask & ~row_mask), "fold: a term that only the engine knows lives on a part of the places: that is not built");
      BLK_REQUIRE(!(t.choice_value & ~t.choice_mask), "fold: a term gives a value to a variable that it does not pick");
      stands_in(held_vectors[c.held.size() + i], t.scale, t.choice_mask, t.choice_value);
    }

    // A Tensor that has no written-out factor and picks a bit in every variable is treated as a
    // single entry.
    const u64 all_vars = ((u64)1 << (k_ + log_ell_)) - 1, place_mask = all_vars & ~row_mask;
    std::vector<Look> looks(c.tensors.size());
    std::vector<std::pair<u64, E>> picked;
    for (size_t i = 0; i < c.tensors.size(); i++) {
      const Tensor &v = c.tensors[i];
      if (tensor_done[i]) continue;
      const Look &x = looks[i] = look_at(v);
      if (v.has_dense() || x.picks() != all_vars) continue;
      picked.push_back({x.zero_lo, v.scalar * picked_factors(v, x, all_vars)});
      tensor_done[i] = 1;
    }

    auto since = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(); };
    seconds_cut_part[0] = since();
    std::vector<unsigned> w_of(c.tensors.size(), 0);
    std::vector<i64> single_row(c.tensors.size(), -1);
    std::vector<char> bulk(c.tensors.size(), 0); // a product form over F on all places of one row
    for (size_t i = 0; i < c.tensors.size(); i++) {
      const Tensor &v = c.tensors[i];
      if (tensor_done[i]) continue;
      const Look &x = looks[i];
      w_of[i] = width_from(dense_mask(v), x.picks(), v.vars(), k_, log_slab);
      const bool one_row = !v.has_dense() && (x.picks() & row_mask) == row_mask;
      if (one_row) single_row[i] = (i64)(x.zero_lo & row_mask);
      bulk[i] = one_row && w_of[i] > 0 && !((x.zero_lo | x.zero_hi) & place_mask) && (x.in_base & place_mask) == place_mask;
    }
    // The address of the pairs of a `bulk` Tensor's place variables (none of them picks a bit).
    // Tensors that were copied from one another share them.
    auto place_pairs = [&](const Tensor &v) { return v.real_pairs() + 2 * __builtin_popcountll(~v.pick_mask() & row_mask); };
    // The `bulk` Tensors of a row that has at least MANY_TERMS of them are written out even if
    // their forms serve other rows too, as those of generator rows at one list of positions on
    // several rows do, unless there are more such rows than forms: a written-out row costs as much
    // as a form. (Forms are counted by address: equal forms at two addresses count twice, which can
    // change the way taken but no value.)
    {
      std::vector<size_t> on_row(all, 0);
      for (size_t i = 0; i < c.tensors.size(); i++)
        if (bulk[i]) on_row[(u64)single_row[i]]++;
      std::vector<const E *> distinct;
      size_t rows_with_many = 0;
      for (size_t n : on_row) rows_with_many += n >= MANY_TERMS;
      for (size_t i = 0; i < c.tensors.size(); i++) {
        if (bulk[i] && on_row[(u64)single_row[i]] < MANY_TERMS) bulk[i] = 0;
        if (bulk[i]) distinct.push_back(place_pairs(c.tensors[i]));
      }
      std::sort(distinct.begin(), distinct.end());
      if (rows_with_many > (size_t)(std::unique(distinct.begin(), distinct.end()) - distinct.begin())) std::fill(bulk.begin(), bulk.end(), 0);
    }
    std::vector<std::string> key(c.tensors.size());
    struct Use {
      u64 row = 0;
      bool many_rows = false;
    };
    std::map<std::string, Use> uses; // which rows of c a form serves
    for (size_t i = 0; i < c.tensors.size(); i++) {
      if (tensor_done[i] || bulk[i]) continue;
      key[i] = key_of(c.tensors[i], k_, w_of[i]);
      const bool one_row = single_row[i] >= 0;
      const u64 row = one_row ? (u64)single_row[i] : 0;
      auto at = uses.find(key[i]);
      if (at == uses.end()) uses.emplace(key[i], Use{row, !one_row});
      else if (!one_row || at->second.row != row) at->second.many_rows = true;
    }
    // own_forms[row]: the forms that serve only that row of c (WRITTEN OUT).
    std::map<u64, std::map<std::string, char>> own_forms;
    for (size_t i = 0; i < c.tensors.size(); i++)
      if (!tensor_done[i] && !bulk[i] && single_row[i] >= 0 && w_of[i] > 0 && !uses[key[i]].many_rows) own_forms[(u64)single_row[i]][key[i]] = 1;

    seconds_cut_part[1] = since() - seconds_cut_part[0];
    Forms forms;
    std::vector<E> split_factors;
    std::vector<FoldSplit> splits;
    std::vector<FoldTerm> terms;
    std::map<std::string, u32> split_of;
    std::unordered_map<const E *, u32> split_at; // `bulk` forms' splits, by their pairs' address
    u64 table_words = 0, base_table_words = 0;
    std::vector<std::vector<u32>> needed(slabs);  // the forms that a slab needs
    std::vector<std::unique_ptr<Cutter>> cutters; // the Tensors that go the ways PIECES and CELLS
    std::vector<size_t> term_tensor;              // the Tensor that a term came from
    auto new_split = [&](unsigned w, bool in_base) { // its factors: the last 2 w of split_factors
      FoldSplit x;
      x.factors_at = (u32)(split_factors.size() / 2 - w), x.w = w, x.w_low = w / 2, x.in_base = in_base;
      u64 &words = in_base ? base_table_words : table_words;
      x.low_at = words, x.high_at = words + ((u64)1 << x.w_low);
      words = x.high_at + ((u64)1 << (w - x.w_low));
      splits.push_back(x);
      return (u32)(splits.size() - 1);
    };
    std::vector<u32> column_of_row(all, ~(u32)0);
    auto column_on = [&](u64 row) {
      if (column_of_row[row] == ~(u32)0) column_of_row[row] = column(slot(row));
      return column_of_row[row];
    };

    for (size_t i = 0; i < c.tensors.size(); i++) {
      const Tensor &v = c.tensors[i];
      if (tensor_done[i] || is_zero(v.scalar)) continue;
      const Look &x = looks[i];
      if (bulk[i]) {
        const E *pairs = place_pairs(v);
        auto found = split_at.find(pairs);
        if (found == split_at.end()) {
          split_factors.insert(split_factors.end(), pairs, pairs + 2 * log_ell_);
          found = split_at.emplace(pairs, new_split(log_ell_, true)).first;
        }
        terms.push_back(FoldTerm{found->second, log_ell_, 0, column_on((u64)single_row[i]), v.scalar * picked_factors(v, x, row_mask)});
        term_tensor.push_back(i);
        continue;
      }
      if (single_row[i] >= 0 && w_of[i] > 0 && !uses[key[i]].many_rows && own_forms[(u64)single_row[i]].size() >= 2) {
        // The highest place variables that pick a bit select a piece; the form on all variables
        // below them is kept as two tables (FoldSplit).
        unsigned w = log_ell_;
        while (w > 0 && ((x.picks() >> (k_ + w - 1)) & 1)) w--;
        const u64 above = all_vars & ~(((u64)1 << (k_ + w)) - 1); // these variables
        const std::string whole = key_of(v, k_, w);
        auto found = split_of.find(whole);
        if (found == split_of.end()) {
          for (unsigned t = 0; t < w; t++) split_factors.push_back(v.lo(k_ + t)), split_factors.push_back(v.hi(k_ + t));
          const u64 below = place_mask & ~above;
          found = split_of.emplace(whole, new_split(w, (x.in_base & below) == below)).first;
        }
        terms.push_back(FoldTerm{found->second, w, (x.zero_lo & above) >> (k_ + w), column_on((u64)single_row[i]), v.scalar * picked_factors(v, x, row_mask | above)});
        term_tensor.push_back(i);
        continue;
      }
      // PIECES and CELLS. For now the host finds only which V the Tensor needs on which slab: that
      // is enough to start the GPU. add_cut(), below, describes its coefficients meanwhile.
      const double before = since();
      cutters.push_back(std::make_unique<Cutter>(v, k_, w_of[i], forms, table_of[i]));
      tensor_done[i] = 1; // its cut is kept (cuts_), not the Tensor
      cutters.back()->reach(log_slab, [&](u64 s, u32 form) { needed[s].push_back(form); });
      seconds_cut_part[3] += since() - before;
    }
    seconds_cut_part[2] = since() - seconds_cut_part[0] - seconds_cut_part[1] - seconds_cut_part[3];

    // Span: the whole slabs that the GPU works on at a time (FoldSetting::span_bytes,
    // most_log_span). A written-out row takes up to 3 * 16 bytes a place: each coordinate's word
    // and its 8 signed bytes.
    const u32 columns = (u32)column_slot.size();
    unsigned log_span = log_slab;
    while (log_span < std::min(log_ell_, setting_.most_log_span) && ((products_->plane_rows() + (u64)columns * 3 * 16) << (log_span + 1)) <= setting_.span_bytes) log_span++;
    const u64 span_width = (u64)1 << log_span, spans = ell_ >> log_span;
    std::vector<std::vector<u32>> lists(spans); // the terms that are written out, by span,
    std::vector<char> has_terms(slabs, 0);      // and whether a slab has any

    // together[column]: the terms of that row that are summed beforehand, by sum_by_layers() or
    // k_fold_two_matrices(). Why log_ell_ <= 28: the left matrix has
    // 3 * 2^(log_ell_ - log_ell_ / 2) rows, of which CardProducts takes 65,535. Why at most 65,535
    // terms: one launch of k_fold_groups() takes so many groups (for the two matrices, CardProducts
    // takes 2^16 terms, one more).
    std::vector<std::vector<u32>> together(column_slot.size());
    for (u32 t = 0; t < terms.size(); t++)
      if (splits[terms[t].split].in_base && terms[t].w == log_ell_ && log_ell_ >= 4 && log_ell_ <= 28) together[terms[t].column].push_back(t);
    for (auto &list : together)
      if (list.size() < MANY_TERMS || list.size() > 65535) list.clear();
    {
      std::vector<char> taken(terms.size(), 0);
      for (const auto &list : together)
        for (u32 t : list) taken[t] = 1, tensor_done[term_tensor[t]] = 1;
      for (u32 t = 0; t < terms.size(); t++) {
        if (taken[t]) continue;
        const u64 from = terms[t].piece << terms[t].w, to = ((terms[t].piece + 1) << terms[t].w) - 1;
        for (u64 s = from >> log_slab; s <= to >> log_slab; s++) has_terms[s] = 1;
        for (u64 s = from >> log_span; s <= to >> log_span; s++) lists[s].push_back(t);
      }
    }
    for (size_t t = 0; t < c.tensors.size(); t++)
      if (!tensor_done[t]) rest_.tensors.push_back(c.tensors[t]);
    for (size_t e = 0; e < c.entries.size(); e++) // into `picked`; an index may occur many times
      if (!entry_done[e]) picked.push_back(c.entries[e]);

    seconds_cut = since();
    forms_made = forms.vectors.size(), written_rows = columns;

    BLK_REQUIRE(terms.size() < lazy::MAC_ROOM, "fold: more written-out terms than a sum on the GPU has room for"); // `k_fold_write_out` relies on it
    Pooled<E> d_split_factors, d_tables(table_words);
    Pooled<F> d_store, d_base_tables(base_table_words);
    Pooled<FoldSplit> d_splits;
    Pooled<FoldTerm> d_terms;
    Pooled<FoldHeld> d_held;
    held_tables_copied = forms.tables_at.size();
    // A form of 2^w entries is multiplied in chunks of 2^chunk_of(w) places.
    auto chunk_of = [&](unsigned w) { return w >= setting_.log_chunk ? setting_.log_chunk : w >= setting_.log_short_chunk ? setting_.log_short_chunk : w; };
    // If all forms are multiplied by bytes, only their bytes are kept: the forms are then written
    // out some at a time, so that only one group's words (FoldSetting::store_words) are in memory
    // beside the bytes of all.
    bool some_at_a_time = true;
    for (const FoldVector &v : forms.vectors) some_at_a_time = some_at_a_time && products_->by_bytes(chunk_of(v.w));
    make_forms(forms, log_slab, d_factors_, d_values_, d_vectors_, d_store, !some_at_a_time);
    store_words_ = forms.store_words;
    up_->send(d_split_factors, split_factors.data(), split_factors.size());
    up_->send(d_splits, splits.data(), splits.size());
    up_->send(d_terms, terms.data(), terms.size());
    if (!splits.empty()) {
      u64 most = 0;
      for (const FoldSplit &x : splits) most = std::max(most, ((u64)1 << x.w_low) + ((u64)1 << (x.w - x.w_low)));
      list_launches[1] = std::max(list_launches[1], (splits.size() + setting_.grid_rows - 1) / setting_.grid_rows);
      for (size_t at = 0; at < splits.size(); at += setting_.grid_rows) {
        const dim3 grid(blocks_for(most, FOLD_THREADS), (unsigned)std::min<size_t>(setting_.grid_rows, splits.size() - at));
        k_fold_splits<<<grid, FOLD_THREADS, 0, stream_>>>(d_splits.get() + at, d_split_factors.get(), d_tables.get(), d_base_tables.get());
        ENGINE_CK(cudaGetLastError());
      }
    }

    if (setting_.trace) ENGINE_CK(cudaStreamSynchronize(stream_));
    seconds_forms = since() - seconds_cut;
    // Sums each list of `together` into a vector of summed_, which then counts as a held vector.
    std::map<std::vector<u32>, Layers> plans; // by the forms' numbers
    for (u32 j = 0; j < columns; j++) {
      const std::vector<u32> &list = together[j];
      if (list.empty()) continue;
      const double t0 = products_->now();
      std::vector<u32> of(list.size());
      for (size_t i = 0; i < list.size(); i++) of[i] = terms[list[i]].split;
      auto plan = plans.find(of);
      if (plan == plans.end()) plan = plans.emplace(of, plan_layers(of, splits, split_factors)).first;
      if (plan->second.d) {
        sum_by_layers(plan->second, list, d_terms.get(), d_splits.get(), d_base_tables.get());
        held.push_back(FoldHeld{reinterpret_cast<const u64 *>(summed_.back().get()), j, E::one(), ell_, 0, 1});
        held_.push_back({row_of_slot[column_slot[j]], held.back()});
        seconds_write_out += products_->now() - t0;
        rows_summed_by_layers++;
        continue;
      }
      // Layers::d is 0: the sum is the product of the two matrices of k_fold_two_matrices().
      const unsigned w_low = log_ell_ / 2, log_room = std::max(2u, ceil_log2(list.size()));
      const u64 n_low = (u64)1 << w_low, n_high = ell_ >> w_low, room = (u64)1 << log_room;
      Pooled<u32> d_together;
      up_->send(d_together, list.data(), list.size());
      Pooled<F> left(3 * n_high * room), right(n_low * room);
      k_fold_two_matrices<<<blocks_for((n_high + n_low) * room, FOLD_THREADS), FOLD_THREADS, 0, stream_>>>(left.get(), right.get(), n_high, n_low, room, d_together.get(),
                                                                                                      (u32)list.size(), d_terms.get(), d_splits.get(), d_base_tables.get());
      ENGINE_CK(cudaGetLastError());
      std::vector<ProductRow> left_rows;
      for (u64 r = 0; r < 3 * n_high; r++) left_rows.push_back(ProductRow{left.get(), r * room, 1, Cell::WORD});
      CardProducts matrices(left_rows, log_room, setting_.way, stream_);
      matrices.raw_words = setting_.raw_words;
      ProductRight ready;
      matrices.slab(0);
      summed_.emplace_back(3 * ell_);
      matrices.prepare(ready, right.get(), (u32)n_low);
      matrices.multiply(ready, summed_.back().get());
      rows_summed_by_matrices++;
      held.push_back(FoldHeld{reinterpret_cast<const u64 *>(summed_.back().get()), j, E::one(), ell_, 0, 1});
      held_.push_back({row_of_slot[column_slot[j]], held.back()});
      seconds_write_out += products_->now() - t0;
    }
    // A written-out row that is just one vector stored by coordinates, with scale 1, is not written
    // out again: the vector itself is multiplied.
    std::vector<const F *> whole(columns, nullptr);
    {
      std::vector<unsigned> parts(columns, 0);
      for (const auto &list : lists)
        for (u32 t : list) parts[terms[t].column] = 2;
      for (const FoldHeld &h : held) parts[h.column]++;
      std::vector<FoldHeld> others;
      for (const FoldHeld &h : held) {
        if (parts[h.column] == 1 && h.step == ell_ && h.scale == E::one()) whole[h.column] = reinterpret_cast<const F *>(h.vector), rows_taken_whole++;
        else others.push_back(h);
      }
      held.swap(others);
    }
    bool some_whole = false, some_written = false;
    for (const F *x : whole) some_whole = some_whole || x, some_written = some_written || !x;
    up_->send(d_held, held.data(), held.size());
    if (setting_.trace) ENGINE_CK(cudaStreamSynchronize(stream_));
    seconds_summing = since() - seconds_cut - seconds_forms;

    // The plan. On each slab the forms that it needs are multiplied with all rows of m; a
    // written-out row of c counts as a form as long as the slab. Columns (TiledColumn: one
    // coordinate of a form) with chunks of one length share tiles (tiled.cuh). Every (slab, form)
    // has a separate range for its sums, so that one launch can gather them all into G.
    constexpr u64 NOWHERE = ~(u64)0;
    Pooled<F> written(some_written ? (size_t)columns * 3 * span_width : 0);
    // Signed bytes, 8 a word: the forms', in the store's order; the written-out rows', on a span.
    Pooled<signed char> d_bytes, written_bytes;
    std::vector<TiledSource> sources, written_sources;
    u64 longest = 0;
    std::vector<size_t> group = {0}; // forms group[i] .. group[i + 1] - 1 are written out together
    if (some_at_a_time) {
      u64 most = 0;
      for (size_t i = 0; i < forms.vectors.size(); i++) {
        const u64 end = forms.vectors[i].at + ((u64)3 << forms.vectors[i].w), first = forms.vectors[group.back()].at;
        if (i > group.back() && end - first <= setting_.store_words && i - group.back() == setting_.grid_rows) list_launches[3] = 2;
        if (i > group.back() && (end - first > setting_.store_words || i - group.back() == setting_.grid_rows)) group.push_back(i);
        most = std::max(most, end - forms.vectors[group.back()].at);
      }
      group.push_back(forms.vectors.size());
      d_store.resize(most);
      form_groups = forms.vectors.empty() ? 0 : group.size() - 1;
    }
    for (size_t i = 0, g = 0; i < forms.vectors.size(); i++) {
      const FoldVector &v = forms.vectors[i];
      if (!products_->by_bytes(chunk_of(v.w))) continue;
      while (some_at_a_time && i >= group[g + 1]) g++;
      const u64 less = some_at_a_time ? forms.vectors[group[g]].at : 0;
      d_bytes.resize(8 * forms.store_words);
      longest = std::max(longest, (u64)1 << v.w);
      for (u64 d = 0; d < 3; d++) sources.push_back(TiledSource{d_store.get() + v.at - less + (d << v.w), d_bytes.get() + 8 * (v.at + (d << v.w)), (u64)1 << v.w, (u64)1 << v.w});
    }
    BLK_REQUIRE((std::uintptr_t)d_bytes.get() % 16 == 0, "fold: a work space that does not begin at a multiple of 16");
    if (columns && products_->by_bytes(chunk_of(log_slab))) {
      written_bytes.resize((size_t)columns * 3 * 8 * span_width);
      BLK_REQUIRE((std::uintptr_t)written_bytes.get() % 16 == 0, "fold: a work space that does not begin at a multiple of 16");
      for (u64 sp = 0; sp < spans; sp++)
        for (u64 j = 0; j < columns; j++)
          for (u64 d = 0; d < 3; d++)
            written_sources.push_back(TiledSource{whole[j] ? whole[j] + d * ell_ + sp * span_width : written.get() + (j * 3 + d) * span_width,
                                                  written_bytes.get() + (j * 3 + d) * 8 * span_width, span_width, span_width});
    }
    std::vector<TiledColumn> tiled;
    std::vector<TiledTile> tiles;
    std::vector<TiledWork> work;                         // the by-bytes tiles' chunks, by span
    std::vector<size_t> work_at(spans + 1, 0);           // where a span's work begins
    std::vector<std::vector<u32>> plain_tiles(spans);    // tiles multiplied in field arithmetic
    std::vector<char> span_writes(spans, 0);             // some slab uses the written-out rows
    std::vector<std::vector<u64>> sums_at(slabs);        // [slab][form]: where the sums begin
    std::vector<std::pair<u32, FoldItem>> written_items; // (slot, item) for the written-out rows
    std::vector<FoldStretch> stretches;                  // sums of forms longer than their chunks
    std::vector<u32> flat_list;
    std::vector<size_t> list_at(spans);
    u64 sum_words = 0;
    // k_fold_write_out() needs a span's terms sorted by column.
    for (u64 sp = 0; sp < spans; sp++) {
      list_at[sp] = flat_list.size();
      std::stable_sort(lists[sp].begin(), lists[sp].end(), [&](u32 a, u32 b) { return terms[a].column < terms[b].column; });
      flat_list.insert(flat_list.end(), lists[sp].begin(), lists[sp].end());
    }
    for (u64 s = 0; s < slabs; s++) {
      const u64 sp = (s << log_slab) >> log_span, in_span = (s << log_slab) & (span_width - 1);
      if (in_span == 0) work_at[sp] = work.size();
      const bool write = columns && (has_terms[s] || !held.empty() || some_whole);
      std::vector<u32> &list = needed[s];
      std::sort(list.begin(), list.end());
      list.erase(std::unique(list.begin(), list.end()), list.end());
      if (list.empty() && !write) continue;
      sums_at[s].assign(forms.vectors.size(), NOWHERE);
      std::map<unsigned, std::vector<TiledColumn>> by_chunk;
      for (u32 f : list) {
        const FoldVector &v = forms.vectors[f];
        sums_at[s][f] = sum_words;
        for (u64 d = 0; d < 3; d++)
          by_chunk[chunk_of(v.w)].push_back(TiledColumn{some_at_a_time ? nullptr : d_store.get() + v.at + (d << v.w), d_bytes.size() ? d_bytes.get() + 8 * (v.at + (d << v.w)) : nullptr, (u64)1 << v.w, sum_words + d, v.w, 3});
        if (v.w > chunk_of(v.w)) stretches.push_back(FoldStretch{sum_words, (u32)(width >> chunk_of(v.w)), (u32)1 << (v.w - chunk_of(v.w))});
        sum_words += (u64)n_rows * (width >> chunk_of(v.w)) * 3;
        pieces_worth += 1.0 / (double)slabs, steps_made[v.w].second++;
      }
      if (write) {
        span_writes[sp] = 1;
        const u64 chunks = width >> chunk_of(log_slab);
        for (u64 j = 0; j < columns; j++) {
          for (u64 d = 0; d < 3; d++)
            by_chunk[chunk_of(log_slab)].push_back(TiledColumn{whole[j] ? whole[j] + d * ell_ + s * width : written.get() + (j * 3 + d) * span_width + in_span,
                                                 written_bytes.size() ? written_bytes.get() + (j * 3 + d) * 8 * span_width + in_span : nullptr, span_width, sum_words + d, log_slab, 3});
          written_items.push_back({column_slot[j], FoldItem{sum_words, (u32)(chunks * 3), 1, E::one()}});
          if (chunks > 1) stretches.push_back(FoldStretch{sum_words, (u32)chunks, (u32)chunks});
          sum_words += (u64)n_rows * chunks * 3;
          written_worth += 1.0 / (double)slabs;
        }
      }
      for (const auto &cls : by_chunk)
        for (size_t at = 0; at < cls.second.size(); at += TILED_COLUMNS) {
          const TiledTile tile{(u32)tiled.size(), (u32)std::min<size_t>(TILED_COLUMNS, cls.second.size() - at), cls.first, in_span};
          tiled.insert(tiled.end(), cls.second.begin() + at, cls.second.begin() + at + tile.columns);
          if (products_->by_bytes(tile.log_k))
            for (u64 chunk = 0; chunk < (width >> tile.log_k); chunk++) work.push_back(TiledWork{(u32)tiles.size(), (u32)chunk});
          else plain_tiles[sp].push_back((u32)tiles.size());
          tiles.push_back(tile);
          steps_made[cls.first].first++;
        }
    }
    work_at[spans] = work.size();
    for (u64 sp = 0; sp < spans; sp++) // long chunks first: short ones then fill the launch's end
      std::stable_sort(work.begin() + work_at[sp], work.begin() + work_at[sp + 1], [&](const TiledWork &a, const TiledWork &b) { return tiles[a.tile].log_k > tiles[b.tile].log_k; });
    seconds_plan = since() - seconds_cut - seconds_forms - seconds_summing;
    Pooled<u32> d_list;
    Pooled<TiledColumn> d_tiled;
    Pooled<TiledTile> d_tiles;
    Pooled<TiledWork> d_work;
    Pooled<TiledSource> d_sources, d_written_sources;
    Pooled<FoldStretch> d_stretches;
    Pooled<F> sums(sum_words);
    up_->send(d_list, flat_list.data(), flat_list.size());
    up_->send(d_tiled, tiled.data(), tiled.size());
    up_->send(d_tiles, tiles.data(), tiles.size());
    up_->send(d_work, work.data(), work.size());
    up_->send(d_stretches, stretches.data(), stretches.size());
    up_->send(d_sources, sources.data(), sources.size());
    up_->send(d_written_sources, written_sources.data(), written_sources.size());
    if (setting_.trace) ENGINE_CK(cudaStreamSynchronize(stream_));
    const auto prepared = std::chrono::steady_clock::now();
    seconds_host = std::chrono::duration<double>(prepared - started).count();

    // The launches: the forms' bytes; then, span by span, the written-out rows, their bytes and the
    // products.
    products_->open(log_span);
    if (some_at_a_time) {
      for (size_t g = 0; g + 1 < group.size() && group[g] < group[g + 1]; g++) {
        const dim3 grid(blocks_for((u64)1 << log_slab, FOLD_THREADS * FOLD_RUNS), (unsigned)(group[g + 1] - group[g]));
        k_fold_vectors<<<grid, FOLD_THREADS, 0, stream_>>>(d_vectors_.get() + group[g], d_factors_.get(), d_values_.get(), d_store.get(), forms.vectors[group[g]].at);
        ENGINE_CK(cudaGetLastError());
        products_->bytes_of(d_sources.get() + 3 * group[g], 3 * (group[g + 1] - group[g]), longest);
      }
      d_store.release();
    } else {
      products_->bytes_of(d_sources.get(), sources.size(), longest);
    }
    for (u64 sp = 0; sp < spans; sp++) {
      if (work_at[sp] == work_at[sp + 1] && plain_tiles[sp].empty()) continue;
      products_->span(sp * span_width);
      if (some_written && span_writes[sp]) {
        const double t0 = products_->now();
        ENGINE_CK(cudaMemsetAsync(written.get(), 0, written.size() * sizeof(F), stream_));
        k_fold_write_out<<<blocks_for(span_width, FOLD_THREADS), FOLD_THREADS, 0, stream_>>>(written.get(), sp * span_width, span_width, d_list.get() + list_at[sp],
                                                                                           (u32)lists[sp].size(), d_terms.get(), d_splits.get(), d_tables.get(),
                                                                                           d_base_tables.get(), d_held.get(), (u32)held.size());
        ENGINE_CK(cudaGetLastError());
        seconds_write_out += products_->now() - t0;
      }
      if (span_writes[sp] && written_bytes.size()) products_->bytes_of(d_written_sources.get() + sp * columns * 3, (size_t)columns * 3, span_width);
      products_->multiply(d_work.get() + work_at[sp], work_at[sp + 1] - work_at[sp], d_tiles.get(), d_tiled.get(), log_slab, sums.get());
      for (u32 i : plain_tiles[sp]) products_->multiply_plain(tiles[i], d_tiled.get(), log_slab, sums.get());
    }

    u64 most_pieces = 0;
    for (const FoldStretch &x : stretches) most_pieces = std::max<u64>(most_pieces, x.chunks / x.many);
    list_launches[2] = std::max(list_launches[2], (stretches.size() + setting_.grid_rows - 1) / setting_.grid_rows);
    for (size_t at = 0; at < stretches.size(); at += setting_.grid_rows) {
      k_fold_pieces<<<dim3(blocks_for((u64)n_rows * most_pieces * 3, FOLD_THREADS), (unsigned)std::min<size_t>(setting_.grid_rows, stretches.size() - at)), FOLD_THREADS, 0, stream_>>>(sums.get(), d_stretches.get() + at, n_rows);
      ENGINE_CK(cudaGetLastError());
    }

    // While the GPU multiplies, the host describes the Tensors' cuts and counts their coefficients
    // on each row of c.
    const double launched = since();
    // Numbers of items by row of c: on sums; on cells, from cuts; on cells, from single entries.
    std::vector<u32> count(all, 0), cut_cell_count(all, 0), cell_count(all, 0);
    for (const auto &x : written_items) count[row_of_slot[x.first]]++;
    for (const auto &cutter : cutters) add_cut(cuts_, *cutter, k_, chunk_of(cutter->w()), count.data(), cut_cell_count.data(), setting_.folded_coefficients);
    for (const auto &x : picked) cell_count[x.first & row_mask]++;
    for (u64 row = 0; row < all; row++)
      if (count[row] || cut_cell_count[row] || cell_count[row]) slot(row);
    const u32 slots = (u32)row_of_slot.size();
    // No row of c has a slot: G stays zero.
    if (slots == 0) {
      ENGINE_CK(cudaStreamSynchronize(stream_));
      products_->release_work();
      if (!setting_.folded_coefficients) forget_coefficients(); // (the stream has been waited for)
      return;
    }
    // Each slot's part of the three lists of items. The host makes the items of the written-out
    // rows and of the single entries, the GPU those of the cuts (k_fold_expand()).
    std::vector<u32> begins(slots + 1, 0), cut_cell_begins(slots + 1, 0), cell_begins(slots + 1, 0);
    for (u32 j = 0; j < slots; j++) {
      const u32 row = row_of_slot[j];
      begins[j + 1] = begins[j] + count[row], cut_cell_begins[j + 1] = cut_cell_begins[j] + cut_cell_count[row], cell_begins[j + 1] = cell_begins[j] + cell_count[row];
    }
    std::vector<u32> next(begins.begin(), begins.end() - 1), fixed_at;
    std::vector<FoldItem> fixed, cell_items = Spare<FoldItem>::take();
    for (const auto &x : written_items) fixed.push_back(x.second), fixed_at.push_back(next[x.first]++);
    cell_items.resize(cell_begins[slots]);
    {
      std::vector<u32> cell_next(cell_begins.begin(), cell_begins.end() - 1);
      for (const auto &x : picked) cell_items[cell_next[slot_of[x.first & row_mask]]++] = FoldItem{x.first >> k_, (u32)(x.first & row_mask), 1, x.second}; // (row_step: the row of c, for end())
    }
    std::vector<u64> flat_sums_at(slabs * forms.vectors.size(), NOWHERE);
    for (u64 s = 0; s < slabs; s++) std::copy(sums_at[s].begin(), sums_at[s].end(), flat_sums_at.begin() + s * forms.vectors.size());
    items_made = (size_t)begins[slots] + cut_cell_begins[slots] + cell_begins[slots];
    BLK_REQUIRE(items_made < lazy::MAC_ROOM, "fold: more coefficients than a sum on the GPU has room for"); // (before the gather, which relies on it)
    held_entries_read = forms.slices_held;
    seconds_items = since() - launched;
    Pooled<u32> d_begins, d_cut_cell_begins, d_cell_begins, d_next, d_cut_cell_next, d_fixed_at, d_slot_of, d_missed(1);
    Pooled<FoldItem> d_items(begins[slots]), d_cut_cell_items(cut_cell_begins[slots]), d_fixed;
    Pooled<FoldItem> &d_cell_items = d_entries_;
    Pooled<u64> d_sums_at;
    Pooled<E> d_gram((size_t)n_rows * slots);
    send_cuts(cuts_, forms);
    up_->send(d_begins, begins.data(), begins.size());
    up_->send(d_cut_cell_begins, cut_cell_begins.data(), cut_cell_begins.size());
    up_->send(d_cell_begins, cell_begins.data(), cell_begins.size());
    up_->send(d_next, next.data(), next.size());
    up_->send(d_cut_cell_next, cut_cell_begins.data(), slots);
    up_->send(d_fixed_at, fixed_at.data(), fixed_at.size());
    up_->send(d_fixed, fixed.data(), fixed.size());
    up_->send(d_slot_of, slot_of.data(), slot_of.size());
    up_->send(d_sums_at, flat_sums_at.data(), flat_sums_at.size());
    up_->send(d_cell_items, cell_items.data(), cell_items.size());
    const bool some_cell_items = !cell_items.empty();
    Spare<FoldItem>::give(std::move(cell_items)); // (it has been copied)
    ENGINE_CK(cudaMemsetAsync(d_gram.get(), 0, d_gram.size() * sizeof(E), stream_));
    ENGINE_CK(cudaMemsetAsync(d_missed.get(), 0, sizeof(u32), stream_));
    const double t0 = products_->now();
    BLK_REQUIRE(slots <= 65535, "fold: more than 65,535 rows of c that are not zero"); // gridDim.y
    if (!fixed.empty()) {
      k_fold_put<<<blocks_for(fixed.size(), FOLD_THREADS), FOLD_THREADS, 0, stream_>>>(d_items.get(), d_fixed.get(), d_fixed_at.get(), (u32)fixed.size());
      ENGINE_CK(cudaGetLastError());
    }
    if (cuts_.first.back()) {
      k_fold_expand<<<blocks_for(cuts_.first.back(), FOLD_THREADS), FOLD_THREADS, 0, stream_>>>(d_items.get(), d_next.get(), d_cut_cell_items.get(), d_cut_cell_next.get(), cuts_.on_card(),
                                                                                              cuts_.d_first.get(), d_sums_at.get(), (u32)forms.vectors.size(), d_slot_of.get(), k_, log_slab,
                                                                                              d_missed.get());
      ENGINE_CK(cudaGetLastError());
    }
    if (begins[slots]) {
      k_fold_gather<<<dim3(blocks_for(n_rows, FOLD_THREADS / 32), slots), FOLD_THREADS, 0, stream_>>>(d_gram.get(), slots, n_rows, sums.get(), d_begins.get(), d_items.get());
      ENGINE_CK(cudaGetLastError());
    }
    if (cut_cell_begins[slots]) {
      k_fold_gather_cells<<<dim3(n_rows, slots), FOLD_THREADS, 0, stream_>>>(d_gram.get(), slots, products_->rows_on_card(), d_cut_cell_begins.get(), d_cut_cell_items.get());
      ENGINE_CK(cudaGetLastError());
    }
    if (some_cell_items) {
      k_fold_gather_cells<<<dim3(n_rows, slots), FOLD_THREADS, 0, stream_>>>(d_gram.get(), slots, products_->rows_on_card(), d_cell_begins.get(), d_cell_items.get());
      ENGINE_CK(cudaGetLastError());
    }
    seconds_gather += products_->now() - t0;
    std::vector<E> small((size_t)n_rows * slots);
    ENGINE_CK(cudaStreamSynchronize(stream_));
    seconds_card = std::chrono::duration<double>(std::chrono::steady_clock::now() - prepared).count();
    d_gram.to_host(small.data());
    u32 missed = 0;
    d_missed.to_host(&missed);
    BLK_REQUIRE(!missed, "fold: a sum is asked for that was not planned");
    for (u32 i = 0; i < n_rows; i++)
      for (u32 j = 0; j < slots; j++) gram_[(size_t)row_of_[i] * all + row_of_slot[j]] += small[(size_t)i * slots + j] * E::theta(power_of_[i]);
    products_->release_work();
    if (!setting_.folded_coefficients) forget_coefficients(); // (the stream has been waited for)
  }

  unsigned k_, left_, log_ell_ = 0, log_slab_ = 0;
  u64 ell_, r_;
  cudaStream_t stream_;
  Uploads *up_ = nullptr; // set while the constructor or `end` runs: uploads their short lists
  FoldSetting setting_;
  std::vector<u32> row_of_;        // of each row over F: the group's row,
  std::vector<unsigned> power_of_; // and the power of X that it stands for
  // For the rows over F: their random tails; their cells and the products with them (tiled.cuh).
  Pooled<ProductRow> tails_;
  std::unique_ptr<CardTiled> products_;
  // gram_[b * n + b'] = < row b of m , row b' of c > for the present n = 2^left_ rows
  std::vector<E> gram_;
  std::vector<E> gamma_;
  // Kept from the constructor for write_out_folded(): the V's of the way PIECES, as
  // k_fold_vectors() reads them, and the cuts.
  Pooled<E> d_factors_, d_values_;
  Pooled<FoldVector> d_vectors_;
  u64 store_words_ = 0;
  Cuts cuts_;                                      // the Tensors of the ways PIECES and CELLS
  Pooled<FoldItem> d_entries_;                     // the single entries; row_step: the row of c
  bool narrow_rows_ = false;                       // some rows have cells of three signed bytes
  std::vector<u32> narrow_, others_;               // rows for `k_fold_combine_bytes`; all others
  std::vector<Pooled<F>> summed_;                  // rows of c summed beforehand, by coordinates
  Form rest_;                                    // Tensors of c that were written out term by term
  std::vector<std::pair<u64, FoldHeld>> held_;   // the held vectors: (row of c, FoldHeld)
};

} // namespace blk
