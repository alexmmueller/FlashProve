// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Inner products of many rows of cells with many vectors over the field F, of several lengths, on
// the GPU, in few launches, for CardFold (fold.cuh). A cell (Cell in encode.cuh) stores one entry
// of a row (ProductRow in products.cuh) as a field element or as an integer in one to three bytes.
// products.cuh handles vectors of one length. The caller's duties, and what released memory
// contains: see CardTiled (`Card` in names: the GPU).
//
// A row's places (the indices of its entries) are divided into slabs of 2^log_slab. A column
// (TiledColumn) is a vector of 2^w entries, w <= log_slab, that multiplies every aligned range of
// 2^w places of one slab. The slab is cut into chunks of 2^log_k places, log_k <= w. For every row,
// column and chunk c, with chunks = 2^(log_slab - log_k) and slab the slab's first place:
//   out[column.out_at + (row * chunks + c) * column.out_step] =
//     sum over p < 2^log_k of entry(row, slab + c 2^log_k + p) * vector[(c 2^log_k + p) mod 2^w]
// A caller that needs a range's sum adds the sums of its 2^(w - log_k) chunks. Short chunks make
// many thread blocks with little work each, which fill the GPU evenly.
//
// ProductWay::PLAIN uses the field's arithmetic. ProductWay::BYTES gives the same results: it cuts
// numbers into signed bytes as products.cuh does, and one kernel multiplies them on the tensor
// cores and combines the 32-bit sums into elements of F, so the sums never go to global memory.
// A plane is signed byte i of every cell of one row. Planes are written for one span at a time:
// 2^log_span places, entire slabs, as many as the caller has memory for.
//
// Public sources. Numbers cut into bytes for the integer tensor cores: see products.cuh.
// The titles and addresses of the works cited below are unchecked.
// [mma] NVIDIA, Parallel Thread Execution ISA, "Warp-level matrix instructions",
//   "Matrix Fragments for mma.m16n8k32": the instruction
//   mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 and which thread of a warp holds which entries.
// [ldmatrix] the same document, "Warp-level matrix load instruction: ldmatrix".
// [CUTLASS] NVIDIA, CUTLASS documentation, "Efficient GEMM in CUDA",
//   https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html: a block's tile is staged in
//   shared memory, each warp computes a part of it from fragments in registers, the accumulators
//   stay in registers over the loop along the summed dimension, and several stages in shared memory
//   hide the latency of the loads (software pipelining).
// [Async] NVIDIA, CUDA Programming Guide, "Asynchronous Data Copies": __pipeline_memcpy_async(),
//   __pipeline_commit(), __pipeline_wait_prior() of cuda_pipeline.h: with them k_tiled_multiply()
//   brings its tiles from device memory straight into shared memory, with no stop in registers.
// [Padding] M. Harris, "An Efficient Matrix Transpose in CUDA C/C++", NVIDIA Technical Blog,
//   https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/: a padded row length in
//   shared memory removes bank conflicts between rows.
#pragma once
#include <cuda_pipeline.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

#include "engine/device.cuh"
#include "engine/encode.cuh" // Cell, cell_at
#include "engine/pooled.cuh"
#include "engine/products.cuh" // ProductRow, ProductWay, product_bytes, product_spacing, product_offset
#include "common/setting.h" // one-off (not upstream): g_double_buffer_planes
#include "field/field.h"
#include "field/field_lazy.cuh"

namespace blk {

// A column (see the formula at the top). multiply_plain() reads `vector`, multiply() reads `bytes`;
// the one that is not read may be null. The address `bytes` and `pitch` must be multiples of 16.
struct TiledColumn {
  const F *vector;          // its 2^w entries
  const signed char *bytes; // BYTES: signed byte j of entry p is bytes[j * pitch + p], j < 8
  u64 pitch;
  u64 out_at;
  u32 w, out_step;
};
struct TiledTile { // up to TILED_COLUMNS columns on one slab, each of at least 2^log_k entries
  u32 first_column, columns;
  u32 log_k;
  u64 from; // the slab's first place, counted from the span's
};
// One entry of multiply()'s `work`: chunk number `chunk` of tiles[tile].
struct TiledWork {
  u32 tile, chunk;
};
struct TiledRow { // where a row's planes are stored
  u32 plane, count; // the first of them, and how many
  u32 spacing;      // plane i has the weight 2^(i * spacing)
  u64 offset;       // a cell's value in F is the sum of (byte i) 2^(i * spacing), plus this
};
// length, pitch and the address `bytes` must be multiples of 4 (not checked).
struct TiledSource { // for `bytes_of`: byte j of vector[p] goes to bytes[j * pitch + p], p < length
  const F *vector;
  signed char *bytes;
  u64 pitch, length;
};

// TILED_STEP: a row's 128 bytes are the size of a cache line; with 64 the kernel took a sixth
// longer. TILED_PITCH: the padding puts the 16-byte pieces of eight neighbouring rows into eight
// different groups of four banks [Padding]. TILED_ROOM: dynamic shared memory, in bytes.
constexpr unsigned TILED_THREADS = 256;
constexpr unsigned TILED_M = 128, TILED_N = 128; // a block's planes, and its rows of columns' bytes
constexpr unsigned TILED_COLUMNS = TILED_N / 8;
constexpr unsigned TILED_STEP = 128;              // places copied to shared memory in one step
constexpr unsigned TILED_STAGES = 3;
constexpr unsigned TILED_PARTS = TILED_STEP / 16; // 16-byte pieces of one row in a step
constexpr unsigned TILED_PITCH = TILED_STEP + 16; // bytes from one row of a tile to the next
constexpr unsigned TILED_STAGE_BYTES = (TILED_M + TILED_N) * TILED_PITCH;
constexpr unsigned TILED_SUMS_PITCH = TILED_N + 8; // 32-bit sums from one plane to the next
constexpr unsigned TILED_ROOM = TILED_STAGES * TILED_STAGE_BYTES;
static_assert(TILED_M * TILED_SUMS_PITCH * sizeof(int) <= TILED_ROOM, "the sums take the room of the tiles");
constexpr unsigned TILED_EACH = TILED_M * TILED_PARTS / TILED_THREADS; // pieces of either tile that a thread brings in
static_assert(TILED_M == TILED_N && TILED_EACH * TILED_THREADS == TILED_M * TILED_PARTS, "the threads share the pieces of either tile evenly");

// Byte j of d[0] .. d[3] in one 32-bit word, d[0]'s lowest.
__device__ __forceinline__ u32 tiled_four(const u64 (&d)[4], int j) {
  return (u32)((d[0] >> (8 * j)) & 255) | (u32)((d[1] >> (8 * j)) & 255) << 8 | (u32)((d[2] >> (8 * j)) & 255) << 16 | (u32)((d[3] >> (8 * j)) & 255) << 24;
}
// Writes planes[(where[y].plane + i) * pitch + p] = signed byte i of row y's cell at place
// from + p, for p < width, y = blockIdx.y. A thread writes four neighbouring places: width and
// pitch must be multiples of 4.
static __global__ void k_tiled_planes(signed char *planes, u64 pitch, const ProductRow *rows, const TiledRow *where, u64 from, u64 width) {
  const u64 p = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (p >= width) return;
  const ProductRow row = rows[blockIdx.y];
  const TiledRow w = where[blockIdx.y];
  u64 d[4]; // a cell's signed bytes, the lowest first
  for (unsigned q = 0; q < 4; q++) {
    const u64 at = row.first + (from + p + q) * row.stride;
    if (row.cell == Cell::WORD) d[q] = lazy::signed_digits(cell_at(row.cells, row.cell, at));
    else if (row.cell == Cell::U8) d[q] = (u64)(__ldg((const u8 *)row.cells + at) ^ 0x80u); // the byte less 128
    else if (row.cell == Cell::S24) {
      const u8 *b = (const u8 *)row.cells + 3 * at;
      d[q] = (u64)__ldg(b) | (u64)__ldg(b + 1) << 8 | (u64)__ldg(b + 2) << 16;
    } else // U16: both bytes less 128. I16: the low byte less 128; the high byte is signed already
      d[q] = (u64)(__ldg((const unsigned short *)row.cells + at) ^ (row.cell == Cell::I16 ? 0x0080u : 0x8080u));
  }
  for (u32 i = 0; i < w.count; i++) *reinterpret_cast<u32 *>(planes + (u64)(w.plane + i) * pitch + p) = tiled_four(d, (int)i);
}
// Writes vectors' signed bytes: source y = blockIdx.y. A thread converts four neighbouring entries.
static __global__ void k_tiled_bytes(const TiledSource *sources) {
  const TiledSource s = sources[blockIdx.y];
  const u64 p = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (p >= s.length) return;
  u64 d[4];
  for (unsigned q = 0; q < 4; q++) d[q] = lazy::signed_digits(s.vector[p + q]);
  for (int j = 0; j < 8; j++) *reinterpret_cast<u32 *>(s.bytes + (u64)j * s.pitch + p) = tiled_four(d, j);
}

// D += A B^T: A is 16 x 32 and B is 8 x 32 signed bytes, D is 16 x 8 sums. With g = lane / 4 and
// t = lane % 4, a thread holds d[0], d[1]: row g, columns 2 t, 2 t + 1; d[2], d[3]: row g + 8, the
// same columns [mma].
// The asm template and the order of its operands are the same as in ggml,
// ggml/src/ggml-cuda/mma.cuh (llama.cpp), MIT; that project's notice is in NOTICE.
__device__ __forceinline__ void tiled_mma(int (&d)[4], const u32 (&a)[4], u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// Loads four blocks of 8 rows x 16 bytes from shared memory. Thread l of the warp gives the address
// (a multiple of 16) of row l % 8 of block l / 8. Thread (g, t) receives bytes 4 t .. 4 t + 3 of
// row g of each block, as tiled_mma() needs its operands [ldmatrix].
__device__ __forceinline__ void tiled_load(u32 (&r)[4], const signed char *mine) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"((u32)__cvta_generic_to_shared(mine))
               : "memory");
}
// The sum of s[j] 2^(8 j + shift) over j < 8, as an element of F. s[j]: the inner product of one
// plane with a vector's bytes j; 2^shift: the plane's weight.
__device__ __forceinline__ F tiled_joined(const int *s, int shift) {
  const long long low = (long long)s[0] + ((long long)s[1] << 8) + ((long long)s[2] << 16) + ((long long)s[3] << 24); // below 2^55 in size
  const long long high = (long long)s[4] + ((long long)s[5] << 8) + ((long long)s[6] << 16) + ((long long)s[7] << 24);
  return lazy::join_digit(low, shift) + lazy::join_digit(high, shift + 32);
}

// BYTES. A thread block multiplies a tile of up to TILED_M planes with a tile of up to
// TILED_COLUMNS columns (TiledTile), 8 rows of bytes each, on one chunk. Both tiles are staged in
// shared memory, TILED_STEP places at a time, in TILED_STAGES buffers; each of the eight warps
// keeps 64 x 32 sums in registers [CUTLASS]. At the chunk's end the sums go to shared memory, and a
// thread joins those of one row and one column.
// Block x: tile of planes x % m_tiles; work[x / m_tiles] gives the tile of columns and the chunk.
// planes: the span's, pitch bytes apart. Tile i has planes_in[i] planes, from plane i TILED_M on;
// no other plane is read. rows_begin[i] .. rows_begin[i + 1] - 1: the rows with planes in tile i.
// ones: plane 0 of every tile is all ones, which gives the vector's sum on the chunk, for rows with
// an offset (TiledRow::offset).
static __global__ void __launch_bounds__(TILED_THREADS, 2)
    k_tiled_multiply(F *out, const signed char *planes, u64 pitch, u32 m_tiles, bool ones, unsigned log_slab, const TiledWork *work, const TiledTile *tiles,
                     const TiledColumn *columns, const TiledRow *rows, const u32 *rows_begin, const u32 *planes_in) {
  extern __shared__ __align__(16) signed char room[];
  const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const u32 m_tile = blockIdx.x % m_tiles;
  const TiledWork mine = work[blockIdx.x / m_tiles];
  const TiledTile tile = tiles[mine.tile];
  const u32 k = 1u << tile.log_k;
  const u64 place = tile.from + ((u64)mine.chunk << tile.log_k);
  const u32 in_tile = planes_in[m_tile];
  // A thread's pieces of 16 bytes in each tile: piece c is bytes 16 (c % TILED_PARTS) .. of row
  // c / TILED_PARTS.
  const signed char *a_from[TILED_EACH], *b_from[TILED_EACH];
#pragma unroll
  for (unsigned i = 0; i < TILED_EACH; i++) {
    const unsigned c = threadIdx.x + i * TILED_THREADS, line = c / TILED_PARTS, part = c % TILED_PARTS;
    a_from[i] = line < in_tile ? planes + ((u64)m_tile * TILED_M + line) * pitch + place + part * 16 : nullptr;
    b_from[i] = nullptr; // the tile has fewer columns
    if ((line >> 3) < tile.columns) {
      const TiledColumn x = columns[tile.first_column + (line >> 3)];
      b_from[i] = x.bytes + (u64)(line & 7) * x.pitch + (place & (((u64)1 << x.w) - 1)) + part * 16;
    }
  }
  // Copies one step of both tiles [Async]. Beyond the chunk's end, and for missing planes, the tile
  // of planes gets zeros, so that whatever the other tile contains there adds nothing. The sums of
  // missing columns are never read.
  auto fetch = [&](u32 step) {
    signed char *to = room + (step % TILED_STAGES) * TILED_STAGE_BYTES;
#pragma unroll
    for (unsigned i = 0; i < TILED_EACH; i++) {
      const unsigned c = threadIdx.x + i * TILED_THREADS, line = c / TILED_PARTS, part = c % TILED_PARTS;
      const bool in = step * TILED_STEP + part * 16 < k;
      signed char *a_to = to + line * TILED_PITCH + part * 16, *b_to = a_to + TILED_M * TILED_PITCH;
      if (in && a_from[i]) __pipeline_memcpy_async(a_to, a_from[i] + (u64)step * TILED_STEP, 16);
      else *reinterpret_cast<uint4 *>(a_to) = make_uint4(0, 0, 0, 0);
      if (in && b_from[i]) __pipeline_memcpy_async(b_to, b_from[i] + (u64)step * TILED_STEP, 16);
    }
  };
  const u32 steps = (k + TILED_STEP - 1) / TILED_STEP;
  for (u32 s = 0; s + 1 < TILED_STAGES; s++) {
    if (s < steps) fetch(s);
    __pipeline_commit();
  }
  // Warp (wm, wn) has 64 planes from 64 wm and 32 rows of columns' bytes from 32 wn (four columns):
  // 4 x 4 results of tiled_mma(). A warp whose planes or columns are all missing skips the
  // products. (Assuming that a multiprocessor's four schedulers take the warps in turn, this
  // numbering gives each of them work if the tiles have over 64 planes and at least five columns.)
  int sum[4][4][4];
#pragma unroll
  for (unsigned m = 0; m < 4; m++)
#pragma unroll
    for (unsigned n = 0; n < 4; n++)
#pragma unroll
      for (unsigned e = 0; e < 4; e++) sum[m][n][e] = 0;
  const unsigned wm = warp & 1, wn = warp >> 1;
  const bool rests = wn * 4 >= tile.columns || wm * 64 >= in_tile;
  // Left operand: tiled_load()'s blocks are (rows 0 .. 7, bytes 0 .. 15), (rows 8 .. 15, the same),
  // (rows 0 .. 7, bytes 16 .. 31), (rows 8 .. 15, the same).
  const unsigned a_at = (wm * 64 + (lane & 7) + ((lane >> 3) & 1) * 8) * TILED_PITCH + ((lane >> 4) & 1) * 16;
  // Right operand, two at a time: blocks (rows 0 .. 7, bytes 0 .. 15), (the same, bytes 16 .. 31),
  // (rows 8 .. 15, bytes 0 .. 15), (the same, bytes 16 .. 31).
  const unsigned b_at = (TILED_M + wn * 32 + (lane & 7) + ((lane >> 4) & 1) * 8) * TILED_PITCH + ((lane >> 3) & 1) * 16;
  for (u32 s = 0; s < steps; s++) {
    __pipeline_wait_prior(TILED_STAGES - 2); // this thread's copies for step s are complete
    __syncthreads();                         // so are all threads'; step s - 1 may be overwritten
    if (s + TILED_STAGES - 1 < steps) fetch(s + TILED_STAGES - 1);
    __pipeline_commit();
    if (rests) continue;
    const signed char *at = room + (s % TILED_STAGES) * TILED_STAGE_BYTES;
#pragma unroll
    for (unsigned half = 0; half < TILED_STEP / 32; half++) {
      u32 a[4][4], b[2][4];
#pragma unroll
      for (unsigned m = 0; m < 4; m++) tiled_load(a[m], at + a_at + m * 16 * TILED_PITCH + half * 32);
#pragma unroll
      for (unsigned h = 0; h < 2; h++) tiled_load(b[h], at + b_at + h * 16 * TILED_PITCH + half * 32);
#pragma unroll
      for (unsigned m = 0; m < 4; m++)
#pragma unroll
        for (unsigned n = 0; n < 4; n++) tiled_mma(sum[m][n], a[m], b[n >> 1][(n & 1) * 2], b[n >> 1][(n & 1) * 2 + 1]);
    }
  }
  __pipeline_wait_prior(0);
  __syncthreads();
  int *sums = reinterpret_cast<int *>(room); // sums[plane * TILED_SUMS_PITCH + column * 8 + j]
#pragma unroll
  for (unsigned m = 0; m < 4; m++)
#pragma unroll
    for (unsigned n = 0; n < 4; n++) {
      int *to = sums + (wm * 64 + m * 16 + (lane >> 2)) * TILED_SUMS_PITCH + wn * 32 + n * 8 + (lane & 3) * 2;
      *reinterpret_cast<int2 *>(to) = make_int2(sum[m][n][0], sum[m][n][1]);
      *reinterpret_cast<int2 *>(to + 8 * TILED_SUMS_PITCH) = make_int2(sum[m][n][2], sum[m][n][3]);
    }
  __syncthreads();
  const u32 first_row = rows_begin[m_tile], n_rows = rows_begin[m_tile + 1] - first_row;
  const u64 chunks = (u64)1 << (log_slab - tile.log_k);
  for (u32 e = threadIdx.x; e < n_rows * TILED_COLUMNS; e += TILED_THREADS) {
    const u32 col = e % TILED_COLUMNS, r = first_row + e / TILED_COLUMNS;
    if (col >= tile.columns) continue;
    const TiledRow row = rows[r];
    F x = F::zero();
    for (u32 i = 0; i < row.count; i++) x += tiled_joined(sums + (row.plane % TILED_M + i) * TILED_SUMS_PITCH + col * 8, (int)(i * row.spacing));
    if (ones && row.offset) x += F{row.offset} * tiled_joined(sums + col * 8, 0);
    const TiledColumn c = columns[tile.first_column + col];
    out[c.out_at + ((u64)r * chunks + mine.chunk) * c.out_step] = x;
  }
}
// The same sums by the field's arithmetic, for one tile. from: the slab's first place, counted from
// the row's first entry. Block (x, y, z): chunk x, row y, column z of the tile.
static __global__ void k_tiled_plain(F *out, const ProductRow *rows, u64 from, unsigned log_slab, unsigned log_k, const TiledColumn *columns) {
  __shared__ F room[TILED_THREADS];
  const TiledColumn c = columns[blockIdx.z];
  const u64 len = (u64)1 << log_k, chunks = (u64)1 << (log_slab - log_k), in = ((u64)blockIdx.x << log_k) & (((u64)1 << c.w) - 1);
  const ProductRow row = rows[blockIdx.y];
  F s = F::zero();
  for (u64 p = threadIdx.x; p < len; p += TILED_THREADS) s += cell_at(row.cells, row.cell, row.first + (from + blockIdx.x * len + p) * row.stride) * c.vector[in + p];
  room[threadIdx.x] = s;
  __syncthreads();
  for (unsigned h = TILED_THREADS / 2; h > 0; h /= 2) {
    if (threadIdx.x < h) room[threadIdx.x] += room[threadIdx.x + h];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[c.out_at + ((u64)blockIdx.y * chunks + blockIdx.x) * c.out_step] = room[0];
}

// Order of calls: open(); for each span, span(), then multiply() for tiles whose log_k by_bytes()
// accepts and multiply_plain() for the others; release_work().
// A failed BLK_REQUIRE (blocks/base.h) or CUDA call ends the process.
// All work goes on the constructor's stream. What is passed in must be complete for that stream
// (written on it, or waited for); nothing here checks that.
// The work space is Pooled memory (prover/engine/pooled.cuh), which is not cleared when it is given
// back: it contains the rows' cells as bytes.
class CardTiled {
public:
  static constexpr unsigned LEAST_LOG_K = 4;  // BYTES: a chunk's bytes begin at multiples of 16
  static constexpr unsigned MOST_LOG_K = 16;  // 2^16 products of signed bytes sum to at most 2^30

  // uploads: on the same stream (Uploads::send() checks that).
  CardTiled(const std::vector<ProductRow> &rows, ProductWay way, cudaStream_t stream, Uploads &uploads) : n_rows_((u32)rows.size()), way_(way), stream_(stream) {
    const OnStream on(stream);
    // One-off (not upstream): see planes_/planes_b_ below. Self-contained: no caller (fold.cuh)
    // needs to change, since span()/multiply() are still called in the same order as upstream;
    // only the buffer and stream that span()'s write kernel lands on changes.
    if (g_double_buffer_planes) {
      dbl_ = true;
      ENGINE_CK(cudaStreamCreateWithFlags(&plane_stream_, cudaStreamNonBlocking));
      for (int i = 0; i < 2; i++) ENGINE_CK(cudaEventCreateWithFlags(&plane_ready_[i], cudaEventDisableTiming));
      for (int i = 0; i < 2; i++) ENGINE_CK(cudaEventCreateWithFlags(&slot_consumed_[i], cudaEventDisableTiming));
      // slot_consumed_ guards a slot's first write too: record it already signaled (no real
      // consumer to wait for) so span()'s first two calls do not block on a stale, unrecorded event.
      for (int i = 0; i < 2; i++) ENGINE_CK(cudaEventRecord(slot_consumed_[i], stream));
    }
    BLK_REQUIRE(rows.size() <= 65535, "tiled products: more than 65,535 rows");
    uploads.send(rows_, rows.data(), rows.size());
    for (const ProductRow &r : rows) ones_ = ones_ || product_offset(r.cell) != 0;
    std::vector<TiledRow> where;
    std::vector<u32> begin = {0}, planes_in;
    u32 tile = 0, taken = ones_ ? 1 : 0; // the present tile, and its planes that are taken
    for (const ProductRow &r : rows) {
      const u32 count = product_bytes(r.cell);
      if (taken + count > TILED_M) { // a row's planes are all in one tile
        planes_in.push_back(taken);
        tile++, taken = ones_ ? 1 : 0;
        begin.push_back((u32)where.size());
      }
      where.push_back(TiledRow{tile * TILED_M + taken, count, product_spacing(r.cell), product_offset(r.cell)});
      taken += count;
    }
    begin.push_back((u32)where.size());
    planes_in.push_back(taken);
    m_tiles_ = (u32)begin.size() - 1, plane_rows_ = (u64)(m_tiles_ - 1) * TILED_M + taken;
    uploads.send(where_, where.data(), where.size());
    uploads.send(begin_, begin.data(), begin.size());
    uploads.send(planes_in_, planes_in.data(), planes_in.size());
  }
  // One-off (not upstream): the stream and events of g_double_buffer_planes, if used, are owned
  // here and released here; nothing else in this class allocates raw CUDA resources beyond what
  // Pooled<T> already manages on its own.
  ~CardTiled() {
    if (plane_stream_) ENGINE_CK(cudaStreamDestroy(plane_stream_));
    for (cudaEvent_t &e : plane_ready_)
      if (e) ENGINE_CK(cudaEventDestroy(e));
    for (cudaEvent_t &e : slot_consumed_)
      if (e) ENGINE_CK(cudaEventDestroy(e));
  }
  // If true, span(), bytes_of() and the multiplications wait for the stream and add their seconds
  // below, for reports.
  bool trace = false;
  // A test can lower grid_rows so that short lists take several launches.
  unsigned grid_rows = 65535; // 1 .. 65,535 (not checked here): descriptors per `bytes_of` launch
  size_t most_launches = 0;   // the most launches that one call of `bytes_of` made
  double seconds_row_bytes = 0, seconds_prepare = 0, seconds_multiply = 0;

  u32 rows() const { return n_rows_; }
  const ProductRow *rows_on_card() const { return rows_.get(); }
  bool by_bytes(unsigned log_k) const { return way_ == ProductWay::BYTES && log_k >= LEAST_LOG_K; }
  // The bytes of work space that open() takes for each place of a span.
  u64 plane_rows() const { return way_ == ProductWay::BYTES ? plane_rows_ : 0; }

  // Takes the work space for one span's planes.
  void open(unsigned log_span) {
    const OnStream on(stream_);
    log_span_ = log_span, opened_ = true;
    if (!by_bytes(log_span) || n_rows_ == 0) return; // PLAIN, or no chunk is long enough for BYTES
    const u64 width = (u64)1 << log_span;
    planes_.resize(plane_rows() * width);
    BLK_REQUIRE((std::uintptr_t)planes_.get() % 16 == 0, "tiled products: a work space that does not begin at a multiple of 16");
    if (ones_) ENGINE_CK(cudaMemset2DAsync(planes_.get(), (size_t)TILED_M * width, 1, width, m_tiles_, stream_));
    // One-off (not upstream): a second, same-sized buffer so span() for the next span can write
    // while multiply() for the current span still reads the other one (see span()/multiply()).
    if (dbl_) {
      const OnStream on_plane(plane_stream_); // one-off (not upstream): planes_b_'s pool allocation
      // is attributed to the stream it is actually written/read on, not stream_ (see release_work()).
      planes_b_.resize(plane_rows() * width);
      BLK_REQUIRE((std::uintptr_t)planes_b_.get() % 16 == 0, "tiled products: a work space that does not begin at a multiple of 16");
      if (ones_) ENGINE_CK(cudaMemset2DAsync(planes_b_.get(), (size_t)TILED_M * width, 1, width, m_tiles_, plane_stream_));
    }
    static const bool set = [] {
      ENGINE_CK_ATTRIBUTE(k_tiled_multiply, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TILED_ROOM);
      return true;
    }();
    (void)set;
  }
  // Turns to the span that begins at place `from`. BYTES: writes its planes.
  void span(u64 from) {
    BLK_REQUIRE(opened_ && !released_, "tiled products: a span is asked for before the work space was taken, or after it was given back");
    from_ = from, turned_ = true;
    if (!planes_.size()) return;
    const u64 width = (u64)1 << log_span_;
    const double t0 = now();
    if (!dbl_) {
      k_tiled_planes<<<dim3(blocks_for(width / 4, TILED_THREADS), n_rows_), TILED_THREADS, 0, stream_>>>(planes_.get(), width, rows_.get(), where_.get(), from, width);
      ENGINE_CK(cudaGetLastError());
    } else {
      // One-off (not upstream): double-buffered span(). Alternates which of planes_/planes_b_ this
      // call writes, on plane_stream_ rather than stream_, so the GPU can run it alongside the
      // previous span's still-running multiply() kernel on stream_ instead of after it. Before
      // reusing a slot, waits (on plane_stream_, not the host) for that slot's last reader
      // (multiply(), via slot_consumed_) to be done; after writing, lets stream_ know the slot it
      // is about to read is ready, without the host blocking on either stream.
      cur_slot_ ^= 1;
      signed char *dst = cur_slot_ ? planes_b_.get() : planes_.get();
      ENGINE_CK(cudaStreamWaitEvent(plane_stream_, slot_consumed_[cur_slot_], 0));
      k_tiled_planes<<<dim3(blocks_for(width / 4, TILED_THREADS), n_rows_), TILED_THREADS, 0, plane_stream_>>>(dst, width, rows_.get(), where_.get(), from, width);
      ENGINE_CK(cudaGetLastError());
      ENGINE_CK(cudaEventRecord(plane_ready_[cur_slot_], plane_stream_));
      ENGINE_CK(cudaStreamWaitEvent(stream_, plane_ready_[cur_slot_], 0));
    }
    seconds_row_bytes += now() - t0;
  }
  // Writes vectors' signed bytes. sources: `count` descriptors on the GPU; longest: the largest of
  // their lengths.
  void bytes_of(const TiledSource *sources, size_t count, u64 longest) {
    const double t0 = now();
    most_launches = std::max(most_launches, (count + grid_rows - 1) / grid_rows);
    for (size_t at = 0; at < count; at += grid_rows) { // a grid has at most 65,535 rows of blocks
      k_tiled_bytes<<<dim3(blocks_for(longest / 4, TILED_THREADS), (unsigned)std::min<size_t>(grid_rows, count - at)), TILED_THREADS, 0, stream_>>>(sources + at);
      ENGINE_CK(cudaGetLastError());
    }
    seconds_prepare += now() - t0;
  }
  // BYTES: `count` entries of `work` on the current span. work, tiles, columns: on the GPU, so
  // nothing here checks them. The caller ensures that each tile's log_k is one that by_bytes()
  // accepts, at most MOST_LOG_K and at most its columns' w; that its `from` is a multiple of
  // 2^log_slab and its slab is inside the span; and that bytes_of() has written its columns' bytes.
  void multiply(const TiledWork *work, size_t count, const TiledTile *tiles, const TiledColumn *columns, unsigned log_slab, F *out) {
    if (n_rows_ == 0 || count == 0) return;
    BLK_REQUIRE(turned_ && !released_ && planes_.size(), "tiled products: a multiplication before the first span, or after the work space was given back");
    BLK_REQUIRE(count * m_tiles_ < ((u64)1 << 31), "tiled products: more blocks than a launch takes");
    const double t0 = now();
    const signed char *src = dbl_ ? (cur_slot_ ? planes_b_.get() : planes_.get()) : planes_.get();
    k_tiled_multiply<<<(unsigned)(count * m_tiles_), TILED_THREADS, TILED_ROOM, stream_>>>(out, src, (u64)1 << log_span_, m_tiles_, ones_, log_slab, work, tiles, columns,
                                                                                         where_.get(), begin_.get(), planes_in_.get());
    ENGINE_CK(cudaGetLastError());
    // One-off (not upstream): lets span()'s NEXT use of this same slot (two spans from now) go
    // ahead once this read is done, instead of waiting for the host.
    if (dbl_) ENGINE_CK(cudaEventRecord(slot_consumed_[cur_slot_], stream_));
    seconds_multiply += now() - t0;
  }
  // All chunks of one tile by the field's arithmetic, under either ProductWay. tile: on the host;
  // columns: on the GPU.
  void multiply_plain(const TiledTile &tile, const TiledColumn *columns, unsigned log_slab, F *out) {
    if (n_rows_ == 0 || tile.columns == 0) return;
    BLK_REQUIRE(turned_, "tiled products: a multiplication before the first span");
    const double t0 = now();
    k_tiled_plain<<<dim3(1u << (log_slab - tile.log_k), n_rows_, tile.columns), TILED_THREADS, 0, stream_>>>(out, rows_.get(), from_ + tile.from, log_slab, tile.log_k,
                                                                                                          columns + tile.first_column);
    ENGINE_CK(cudaGetLastError());
    seconds_multiply += now() - t0;
  }

  // Gives back the work space. After it span() and multiply() fail; rows_on_card() stays valid.
  void release_work() {
    // One-off (not upstream): plane_stream_'s work must be visibly done before planes_b_ (whose
    // pool allocation is attributed to plane_stream_, see open()) is freed, since nothing else
    // drains plane_stream_ on its own; a full host-side wait here is fine (release_work() already
    // runs at request end, not in the per-span hot path).
    if (dbl_) ENGINE_CK(cudaStreamSynchronize(plane_stream_));
    planes_.release(), planes_b_.release(), released_ = true;
  }

  // The time in seconds, after a wait for the stream; 0 if `trace` is false.
  double now() const {
    if (!trace) return 0;
    ENGINE_CK(cudaStreamSynchronize(stream_));
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }

private:
  u32 n_rows_, m_tiles_ = 0;
  u64 plane_rows_ = 0; // planes in the work space, up to the last one that a row fills
  unsigned log_span_ = 0;
  bool ones_ = false; // some row has a nonzero TiledRow::offset
  bool opened_ = false, turned_ = false, released_ = false;
  ProductWay way_;
  cudaStream_t stream_;
  u64 from_ = 0;
  Pooled<ProductRow> rows_;
  Pooled<TiledRow> where_;
  Pooled<u32> begin_, planes_in_;
  Pooled<signed char> planes_;
  // One-off (not upstream): g_double_buffer_planes's state. planes_b_ is the second buffer;
  // cur_slot_ (0: planes_, 1: planes_b_) is which one the last span() call wrote and multiply()
  // must therefore read; plane_stream_ is where span()'s write kernel runs when doubled, instead
  // of stream_; plane_ready_[s]/slot_consumed_[s] cross-stream-synchronize slot s's write and read
  // without any host-side wait. All no-ops (dbl_ stays false) unless -double-buffer-planes is set.
  bool dbl_ = false;
  int cur_slot_ = 0;
  cudaStream_t plane_stream_ = nullptr;
  cudaEvent_t plane_ready_[2] = {nullptr, nullptr};
  cudaEvent_t slot_consumed_[2] = {nullptr, nullptr};
  Pooled<signed char> planes_b_;
};

} // namespace blk
