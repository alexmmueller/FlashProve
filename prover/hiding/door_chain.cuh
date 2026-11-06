// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Translates the linear equation that ends a sub-proof (for one part of the forward pass) from an
// `hd::Door` (prover/hiding/door.cuh) into a blk::ReqEquation (blocks/iop/req_rows.h) and closes
// the sub-proof's link with it (ChainRun::close(), prover/prover/chain_run.cuh). A `link`, or
// segment, is a sub-proof's part of a request's proof (blocks/iop/req.h).
//
// An hd::Term (prover/hiding/door_terms.cuh) is a committed array with pairs of vectors. Each
// hd::Pair (cv, L, c0, c1) becomes one blk::ReqTerm, with the weight cv[r] * L[c] on element (r, c)
// of the array, or 0 if c is outside [c0, c1). A ReqTerm takes each factor as a product form (two
// values for each bit of the index) or, for at most one of the two, as a written-out vector. L is
// written out, so the weights on the rows must be a product form, by Term::kind:
//   SYMBOLS   (symbols of three cells) the product of Term::point[j] over the row number's set bits
//   ELEMENTS  (elements of blk::E) the array has one row, with the weight 1
//   CELLS16   (16-bit cells) the form in Pair::form; stating none (PLAIN) is an internal error
// The form is compared with cv only for CELLS16, and only under the test variable
// CHAIN_CHECK_FORMS, which no test in this tree sets.
#pragma once
#include "hiding/door_values.cuh"
#include "hiding/door_vectors.cuh"
#include "prover/chain_host.cuh"
#include <algorithm>
#include <memory>
namespace hd {
static ChainHost *the_chain = nullptr;
static bool the_chain_puts_vectors = false; // set by link_frame() alone

// Batched closing (one-off, not upstream; toggled by -batch-fetch -> g_batch_fetch_mode,
// chain_run.cuh): defers the several small device->host reads one closing step needs to a single
// cudaStreamSynchronize() in chain_apply_eq_patches() below, instead of chain_fetch's one sync per
// read. Each function writes a correctly-shaped placeholder into its return value immediately (so
// every caller's control flow, including code that appends more entries afterward, runs unchanged)
// and records what to overwrite once chain_fetch_sync_all() has run. term_index is the index this
// term will get in eq.terms: valid because, at every call site below, nothing else pushes into
// eq.terms between this call and the eq.terms.push_back(t) that follows it in the same loop
// iteration of close_by_frame() (checked by reading that loop).
struct PendingEqPatch {
  enum Kind { Equality, SetBits, Written } kind;
  size_t term_index = 0, fetch_idx = 0, c0 = 0, c1 = 0, n = 0;
};
static std::vector<PendingEqPatch> g_pending_eq_patches;

// eq(point, .) as a product form. Coordinate 0 of the point belongs to the top bit of the index;
// blk::IndexWeights counts bits from the lowest.
static inline blk::IndexWeights chain_equality(const T *point_on_card, int bits, size_t term_index = 0) {
  if (!g_batch_fetch_mode) {
    std::vector<blk::E> point = chain_fetch(point_on_card, (size_t)bits, cudaStreamPerThread);
    std::reverse(point.begin(), point.end());
    return blk::IndexWeights::eq(point);
  }
  const size_t idx = chain_fetch_queue(point_on_card, (size_t)bits, cudaStreamPerThread);
  g_pending_eq_patches.push_back({PendingEqPatch::Equality, term_index, idx});
  blk::IndexWeights w;
  w.lo.assign((size_t)bits, blk::E::one()), w.hi.assign((size_t)bits, blk::E::one()); // placeholder
  return w;
}
// The product form in which a set bit of the index has the factor point[j] and a clear bit the
// factor 1. Element 0 belongs to the top bit.
static inline blk::IndexWeights chain_set_bits(const T *point_on_card, int bits, size_t term_index = 0) {
  if (!g_batch_fetch_mode) {
    blk::IndexWeights w;
    w.hi = chain_fetch(point_on_card, (size_t)bits, cudaStreamPerThread);
    std::reverse(w.hi.begin(), w.hi.end());
    w.lo.assign(w.hi.size(), blk::E::one());
    return w;
  }
  const size_t idx = chain_fetch_queue(point_on_card, (size_t)bits, cudaStreamPerThread);
  g_pending_eq_patches.push_back({PendingEqPatch::SetBits, term_index, idx});
  blk::IndexWeights w;
  w.hi.assign((size_t)bits, blk::E::one()), w.lo.assign((size_t)bits, blk::E::one()); // placeholder
  return w;
}
// A written-out vector on the host: n entries from the GPU, with 0 outside [c0, c1).
static inline blk::IndexWeights chain_written(const T *on_card, size_t n, size_t c0, size_t c1, size_t term_index = 0) {
  if (!g_batch_fetch_mode) {
    auto v = std::make_shared<std::vector<blk::E>>(chain_fetch(on_card, n, cudaStreamPerThread));
    for (size_t c = 0; c < n; c++)
      if (c < c0 || c >= c1) (*v)[c] = blk::E::zero();
    return blk::IndexWeights::written(v);
  }
  const size_t idx = chain_fetch_queue(on_card, n, cudaStreamPerThread);
  g_pending_eq_patches.push_back({PendingEqPatch::Written, term_index, idx, c0, c1, n});
  auto v = std::make_shared<std::vector<blk::E>>(n, blk::E::zero()); // placeholder, right size
  return blk::IndexWeights::written(v);
}
// Applies every patch recorded since the last chain_fetch_batch_reset(), using data that
// chain_fetch_sync_all() (called by the caller just before this) has made ready.
static inline void chain_apply_eq_patches(blk::ReqEquation &eq) {
  for (const PendingEqPatch &p : g_pending_eq_patches) {
    if (p.kind == PendingEqPatch::Written) {
      std::vector<blk::E> v = chain_fetch_take(p.fetch_idx);
      for (size_t c = 0; c < p.n; c++)
        if (c < p.c0 || c >= p.c1) v[c] = blk::E::zero();
      eq.terms[p.term_index].on_columns.values = std::make_shared<const std::vector<blk::E>>(std::move(v));
    } else {
      std::vector<blk::E> point = chain_fetch_take(p.fetch_idx);
      std::reverse(point.begin(), point.end());
      blk::IndexWeights &w = eq.terms[p.term_index].on_rows;
      // Overwrites only the first point.size() entries: chain_equality's caller (chain_rows_of,
      // the EQUALITY branch) can append more real (non-fetched) entries after the placeholder,
      // which must survive untouched; for SetBits nothing is ever appended, and point.size()
      // covers the whole vector.
      for (size_t t = 0; t < point.size(); t++) {
        if (p.kind == PendingEqPatch::Equality) w.lo[t] = blk::E::one() - point[t];
        else w.lo[t] = blk::E::one();
        w.hi[t] = point[t];
      }
    }
  }
  g_pending_eq_patches.clear();
}
// The same on the GPU, one thread for each entry: the launch must cover n, which is not checked.
__global__ static void k_door_factor(T *to, const T *from, size_t n, size_t c0, size_t c1) {
  const size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (c < n) to[c] = c >= c0 && c < c1 ? from[c] : T{};
}
// Sets w to a pair's weights on the R rows of its array, from Pair::form. Returns false for PLAIN.
static inline bool chain_rows_of(const Pair &p, size_t R, blk::IndexWeights &w, size_t term_index) {
  const unsigned row_bits = blk::ceil_log2(R);
  if (p.form == EQUALITY) {
    BLK_REQUIRE(p.bits >= 0 && (unsigned)p.bits <= row_bits && (p.first >> p.bits << p.bits) == p.first && p.first < (size_t(1) << row_bits),
                "door: an equality vector's block is not aligned or lies beyond the array's rows");
    w = chain_equality(p.point, p.bits, term_index);
    for (unsigned t = (unsigned)p.bits; t < row_bits; t++) {  // the bits above pick the block
      const bool one = (p.first >> t) & 1;
      w.lo.push_back(one ? blk::E::zero() : blk::E::one()), w.hi.push_back(one ? blk::E::one() : blk::E::zero());
    }
    return true;
  }
  if (p.form == ONES) {
    w = blk::IndexWeights();
    w.lo.assign(row_bits, blk::E::one()), w.hi.assign(row_bits, blk::E::one());
    return true;
  }
  if (p.form == SET_BITS) {
    BLK_REQUIRE(p.bits >= 0 && (unsigned)p.bits == row_bits, "door: a product over the set bits has not one factor for each bit of the row's number");
    w = chain_set_bits(p.point, p.bits, term_index);
    return true;
  }
  return false;
}
static inline blk::E chain_weight_at(const blk::IndexWeights &w, size_t r) {
  blk::E x = blk::E::one();
  for (size_t t = 0; t < w.lo.size(); t++) x *= ((r >> t) & 1) ? w.hi[t] : w.lo[t];
  return x;
}

// The closing function, which hd::close() calls; the frame is a request's proof outside its
// sub-proofs (blocks/iop/req.h). who and g name the sub-proof in printed lines. tr must be the
// request's transcript (a hash chain: blocks/chain/chain.h), or the process exits with code 4.
// D's vectors must be filled before the call, by work that is complete or queued on this thread's
// cudaStreamPerThread; this is not checked. Returns 0, or 4 after printing an error.
//
// A term's array is found in the layout by the link that committed it and its name there, Term::nm:
// "M<mu>" for a symbol matrix (prover/prover/chain_frame.cuh), "U3", "ZZ", "BZ"
// (prover/prover/chain_moments.cuh). A foreign matrix, one that an earlier link committed, takes
// both from ChainLink::reads: the n-th foreign term of D.t uses the n-th entry (a convention of
// this file; Prover::chain_make() gives a link at most one entry). The first D.ns entries of D.LS0
// are the coefficients on the link's mask row, whose entry i masks the sub-proof's scalar i.
// ReqEquation::target is not set: the prover computes no constant, and the fold in
// ChainRun::close() takes its starting claim from its own first round, that is, from the committed
// arrays (under the test variable CHAIN_ROW_VALUE, from the equation's value on them).
static int close_by_frame(const Door &D, Tr *tr, const char *who, int g) {
  ChainHost &C = *the_chain;
  if (D.link < 0 || D.link >= C.links()) {
    printf("internal error: door: %s %d: the link number is not set or is out of range\n", who, g);
    return 4;
  }
  C.hold_the_transcript(tr);
  C.request->take_back(); // the hash chain returns to the host: no more sub-proof challenges
  sc_door::cheat_say_seen(who, g); // a test hook: empty in this repository
  const ChainLink &L = C.plan[D.link];
  blk::ReqEquation eq;
  size_t foreign_seen = 0, pairs = 0;
  // The pairs' vectors L stay on the GPU: each pair gets a range of the link's one buffer
  // (ChainRun::take_factors()), a copy of L with zeros outside [c0, c1). Two exceptions go to the
  // host:
  // (1) A term on an array whose oracle message (rows sent together as Merkle roots) is still kept
  //     after this link, because a later link reads it (ReqTerm::born + ReqMessage::life > this
  //     link, which the public shape decides). The closing does not fold such a term but passes it
  //     on into the next link (blocks/iop/fold.h), where the factor is read again after
  //     ChainRun::close() has released this link's buffer. The room that such pairs leave unused at
  //     the buffer's end is not initialised; no term points into it, but CardFold::make_forms()
  //     (prover/engine/fold.cuh) can copy it.
  // (2) Every term, under the test variable CHAIN_L_ON_HOST. With the same private randomness both
  //     ways must give the same proof, byte for byte; no test in this tree sets the variable.
  const bool all_on_host = TESTENV("CHAIN_L_ON_HOST") != nullptr;
  size_t room = 0, held_elements = 0, host_elements = 0;
  if (!all_on_host)
    for (const Term &x : D.t) room += x.K * x.v.size();  // room for every pair
  T *const factors = room ? C.request->take_factors(room) : nullptr;
  std::vector<std::string> names; // for a test's files: each term's array as the frame has it: "<the link that committed it>:<its name there>"
  for (const Term &x : D.t) {
    blk::ReqTerm t;
    t.born = (size_t)D.link, t.array = x.nm;
    if (x.foreign) {
      if (foreign_seen >= L.reads.size()) {
        printf("door: the link %s reads no array of an earlier link for the matrix %s, which another sub-proof committed\n", L.name.c_str(),
               x.nm.c_str());
        return 4;
      }
      t.born = (size_t)L.reads[foreign_seen].first, t.array = L.reads[foreign_seen].second;
      foreign_seen++;
    }
    names.push_back(std::to_string(t.born) + ":" + t.array);
    const blk::ReqArray *const array = C.fixed.layout.find(t.born, t.array);
    if (!array) {
      printf("internal error: door: %s %d: the layout has no array %s of link %zu\n", who, g, t.array.c_str(), t.born + 1);
      return 4;
    }
    const bool still_kept = t.born + C.fixed.shape.link[t.born].messages[array->message].life > (size_t)D.link;
    for (const Pair &p : x.v) {
      if (!factors || still_kept || !x.K) {
        t.on_columns = chain_written(p.L, x.K, p.c0, p.c1, eq.terms.size());
        host_elements += x.K;
      } else {
        k_door_factor<<<(unsigned)((x.K + 255) / 256), 256, 0, cudaStreamPerThread>>>(factors + held_elements, p.L, x.K, p.c0, p.c1);
        CK(cudaGetLastError());
        t.on_columns = blk::IndexWeights::held_by_engine(C.request->factors_table(), held_elements, x.K, room);
        held_elements += x.K;
      }
      if (x.kind == SYMBOLS) {
        t.scale = blk::E::one(), t.on_rows = chain_set_bits(x.point, x.nb, eq.terms.size());
        eq.terms.push_back(t);
      } else if (x.kind == ELEMENTS) {
        t.scale = blk::E::one(), t.on_rows = blk::IndexWeights::eq({});
        eq.terms.push_back(t);
      } else if (chain_rows_of(p, x.R, t.on_rows, eq.terms.size())) {
        t.scale = blk::E::one();
        if (TESTENV("CHAIN_CHECK_FORMS")) {
          const std::vector<blk::E> cv = chain_fetch(p.cv, x.R, cudaStreamPerThread);
          for (size_t r = 0; r < x.R; r++)
            if (!(chain_weight_at(t.on_rows, r) == cv[r])) {
              printf("internal error: door: %s %d: array %s: the product form (kind %d) that a pair states for its weights on the rows differs from the vector at row %zu\n",
                     who, g, x.nm.c_str(), (int)p.form, r);
              return 4;
            }
        }
        eq.terms.push_back(t);
      } else {
        // A pair without a form could go as one term for each row, which is correct but costly.
        // Measured on a small request's passes: 90 s and 46 GB of host memory with one term for
        // each row of the approximate range check's pairs, 3 s and 1.4 GB with their product forms.
        // The two builds differed in other ways too; no run has changed this alone.
        printf("internal error: door: %s %d: array %s: a pair states no product form for its weights on the rows (hd::Pair::form)\n", who, g, x.nm.c_str());
        return 4;
      }
      pairs++;
    }
  }
  if (TESTENV("CHAIN_TIMES"))
    printf("# chain: link %d: the door: %zu arrays, %zu pairs; of their written-out vectors %zu elements stay on the GPU, %zu go to the host\n", D.link + 1, D.t.size(), pairs,
           held_elements, host_elements);
  if (const char *folder = the_chain_puts_vectors ? TESTENV("CHAIN_PUT_VECTORS") : nullptr) {
    // A test variable: the name of an existing directory. Only for runs on synthetic weights, and
    // of one request: the files are created, never overwritten, so a second request, or a second run
    // into the same directory, ends with the internal error below. The files stay on the machine.
    //   <folder>/link_<N>.vectors: the link's coefficient vectors. The verifier computes them by
    //     separate code, and a test build of it (not part of this repository) compares the two results, entry by entry
    //     (prover/hiding/door_vectors.cuh; verifier/tests/coefficient_vectors.h).
    //   <folder>/link_<N>.values: each pair's virtual row, the sum over r of cv[r] * row r of the
    //     committed array (computed from the cells at Term::data), and the first D.ns entries of
    //     the link's mask row: data of the request and the model, and private randomness. With them
    //     that build lists which single claims of a malicious prover's link are false
    //     (prover/hiding/door_values.cuh; verifier/tests/claims_report.h).
    // TESTENV() reads no variable in a production build (`SERVING_BUILD`). Prover::read_isk()
    // rejects this variable at start-up unless the weights are synthetic and the seed is a test
    // seed (made_up_under_a_tests_seed()). Nothing is written here unless link_frame(), the only
    // function that installs this closing function, was told that both hold.
    CK(cudaStreamSynchronize(cudaStreamPerThread));
    const std::string stem = std::string(folder) + "/link_" + std::to_string(D.link + 1);
    std::vector<T> mask_row(D.ns);
    if (D.ns) CK(cudaMemcpy(mask_row.data(), L.masks, D.ns * sizeof(T), cudaMemcpyDeviceToHost));
    for (const char *kind : {".vectors", ".values"}) {
      const std::string path = stem + kind;
      FILE *o = fopen(path.c_str(), "wbx"); // x: create, or fail if anything has the name, a symbolic link included
      if (o && kind[2] == 'e') put_door_vectors(o, door_vectors(D.t, &names));
      if (o && kind[2] == 'a') put_door_values(o, door_values(D.t, ERAD, &names), mask_row);
      if (!o || fclose(o) != 0) {
        printf("internal error: door: %s %d: the test's file %s cannot be created (it must not exist yet), or was not written whole\n", who, g, path.c_str());
        return 4;
      }
    }
  }
  if (g_batch_fetch_mode) {
    const size_t ls0_idx = chain_fetch_queue(D.LS0, D.ns, cudaStreamPerThread);
    chain_fetch_sync_all(cudaStreamPerThread); // the one sync for every read this closing step queued
    chain_apply_eq_patches(eq);
    eq.on_masks = chain_fetch_take(ls0_idx);
    chain_fetch_batch_reset();
  } else {
    eq.on_masks = chain_fetch(D.LS0, D.ns, cudaStreamPerThread);
  }
  C.request->close(std::move(eq));
  return 0;
}
// Installs close_by_frame() as hd::close_fn and keeps C's address. puts_vectors: whether
// CHAIN_PUT_VECTORS may write its files in this process; Prover::chain_make() passes
// made_up_under_a_tests_seed() (prover/common/made_up_alone.h).
static inline void link_frame(ChainHost &C, bool puts_vectors) {
  the_chain = &C;
  the_chain_puts_vectors = puts_vectors;
  close_fn = close_by_frame;
}
} // namespace hd
