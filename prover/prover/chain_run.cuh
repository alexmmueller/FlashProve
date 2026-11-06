// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Runs, on the GPU prover and outside the sub-proofs, one request's first hash chain: its own
// Fiat-Shamir transcript up to the seal, the digest that ends it and on which later chains build.
//
// blocks/iop/req.h fixes the order of a request's proof (the frame): segments (`link` in
// identifiers), each with a group of oracle messages (rows sent as Merkle roots) and closed into
// the next, then one proximity test. A segment carries one sub-proof, the argument for one part of
// the forward pass. ChainRun runs the frame on the GPU engine, between the sub-proofs' kernels:
//   - it reads the arrays that a sub-proof commits to from GPU memory (prover/engine/lay.cuh);
//   - it commits, as the segment's mask row, to the mask entries: the random elements with which
//     the sub-proof's kernels mask the scalars that they send;
//   - the CPU runs the chain: roots, samples, positions, batching and folds. It lends the chain to
//     the GPU for the rounds that the sub-proofs' kernels run (prover/hiding/transcript_shell.cuh).
// Order of calls for a request: begin(); for each segment first(), later() zero or more times,
// close(); then finish(); the caller's own rounds on channel() (a `ProverRounds`), up to the
// chain's seal and end; detach().
#pragma once
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <memory>
#include <set>
#include <vector>

#include "common/end_name.h"
#include "common/setting.h"  // TESTENV
#include "prover/stage_clock.h"  // one-off (not upstream): stage_clock, for a diagnostic mark in close()
#include "engine/engine_gpu.cuh"
#include "engine/lay.cuh"
#include "hiding/lost_copies_door.cuh"
#include "hiding/transcript.cuh"
#include "iop/req.h"
#include "iop/req_arrays.h"
#include "iop/req_rows.h"
#include "iop/row_value.h"
#include "prover/request_part.h"


// With CHAIN_TIMES set in the environment (the production build does not read it), each timed step
// of ChainRun prints one line: the segment, the step, its wall-clock seconds including the wait for
// the GPU stream, and the process's resident memory at the step's end.
struct ChainClock {
  const char *what;
  size_t link;
  cudaStream_t stream;
  double began = 0;
  bool on;
  static double now() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
  }
  ChainClock(const char *w, size_t l, cudaStream_t s) : what(w), link(l), stream(s), on(true) { // one-off (not upstream): forced on; normally TESTENV("CHAIN_TIMES") != nullptr, but
    // cp_prove's reduced_env_start.h re-exec strips TESTENV's underlying environment variable
    // before main() runs, so the runtime switch can never fire; hardcoding `on` is a source-level
    // (not environment-level) way to use this existing instrumentation without touching that
    // anti-LD_PRELOAD mechanism.
    if (on) began = now();
  }
  ~ChainClock() {
    if (!on) return;
    cudaStreamSynchronize(stream);
    long pages = 0, resident = 0;
    if (FILE *f = fopen("/proc/self/statm", "r")) {
      if (fscanf(f, "%ld %ld", &pages, &resident) != 2) resident = 0;
      fclose(f);
    }
    timespec wall;  // the step's end in seconds since the epoch, to align the lines with other logs
    clock_gettime(CLOCK_REALTIME, &wall);
    printf("# chain: link %zu: %-12s %8.3f s, %7.3f GB resident, ends at %.3f\n", link + 1, what, now() - began, (double)resident * (double)sysconf(_SC_PAGESIZE) / 1e9,
           (double)wall.tv_sec + 1e-9 * (double)wall.tv_nsec);
    fflush(stdout);
  }
};

// Fixed at set-up; the same for all requests. `shape`: the segments and their oracle messages.
// `layout`: where each array is in those messages.
struct ChainShape {
  blk::Codes codes;
  blk::ReqShape shape;
  blk::ReqLayout layout;
  blk::ArraySources sources;  // the prover's only: where its cells are stored
};
// The public members of ChainShape as one request's chain binds them in its first query, the hash
// input that fixes parameters and statement before the prover speaks (blocks/chain/public.h).
struct ChainPublic {
  blk::Public<blk::Codes> codes;
  blk::Public<blk::ReqShape> shape;
  blk::Public<blk::ReqLayout> layout;
};

// Converts a field element from the kernels' type `T` to the library's `blk::E`.
static inline blk::E chain_element(const T &x) {
  blk::E e;
  for (int k = 0; k < EDEG; k++) e.c[k].v = coord(x, k);
  return e;
}
// n elements from GPU memory.
// Diagnostic counters (one-off, not upstream): how often this blocking fetch path is used and how
// much data it moves, to check whether it is the loop-of-small-syncs anti-pattern suspected from
// the stage_clock breakdown (door_chain.cuh's close_by_frame calls this once per term/pair).
static unsigned long long g_chain_fetch_calls = 0, g_chain_fetch_elements = 0;
// Set once from args.batch_fetch (inputs.cuh) right after parse(): true runs the batched variant
// (chain_fetch_queue/_sync_all/_take below), false runs the original one-sync-per-call chain_fetch,
// unchanged. Both live in the same binary so -batch-fetch toggles between them at the command line.
static bool g_batch_fetch_mode = false;
static inline std::vector<blk::E> chain_fetch(const T *on_card, size_t n, cudaStream_t stream) {
  g_chain_fetch_calls++, g_chain_fetch_elements += n;
  std::vector<T> got(n);
  if (n) CK(cudaMemcpyAsync(got.data(), on_card, n * sizeof(T), cudaMemcpyDeviceToHost, stream));
  CK(cudaStreamSynchronize(stream));
  std::vector<blk::E> out(n);
  for (size_t i = 0; i < n; i++) out[i] = chain_element(got[i]);
  return out;
}
// Batched variant: chain_fetch_queue() issues the async copy only (no sync) and returns a handle;
// call chain_fetch_sync_all() once after every read of a closing step has been queued; then
// chain_fetch_take() converts each handle's data, in any order, any number of times. Buffers stay
// owned by g_pending_got (never erased mid-step, so earlier handles stay valid) until
// chain_fetch_batch_reset(), called once per close_by_frame() after its reads are all taken.
static std::vector<std::vector<T>> g_pending_got;
static inline size_t chain_fetch_queue(const T *on_card, size_t n, cudaStream_t stream) {
  g_chain_fetch_calls++, g_chain_fetch_elements += n;
  g_pending_got.emplace_back(n);
  if (n) CK(cudaMemcpyAsync(g_pending_got.back().data(), on_card, n * sizeof(T), cudaMemcpyDeviceToHost, stream));
  return g_pending_got.size() - 1;
}
static inline void chain_fetch_sync_all(cudaStream_t stream) {
  if (!g_pending_got.empty()) CK(cudaStreamSynchronize(stream));
}
static inline std::vector<blk::E> chain_fetch_take(size_t idx) {
  const std::vector<T> &got = g_pending_got[idx];
  std::vector<blk::E> out(got.size());
  for (size_t i = 0; i < got.size(); i++) out[i] = chain_element(got[i]);
  return out;
}
static inline void chain_fetch_batch_reset() { g_pending_got.clear(); }

class ChainRun {
public:
  // fixed: kept by reference. A stretch is a run of rounds for which the chain is lent to the GPU.
  // most_rounds, most_draws: the most rounds, and the most challenges drawn, in one stretch; the
  // kernels trap beyond them.
  ChainRun(const ChainShape &fixed, unsigned most_rounds, unsigned most_draws) : fixed_(fixed) { stretch_.init(most_rounds, most_draws); }

  // The transcript object in GPU memory (the lent chain) that the sub-proofs' kernels use. Its
  // address never changes.
  Tr *tr() const { return stretch_.tr; }

  // Starts a request. binder: of the request's first chain, with every public input bound.
  // bound: the public members of `fixed` as bound there (not compared).
  // random_bytes: the chain's salt, from the prover's private randomness (`coins` in identifiers).
  // The engine `on` and `coins` must outlive the request's second chain; requests of one batch
  // share an engine.
  // stream: that of the sub-proofs' kernels and of `on` (not checked): reads of GPU memory are
  // ordered after this stream's work only.
  void begin(std::unique_ptr<blk::Binder> binder, const ChainPublic &bound, const blk::Digest &random_bytes, blk::Coins &coins, blk::EngineGpu &on,
             cudaStream_t stream) {
    end();
    stream_ = stream, bound_ = bound;
    binder_ = std::move(binder);
    engine_ = &on;
    proof_ = std::make_unique<blk::Writer>();
    blk::EngineGpu *engine = engine_;
    channel_ = std::make_unique<blk::ProverRounds>(*binder_, random_bytes, *proof_,
                                                   [engine](const blk::ChainPrefix &p, unsigned bits) { return engine->search_nonce(p, bits); });
    context_ = std::make_unique<blk::ProverContext>(bound_.codes, *channel_, *engine_, coins);
    frame_ = std::make_unique<blk::ReqProver>(*context_, bound_.shape);
    rows_ = std::make_unique<blk::ReqRows<blk::ProverContext>>(*context_, *frame_, bound_.layout);  // it checks the layout against shape and codes
    tables_ = std::make_unique<blk::CardTables>(*engine_, stream);
  }

  // Sends the first prover message of the next segment: its commit point 0 (`moment` in
  // identifiers: a prover message at which a sub-proof's arrays are committed).
  // at: pointers into GPU memory for the commit point's arrays.
  // slots: the sub-proof's first message slot; a slot is the element in GPU memory from which one
  //   scalar is sent. The scalars must be sent from consecutive slots that start there (tr_open(),
  //   in prover/hiding/transcript_shell.cuh, traps otherwise). Mask entry i belongs to slot i.
  // masks, entries: the mask entries that the sub-proof uses, in GPU memory. The sub-proof has
  //   drawn them, uniform and fresh (not checked here). They become the first entries of the
  //   segment's mask row; the rest of the row is drawn here.
  // at_start: for the first segment only: oracle messages that belong to no segment. The first of
  //   them is the index-mask message: masks for the request's claims on the weight table
  //   (index_mask_entry()).
  // word: null, or a 32-bit word in GPU memory that the prover makes known only now: the next
  //   token, which the output head computes. It is sent as the segment's one plain scalar, in a
  //   round of its own right before this prover message, so the chain binds it before the roots,
  //   the seed of the message's samples, the previous segment's positions and the sub-proof's first
  //   challenge.
  // On return the chain is lent. If no segment is left, the shape is indexed out of range before
  // the frame's check.
  void first(const std::vector<const void *> &at, const T *slots, const T *masks, size_t entries, std::vector<blk::OracleDraft> at_start = {},
             const u32 *word = nullptr) {
    take_back();
    const bool index_masks_go_out = !at_start.empty();
    BLK_REQUIRE(slots != nullptr, "the chain: a link's first moment does not say where its run of slots begins");
    slots_go_on_at_ = slots, scalars_of_the_link_ = 0, mask_entries_of_the_link_ = entries;  // a new sub-proof: its first scalar is sent from `slots` and has mask entry 0
    const size_t link = frame_->link();
    const ChainClock clock("first", link, stream_);
    blk::Table row(blk::E_DEGREE, 1, fixed_.codes.mask[fixed_.shape.mask_code].ell);
    BLK_REQUIRE(entries < row.len, "the chain: the sub-proof uses more mask entries than the mask row has beside its pad");
    std::vector<blk::E> drawn = chain_fetch(masks, entries, stream_);
    // Fault-injection hook (prover/tests/lost_pad_copies.sh): zeroes words of the fetched copy, to
    // which the segment commits as its mask row, while the mask entries in GPU memory stay right.
    // No self-check of the prover looks at this copy: the proof is then false and its scalars are
    // still masked.
    lost_copies_door::strike("slot_pads_fetched", drawn.data(), EDEG * entries, false);
    for (size_t j = 0; j < row.len; j++) row.set(0, j, j < entries ? drawn[j] : context_->coins.e());
    // A pad is the random element that masks the answer to an out-of-domain sample. Its places in
    // the row are zeroed: SEND, the block that sends oracle messages (blocks/iop/send.h), draws
    // every pad itself and ends the process if a place is not zero.
    for (const blk::ReqMessage &m : fixed_.shape.link[link].messages)
      if (m.shape.mask_rows) {
        for (const blk::PadPlace &p : blk::Send::pad(fixed_.codes, m.shape)) {
          BLK_REQUIRE(p.part == blk::Part::MASK && p.index >= entries, "the chain: the sub-proof's mask entries reach a pad's place");
          row.set_entry(p.index, blk::E::zero());
        }
        break;
      }
    std::vector<blk::E> plain;
    if (word) {
      u32 w = 0;
      CK(cudaMemcpyAsync(&w, word, sizeof w, cudaMemcpyDeviceToHost, stream_));
      CK(cudaStreamSynchronize(stream_));
      plain.push_back(blk::E::from(blk::from_u64(w)));
    }
    std::vector<blk::TableId> laid;
    {
      const ChainClock part("first: lay", link, stream_);  // writes the arrays into the engine's tables
      laid = tables_->lay(fixed_.codes, fixed_.shape, fixed_.layout, fixed_.sources, link, 0, at);
    }
    {
      // The frame's part. After the first segment it includes the previous segment's code switch
      // (blocks/iop/cs.h).
      const ChainClock part("first: frame", link, stream_);
      own_ = frame_->first(laid, std::move(at_start), std::move(row), std::move(plain));
    }
    if (index_masks_go_out) {
      hold_the_index_masks();
      blk::Table &sent = context_->behind[frame_->at_start().at(0)].mask;
      lost_copies_door::strike("index_masks_sent", sent.a.data(), sent.a.size(), false);  // test hook: loses a value after the check of the whole table and before a sub-proof takes it
    }
    if (link > 0) tables_->give_back(link - 1);  // opened and dropped in `ReqProver::first` above
    lend();
  }
  // Sends the oracle messages of the segment's next commit point. at: as in first(). On return the
  // chain is lent.
  void later(const std::vector<const void *> &at) {
    take_back();
    const ChainClock clock("later", frame_->link(), stream_);
    const std::vector<blk::OracleId> ids =
        frame_->later(tables_->lay(fixed_.codes, fixed_.shape, fixed_.layout, fixed_.sources, frame_->link(), moment_of_next(), at));
    own_.insert(own_.end(), ids.begin(), ids.end());
    lend();
  }
  // Ends the segment's sub-proof and closes the segment. equation: the sub-proof's one linear
  // equation (blocks/iop/req_rows.h); the caller leaves its `target` unset. On return the chain is
  // not lent.
  void close(blk::ReqEquation equation) {
    take_back();
    // One-off (not upstream): close() is shared by every sub-proof's closing (layer, attention,
    // head), but this mark is only meaningful for the layer sub-proof's (stage_clock.h's eqcol_*
    // buckets); guarded so it doesn't relabel the attention/head stages' own time.
    const bool localize_here = stage_clock.current() == StageClock::EqColBuild;
    if (localize_here) stage_clock.at(StageClock::EqColRow);
    // Every scalar that the sub-proof sent must have a mask entry of its own. Three checks:
    // (1) here: the segment's stretches sent as many scalars as the sub-proof drew mask entries,
    // and the equation has as many coefficients on mask entries; (2) the scalars are in one
    // unbroken run of slots, over all the segment's stretches; (3) the run begins at the slot that
    // first() was given. tr_open() traps if (2) or (3) fails: lend() tells each stretch where the
    // last one ended. Otherwise a scalar would be sent under another scalar's entry or unmasked,
    // and no check in cp_prove would notice.
    BLK_REQUIRE(scalars_of_the_link_ == mask_entries_of_the_link_ && equation.on_masks.size() == mask_entries_of_the_link_,
                "the chain: the sub-proof has sent another number of scalars than it drew mask entries, or its equation has another number of coefficients on them");
    const size_t link = frame_->link();
    blk::Row row;
    {
      const ChainClock clock("row", link, stream_);
      row = rows_->row(equation);
    }
    if (TESTENV("CHAIN_TIMES")) {  // the sizes of the equation and of its claim (`row`)
      size_t tensors = 0, with_factor = 0, held = 0, entries = 0, written = 0;
      std::set<const void *> factors;
      for (const blk::MainForm &f : row.forms) {
        tensors += f.c.tensors.size(), entries += f.c.entries.size();
        for (const blk::Tensor &t : f.c.tensors)
          if (t.is_held()) {
            held++;  // prover/hiding/door_chain.cuh prints how many elements stay on the GPU
          } else if (t.has_dense()) {
            with_factor++;
            if (factors.insert(t.values.get()).second) written += t.values->size();
          }
      }
      printf("# chain: link %zu: the equation: %zu terms, %zu coefficients on masks; its row: %zu forms, %zu tensors (%zu with a written-out factor on the host, %zu with one that the engine holds), %zu entries, "
             "%zu runs, %zu units; %zu distinct written-out factors of %zu elements in all\n",
             link + 1, equation.terms.size(), equation.on_masks.size(), row.forms.size(), tensors, with_factor, held, entries, row.runs.size(), row.units.size(),
             factors.size(), written);
    }
    // The claim goes to the frame without its value (`target`). The sub-proofs' kernels
    // never form the claims' constants, so the prover does not know its equation's value on the
    // committed tables. It need not: the value is not sent and nothing is compared with it. It
    // would only be the claim from which the closing's fold starts, and the fold takes that from
    // its own first round (blocks/iop/fold.h, FoldShape::target_from_the_rounds). Evaluating the
    // claim here would be a whole further pass over the segment's arrays, with its kernel launches,
    // for each request.
    // CHAIN_ROW_VALUE, a test variable, asks for that evaluation. With the same private randomness
    // both ways must give the same proof, byte for byte. No test in this tree sets the variable.
    const bool evaluate = TESTENV("CHAIN_ROW_VALUE") != nullptr;
    if (evaluate) {
      const ChainClock clock("value", link, stream_);
      row.target = blk::row_value(*context_, row);
    }
    context_->st.add_row(frame_->group(), std::move(row));
    if (localize_here) stage_clock.at(StageClock::EqColClose);
    {
      const ChainClock clock("close", link, stream_);
      frame_->close(evaluate);
    }
    give_factors_back();  // after the closing's fold: `drop` fails while a fold reads the table
    // Self-check under CHAIN_CHECK_FORMS, a test variable that prover/hiding/door_chain.cuh reads
    // too: the claim that the closing leaves holds on the prover's tables. A proof's bytes cannot
    // show this: they do not depend on the prover's copy of that claim. No test in this tree sets
    // this variable either.
    if (TESTENV("CHAIN_CHECK_FORMS") && !frame_->carried_claim_holds()) {
      printf("internal error: the chain: link %zu: the claim that the closing has left does not hold on the prover's tables\n", link + 1);
      exit(4);
    }
  }
  // Returns a fresh GPU buffer for the vectors of weights on columns (ReqTerm::on_columns) in the
  // present segment's equation, which the sub-proof's kernels write out: millions of elements a
  // segment. The engine reads them there (blocks/iop/tensor.h, Tensor::held), and the CPU does not
  // copy them. The buffer is not initialized. The caller fills one range for each vector, on this
  // thread's stream, before close(), which releases the buffer. The engine cannot read the
  // sub-proof's own arrays instead: they are in working memory that can be overwritten before the
  // next turn (a request's work on one layer group), and nothing checks that they are zero outside
  // the range that counts.
  T *take_factors(size_t elements) {
    static_assert(sizeof(T) == blk::E_DEGREE * sizeof(blk::u64) && sizeof(blk::E) == sizeof(T), "an element in the kernels' memory is the engine's three words");
    BLK_REQUIRE(factors_table_ == blk::NO_ID && elements > 0, "the chain: a link's factors are taken twice, or none is asked for");
    factors_.resize(elements * blk::E_DEGREE);
    factors_table_ = engine_->adopt(factors_.get(), blk::Cell::WORD, blk::E_DEGREE, 1, elements);
    return reinterpret_cast<T *>(factors_.get());
  }
  // The engine's id of that buffer, a table of one row; NO_ID if none is taken.
  blk::TableId factors_table() const { return factors_table_; }
  // Releases the buffer, if one is taken. close() and end() call it.
  void give_factors_back() {
    if (factors_table_ != blk::NO_ID) engine_->drop(factors_table_);
    factors_table_ = blk::NO_ID;
    factors_.release();
  }

  // Call after the last segment's close(): runs the last code switch and the proximity test. The
  // chain does not end here: the caller goes on with its own rounds on channel().
  void finish() {
    const size_t links = fixed_.shape.link.size();
    const ChainClock clock("finish", links, stream_);
    frame_->finish();
    tables_->give_back(links - 1);
  }

  // Takes the next free entry of the index-mask message: the request's oracle message of mask rows
  // for its claims on the weight table. It went out in the request's first prover message and is
  // tested on another chain, not by this frame. The entry's value is copied to GPU memory at
  // `where`, and its `MaskSlot` is kept in index_slots(). Call it only after first() has run for
  // the request's first segment: before that the statement has no entries of the message to give
  // out. The value must be in one oracle message only, so `where` must be outside the mask entries
  // that first() fetches (not checked).
  void index_mask_entry(T *where) {
    blk::MaskSlot slot = context_->st.masks(frame_->at_start().at(0)).take();
    const blk::E value = context_->mask_value(slot);
    // [own-test-index-mask] Compares the entry, before use, with the random value that was drawn
    // for it, as hold_the_index_masks() does for the whole table. The entry, c, masks a value that
    // no entry of the sub-proof's own mask row masks: v = fv + beta c, with fv a linear
    // form of the weights (prover/head/claims_lib.cuh, kG_merge_end()). After that earlier check,
    // this one can find only a value lost after the table's sending. Such a loss gives a false
    // proof, which a verifier rejects too. The check asks only for the drawn
    // value: 0 passes if 0 was drawn, so it does not change c's distribution.
    if (!index_mask_coin_) index_masks_cannot_be_held();
    {  // a block for the three names
      const blk::Table &table = context_->behind[slot.oracle()].mask;
      const u64 row = slot.index() & ((u64(1) << table.log_rows) - 1), place = slot.index() >> table.log_rows;  // as in `Table::entry` (blocks/engine/table.h); coordinate k is a[(row * len + place) * deg + k]
      for (unsigned k = 0; k < blk::E_DEGREE; k++)
        if (!index_mask_coin_((row * table.len + place) * table.deg + k, value.c[k])) {
          printf("internal error (own-test-index-mask): an entry of the request's index-mask message is not the random value that was drawn for it. The program or the machine has a fault; nothing of this "
                 "run may be used\n");
          name_the_end("own-test-index-mask");
          exit(4);
        }
    }
    const u64 c[EDEG] = {value.c[0].v, value.c[1].v, value.c[2].v};
    const T x = ext_from(c);
    CK(cudaMemcpyAsync(where, &x, sizeof x, cudaMemcpyHostToDevice, stream_));
    CK(cudaStreamSynchronize(stream_));  // x is about to go out of scope
    index_slots_.push_back(std::move(slot));
    index_value_ = value;
  }
  // Whether `cell`, in GPU memory, still has the value of the entry that was taken last. For a
  // check after the segment's last kernel: a lost copy would leave zeros there (the value would be
  // sent unmasked) or the previous segment's entry (one mask for two values). False while
  // index_slots() is empty.
  bool index_mask_entry_holds(const T *cell) const { return !index_slots_.empty() && chain_fetch(cell, 1, stream_)[0] == index_value_; }
  // For [own-test-index-mask]. The code that draws the index-mask message's table calls this right
  // after the draw, and after begin(), which clears both functions. In the prover program cp_prove
  // both recompute the draw by a second implementation of the rule that derives private random
  // values. coin(i, x): whether x is element i of the draw. differ(values, n, left_out): how many
  // of the draw's n elements, read at `values`, are not the drawn ones; left_out: indices to skip,
  // ascending. Without this call the process exits at the first check that needs either function.
  // Like a failed check, that is exit(4) after name_the_end(); going by that name, cp_prove's exit
  // handler (prover/prover/ends.h) replaces the 4 with its exit code for a failed self-check.
  void index_masks_were_drawn(std::function<bool(u64, blk::F)> coin, std::function<u64(const blk::F *, u64, const std::vector<u64> &)> differ) {
    index_mask_coin_ = std::move(coin), index_masks_differ_ = std::move(differ);
  }
  // The `MaskSlot`s of the entries taken, in that order; each can be used once. The caller moves
  // each into the pending claim on the weight table that names it; detach() requires an empty list.
  std::vector<blk::MaskSlot> &index_slots() { return index_slots_; }

  // take_back(), lend(): for the caller's own rounds between the calls above, on the CPU or in
  // kernels. take_back() waits for the stream's work, replays the stretch's rounds on the CPU and
  // ends the process unless it reaches the GPU's chain value. While the chain is lent, sending an
  // item, closing a round or ending the chain on channel() ends the process
  // (blocks/chain/rounds.h).
  void take_back() {
    if (lent_) {
      drawn_ = stretch_.end(*channel_, stream_);
      slots_go_on_at_ = stretch_.next_slot, scalars_of_the_link_ += stretch_.sent_scalars;
    }
    lent_ = false;
  }
  void lend() {
    if (!lent_) stretch_.begin(*channel_, stream_, slots_go_on_at_);
    lent_ = true;
  }
  // The challenges drawn in the stretch that was taken back last.
  const std::vector<blk::E> &drawn() const { return drawn_; }

  blk::ProverRounds &channel() { return *channel_; }
  blk::ProverContext &context() { return *context_; }
  const blk::ReqFrame &frame() const { return *frame_; }
  // The chain's bytes, complete once the caller has ended the chain on channel().
  const std::vector<blk::u8> &bytes() const { return proof_->bytes(); }

  // Call when the first chain is sealed and ended (not checked here). Moves binder, bytes, channel
  // and context into `part`, where they wait for the request's second chain
  // (prover/prover/batch_opening.h). The caller fills in lin, lin_first and seal. The run is then
  // free for another request.
  void detach(RequestPart &part) {
    BLK_REQUIRE(!lent_ && context_ && index_slots_.empty(), "the chain: the run is not at its end, or an index-mask entry was taken and stands in no row");
    part.binder = std::move(binder_), part.first = std::move(proof_), part.channel = std::move(channel_), part.context = std::move(context_);
    end();
  }
  // Ends the run, finished or not, and releases everything of the request that it still owns.
  void end() {
    lent_ = false;
    if (engine_) give_factors_back();
    tables_.reset(), rows_.reset(), frame_.reset(), context_.reset(), channel_.reset(), proof_.reset(), binder_.reset();
    engine_ = nullptr;
    own_.clear(), drawn_.clear(), index_slots_.clear();
    index_value_ = blk::E::zero();
    index_mask_coin_ = nullptr, index_masks_differ_ = nullptr;
  }

private:
  [[noreturn]] static void index_masks_cannot_be_held() {
    printf("internal error (own-test-index-mask): the prover was not told which random draw made the request's index-mask message, so its self-check cannot run. The program has a fault; nothing of this run may be used\n");
    name_the_end("own-test-index-mask");
    exit(4);
  }
  // [own-test-index-mask] Compares the whole table of the index-mask message, right after its
  // sending, with the second implementation (index_masks_were_drawn()), then its random tail. The
  // sub-proofs take some of the table's entries (index_mask_entry()); the test on the weight table
  // reads others, as masks of sums that are linear in the weights. This CPU table is their one
  // source: the GPU's cell, index_value_, the pending claims and the commitment are all copies of
  // it, so they agree whatever it contained at its sending. A table that lost values between the
  // draw and its sending would give a proof that verifies, with fv (index_mask_entry()) or such a
  // sum unmasked. The pad's places are not compared here: Send::fresh_mask() leaves them zero and
  // send_prove() fills them with draws of its own, which it reads back itself
  // (own-test-pad-kept, blocks/iop/send.h).
  // Not seen either: the copies that send_oracles() gave the engine to encode, made before this
  // check. The counterpart for the key file is not_as_drawn() (blocks/iop/send.h).
  void hold_the_index_masks() {
    if (!index_masks_differ_) index_masks_cannot_be_held();
    const blk::Table &table = context_->behind[frame_->at_start().at(0)].mask;
    const blk::OracleShape &shape = fixed_.shape.at_start.at(0).shape;
    std::vector<u64> left_out;  // the pad's places
    if (shape.answers > 0)
      for (const blk::PadPlace &p : blk::Send::pad(fixed_.codes, shape)) {
        BLK_REQUIRE(p.part == blk::Part::MASK, "the chain: the index-mask message has a pad outside its mask rows");
        const u64 row = p.index & ((u64(1) << table.log_rows) - 1), place = p.index >> table.log_rows;  // as in index_mask_entry()
        for (unsigned k = 0; k < blk::E_DEGREE; k++) left_out.push_back((row * table.len + place) * table.deg + k);
      }
    std::sort(left_out.begin(), left_out.end());
    if (const u64 differ = index_masks_differ_(table.a.data(), table.a.size(), left_out)) {
      printf("internal error (own-test-index-mask): %llu of the %llu checked values of the request's index-mask message are not the random values that were drawn for it. The program or the machine has a fault; "
             "nothing of this run may be used\n", (unsigned long long)differ, (unsigned long long)(table.a.size() - left_out.size()));
      name_the_end("own-test-index-mask");
      exit(4);
    }
    // The random tail: the random coefficients that send_oracles() (blocks/iop/context.h) draws for
    // each mask row's codeword. send_oracles() records the draw's number (`tail_draw`); by that
    // number the request's generator counts the coefficients that differ from the draw
    // (Coins::not_the_coins(), which cp_prove's `differ` calls too). With a generator that does not
    // number its draws (Coins::NO_DRAW), such as a unit test's, the tail is not compared.
    const blk::ProverOracle &sent = context_->behind[frame_->at_start().at(0)];
    if (const u64 differ = sent.tail_draw == blk::Coins::NO_DRAW ? 0 : context_->coins.not_the_coins(sent.tail_draw, sent.mask_tail.a.data(), sent.mask_tail.a.size(), {})) {
      printf("internal error (own-test-index-mask): %llu of the %llu values of the random tail of the request's index-mask message are not the random values that were drawn for it. The program or the machine has a fault; "
             "nothing of this run may be used\n", (unsigned long long)differ, (unsigned long long)sent.mask_tail.a.size());
      name_the_end("own-test-index-mask");
      exit(4);
    }
  }
  // The commit point that later() is about to run: that of the segment's first oracle message that
  // has not been sent. That one is left is not checked here.
  unsigned moment_of_next() const { return fixed_.shape.link[frame_->link()].messages[own_.size()].moment; }

  const ChainShape &fixed_;
  ChainPublic bound_;
  Stretch stretch_;
  const T *slots_go_on_at_ = nullptr;                                // where the present segment's next scalar must be sent from; never null after `first`
  size_t scalars_of_the_link_ = 0, mask_entries_of_the_link_ = 0;  // scalars sent in the segment's stretches so far; `entries` of `first`
  cudaStream_t stream_ = 0;
  bool lent_ = false;
  blk::EngineGpu *engine_ = nullptr;
  std::unique_ptr<blk::Binder> binder_;
  std::unique_ptr<blk::Writer> proof_;
  std::unique_ptr<blk::ProverRounds> channel_;
  std::unique_ptr<blk::ProverContext> context_;
  std::unique_ptr<blk::ReqProver> frame_;
  std::unique_ptr<blk::ReqRows<blk::ProverContext>> rows_;  // makes a segment's claim from its equation (blocks/iop/req_rows.h)
  std::unique_ptr<blk::CardTables> tables_;
  blk::Card<blk::u64> factors_;                // take_factors()
  blk::TableId factors_table_ = blk::NO_ID;
  std::vector<blk::OracleId> own_;  // the present segment's oracle messages so far
  std::vector<blk::E> drawn_;
  std::vector<blk::MaskSlot> index_slots_;
  blk::E index_value_ = blk::E::zero();  // of the entry that was taken last
  std::function<bool(u64, blk::F)> index_mask_coin_;  // see `index_masks_were_drawn`. May hold a bare pointer to the request's `coins`, which `clear_slot` (prover/prover/slot.h) destroys; `end` drops it
  std::function<u64(const blk::F *, u64, const std::vector<u64> &)> index_masks_differ_;  // likewise
};
