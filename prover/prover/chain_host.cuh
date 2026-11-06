// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// What a request's sub-proofs (one argument per part of the forward pass) share when they commit to
// their arrays. The plan: which arrays each one commits to, at which commit point, and how each
// is read from GPU memory. `ChainShape` is what follows for all requests from the plan, the layout
// file (`layout.txt`) and the parameter file (`policy.txt`). Each request in progress has a
// ChainRun (prover/prover/chain_run.cuh). A caller must select a request's run, by enter(), and its
// attempt (below), by hz::set_attempt() or hz::resume_attempt(), together: see hold_the_attempt().
//
// A request's proof has a sequence of segments (`link` in identifiers and messages), one per
// sub-proof. The frame (blocks/iop/req.h) commits to each segment's arrays in oracle messages
// (encoded rows under Merkle roots), takes an out-of-domain sample of each, and closes the
// segment's one linear equation over the arrays into the next segment. Each of the NGr layer groups
// has a layer sub-proof, for its layers except the attention function, and an attention sub-proof.
// The order: layer 0, attention 0, layer 1, ..., output head. In identifiers and messages:
//  `chain`   the sequence of segments; the proof's first Fiat-Shamir transcript is "the hash chain"
//  `moment`  commit point, one of a sub-proof's numbered commitments (prover/hiding/moments.cuh)
//  `wave`    batch pass: the requests that are proved in step. They take turns, and all of them
//            finish one layer group before any of them starts the next.
//  `attempt` the proving of one request; randomness is derived from its counter (hz::Replay::ctr)
#pragma once
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "frame/req_frame.h"
#include "hiding/late.cuh"
#include "hiding/coins_long.cuh"
#include "hiding/place.cuh"
#include "hiding/product_test.cuh"
#include "prover/chain_run.cuh"

// An array as a sub-proof's set-up declares it: a matrix over the field F or its extension E.
struct ChainArray {
  std::string name;
  size_t R = 1, K = 1;       // rows and columns of the matrix of committed elements
  blk::ArraySource source;   // how element (r, c) is read, and from which of the moment's pointers
};
struct ChainLink {
  std::string name;
  u32 gid = 0;  // the sub-proof's public identifier, a parameter in the hash chain's first query
  // The arrays of each commit point, in the order in which the layout file numbers them. At the
  // first: those that only this segment reads, then those that later segments read too.
  std::vector<std::vector<ChainArray>> moment;
  // Mask-row entries in GPU memory, drawn by the sub-proof before first() to mask its sent scalars.
  const T *masks = nullptr;
  size_t mask_entries = 0;
  // Earlier segments' arrays in this segment's equation, each as (committing segment, name there),
  // in the order of the `foreign` terms in hd::close_by_frame() (prover/hiding/door_chain.cuh).
  std::vector<std::pair<int, std::string>> reads;
  int find(int m, const std::string &array) const {
    for (size_t a = 0; a < moment[m].size(); a++)
      if (moment[m][a].name == array) return (int)a;
    return -1;
  }
};

// The plan, the shapes and the runs. A prover has one (Prover::chain).
struct ChainHost {
  int NGr = 0; // number of layer groups
  // Segments count from 0 here and in the frame's calls, from 1 in the layout file.
  int layer(int g) const { return 2 * g; }
  int attention(int g) const { return 2 * g + 1; }
  int head() const { return 2 * NGr; }
  int links() const { return 2 * NGr + 1; }

  std::vector<ChainLink> plan; // set-up: sub-proofs add later commit points, `chain_make` the rest
  ChainShape fixed;            // made once, by links_from() and settle()
  // One run per request of a batch pass, created at set-up. enter(j) points `request` at run j.
  std::vector<std::unique_ptr<ChainRun>> in_step;
  // One-off (not upstream): one CUDA stream per in_step slot, used only if g_parallel_streams
  // (common/setting.h) is set; else every slot's stream is cudaStreamPerThread, as upstream. The
  // requests of a batch pass are cryptographically independent of one another (separate
  // Fiat-Shamir transcripts, separate nonces), so nothing about correctness requires funneling
  // their GPU work through one stream; GPU memory used by Pooled<T> (prover/engine/pooled.cuh)
  // comes from CUDA's stream-ordered pool allocator, which is built to be shared safely by
  // several streams (card_arena(), the one case that is not, is never engaged by this binary:
  // InArena is not instantiated anywhere outside arena.h in this repository).
  std::vector<cudaStream_t> streams;
  cudaStream_t stream_of(int member) const { return streams.empty() ? cudaStreamPerThread : streams[member]; }
  ChainRun *request = nullptr;
  int member = 0; // the index in `in_step` of the run that `enter` selected
  void enter(int member_) { member = member_, request = in_step[member_].get(); }
  // Before an exit on an internal error: releases what the runs hold of their requests.
  void wipe_the_wave() {
    for (auto &r : in_step) r->end();
  }

  // hold_the_attempt() checks this rule: a request's begin(), first(), later(), close() and
  // finish() must run while that request's attempt is the current one (hz::replay_()), because
  // prover/hiding/trees.cuh derives randomness from the current attempt's numbers, whichever run is
  // selected. If code drew it for one request under another's attempt, two requests could get the
  // same random values. Under a secret key the two would then hide nothing from each other, and no
  // verifier would notice. begin()'s and finish()'s callers make the check; close()'s does not.
  static constexpr u64 NOT_BEGUN = ~0ull;
  // For each run, the counter of the attempt under which begin() was called, or NOT_BEGUN.
  std::vector<u64> begun_under;
  void hold_the_attempt(const char *call) const {
    const hz::Replay &at_hand = hz::replay_();
    if (at_hand.set && at_hand.ctr == begun_under[member]) return;
    printf("internal error: the chain: %s for request %d of the wave, begun under attempt %llu, but ", call, member,
           (unsigned long long)begun_under[member]);
    if (at_hand.set)
      printf("attempt %llu is at hand\n", (unsigned long long)at_hand.ctr);
    else
      printf("no attempt is at hand\n");
    exit(4);
  }

  u32 *no_words = nullptr; // GPU memory: ROOTW zero words, for `chain_bind` (chain_frame.cuh)
  std::vector<std::vector<const void *>> first_pointers; // per segment: its first arrays' pointers
  // Set by chain_make(). Makes the first prover message's oracle messages that are in no segment.
  std::function<std::vector<blk::OracleDraft>()> at_start;

  void init(int layer_groups) {
    NGr = layer_groups;
    plan.assign(links(), {});
    first_pointers.assign(links(), {});
    hz::alloc(no_words, (size_t)ROOTW * 4, hz::OWN);
    CK(cudaMemset(no_words, 0, (size_t)ROOTW * 4));
  }

  // The plan as blk::req_frame() takes it, `life` from later segments' `reads`. Null or the fault.
  const char *declare(std::vector<blk::FrameLink> &declared) const {
    declared.clear();
    for (const ChainLink &l : plan) {
      blk::FrameLink x;
      x.name = l.name;
      for (const std::vector<ChainArray> &m : l.moment) {
        x.moment.emplace_back();
        for (const ChainArray &a : m) x.moment.back().push_back(blk::FrameArray{a.name, a.R, a.K, a.source.cells == blk::ArrayCells::ELEMENTS, 0});
      }
      declared.push_back(x);
    }
    for (size_t l = 0; l < plan.size(); l++) // `life`: segments from an array to its last reader
      for (const std::pair<int, std::string> &r : plan[l].reads) {
        if (r.first < 0 || (size_t)r.first >= l) return "a link reads an array of a link that does not come before it";
        for (blk::FrameArray &a : declared[(size_t)r.first].moment.at(0))
          if (a.name == r.second) a.life = std::max<unsigned>(a.life, (unsigned)(l - (size_t)r.first));
      }
    return nullptr;
  }
  // Prints the arrays for prover/tools/pack_layout.py --arrays, in the format of the verifier's
  // `cp_verify -plan arrays`: "array: link moment group class life index rows columns low name".
  // link, moment, group (here one oracle message): from 1; index in the moment: from 0. Arrays that
  // no later segment reads share a group: over F the first (class N), over E the next (W). Every
  // other array has its own (B). low: r if row_stride <= column_stride, else c; only a hint.
  void print_arrays(FILE *to) const {
    std::vector<blk::FrameLink> declared;
    if (const char *fault = declare(declared)) {
      fprintf(to, "arrays: %s\n", fault);
      return;
    }
    for (size_t l = 0; l < declared.size(); l++)
      for (size_t m = 0; m < declared[l].moment.size(); m++) {
        const std::vector<blk::FrameArray> &arrays = declared[l].moment[m];
        bool over_F = false, over_E = false;
        for (const blk::FrameArray &a : arrays)
          if (!a.life) (a.over_E ? over_E : over_F) = true;
        unsigned kept = (over_F ? 1 : 0) + (over_E ? 1 : 0);
        for (size_t i = 0; i < arrays.size(); i++) {
          const blk::FrameArray &a = arrays[i];
          const blk::ArraySource &s = plan[l].moment[m][i].source;
          const unsigned group = a.life ? ++kept : a.over_E && over_F ? 2 : 1;
          fprintf(to, "array: %zu %zu %u %c %u %zu %llu %llu %c %s\n", l + 1, m + 1, group, a.life ? 'B' : a.over_E ? 'W' : 'N', a.life, i,
                  (unsigned long long)a.rows, (unsigned long long)a.columns, s.row_stride <= s.column_stride ? 'r' : 'c', a.name.c_str());
        }
      }
  }
  // Set-up, step 1 of 2, once the plan is complete: derives the segments' oracle messages and their
  // layout from the plan and the layout file's text. answers: the number of answers to each sample.
  const char *links_from(const std::string &layout, unsigned answers) {
    std::vector<blk::FrameLink> declared;
    if (const char *fault = declare(declared)) return fault;
    blk::LayoutLines lines;
    if (const char *fault = blk::layout_read(layout, lines)) return fault;
    return blk::req_frame(declared, lines, answers, fixed.shape, fixed.layout);
  }
  // Step 2: takes the codes and the whole shape that blk::sized() (blocks/frame/files.h) made from
  // the parameter file and step 1's shape, and checks them. Both steps return null or the fault.
  const char *settle(const blk::Codes &codes, const blk::ReqShape &whole) {
    fixed.codes = codes, fixed.shape = whole;
    if (const char *fault = blk::req_fault(fixed.codes, fixed.shape)) return fault;
    if (const char *fault = blk::req_layout_fault(fixed.codes, fixed.shape, fixed.layout)) return fault;
    fixed.sources.clear();
    for (size_t l = 0; l < plan.size(); l++) {
      fixed.sources.emplace_back();
      for (const blk::ReqArray &a : fixed.layout.link[l]) {
        const unsigned m = fixed.shape.link[l].messages[a.message].moment;
        fixed.sources.back().push_back(plan[l].moment[m][(size_t)plan[l].find((int)m, a.name)].source);
      }
      if (plan[l].mask_entries + fixed.shape.answers > fixed.codes.mask[fixed.shape.mask_code].ell) return "a link's scalars outnumber the mask row's entries";
      for (const std::pair<int, std::string> &r : plan[l].reads) {  // are these arrays still kept?
        const blk::ReqArray *a = fixed.layout.find((size_t)r.first, r.second);
        if (!a || r.first < 0 || (size_t)r.first >= l || (size_t)r.first + fixed.shape.link[(size_t)r.first].messages[a->message].life < l)
          return "a link reads an array of an earlier link that the layout file does not keep so long";
      }
    }
    return blk::req_sources_fault(fixed.shape, fixed.layout, fixed.sources);
  }

  // Commits to the segment's first arrays and mask row and carries over the previous one's closed
  // claim, in one prover message; then lends the hash chain to the GPU via tr. It is not checked
  // that the run is at segment `link`. y: null or the next token in GPU memory, sent here unmasked.
  void first(int link, Tr *tr, const u32 *y = nullptr) {
    hold_the_attempt("first()");
    hold_the_transcript(tr);
    // Scalar i is stored PAD_STRIDE (prover/hiding/sumcheck_door.cuh) elements before mask entry i.
    request->first(first_pointers[link], plan[link].masks - PAD_STRIDE, plan[link].masks, plan[link].mask_entries,
                   link == 0 && at_start ? at_start() : std::vector<blk::OracleDraft>(), y);
  }
  // Runs a later commit point: takes the hash chain back, commits, samples, lends the chain again.
  void later(int link, int moment, const std::vector<const void *> &arrays, Tr *tr) {
    (void)link, (void)moment; // the frame counts both and ends the process if a moment is left out
    hold_the_attempt("later()");
    hold_the_transcript(tr);
    request->later(arrays);
  }
  void hold_the_transcript(const Tr *tr) const {
    if (tr == request->tr()) return;
    printf("internal error: the chain: a sub-proof runs on another transcript object than its request's\n");
    exit(4);
  }
};
