// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Proves the requests (`member` in identifiers) of a batch pass side by side. A batch pass (`wave`
// in identifiers; "wave" or "pass" in messages) is the part of a batch, at most -in-step requests
// (an option of the prover program cp_prove), proved on one pass over the weights. A turn is one
// request's work on one layer group or, last, on the output head: each request has its turn in a
// group before any goes on to the next group.
// The sub-proofs (`fork` in identifiers; each proves one part of the forward pass) work at fixed
// GPU addresses, the working memory, so the requests take turns on it. `place` is a request's index
// in its batch pass. Place p has its own ChainRun, which runs the part of a request's proof outside
// its sub-proofs (prover/prover/chain_run.cuh), and state buffer p: a bank (MemberState), GPU
// memory that keeps the request's part of the working memory between turns.
// prove_wave() (prover/prover/wave_order.cuh) calls seat_begins(), seat_turn() for each layer group
// and seat_ends() for each non-empty slot (a request's index in its batch), then wave_has_ended().
// It must call seat_begins() in ascending order of Slot::attempt, the counter of the slot's
// blinding context, which is the order of the slots in their batch: hz::set_attempt() exits on a
// counter that is not above every earlier one under the same key. The calls maintain:
//  - One path: every non-empty slot takes every step, whatever its failure word (`flags` in
//    identifiers) holds. (A request with a bit set in it gets a `.failed` file and no proof.)
//  - A slot's failure word is its own. Host and GPU have one live word each. The host's is the
//    slot's from take_the_seat() to leave_the_seat(), the GPU's from MemberState::begin() or load()
//    to save() or end(), in stream order, for kernels whose stream has by then joined the caller's.
//  - Every turn starts on the same memory. Before each turn, even if the same slot had the previous
//    one, the working memory is restored (`wipe` in identifiers and messages): zeroed, or given
//    back its content at the end of set-up, except what MemoryAsSetUp skips. Then the slot's state
//    is copied in from its bank. So what a turn finds in the restored memory, even where a step
//    writes nothing, depends only on the slot's own state, not on the order of the turns or on the
//    other requests. With one state buffer a batch pass has one request, so turns skip the restore.
//  - After a batch pass the GPU memory in ForkMemory's record is as set-up left it or, if allocated
//    later, zero, except the weights, the buffers they are streamed into and ForkMemory::laid_once.
//    Restores reach only parts of ForkMemory's record (prover/prover/memory_as_set_up.cuh says
//    which) and the sub-proofs' scratch memory (the slab), not prover/engine's pool
//    (docs/limits-all-rows.tsv, row H39.5).
//  - A slot's private random values are derived only under its own blinding context (`attempt`),
//    whose counter enters every derivation: hz::coink() exits if none is current, and before
//    ChainRun::begin(), first(), later() or finish() hold_the_attempt() exits under another slot's.
// In identifiers `seat_` is a slot (seat_begins()) or a state buffer (`seat_at_hand`).
#pragma once
#include "common/refusal_test.h"
#include "prover/member_state.cuh"
#include "prover/prover.h"

// Lists what a turn loads from its bank (`reads`) and saves there (`writes`). Turn g < NGr: layer
// group g. NGr (loads only): output head and request's end. NGr + 1 (saves only): begin_request().
inline void Prover::plan_member_state(int members) {
  using Buffers = MemberState::Buffers;
  if (fork_memory.test.own_memory) {
    printf("memory_test: the mode own is not supported, because requests keep their state in separate state buffers; "
           "use the mode scribble or disjoint\n");
    exit(2);
  }
  enum { Small, Stream, Border, Late, Pads, Results };
  // Pointer and allocated size of the entire buffer p.
  auto whole = [](void *p) {
    return std::make_pair(p, fork_memory.device_buffers.at(p));
  };
  Buffers small;
  for (void *p : {(void *)hdr, (void *)tokd, (void *)rootCP, (void *)CPh, (void *)efs, (void *)Aef, (void *)err,
                  (void *)errc, (void *)errf, (void *)slot_pads_agree, (void *)filler_is_zero})
    small.push_back(whole(p));
  small.push_back(whole(tr)); // set-up's buffer; `take_the_seat` points `tr` at a `ChainRun`'s own
  for (auto *Hd : AO) small.push_back(whole(Hd->errK));
  small.push_back(whole(attF.error_words()));
  std::vector<MemberState::Turn> turns(NGr + 2);
  for (int g = 0; g <= NGr; g++) {
    auto &t = turns[g];
    t.reads.push_back({Small, small});
    t.reads.push_back({Pads, pad_tree_buffers});
    if (g > 0) {
      t.reads.push_back({Stream, {whole(xin[g])}});
      t.reads.push_back({Border, {whole(XBv[g])}});
      t.reads.push_back({Late, fork_memory.lasting_of(g - 1)});
    }
    if (g < NGr) {
      t.writes.push_back({Small, small});
      t.writes.push_back({Pads, pad_tree_buffers});
      t.writes.push_back({Stream, {whole(xin[g + 1])}});
      t.writes.push_back({Border, {whole(XBv[g + 1])}});
      t.writes.push_back({Late, fork_memory.lasting_of(g)});
      t.writes.push_back({Results + g, fork_memory.results_of(g)});
    } else {
      for (int q = 0; q < NGr; q++) t.reads.push_back({Results + q, fork_memory.results_of(q)});
    }
  }
  turns[NGr + 1].writes.push_back({Small, small});
  turns[NGr + 1].writes.push_back({Pads, pad_tree_buffers});
  fork_memory.purpose = "the requests' state between layer groups";
  member_state.build(members, turns);
  fork_memory.purpose = "other buffers of the prover";
  printf("# proving %d requests at a time; each needs %.4f GB of GPU memory for its state between layer groups\n",
         members, member_state.bytes_a_member / 1e9);
}

// Call it once, at the end of set-up: after chain_make() and before the first request is read. It
// allocates the banks and calls MemoryAsSetUp::look(), whose scan decides what restores bring back.
inline int Prover::set_up_the_seats() {
  // No code fills chain_own_regions: it gets one empty list per state buffer.
  if (chain_own_regions.empty()) chain_own_regions.resize(args.in_step);
  chain_seat_regions.resize(chain_own_regions.size());
  seats = (int)chain_own_regions.size();
  if (args.in_step < 1 || args.in_step > SLOTS_IN_STEP_MOST || args.batch < 1 || args.batch > SLOTS_MOST ||
      seats != args.in_step) {
    printf("internal error: %d requests at a time (1 to %d are supported) with %d state buffers, in batches of %d (1 "
           "to %d)\n",
           args.in_step, SLOTS_IN_STEP_MOST, seats, args.batch, SLOTS_MOST);
    return 4;
  }
  plan_member_state(seats);
  seats_state = kept_between_turns();
  // Zeroed before look(), which would take leftovers of set-up in these regions for constants.
  // (No code fills chain_shared_regions either.)
  for (auto &b : chain_shared_regions) CK(cudaMemset(b.first, 0, b.second));
  for (auto &regions : chain_own_regions)
    for (auto &b : regions) CK(cudaMemset(b.first, 0, b.second));
  CK(cudaDeviceSynchronize());
  memory_as_set_up.look();
  if (fork_memory.test.left_behind) fork_memory.look_at_what_stays();
  return 0;
}

// A stub of a test hook: `refusal_test::seat_bits()` returns 0 in every build from these
// sources, so this function sets no bit of the failure word: the bits are 0 and no kernel is
// launched. `where` and n name the point of its proof that slot s has reached.
// Public sources: such a hook is a fail point (here a stub, which acts in no build), as
// in the Rust crate fail, which is inspired by FreeBSD's: The Rust crate "fail" (a fail point
// implementation for Rust), crate documentation, https://docs.rs/fail. That crate's fail points act
// in host code; here a hook also names a batch slot.
// Public sources, for "in stream order" in this file's head comment: NVIDIA, CUDA C++ Programming
// Guide, "Asynchronous Concurrent Execution", "Streams",
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html.
// See also: NVIDIA, CUDA Programming Guide, "Asynchronous Execution",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html,
// sections "CUDA Streams" and "CUDA Stream Ordering".
static inline void raise_what_a_test_asks(const Slot &s, const char *where, int n) {
  raise_flags(refusal_test::seat_bits(s.j, where, n, false));
}

// Makes slot s the one that the next steps work on, with the ChainRun and state buffer of `place`.
inline void Prover::take_the_seat(Slot &s, int place) {
  if (place < 0 || place >= seats) {
    printf("internal error: slot %d was given state buffer %d of %d\n", s.j, place, seats);
    exit(4);
  }
  chain.enter(place);
  tr = chain.request->tr(); // each `ChainRun` has its own transcript object, at a fixed address
  seat_at_hand = place;
  enter_slot(s);
  rep = s.attempt;
}
// Ends the steps' work on slot s for now.
inline void Prover::leave_the_seat(Slot &s) {
  leave_slot(s);
}

// Exits on an internal error, such as a failed check of the plan. Meant to depend on no request.
[[noreturn]] static inline void road_is_broken(ChainHost &chain, int rc) {
  chain.wipe_the_wave();
  exit(rc);
}

// Starts the slot's request.
inline void Prover::seat_begins(Slot &s, int place) {
  stop_if_asked();
  weight_stream.count_waits = args.measure && args.stream_weights; // wave_has_ended() prints what was counted
  SERVING_OR(, stage_clock.on = args.measure); // likewise
  take_the_seat(s, place);
  stage_clock.at(StageClock::Start);
  begin_request(s);
  stage_clock.at(StageClock::Turns);
  raise_what_a_test_asks(s, "begin", -1);
  member_state.save(place, NGr + 1);
  hz::park_attempt();
  leave_the_seat(s);
}

// The slot's turn in layer group g; it loads state that turn g - 1 or, if g = 0, seat_begins()
// saved. With one state buffer no other request takes a turn in between, so memory is restored only
// at a request's start and a batch pass's end: a turn finds what the request's earlier turns left
// (docs/limits-all-rows.tsv, row H39.5).
// Public source for leaving out a restore where no one else comes in between: the Linux kernel's
// documentation, "L1D Flushing", has a similar rule for a cache and the process of the next task;
// here the number of state buffers, fixed at set-up, decides. The omission is a saving, not a need.
inline void Prover::seat_turn(Slot &s, int place, int g) {
  stop_if_asked();
  take_the_seat(s, place);
  hz::resume_attempt(s.attempt);
  if (seats > 1) wipe_the_working_memory(Before::ATurn); // with one state buffer: see this file's head
  member_state.load(place, g);
  raise_what_a_test_asks(s, "turn", g);
  if (g == 0 && fork_memory.test.failed_seat == s.j) raise_flags(UNFIT); // the memory test's switch; -1 elsewhere
  if (int rc = prove_layer_group(g)) road_is_broken(chain, rc);
  stage_clock.at(StageClock::Turns);
  member_state.save(place, g);
  if (fork_memory.test.changed_table && !batch.number && s.j == 0 && g == NGr - 1) // a switch of the memory test
    memory_as_set_up.change_a_table(cudaStreamPerThread);
  hz::park_attempt();
  leave_the_seat(s);
}

// The slot's last turn: the output head's sub-proof and the request's end, up to its first chain's
// seal, a leaf of the batch's binding tree. Public source for the omitted restore: see seat_turn().
inline void Prover::seat_ends(Slot &s, int place) {
  stop_if_asked();
  take_the_seat(s, place);
  hz::resume_attempt(s.attempt);
  if (seats > 1) wipe_the_working_memory(Before::TheHeadsTurn);
  member_state.load(place, NGr);
  if (int rc = prove_head_fork()) road_is_broken(chain, rc);
  raise_what_a_test_asks(s, "head", -1);
  stage_clock.at(StageClock::End);
  finish_request(s);
  stage_clock.at(StageClock::Turns);
  leave_the_seat(s);
  opening_of_the_batch->order->put(s.part);
}

// Call it when every slot of a batch pass has ended.
inline void Prover::wave_has_ended() {
  stage_clock.at(StageClock::Turns); // -measure's last border: the wipe below waits for the GPU
  wipe_behind_the_wave();
  // ForkMemory's peak (record and mapped slab) must not rise after this: a rise is only reported.
  if (!waves_ended++) fork_memory.hold_to_what_is_taken();
  // How far the restore before a turn reaches into the slab (the sub-proofs' scratch memory), and
  // the most that fold_group_for_opening() decodes into its start. No test checks these numbers.
  if (args.measure)
    printf("# memory as set-up left it: behind wave %d: of the slab's %zu mapped bytes the sub-proofs have taken at most %zu, "
           "the stretch lent to the chain library ends at %zu, and a wipe before a turn zeroes the first %zu; one unit's "
           "decoded weights take at most %zu\n",
           waves_ended, fork_memory.mapped, fork_memory.peak, fork_memory.lent_to, MemoryAsSetUp::slab_bytes(0),
           decoding_needs_at_most());
  if (args.measure) { // this batch pass's time and GPU memory; a batch's line sums the times
    size_t free_bytes = 0, total = 0;
    CK(cudaMemGetInfo(&free_bytes, &total));
    const WeightStream::Waited w = weight_stream.waited();
    printf("# wave %d has ended: %.3f s since the epoch; %.3f of %.3f GB of GPU memory are in use; the turns' stream stood still for weights %.3f s, in "
           "%zu of %zu asks for more than a millisecond\n",
           waves_ended, epoch(), (total - free_bytes) / 1e9, total / 1e9, w.seconds, w.long_ones, w.asks);
    // Time by stage (prover/prover/stage_clock.h); the numbers include the wait for weights.
    SERVING_OR(, if (!closed_front.say_closed("measure: pass %d stages%s; of these the stream stood still for weights %.3f\n", waves_ended, stage_clock.pass_has_ended().c_str(), w.seconds)) stop(End::WriteFailed));
    // Diagnostic (one-off, not upstream): how many blocking chain_fetch() round-trips this pass made.
    SERVING_OR(, printf("measure: chain_fetch calls %llu elements %llu\n", g_chain_fetch_calls, g_chain_fetch_elements));
  }
  // With -overlap, the_rider_folds() decodes weights into the slab between turns. Only the restore
  // before the next turn keeps them from its request, so it must reach as far. Neither number is
  // meant to depend on a request, and the reach never falls. No decoding precedes the first check.
  if (args.overlap && decoding_needs_at_most() > MemoryAsSetUp::slab_bytes(0)) {
    printf("internal error: with -overlap one unit's decoded weights take up to %zu bytes of the sub-proofs' slab, and a wipe "
           "before a turn zeroes only the first %zu\n",
           decoding_needs_at_most(), MemoryAsSetUp::slab_bytes(0));
    road_is_broken(chain, 4);
  }
  // A stub of a test hook: `refusal_test::between_bits()` returns 0 in every build from these
  // sources, so no bit is set here. Public sources for such hooks: see raise_what_a_test_asks().
  raise_flags(refusal_test::between_bits(false));
}
inline bool Prover::memory_grew() const {
  return fork_memory.taken_beyond_the_hold();
}

// Prints one summary line for the operator at the service's end.
inline void Prover::say_of_the_memory() const {
  printf("# memory as set-up left it: at the end: %d waves; a wipe before a turn zeroes %.3f GB, one at a wave's end %.3f "
         "GB; long stretches of constants that a run changed and that were laid again: %zu; GPU memory grew behind the "
         "first wave: %d\n",
         waves_ended, memory_as_set_up.bytes_zeroed(0) / 1e9, memory_as_set_up.bytes_zeroed(1) / 1e9,
         memory_as_set_up.tables_laid_again, (int)memory_grew());
}
