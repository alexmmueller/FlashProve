// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Proving one request (`member` in identifiers): begin_request(), prove_layer_group() for each
// layer group, prove_head_fork(), finish_request(); before them, functions that restore GPU memory.
// In a batch pass (`wave`: one pass over the weights) requests take turns, each on a layer group or
// the output head; a state buffer (`seat_`) keeps a request's state in between. `fork`: sub-proof.
// Every request is meant to take the same path: no stage ends early or is shortened because of a
// request's values, challenges or private random values (`coins`). A failed check sets a bit of the
// request's failure word (`flags`; prover/common/slot_flags.h), and the work goes on to the end, on
// a substitute value where the check has one; a request with a bit set gets no proof. Only a fault
// of the machine or the program ends the process: a failed CUDA call or internal check, no memory.
// Known exception: the routed experts' products (prover/weights_blocks/experts.cuh) follow the
// routing. A tensor of type Q4_K or Q5_K that pk_fits() accepts takes one launch with a public grid:
// only the kernel's time depends on the routing. Otherwise (so for every tensor of type Q8_0, F16 or
// BF16), or with EXPERTS_IN_ONE_LAUNCH at 0, so do the launches' number and sizes. The host always
// copies the routing tables and zeroes its copy. The routing is a function of the prompt and the
// weights: no challenge or private randomness reaches a router. Only the failure bits UNFIT and INDEX
// put substitute values into the layers' forward pass and so can change the routing. Under other bits
// the launches are as without failures, except after a failed self-check of an opened set of columns;
// no test in this tree checks this.
#pragma once
#include "prover/memory_as_set_up.cuh"
#include "prover/prover.h"

// Test switch fork_memory.test.stale (cp_prove never sets it): fills with 0xEE what is zeroed next.
inline void Prover::lay_stale_bytes(const ForkMemory::Buffers &list) {
  for (auto &b : list) CK(cudaMemsetAsync(b.first, 0xEE, b.second, cudaStreamPerThread));
}
// Same switch, after the zeroing: counts the listed buffers' bytes that are not zero and prints the
// count in a line with `when`; ends the process if it is not 0. kept: the zeroing was left out.
inline void Prover::count_stale_bytes(const ForkMemory::Buffers &list, const char *when, bool kept) {
  static unsigned long long *left = nullptr;
  if (!left) CK(cudaMallocManaged(&left, 8));
  CK(cudaStreamSynchronize(cudaStreamPerThread));
  *left = 0;
  size_t bytes = 0;
  for (auto &b : list) {
    k_fork_memory_count_not<<<1024, 256, 0, cudaStreamPerThread>>>((const unsigned char *)b.first, b.second, 0, left);
    bytes += b.second;
  }
  CK(cudaStreamSynchronize(cudaStreamPerThread));
  printf("test mode: seat %d%s: %zu buffers (%zu bytes in all) were overwritten with 0xEE%s; bytes that are not "
         "zero: %llu%s\n",
         g_slot_entered ? g_slot_entered->j : -1, when, list.size(), bytes, kept ? " and, as this mode asks, NOT zeroed" : ", then zeroed", *left,
         *left ? "; the program ends, because memory that must be zero is not" : "");
  fflush(stdout);
  if (*left) exit(3);
}

// Returns the state buffers, which the restore before a turn must leave alone: every request's bank
// (its state as MemberState saves it), the saved failure words, the regions in chain_seat_regions.
// Public source: zeroing only selected GPU memory: see prover/prover/memory_as_set_up.cuh, wipe().
inline ForkMemory::Buffers Prover::kept_between_turns() const {
  ForkMemory::Buffers kept = member_state.all_banks();
  for (auto &regions : chain_seat_regions) kept.insert(kept.end(), regions.begin(), regions.end());
  return kept;
}

// Restores the working memory before a turn, in stream order, so that the turn finds there nothing
// of other requests: zeroes the sub-proofs' scratch memory up to slab_bytes(0) and gives the rest
// of ForkMemory's record its content at the end of set-up. Not restored: long tables of constants
// (see MemoryAsSetUp::SHORT_MOST), what of_the_record(0) leaves out, memory outside the record, the
// state buffers. Before::AnAttempt (a request starts) also zeroes its bank and chain_own_regions.
inline void Prover::wipe_the_working_memory(Before before) {
  const bool turn = before != Before::AnAttempt;
  const char *const when = before == Before::AnAttempt ? ""
                           : before == Before::ATurn   ? ", before a turn in step"
                                                       : ", before the head's turn in step";
  ForkMemory::Buffers own; // the parts of the state buffer that are zeroed when a request starts
  if (!turn) {
    own = chain_own_regions[seat_at_hand];
    own.push_back(member_state.bank(seat_at_hand));
  }
  const ForkMemory::Test &test = fork_memory.test;
  const bool skipped = test.skip_zero_shipped || (turn && test.skip_zero_turns);
  ForkMemory::Buffers looked_at;
  if (test.stale) {
    // Prints how far the zeroing reaches into the sub-proofs' scratch memory, when that changes.
    static size_t said = ~(size_t)0;
    if (said != MemoryAsSetUp::slab_bytes(0))
      printf("test mode: the slab is to be zeroed as far as %zu bytes, of %zu that are mapped\n",
             said = MemoryAsSetUp::slab_bytes(0), fork_memory.mapped);
    looked_at = memory_as_set_up.what_is_zeroed(0, seats_state);
    looked_at.insert(looked_at.end(), own.begin(), own.end());
    lay_stale_bytes(looked_at);
  }
  if (!skipped) {
    memory_as_set_up.wipe(0, seats_state, cudaStreamPerThread);
    for (auto &b : own) CK(cudaMemsetAsync(b.first, 0, b.second, cudaStreamPerThread));
  }
  if (test.stale) count_stale_bytes(looked_at, when, skipped);
}

// Call it when a batch pass's last request has ended. Like wipe_the_working_memory(), but it also
// restores the state buffers and the requests' own memory and zeroes all mapped scratch memory. Not
// restored: what of_the_record(1) leaves out, memory outside the record. It checks the long tables
// of constants, zeroes `hd` (the host's copy of a request's statement) and waits for the GPU.
inline void Prover::wipe_behind_the_wave() {
  ForkMemory::Buffers looked_at;
  if (fork_memory.test.stale) {
    looked_at = memory_as_set_up.what_is_zeroed(1, {});
    lay_stale_bytes(looked_at);
  }
  memory_as_set_up.wipe(1, {}, cudaStreamPerThread);
  if (fork_memory.test.stale) count_stale_bytes(looked_at, ", behind a wave", false);
  // Non-zero: a long table of constants changed since the last check. Nothing restores it between
  // turns, so a request may have read what another wrote: no request of the batch gets its files.
  if (memory_as_set_up.hold_the_tables()) fail_the_batch(End::Stopped);
  explicit_bzero(hd.data(), hd.size() * sizeof hd[0]);
  if (fork_memory.test.left_behind) fork_memory.look_at_what_stays();
}

// Runs the exact forward pass of layer group g, which also writes the cells (committed small
// integers) and lookup keys, then computes the multiplicities of the keys of every lookup call.
inline void Prover::forward_and_counts(int g) {
  Fork &K = FK[g];
  if (g > 0) {
    CK(cudaMemcpyAsync(K.g.xs(0), xin[g], (size_t)d * NT * sizeof(ResidualInt), cudaMemcpyDeviceToDevice));
    // A test hook, empty in this repository (prover/common/lie.cuh). In a test build outside this
    // repository it makes the group's input differ from what the group before committed to.
    lie::in_memory<ResidualInt>(g * Lam, lie::BORDER, lie::STREAM, cudaStreamPerThread, [&](long long row, long long token, long long) {
      return row >= 0 && row < d && token >= 0 && token < NT ? K.g.xs(0) + row * NT + token : nullptr;
    });
  }
  K.g.forward(1); // its range flags go to errf (begin_request())
  fork_memory.arena_idle();
  for (int c = 0; c < NC; c++) K.F->counts(c, K.g.b.key[c], K.pub + layer_forest.first[c], K.padded[c].data());
  K.F->planes();
  // The multiplicities' bits, ordered by (call, table row, bit position), are the matrix b.
  CK(cudaMemcpyAsync(K.M[K.b.first].v, K.F->bits(0), nb, cudaMemcpyDeviceToDevice));
  fork_memory.arena_idle();
}

// Starts the request of slot s (its place in the batch) in the current state buffer. s.tokens: NT
// tokens, each below the vocabulary size, then r < e <= NT (the proof is about positions r to
// e - 1). The caller must have validated them: here only the length, r and e are checked again.
inline void Prover::begin_request(Slot &s) {
  if ((int)s.tokens.size() != NT + 2 || s.tokens[NT] >= s.tokens[NT + 1] || s.tokens[NT + 1] > (u32)NT) {
    printf("internal error: a slot holds %zu words, but a request under this parameter file has %d tokens and two "
           "positions r < e <= %d\n",
           s.tokens.size(), NT, NT);
    exit(4);
  }
  rep = s.attempt;
  // The forward passes set their flags directly in errf (zeroed below, read by finish_request()): a
  // kernel that copied each pass's flags there could fail to launch unnoticed and lose them.
  for (auto &K : FK) K.g.raise_flags_in(errf);
  HH.h.raise_flags_in(errf);
  wipe_the_working_memory(Before::AnAttempt);
  member_state.begin(seat_at_hand); // from here on the GPU's live words are this request's
  CK(cudaMemcpy(tokd, s.tokens.data(), ((size_t)NT + 2) * 4, cudaMemcpyHostToDevice));
  // The tokens are hashed in place and not with hash_words(), which allocates
  // and frees a buffer: freeing GPU memory waits for the whole device.
  CK(cudaMemcpyAsync(hdr + atTokens, treehash((const u32 *)tokd, (size_t)NT + 2), 4 * DW, cudaMemcpyDeviceToDevice,
                     cudaStreamPerThread));
  // Begins the request's blinding context (`attempt`): counter, kind and slot number enter every
  // private random value that hz derives under it. The counter must exceed all used before under
  // the batch's key, or the call exits: the nonce is 0, so a repeated counter could repeat values.
  hz::set_attempt(rep, hz::lng::REQUEST, (u32)s.j);
  s.pad_cells_held = 0, s.pad_links_held.assign(chain.plan.size(), 0), s.pads_differ = false; // [own-test-slot-pads]
  // A [tag] marks a part of the self-check of that name. The error words accumulate over a request.
  for (u32 *word : {err, errc, errf}) CK(cudaMemsetAsync(word, 0, 4, cudaStreamPerThread));
  // Two counts accumulate too: of slot pads (mask entries: each masks one scalar that a sub-proof
  // sends) and of filler cells (padding). Each starts not at 0 but at eight equal bytes, 1 +
  // requests_begun mod 255; finish_request() subtracts that word. So a word that was not reset, and
  // still has an earlier request's sum, does not pass for this request's count: requests of one
  // shape have equal sums, but two different start words are over 2^56 apart, far above any count.
  // A start recurs after 255 requests: never within a batch pass (SLOTS_IN_STEP_MOST at most).
  s.counts_start = (unsigned char)(1 + requests_begun++ % 255);
  CK(cudaMemsetAsync(slot_pads_agree, s.counts_start, 8, cudaStreamPerThread));
  CK(cudaMemsetAsync(filler_is_zero, s.counts_start, 8, cudaStreamPerThread));
  for (auto *Hd : AO) CK(cudaMemsetAsync(Hd->errK, 0, 16, cudaStreamPerThread));
  CK(cudaMemsetAsync(attF.error_words(), 0, 16, cudaStreamPerThread));
  CK(cudaMemsetAsync(hdr + atY, 0, 4, cudaStreamPerThread));
  // Starts the request's first chain: its hash chain up to its seal (the digest that ends it). What
  // ForkMemory allocates in the call is labelled the request's own: not restored before a turn.
  fork_memory.purpose = MemoryAsSetUp::A_REQUESTS_OWN;
  chain_begin_request();
  fork_memory.purpose = "other buffers of the prover";
}

// Proves the request's layer sub-proof G_g and attention sub-proof A_g. Non-zero: internal error.
inline int Prover::prove_layer_group(int g) {
  if (int rc = g_commit(g)) return rc; // -measure's stages forward and commit (stage_clock.h)
  stage_clock.at(StageClock::Tables);
  if (int rc = g_product_tests(g)) return rc;
  if (int rc = g_weight_tables(g)) return rc;
  stage_clock.at(StageClock::Sums);
  if (int rc = g_norm_sums(g)) return rc;
  if (int rc = g_token_sumchecks(g)) return rc;
  stage_clock.at(StageClock::Rows);
  if (int rc = g_stack(g)) return rc;
  if (int rc = g_gated_products(g)) return rc;
  if (int rc = g_row_passes(g)) return rc;
  stage_clock.at(StageClock::Fold);
  if (int rc = g_fold(g)) return rc;
  stage_clock.at(StageClock::Walk);
  if (int rc = g_walk(g)) return rc;
  stage_clock.at(StageClock::EqColBuild);
  if (int rc = g_equations_and_columns(g)) return rc;
  fork_memory.run_end();
  return prove_attention_fork(g); // -measure's stages forward and attention
}

// Proves the head's sub-proof and sets the statement's next token (`y`). Non-zero: internal error.
inline int Prover::prove_head_fork() {
  stage_clock.at(StageClock::Forward);
  fork_memory.run_begin(NGr);
  if (weight_stream.ends_by_turns) weight_stream.experts_ready(weight_stream.end_unit[1]);
  // With -memory rest-by-turns, the head's weights were not on the GPU when set-up wrote the cells.
  if (!fork_memory.test.own_memory || weight_stream.ends_by_turns) HH.weights();
  HH.phase1(xin[NGr], tokd, (const u32 *)tokd + NT); // its flags go to errf (begin_request())
  // Like g_commit(), counts the head's zero filler cells before HH.prove() commits to its matrices.
  for (auto &range : HH.filler)
    if (range.second) k_count_zero_bytes<<<32, 256>>>(range.first, range.second, filler_is_zero);
  CK(cudaGetLastError());
  // Nothing after the sub-proof's forward pass reads the head's weights, because the proof
  // works on the cells that weights() wrote. So the weights can be reported as used here.
  if (weight_stream.ends_by_turns) weight_stream.experts_used(weight_stream.end_unit[1]);
  k_sety<<<1, 1>>>(hdr, atY, HH.h.y);
  chain_bind(chain.head());
  HH.door.link = chain.head();
  ch::host.blob = [this] {
    size_t o = 0;
    for (size_t q = iSecH; q < iSecHe; q++) {
      if (SC[q].words) CK(cudaMemcpyAsync(out + o, SC[q].p, SC[q].words * 4, cudaMemcpyDeviceToDevice));
      o += SC[q].words;
    }
    return (const u32 *)out;
  };
  stage_clock.at(StageClock::Head);
  if (int rc = HH.prove(tr, sd, rsc, err, errc)) return rc;
  fork_memory.run_end();
  return 0;
}

// Ends the request: finishes its first chain, collects the failure word and fills slot s. No step
// branches on that word, except after a failed self-check: a printed line, a_self_check_failed(),
// withhold(). ChainRun records no failure: it reaches the seal or a failed check ends the process.
inline void Prover::finish_request(Slot &s) {
  lie::report(); // a test hook: empty in this repository
  // The last segment's (`link`) code switch (blocks/iop/cs.h: re-encodes rows under new codes) and
  // the last proximity test; then the first part of the weight table's test (`lin`), and the seal.
  chain.hold_the_attempt("finish()");
  fork_memory.check_what_is_lent(false);
  chain.request->take_back();
  chain.request->finish();
  stage_clock.at(StageClock::EndSeal);
  chain_member_and_seal(s); // fills s.part.lin_first and s.part.seal
  stage_clock.at(StageClock::EndAfter);
  chain.request->channel().finish_inside_a_file(); // the file goes on with other chains' bytes
  // [own-test-engine-coins] Checks a sample of the private random values before bytes go to disk.
  coins_hold(s.coins.get(), "a request's");
  unsigned bits = 0;

  bool slot_pads_held; // [own-test-slot-pads]
  {
    // [own-test-slot-pads] Each sub-proof's slot pads must have been compared exactly once with a
    // second implementation of their derivation (slot_pads_second.cuh), every 64-bit cell equal.
    unsigned long long pad_cells = HH.door.ns;
    bool held = !s.pads_differ && s.pad_links_held.size() == chain.plan.size() && s.pad_links_held[(size_t)chain.head()] == 1;
    for (int g = 0; g < NGr; g++) {
      pad_cells += FK[g].h->door.ns + AG[g].o->door.ns;
      held = held && s.pad_links_held[(size_t)chain.layer(g)] == 1 && s.pad_links_held[(size_t)chain.attention(g)] == 1;
    }
    held = held && pad_cells && s.pad_cells_held == EDEG * pad_cells;
    slot_pads_held = held;
    bits |= held ? 0 : OWN_TEST;
    if (!held) a_self_check_failed("own-test-slot-pads");
    if (!held) // for the operator; nothing depends on this line
      printf("a self-check failed (own-test-slot-pads): secret random values that hide a proof's contents are not what the batch's key "
             "gives, or were not all checked (a fault of the GPU or the program); the request gets no proof, and the service ends at the next end of a batch\n");
  }

  // The failure word. end() saves the GPU's live word, in which kernels set the current request's
  // bits, then waits for the whole GPU. The saved word is sure to have all of the request's bits
  // only if every stream that worked for the request has joined this thread's stream
  // (docs/limits-all-rows.tsv, row T11.1.1); end() itself joins those registered in
  // prover/common/side_streams.h. A bit that a kernel on another stream sets after the save belongs
  // to no request and fails the whole batch (end_the_batch()).
  bits |= member_state.end(seat_at_hand);
  CK(cudaGetLastError());
  {
    // An attention sub-proof's errors() (prover/attn/fork.cuh): bit 0, a forward-pass value outside
    // its range; bit 1, the approximate range check's flag (an entry outside its window); bit 2, a
    // lookup forest's error word, always 0; bit 3, its cells of q, k, v and o differ from the layer
    // sub-proof's, as copy_qkv_for_attention() kept them. In errf only the head sets E_BOUNDS
    // (prover/head/forward.cuh): the checked tokens exceed the parameter file's bound theta or cap.
    u32 attention = 0, forward = 0, counts = 0, folds = 0;
    for (auto *Hd : AO) attention |= Hd->errors();
    const u32 output = attF.errors(); // read outside the || below, so that its GPU copy always runs
    CK(cudaMemcpy(&forward, errf, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&counts, errc, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&folds, err, 4, cudaMemcpyDeviceToHost));
    bits |= (attention & 5u || output || forward & ~(u32)E_BOUNDS || counts ? UNFIT : 0) |
            (forward & (u32)E_BOUNDS ? ABOVE : 0) | (folds || attention & 2u ? WINDOW : 0) |
            (attention & 8u ? OWN_TEST : 0);
    // hz::Tree::slot_pads_agree() must also find every cell of the slot pads to be what the batch's
    // key gives, twice: after the host fetches a copy to commit to, and after their last use. Like
    // the comparison above, it reads the sub-proof's GPU array, not that copy. Any other count: the
    // GPU skipped a draw, a check, a copy of the pads or of this count, or a cell was overwritten.
    const unsigned long long counts_began_at = s.counts_start * 0x0101010101010101ull; // not 0: see begin_request()
    unsigned long long agree = 0, cells = HH.door.ns;
    for (int g = 0; g < NGr; g++) cells += FK[g].h->door.ns + AG[g].o->door.ns;
    cells *= 2 * EDEG;
    CK(cudaMemcpy(&agree, slot_pads_agree, 8, cudaMemcpyDeviceToHost));
    agree -= counts_began_at;
    bits |= agree != cells ? OWN_TEST : 0;
    if (agree != cells) a_self_check_failed("own-test-slot-pads-recount"); // after the comparison by value: a lost pad fails both, and the first name stays
    if (agree != cells) // for the operator; nothing depends on this line
      printf("slot %d: a self-check failed (own-test-slot-pads-recount): %llu of the %llu cells of the sub-proofs' slot pads in their two copies were found "
             "to be the coins that the key gives (a fault of the GPU or the program); the request gets no proof, and the service ends at the next end of a batch\n",
             s.j, agree, cells);
    // Every filler cell must have been counted as zero: g_commit() counts in the matrices af, ad,
    // ar, qa, prove_attention_fork() in as, ah, ao, bA, qa, prove_head_fork() in HH.h's hf, hd, hr.
    // Otherwise the GPU skipped a zeroing, a count or a copy of the sum, or a cell was overwritten.
    unsigned long long found_zero = 0, filler_cells = 0;
    for (auto &range : HH.filler) filler_cells += range.second;
    for (int g = 0; g < NGr; g++) {
      for (auto &range : FK[g].filler) filler_cells += range.second;
      for (size_t n : AG[g].o->A.zero_fill) filler_cells += n;
    }
    CK(cudaMemcpy(&found_zero, filler_is_zero, 8, cudaMemcpyDeviceToHost));
    found_zero -= counts_began_at;
    bits |= found_zero != filler_cells ? OWN_TEST : 0;
    // For the operator; nothing depends on this line. It prints the number of filler cells, which
    // follows from the configuration files, and not the count, which depends on hidden memory.
    if (found_zero != filler_cells)
      printf("slot %d: a self-check failed: the sub-proofs' matrices have %llu filler cells (padding), and another "
             "number of them was found to be zero before the first commitments (a fault of the GPU or the program); "
             "the request gets no proof\n",
             s.j, filler_cells);
  }
  raise_flags(bits); // the host's live word is this slot's until leave_slot()

  // Moves the first chain's bytes into the slot, where a writer thread reads them
  // (stage_the_proofs()): only after this thread has read its CUDA status, above
  // (prover/docs/cuda-status.md, "The rule"). The request's context moves too, for the second
  // chain, which also needs the batch's engine and s.coins.
  chain.request->detach(s.part);
  chain.begun_under[chain.member] = ChainHost::NOT_BEGUN;

  // The next token, read from the statement.
  CK(cudaMemcpy(&s.y, hdr + atY, 4, cudaMemcpyDeviceToHost));

  // The proof file: header, first chain (s.part.first; the request's own), its place, shared chain
  // (one for the whole batch), second chain (the request's own again). The place is not the slot's
  // index: it is the request's leaf number in the batch's binding tree (a Merkle tree over the
  // requests' seals), which BatchOpening::put() takes from a random arrangement of the leaves.
  // [own-test-slot-pads] The chain cannot detect wrong slot pads, so it has produced bytes,
  // which such pads do not hide. The failure bit withholds them; they are also zeroed here.
  if (!slot_pads_held) s.part.withhold();

  hz::end_attempt(); // ends the blinding context, after waiting for the whole GPU
}
