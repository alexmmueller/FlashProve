// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Members of Prover that set up and start the frame of a request's proof (blocks/iop/req.h), which
// commits to the arrays of the sub-proofs (`FK` layers, `AG` attention, `HH` output head). `link`:
// a sub-proof's segment of the proof; `moment`: commit point. Terms: prover/prover/chain_host.cuh.
#pragma once
#include <fstream>
#include <sstream>
#include "frame/check.h"
#include "frame/files.h"
#include "frame/heads.h"
#include "frame/hiding_conditions.h"
#include "iop/member.h"
#include "engine/card_plan.h"
#include "common/setting.h" // one-off (not upstream): g_parallel_streams
#include "prover/chain_host.cuh"

// Fills the plan of a segment's first commit point: the sub-proof's own matrices of packed cells,
// their GPU addresses (the same for all requests) and where its mask entries are. Matrices that no
// other sub-proof reads (shr == 0) come first, then the border matrices. A layer sub-proof has two:
// qa (queries, keys, values, attention's outputs), which the next segment reads, and the residual
// stream, read two segments on. ChainHost::settle() checks that the layout keeps each that long.
inline void Prover::chain_first_moment(int link, const std::string &name, u32 gid, HiddenTail &HT, const T *masks, size_t mask_entries) {
  ChainLink &L = chain.plan[link];
  L.name = name, L.gid = gid;
  L.masks = masks, L.mask_entries = mask_entries;
  if (L.moment.empty()) L.moment.resize(1);
  L.moment[0].clear();
  auto &at = chain.first_pointers[link];
  at.clear();
  for (int pass = 0; pass < 2; pass++)
    for (int mu = 0; mu < HT.NM; mu++) {
      if (HT.cls[mu] || (HT.shr[mu] != 0) != (pass == 1)) continue;
      ChainArray a;
      a.name = "M" + std::to_string(mu);
      a.R = (HT.M[mu].m + 2) / 3, a.K = HT.M[mu].k;
      // One symbol is three neighbouring signed byte cells of a column, in radix 2^ERAD; the last symbol of a column may be short.
      a.source.cells = blk::ArrayCells::SYMBOLS, a.source.symbol_bits = ERAD;
      a.source.row_stride = 3, a.source.column_stride = a.source.cells_in_a_column = HT.M[mu].m;
      a.source.pointer = (unsigned)at.size();
      L.moment[0].push_back(a);
      at.push_back(HT.M[mu].v);
    }
}

// blk::configure() takes a plain function for what this program adds to the public files (segments'
// oracle messages and identifiers, sub-proofs' rounds, the layout), so that only the files decide
// the result. Yet chain_shapes_of() also reads the sub-proofs' set-ups, through chain_shapes_from,
// which chain_make() sets for its configure() call, before any request. The shapes do not come
// from the files alone. Nothing in this program compares the result with the verifier's reading of
// the files; prover/tests/frame_as_the_verifier_reads.cu does, for a test.
static Prover *chain_shapes_from = nullptr;
static blk::Shapes chain_shapes_of(const std::vector<std::string> &files, const blk::Parameters &par) { return chain_shapes_from->chain_shapes(files, par); }
inline blk::Shapes Prover::chain_shapes(const std::vector<std::string> &files, const blk::Parameters &par) {
  blk::Shapes s;
  if (const char *fault = chain.links_from(files.at(1), par.answers)) return s.refused = "plan-layout: " + layout_beside(args.policy) + ": " + fault, s;
  chain.fixed.shape.link.back().plain = 1; // the next token is sent in its own round, before the head's first prover message
  s.sub.request = chain.fixed.shape;
  s.sub.open_rows.assign((size_t)chain.links(), 1); // a sub-proof ends in one linear claim (`row`)
  // One pending claim (`pending_rows`) on the weight table per layer sub-proof, one for the head.
  s.sub.pending_rows = (blk::u64)NGr + 1;
  // Shape of the committer's range proof (not made here), with the weight matrix W's row count.
  const char *units = nullptr;
  s.sub.ranges = chain_ranges(par.block_log, units); // the parameter file's number of column variables: `first_bound` (set in blocks/frame/wf_units.h) depends on it
  if (units) return s.refused = std::string("plan-words: W's layout: ") + units, s;
  if (s.sub.ranges.matrix_rows != closer->parameters().m0) return s.refused = "plan-words: W's layout has another number of rows than the keeper's unit has from the model's shape", s;
  // One request's approximate range checks (`tail` in identifiers), for the hiding conditions.
  auto add = [&s](const HiddenTail &tail) {
    for (const blk::RangeCheck &c : tail.range_checks()) s.range_checks.push_back(c);
  };
  for (int g = 0; g < NGr; g++) add(FK[g].h->HT), add(AG[g].o->HT);
  add(HH.HT);
  s.sub.plan = [this](blk::Plan &p, size_t link, unsigned moment) { chain_link_plan.at(link)(p, moment); };
  s.layout = chain.fixed.layout;
  for (const ChainLink &l : chain.plan) s.identifiers.push_back(l.gid);
  return s;
}

// Call after setup_opening() and all sub-proofs' set-ups. It completes `chain.plan`, makes `frame`
// and one ChainRun per -in-step request. Returns 0 or 2 (files rejected).
inline int Prover::chain_make() {
  // The owner's n-th border matrix: its segment, and its name at the owner's first commit point.
  auto border = [&](int owner_link, HiddenTail &HT, int n) {
    for (int mu = 0; mu < HT.NM; mu++)
      if (!HT.cls[mu] && HT.shr[mu] != 0 && n-- == 0) return std::make_pair(owner_link, "M" + std::to_string(mu));
    printf("internal error: the chain: a link has fewer border matrices than later links read from it\n");
    exit(4);
  };
  for (int g = 0; g < NGr; g++) {
    HFk &Q = *FK[g].h;
    chain_first_moment(chain.layer(g), "G" + std::to_string(g), (u32)(1 + g), Q.HT, MSG + NSM, Q.door.ns);
    if (g) chain.plan[chain.layer(g)].reads = {border(chain.layer(g - 1), FK[g - 1].h->HT, 1)}; // the residual stream
  }
  for (int g = 0; g < NGr; g++) {
    AForkHid &Hd = *AG[g].o;
    chain_first_moment(chain.attention(g), "A" + std::to_string(g), (u32)(2 + NGr + g), Hd.HT, Hd.slot_pads(), Hd.door.ns);
    chain.plan[chain.attention(g)].reads = {border(chain.layer(g), FK[g].h->HT, 0)}; // qa; the residual stream only passes by
  }
  chain_first_moment(chain.head(), "H", (u32)(1 + NGr), HH.HT, MSG + NSM, HH.door.ns);
  chain.plan[chain.head()].reads = {border(chain.layer(NGr - 1), FK[NGr - 1].h->HT, 1)};

  // Test builds only: prints the arrays that the sub-proofs declare, the input for a layout file.
  if (TESTENV("CHAIN_ARRAYS")) chain.print_arrays(stdout);
  // There is no default layout; prover/tools/pack_layout.py --arrays makes a file from those lines.
  if (!layout_file) {
    printf("policy: plan-layout: %s: there is no layout file\n", layout_beside(args.policy).c_str());
    return 2;
  }
  // configure() derives codes, numbers of positions, the last proximity test's stages and the round
  // plans of a proof's three hash chains. chain_link_plan adds the rounds after each commit point.
  chain_link_plan.assign((size_t)chain.links(), nullptr);
  for (int g = 0; g < NGr; g++) {
    chain_link_plan[(size_t)chain.layer(g)] = [this, g](blk::Plan &p, unsigned moment) { FK[g].h->plan_rounds(p, moment); };
    chain_link_plan[(size_t)chain.attention(g)] = [this, g](blk::Plan &p, unsigned moment) { AG[g].o->plan_rounds(p, moment); };
  }
  chain_link_plan[(size_t)chain.head()] = [this](blk::Plan &p, unsigned moment) { HH.plan_rounds(p, moment); };
  chain_shapes_from = this;
  frame = std::make_unique<blk::Configured>(blk::configure(configuration_files(), chain_shapes_of));
  chain_shapes_from = nullptr;
  {
    // configure() has tested the hiding conditions (blocks/frame/hiding_conditions.h) on the
    // shapes, plans and range checks from chain_shapes(). No verifier tests them in a proof.
    // read_inputs() tested the part that follows from the parameter file (`policy.txt`) alone. The
    // seed and the weights are already in memory: the set-ups come first in run(). configure() must
    // be called before (a) the first count of a batch's columns on a leaf table, (b) any request is
    // taken, (c) anything computed from the seed, the weights or private randomness (`coins` in
    // identifiers) is written to a file, a cache or a socket. (a), (b): batches begin in serve(),
    // after chain_make(). No build continues if a condition is missed, test seed or not.
    const bool hiding = frame->refused().rfind("hiding: ", 0) == 0;
    if (frame->refused().empty() || hiding)
      if (!(!hiding ? closed_front.say_closed("p258 the parameter file and the sub-proofs' shapes meet the conditions for hiding\n")
                    : closed_front.say_closed("p258 the parameter file and the sub-proofs' shapes miss a condition for hiding: %.300s\n",
                                              frame->refused().c_str() + 8))) // at most 300 bytes: the line fits say_closed()'s buffer
        stop(End::WriteFailed);
    if (hiding && !isk_test_seed) {
      isk_fault = "prover-only-hiding-conditions";
      name_the_end(isk_fault); // for the last line
      return 2;
    }
  }
  if (!frame->refused().empty()) {
    // chain_shapes() has already labelled its own rejections "plan-layout: " or "plan-words: ".
    const bool labelled = frame->refused().rfind("plan-layout: ", 0) == 0 || frame->refused().rfind("plan-words: ", 0) == 0;
    printf(labelled ? "policy: %s\n" : "policy: plan-words: %s\n", frame->refused().c_str());
    return 2;
  }
  if (const char *fault = chain.settle(frame->sized().proof_codes, frame->sized().request)) {
    printf("policy: plan-layout: %s: %s\n", layout_beside(args.policy).c_str(), fault);
    return 2;
  }
  size_t rows_F = 0, rows_E = 0, blocks = 0;
  for (const blk::ReqLink &l : chain.fixed.shape.link)
    for (const blk::ReqMessage &m : l.messages) (m.shape.main_deg == 1 ? rows_F : rows_E) += m.shape.main_rows;
  for (const std::vector<blk::ReqArray> &l : chain.fixed.layout.link)
    for (const blk::ReqArray &a : l) blocks += a.laid.blocks.size();
  printf("# the chain: %d links; %zu rows over F, %zu rows over E, %zu blocks\n", chain.links(), rows_F, rows_E, blocks);

  const int in_step = std::max(1, args.in_step);
  // For the operator: the GPU memory that the -in-step requests take through the engine, computed
  // from the shape. Tables count as CardTables::lay() stores them, a row in cells of 1, 2, 3 or 8
  // bytes; 8 bytes for every cell would give far more than one more request takes.
  blk::card_plan_print(stdout,
                       blk::card_plan(chain.fixed.codes, chain.fixed.shape, blk::CardKeep{false, false, false}, frame->sized().d.set.salt_bytes, frame->sized().d.link.t,
                                      [this](size_t link, size_t m, const blk::OracleShape &) {
                                        return blk::CardTables::table_bytes(chain.fixed.codes, chain.fixed.shape, chain.fixed.layout, chain.fixed.sources, link, m);
                                      }),
                       (unsigned)in_step);
  unsigned most_rounds = 0, most_draws = 0; // upper bounds between `lend` and `take_back`: the first chain's totals
  for (const blk::PlanRound &r : frame->proof().chains.at(0).plan->rounds_) most_rounds += (unsigned)r.count, most_draws += (unsigned)(r.count * r.draw_E);
  for (int j = 0; j < in_step; j++) chain.in_step.emplace_back(new ChainRun(chain.fixed, most_rounds, most_draws));
  // One-off (not upstream): see ChainHost::streams (chain_host.cuh). Created once, kept for the
  // program's life (cp_prove's requests of a batch pass are short-lived but the slots are not).
  if (g_parallel_streams) {
    chain.streams.resize(in_step);
    for (cudaStream_t &s : chain.streams) CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  }
  chain.begun_under.assign(in_step, ChainHost::NOT_BEGUN);
  chain.enter(0);
  // Index-mask message (masks for claims on the weight table): mask rows only, uniform and fresh.
  chain.at_start = [this] {
    std::vector<blk::OracleDraft> drafts;
    for (const blk::OracleNotice &n : chain.fixed.shape.at_start) {
      blk::OracleDraft d;
      d.shape = n.shape, d.leaf_class = n.leaf_class;
      blk::Coins &coins = chain_coins();
      d.mask = blk::Send::fresh_mask(chain.fixed.codes, n.shape, coins);
      if (drafts.empty()) {  // the index-mask message is the first (ChainRun::index_mask_entry(): at_start().at(0))
        // [own-test-index-mask] Notes this draw's stream while it is the generator's last. ChainRun
        // compares the sent table and entries taken later with EngineCoins' second implementation.
        auto *own = dynamic_cast<hz::lng::EngineCoins *>(&coins);
        if (!own) {
          printf("internal error: a request's coins are not this program's own generator: its index-mask entries cannot be held against the second writing\n");
          name_the_end("own-test-foreign-coins");
          exit(4);
        }
        const u32 stream = own->last_stream();
        chain.request->index_masks_were_drawn([own, stream](u64 i, blk::F x) { return own->is_the_coin(stream, i, x); },
                                              [own, stream](const blk::F *values, u64 n, const std::vector<u64> &left_out) { return own->not_the_coins(stream, values, n, left_out); });
        // Fault-injection hook, empty in a production build (row E29.1): can zero drawn values before sending.
        lost_copies_door::strike("index_masks_drawn", d.mask.a.data(), d.mask.a.size(), false);
      }
      drafts.push_back(std::move(d));
    }
    return drafts;
  };
  hd::link_frame(chain, made_up_under_a_tests_seed());
  return 0;
}

// Starts a request's first hash chain. bind_request_parameters() and bind_request_statement()
// (blocks/frame/check.h), which the verifier calls too, put the public inputs into its first
// query. Its item "request" is the word list `hd` (prover/prover/header.cuh), little-endian, with 0
// for the next token, which the head computes later and ChainHost::first() sends. The item after
// it is the request file's own words (the tokens, r and e), little-endian. The chain's salt, 32
// fresh random bytes, is neither parameter nor statement; the chain's bytes begin with it.
inline void Prover::chain_begin_request() {
  if (!g_slot_entered || !opening_of_the_batch || !opening_of_the_batch->order) {
    printf("internal error: the chain: a request starts while no slot is entered, or before its batch's columns are counted\n");
    exit(4);
  }
  CK(cudaMemcpy(hd.data(), hdr, NH * 4, cudaMemcpyDeviceToHost));
  std::unique_ptr<blk::Binder> binder(new blk::Binder(blk::LABEL_FIRST_CHAIN, frame->proof().chains.at(0).plan));
  blk::RequestPublic x = blk::bind_request_parameters(*binder, *frame);
  std::vector<blk::u8> words, file_words;
  file_words.reserve(4 * g_slot_entered->tokens.size());
  for (int i = 0; i < NH; i++)
    for (int b = 0; b < 4; b++) words.push_back((blk::u8)(hd[(size_t)i] >> (8 * b)));
  for (uint32_t word : g_slot_entered->tokens)
    for (int b = 0; b < 4; b++) file_words.push_back((blk::u8)(word >> (8 * b)));
  // The statement: the batch's leaf table (`IndexName`), that table's number, the request, its words.
  blk::bind_request_statement(*binder, x, opening_of_the_batch->table, opening_of_the_batch->entry, words, file_words);
  explicit_bzero(file_words.data(), file_words.size()); // this copy alone: the binder's list and x.file_words hold the words until they are freed
  g_slot_entered->part.lin = x.lin;
  const ChainPublic bound{x.lin.codes, x.shape, x.layout};
  chain.begun_under[chain.member] = (u64)rep;
  chain.hold_the_attempt("begin()");
  blk::u8 bytes[blk::DIGEST_BYTES];
  chain_coins().bytes(bytes, sizeof bytes);
  blk::Digest random_bytes;
  blk::digest_from_bytes(bytes, random_bytes);
  chain.request->begin(std::move(binder), bound, random_bytes, chain_coins(), chain_engine(), chain.stream_of(chain.member));
  tr = chain.request->tr();
}
inline blk::EngineGpu &Prover::chain_engine() { return *opening_of_the_batch->engine; }

// Call it before a sub-proof. It binds the callbacks of ch::host (prover/hiding/moments.cuh) to the
// sub-proof's segment, except `blob`, which the caller sets.
inline void Prover::chain_bind(int link) {
  ch::host.masking_pads_hold = [this, link](const void *pads, size_t slots) {
    if (!g_slot_entered) {
      printf("internal error (own-test-slot-pads): a sub-proof closes while no slot is entered\n");
      name_the_end("own-test-slot-pads");
      exit(4);
    }
    static_assert(sizeof(T) == 8 * EDEG, "hold() reads an element as EDEG cells of 8 bytes, one behind the other");
    Slot &s = *g_slot_entered;
    bool equal = false;
    s.pad_cells_held += slot_pads_second.hold(pads, slots, chain.plan[link].gid, s8::ver(), (u32)s.j, equal);
    if (!equal || (size_t)link >= s.pad_links_held.size() || s.pad_links_held[(size_t)link]++) s.pads_differ = true;
  };
  // The frame keeps no copy of the mask entries on the GPU: it fetches the sub-proof's own array in first() and commits to what it fetched.
  ch::host.kept_pads = [this, link](size_t) -> const void * { return chain.plan[link].masks; };
  // No root is kept in GPU memory: code that reads a commit point's ROOTW words gets zeros.
  ch::host.words = [this](int) { return (const u32 *)chain.no_words; };
  ch::host.take_back = [this]() -> blk::ProverRounds & {
    chain.hold_the_attempt("take_back()");
    chain.request->take_back();
    return chain.request->channel();
  };
  ch::host.lend = [this] { chain.request->lend(); };
  ch::host.index_mask_entry = [this](T *where) {
    chain.hold_the_attempt("index_mask_entry()");
    chain.request->index_mask_entry(where);
  };
  ch::host.index_mask_entry_holds = [this](const T *cell) {
    if (!g_slot_entered) {
      printf("internal error (own-test-slot-pads): a sub-proof closes while no slot is entered\n");
      name_the_end("own-test-slot-pads");
      exit(4);
    }
    if (!chain.request->index_mask_entry_holds(cell)) g_slot_entered->pads_differ = true;
  };
  ch::host.later = [this, link](int m, const std::vector<const void *> &at, Tr *t) {
    StageClock::Within frame(stage_clock, StageClock::FrameLater);
    chain.later(link, m, at, t);
  };
  ch::host.first = [this, link](Tr *t, const u32 *y) {
    StageClock::Within frame(stage_clock, StageClock::FrameFirst);
    chain.first(link, t, y);
  };
}
