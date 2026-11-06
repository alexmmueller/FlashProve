// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// With -measure, the prover program cp_prove prints one line per batch pass that splits its time
// among stages of the prover's code: "measure: pass <n> stages ..". A batch pass (`pass` in
// identifiers and output) is the work of the requests that advance together through all layer
// groups (runs of consecutive layers) and the output head (prover/prover/wave_order.cuh). Only a
// test build measures: a production build (SERVING_BUILD) rejects -measure.
//
// The prover calls at(s) where stage s begins. A call reads the host's clock and records a CUDA
// event on the calling thread's stream, which only queues a marker: no call waits for the GPU or
// touches proof data. A stage gets two sums over its intervals, each from at(s) to the next call:
//   on the stream: the time between the two calls' events, that is, how long the GPU's queue took,
//                  including the time that it sat idle waiting for the host;
//   on the host:   the time between the two calls by the host's clock, that is, what the calling
//                  thread spent queueing work, computing or waiting.
// The two differ where the host runs ahead of the GPU. Over all stages, each adds up to the time
// from the batch pass's first call to its last; the wipe of GPU memory after it is not counted.
#pragma once
#include "common/common.h"
#include <cuda.h>
#include <ctime>
#include <string>
#include <vector>

// Nothing here is locked: all calls must come from one thread. In a production build at() and
// Within do nothing, and `on` and pass_has_ended() do not exist: wrap their uses in SERVING_OR()
// (prover/common/setting.h).
//
// The stages, by their printed names. Functions without a class are Prover's. The attention
// sub-proof and the sub-proof of the output head (the arguments for these parts of the forward
// pass) are not split further.
//   start      begin_request()
//   forward    the forward passes of a layer group, of its attention and of the output head
//   commit     g_commit() after its forward pass
//   tables     g_product_tests(), g_weight_tables()
//   sums       g_norm_sums(), g_token_sumchecks()
//   rows       g_stack(), g_gated_products(), g_row_passes()
//   closing    g_fold(), g_walk(), g_equations_and_columns()
//   attention  the attention sub-proof after its forward pass
//   head       the sub-proof of the output head after its forward pass
//   frame      ChainHost::first() and ChainHost::later() (calls into the request's first hash
//              chain, a Fiat-Shamir transcript), wherever they occur; their time is taken out of
//              the surrounding stage
//   end        finish_request()
//   turns      everything else: wipes, saving and loading a request's state and, with -overlap,
//              work on an earlier batch's opening (its test on the weight table) between two turns
//              (a turn is one request's work on one layer group or on the output head)
struct StageClock {
  // Split for a one-off diagnostic (not upstream): Closing -> Fold/Walk/EqCol, Frame ->
  // FrameFirst/FrameLater, End -> End/EndSeal/EndAfter, to localize the cost within each bucket.
  // EqCol itself split further into EqColBuild (close_by_frame's term loop + batch patches),
  // EqColRow and EqColClose (ChainRun::close()'s own "row" and "close" sub-steps).
  enum Stage { Start, Forward, Commit, Tables, Sums, Rows, Fold, Walk, EqColBuild, EqColRow, EqColClose, Attention, Head, FrameFirst, FrameLater, End, EndSeal, EndAfter, Turns, STAGES };
#ifdef SERVING_BUILD
  void at(Stage) {}
  struct Within {
    Within(StageClock &, Stage) {}
  };
#else
  bool on = false; // -measure
  void at(Stage s) {
    if (!on) return;
    if (marks.size() == events.size()) {
      events.emplace_back();
      CK(cudaEventCreate(&events.back()));
    }
    CK(cudaEventRecord(events[marks.size()], cudaStreamPerThread));
    marks.push_back({s, now()});
    stage = s;
  }
  // One-off (not upstream): lets a caller elsewhere in the file (ChainRun::close(), shared by
  // every sub-proof's closing, not just the layer sub-proof's) check the current stage before
  // switching, so a mark meant only for one caller's window doesn't mislabel another's (e.g. the
  // attention or head sub-proof's own closing, which reuses the same close() function).
  Stage current() const { return stage; }
  // Switches to stage s until the end of the enclosing block, then back to the stage before.
  struct Within {
    StageClock &clock;
    const Stage back;
    Within(StageClock &c, Stage s) : clock(c), back(c.stage) { clock.at(s); }
    ~Within() { clock.at(back); }
  };
  // Call it at the end of a batch pass, when the calling thread has waited for the GPU since the
  // last at(), so that every event has completed; in cp_prove, MemoryAsSetUp::hold_the_tables() is
  // that wait. Returns the line's text after "stages" (its last item, "marks", is the number of
  // at() calls) and clears the marks.
  // It reads the events with the driver's cuEventElapsedTime(), which returns its status to the
  // caller: the runtime's cudaEventElapsedTime() might store cudaErrorNotReady in the thread's
  // last-error variable, which the program reads as the status of a kernel launch.
  // A test that is not part of this repository checks that a successful driver call leaves that
  // variable unchanged; no test covers a failing one. If a call fails, the line reports it and
  // gives only the host's numbers.
  std::string pass_has_ended() {
    static const char *const NAMES[STAGES] = {"start", "forward", "commit", "tables", "sums", "rows", "fold", "walk", "eqcol_build", "eqcol_row", "eqcol_close", "attention", "head", "frame_first", "frame_later", "end", "end_seal", "end_after", "turns"};
    double stream[STAGES] = {0}, host[STAGES] = {0};
    bool all_came = true;
    for (size_t i = 0; i + 1 < marks.size(); i++) {
      float ms = 0;
      all_came = all_came && cuEventElapsedTime(&ms, (CUevent)events[i], (CUevent)events[i + 1]) == CUDA_SUCCESS;
      stream[marks[i].stage] += ms / 1e3, host[marks[i].stage] += marks[i + 1].host - marks[i].host;
    }
    std::string line = all_came ? " (seconds on the stream/on the host)" : " (seconds on the host; an event had not come at the pass's end, so the stream's are left out)";
    char one[64];
    for (int s = 0; s < STAGES; s++) {
      if (all_came)
        snprintf(one, sizeof one, " %s %.3f/%.3f", NAMES[s], stream[s], host[s]);
      else
        snprintf(one, sizeof one, " %s %.3f", NAMES[s], host[s]);
      line += one;
    }
    snprintf(one, sizeof one, " marks %zu", marks.size());
    marks.clear();
    return line + one;
  }

private:
  struct Mark {
    Stage stage;
    double host;
  };
  static double now() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
  }
  Stage stage = Turns;
  std::vector<Mark> marks;
  std::vector<cudaEvent_t> events; // kept from pass to pass
#endif
};
// `static`: one clock per translation unit.
static StageClock stage_clock;
