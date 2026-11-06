# GPU prover optimization investigation (cp_prove)

Status: **no verified speedup found.** This document records a systematic, instrumented
search for the "meaningful optimizations ... in prover time, prover memory" that the
FlashProve technical report (§7, Limitations) says were not incorporated. It exists so the
negative results are not lost, and so the next person does not re-walk the same dead ends
without the evidence that closed them.

All changes here are one-off diagnostic/experimental code, gated behind new CLI flags that
default to off. With no flags set, `cp_prove`'s behavior and output are unchanged from
upstream. None of this is proposed for merging upstream as-is; it is a lab notebook.

Everything below was tested on `prover/examples/target_small_f2_roomy` (the small synthetic
model), on rented H100s (Vast.ai), using `-measure` and the repo's own `BLK_FOLD_PRINT` /
`CHAIN_TIMES` instrumentation (the latter force-enabled at the source level, see below, since
the binary's own anti-`LD_PRELOAD` re-exec strips the environment variables that would
normally gate them). Every timing claim was repeated 3+ times; every correctness claim was
checked with `cp_verify` returning `YES`.

## Scope

Per the report, "meaningful optimizations" fall into two different layers:

1. **Protocol/IOP design** ("more careful application of the IOP and zkML toolkits") —
   genuine cryptography research: new sumcheck/fold constructions, code-rate choices,
   soundness/efficiency tradeoffs. Out of reach without deep IOP-theory expertise.
2. **Hash-based compilation** ("fine-tuning... Steps 2/3, the BCS transformation") —
   explicitly called "engineering" in the report's own technical overview (§4). This is
   the layer we focused on: CUDA kernel structure, memory/stream management, concurrency.

We measured where time actually goes in a pass (toy scale, `-measure`, stage_clock's own
stream/host split): the fold/sumcheck closing (`eqcol_close`) and the hash-chain commitment
(`frame_first`) are comparable in size; `eqcol_close` is somewhat larger. Six hypotheses
were tested against these hot spots.

## What's in this branch

All changes are off by default. CLI flags, each independently toggleable:

| Flag | File(s) | What it does |
|---|---|---|
| `-batch-fetch` | `prover/hiding/door_chain.cuh`, `prover/prover/chain_run.cuh` | Batches `chain_fetch()`'s per-call blocking `cudaMemcpyAsync`+`cudaStreamSynchronize` into one sync per `close_by_frame()` call instead of one per fetch (was: 75 blocking round-trips/request at toy scale). |
| `-parallel-fold` | `prover/engine/fold.cuh` | Threads `CardFold::bind()`'s host-side O(n²) fold loop across a small, capped thread pool. |
| `-parallel-streams` | `prover/prover/chain_host.cuh`, `prover/prover/chain_frame.cuh` | Gives each `in_step` slot (concurrent request) its own CUDA stream instead of all sharing `cudaStreamPerThread`. **Known to have an intermittent correctness bug — see below. Do not use.** |
| `-double-buffer-planes` | `prover/engine/tiled.cuh` | Double-buffers `CardTiled::planes_` (the span-by-span GPU scratch space for tiled multiplication) so span N+1's write kernel can run on a second internal stream while span N's multiply kernel still reads the other buffer. Correct, no measured gain at toy scale (see below). |

Also: `prover/prover/stage_clock.h`, `request.cuh`, `chain_frame.cuh`, `chain_run.cuh` split
several of the built-in `-measure` stages (`Closing`→`Fold`/`Walk`/`EqColBuild`/`EqColRow`/
`EqColClose`; `Frame`→`FrameFirst`/`FrameLater`; `End`→`End`/`EndSeal`/`EndAfter`) to localize
cost within each bucket — pure instrumentation, no behavior change. `chain_run.cuh`'s
`ChainClock` (gated upstream by `TESTENV("CHAIN_TIMES")`, which the binary's own anti-tamper
re-exec strips before `main()`) is force-enabled at the source level (`on(true)` instead of
`on(TESTENV(...) != nullptr)`) so it can be used at all; this is why it is a source change and
not an env var.

## Results (six hypotheses, five negative, one unproven at scale)

### 1. Batch `chain_fetch`'s blocking syncs — no gain
`close_by_frame()` made 75 blocking `cudaMemcpyAsync` + `cudaStreamSynchronize` round-trips per
request (~422 elements/call), a textbook "many small syncs" anti-pattern. Implemented batching
(`-batch-fetch`), verified correct (`cp_verify` → YES), measured **zero difference** vs.
baseline across 3 runs each. An isolated CUDA microbenchmark (75x async+sync-each vs.
async+sync-once, same H100) explained why: per-call `cudaStreamSynchronize` overhead on this
CUDA 13.2 / driver 595.71 combination is ~0.3µs — the classic anti-pattern's assumed cost
(tens of µs) doesn't hold on current hardware/toolkits.

### 2. Thread-parallelize `CardFold::bind()` — net slower
`bind()` is a pure CPU, single-threaded O(n²) loop over a dense Gram matrix (confirmed via
stage_clock: stream time == host time during this window). Parallelized it (`-parallel-fold`)
across a thread pool. At 128 threads (matching `hardware_concurrency()` on a 128-vCPU host):
**~3x slower**. Capped at 8 threads: still ~15-20% slower. `CardFold` is capped at ≤1024 rows
by the original authors' own design (`prover/engine/fold.cuh`'s header comment) — the
per-round workload is too small for OS thread-creation cost to pay off, at any thread count
tested.

### 3. Per-request CUDA streams — real race condition, abandoned
Every `in_step` slot's `ChainRun` was bound to the literal `cudaStreamPerThread`, serializing
cryptographically-independent requests through one stream. Gave each slot its own stream
(`-parallel-streams`). Correctness held in most trials (`cp_verify` → YES on all members,
repeatedly) but **failed intermittently** (~1 in 11 runs) with `internal error: a stretch of
rounds ends inside a sum-check` (`prover/hiding/transcript_shell.cuh`). Root cause (found, not
fixed): `EngineGpu` (`prover/engine/engine_gpu.cuh`) is one object shared by every `in_step`
slot, with its own internal `stream_` used for `CardFold` construction and
`search_nonce()`/proof-of-work — meaning even a fully-safe version of this change would only
ever affect the secondary hash-chain layer, not the dominant fold compute, which stays
serialized through the shared engine regardless. **Not pursued further**: fixing the race
would require auditing/reworking `EngineGpu`'s internal state-sharing assumptions, which is
out of scope for a one-off diagnostic change, and the upside is capped even if fixed.

### 4. Buffer reuse in `bind()` — no gain, and a non-obvious reason why
Hypothesis: `bind()`'s two `std::vector<E>` allocations per round could be made persistent/reused
instead of freshly allocated+destroyed every call. Tested **locally, free, no GPU**, with a
realistic microbenchmark (full-array writes, not just touching one element): naive reuse via
`resize()` was actually **~15-43% slower**, because `std::vector<T>(n)` value-initializes
(zeros) its elements, and a *fresh* allocation gets that zeroing nearly free via the OS's
copy-on-write zero pages, while a *reused*, already-dirty buffer needs genuine memory writes to
re-zero. A corrected variant (skip zero-init entirely, since the real loop overwrites every
element before reading it) showed **no measurable difference** once modeled with full,
realistic memory traffic — the "wasted" zero-init is a small fraction of total write volume
once every byte gets touched anyway.

### 5. Simplify `fold_pair`'s field arithmetic — already minimal
`fold_pair(a, b, g) = a + g * (b - a)` (`blocks/field/mle.h`) is one multiply, one subtract, one
add in the extension field — the standard linear-interpolation formula, already algebraic
minimum. No redundant computation to hoist. Not touched (and shouldn't be, without deep
finite-field/SIMD expertise: silently wrong modular arithmetic would produce an unsound proof
that may still look "accepted").

### 6. Double-buffer `CardTiled::planes_` — correct, unproven at scale
`CardTiled` (the per-span GPU tiled-multiplication engine inside `CardFold`) reused one scratch
buffer (`planes_`) across spans — `span()` overwrites it, `multiply()` immediately reads it
back, serializing every span. Same pattern found independently at two other sites
(`row_pass_scratch` in `prover/weights_blocks/block_weights.cuh`; the single shared `EngineGpu`
stream above) — all three are the *same* deliberate memory-for-concurrency tradeoff, consistent
with the report's own "prover memory" framing, not an oversight. (`row_pass_scratch`'s call
site also interacts with CUDA graph capture — `cudaStreamIsCapturing` checks in
`prover/prover/layer_prove.cuh` — making it a materially riskier place to add concurrency than
`CardTiled`, which has no such entanglement; that's why this change targets `CardTiled`.)

Implemented double-buffering (`-double-buffer-planes`): two `planes_` buffers, a second
internal CUDA stream for `span()`'s write kernel, per-slot `cudaEvent_t`s so the GPU scheduler
can overlap span N+1's write with span N's multiply without any host-side wait. Self-contained
inside `CardTiled` — no caller (`fold.cuh`) needed to change. Verified correct (`cp_verify` →
YES, 3/3). **No measurable timing difference** vs. baseline across 3 runs each
(`eqcol_close` 0.072s vs. 0.073s, every other stage matching within noise).

Why this might still be real at full scale, untested: our own `BLK_FOLD_PRINT` data shows the
span-processing phase totals only ~0.039s summed across all 25 `CardFold` instances in a toy
pass — meaning each instance likely has very few spans, each kernel already microseconds-small.
There isn't enough work per span at toy scale for pipelining to show above measurement noise.
At full Llama4 Maverick scale (far more spans, far larger per-span kernels) this could behave
differently, but that is not something we could test against the toy example, and is not
claimed here.

## Independent corroboration

A separate, apparently well-resourced research fork
([wakesync/flashprove](https://github.com/wakesync/flashprove)) ran its own GPU experiments
(H100 vs. H200, memory-bound prover hypothesis) and reached the same kind of result: PR #44
closed issue #43 "as a measured negative" — the observed 1.56x H200 throughput advantage was
fully explained by host-speed variance between rental boxes, not GPU memory. Their own
prover/commit speedup tracking issue (#39) remains open and unresolved. This is independent
evidence (different team, different methodology, real GPU spend) that the GPU-prover layer
does not have an easily-reachable systems-level win sitting on the table — consistent with
what this investigation found.

(That fork *did* land a real, verified, merged CPU-side win — PR #27, ~17.8% faster
`cp_verify` via a small-object optimization on big-integer "limb" allocation in
`blocks/numbers/real.h` — but that's the verifier, not the GPU prover, and out of scope for
this document, which is GPU-only per the investigation's own scope.)

## Cost

~36 GPU-hours across four H100 rentals (Vast.ai), ~$93 total. A meaningful fraction of that
(~30 hours) was instances left running during local code reading/writing that didn't need the
GPU, not time spent on builds or tests — a process mistake, not a necessary cost of the
investigation. The actual GPU-touching work (builds + test runs) was on the order of a few
hours.

## Bottom line

Six real, implemented, measured hypotheses; five negative with understood mechanisms, one
correct-but-unproven-at-the-only-scale-we-could-test. No verified speedup. The report's claim
that "meaningful optimizations" exist is plausible and likely correct, but — based on this
investigation plus independent corroboration from a separate team — that headroom appears to
sit in the IOP/protocol-design layer (genuine cryptography research) rather than in reachable
CUDA systems engineering, or it requires full-model-scale testing this investigation's budget
could not reach.
