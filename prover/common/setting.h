// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Compile-time constants that the sources of the committer program cp_commit and the prover program
// cp_prove share: first the macros that separate the production build (SERVING_BUILD in the code)
// from test builds, then parameters and bounds of the protocol, then the hash's widths and the
// lengths of hashed lists. Many of the values belong to the proof format: the sources of the
// verifier program cp_verify (verifier/src) and scales/policy_tables.py repeat some of them, so a
// change here must be made there too. The parameter file is policy.txt. F is the field of p
// elements (prover/field/field_names.h). Some constants are read by no code; their comments say so.
// A backquoted name in parentheses after a check, such as (`hid-tau`), is that check's rejection
// name, which the program prints when it rejects.
#pragma once
#include "hash_commitment/hf_format.h" // DIGEST_WORDS, hf_version_word(), HF_BAR
#include "common/state8.h"            // policy.txt's target and bits of proof of work (s8::now)
#include "field/field_names.h"        // FIELD_P, FIELD_BITS
// A production build ("serving build" in printed lines) is compiled with SERVING_BUILD.
// prover/build.sh sets it for the targets `serving`, `serving_cp_prove` and `serving_cp_commit`,
// and run without a target it builds `serving`. SERVING_OR(a, b) is a in a production build and b
// in every other build; the argument that is not chosen is not compiled.
#ifdef SERVING_BUILD
#if defined(CP_PROVE_NO_MAIN) || (defined(CHAIN_STAND_IN) && CHAIN_STAND_IN)
#error "SERVING_BUILD cannot be combined with CP_PROVE_NO_MAIN or CHAIN_STAND_IN, which build test programs"
#endif
#if defined(HC_COMMIT_NO_MAIN) || defined(CP_COMMIT_NO_MAIN) || defined(HC_COMMIT_PHASE_A_ONLY) || defined(HC_COMMIT_FOLD_NO_CODE) ||                \
    (defined(KEYS_FROM_A_FILE_PART) && KEYS_FROM_A_FILE_PART)
#error "SERVING_BUILD cannot be combined with HC_COMMIT_NO_MAIN, CP_COMMIT_NO_MAIN, HC_COMMIT_PHASE_A_ONLY, HC_COMMIT_FOLD_NO_CODE or KEYS_FROM_A_FILE_PART, which build test programs or parts of programs"
#endif
#if defined(REFUSAL_TEST_WORDS) && REFUSAL_TEST_WORDS
#error "SERVING_BUILD cannot be combined with REFUSAL_TEST_WORDS, which builds test programs"
#endif
#ifdef LR_TEST_KEYS
#error "SERVING_BUILD does not go with LR_TEST_KEYS, a test switch that enables nothing"
#endif
#if defined(TEST_KEYS_FROM_A_FILE) && TEST_KEYS_FROM_A_FILE
#error "SERVING_BUILD does not go with TEST_KEYS_FROM_A_FILE, which makes test programs with keys from a file"
#endif
#define SERVING_OR(a, b) a
// For TESTENV() below. A call, not a bare nullptr, so that code made dead by the null result causes
// no compiler warning.
static inline const char *no_test_variable() {
  return nullptr;
}
#else
#define SERVING_OR(a, b) b
#endif
// No other source reads TEST_KEYS_FROM_A_FILE, and no build script sets it.
#ifndef TEST_KEYS_FROM_A_FILE
#define TEST_KEYS_FROM_A_FILE 0
#endif
// BUILT_FOR_TESTS_ALONE is 1 in every build except the production build. Every other cp_commit and
// cp_prove, and every program that includes their sources, accepts the test options, and some of
// these builds drop a mask (a random value added to a sent value), send a false value or print
// hidden values, under the honest program's name. Where the macro is 1:
//   - the program rejects a model's weight files as soon as it knows which weights the run names
//     (`tests-build-on-a-model`: prover/hash_commitment/hf_commit.cu; read_isk(),
//     prover/prover/isk.cuh);
//   - its first lines of output and its line "setting:" say "for tests alone";
//   - the last line of cp_commit never says "can be handed out".
// The first of these is to keep every program that accepts a test option away from a model's weight
// files.
// On such files the test options would take the following away. Under -test-seed every mask, salt
// and pad follows from the file that the option names. So the key file (`ivk/key.bin`) and the leaf
// files (`ivk/leaf/<k>.bin`), which are derived from the weights, would hide the weights from no
// one who holds the option's file, and the same person could unmask each proof's opened columns.
// cp_commit's worst-case check of the weights would be skipped unless -worst-case or
// -worst-case-table asks for it.
// -test-bits accepts public files that a run at 128 bits (BAR, below) rejects: a lower target (line
// `sec`: the bits that the programs' error accounting must reach), against which the conditions for
// hiding are checked too; a parameter file outside the standard parameter form, such as one without
// the clamps (lines `rcl`, `gcl`); no layout.txt.
// -test-coins reads each batch's 32 random bytes from a file, and then nothing is hidden from
// whoever has that file.
// One option that is no test option writes to disk what hiding rests on, and the production builds
// of both programs accept it: -keep-for-the-audit. cp_commit keeps the 32 bytes of system
// randomness that went into the key file and into each leaf file; cp_prove keeps the 32 random
// bytes of its one batch. The bytes go into the subdirectory `audit` of the secret key directory
// (`isk`), for the audit with the secret key (tools/audit/README.md). With the seed
// (`isk/seed.bin`) they reveal every mask in those files and every mask, pad and salt of that
// batch's proofs. No macro restricts the option, only checks at run time. cp_commit rejects it on a
// model's weight files (`cp-commit-kept-on-a-model`: prover/hash_commitment/hf_commit.cu). So does
// cp_prove (`cp-prove-kept-on-a-model`: read_isk()), which also rejects it when run as a service
// (`cp-prove-kept-in-a-service`). With or without the option, cp_prove rejects a secret key
// directory whose `audit` is not empty or cannot be listed (`cp-kept-for-the-audit`: read_isk());
// that check sees that nothing is in `audit`, not that the audit has said YES: emptying `audit` by
// hand satisfies it. That nothing of the commitment and none of that batch's responses is handed
// out before the audit's YES is left to the operator (docs/operators-duties.md).
// A test that builds an altered copy of the sources changes the 0 below to 1 in the copy, so that
// a production build of the copy fails the static_assert below.
#ifdef SERVING_BUILD
#define BUILT_FOR_TESTS_ALONE 0
#else
#define BUILT_FOR_TESTS_ALONE 1
#endif
// BUILT_SAYS: part of the first lines of output and of the rejection message. BUILT_SETTING: part
// of the line "setting:".
#define BUILT_SAYS "BUILT FOR TESTS ALONE (every build is, except prover/build.sh run without an argument)"
#if BUILT_FOR_TESTS_ALONE
#define BUILT_SETTING "; built for tests alone"
#else
#define BUILT_SETTING ""
#endif
static_assert(!(BUILT_FOR_TESTS_ALONE && SERVING_OR(1, 0)), "THE build from a changed copy of the sources, into which a test has laid the mark of a build for tests alone");
// TESTENV(name) is getenv(name), for an environment variable that only tests or tuning set. In a
// production build it is a null pointer, and the variable's name is not compiled into the program.
#define TESTENV(name) SERVING_OR(no_test_variable(), getenv(name))
// E, the field from which challenges are drawn, is the extension of F of degree EDEG. EWORDS: the
// 32-bit words of one element of E. EBITS: the length of an element of E in bits.
#define EDEG 3
#define EWORDS (2 * EDEG)
#define EBITS (FIELD_BITS * EDEG)
// ETAU: the number of random combinations in the approximate range check of a sub-proof (the
// argument for one part of the forward pass; `fork` in names). The default is 128, the target in
// bits, because that check has no proof of work. The parameter file's line `tau` must equal ETAU
// (`hid-tau`), and cp_verify rejects a file whose target exceeds `tau` (`policy-tau-hb`).
// prover/build.sh passes on ETAU from its environment; its production targets reject the variable.
// ENJ: printed in the two lines named at ETAUF below, and a factor in one buffer's size
// (hash_words(), prover/attn/fork.cuh).
#ifndef ETAU
#define ETAU 128
#endif
#define ENJ 5
// ETAUF, ENJW: cp_prove only prints them, in the lines "p258 setting" (print_lines(),
// prover/prover/closed_front.h) and "setting:" (print_setting(), prover/prover/base.cuh).
#define ENJW 5
#define ETAUF 168
// The bounds of cp_commit and cp_prove on lkW, the last number of the parameter file's line
// `table` (`w-shape-lk`): W, the matrix of the committed weights, has 2^lkW columns.
#define ELKW_LEAST 4
#define ELKW_MOST 18
// W has at most 2^ELMW rows (`w-shape-mW`). No line of the parameter file can change the bound:
// cp_verify rejects a line `lmW` (`policy-call-line`).
#define ELMW 22
static_assert(ELMW == 22, "ELMW must be 22: another row limit for W* would need the line lmW in the parameter file, "
                          "which this version word does not allow");
// In a request, one product tree of a lookup forest (the trees of one group of a sub-proof's lookup
// calls; "identity" in names and messages) has at most 2^cap keys. The cap is the parameter file's
// line `lidb` (absent: 22; present: 23 to 30), which Prover::read_inputs()
// (prover/prover/inputs.cuh) stores in identity_cap(). The file's iota lines, which state the
// number or the size of a lookup call's trees, depend on it. EIDB is the largest cap that a build
// supports; read_inputs() rejects a larger one (`plan-identity-cap`). The default is 28, the cap of
// the full-size example configuration (prover/examples/target_full_f2); at that cap full-size runs
// have been made. prover/build.sh passes on EIDB=22, 24 or 28 from its environment; its production
// targets reject the variable.
#ifndef EIDB
#define EIDB 28
#endif
static_assert(EIDB == 22 || EIDB == 24 || EIDB == 28, "EIDB must be 22, 24 or 28");
// The cap of this run: 22 in a program that never calls Prover::read_inputs().
inline int &identity_cap() {
  static int cap = 22;
  return cap;
}
// In a sub-proof's range check, every entry of a random combination Z must be in [-2^ELBZ, 2^ELBZ).
#define ELBZ 17
#include <cstdlib>
// SLOT_COUNT=1 builds test programs. With it, cp_prove's sources check at the end of each sub-proof
// that every slot of the message array was written exactly once, with a mask entry that is not
// zero (prover/hiding/slot_count.cuh). No test of this repository builds cp_prove this way.
#ifndef SLOT_COUNT
#define SLOT_COUNT 0
#endif
static_assert(!(SLOT_COUNT && SERVING_OR(1, 0)),
              "SLOT_COUNT=1 builds a test program and cannot be combined with a serving build");
// shapeF() (prover/hiding/trees.cuh) and table_shape_at()
// (prover/hash_commitment/hf_opening_params.h) give a row commitment no code longer than 2^ELNO. A
// row commitment: a matrix's rows, encoded, their columns hashed into one Merkle tree.
#define ELNO 24
// No code reads ELIDO.
#define ELIDO 28
// No code under prover/ reads DRAW_TRIES.
// A proof's hash chains draw their challenges from hash output in blocks/chain/chain.h, under a
// cap of their own, blk::DRAW_TRIES = 6 (blocks/base.h): after that many rejected 64-bit words a
// field element is 0.
constexpr unsigned DRAW_TRIES = 3;
// The widths of the hash (BLAKE2b) in 32-bit words. DW: a digest as it is sent, stored or put into
// a list: the function's result cut to DW words. SW: a state or chaining value inside the function.
// BLKW: a block. DIGEST_WORDS must be 8 (prover/hash_commitment/hf_table_format.h), so what is
// written below for DW = 10 is never used.
constexpr int DW = DIGEST_WORDS, SW = 16, BLKW = 32;
// Other lengths in 32-bit words. SALTW: a tree leaf's salt, SALTW / 2 symbols of 64 bits; it does
// not depend on DW. COINKEYW: a key of the generator of private randomness (ChaCha20:
// prover/common/coin_chacha.h). NONCEW: the nonce in that generator's label. RETRYQ is no length:
// try number tr >= 1 of a random value has the retry word RETRYQ * tr + q, where q is the value's
// index in its block, and the first try has the retry word 0. SESSW, LAMW: lengths of two further
// salts, which cp_commit and cp_prove use only in the derived lengths below.
constexpr int SALTW = 10;
constexpr int SESSW = SALTW, LAMW = DW, COINKEYW = 8, NONCEW = 4, RETRYQ = 8;
static_assert(LAMW == DW, "LAMW must equal DW: a leaf's list and a node's list have the same length");
// FILLC: a column of a row commitment is filled to a multiple of FILLC symbols
// (prover/hiding/trees.cuh). SH2_DIGB: the bytes that the search for a row commitment's shape, in
// the same file, counts for each digest of a Merkle path when it compares proof sizes; it is not
// derived from DW. SLABR: the symbols of a column, 64 bits each, that fill one block of the hash.
constexpr int FILLC = 8, SH2_DIGB = 32, SLABR = BLKW / 2;
// BAR: the requested bits below which no answer of cp_verify is a verification. cp_commit
// requests that many bits of it, and another number only under -test-bits.
constexpr int BAR = HF_BAR; // 128
// Lengths derived from these, in 32-bit words except HEXW. TBLW: the parameter file's
// lines with the four public tables' digests, one line for each word. ROOTW: what a Fiat-Shamir
// transcript absorbs with a Merkle root: three words, then the root. S120W: the list hashed under
// the domain-separation number 120; S120PAD: its zero words, which keep a digest in one block.
// THPW: a chunk of the tree hash (see TH_LIST). HEXW: a digest's hex digits.
// Under prover/, only this file reads MRW, DJW, S75W, HEXW, S107W, LEAFW, S121W, NODEW,
// S121PAD, S124W and PATHW(). verifier/src/vf_widths_and_bounds.h defines the hash's three widths and
// TBLW, HEXW and THPW separately, with the same values. The commitment record's lengths are in
// prover/hash_commitment/hf_table_format.h (HF_RECORD_WORDS, HF_REC0_WORDS).
constexpr int TBLW = 4 * DW, ROOTW = DW + 3;
constexpr int MRW = 2 + DW, DJW = 2 + 3 * DW, LEAFW = LAMW + DW, NODEW = 2 * DW;
constexpr int S75W = DW + 1, S107W = SESSW + 2 + 4 * DW, S124W = 3 * DW;
constexpr int S121PAD = (3 + 2 * DW) % BLKW + DW > BLKW ? BLKW - (3 + 2 * DW) % BLKW : 0, S121W = 4 + 3 * DW + S121PAD;
constexpr int S120PAD = (3 + 3 * DW) % BLKW + DW > BLKW ? BLKW - (3 + 3 * DW) % BLKW : 0;
constexpr int S120W = 3 + 3 * DW + S120PAD + DW, HEXW = 8 * DW, THPW = 16 * DW;
// The tree hash reduces a list of words to one digest, level by level. Its two domain-separation
// numbers: the chunks (THPW words each) of the list itself are hashed under TH_LIST, and the chunks
// of every level of digests above it under TH_UPPER.
constexpr unsigned TH_LIST = 10, TH_UPPER = 11;
constexpr int PATHW(int leafd) {
  return 1 + LAMW + DW * leafd;
}
static_assert(DW != 10 || (TBLW == 40 && ROOTW == 13 && MRW == 12 && DJW == 32 && LEAFW == 20 && NODEW == 20 &&
                           S75W == 11 && S107W == 52 && S121PAD == 9 && S121W == 43 && S124W == 30 && S120PAD == 0 &&
                           S120W == 43 && HEXW == 80 && THPW == 160 && PATHW(6) == 71),
              "one of the lengths of the format's lists has changed, at DW = 10");
static_assert(DW != 8 || (TBLW == 32 && ROOTW == 11 && MRW == 10 && DJW == 26 && LEAFW == 16 && NODEW == 16 &&
                          S75W == 9 && S107W == 44 && S121PAD == 0 && S121W == 28 && S124W == 24 && S120PAD == 5 &&
                          S120W == 40 && HEXW == 64 && THPW == 128 && PATHW(6) == 57),
              "one of the lengths of the format's lists has changed, at DW = 8");
// The version word: blk::SERVED_VERSION (blocks/frame/version.h), four characters that only
// identify the format, the field and the hash ("H821" at p = 2^64 - 2^32 + 1). It is an int here
// because messages print it with %d.
#define VERSION_WORD ((int)hf_version_word())
// In a matrix under a sub-proof's range check, a symbol packs up to three cells (small integers)
// of adjacent rows in one column: cell[3 i] + 2^ERAD cell[3 i + 1] + 2^(2 ERAD) cell[3 i + 2].
#define ERAD 20
static_assert((1ll << ERAD) >= 4 * (1ll << ELBZ) - 1, "the radix 2^ERAD must be at least 4 * 2^ELBZ - 1");
static_assert((unsigned long long)((4 * (1ll << ELBZ) - 2) * (1 + (1ll << ERAD) + (1ll << (2 * ERAD)))) < FIELD_P,
              "a symbol of three packed cells must stay below p");
// An entry of Z plus 2^ELBZ is split into a low digit of EZD0 bits and a high digit of EZD1 bits.
#define EZD0 10
#define EZD1 (ELBZ + 1 - EZD0)
static_assert(ETAU >= 32 && ETAU % 32 == 0, "ETAU must be a multiple of 32: support and sign bits are kept in whole words");
static_assert(((ETAU & (ETAU - 1)) & ((ETAU & (ETAU - 1)) - 1)) == 0,
              "ETAU must be a power of two or the sum of two: the digits' rows lie in one or two pieces of a lookup's keys");
// One-off (not upstream): toggles per-request CUDA streams for a batch pass (chain_frame.cuh,
// chain_host.cuh), set from args.parallel_streams via -parallel-streams. See chain_host.cuh.
static bool g_parallel_streams = false;
// One-off (not upstream): double-buffers CardTiled's planes_ work space (engine/tiled.cuh) so
// span()'s plane-writing kernel for span N+1 can run concurrently with multiply()'s kernel for
// span N, on a second internal stream, instead of serializing every span through one buffer. Set
// from args.double_buffer_planes via -double-buffer-planes.
static bool g_double_buffer_planes = false;
static_assert(ETAU <= 256 && ETAUF <= 256, "ETAU and ETAUF must be at most 256: a column's coefficients are read from "
                                           "the first 256 bits of two hash outputs");
// One-off (not upstream): toggles CardFold::bind()'s threaded path (prover/engine/fold.cuh), set
// from args.parallel_fold (prover/prover/inputs.cuh) via -parallel-fold. Declared here, a header
// included from nearly everywhere unconditionally, so every translation unit that compiles
// cp_prove.cu's sources sees the same declaration regardless of include order elsewhere.
static bool g_parallel_fold = false;
