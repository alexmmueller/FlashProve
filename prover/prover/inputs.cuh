// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Three steps of the start-up of cp_prove, the prover program, in the order in which Prover::run()
// (service.cuh) calls them:
//   read_inputs()  reads the command line, isk.txt in the secret key directory (isk.cuh) and the
//                  configuration (the parameter file policy.txt, the quantization scales KL.bin,
//                  the layout file layout.txt), rejects what this build does not support and
//                  derives the sizes used everywhere
//   make_tables()  builds the public tables
//   plan_memory()  sets up the weights and allocates the GPU memory that all sub-proofs share
// Each returns 0, or 2 if it rejects its inputs; read_inputs() also 1 (bad command line, output
// that cannot be written, no GPU). CK() ends the process if its CUDA call fails. cudaMalloc(),
// cudaFree(), cudaMemset() and cudaMemcpy() are macros here, defined at the end of fork_memory.h.
// Identifiers say "fork" for a sub-proof.
#pragma once
#include "common/blas.h"
#include "engine/fold.cuh"  // one-off (not upstream): g_parallel_fold
#include "prover/prover.h"
#include "prover/slot_pads_second.cuh"
#include <dirent.h>
#include "common/pad_tree_rows.h"
#include "common/parameters_hash.h"
#include "frame/hiding_conditions.h" // ../blocks: what the arguments for hiding ask of a parameter file

inline int Prover::read_inputs() {
  process_start = epoch();
  pt_alloc = [](void **p, size_t n, int kind) {
    // The layer sub-proofs' lookup multiplicities and their bit masks (kind 1; each kind is defined
    // at pt_alloc, prover/hiding/product_test.cuh) are written before the product tests (the
    // lookups' grand-product checks) and read in them. Other steps use the arena, the layer
    // sub-proofs' shared scratch memory (prover/prover/fork_memory.h), in between, so these buffers
    // are allocated outside it.
    const bool counts = kind == 1 && fork_memory.arena_open;
    const cudaError_t e = fork_alloc(p, n,
                                     kind == 0 || kind == 3 || counts ? Place::Own
                                     : hz::g_late                     ? Place::Late
                                                                      : Place::Fork);
    if (counts && e == cudaSuccess) fork_memory.of_a_request.push_back({*p, n});
    if (kind == 3 && e == cudaSuccess) fork_memory.laid_once.push_back({*p, n});
    return e;
  };
  forward_scratch_alloc = forward_pass_alloc = [](void **p, size_t n) {
    return fork_alloc(p, n, Place::Fork);
  };
  forward_pass_buffers_last = false;
  hz::alloc_hook = [](void **p, size_t n, hz::Place k) {
    return fork_alloc(p, n,
                      k == hz::FORK    ? (hz::g_late ? Place::Late : Place::Fork)
                      : k == hz::SHAPE ? Place::Shape
                      : k == hz::OWN   ? Place::Own
                                       : Place::Lasting);
  };
  // An attention sub-proof uses its region (prover/hiding/late.cuh) only in the first half of its
  // run, so its Late buffers reuse that memory as far as they fit.
  hz::region_hook = [](size_t n) {
    void *p = nullptr;
    fork_memory.early_begin();
    CK(fork_alloc(&p, n, Place::Fork));
    fork_memory.early_end();
    return p;
  };
  hz::late_hook = [] {
    fork_memory.late_begin();
  };
  if (!args.parse(argc, argv)) return 1;
  g_batch_fetch_mode = args.batch_fetch; // chain_run.cuh: toggles close_by_frame's fetch strategy
  g_parallel_fold = args.parallel_fold;  // engine/fold.cuh: toggles CardFold::bind()'s threading
  g_parallel_streams = args.parallel_streams; // chain_frame.cuh/chain_host.cuh: per-slot CUDA streams
  g_double_buffer_planes = args.double_buffer_planes; // engine/tiled.cuh: CardTiled's planes_ buffer
  if (int rc = read_isk()) return rc;
  if (!name_what_shows()) return 1; // the first lines of output
  // Exit handlers run in the reverse order of their registration, so the CUDA runtime must register
  // its handlers before arm_ends() registers ours, or ours could not zero GPU memory. GPU 0 is also
  // the GPU of host threads that set no device, such as those of the product tests and of a batch's
  // opening.
  if (cudaSetDevice(0) != cudaSuccess) {
    printf("CUDA finds no GPU\n");
    return 1;
  }
  // This limit says how many bytes of stack every thread of a kernel gets
  // (NVIDIA, CUDA Runtime API reference, cudaDeviceSetLimit()). It is set to what the kernels need,
  // so that the runtime reserves no more GPU memory for stacks than that. The test suite does not
  // check, in the built programs, that every kernel's stack frame is within 800 bytes.
  CK(cudaDeviceSetLimit(cudaLimitStackSize, 800));
  cudaFree(0);
  arm_ends();
  // prover/hiding/trees.cuh, which derives the sub-proofs' private random values (`coins` in
  // identifiers), calls this hook from key_refuse(), replay_refuse() and its checks on drawn cells,
  // after a message on stderr and before exit(2): for example at a key of zeros, a value requested
  // without a key or after the wipe, a cell drawn twice, an attempt ended twice. Each is a fault
  // of the program. These checks read keys, counters and cell numbers and no random value. All get
  // one name in the last line of output; its prefix "own-test-" makes the exit code 9, not exit()'s
  // 2 (stop(), prover/prover/ends.h). The hook is static: nothing sets the one in
  // prover/hash_commitment/hf_keeper.cu, a separately compiled part of cp_prove.
  hz::refusal_is_told_to = [](const char *) { name_the_end("own-test-coin-library"); };
  umask(077);
  static Prover *const the_prover = this;
  g_end_wipe = [] {
    const bool zeroed = fork_memory.wipe();
    hz::key32_wipe();
    slot_pads_second.wipe_the_key();
    the_prover->wipe_the_seed();
    the_prover->batch.for_the_library.wipe();
    explicit_bzero(&the_prover->batch.key_word, sizeof the_prover->batch.key_word);
    for (Prover::Opening &o : the_prover->opening)
      o.rode.for_the_library.wipe(), explicit_bzero(&o.rode.key_word, sizeof o.rode.key_word);
    return zeroed;
  };
  if (args.test_coins.empty())
    if (const char *word = test_word_in_the_environment()) {
      printf("the environment holds %s, a test variable" SERVING_OR("", "; test variables are accepted only with -test-coins") "\n", word);
      return 2;
    }
  // Test builds only: a fill byte for every buffer that the GPU engine allocates or reuses
  // (`card_test_byte`, prover/engine/device.cuh), to expose reads of memory that was never written.
  // prover/weights_commitment/lent_memory.h reads the same variable for the memory that it lends.
  SERVING_OR(, if (const char *byte = TESTENV("LENT_POISON")) blk::card_test_byte = (int)strtoul(byte, nullptr, 16) & 255;)
  // The fault-injection hooks of prover/hiding/lost_coins_door.cuh and
  // prover/hiding/lost_copies_door.cuh act only in a test run (-test-coins), and only on synthetic
  // weights under a test seed.
  lost_coins_door::open_if(!args.test_coins.empty());
  near_host_pin(0);
  // Reads the parameter file once, accepts only one textual form of it and checks its numbers
  // against the table of limits (limit_missed(), prover/common/parameter_file.h). The table bounds
  // the numbers that size a buffer, count a loop or serve as a shift count, a divisor or an index.
  if (!policy.read(args.policy)) return 2;
  // Checks the version word, and stores the error accounting's target in bits (line `sec`) and the
  // bits of proof of work (line `work positions`), which policy.read() has bounded.
  if (const std::string why = s8::set(policy.frame); !why.empty()) {
    printf("policy: %s\n", why.c_str());
    return 2;
  }
  if (isk_version_word != s8::ver()) {
    printf("refused (cp-isk): isk.txt has the version word %u and its parameter file %u\n", isk_version_word, s8::ver());
    return refused("cp-isk");
  }
  {
    // The conditions that the arguments for hiding put on a parameter file
    // (blocks/frame/hiding_conditions.h). Only the part on the weight table is checked here, which
    // follows from the file alone; chain_make() (prover/prover/chain_frame.cuh) runs the complete
    // test, blk::hiding_conditions_missed(), through blk::configure(), once the sub-proofs' shapes
    // are known. Here a file that misses a condition is rejected unless isk.txt says "seed test".
    // No key protects that line, and a production build (`SERVING` in identifiers) has already
    // rejected it, in read_isk(). cp_commit -test-seed writes the line, and nothing is hidden on
    // such a commitment. Every number in the printed line follows from the parameter file alone.
    const std::string missed = blk::hiding_conditions_missed_by_the_file(policy.frame);
    if (!(missed.empty()
              ? closed_front.say_closed("p258 the parameter file meets the conditions for hiding that follow from it alone "
                                        "(their names and what each asks: blocks/frame/hiding_conditions.h) "
                                        "(these conditions count positions and masks; they say nothing of how a record's masks are made)\n")
              : closed_front.say_closed("p258 the parameter file misses conditions for hiding "
                                        "(their names and what each asks: blocks/frame/hiding_conditions.h): %.300s\n",
                                        missed.c_str()))) // at most 300 bytes: the line fits say_closed()'s buffer
      stop(End::WriteFailed);
    if (!missed.empty() && !isk_test_seed) {
      isk_fault = "prover-only-hiding-conditions";
      name_the_end(isk_fault); // for the last line
      return 2;
    }
  }
  hz::COINS_WORD = s8::ver();
  lt = policy["lt"];
  Lr = policy["L"];
  ll = policy["ll"];
  d = policy["d"];
  f = policy["f"];
  la = policy["la"];
  nmin = policy["nmin"];
  D = policy["D"];
  lmb = policy["lmb"];
  seed = policy["seed_lo"] | ((u64)policy["seed_hi"] << 32);
  Lam = 1 << ll;
  NGr = Lr / Lam;
  // The Lr layers form NGr layer groups of Lam layers. Sub-proof numbers run from 0 to 2 * NGr + 1:
  // a layer and an attention sub-proof for each group, one for the output head, and number 0, whose
  // randomness streams belong to the request's pad tree CP. The format allows 59 numbers, 0 to 58:
  // the verifier's groupsBaseHeld() (verifier/src/cp_verify.cpp) rejects 2 + 2 * NGr > 59.
  // HF_STREAMS (prover/hash_commitment/hf_message_format.h) reserves the streams from number 60's on
  // for the opening. So NGr is at most 28, which limit_missed() has checked.
  // Sizes set below: Pd and Pf are d, the residual stream's width, and f, a dense layer's hidden
  // width, rounded up to powers of two. A layer writes NW states of the residual stream and has NN
  // norms, so a group has NI norm instances. LA = 2^la cells (2^ls features, NT tokens each) form a
  // block; a layer sub-proof evaluates its matrices af, qa, xg and xn (Fork) block by block.
  chain.init(NGr);
  NT = 1 << lt;
  ls = la - lt;
  lPd = l2i(d);
  Pd = 1 << lPd;
  lPf = l2i(f);
  Pf = 1 << lPf;
  NW = 2;
  NN = NW;
  NI = NN * Lam;
  lNI = l2i(NI);
  H = policy["H"];
  Gk = policy["G"];
  dh = policy["dh"];
  dq = H * dh;
  dkv = Gk * dh;

  ldh = l2i(dh);
  LA = (size_t)1 << la;
  hb_ = (int)policy.frame.sec;
  print_setting(hb_);
  closed_front.hb = hb_;
  lkW = (int)policy.frame.block_log;
  // A batch's binding tree, a Merkle tree over its requests' first hash chains, has as many leaves
  // as the file's line `slots` says: a power of two, and the most requests in a batch.
  // The numbers of the line `table` are checked later, by
  // HfParams::derive() (prover/hash_commitment/hf_opening_params.h), because whether they are valid
  // also depends on the number of committed rows.
  static_assert(PLACES_MOST == 1 << parameter_file::LEAFD_LIMIT, "the table of limits admits leafd up to 8");
  rkR = 1;
  identity_cap() = (int)policy.value("lidb");
  if (identity_cap() > EIDB) {
    printf("policy: plan-identity-cap: the parameter file allows lookup identities of up to 2^%d keys (line lidb; 22 "
           "if absent), and this build supports at most 2^%d (compile-time constant EIDB)\n",
           identity_cap(), EIDB);
    return 2;
  }
  if (policy.has("lidb"))
    printf("# the parameter file allows lookup identities of up to 2^%d keys (line lidb)\n", identity_cap());
  gate_key_step = (int)policy.value("eg");
  gate_values_most = (int)policy.value("smax");
  gate_clamped = policy.has("gcl");
  xl = (int)policy.value("xl"); // the number of limbs of the residual stream
  if (xl == 5) printf("# the stream has five limbs (line xl)\n");
  if (args.batch > (int)policy.frame.slots) {
    printf("-batch %d is too large: a batch has at most %d requests (line slots of the "
           "parameter file)\n",
           args.batch, (int)policy.frame.slots);
    return 2;
  }
  NC = (int)policy.cl.size();
  NC1 = NC;
  static_assert(MAX_CALLS == 8, "the table of limits admits 1 to 8 call lines");
  NE = policy["NE"];
  fe = policy["fe"];
  mo = NE > 0;
  LamM = mo ? Lam / 2 : 0;
  LamD = Lam - LamM;
  lfe = l2i(fe);
  KLW = 13;
  // KL.bin: a record of KLW 32-bit words per layer and, after all records, three words (kr, kgR,
  // kuR) per MoE layer.
  {
    size_t words = (size_t)KLW * Lr;
    for (int l = 0; l < Lr; l++) words += is_moe_layer(l) ? 3 : 0;
    KL.assign(words, 0);
    FILE *kf = fopen(args.layer_words.c_str(), "rb");
    if (!kf || fread(KL.data(), 4, KL.size(), kf) != KL.size()) {
      printf("layer constants: the file of quantization scales cannot be read, or it is shorter than 13 words per "
             "layer plus 3 words per layer with experts\n");
      return 2;
    }
    if (fgetc(kf) != EOF) {
      printf("layer constants: the file of quantization scales is longer than 13 words per layer plus 3 words per "
             "layer with experts\n");
      return 2;
    }
    fclose(kf);
    // Attention's shift serves as a shift count and in a divisor at set-up (APlan::shd(),
    // prover/attn/plan.cuh), so it is checked here, before any use, against the verifier's range.
    for (int l = 0; l < Lr; l++)
      if (KL[(size_t)KLW * l + 11] < 4 || KL[(size_t)KLW * l + 11] > 12) {
        printf("layer constants: af-policy-sh: layer %d: word 11, attention's shift, must be from 4 to 12\n", l);
        return 2;
      }
  }
  // layout.txt, in the parameter file's directory, is read once, here. chain_make()
  // (chain_frame.cuh) rejects a run without the file, because the format has no default layout, and
  // passes the bytes to blk::configure().
  {
    const parameter_file::Found found = parameter_file::read_once(layout_beside(args.policy), layout_bytes, parameter_file::LAYOUT_FILE_MOST);
    layout_file = found == parameter_file::Found::Read;
    if (found == parameter_file::Found::Unreadable) {
      printf("policy: plan-layout: layout.txt is there and cannot be read, is a device or a folder, or has more than 2^26 bytes\n");
      return 2;
    }
  }
  // h_par, blk::configuration_digest() of the three configuration files' bytes, as cp_commit forms
  // it. the_ranges_are_guarded() (prover/prover/isk.cuh) compares it with isk.txt's line `ranges`;
  // a test build can skip that.
  {
    const blk::Digest d = blk::configuration_digest(configuration_files());
    parameters_hash_words.assign(d.w, d.w + 8);
  }
  printf("# the parameter files' digest: hp=");
  for (const u32 w : parameters_hash_words) printf("%08x", w);
  printf("\n");
  // The forward kernels support the layers' scales only in certain ranges, which are checked here,
  // not where the scales are used, so that a value out of range ends the run at its start.
  {
    LayerShape s;
    s.prof = mo ? 3 : 2;
    s.xl = xl;
    digit_forms(s);
    for (int l = 0; l < Lr; l++)
      if (!layer_words_fit(layer_constants(l), l, s)) return 2;
  }
  io = std::vector<int>(NC);
  kap = std::vector<int>(NC);
  kk_ = std::vector<int>(NC);
  t0_ = std::vector<int>(NC);
  knd = std::vector<int>(NC);
  for (int c = 0; c < NC; c++) {
    io[c] = policy["iota_" + std::to_string(c)];
    // Every lookup call has 2^5 = 32 bits for each multiplicity. A lookup identity (a tree of the
    // lookup forest) has at most 2^identity_cap() <= 2^EIDB <= 2^28 keys (the check on lidb above),
    // but a multiplicity counts keys of a whole call, over all its identities. forest_plan()
    // (prover/hiding/forest_plan.h) rejects a group of calls with 2^32 keys or more, so no
    // multiplicity can overflow. With fewer bits, whether one overflows would depend on hidden
    // values. Nothing reads kap: ForestPlan::plane_bits() has the 5 built in.
    kap[c] = 5;
    // limit_missed() has rejected every kind except these four and FL, which becomes 4, and every
    // call whose name does not begin with its kind. kk_: bits of a key; t0_: the padding key.
    const std::string &k = policy.cl[c].kind;
    knd[c] = k == "R16" ? 0 : k == "R8" ? 1 : k == "FC" ? 2 : k == "FS" ? 3 : 4;
    kk_[c] = knd[c] == 1 ? 8 : 16;
    t0_[c] = knd[c] >= 2 ? 32768 : 0;
  }

  if (int rc = named_paths_open()) return rc;
  // Rejects a GPU with more than GPU_IN_USE_AT_THE_START bytes in use: another process is then
  // assumed to use it. Running out of GPU memory in the middle of a batch fails every request in
  // it, whereas a rejection here costs nothing: no key has been drawn yet. The check does not see a
  // process that comes later or that has not yet taken its memory, and it does not show that the
  // GPU has enough memory for the service. In a test run (-test-coins, which a production build
  // does not accept) it only warns, so that a small test can run beside another program.
  {
    size_t free_bytes = 0, total_bytes = 0;
    CK(cudaMemGetInfo(&free_bytes, &total_bytes));
    printf("# GPU memory: %.3f GB are in use at the start of this run, including its own CUDA context\n",
           (total_bytes - free_bytes) / 1e9);
    if (total_bytes - free_bytes > GPU_IN_USE_AT_THE_START) {
      const bool a_tests_run = !args.test_coins.empty();
      printf("%s (gpu-in-use): more than %zu GiB (%.3f GB) of this GPU's memory are in use, so something else is taken to be "
             "on it%s\n",
             a_tests_run ? "warning" : "refused", GPU_IN_USE_AT_THE_START >> 30, GPU_IN_USE_AT_THE_START / 1e9,
             a_tests_run ? SERVING_OR("", "; this is a test's run (-test-coins), which goes on")
                         : ": no key was drawn: give the service a GPU that nothing else uses");
      if (!a_tests_run) return refused("gpu-in-use");
    }
  }
  return 0;
}

// Checks that the directory given with -in or -spool can be opened, and that the spool directory is
// writable, so that a mistyped path costs no set-up. Nothing is read. read_isk() checks the secret
// key directory's files. Returns 0, or 2 after printing why.
inline int Prover::named_paths_open() {
  struct Named {
    const char *option;
    std::string path;
    bool folder;
    bool written = false; // only the spool directory: directories and files are created in it
  };
  std::vector<Named> named;
  if (args.one_batch())
    named.push_back({"-in", args.in, true});
  else
    named.push_back({"-spool", args.spool, true, true});
  for (const Named &n : named) {
    const int opened = ::open(n.path.c_str(), O_RDONLY | O_CLOEXEC | O_NOCTTY | (n.folder ? O_DIRECTORY : 0));
    if (opened < 0) {
      printf("refused (cp-spool): the path given with %s is not there or cannot be opened\n", n.option);
      return refused("cp-spool");
    }
    ::close(opened);
    // Creating a file in a directory needs write and search permission on it. access() checks for
    // the real user, who is also the writing user unless the program is installed set-user-ID or
    // set-group-ID. Nothing is created here, and a full disk or a quota is not detected.
    if (n.written && access(n.path.c_str(), W_OK | X_OK)) {
      printf("refused (spool-not-writable): the spool directory is not writable\n");
      return refused("spool-not-writable");
    }
  }
  return 0;
}

// Builds the public tables Ts (SiLU), Sg (sigmoid) and Te (attention's exponential), and the lookup
// tables tFC (clamp), tFL (SiLU) and, for a model with experts, tFS (sigmoid), whose entries pack a
// key and its value. build_header() (header.cuh) hashes Ts, Te and, with experts, FSw (tFS's
// values) into the statement's header.
// Source for non-linear functions by lookup tables in a proof of inference (ZKML): Chen, Waiwitlikhit, Stoica, Kang,
// "ZKML: An Optimizing System for ML Inference in Zero-Knowledge Proofs", EuroSys 2024, section 5.1; a table with the
// two columns x and f(x), there for ReLU: section 3.
// Source for SiLU and the softmax's exponential by tables: Yoon, Zhang, Liao, "Bits Under ZK-LLM: Evaluating
// Zero-Knowledge-Friendly Quantization for Verifiable Private LLM Inference", arXiv:2609.36437, section 4.
// Source for a public table of all input-output pairs: Hao, Chen, Li, Weng, Zhang, Yang, Zhang, "Scalable
// Zero-knowledge Proofs for Non-linear Functions in Machine Learning", USENIX Security 2024, section 3.5.
// Source for a table of arguments and a table of values: Sun, Li, Zhang, "zkLLM: Zero Knowledge Proofs for Large
// Language Models", arXiv:2404.16109 (ACM CCS 2024), section 4. There the two are joined by a random linear
// combination; here the pair is packed at a fixed radix.
inline int Prover::make_tables() {
  dev_info();
  // Creates this thread's cuBLAS handle (prover/common/blas.h) before the GPU's memory is planned,
  // because creating it allocates memory. It has no workspace yet: each user sets one before its
  // products.
  blas::blas(cudaStreamPerThread, nullptr, 0);
  grm::set_up(); // also creates the cuBLASLt handle (hiding/gram_products.cuh)
  g_st = cudaStreamPerThread;
  Ts = std::vector<short>(65536);
  for (int t = 0; t < 65536; t++) {
    const int top = 64 << gate_key_step, most = gate_values_most;
    int c = t - 32768;
    c = c < -top ? -top : c > top - 1 ? top - 1 : c;
    double z = c / (double)(1 << gate_key_step), s = z / (1 + exp(-z)) * 64;
    double r = s < 0 ? -floor(-s + 0.5) : floor(s + 0.5);
    Ts[t] = (short)(r < -most ? -most : r > most ? most : r);
  }
  CK(cudaMalloc(&Tsd, 65536 * 2));
  CK(cudaMemcpy(Tsd, Ts.data(), 65536 * 2, cudaMemcpyHostToDevice));
  tFC = std::vector<u64>(65536);
  tFL = std::vector<u64>(65536);
  for (int t = 0; t < 65536; t++) {
    int c = t - 32768;
    c = c < -2047 ? -2047 : c > 2047 ? 2047 : c;
    tFC[t] = (u64)t + ((u64)(c + 2048) << 28);
    tFL[t] = (u64)t + ((u64)(Ts[t] + 4096) << 28);
  }
  CK(cudaMalloc(&tFCd, 65536 * 8));
  CK(cudaMalloc(&tFLd, 65536 * 8));
  CK(cudaMemcpy(tFCd, tFC.data(), 65536 * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(tFLd, tFL.data(), 65536 * 8, cudaMemcpyHostToDevice));
  Sg = std::vector<int>(65536);
  for (int t = 0; t < 65536; t++) {
    Sg[t] = (int)(16 * floor(4096.0 / (1 + exp(-(t - 32768) / 256.0)) + 0.5));
  }
  Sigd = nullptr;
  tFSd = nullptr;
  tFS = std::vector<u64>(65536, 0);
  FSw = std::vector<int>(65536, 0);
  if (mo) {
    for (int t = 0; t < 65536; t++) {
      // With the line gcl, the table's first half is Sg[16384 .. 49151]. Entry t of the second half
      // is t - 49152 clamped to +-127, plus 128, placed at bit 28 with the flag 2^40 added.
      const int c = t - 49152;
      FSw[t] = !gate_clamped ? Sg[t] : t < 32768 ? Sg[t + 16384] : (c < -127 ? -127 : c > 127 ? 127 : c) + 128;
      tFS[t] = gate_clamped && t >= 32768 ? (u64)t + ((u64)FSw[t] << 28) + (1ull << 40) : (u64)t + ((u64)FSw[t] << 17);
    }
    CK(cudaMalloc(&Sigd, 65536 * 4));
    CK(cudaMemcpy(Sigd, Sg.data(), 65536 * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&tFSd, 65536 * 8));
    CK(cudaMemcpy(tFSd, tFS.data(), 65536 * 8, cudaMemcpyHostToDevice));
  }
  {
    int S_, wd_;
    if (!APlan::decodeS(policy["S"], S_, wd_)) {
      printf("policy: the value S = %d is not an allowed setting\n", (int)policy["S"]);
      return 2;
    }
    Te = APlan::expTable(S_, policy["rx"]);
  }
  return 0;
}

// Row length of weight kind k (KN, prover.h) in one-byte cells.
inline size_t Prover::kind_row_length(int k) {
  return (size_t)(k == 0 ? 64 : k == 1 ? dkv : k == 3 || k == 4 ? f : k >= 5 && k <= 8 ? fe : k == 11 ? NE : d);
}

// The smallest p for which the 2 * kind_cells(kd) bytes of weight kind kd split into p equal parts
// of whole rows, each of at most 5 * 2^29 bytes.
inline size_t Prover::parts_of_kind(int kd) {
  const size_t by = 2 * kind_cells(kd), rows = by / kind_row_length(kd);
  size_t p = 1;
  while (by / p > ((size_t)5 << 29) || rows % p) {
    p++;
  }
  return p;
}

inline bool Prover::has(const std::vector<std::string> &v, const std::string &s) {
  return std::find(v.begin(), v.end(), s) != v.end();
}

// Sets up the source of the weights and allocates what all sub-proofs share, including the residual
// stream between layer groups (XBv, xin), the message array (MSG), the request's tokens and range
// (tokd), the buffer for weight cells (`slab`) and the request's pad tree (CP).
inline int Prover::plan_memory() {
  nAF = policy["NAf"];
  nAD = policy["NAd"];
  iQA = nAF + nAD + 2;
  iXN = nAF + nAD + 3; // iXN, iXG: indices in Fork::M of borders g + 1 and g (see XBv below)
  iXG = iXN + 1;
  kinds = mo ? std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11} : (std::vector<int>{0, 1, 2, 3, 4, 9});
  // Called before the weights are read: in a test build, a value that it alters may be a weight.
  lie::from_the_environment(); // a test hook: empty in this repository (prover/common/lie.cuh)
  if (args.weights == "generator" && args.fold.empty()) {
    auto *w = new GeneratedWeights;
    w->seed = seed, w->Lam = Lam, w->d = d, w->H = H, w->G = Gk, w->dh = dh, w->f = f, w->fe = fe, w->NE = NE;
    w->V = (int)policy["V"], w->experts = mo;
    weights.reset(w);
  } else { // made-up-blocks, or a model's files (args.fold)
    block_weights = new wb::BlockWeights;
    block_weights->between_tensors = stop_if_asked;
    weights.reset(block_weights);
    wb::Shape &s = block_weights->shape;
    s.layers = Lr, s.Lam = Lam, s.d = d, s.H = H, s.G = Gk, s.dh = dh, s.f = f, s.fe = fe, s.NE = NE;
    s.V = (int)policy["V"], s.experts = mo;
    if (args.fold.empty()) {
      auto *source = new wb::MadeUpSource;
      block_source.reset(source);
      source->seed = seed, source->shape = s;
    } else {
      // A test build has no call here that reads a model's *.gguf files (prover/common/setting.h).
#if BUILT_FOR_TESTS_ALONE // `read_isk` rejects this case, but by `cp::made_up`, not by args.fold
      printf("internal error (tests-build-fence): a build for tests alone has come to a model's files, which it refuses long before: nothing of them is opened. The program has a fault; nothing of this run may be used\n");
      stop(End::Refused);
#else
      auto *source = new wb::GgufSource;
      block_source.reset(source);
      source->shape = s;
      source->open(args.fold, args.shards);
#endif
    }
    const double t0 = now();
    if (args.stream_weights) {
      fork_memory.purpose = "weights kept on the GPU";
      if ((args.one_room_for_experts || args.experts_by_layer) && !mo) {
        printf("-memory: one-room-for-experts, experts-by-layer and tensor-rooms require a model with routed "
               "experts\n");
        return 2;
      }
      weight_stream.one_room_for_experts = args.one_room_for_experts || args.experts_by_layer;
      weight_stream.experts_by_layer = args.experts_by_layer;
      weight_stream.experts_rooms = args.tensor_rooms ? args.tensor_rooms : args.experts_rooms;
      weight_stream.experts_by_tensor = args.tensor_rooms > 0;
      weight_stream.routed_layers_a_group = Lam / 2, weight_stream.huge_pages = args.huge_pages;
      if (args.tensor_rooms && args.tensor_rooms < 3 * (Lam / 2)) {
        printf("-memory: tensor-rooms=%d is too small: a layer group of this model has %d routed tensors; give at "
               "least that many\n",
               args.tensor_rooms, 3 * (Lam / 2));
        return 2;
      }
      weight_stream.one_room_for_the_rest = weight_stream.ends_by_turns = args.rest_by_turns;
      block_weights->fill(*block_source, weight_stream);
      weight_stream.start();
      weights_ready = [this](int g) {
        weight_stream.ready(g);
      };
      experts_done = [this](int g) {
        weight_stream.experts_done(g);
      };
      weights_done = [this](int g) {
        weight_stream.done(g);
      };
      weights_forget = [this] {
        weight_stream.restart();
      };
      printf("# the layer groups' weights are kept in host memory, %.3f GB in all, and loaded group by group; the "
             "largest group has %.3f GB\n",
             weight_stream.total_bytes() / 1e9, weight_stream.largest_group() / 1e9);
    } else {
      wb::Placement own;
      fork_memory.purpose = "weights kept on the GPU";
      block_weights->fill(*block_source, own);
    }
    fork_memory.purpose = "other buffers of the prover";
    weights_read_seconds = now() - t0;
    printf("# weights read as blocks in %.1f s\n", weights_read_seconds);
  }
  FK = std::vector<Fork>(NGr);
  XBv = std::vector<i8 *>(NGr + 1);
  mx = (size_t)1 << policy["lmx"];
  kx = ((size_t)xl * d * NT + mx - 1) / mx;
  // Layer sub-proof g reads border g, the residual stream before group g, and writes border g + 1;
  // the head reads the last border. Nothing reads border g once layer sub-proof g (for the last
  // border, the head) has run, so two buffers alternate for all borders, as for xin below.
  for (size_t g = 0; g < XBv.size(); g++) {
    if (g >= 2) {
      XBv[g] = XBv[g - 2];
      continue;
    }
    CK(cudaMalloc(&XBv[g], kx * mx));
    CK(cudaMemset(XBv[g], 0, kx * mx));
  }

  MSG = nullptr;
  NSM = PAD_STRIDE;
  {
    CK(cudaMalloc(&MSG, (3 + SLOT_COUNT) * NSM * sizeof(T)));
    CK(cudaMemset(MSG, 0, (3 + SLOT_COUNT) * NSM * sizeof(T)));
    // The only write of PADOFF in cp_prove: in every message array, the mask entry (`pad`) of
    // message slot i is PADOFF elements after the slot (prover/hiding/sumcheck_door.cuh).
    const long long po = (long long)NSM;
    CK(cudaMemcpyToSymbol(PADOFF, &po, 8));
    sc_door::cheat_from_the_environment(); // a test hook: empty in this repository
  }
  Vv = (int)policy["V"];
  tokd = nullptr;
  embC = nullptr;
  npE = 1;
  {
    std::vector<int> tk(NT + 2); // tokens, then r, e (range r to e - 1): here the last position
    tk[NT] = NT - 1, tk[NT + 1] = NT;
    for (int t = 0; t < NT; t++) {
      // Placeholder tokens; begin_request() (request.cuh) writes a request's words over them.
      tk[t] = (int)(mix64(seed + ((u64)3 << 40) + t) % (u64)Vv);
    }
    CK(cudaMalloc(&tokd, tk.size() * 4));
    CK(cudaMemcpy(tokd, tk.data(), tk.size() * 4, cudaMemcpyHostToDevice));
    const size_t by = (size_t)4 * Vv * d;
    if (weights->embedding_in_place()) {
      embC = (i8 *)weights->embedding_in_place();
    } else {
      while (by / npE > ((size_t)5 << 29) || ((size_t)4 * Vv) % npE) {
        npE++;
      }
      CK(cudaMalloc(&embC, by / npE));
    }
  }
  merge_mask = nullptr;
  slab = nullptr;
  {
    CK(cudaMalloc(&merge_mask, sizeof(T)));
    size_t slabB = 0;
    for (int kd : kinds) {
      if (!weights->lays_out_cells(kd)) continue;
      slabB = std::max(slabB, 2 * kind_cells(kd) / parts_of_kind(kd));
    }
    CK(cudaMalloc(&slab, slabB));
  }
  fork_memory.init(2 * NGr + 2);
  rootCP = nullptr;
  CPh = nullptr;
  {
    const int NFW = NGr + 1;
    fork_memory.also_list = &pad_tree_buffers;
    // CP has CP_ROWS rows, each with a random tail for CP_LISTS lists of positions
    // (prover/common/pad_tree_rows.h). No code draws, commits to or opens its rows: outside this
    // function only its width K and its stream number are read (prover/head/hidden_fork.cuh,
    // prover/prover/stream_plan.cuh), and plan_member_state() (prover/prover/in_step.cuh) lists its
    // buffers in a request's state between layer groups.
    CP.setup("CP", 2, 0, NFW, CP_LISTS, CP_ROWS, hz::ATTEMPT_STREAMS_FROM + 8u, s8::hb_r());
    CP.tnum = TREE_CP;
    fork_memory.also_list = nullptr;
    CK(cudaMalloc(&rootCP, 4 * DW));
    CK(cudaMalloc(&CPh, CP.sh.Kc * sizeof(T)));
  }
  rscB = 0;
  thW = 0;

  ai = 0;
  xin = std::vector<ResidualInt *>(NGr + 1);
  for (size_t g = 0; g < xin.size(); g++) {
    if (g >= 2) {
      xin[g] = xin[g - 2];
      continue;
    }
    // The sub-proofs' other buffers are allocated in their set-ups through fork_alloc(). A buffer's
    // Place decides which sub-proofs share it and how long its content is valid (fork_memory.h).
    CK(cudaMalloc(&xin[g], (size_t)d * NT * sizeof(ResidualInt)));
  }
  return 0;
}
