// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Command line of the prover program cp_prove: option values and defaults, usage text, parser.
// read_isk() (isk.cuh) sets the fields that come from isk, the secret key directory that cp_commit
// writes, and checks options against it. A leaf table ("leaf"): one of a commitment's weight
// tables, each with its own masks. SERVING_OR(a, b): a in a production ("serving") build, else b.
#pragma once
#include "prover/closed_front.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

struct Arguments {
  std::string isk;     // the value of -isk; cp_prove writes only its leaf/<k>/count.bin and audit/
  std::string in, out; // one batch: the directory of request files; the new directory for responses
  std::string spool;   // a service: the spool directory, for requests and responses (spool.h)
  bool one_batch() const { return spool.empty(); }
  // Set by read_isk(): isk/config/policy.txt, KL.bin (the scales). layout.txt is also read there.
  std::string policy, layer_words;
  int batch = 1;   // largest size of a batch, whose requests share one opening; with -in: its size
  int in_step = 1; // how many of them share a pass over the weights, by layer group (2^`ll` layers)
  bool measure = false;        // print lines that begin with "measure:", with each batch's times
  // One-off (not upstream): batches the several small device->host syncs that close_by_frame()
  // (prover/hiding/door_chain.cuh) makes per sub-proof into one, instead of one sync per fetch.
  bool batch_fetch = false;
  // One-off (not upstream): threads CardFold::bind()'s host-side fold loop (prover/engine/fold.cuh).
  bool parallel_fold = false;
  // One-off (not upstream): gives each in_step slot (chain_host.cuh) its own CUDA stream instead
  // of sharing cudaStreamPerThread, letting a batch's cryptographically-independent requests'
  // kernels run concurrently instead of fully serialized through one stream.
  bool parallel_streams = false;
  // One-off (not upstream): double-buffers CardTiled's planes_ work space (engine/tiled.cuh).
  bool double_buffer_planes = false;
  bool end_when_empty = false; // end, as with the stop file, when a batch is due and no request is waiting
  bool keep_for_the_audit = false; // with -in: keep the batch's 32 system bytes in isk/audit/
  long long most_batches = 0;  // end, as with the stop file, after this many batches; 0: no limit
  // If the next batch is ready when a batch's passes end, the batch is set aside: each pass
  // ("walk") of its opening of the weight table uses what a later batch's pass loads, or runs alone
  // (service.cuh, prove_the_batch()). Each batch draws its own randomness (`coins`) and leaf table.
  bool overlap = false;
  // Sets `overlap`, with up to two batches set aside; one pass then serves one walk of each. Set-up
  // makes one opening more than batches can be set aside (opening_in_process.cuh, setup_opening()).
  bool overlap_two = false;
  // Set by read_isk() from isk.txt's line "weights": "generator" (the line says "generated") or
  // "made-up-blocks" for synthetic weights, the latter stored like a model's files; or "files".
  std::string weights;
  std::string fold; // "files": the directory with wsums.txt, fold.txt; `shards`: its *.gguf files
  std::vector<std::string> shards;
  // Set by read_isk() unless "generator": weights stay on the host, sent to the GPU by layer group.
  bool stream_weights = false;
  // Test options; a production build has none, and the fields keep their defaults. test_coins: a
  // file whose bytes 32 b .. 32 b + 31 replace the 32 system bytes (rho) of batch b; the service
  // ends when they are used up. The next two need test_coins. test_leaf: every batch uses this leaf
  // table; -1: the batch's first key, derived from the seed, rho and a digest, selects a live one.
  // test_no_requests: the requests are left out of that digest, so batches with the same rho have
  // the same coins: to test what a request changes in other slots' files. read_isk() accepts
  // test_coins only if isk.txt says "seed test", a line that no key protects.
  std::string test_coins;
  long long test_leaf = -1;
  bool test_no_requests = false;
  // -public-weights (prover/common/public_weights.h; production builds only): the 32-byte file from
  // which cp_commit derived the seed, so nothing is hidden. read_isk() accepts the option and
  // isk.txt's line "seed public" only together; read_the_seed() checks seed.bin against that seed.
  std::string public_weights;

  // The keywords of -memory: the fields down to close_on_the_slab, typed with - for _ and = before
  // a number (tensor-rooms=10). All but the last can save GPU memory, most at a cost in time. They
  // must change no proof byte. memory_word() checks ranges; parse() rejects some combinations.
  // one_room_for_experts: the GPU holds the routed experts of one layer group at a time, not two.
  bool one_room_for_experts = false;
  // The routed experts are loaded one layer at a time into 'experts_rooms' GPU buffers (1 to 4),
  // each for one layer's routed experts, however many layers a layer group has. A group with more
  // such layers than buffers loads them again in every request's turn (its work on a layer group).
  bool experts_by_layer = false;
  int experts_rooms = 1;
  // Routed experts are loaded tensor by tensor (three per layer) into this many GPU buffers, 3 to
  // 16, at least a layer group's routed tensors. 0: off. A finer experts_by_layer, which it sets.
  int tensor_rooms = 0;
  // With experts_by_layer: a layer group's other weights have one GPU buffer instead of two,
  // and the embedding and the head are loaded into the experts' buffers when they are used.
  bool rest_by_turns = false;
  // A sub-proof (`fork` in identifiers) covers one part of the forward pass. How many of the lowest
  // levels of its lookups' product trees (prover/hiding/forest.cuh) are recomputed from the lookup
  // keys, not stored, 2 to 5: more levels take less memory, more time. tail_rebuilt: the two range
  // lookups that end every sub-proof (prover/hiding/tail_f.cuh). The rest: each type's own lookups.
  int layer_rebuilt = 3, attention_rebuilt = 2, head_rebuilt = 4, tail_rebuilt = 3;
  // A layer sub-proof's Place::Late buffer (fork_memory.h) that does not fit in the scratch memory
  // of the sub-proof's first half shares addresses with the forward pass's lookup keys.
  bool late_on_keys = false;
  // In GB (10^9 bytes), 0 to 80; -1 (not given) or 0: off. The slab is the sub-proofs' shared GPU
  // scratch memory. Above 0: the attention and head sub-proofs' buffers begin that far after its
  // start, or at ForkMemory::arena_first if that comes first (service.cuh, run()). Nothing in this
  // tree uses the space that they leave: ForkMemory::lend(), which can lend it, has no caller.
  double close_on_the_slab = -1;
  // -huge-pages: ask Linux for huge pages (madvise(), MADV_HUGEPAGE) for the host copy of the
  // weights (pinned_host_memory(), prover/prover/weight_stream.cuh). Off by default: where
  // /sys/kernel/mm/transparent_hugepage/defrag has its default, `madvise`, a region with that
  // request "will enter direct reclaim like always" (`always` is another value of that file;
  // https://docs.kernel.org/admin-guide/mm/transhuge.html), so on a machine short of free 2 MiB
  // blocks, filling the memory can take far longer. Without the option, the file `enabled` beside
  // `defrag` decides whether the memory gets huge pages. No byte of any file depends on the option.
  bool huge_pages = false;

  // A name for a list of -memory keywords.
  struct Preset {
    const char *name, *words;
  };
  static const std::vector<Preset> &presets() {
    static const std::vector<Preset> list = {
        // For a full-size model.
        {"lean", "late-on-keys,tensor-rooms=10"},
    };
    return list;
  }

  static void usage() {
    printf(
        SERVING_OR("This is a serving build. Until it reads its secrets it says what it does and, where it refuses, why. From the line \"p258 closed\" on it prints a fixed list of lines and none of a batch's "
                   "internal times; its last line is \"p258 end <status>\", and behind a status other than 0 the name of the refusal, the self-check or the machine's fault that ended it, if there is one "
                   "(prover/docs/ends-in-short.md lists the names). These lines are for the operator: hand none of them on to a client. It reads no test option and no test or tuning variable. "
                   "It starts itself a second time with a reduced environment: of its caller's variables it keeps HOME, TMPDIR, CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER (as root it reads SUDO_UID once, for the paths named with -spool, -isk, -out and -clear, and then drops it too). HOME AND TMPDIR MUST BE SET, each to a "
                   "folder that exists, by an absolute path of at most 400 characters out of letters, digits and / . _ -\n",
                   "")
        "usage: cp_prove -isk <folder> -in <folder of request files> -out <new folder> [options]\n"
        "       cp_prove -isk <folder> -spool <folder> [options]\n"
        "       cp_prove -clear <folder>\n"
        "The first form proves one batch and ends. The second serves batch after batch until it is told to stop. The third removes "
        "responses that were never released, and ends. For each "
        "batch the program draws 32 fresh bytes, and from them, the commitment's seed and the batch's requests come one of the "
        "commitment's live leaves and all coins of the batch. Without -keep-for-the-audit it writes one thing to isk: before it uses the leaf that a batch has drawn, "
        "it appends to that leaf's count.bin one byte for each column of the stored table that the batch's opening shows; the file's length is the count, and the program takes no further batch when a live "
        "leaf's count leaves no room for one more batch under the number of columns that its masks bear (cp_commit prints the number). It counts in this folder only: never serve one leaf from two "
        "folders. Otherwise it keeps nothing from one batch to the next. "
        "Masking is tested; that it hides the weights is not proved; what was run: docs/how-hiding-is-checked.md.\n"
        "  -isk <folder>     the folder that cp_commit wrote: seed.bin, isk.txt, config/ and leaf/<k>/ for every live leaf k "
        "(isk.txt's first, or 0, up to M - 1). If such a leaf's folder is missing, or its count.bin cannot be written, the program does not start. isk.txt is read once, at the "
        "start\n"
        "  -in <folder>      it holds nothing but files <name>.req, <name> of 1 to 64 characters out of A-Z a-z 0-9 _ -; they are "
        "taken in the bytewise order of their names. Each is a request: T tokens and the two words r and e (the proof "
        "covers positions r to e - 1) as 32-bit little-endian words. At most as many requests as the parameter file's line slots says, "
        "and at most 64\n"
        "  -out <folder>     must not exist; or it holds nothing but .tmp, with a mark.txt in it, left by a batch that was killed before it released anything: it is then taken, and what is staged there is removed first. Name it by its full path from /: a path is refused as for -clear. It gets <name>.y (the answer, one word) and <name>.proof for each request "
        "that has a proof, and the one empty file <name>.failed for one that has none. While the batch runs, and after one that was killed, crashed or could not remove what it had staged, .tmp in it holds mark.txt and what is staged; after any other failed batch .tmp is empty: name a new folder then\n"
        "  -spool <folder>   an existing folder of yours with mode 0700 (mkdir -m 700). It is the service's alone: keep nothing of your own in it. It must be empty or hold a spool directory's mark.txt, a small file that cp_prove writes into an empty folder: on any other folder cp_prove does not start, and touches nothing that is there. "
        "A folder counts as empty if it holds nothing but what a client may put there before the first start: in/ with waiting requests, an empty out/, an empty file stop (a start removes stop). "
        "cp_prove does not check which commitment a folder was used for: use one folder for one commitment, or clients' checks of the proofs fail. A request is the file in/<id>.req: the request's words as above, followed "
        "by 32 random bytes. <id> is the hash of the file's bytes (prover/prover/spool.h; driver/spool.py lay submits a "
        "request). The answer is out/<id>.y and then out/<id>.proof, or the empty out/<id>.failed. Before the last of them comes "
        "out/<id>.tag, for the front end alone (prover/prover/spool.h). The empty file stop ends the process after the current batch\n"
        "  -batch B          with -spool: the largest number of requests in a batch: 1 to 64, and at most the parameter file's line slots; "
        "default 1\n"
        "  -in-step k        how many requests of a batch are proved together, layer group by layer group, in one pass "
        "over the weights: 1 to 16, and at most the batch; default 1\n"
        "  Timing is not hidden: a batch's time is visible and, through it, the number of requests in the batch, "
        "something about their routing, and whether one of them failed\n"
        "  -memory <words>   other layouts of GPU memory; the proofs stay the same. <words> is lean, the list of the full-size runs, or a comma-separated list of keywords, in which lean may appear; if a "
        "keyword occurs twice, the later occurrence counts (the keywords: prover/prover/arguments.h, \"The keywords of -memory\")\n"
        "  -huge-pages       ask Linux for huge pages for the host copy of the weights (generated weights have no host copy). Without it the machine's own setting (/sys/kernel/mm/transparent_hugepage/enabled) decides. Off by default, "
        "because on a machine that is short of free 2 MiB blocks the request can make the start take far longer. The proofs stay the same\n"
        SERVING_OR("", "  -measure          print lines that begin with \"measure:\" and contain each batch's times\n")
        "  cp_prove does not check the proofs that it writes: each response gets its final name as soon as its batch is written. After a fault of the machine or the program a proof can be false and can hold values that are not masked: run cp_verify on each response before you hand it to anyone\n"
        "  Exit status. 0: the program was told to end, and every response that it owed has its final name. 9: one of "
        "the program's self-checks on secret random values or keys has failed (a name that begins own-test-): the machine or the program may lose random values, and a proof made with a lost one verifies and hides less. One of these names, own-test-coin-library, can also mean, at a start and with a line that says so, that the operating system gave no memory page that a fork wipes (it needs Linux 4.14 or later): then no self-check has failed. "
        "Nothing that has no final name at that moment leaves, with three exceptions (prover/docs/ends-in-short.md, \"The three exit statuses\"), and the next start answers those requests failed. Find out why before you start the program again, on this machine above all; do not let a supervisor restart it "
        "after any exit but 0, or when it was killed. 1: every other end: a wrong call, a refusal, a fault of the machine, a signal; a few self-checks on secret random values or keys end with 1 too, not with 9: look a name up (prover/docs/ends-in-short.md, \"After status 1\"), and treat a last line \"p258 end 1\" with no name as a 9 until you know its cause\n"
        "  -keep-for-the-audit  with -in and -out, on made-up weights: keep the 32 bytes that the batch takes from the operating system's random generator in <isk>/audit/batch.fresh, for one audit with the key "
        "(tools/audit/audit_batch.sh). The audit computes the proofs' random values again and compares them; it erases the file only in a call whose last word is --erase-kept-files, and not always then: count the file as gone only if the audit's last line says so. cp_prove never erases it. Together with seed.bin the file "
        "opens every mask of that batch's proofs: until it is erased, hiding does not hold for that batch. Hand out none of the batch's responses unless the audit has said YES; after a NO or a NOT WHOLE no part of the commitment that the batch was proved under may ever leave the machine (docs/operators-duties.md, \"The audit with the key\"). While anything is in <isk>/audit/, every start on that folder is refused: audit first, then serve. "
        "The option changes no byte of a proof. Refused with -spool and on a model's files\n"
        "  -end-when-empty   with -spool: exit, as for the stop file, when a batch is due and no request is waiting\n"
        "  -most-batches n   with -spool: exit, as for the stop file, after n batches\n"
        "  -overlap          with -spool: a batch's opening reads the weights on the next batch's passes, one of its walks on "
        "each, if that batch stands ready; it takes GPU memory for a second keeper of the weight table and a second engine, and wants "
        "-in-step of at least 2, also with -batch 1, and -memory with tensor-rooms (as in lean). Each batch has its own coins and its own leaf. It is refused on a model's files (overlap-on-a-model): it runs on "
        "made-up weights alone\n"
        "  -overlap-two      like -overlap (which it turns on), but the openings of two batches can be in progress at once, so one "
        "pass over the weights serves one walk of each. It helps only if a batch fits in one pass (-batch at most -in-step). A "
        "response comes when the passes of the next two batches are nearly done. It takes GPU memory for a third keeper and a "
        "third engine\n"
        SERVING_OR(
        "  -public-weights <file>  serve a commitment that cp_commit created with -public-weights and the same file of 32 bytes (isk.txt has the line \"seed public\"). The seed is derived from that file, so "
        "nothing is hidden from anyone who has it. With this option the weights are not hidden, and nothing has been argued on whether the requests of one batch stay hidden from one another; serve other people's requests under it one request a batch, or not at all. "
        "Such a commitment is served only with this option, and the option is accepted for no other commitment. Never use it for weights that must stay hidden\n", "")
#ifndef SERVING_BUILD
        "  -test-coins <file>  for tests: the 32 bytes of batch b are bytes 32 b .. 32 b + 31 of the file, and the program "
        "ends when they are used up. Nothing is then hidden from whoever has the file\n"
        "  -test-leaf <k>    with -test-coins alone: every batch is held on leaf k, a live one\n"
        "  -test-no-requests with -test-coins alone: a batch's coins do not depend on its requests' words\n"
        "  -test-coins is taken only where isk.txt says \"seed test\", as it does on a commitment made with "
        "-test-seed. No key protects that line, and whoever can write isk.txt can add it: never add it to a commitment that was made without -test-seed. Where the line stands on a folder of weights "
        "the program does not start: on a model's files no option for tests is taken\n"
#endif
        "  -clear <folder>   as the only two arguments: remove the responses that another run of cp_prove wrote to out/.tmp (-spool) or .tmp (-out) in "
        "this folder and never released: give them to no one. Run this on a folder in which a cp_prove "
        "has worked and ended with a status other than 0, or was cut off by a kill or a loss of power. Name no other folder, and name it by its "
        "full path from /: a path is refused if it does not begin with /, or if a part of it is a link or lies in a folder that an account other than root and you can write (a "
        "folder with the sticky bit, as /tmp, passes if the part in it is root's or yours). out/.tmp is cleared only if <folder>/mark.txt says that <folder> is a spool directory, .tmp only if .tmp/mark.txt says that "
        "cp_prove -out made it. If either holds something and has no such file, or holds a name at its top that cp_prove never uses there, nothing is touched, and the printed lines say why. "
        "Otherwise the call knows what cp_prove staged by names alone. One level below out/.tmp or .tmp, the staging "
        "folder, entries named y, proof or failed go (files are overwritten with zeros first), with the folders that are then empty or were "
        "empty. Every other entry is left and listed. Status 0: none is left; 1: a "
        "wrong call, a folder that is refused, or something that is left or could not be examined: the printed lines explain what to do; 2: nothing "
        "was touched and nothing is known to be wrong: a cp_prove is running there, or there is no such folder, or it has no out/.tmp and no .tmp\n"
        "  --setting         as the only argument: print the compile-time constants (prover/common/setting.h) and exit\n"
        "  -h, --help        as the only argument: print this text and exit\n");
  }
  // Returns the number in 'v' (digits with at most one point), or -2, which no option accepts.
  static double number(const char *v) {
    return v[0] && strspn(v, "0123456789.") == strlen(v) && strchr(v, '.') == strrchr(v, '.') && strcmp(v, ".")
               ? atof(v)
               : -2;
  }
  // Returns the integer that 'v' contains, in at most nine digits, or -1.
  static long long whole(const char *v) {
    return v[0] && strlen(v) <= 9 && strspn(v, "0123456789") == strlen(v) ? atoll(v) : -1;
  }
  // fault() prints usage() and the fault and returns false; in isk_on_the_line() it prints nothing.
  static inline bool a_look_ahead = false;
  static bool fault(const std::string &what) {
    if (a_look_ahead) return false;
    usage();
    printf("cp_prove: %s\n", what.c_str());
    name_the_end("usage");
    return false;
  }
  // Applies one -memory keyword, "name" or "name=value"; false if unknown or value out of range.
  bool memory_word(const std::string &word) {
    const size_t equals = word.find('=');
    const std::string name = word.substr(0, equals);
    if (equals == std::string::npos) {
      bool *const flag = name == "one-room-for-experts"   ? &one_room_for_experts
                         : name == "experts-by-layer"     ? &experts_by_layer
                         : name == "rest-by-turns"        ? &rest_by_turns
                         : name == "late-on-keys"         ? &late_on_keys
                                                          : nullptr;
      if (flag) *flag = true;
      return flag != nullptr;
    }
    const char *const value = word.c_str() + equals + 1;
    if (name == "experts-rooms") return (experts_rooms = (int)whole(value)) >= 1 && experts_rooms <= 4;
    if (name == "tensor-rooms") return !(tensor_rooms = (int)whole(value)) || (tensor_rooms >= 3 && tensor_rooms <= 16);
    int *const levels = name == "layer-rebuilt"       ? &layer_rebuilt
                        : name == "attention-rebuilt" ? &attention_rebuilt
                        : name == "head-rebuilt"      ? &head_rebuilt
                        : name == "tail-rebuilt"      ? &tail_rebuilt
                                                      : nullptr;
    if (levels) return (*levels = (int)whole(value)) >= 2 && *levels <= 5;
    if (name != "close-on-the-slab") return false;
    close_on_the_slab = number(value);
    return close_on_the_slab >= 0 && close_on_the_slab <= 80;
  }
  // The value of -isk, or "" if parse() rejects the command line. Prints and creates nothing. For
  // cp_prove.cu, which reads isk.txt before the real parse. Calls parse(), so there is one parser.
  static std::string isk_on_the_line(int argc, char **argv) {
    Arguments scratch;
    a_look_ahead = true;
    const bool good = scratch.parse(argc, argv);
    a_look_ahead = false;
    return good ? scratch.isk : std::string();
  }
  // Reads the command line into the fields and checks ranges and combinations; false after fault().
  // The argument after an option that takes a value is that value, even if it is itself an option.
  // Paths are stored as given. main() (cp_prove.cu) handles -clear, --setting, -h and --help.
  bool parse(int argc, char **argv) {
    std::string memory;
    bool batch_given = false;
    for (int i = 1; i < argc; i++) {
      const std::string a = argv[i];
      bool *const flag = a == "-measure"          ? &measure
                         : a == "-batch-fetch"    ? &batch_fetch
                         : a == "-parallel-fold"  ? &parallel_fold
                         : a == "-parallel-streams" ? &parallel_streams
                         : a == "-double-buffer-planes" ? &double_buffer_planes
                         : a == "-end-when-empty" ? &end_when_empty
                         : a == "-keep-for-the-audit" ? &keep_for_the_audit
                         : a == "-overlap"        ? &overlap
                         : a == "-overlap-two"    ? &overlap_two
                         : a == "-huge-pages"     ? &huge_pages
                         SERVING_OR(, : a == "-test-no-requests" ? &test_no_requests)
                                                  : nullptr;
      if (flag) {
        *flag = true;
        continue;
      }
      std::string *const text = a == "-isk"          ? &isk
                                : a == "-in"         ? &in
                                : a == "-out"        ? &out
                                : a == "-spool"      ? &spool
                                : a == "-memory"     ? &memory
                                SERVING_OR(: a == "-public-weights" ? &public_weights, )
                                SERVING_OR(, : a == "-test-coins" ? &test_coins)
                                                     : nullptr;
      if (!text && a != "-batch" && a != "-in-step" && a != "-most-batches"
          SERVING_OR(, && a != "-test-leaf"))
        return fault("unknown option " + a);
      if (i + 1 >= argc) return fault(a + " needs a value");
      const char *v = argv[++i];
      if (text == &public_weights && !*v) return fault("-public-weights takes the name of a file");
      if (text == &public_weights && !public_weights.empty()) return fault("-public-weights is given twice");
      if (text)
        *text = v;
      else if (a == "-batch")
        batch = (int)whole(v), batch_given = true;
      else if (a == "-in-step")
        in_step = (int)whole(v);
#ifndef SERVING_BUILD
      else if (a == "-test-leaf")
        test_leaf = whole(v) >= 0 ? whole(v) : -2;
#endif
      else
        most_batches = whole(v) > 0 ? whole(v) : -1;
    }
    if (overlap_two) overlap = true;
    if (isk.empty()) return fault("-isk is required");
    if (spool.empty() ? in.empty() || out.empty() : !in.empty() || !out.empty())
      return fault("give either -in and -out (one batch) or -spool (a service)");
    if (one_batch() && (batch_given || end_when_empty || most_batches || overlap))
      return fault("-batch, -end-when-empty, -most-batches, -overlap and -overlap-two go with -spool: with -in the batch is the "
                   "folder's requests");
    if (batch < 1 || batch > 64) return fault("-batch takes 1 to 64");
    // With -overlap even a batch of one request needs two state buffers: see the last check below.
    if (in_step < 1 || in_step > 16 || (!one_batch() && in_step > std::max(batch, overlap ? 2 : 1)))
      return fault("-in-step takes 1 to 16 and at most -batch (with -overlap or -overlap-two and -batch 1: 2)");
    if (most_batches < 0) return fault("-most-batches takes a number of at least 1 with at most nine digits");
#ifndef SERVING_BUILD
    if (test_leaf == -2) return fault("-test-leaf takes a leaf's number");
    if (test_leaf >= 0 && test_coins.empty()) return fault("-test-leaf goes with -test-coins alone");
    if (test_no_requests && test_coins.empty()) return fault("-test-no-requests goes with -test-coins alone");
#endif
#ifdef SERVING_BUILD
    static_assert(!CLOSED_KEEPS_THE_TEST_WORD, "SERVING_BUILD and CLOSED_KEEPS_THE_TEST_WORD cannot both be set");
    // The times depend on the routing of every request's tokens.
    if (measure)
      return fault(
          "refused (serving-build): this program prints none of a batch's internal times: it does not take -measure");
#endif
#if CLOSED_KEEPS_THE_TEST_WORD
    if (test_coins.empty()) return fault("this is a test build: it requires -test-coins");
#endif

    // -memory. A preset's name in the list stands for its keywords. A keyword's last value counts.
    auto each_word = [](const std::string &list, auto &&take) {
      for (size_t from = 0; from < list.size();) {
        const size_t comma = std::min(list.find(',', from), list.size());
        if (!take(list.substr(from, comma - from))) return false;
        from = comma + 1;
      }
      return true;
    };
    std::string refused;
    auto one_word = [&](const std::string &word) {
      return memory_word(word) || (refused = word, false);
    };
    if (!each_word(memory, [&](const std::string &word) {
          for (const Preset &p : presets())
            if (word == p.name) return each_word(p.words, one_word);
          return one_word(word);
        }))
      return fault("-memory: \"" + refused + "\" is not a word that -memory accepts, or its value is out of range");
    if (late_on_keys && close_on_the_slab > 0)
      return fault("-memory: late-on-keys cannot be combined with close-on-the-slab above 0");
    if (tensor_rooms && (one_room_for_experts || experts_by_layer || experts_rooms > 1 || rest_by_turns))
      return fault("-memory: tensor-rooms cannot be combined with one-room-for-experts, experts-by-layer, "
                   "experts-rooms or rest-by-turns");
    if (tensor_rooms) experts_by_layer = true;
    if (experts_by_layer && one_room_for_experts)
      return fault("-memory: experts-by-layer cannot be combined with one-room-for-experts");
    if (experts_rooms > 1 && !experts_by_layer) return fault("-memory: experts-rooms needs experts-by-layer");
    if (rest_by_turns && !experts_by_layer) return fault("-memory: rest-by-turns needs experts-by-layer");
    // In a pass whose weights an opening also uses, each routed tensor must keep its buffer until
    // the last turn on its layer group, and the embedding and the head must stay on the GPU. The
    // opening decodes weights into the start of the sub-proofs' scratch memory between turns. Only
    // with 2 or more state buffers (in_step) do seat_turn() and seat_ends() (in_step.cuh) zero that
    // start before a turn; wave_has_ended() exits if the decoded weights could reach further.
    if (overlap && (!tensor_rooms || in_step < 2))
      return fault("-overlap and -overlap-two require -memory with tensor-rooms (as in lean) and -in-step of at least 2");
    return true;
  }
};
