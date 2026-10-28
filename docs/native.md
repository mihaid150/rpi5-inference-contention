# Native workers

The release includes the C++ implementation actually used by the Qwen study,
with comments removed and formatting normalized. It does not replace these
workers with a Python simulation.

| Source | Function |
|---|---|
| `route_worker.cpp` | Binary stage protocol, KV export/import, precision transition and memory telemetry |
| `mapped_shard.hpp` | Mapped weight lifetime and explicit release |
| `llama_cpu_executor.cpp` | Partial Qwen layer execution, attention and KV state |
| `qwen_math.hpp`, `qwen_trace.hpp` | Qwen mathematical helpers and optional tracing |
| `legacy_decode.cpp` | Restricted microbatch native decode reference |
| `attention_reference.cpp` | Frozen reference channel for the later prefill qualification |
| `prefill_worker.cpp` | Prompt boundary, batching/repacking, attention profiling and omission |

`qwen-release-worker` and `qwen-route-worker` are built from the same source.
The release build enables destruction and unmapping before target weights are
loaded. The control keeps the alternate mapping available. The integration
test compares their continuation and checks the actual process mappings.

## Build ARM64 binaries on either x86 or ARM64

The following image cross-compiles with the AArch64 GNU toolchain. It executes
only build-host programs, so ARM emulation is not needed:

```bash
docker build --platform linux/arm64 -f deploy/Dockerfile.native -t shared-budget-native:local .
```

It builds unmodified llama.cpp at the frozen revision for the standard and
partitioned workers, then a separate private checkout with the fixed attention
graph patch. The original checkout is not modified. Libraries for the private
and standard runtimes remain separate.

To export compiled artifacts without executing them:

```bash
docker create --name paper-native-export shared-budget-native:local /unused
docker cp paper-native-export:/native workspace/native
docker cp paper-native-export:/poc workspace/poc
docker cp paper-native-export:/reference workspace/reference
docker cp paper-native-export:/standard workspace/standard
docker rm paper-native-export
```

Create `workspace/` first. These outputs are ignored by Git. Executing the
exported ARM64 binaries requires an ARM64 Linux host.

## Build and test on the current host

A native build can use an existing pinned llama.cpp checkout and its built
shared libraries. This is software qualification on that host, not evidence
of ARM performance:

```bash
cmake -S native -B workspace/native-build -DLLAMA_SOURCE_DIR=/absolute/path/to/llama.cpp -DLLAMA_LIBRARY_DIR=/absolute/path/to/llama-build/bin -DCMAKE_BUILD_TYPE=Release
cmake --build workspace/native-build -j2
ctest --test-dir workspace/native-build --output-on-failure
```

To build prefill workers too, add `-DBUILD_PREFILL=ON`. Actual attention skipping
requires the private patched runtime; compilation against an unmodified runtime
alone does not activate the graph change.

For the optional real-shard integration test, supply compatible small Qwen
stage-0 Q4/Q3 shards on the same host:

```bash
DLI_NATIVE_BIN="$PWD/workspace/native-build" DLI_TEST_Q4_SHARD=/path/to/q4-stage-0.gguf DLI_TEST_Q3_SHARD=/path/to/q3-stage-0.gguf .venv/bin/python -m unittest discover -s tests -p test_native_release.py -v
```

Without those artifacts this optional test is explicitly skipped. It does not
download a substitute model or claim to validate the 14B ARM measurements.
