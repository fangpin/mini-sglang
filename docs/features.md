# Features of Mini-SGLang

## Online Serving

Mini-SGLang supports online serving with an OpenAI-compatible API server. It provides the standard `/v1/chat/completions` endpoint, allowing seamless integration with existing tools and clients. For detailed command-line arguments and configuration options, run `python -m minisgl --help`.

## Interactive Shell Mode

For demonstration and testing purposes, an interactive shell mode is available. In this mode, users can input prompts directly, and the LLM will generate responses in real-time. The shell automatically caches chat history to maintain context. To clear the conversation history and start a new session, use the `/reset` command.

Example:

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

## Distributed Serving

To scale performance across multiple GPUs, Mini-SGLang supports Tensor Parallelism (TP). You can enable distributed serving by specifying the number of GPUs with the `--tp n` argument, where `n` is the degree of parallelism.

## Supported Models

Our framework currently supports the following dense model architectures:

- [`Llama-3`](https://huggingface.co/collections/meta-llama/llama-31) series
- [`Qwen-3`](https://huggingface.co/collections/Qwen/qwen3) series (including MoE)
- [`Qwen-2.5`](https://huggingface.co/collections/Qwen/qwen25) series

## Chunked Prefill

Chunked Prefill, a technique introduced by [Sarathi-Serve](https://arxiv.org/abs/2403.02310), is enabled by default. This feature splits long prompts into smaller chunks during the prefill phase, significantly reducing peak memory usage and preventing Out-Of-Memory (OOM) errors in long-context serving. The chunk size can be configured using `--max-prefill-length n`. Note that setting `n` to a very small value (e.g., 128) is not recommended as it may significantly degrade performance.

## Page Size

You can specify the page size of the system using the `--page-size` argument.

## Delayed Prefill Admission

By default, the scheduler runs a prefill batch whenever one can be formed, then falls
back to decode. To accumulate new prefills while existing requests decode, enable a
waiting-time threshold, a request-count threshold, or both:

```bash
python -m minisgl --model Qwen/Qwen3-0.6B \
  --delay-prefill-wait-sec 0.05 \
  --delay-prefill-min-batch-size 8
```

New prefills become eligible when **either** the oldest waiting request reaches the
time threshold **or** the number of new waiting requests reaches the count threshold.
With no runnable decode requests, prefills are eligible immediately. Both thresholds
default to zero (disabled); each zero threshold is ignored when the other is enabled.
The corresponding `SchedulerConfig` fields are `delay_prefill_wait_sec` and
`delay_prefill_min_batch_size`, including for offline inference.

Waiting time starts when the scheduler queues a request and uses a monotonic clock.
TP ranks make queue- and count-based decisions locally. When admission depends on
waiting time, rank 0 evaluates the clock and broadcasts the decision so every rank
selects the same phase. Already started chunked prefills continue without waiting for
new requests; they do not count toward the request threshold. Prefix-cache matches still count as
new requests until their first chunk is scheduled. Actual batches remain subject to
the existing token budget, request slots, and KV cache capacity. If eligible prefills
cannot allocate resources, decode continues.

Decode requests stay in the decode manager during prefill batches, and both normal
and overlap scheduling retain their usual output processing. Add
`--delay-prefill-debug` to log admission decisions, new waiting request counts, oldest
wait time, and decode counts on rank 0. The default is no extra logging.

This policy trades extra prefill queueing and occasional decode pauses for potentially
more consecutive decode steps and larger prefill batches. The time threshold is not a
TTFT bound. A count-only threshold can delay sparse arrivals for as long as decode
remains active. Sustained prefill backlog can produce consecutive prefill batches;
there is no limit on the length of a prefill burst.

Validate output completeness before measuring performance. Compare disabled and
enabled runs with the same model, CUDA graph settings, input/output lengths, cache
state, and arrival sequence. Measure actual delivered output tokens/s, TTFT, and
inter-token latency percentiles under sparse, bursty, and sustained traffic; an
improvement in average decode latency alone does not establish a serving benefit.
CPU scheduling regressions can be run with
`PYTHONPATH=python pytest tests/core/test_delay_prefill.py`; model correctness and
performance additionally require a Linux CUDA host.

## Attention Backends

Mini-SGLang integrates high-performance attention kernels, including [`FlashAttention`](https://github.com/Dao-AILab/flash-attention) (`fa`), [`FlashInfer`](https://github.com/flashinfer-ai/flashinfer) (`fi`) and [`TensorRT-LLM fmha`](https://github.com/NVIDIA/TensorRT-LLM) (`trtllm`). It supports using different backends for the prefill and decode phases to maximize efficiency. For example, on NVIDIA Hopper GPUs, `FlashAttention 3` is used for prefill and `FlashInfer` for decode by default.

You can specify the backend using the `--attn` argument. If two values are provided (e.g., `--attn fa,fi`), the first specifies the prefill backend and the second the decode backend. Note that some attention backend might override the user-provided page size (e.g. `trtllm` only supports page size 16,32,64).

## CUDA Graph

To minimize CPU launch overhead during decoding, Mini-SGLang supports capturing and replaying CUDA graphs. This feature is enabled by default. The maximum batch size for CUDA graph capture can be set with `--cuda-graph-max-bs n`. Setting `n` to `0` disables this feature.

## Radix Cache

Adopting the original design from [SGLang](https://github.com/sgl-project/sglang.git), Mini-SGLang implements a Radix Cache to manage the Key-Value (KV) cache. This allows the reuse of KV cache for shared prefixes across requests, reducing redundant computation. This feature is enabled by default but can be switched to a naive cache management strategy using `--cache naive`.

![radix](https://lmsys.org/images/blog/sglang/radix_attn.jpg)
*Illustration of Radix Attention from [LMSYS Blog](https://lmsys.org/blog/2024-01-17-sglang/).*

## Overlap Scheduling

To further reduce CPU overhead, Mini-SGLang employs overlap scheduling, a technique proposed in [NanoFlow](https://arxiv.org/abs/2408.12757). This approach overlaps the CPU scheduling overhead with GPU computation, improving overall system throughput.

![overlap](https://lmsys.org/images/blog/sglang_v0_4/scheduler.jpg)
*Illustration of Overlap Scheduling from [LMSYS Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/).*
