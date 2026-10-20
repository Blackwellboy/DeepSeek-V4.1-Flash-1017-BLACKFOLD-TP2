# Credits

BLACKFOLD puts together other people's work. Our own part is listed at the end.

## Model and weights

- **DeepSeek-AI**: [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT), its Engram
  tables, tokenizer, `encoding/` and the official V4.1 chat template. Pinned at `1bae0e10`.
- **LibertAIDAI**: [DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
  (MIT), the EXL3 pack these weights are built on. Pinned at `afc7aa0a`. Its card credits
  [Mia-AiLab](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) for the earlier tensor layout and per-tensor bitrates.
- **turboderp / turboderp-org**: [exllamav3](https://github.com/turboderp-org/exllamav3) (MIT), the EXL3 format and quantizer,
  and the expert kernels TensorFold's native engine builds on.

## Refusal removal

- **p-e-w**: [Heretic](https://github.com/p-e-w/heretic) (AGPL-3.0), the method. No Heretic source is redistributed here.
- **pqhaz**: [DeepSeek-V4.1-Flash-1017-heretic](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic) (MIT), the
  refusal directions and settings re-encoded into EXL3 for NIGHTFALL: strength 2.5, attention `wo_b` only, blocks 2 to 39.
  Pinned at `12d58ad0`.

## Engine

- **ashhart**: [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0), the inference engine.
- **bertholomus (BertholomusAI)**: the DeepSeek-V4.1 fork [bertholomus/TensorFold](https://github.com/bertholomus/TensorFold),
  including the native Zig `dsv41` family, the RDMA ring, the Python TP2 engine and `kit_bench.py`, and the
  [TP2 recipe and kernel kit](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10).
- Upstream fixes included at `629c524`: the shared-prefix placement fix, upstream PR #26 by Manrique / hashd1ve; the
  retained-prompt-extent fix by bertholomus from sayyidfareed's report; template changes by Nathan McBride; truncated
  UTF-8 handling by Jay Leaton; the configurable RDMA GID index by sayyidfareed.
- **NVIDIA**: the `nvcr.io/nvidia/pytorch:26.07-py3` container, CUDA, NCCL.

## Recipes, tools and ideas we used

- **lucataco**: [DeepSeek-V4.1-Flash-2x-DGX-Spark](https://github.com/lucataco/DeepSeek-V4.1-Flash-2x-DGX-Spark) (Apache-2.0),
  the two-Spark recipe our first serve came from, its `bench/run-bench.sh` and its seeded MMLU/GSM8K evaluation with
  DeepSeek API reference answers. lucataco credits sudoingX's [dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41).
- **tonyd2wild**: [DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark), the
  TensorFold TP2 lane we compare against, `v41bench_tf.py`, and the published two-rail measurements and runtime-address
  approach for the second RoCE twin.
- **MiaAI-Lab**: [DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks),
  whose notes on using both PCIe twins of a port as rails (code by webzone) and on matched GPU clocks informed our setup.
  None of its patches are included.
- [HarmBench](https://github.com/centerforaisafety/HarmBench), [StrongREJECT](https://github.com/dsbowen/strong_reject) and
  JailbreakBench behaviours: the refusal evaluations reported on the NIGHTFALL card.

## Blackwellboy

- Re-encoded pqhaz's edited `wo_b` weights into EXL3 and built the NIGHTFALL patch and assembly scripts.
- Found why native TensorFold took more verification rounds than the Python engine (copy windows plus the live overhead
  estimate) and wrote patches 0002 and 0003; wrote patch 0001 from observed streaming failures.
- Two-rail setup with GID alignment, the controlled A/B trials, qualification and every measurement in `results/`.
