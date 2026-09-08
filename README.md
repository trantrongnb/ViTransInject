# ViTransInject

Code for research on prompt injection against Vietnamese LLMs (bilingual/lookalike/teencode/no-diacritic variants, guard defenses).

## Repo structure

```
configs/    # templates/config for generating injections
scripts/
  eval/     # run RQ1/RQ2/RQ3 pipelines, BTU judge (with/without defense)
  analysis/ # statistical analysis (McNemar test, ...)
visual/     # figure-generation scripts + result plots (fig1, fig2)
docs/       # project landing pages (static HTML; see docs/README.md)
```

The following directories are **not** included in this repo (see `.gitignore`):

- `data/` — dataset (seeds, attack, variants, final, ...)
- `results/` — experiment results (output of scripts/eval)
- `llama.cpp/`, `models/` — llama.cpp build and GGUF weights, download separately
- `external/` — third-party data (UIT-ViQuAD2.0, vietnews)
- `scripts/gen/` — data-generation scripts

## Downloading the data

The dataset is hosted publicly on Hugging Face:

```bash
huggingface-cli download trongnb/ViTransInject --repo-type dataset --local-dir data
```

or in Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="trongnb/ViTransInject", repo_type="dataset", local_dir="data")
```

Dataset page: https://huggingface.co/datasets/trongnb/ViTransInject

## Running inference (llama.cpp)

1. Clone and build [llama.cpp](https://github.com/ggml-org/llama.cpp) into the `llama.cpp/` directory.
2. Download the GGUF models (Llama-3.2-1B, Qwen2.5, Gemma-2-2B, Sailor2-1B, ...) into the `models/` directory.
3. Start the server, e.g.:

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-server \
  -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --port 8080 -ngl 99 --alias llama-3.2-1b &
```

## Environment variables

`scripts/eval/shared/llm.py` calls the DeepSeek API for judging. Set:

```bash
export DEEPSEEK_API_KEY="your_api_key"
```

## Running the evaluation pipeline

```bash
python scripts/eval/rq1_pipeline.py
python scripts/eval/rq2_variants.py
python scripts/eval/rq3_guard.py
python scripts/eval/judge_btu.py
python scripts/eval/judge_btu_defense.py
```
