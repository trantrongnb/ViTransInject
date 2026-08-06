# ViTransInject

Code cho nghiên cứu prompt injection trên các mô hình LLM tiếng Việt (bilingual/lookalike/teencode/no-diacritic variants, guard defenses).

## Cấu trúc repo

```
configs/    # templates/config cho việc sinh injection
scripts/
  eval/     # chạy pipeline RQ1/RQ2/RQ3, judge BTU (có/không defense)
  analysis/ # phân tích thống kê (McNemar test, ...)
visual/     # script + hình vẽ kết quả (fig1, fig2)
```

Các thư mục sau **không** được đưa vào repo này (xem `.gitignore`):

- `data/` — dữ liệu (seeds, attack, variants, final, ...)
- `results/` — kết quả thực nghiệm (output của scripts/eval)
- `llama.cpp/`, `models/` — llama.cpp build và trọng số GGUF, tải riêng
- `external/` — dữ liệu bên thứ ba (UIT-ViQuAD2.0, vietnews)
- `docs/` — báo cáo/tài liệu nội bộ
- `scripts/gen/` — script sinh dữ liệu

## Tải dữ liệu

Dataset được host công khai trên Hugging Face:

```bash
huggingface-cli download trongnb/ViTransInject --repo-type dataset --local-dir data
```

hoặc bằng Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="trongnb/ViTransInject", repo_type="dataset", local_dir="data")
```

Dataset page: https://huggingface.co/datasets/trongnb/ViTransInject

## Chạy inference (llama.cpp)

1. Clone và build [llama.cpp](https://github.com/ggml-org/llama.cpp) vào thư mục `llama.cpp/`.
2. Tải các model GGUF (Llama-3.2-1B, Qwen2.5, Gemma-2-2B, Sailor2-1B, ...) vào thư mục `models/`.
3. Chạy server, ví dụ:

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-server \
  -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --port 8080 -ngl 99 --alias llama-3.2-1b &
```


## Chạy pipeline đánh giá

```bash
python scripts/eval/rq1_pipeline.py
python scripts/eval/rq2_variants.py
python scripts/eval/rq3_guard.py
python scripts/eval/judge_btu.py
python scripts/eval/judge_btu_defense.py
```
