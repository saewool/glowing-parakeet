# glowing-parakeet

Script `train_llm_pytorch.py` huấn luyện một mô hình GPT mini bằng **PyTorch**.

## Điểm chính
- Tự động tải **nhiều dataset** từ internet và ghép lại để train.
- Tokenization mức ký tự (char-level).
- Huấn luyện trên CPU/GPU.
- Cải tiến train với `AdamW + weight_decay + gradient clipping`.
- Lưu artifacts sau train: `model_test.pt`, `sample.txt`, `metrics.json`.

## Dataset có sẵn
- `tiny_shakespeare`
- `alice`
- `nietzsche`
- `metamorphosis`

Ví dụ train với nhiều dataset:
```bash
python3 train_llm_pytorch.py \
  --datasets tiny_shakespeare,alice,metamorphosis \
  --max-steps 200 \
  --batch-size 32 \
  --block-size 128
```

## Chạy rất nhanh để test
```bash
python3 train_llm_pytorch.py \
  --datasets tiny_shakespeare,alice \
  --max-steps 5 \
  --eval-interval 1 \
  --eval-iters 2 \
  --batch-size 8 \
  --n-embed 64 \
  --n-heads 4 \
  --n-layers 2 \
  --sample-tokens 80 \
  --output-dir artifacts
```

## CI/CD
Repository có GitHub Actions workflow tại `.github/workflows/ci.yml`:
- Cài Python 3.11
- Cài `torch` bản CPU
- Chạy syntax check cho script
- Chạy train test với cấu hình nhỏ
- Upload artifacts (`artifacts/`) để lưu model test + metrics + sample text
