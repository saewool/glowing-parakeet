# glowing-parakeet

Script `train_llm_pytorch.py` huấn luyện một mô hình GPT mini bằng **PyTorch**.

## Điểm chính
- Tự động tải dataset Tiny Shakespeare từ internet.
- Tokenization mức ký tự (char-level).
- Huấn luyện ngay trên CPU/GPU.
- In loss/perplexity định kỳ và sinh văn bản mẫu sau train.

## Chạy nhanh
```bash
python3 train_llm_pytorch.py --max-steps 200 --batch-size 32 --block-size 128
```

## Chạy rất nhanh để test
```bash
python3 train_llm_pytorch.py --max-steps 5 --eval-interval 1 --eval-iters 2 --batch-size 8 --n-embed 64 --n-heads 4 --n-layers 2 --sample-tokens 80
```

## CI/CD
Repository đã có GitHub Actions workflow tại `.github/workflows/ci.yml`:
- Cài Python 3.11
- Cài `torch` bản CPU
- Chạy syntax check cho script
- Chạy thử train end-to-end với cấu hình nhỏ để xác nhận script hoạt động
