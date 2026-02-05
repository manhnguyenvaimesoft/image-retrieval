# Backend User Guide

## Các bước:
1. Thêm .env với các biến:
    - `SECRET_KEY`
    - `ALGORITHM`
    - `ACCESS_TOKEN_EXPIRE_MINUTES`
    - `YOLO_MODEL_PATH`

2. Run:
    - Run API: `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
