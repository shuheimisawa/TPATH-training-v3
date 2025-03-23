@echo off
call fresh_venv\Scripts\activate.bat
python scripts/train.py --config experiments/configs/baseline.yaml --data-dir data_test/processed --checkpoint-dir experiments/checkpoints --log-dir experiments/logs 