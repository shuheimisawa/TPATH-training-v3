@echo off
python scripts/train_enhanced.py --config experiments/configs/improved_v1.yaml --data-dir data_test/processed --checkpoint-dir experiments/checkpoints --log-dir experiments/logs 