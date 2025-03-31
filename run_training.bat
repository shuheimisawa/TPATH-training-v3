@echo off
setlocal

:: Activate virtual environment
call fresh_venv\Scripts\activate.bat

:: Run the training script
python scripts/train.py --config experiments/configs/baseline.yaml --data-dir data_test/processed --checkpoint-dir experiments/checkpoints --log-dir experiments/logs

:: Deactivate virtual environment
deactivate

endlocal 