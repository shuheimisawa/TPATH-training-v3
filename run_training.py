import subprocess
import sys
import os

def run_training():
    # Get the path to the virtual environment Python interpreter
    venv_python = os.path.join('fresh_venv', 'Scripts', 'python.exe')
    
    # Command arguments
    args = [
        venv_python,
        'scripts/train.py',
        '--config', 'experiments/configs/baseline.yaml',
        '--data-dir', 'data_test/processed',
        '--checkpoint-dir', 'experiments/checkpoints',
        '--log-dir', 'experiments/logs'
    ]
    
    # Run the training script
    try:
        process = subprocess.run(args, check=True)
        return process.returncode
    except subprocess.CalledProcessError as e:
        print(f"Training script failed with return code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"An error occurred: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(run_training()) 