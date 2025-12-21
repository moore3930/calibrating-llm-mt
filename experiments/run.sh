# Tower-7B-Mistral with KIWI-XXL
# sbatch train_tower_calibration_7B.sh 1.0 1.0 0.0 5e-5 gpt-4o-mini-16-1.0-98-new 16 kiwi-xxl.score TowerInstruct-Mistral-7B-v0.2
sbatch test_inference.sh 1.0 1.0 0.0 5e-5 gpt-4o-mini-16-1.0-98-new 16 kiwi-xxl.score TowerInstruct-Mistral-7B-v0.2

# Rebuttal  trans==4.32
# sbatch train_llama_calibration_7B.sh 1.0 0.0 0.0 5e-5 rebuttal 5 kiwi-xxl.score Mistral-7B-Instruct-v0.3
# sbatch train_llama_calibration_7B.sh 1.0 0.0 1.0 5e-5 rebuttal 5 kiwi-xxl.score Mistral-7B-Instruct-v0.3
# sbatch train_llama_calibration_7B.sh 1.0 1.0 0.0 5e-5 rebuttal 5 kiwi-xxl.score Mistral-7B-Instruct-v0.3
# sbatch train_llama_calibration_7B.sh 1.0 1.0 0.0 3e-5 rebuttal 5 kiwi-xxl.score Mistral-7B-Instruct-v0.3
# sbatch train_llama_calibration_7B.sh 1.0 1.0 0.0 1e-5 rebuttal 5 kiwi-xxl.score Mistral-7B-Instruct-v0.3
# sbatch train_llama_calibration_7B.sh 1.0 1.0 0.0 7e-5 rebuttal 5 kiwi-xxl.score Mistral-7B-Instruct-v0.3
# sbatch train_llama_calibration_7B.sh 1.0 1.0 0.0 1e-4 rebuttal 5 kiwi-xxl.score Mistral-7B-Instruct-v0.3

# sbatch train_llama_calibration_7B.sh 1.0 1.0 0.0 5e-5 rebuttal 5 kiwi-xxl.score Mistral-7B-v0.1


# Rebuttal, Zhu
# sbatch train_llama_calibration_7B.sh 1.0 1.0 0.0 5e-5 rebuttal 5 kiwi-xxl.score TowerInstruct-Mistral-7B-v0.2

#### RUNNING ####
# sbatch train_llama_calibration_7B.sh 1.0 0.0 0.0 5e-5 rebuttal 5 kiwi-xxl.score TowerInstruct-Mistral-7B-v0.2
# sbatch train_llama_calibration_7B.sh 1.0 0.0 1.0 5e-5 rebuttal 5 kiwi-xxl.score TowerInstruct-Mistral-7B-v0.2


# Rebuttal
# sbatch train_tower_calibration_7B_doc.sh 1.0 1.0 0.0 5e-5 gpt-4o-mini-16-1.0-98-new 16 kiwi-xxl.score TowerInstruct-Mistral-7B-v0.2
