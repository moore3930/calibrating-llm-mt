#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=01-00:00:00

#SBATCH -o /gpfs/work4/0/gus20642/dwu18/log/out.calibration.%j.o
#SBATCH -e /gpfs/work4/0/gus20642/dwu18/log/out.calibration.%j.e

source activate py38cuda11
# source activate py39

export HF_HUB_CACHE=/gpfs/work4/0/gus20642/dwu18/cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

evaluate_lang_directions() {
    # Parameters
    local TEST_DATASET="$1"  # Test dataset name
    local BASE_SYS="$2"     # Base system directory

    # Define language directions (can customize or pass as parameter if needed)
    local LANG_DIRECTIONS=("en-de" "en-es" "en-ru" "en-zh" "en-fr" "en-nl" "en-it" "en-pt" "en-ko") # tower-1 langs

    # Define base source and target directories
    local BASE_SRC="/gpfs/work4/0/gus20642/dwu18/project/value_finetuning/src/llama_recipes/customer_data/${TEST_DATASET}/test"
    local BASE_TGT=$BASE_SRC
    # local BASE_TGT="/gpfs/work4/0/gus20642/dwu18/project/value_finetuning/src/llama_recipes/customer_data/${TEST_DATASET}/test"
    

    # Loop through each language direction
    for LANG_DIR in "${LANG_DIRECTIONS[@]}"; do
        # Extract source and target language codes
        local SRC_LANG=$(echo $LANG_DIR | cut -d'-' -f1)
        local TGT_LANG=$(echo $LANG_DIR | cut -d'-' -f2)

        # Define the file paths
        local SRC_FILE="${BASE_SRC}/${LANG_DIR}/test.${LANG_DIR}.${SRC_LANG}"
        local TGT_FILE="${BASE_TGT}/${LANG_DIR}/test.${LANG_DIR}.${TGT_LANG}"
        local SYS_FILE="${BASE_SYS}/${LANG_DIR}/hyp.${LANG_DIR}.${TGT_LANG}"

        # Define the output score files
        local COMET_SCORE_FILE="./${BASE_SYS}/${LANG_DIR}/comet.score"
        local XCOMET_SCORE_FILE="./${BASE_SYS}/${LANG_DIR}/xcomet.score"
        local KIWI_SCORE_FILE="./${BASE_SYS}/${LANG_DIR}/kiwi.score"
        local KIWI_XL_SCORE_FILE="./${BASE_SYS}/${LANG_DIR}/kiwi-xl.score"
        local KIWI_XXL_SCORE_FILE="./${BASE_SYS}/${LANG_DIR}/kiwi-xxl.score"

        echo "Calculating COMET scores for ${LANG_DIR}..."

        # Run COMET scoring
        comet-score -s $SRC_FILE -t $SYS_FILE -r $TGT_FILE --model Unbabel/wmt22-comet-da >> $COMET_SCORE_FILE
        comet-score -s $SRC_FILE -t $SYS_FILE --model Unbabel/XCOMET-XXL >> $XCOMET_SCORE_FILE
        comet-score -s $SRC_FILE -t $SYS_FILE --model Unbabel/wmt22-cometkiwi-da >> $KIWI_SCORE_FILE
        comet-score -s $SRC_FILE -t $SYS_FILE --model Unbabel/wmt23-cometkiwi-da-xl >> $KIWI_XL_SCORE_FILE
        comet-score -s $SRC_FILE -t $SYS_FILE --model Unbabel/wmt23-cometkiwi-da-xxl >> $KIWI_XXL_SCORE_FILE

        echo "Finished ${LANG_DIR}"
    done

    echo "All language directions processed!"
}



################## MAIN ##################

ALPHA=$1
BETA=$2
GAMA=$3
LR=$4
SUBSET=$5
LIST_SIEZ=$6
METRIC=$7
BASE_MODEL=$8

echo "ALPHA is set to: $ALPHA"
echo "BETA is set to: $BETA"
echo "GAMA is set to: $GAMA"
echo "LR is set to: $LR"
echo "Subset is set to: $SUBSET"
echo "List size is set to: $LIST_SIEZ"
echo "Base_model is set to: $BASE_MODEL"

SETTING=${ALPHA}-${BETA}-${GAMA}-${LR}-${METRIC}-debug2
TEST_DATASET=wmt24_testset
CKP_DIR=/gpfs/work4/0/gus20642/dwu18/project/calibrating-llm-mt/experiments/checkpoints

echo "CKP: $CKP_DIR/$BASE_MODEL/calibration/${SUBSET}/${SETTING}"
echo "RESULTS: results/$BASE_MODEL/calibration/${TEST_DATASET}/${SUBSET}/${SETTING}-beam5"
echo "SCORES: scores/$BASE_MODEL/calibration/${SUBSET}/${SETTING}/0/wmt-qe-22-test"

# Train
python -m llama_recipes.calibration --use_peft --peft_method lora \
        --model_name Unbabel/$BASE_MODEL \
        --output_dir $CKP_DIR/$BASE_MODEL/calibration/${SUBSET}/${SETTING} \
        --dataset calibration \
        --subset_name ${SUBSET} \
        --metric ${METRIC} \
        --batching_strategy padding \
        --num_epochs 1 \
        --lr $LR \
        --batch_size_training 32 \
        --val_batch_size 32 \
        --gradient_accumulation_steps 8 \
        --alpha $ALPHA \
        --beta $BETA \
	    --gama $GAMA \
        --lang_pairs "en-de,en-fr,en-nl,en-it,en-es,en-pt,en-ko,en-ru,en-zh" \
        --listwise_loss \
        --list_size $LIST_SIEZ \
        --use_wandb

# Test
for EPOCH in 0; do
    BASE_SYS=results/$BASE_MODEL/calibration/${TEST_DATASET}/${SUBSET}/${SETTING}-beam5/${EPOCH}
    python inference_formal.py --model_name Unbabel/$BASE_MODEL \
            --peft_model $CKP_DIR/$BASE_MODEL/calibration/${SUBSET}/${SETTING}/${EPOCH} \
            --dataset ${TEST_DATASET} \
            --val_batch_size 8 \
            --do_sample False \
            --output_dir ${BASE_SYS} \
            --lang_pairs en-de,en-fr,en-nl,en-it,en-es,en-pt,en-ko,en-ru,en-zh \
            --beam_size 5
    evaluate_lang_directions ${TEST_DATASET} ${BASE_SYS}
done

:<<!
# Spearman
# Get scores (log_probs) on WMT22 Metric Data (Train & Test Set)
for EPOCH in 0; do
    python reward_inference.py --model_name Unbabel/$BASE_MODEL \
        --peft_model $CKP_DIR/$BASE_MODEL/calibration/${SUBSET}/${SETTING}/${EPOCH} \
        --dataset da_dataset \
        --subset_name wmt-qe-2022.test.csv \
        --val_batch_size 32 \
        --do_sample False \
        --output_dir scores/$BASE_MODEL/calibration/${SUBSET}/${SETTING}/${EPOCH}/wmt-qe-22-test \
        --lang_pairs en-de,en-ru
done
!
