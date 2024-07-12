set -e

# assumes 4 nodes, each with 8 GPUs
DATA_DIR=./atlas_data
#SIZE=large # lets use large, (slower than base, but still quite fast and accessible, but less accurate than xl or xxl)
SIZE=base # lets use large, (slower than base, but still quite fast and accessible, but less accurate than xl or xxl)

# download the NQ data
python preprocessing/download_model.py --model models/atlas/${SIZE} --output_directory ${DATA_DIR}

SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=nq-full-shot-example
TRAIN_STEPS=50000

#    --index_mode flat \
python train.py \
    --shuffle \
    --train_retriever \
    --gold_score_mode pdist \
    --use_gradient_checkpoint_reader \
    --use_gradient_checkpoint_retriever \
    --precision fp32 \
    --shard_optim --shard_grads \
    --temperature_gold 0.01 --temperature_score 0.01 \
    --refresh_index -1 \
    --query_side_retriever_training \
    --target_maxlength 16 \
    --reader_model_type google/t5-${SIZE}-lm-adapt \
    --dropout 0.1 --weight_decay 0.01 --lr 4e-5 --lr_retriever 4e-5 --scheduler linear \
    --text_maxlength 200 \
    --model_path "${DATA_DIR}/models/atlas/${SIZE}" \
    --train_data "/localhome/qingyu/FiD_data/NQ/train.jsonl" \
    --eval_data "/localhome/qingyu/FiD_data/NQ/dev.jsonl" \
    --per_gpu_batch_size 1 \
    --n_context 100 \
    --use_file_passages \
    --retriever_n_context 40 \
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${SAVE_DIR} \
    --eval_freq ${TRAIN_STEPS} \
    --log_freq 1000 \
    --total_steps ${TRAIN_STEPS} \
    --warmup_steps 3000 \
    --save_freq ${TRAIN_STEPS} \
    --write_results \
    --task qa \
    --index_mode faiss \
    --passages "${DATA_DIR}/corpora/wiki/enwiki-dec2018/text-list-100-sec.jsonl" "${DATA_DIR}/corpora/wiki/enwiki-dec2018/infobox.jsonl" \
    --save_index_path ${SAVE_DIR}/${EXPERIMENT_NAME}/saved_index
