# Generated by "generate_commands.py"

USE_HOROVOD=${1:-0}  # Horovod flag. 0 --> not use horovod, 1 --> use horovod
VERSION=${2:-2.0}   # SQuAD Version
MODEL_NAME={{ model_name }}
BATCH_SIZE={{ batch_size }}
NUM_ACCUMULATED={{ num_accumulated }}
EPOCHS={{ epochs }}
LR={{ lr }}
WARMUP_RATIO={{ warmup_ratio }}
WD={{ wd }}
MAX_SEQ_LENGTH={{ max_seq_length }}
MAX_GRAD_NORM={{ max_grad_norm }}
LAYERWISE_DECAY={{ layerwise_decay }}

# Prepare the Data
nlp_data prepare_squad --version ${VERSION}

RUN_SQUAD_PATH=$(dirname "$0")/../run_squad.py

# Run the script
if [ ${USE_HOROVOD} -eq 0 ];
then
  RUN_COMMAND="python3 ${RUN_SQUAD_PATH} --gpus 0,1,2,3"
else
  RUN_COMMAND="horovodrun -np 4 -H localhost:4 python3 ${RUN_SQUAD_PATH} --comm_backend horovod"
fi
${RUN_COMMAND} \
    --model_name ${MODEL_NAME} \
    --data_dir squad \
    --output_dir fintune_${MODEL_NAME}_squad_${VERSION} \
    --version ${VERSION} \
    --do_eval \
    --do_train \
    --batch_size ${BATCH_SIZE} \
    --num_accumulated ${NUM_ACCUMULATED} \
    --layerwise_decay ${LAYERWISE_DECAY} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --warmup_ratio ${WARMUP_RATIO} \
    --wd ${WD} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --overwrite_cache
