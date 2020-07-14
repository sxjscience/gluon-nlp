nlp_data prepare_glue --benchmark glue
CTX=gpu0

for TASK in cola sst mrpc sts qqp qnli rte wnli
do
    TRAIN_FILE=glue/${TASK}/train.pd.pkl
    DEV_FILE=glue/${TASK}/dev.pd.pkl
    TEST_FILE=glue/${TASK}/test.pd.pkl
    python run_text_classification.py \
     --do_train \
     --train_file ${TRAIN_FILE} \
     --dev_file ${DEV_FILE} \
     --test_file ${TEST_FILE} \
     --task ${TASK} \
     --batch_size 32 \
     --num_accumulated 1 \
     --ctx ${CTX}
done

python run_text_classification.py \
     --do_train \
     --train_file glue/mnli/train.pd.pkl \
     --dev_file glue/mnli/dev_matched.pd.pkl \
     --test_file glue/mnli/test_matched.pd.pkl \
     --task mnli \
     --batch_size 32 \
     --num_accumulated 1 \
     --ctx ${CTX}

python run_text_classification.py \
     --do_train \
     --train_file glue/mnli/train.pd.pkl \
     --dev_file glue/mnli/dev_mismatched.pd.pkl \
     --test_file glue/mnli/test_mismatched.pd.pkl \
     --task mnli \
     --batch_size 32 \
     --num_accumulated 1 \
     --ctx ${CTX}
