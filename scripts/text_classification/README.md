# Text Classification with Pretrained Models

## Solve GLUE and SuperGLUE with AutoML + NLP
We describe how to finetune a pretrained model for text classification. Here, we use the 
GLUE/SuperGLUE as the example.

- Get the dataset using the following command. For more details, you may refer 
to [datasets/general_nlp_benchmark](../datasets/general_nlp_benchmark).

```
nlp_data prepare_glue --benchmark glue
nlp_data prepare_glue --benchmark superglue
```

- Run training script

```bash
TASK=mnli
TRAIN_FILE=glue/${TASK}/train.pd.pkl
DEV_FILE=glue/${TASK}/dev_matched.pd.pkl
TEST_FILE=glue/${TASK}/test_matched.pd.pkl
python run_text_classification.py \
     --do_train \
     --train_file ${TRAIN_FILE} \
     --dev_file ${DEV_FILE} \
     --test_file ${TEST_FILE} \
     --task ${TASK} \
     --batch_size 32 \
     --num_accumulated 1 \
     --gpus 0
```


```bash
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
     --ctx gpu0
done
```

## Kaggle Competitions

First of all, install the `kaggle` API toolkit following the documentation in 
[Kaggle API Docs](https://www.kaggle.com/docs/api).

### NLP 