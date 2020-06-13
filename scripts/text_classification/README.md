# Text Classification with Pretrained Models
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
TASK=glue/sst
FEATURE_COLUMNS=sentence
LABEL_COLUMNS=label
TRAIN_FILE=${TASK}/train.pd.pkl
DEV_FILE=${TASK}/dev.pd.pkl
TEST_FILE=${TASK}/test.pd.pkl
python run_text_classification.py \
     --do_train \
     --train_file ${TRAIN_FILE} \
     --dev_file ${DEV_FILE} \
     --test_file ${TEST_FILE} \
     --feature_columns ${FEATURE_COLUMNS} \
     --label_columns ${LABEL_COLUMNS}
```
