# Text Classification with Pretrained Models

## Solve GLUE Tasks with AutoML + NLP
We describe how to finetune a pretrained model for text classification. Here, we use the 
GLUE/SuperGLUE as the example.

- Get the dataset using the following command. For more details, you may refer 
to [datasets/general_nlp_benchmark](../datasets/general_nlp_benchmark).

```
nlp_data prepare_glue --benchmark glue
nlp_data prepare_glue --benchmark superglue
```

- Run GLUE Benchmark

```bash
bash run_glue.sh
```

## Solve Text Classification Benchmarks

```bash
python run_text_classification.py \
     --do_train \
     --train_file glue/mnli/train.pd.pkl \
     --dev_file glue/mnli/dev_mismatched.pd.pkl \
     --test_file glue/mnli/test_mismatched.pd.pkl \
     --task mnli \
     --batch_size 32 \
     --num_accumulated 1 \
     --ctx gpu0
```

## Kaggle Competitions

First of all, install the `kaggle` API toolkit following the documentation in 
[Kaggle API Docs](https://www.kaggle.com/docs/api).

### NLP Beginner

### Google 