# Multimodal Tasks

## Image-Captioning

### SentEval COCO Dataset
We provide the COCO dataset used to profile sentence embeddings in 
[SentEval](https://github.com/facebookresearch/SentEval):

The image embeddings from this dataset are ResNet-101 2048d image embeddings. 
Each image is assigned with 5 embeddings

```
train/valid/test
├── image_features.npy
├── image_to_caption_ids.npy
├── captions.pkl
```
```
nlp_data prepare_senteval_coco --save-dir .
```

| Dataset            | #Train Samples | #Valid Samples | #Test Samples |
|--------------------|----------------|----------------|---------------|
| SentEval MSCOCO    |   113287       | 5000           |  5000         |

