# Benchmarking the Performance of NLP Backbones

We benchmark the training, inference latency + memory usage of the NLP backbones.
For comparison, we also provide the numbers of 

## Backbones in HuggingFace

We use the [huggingface benchmark](https://github.com/huggingface/transformers/tree/master/examples/benchmarking) 
to benchmark the training + inference speed of common workloads in NLP. 

```bash
python3 -m pip install -U -r requirements.txt --user
python3 benchmark_hf.py
```

## GluonNLP Backbones based on MXNet

```bash
python3 -m pip install -U -r requirements.txt --user
python3 benchmark_gluonnlp.py
```