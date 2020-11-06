#!/bin/bash

if [[ "$1" == "serve" ]]; then
    echo -e "@ entrypoint -> launching serving script \n"
    python3 sagemaker_serve.py
else
    echo -e "@ entrypoint -> launching training script \n"
    python3 sagemaker_train.py
fi
