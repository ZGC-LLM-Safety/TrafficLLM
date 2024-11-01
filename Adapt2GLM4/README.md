# Adapting TrafficLLM to GLM4

This dir contains the new adapting codes for [GLM4](https://github.com/THUDM/GLM-4). GLM4 is the most recently released version of ChatGLM, which has a faster tuning and inference speed than ChatGLM2. You can download the model and codes from the official repo and use the codes in this dir as reference for tuning, inference, and evaluating with TrafficLLM. 

The dir tree is shown as follows:
```shell
.
├── FT
│   ├── configs
│   │   └── lora.yaml
│   ├── finetune.py
│   ├── infer.sh
│   ├── inference.py
│   ├── requirements.txt
│   └── train.sh
├── Preprocess.py  # Transfer the format of training data for GLM4
└── README.md
```
Many thanks to GLM4. Thanks for their wonderful work.

Note: Please upgrade the transformers version required by the GLM-4-9B-Chat model to 4.44.0. Re-pull all files except for the model weights (*.safetensor files and tokenizer.model), and update dependencies strictly according to FT/requirements.txt.
