<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<h1 align="center"> <p>🤗 PEFT-RoSA</p></h1>

This repository is a fork of the [huggingface Parameter-Efficient Fine-Tuning (PEFT) library](https://github.com/huggingface/peft), containing the official implementation for the paper [Robust Adaptation (RoSA)](https://arxiv.org/abs/2401.04679). The RoSA-related code can be found in [`src/peft/tuners/rosa/`](https://github.com/IST-DASLab/peft-rosa/tree/main/src/peft/tuners/rosa). Also [here](https://github.com/IST-DASLab/RoSA), we have integrated this library into [MosaicML's llm-foundry](https://github.com/mosaicml/llm-foundry), containing the experiments reported in the paper. 

## Installation
1. Make sure you have [pytorch](https://pytorch.org/) installed. Preferably, install pytorch using conda instead of pip to ensure the dependencies are installed correctly.
2. Install the [*spops*](https://github.com/IST-DASLab/spops) library, which we use under the hood to perform sparse operations. Simply run 
```
pip install git+https://github.com/IST-DASLab/spops.git
```

3. Finally, clone this repository and run 
```
pip install -e .
```

## Usage
The usage is almost identical to LoRA in the PEFT library, with some extra configuration parameters in `RosaConfig` + a single line of code adding a `RosaScheduler`. The required changes are shown in the code block below.

```
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_model, TaskType
from peft.tuners.rosa import RosaConfig, RosaScheduler
model_name_or_path = "bigscience/mt0-large"

peft_config = RosaConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,

    d=0.006,                                |
    spa_num_grads=1,                        |   <---- the new config parameters 
    grad_acc_mode='mean_squared',           |
    schedule='wl64'                         |
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)

trainer = Trainer(
    model=model,
    ...,
    callbacks=[RosaScheduler(model)]        |   <---- add RosaScheduler as a callback
)
```

The new config parameters, in line with the paper, are the following ones:
- `d`: Density of the sparse adaptation matrix. 
- `spa_num_grads`: How many batches of gradients to use for RoSA mask generation. One batch (the default value) usually works well.
- `grad_acc_mode`: How to accumulate the gradients. `mean_squared` corresponds to the empirical diagonal Fisher estimation, while `mean` corresponds to simply averaging the gradients. `mean_squared` is default, and usually obtains good results.
- `schedule`: TL;DR just use `wl64` to warm up with only low-rank adapter for 64 steps, and then start collecting gradients for sparse adaptation's mask generation. See the next section for a complete guide.


Finally, just add `RosaScheduler(model)` as a callback to the Trainer. `RosaScheduler` is also compatible with [MosaicML's composer](https://github.com/mosaicml/composer) (just add it as an Algorithm). Additionally, you can customize it for any other framework by calling scheduler's `_on_step_begin()` and `_on_step_end()` before forward and after backward, respectively.


## RoSA Schedule
The `schedule` argument in `RosaConfig` determines when each of low-rank and sparse adapters should be active, and when to generate the sparsity masks. The (currently) supported options are discussed below.

- `default`: the sparse and low-rank adapters will be enabled as soon as possible. This means that the low-rank adapter is always activated, and gradient collection will start right away and the sparse adapter will be activated once the masks are generated.
- `lora_only`: the low-rank adapter is always active, while the sparse adapter is disabled.
- `spa_only`: the low-rank adapter is always disabled, while the sparse adapter will be activated once enough gradients are collected.
- `wl64` (or `wl` + any number): start by fine-tuning the low-rank adapter alone for 64 steps, then collect gradients as long as needed and activate sparse adaptation.

Finally, as discussed in the paper, we found it beneficial to warm up with low-rank adapter only (`wl64` schedule), generate the masks, and then restart the training with both adapters activated. To do this, we suggest following the steps below, taking advantage of three extra parameters in `RosaConfig`.

1. First run your RoSA training with `schedule=wl64`, `mask_save_path=./tmp_mask`, and `terminate_after_mask_generation=True` passed into `RosaConfig`, which saves the generated mask (after low-rank warmup) in the `./tmp_mask` file and terminates the run.
2. Re-run the training with `schedule=default` and `mask_load_path=./tmp_mask`, which loads the masks directly from the file and activates both low-rank and sparse adapters right away.


## Citation
If you plan to use our work in your projects, please consider citing our paper:

```
@article{nikdan2024rosa,
  title={RoSA: Accurate Parameter-Efficient Fine-Tuning via Robust Adaptation},
  author={Nikdan, Mahdi and Tabesh, Soroush and Crnčević, Elvir and Alistarh, Dan},
  journal={arXiv preprint arXiv:2401.04679},
  year={2024}
}
```
