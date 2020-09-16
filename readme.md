# BYOL-PyTorch

PyTorch implementation of [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733) with DDP (DistributedDataParallel) and [Apex](https://github.com/NVIDIA/apex) Amp (Automatic Mixed Precision).


## Requirements

```
python>=3.6.9
pytorch>=1.4.0
tensorboardx
opencv-python==4.2.0.34
pyyaml==5.3.1
apex
```

This repo supposes using `torch.distributed.launch` to start training, for example:

```bash
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="" --master_port=12345 byol_main.py
```

## Implementation Details

1. Use `apex` or `pytorch>=1.4.0` for `SyncBatchNorm`
2. **Pay attention to the data augmentations**, which are slightly different from those in SimCLR, especially the probability of applying `GaussianBlur` and `Solarization` for different views (see Table 6 of the paper)
3. In both training and evaluation, they normalize color channels by subtracting the average color and dividing by the standard deviation, computed on ImageNet, after applying the augmentations (even with the specially designed augmentations)
4. Increase target model momentum factor with a cosine rule
5. Exclude `biases` and `batch normalization` parameters from both `LARS adaptation` and `weight decay`
6. The correct order for model wrapping: `convert_syncbn` -> `cuda` -> `amp.initialize` -> `DDP`


## Reproduced Results

### Linear Evaluation Protocol

Here we post our reproduced results with hyper parameters in [train_config.yaml](./config/train_config.yaml) using 32x Nvidia V100 (32GB) GPU cards, indicating a global batch size of 4096.

Under this setup, reference accuracies for 300 epochs are 72.5% (top-1) and 90.8% (top-5), as reported in Section F of the paper.

| Train Epoch | Classifier Train Epoch | Classifier LR   | Top-1 ACC | Top-5 ACC |
|-------------|------------------------|-----------------|-----------|-----------|
| 100         | 100                    | [1., 0.]/Cosine | 64.34%    | 85.95%    |
| 200         | 100                    | [1., 0.]/Cosine | 69.48%    | 89.07%    |
| 300         | 100                    | [1., 0.]/Cosine | 72.29%    | 90.90%    |


## Model Checkpoints

We release our BYOL checkpoint of ResNet-50 model trained on ImageNet (ILSVRC 2012) for 300 epochs on [Google Drive](https://drive.google.com/file/d/1TLZHDbV-qQlLjkR8P0LZaxzwEE6O_7g1/view?usp=sharing).

Note:
This checkpoint file is created using legacy codebase, which saves the `online_backbone`, `online_projection`, `predictor` and target network modules separately. And the parameter names are slightly different from `ResNet50` in `torchvision.models`.
Concretely, parameters in `checkpoint['online_backbone']` is probably what you want, from `conv1` to `avgpool`, yielding 318 parameters.

You can refer to [utils/load_and_convert.py](./utils/load_and_convert.py) for the parameter name agnostic loading and converting (to torchscript).

## Remarks

I have modified this repo (remove some code) to meet the compliance issues from my former employer.
If you have forked or saved the legacy versions, please **delete** them.
