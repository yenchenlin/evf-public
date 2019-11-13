# evf-public

This repo hosts the code for [Experience-embedded Visual Foresight](http://yenchenlin.me/evf/).

**Disclaimer:** code is hugely borrowed from Stochasitic Adversarial Video Prediction (SAVP) [[paper](https://arxiv.org/abs/1804.01523) | [code](https://github.com/alexlee-gk/video_prediction)]

![](http://yenchenlin.me/evf/animation.gif)

## Getting Started

### Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN


### Installation

- Clone this repo:

```
git clone git@github.com:yenchenlin/evf-public.git
cd evf-public
```

- Install dependencies

```
pip install -r requirements.txt
```

- TensorFlow >= 1.9
- Install [ffmpeg](https://ffmpeg.org/), used to generate GIFs for visualization.

### Download Omnipush

```
bash ./dataset/download_data.sh 
```

To train SAVP, pre-process the dataset into tfrecords.

```
python ./dataset/generate_tfrecords.py
```

To verify everything works correctly, `dataset` should contain the following directories.

```
dataset
├── omnipush            # raw image files, will be used for SVG
├── omnipush-tfrecords  # tfrecords, will be used for SAVP
└── ...
```

### Training

To make sure dependencies are met, run **Debug** command first.

#### Debug

```
CUDA_VISIBLE_DEVICES=0 python scripts/train_evf.py --input_dir dataset/omnipush-tfrecords/ --dataset omnipush --dataset_hparams use_state=True,sequence_length=12 --model evf --model_hparams_dict hparams/bair_action_free/ours_vae_l1/debug.json --model_hparams batch_size=4 --output_dir logs/tmp/ours_vae_l1 --summary_freq 1 --image_summary_freq 1 --eval_summary_freq 1 --accum_eval_summary_freq 1 --debug_num_datasets 2
```

#### EVF

```
python scripts/train_evf.py --input_dir dataset/omnipush-tfrecords/ --dataset omnipush --dataset_hparams use_state=True,sequence_length=12 --model evf --model_hparams_dict hparams/bair_action_free/ours_vae_l1/debug.json --model_hparams batch_size=8 --output_dir logs/evf
```

#### SAVP

```
python scripts/train.py --input_dir dataset/omnipush-tfrecords/ --dataset omnipush --dataset_hparams use_state=True,sequence_length=12 --model savp --model_hparams_dict hparams/bair_action_free/ours_vae_l1/debug.json --model_hparams batch_size=8 --output_dir logs/savp-vae 
```
