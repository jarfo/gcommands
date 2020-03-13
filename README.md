# Speech Commands Recognition

Training Deep Learning models using [Google Speech Commands Dataset](https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html), implemented in [PyTorch](http://pytorch.org).
<!-- This repo contains a data loader for the Kaldi data set format, as well as training scripts for single-word neural net models, written in [PyTorch](http://pytorch.org). -->

## Features
* Training and testing basic ConvNets and TDNNs.
* Standard Train, Test, Valid folders for the Google Speech Commands Dataset.
* Dataset loader for standard [Kaldi](https://github.com/kaldi-asr/kaldi) speech data folders (files and pipes).

## Requirements

* Install [PyTorch](https://github.com/pytorch/pytorch#installation)
* Install [SoX](http://sox.sourceforge.net/)

To install SoX on Mac with [Homebrew](https://brew.sh):

```brew install sox```

on Linux:

```sudo apt-get install sox```


* Install [LibRosa](https://github.com/librosa/librosa) with pip:

```pip install librosa```

## Usage

### Google Speech Commands Dataset
To download and extract the [Google Speech Commands Dataset](https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html) run the following command:
```
./download_audio.sh
```

### Training
Use `python run.py --help` for more parameters and options.

```
python run.py --arc VGG16 --checkpoint VGG16 --num_workers 10
```

### Results (Isolated word recognition, 36 words)
Accuracy results for the validation and test sets using the default parameters (VGG16) and with data augmentation (VGG16 + sp) 

| Model | Valid acc. | Test acc.| parameters and options |
| ------------- | ------------- | ------------- | ------------- | 
| VGG16 | 95.5% | 95.9% | default |
| VGG16 + sp | 96.0% | 96.3% | --train_path data/train_training_sp |

The augmented training dataset train_training_sp is an speed perturbed version of the train_training dataset. It was obtained using the [Kaldi](https://github.com/kaldi-asr/kaldi) script [perturb_data_dir_speed_3way.sh](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/utils/data/perturb_data_dir_speed_3way.sh)
