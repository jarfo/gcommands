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

### Results (Isolated word recognition, 31 words)
Accuracy results for the train, validation and test sets using the default parameters (VGG16). 

| Model | Train acc. | Valid acc. | Test acc.|
| ------------- | ------------- | ------------- | ------------- |
Work in progress...
