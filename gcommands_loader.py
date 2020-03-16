# coding=utf-8
# Copyright 2018 jose.fonollosa@upc.edu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data generator for the Google speech commands data set using standard Kaldi data folders."""

from __future__ import print_function, division

import os.path
import subprocess
import struct
import wave

import numpy as np
from mfsc import mfsc
import torch
import torch.utils.data as data

# Words for Google Speech Commands v0.02 plus 'silence' for noise recordings
CLASSES = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 
           'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 
           'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero',
           'silence']

print("Number of labels:", len(CLASSES))

# pylint: disable=ungrouped-imports
try:
    from subprocess import DEVNULL # python3
except ImportError:
    DEVNULL = open(os.devnull, 'wb')


def wav_read(pipe):
    if pipe[-1] == '|':
        tpipe = subprocess.Popen(pipe[:-1], shell=True, stderr=DEVNULL, stdout=subprocess.PIPE)
        audio = tpipe.stdout
    else:
        tpipe = None
        audio = pipe
    try:
        wav = wave.open(audio, 'r')
    except EOFError:
        print('EOFError:', pipe)
        exit(-1)
    sfreq = wav.getframerate()
    assert wav.getsampwidth() == 2
    wav_bytes = wav.readframes(-1)
    npts = len(wav_bytes) // wav.getsampwidth()
    wav.close()
    # convert binary chunks
    wav_array = np.array(struct.unpack("%ih" % npts, wav_bytes), dtype=float) / (1 << 15)
    return wav_array, sfreq


def get_classes():
    classes = CLASSES
    weight = None
    class_to_id = {label: i for i, label in enumerate(classes)}
    return classes, weight, class_to_id


def get_segment(wav, seg_ini, seg_end):
    nwav = None
    if float(seg_end) > float(seg_ini):
        if wav[-1] == '|':
            nwav = wav + ' sox -t wav - -t wav - trim {} ={} |'.format(seg_ini, seg_end)
        else:
            nwav = 'sox {} -t wav - trim {} ={} |'.format(wav, seg_ini, seg_end)
    return nwav


def make_dataset(kaldi_path, class_to_id):
    text_path = os.path.join(kaldi_path, 'text')
    wav_path = os.path.join(kaldi_path, 'wav.scp')
    segments_path = os.path.join(kaldi_path, 'segments')

    with open(text_path, 'rt') as text:
        key_to_word = dict()
        for line in text:
            key, word = line.strip().split(' ', 1)
            key_to_word[key] = word

    with open(wav_path, 'rt') as wav_scp:
        key_to_wav = dict()
        for line in wav_scp:
            key, wav = line.strip().split(' ', 1)
            key_to_wav[key] = wav

    wavs = []
    if os.path.isfile(segments_path):
        with open(segments_path, 'rt') as segments:
            for line in segments:
                key, wav_key, seg_ini, seg_end = line.strip().split()
                wav_command = key_to_wav[wav_key]
                word = key_to_word[key]
                word_id = class_to_id[word]
                wav_item = [key, get_segment(wav_command, seg_ini, seg_end), word_id]
                wavs.append(wav_item)
    else:
        for key, wav_command in key_to_wav.items():
            word = key_to_word[key]
            word_id = class_to_id[word]
            wav_item = [key, wav_command, word_id]
            wavs.append(wav_item)

    return wavs


def param_loader(path, window_size, window_stride, window, normalize, max_len):
    y, sfr = wav_read(path)

    param = mfsc(y, sfr, window_size=window_size, window_stride=window_stride, window=window, normalize=normalize, log=False, n_mels=40, preemCoef=0, melfloor=1.0)

    # Add zero padding to make all param with the same dims
    if param.shape[1] < max_len:
        pad = np.zeros((param.shape[0], max_len - param.shape[1]))
        param = np.hstack((pad, param))

    # If exceeds max_len keep last samples
    elif param.shape[1] > max_len:
        param = param[:, -max_len:]

    param = torch.FloatTensor(param)

    return param


class Loader(data.Dataset):
    """A google commands data set loader using Kaldi data format::

    Args:
        root (string): Kaldi directory path.
        transform (callable, optional): A function/transform that takes in a spectrogram
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the param to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_id (dict): Dict with items (class_name, class_index).
        wavs (list): List of (wavs path, class_index) tuples
        STFT parameters: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=99):
        classes, weight, class_to_id = get_classes()
        wavs = make_dataset(root, class_to_id)
        if not wavs:
            raise RuntimeError("Found 0 segments in '" + root + "'. Folder should be in standard Kaldi format")  # pylint: disable=line-too-long

        self.root = root
        self.wavs = wavs
        self.classes = classes
        self.weight = torch.FloatTensor(weight) if weight is not None else None
        self.class_to_idx = class_to_id
        self.transform = transform
        self.target_transform = target_transform
        self.loader = param_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (params, target) where target is class_index of the target class.
        """
        key, path, target = self.wavs[index]
        params = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)  # pylint: disable=line-too-long
        if self.transform is not None:
            params = self.transform(params)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return key, params, target

    def __len__(self):
        return len(self.wavs)
