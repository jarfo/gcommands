#!/bin/bash

audio_path=data/speech_commands/train/audio

mkdir -p $audio_path

n=`find $audio_path -name '*.wav' | wc -l`

pwd=`pwd`

if (( $n < 64726 )); then
  wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
  cd $audio_path
  tar -xvf $pwd/speech_commands_v0.01.tar.gz 
  cd $pwd
fi
