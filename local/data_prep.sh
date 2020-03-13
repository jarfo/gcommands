#!/bin/bash

segments=true

. path.sh

if [ $# != 2 ]; then
  echo "Usage: data_prep.sh /path/to/database local_dir"
  exit 1;
fi

data_dir=$1
name=`basename $data_dir`
local_dir=$2
mkdir -p $local_dir

# Audio data directory check
if [ ! -d $data_dir/audio ]; then
  echo "Error: data_prep.sh requires a directory argument with a subdir named audio"
  exit 1;
fi

# find audio files
find $data_dir/audio -name *.wav | \
  awk -F "/" '{print $(NF)"_"$(NF-1), $0}' | \
  sed 's|.wav||' | sort \
  > $local_dir/wav.scp

# create utt2spk file
cut -d' ' -f 1 $local_dir/wav.scp | \
  awk -F "_" '{print $0, $1}' \
  > $local_dir/utt2spk

if [ "$segments" = "true" ]; then
  # create temporal utt2dur file
  utils/data/get_utt2dur.sh $local_dir

  utils/data/get_segments_for_data.sh $local_dir > $local_dir/segments
  python utils/data/get_uniform_subsegments.py \
    --max-segment-duration=1 --overlap-duration=0 --max-remaining-duration=0 \
    $local_dir/segments \
    | grep -v nohash \
    | shuf \
    > $local_dir/sil_segments
    grep nohash $local_dir/segments > $local_dir/segments0
    cat $local_dir/sil_segments >> $local_dir/segments0
    sort $local_dir/segments0 > $local_dir/segments

  # create text file
  cut -d' ' -f 1 $local_dir/segments | \
    awk '{b = gensub(/_-.*/, "", "g", $0); n = split(b,A,"_"); print $0, A[n]}' \
    > $local_dir/text

  # update utt2spk file
  cut -d' ' -f 1 $local_dir/segments | \
    awk -F "_" '{print $0, $1}' \
    > $local_dir/utt2spk

  # update utt2dur
  utils/data/get_utt2dur.sh $local_dir

else
  # create text file
  cut -d' ' -f 1 $local_dir/wav.scp | \
    awk '{b = gensub(/_$/, "", "g", $0); n = split(b,A,"_"); print $0, A[n]}' \
    > $local_dir/text
fi

utils/utt2spk_to_spk2utt.pl \
  < $local_dir/utt2spk > $local_dir/spk2utt || exit 1;

utils/validate_data_dir.sh --no-feats $local_dir

for list in testing validation; do
  cat $data_dir/audio/${list}_list.txt | \
    awk -F "/" '{print $(NF)"_"$(NF-1)}' | \
    sed 's|.wav||' | sort \
  > $local_dir/$list
done

if [ "$segments" = "true" ]; then
  sort $local_dir/segments $local_dir/validation $local_dir/testing | cut -d' ' -f 1 | \
    uniq -u \
    > $local_dir/training
else
  sort $local_dir/wav.scp $local_dir/validation $local_dir/testing | cut -d' ' -f 1 | \
    uniq -u \
    > $local_dir/training
fi

for list in testing validation training; do
  subset_data_dir.sh --utt-list $local_dir/$list $local_dir ${local_dir}_${list}
  utils/validate_data_dir.sh --no-feats ${local_dir}_${list}
done
