---
kernelspec:
  display_name: Python 3
  language: python
  name: python3
language_info:
  codemirror_mode:
    name: ipython
    version: 3
  file_extension: .py
  mimetype: text/x-python
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
  version: 3.6.4
---



## ORIGINAL file tree from git repository

- clone git respository from https://github.com/allyoushawn/timit_gas
```
alvin@ai-labs-alvin:~/cei/git_repository/timit_gas$ tree -a -P '*sh' -I '.git|.backup'
.
├── local_rnn
├── prepare_timit
│   ├── align_dev.sh
│   ├── clean.sh
│   ├── gen_bounds.sh                 <---- major command
│   ├── gen_data.sh                   <---- major command (3)
│   ├── gen_noisy_data.sh
│   ├── material
│   ├── produce_label.sh
│   ├── script
│   │   ├── 00.train_lm.sh
│   │   ├── 01.format.sh
│   │   ├── 02.extract.feat.sh        <---- major command
│   │   ├── 03.mono.train.sh
│   │   ├── 04a.01.mono.mkgraph.sh
│   │   ├── 04a.02.mono.fst.sh
│   │   ├── 04b.mono.viterbi.sh
│   │   ├── 05.tree.build.sh
│   │   ├── 06.tri.train.sh
│   │   ├── 07a.01.tri.mkgraph.sh
│   │   ├── 07a.02.tri.fst.sh
│   │   ├── 07b.tri.viterbi.sh
│   │   ├── 08.mlp.train.sh
│   │   ├── 09.mlp.decode.sh
│   │   └── all.sh
│   ├── setup.sh                      <---- major command (1)
│   └── utility
│       ├── parse_options.sh
│       └── pretrain_dbn.sh
├── run.sh                            <---- major command
├── sh_tools
│   ├── decode_multi_model.sh
│   ├── extract_subset_data.sh
│   └── gen_wav_scp_from_corpus.sh    <---- major command (2)
└── utils
```

## SUMMARY

- STEP1 download TIMIT corpus
- STEP2 convert TIMIT corpus with lower case
- STEP3 install kaldi
- STEP4 sh_tool/gen_wav_scp_from_corpus.sh
- copy STEP4 output to prepare_limit
- STEP5 prepare_timit/gen_data.sh
- How to run AE-GAS by ./run.sh
  - code modification
    - /timit_gas/local_rnn/local_cell.py
    - ~/conda/lib/python3.6/site-packages/tensorflow/python/ops/rnn_cell_impl.py

## STEP 1 download TIMIT corpus

- download corpus from http://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3
- create a symblic link timit_corpus to TIMIT corpus
- create a empty folder with name timit_feature for variance feat_loc in ./run.sh

## STEP 2 preprocess TIMIT with lowcase

TIMIT corpus is upper case format, in this code, all TIMIT corpus content is handle by low case,
under timit_corpus folder, do the following two command
```
- for f in `find ./ -type d -maxdepth 3`; do mv -v "$f" "`echo $f | tr '[A-Z]' '[a-z]'`"; done 
- for f in `find ./ -type f `; do mv -v "$f" "`echo $f | tr '[A-Z]' '[a-z]'`"; done
```

alvin@ai-labs-alvin:~/cei/notebook_home/ai_lab_sound/timit_gas$ tree -a -P timit_corpus --matchdirs -l -L 3 -I

```
.
├── local_rnn
├── prepare_timit
├── sh_tools
├── timit_corpus -> /media/alvin/disk_D/dataset/TIMIT   <--- create symbolic link here
│   ├── DOC
│   ├── test                                            <--- 1600 wav under test forlder
│   │   ├── dr1                                         <--- 110
│   │   │   ├── faks0 
│   │   │   ├── fdac1
.   .   .   .   ...
│   │   │   └── ... 
│   │   ├── dr2                                         <--- 260
│   │   ├── dr3                                         <--- 260
│   │   ├── dr4                                         <--- 320
│   │   ├── dr5                                         <--- 280
│   │   ├── dr6                                         <--- 110
│   │   ├── dr7                                         <--- 230
│   │   └── dr8                                         <--- 110
│   └── train                                           <--- 4620 wav under train folder
│       ├── dr1                                         <--- 380
│       ├── dr2                                         <--- 760
│       ├── dr3                                         <--- 760
│       ├── dr4                                         <--- 680
│       ├── dr5                                         <--- 700
│       ├── dr6                                         <--- 350
│       ├── dr7                                         <--- 770
│       └── dr8                                         <--- 220
├── timit_feature
└── utils
```

## STEP 3 install kaldi

```
1 kaldi/tools$ sudo apt-get install libatlas3-base
2 kaldi/tools$ sudo apt-get install automake autoconf libtool subversion
3 kaldi/tools$ make
  * (option) kaldi/tools/extras$ ./check_dependencies 
4 tools/extras$ ./install_irstlm.sh
5 kaldi/src$ ./configure
6 kaldi/src$ make depend
7 kaldi/src$ make

alvin@ai-labs-alvin:~/cei/notebook_home/ai_lab_sound/timit_gas$ tree -d -L 1

.
├── kaldi                                               <--- kaldi root
├── local_rnn
├── prepare_timit
├── sh_tools
├── timit_corpus -> /media/alvin/disk_D/dataset/TIMIT
├── timit_feature
└── utils
```

## STEP 4 prepare_timit/setup.sh
```shell
#specify the followings
kaldi_root=/home/alvin/cei/notebook_home/ai_lab_sound/timit_gas/kaldi
timit_root=/home/alvin/cei/notebook_home/ai_lab_sound/timit_gas/timit_corpus
export feat_loc=/home/alvin/cei/notebook_home/ai_lab_sound/timit_gas/timit_feature
```

alvin@ai-labs-alvin:~/cei/notebook_home/ai_lab_sound/timit_gas$ tree -a -L 2 -P 'prepare_timit' --matchdirs -I '.vscode|.ipynb*'
```
.
├── kaldi
├── local_rnn
├── prepare_timit
│   ├── align_dev.sh
│   ├── clean.sh
│   ├── gen_bounds.sh
│   ├── gen_data.sh
│   ├── gen_noisy_data.sh
│   ├── material
│   ├── produce_label.sh
│   ├── script
│   ├── setup.sh                          <--- update path infomration 
│   └── utility
├── sh_tools
├── timit_feature
└── utils
```

## STEP 5 sh_tool/gen_wav_scp_from_corpus.sh

alvin@ai-labs-alvin:~/cei/notebook_home/ai_lab_sound/timit_gas$ tree -a -L 2 -P 'sh_tools' --matchdirs -I '.vscode|.ipynb*'

```
.
├── kaldi
├── local_rnn
├── prepare_timit
├── sh_tools
│   ├── decode_multi_model.sh
│   ├── extract_subset_data.sh
│   ├── gen_wav_scp_from_corpus.1.sh      <--- update version
│   ├── gen_wav_scp_from_corpus.sh        <--- original version
│   ├── test.wav.scp                      <--- output file
│   └── train.wav.scp                     <--- output file
├── timit_feature
└── utils

```

* the modification point for this file is
  1. path update for variance timit 
  2. path update for variance kaldi_tool_header
  3. seg[8] --> seg[10] , # number 8 is depended on '/' number for full kaldi path
  4. seg[9] --> seg[11] , # number 9 is depended on '/' number for full kaldi path

``` shell
#!/bin/bash

#timit=/media/hdd/csie/corpus/timit
#kaldi_tool_header="/home/allyoushawn/Kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav"

timit=/home/alvin/cei/notebook_home/ai_lab_sound/timit_gas/timit_corpus
kaldi_tool_header="/home/alvin/cei/notebook_home/ai_lab_sound/timit_gas/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav"

for target in train test; do
    rm -f ${target}.wav.scp
    dir=${timit}/${target}
    for file in $(find $dir | grep .wav); do

        while IFS='/' read -ra seg; do
          while IFS='.' read -ra seg2; do
            # echo "${seg[8]}_${seg2[0]} $kaldi_tool_header $file |" >>${target}.wav.scp
            echo "${seg[10]}_${seg2[0]} $kaldi_tool_header $file |" >>${target}.wav.scp
            # done <<< ${seg[9]}
          done <<< ${seg[11]}
        done <<< $file
    done
done
```

## STEP 6 prepare_timit/gen_data.sh

* copy STEP4 output(test.wav.scp, train.wav.scp) to ./prepare_timit
* under ./prepare_timit, exectue ./gen_data


alvin@ai-labs-alvin:~/cei/notebook_home/ai_lab_sound/timit_gas$  tree -fdl | fgrep -v 'train' | fgrep -v 'test' | fgrep -v 'kaldi/' | fgrep -v 'DOC' | sed -r 's%\./.*/%%' | sed -r 's%\./%%'

``` shell
.
├── bounds                <--- created by ./prepare_timie/gen_data.sh 
│   └── phn               <--- created by ./prepare_timie/gen_data.sh 
├── feature_scp           <--- created by ./prepare_timie/gen_data.sh 
├── kaldi
├── local_rnn
├── prepare_timit
│   ├── material
│   ├── script
│   └── utility
├── sh_tools
├── timit_corpus -> /media/alvin/disk_D/dataset/TIMIT
├── timit_feature
└── utils


feature_scp               bounds
├── dev.scp               └── phn
├── test.scp                  ├── dev.phn
└── train.scp                 ├── test.phn
                              └── train.phn
```

## EXPECTED TREE STRUCT

```
.
├── bounds                                              <--- generated by STEP6 
│   └── phn                                             <--- generated by STEP6
│       ├── phn/dev.phn                                 <--- generated by STEP6
│       ├── phn/test.phn                                <--- generated by STEP6
│       └── phn/train.phn                               <--- generated by STEP6  
├── feature_scp                                         <--- generated by STEP6
│   ├── dev.scp                                         <--- generated by STEP6
│   ├── test.scp                                        <--- generated by STEP6
│   └── train.scp                                       <--- generated by STEP6
├── kaldi                                               <--- STEP3
├── local_rnn
├── prepare_timit
│   ├── align_dev.sh
│   ├── clean.sh
│   ├── gen_bounds.sh
│   ├── gen_data.sh
│   ├── gen_noisy_data.sh
│   ├── material
│   ├── produce_label.sh
│   ├── script
│   ├── setup.sh                                        <--- update from STEP4
│   ├── utility
│   ├── test.wav.scp                                    <--- manually copy after STEP 5
│   └── train.wav.scp                                   <--- manually copy after STEP 5
├── run.sh
├── sh_tools
│   ├── decode_multi_model.sh
│   ├── extract_subset_data.sh
│   ├── gen_wav_scp_from_corpus.1.sh                    <--- updated on STEP5
│   ├── gen_wav_scp_from_corpus.sh
│   ├── test.wav.scp                                    <--- generated by STEP5
│   └── train.wav.scp                                   <--- generated by STEP5
├── timit_corpus -> /media/alvin/disk_D/dataset/TIMIT   <--- prepare symbolic link on STEP2, **lower case
│   ├── test
│   └── train
├── timit_feature                                       <--- generated by STEP6
│   ├── test.13.scp                                     <--- generated by STEP6
│   ├── test.39.cmvn.scp                                <--- generated by STEP6
│   ├── test.39.scp                                     <--- generated by STEP6
│   ├── train.13.scp                                    <--- generated by STEP6
│   ├── train.39.cmvn.scp                               <--- generated by STEP6
│   └── train.39.scp                                    <--- generated by STEP6
└── utils

```

## how run AE-GAS (Auto-Encoder Gate Activation Signal) by ./run.sh

### code modification

* original timit_gas code from github is wrote by tensorflow 1.1, need modify 2 patchs before perform ./run.sh for tensorflow 1.8
  > - ./timit_gas/local_rnn/local_cell.py
  > - ~/conda/lib/python3.6/site-packages/tensorflow/python/ops/rnn_cell_impl.py

### ./timit_gas/local_rnn/local_cell.py
***
change from 
```python
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
```
to
```python
from tensorflow.python.ops.rnn_cell_impl import RNNCell as RNNCell
```

### ~/conda/lib/python3.6/site-packages/tensorflow/python/ops/rnn_cell_impl.py
***
  > - add the following code to rnn_cell_impl.py

---
```python
def _state_size_with_prefix(state_size, prefix=None):
    """Helper function that enables int or TensorShape shape specification.
    This function takes a size specification, which can be an integer or a
    TensorShape, and converts it into a list of integers. One may specify any
    additional dimensions that precede the final state size specification.
    Args:
      state_size: TensorShape or int that specifies the size of a tensor.
      prefix: optional additional list of dimensions to prepend.
    Returns:
      result_state_size: list of dimensions the resulting tensor size.
    """
    result_state_size = tensor_shape.as_shape(state_size).as_list()
    if prefix is not None:
      if not isinstance(prefix, list):
        raise TypeError("prefix of _state_size_with_prefix should be a list.")
      result_state_size = prefix + result_state_size
    return result_state_size
```
