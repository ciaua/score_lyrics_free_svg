Score- and Lyrics-Free Singing Voice Generation
===============================================

This repository contains the generation code for our paper "Score and Lyrics-Free Singing Voice Generation". The goal is to generate singing voices unconditionally or given an accompaniment track.

You can find the paper at: [Score- and Lyrics-Free Singing Voice Generation](https://arxiv.org/abs/1912.11747)

The model is implemented with PyTorch. 

Note that we have a newer version for unconditional audio generation. You can find the repo at: [Unconditional Audio Generation with GAN and Cycle Regularization](https://github.com/ciaua/unagan).


## Getting Started


### Install requirements

```
pip install -r requirements.txt
```


### Generate singing voices

### Accompanied singing
```
python generate_singing.py --condition 5 --cuda_id 0
```

### Free singing
```
python generate_singing.py --condition 0 --cuda_id 0
```

## Vocoder
For converting mel-spectrograms to audios, we use the an [WaveRNN](https://arxiv.org/abs/1802.08435) version that is implemented by [fatchord](https://github.com/fatchord/WaveRNN).


### Training scripts
The training scripts will need training data which are not included in this repository.

```
./scripts/train.accompanied_singer.py
./scripts/train.free_singer.py
```
