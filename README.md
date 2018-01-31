# VoxelNet-tensorflow

A tensorflow implementation for [VoxelNet](https://arxiv.org/abs/1711.06396).

# Usage

0. have a look at `config.py` for model configurations, and make sure that you are using `Python3.5+`.
1. run `setup.py` to build the Cython module.
```bash
$ python setup.py build_ext --inplace
```
2. run `train.py`. Some cmdline parameters is needed, just check `train.py` for them.
3. launch a tensorboard and wait for the training result.

# Data augmentation
Since [c928317](https://github.com/jeasinema/tf_voxelnet/commit/c928317169f1bf23e2157dab20cb402bddb8ffe0), data augmentation is done in an online manner, so there is no need for generating augmented samples.

# Result

TBD

# TODO

- [X] nan and inf bugs fix
- [X] multicard support
- [X] data augmentation

# Acknowledgement

Thanks to [@ring00](https://github.com/ring00) for the implementation of VFE layer and **Jialin Zhao** for the implementation of the RPN.
