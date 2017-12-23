# tf_voxelnet

A tensorflow inplementation for [voxelnet](https://arxiv.org/abs/1711.06396)

# Usage

0. have a look at `config.py` for model configurations, and make sure that you are using `Python3.5+`
1. run `setup.py` to build the Cython module
```bash
$ python setup.py build_ext --inplace
```
2. run `utils/preprocess.py` for pointcloud preprocess(attention to use correct path)
3. run `train.py`
4. launch a tensorboard and wait for the training process

# Data augmentation
run `data_aug.py`.

# Result

TBD

# TODO

- [X] nan and inf bugs fix
- [X] multicard support
- [X] data augmentation

# Acknowledgement

Thanks to[@ring00](https://github.com/ring00) for the implementation of VFE layer and **Jialin Zhao** for the implementation of the RPN
