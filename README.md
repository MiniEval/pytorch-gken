# Graph Keyframe Evaluation Network - PyTorch implementation
Official Pytorch implementation of the ACM MM'21 paper: Keyframe Extraction from Motion Capture Sequences with Graph based Deep Reinforcement Learning

# Prerequisites
- [Python 3.9](https://www.python.org/)
- [PyTorch 3.9](https://github.com/pytorch/pytorch) with CUDA
- [PyTorch Geometric 1.7.2](https://github.com/rusty1s/pytorch_geometric) with CUDA
- [NumPy v1.21.0](https://github.com/numpy/numpy)
- [Matplotlib 3.4.2](https://github.com/matplotlib/matplotlib)

# Usage
## Training
To train the model, download our preprocessed CMU Mocap dataset from the releases page and place the `/train_data` and `/test_data` folders in the repository root. `train.py` is used as follows:

`python train.py [NUMBER OF EPISODES]`

The paper uses 5000 episodes to train the model.

We update two Matplotlib diagrams per 200 episodes. The scatter plot displays <img src="https://render.githubusercontent.com/render/math?math=R_1"> rewards over time, while the line chart displays Q-loss over time.


### Custom datasets
We provide a `CMU_parse(file, start=1, frame_skip=5)` function in `bvh.py`. This function properly formats motion capture data from the BVH conversion of the CMU Mocap Dataset into a NumPy tensor. The NumPy tensor can be saved as a file that is used by our `dataloader.py` The default parameters of `CMU_parse` scale the BVH file down from 120FPS to 24FPS.

## Evaluation
To evaluate with our model, download our pre-trained model from the releases page and place `model.pt` in the repository root. Alternatively, you may train your own model with `train.py`. 

The evaluation accepts BVH files as input, using the CMU Mocap skeleton format. `eval.py` is used as follows:

`python eval.py [BVH FILE] [NUMBER OF KEYFRAMES]`

# Human annotations

In the releases page, we provide five sets of human annotations in `Keyframe Extraction - Demonstration.blend`, which can be opened using [Blender](https://www.blender.org/).
