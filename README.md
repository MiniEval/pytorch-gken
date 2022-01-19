# Graph Keyframe Evaluation Network - PyTorch implementation
Official Pytorch implementation of the ACM MM'21 paper: [Keyframe Extraction from Motion Capture Sequences with Graph based Deep Reinforcement Learning]{https://dl.acm.org/doi/10.1145/3474085.3475635}

# Prerequisites
- [Python 3.9](https://www.python.org/)
- [PyTorch 3.9](https://github.com/pytorch/pytorch) with CUDA
- [PyTorch Geometric 1.7.2](https://github.com/rusty1s/pytorch_geometric) with CUDA
- [NumPy v1.21.0](https://github.com/numpy/numpy)
- [Matplotlib 3.4.2](https://github.com/matplotlib/matplotlib)

# Usage
## Training
To train the model, download our preprocessed CMU Mocap dataset from the [releases](https://github.com/MiniEval/pytorch-gken/releases/tag/1) page and place the `/train_data` and `/test_data` folders in the repository root. `train.py` is used as follows:

`python train.py [NUMBER OF EPISODES]`

The paper uses 5000 episodes to train the model.

We update two Matplotlib diagrams per 200 episodes. The scatter plot displays <img src="https://render.githubusercontent.com/render/math?math=R_1"> rewards over time, while the line chart displays Q-loss over time.


### Custom datasets
We provide a `CMU_parse(file, start=1, frame_skip=5)` function in `bvh.py`. This function properly formats motion capture data from the [BVH conversion of the CMU Mocap Dataset[(https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture) into a NumPy tensor. The NumPy tensor can be saved as a file that is used by our `dataloader.py` The default parameters of `CMU_parse` scale the BVH file down from 120FPS to 24FPS.

## Evaluation
To evaluate with our model, download our pre-trained model from the [releases](https://github.com/MiniEval/pytorch-gken/releases/tag/1) page and place `model.pt` in the repository root. Alternatively, you may train your own model with `train.py`. 

The evaluation accepts BVH files as input, using the CMU Mocap skeleton format. `eval.py` is used as follows:

`python eval.py [BVH FILE] [NUMBER OF KEYFRAMES]`

# Human annotations
In the [releases](https://github.com/MiniEval/pytorch-gken/releases/tag/1) page, we provide five sets of human annotations in `Keyframe Extraction - Demonstration.blend`, which can be opened using [Blender](https://www.blender.org/).

# Acknowledgements
The motion capture data used in this project was obtained from [mocap.cs.cmu.edu](https://mocap.cs.cmu.edu). The database was created with funding from the American National Science Foundation, under EIA-0196217.

The BVH file parsing module was written by [20tab srl](https://github.com/20tab/bvh-python).
