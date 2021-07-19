import sys
import os
import torch

from agent import KeyframeExtractor
from env import Env
from bvh import CMU_parse


def input_error():
    print("Usage: python eval.py [BVH FILE] [NUMBER OF KEYFRAMES]")
    sys.exit()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        if not os.path.isfile(sys.argv[1]):
            print("BVH file", sys.argv[1], "cannot be found.")
            sys.exit()
        try:
            num_keys = int(sys.argv[2])
        except ValueError:
            input_error()
    else:
        input_error()

    if not os.path.isfile("./model.pt"):
        print("Trained model cannot be found. Please download a pre-trained model, or train a new model using train.py")
        sys.exit()

    print("Processing BVH file...", end="")
    with open(sys.argv[1], "r") as f:
        bvh_file = f.read()
        f.close()
    bvh_input = CMU_parse(bvh_file)
    print("done!")

    print("Loading GKEN model...", end="")
    agent = KeyframeExtractor(gamma=0.99, eps=1.0, eps_factor=2000, learning_rate=0.0004, training_len=72, mem_size=1,
                              batch_size=1, tau=(200, 1.0))
    agent.load_models()
    agent.policy_net.eval()
    print("done!")

    print("Evaluating keyframes...")
    env = Env(bvh_input, num_keys)
    env_state, remaining_actions = env.get_state()

    done = False
    state = torch.tensor(env_state, dtype=torch.float)
    while not done:
        remain = torch.tensor(remaining_actions / state.shape[1]).view(1, 1)
        action = agent.select_action(state.unsqueeze(0), remain, env.get_keyframes()[1:-1], use_eps=False)
        next_state, next_remaining_actions, _, done, result = env.step(action)
        if not done:
            state = torch.tensor(next_state, dtype=torch.float)
            remaining_actions = next_remaining_actions

    keyframe_mask = result == 1
    keyframe_idx = []
    for k in range(keyframe_mask.shape[0]):
        if keyframe_mask[k].item():
            keyframe_idx.append(k)
    print("done!")

    print("\n24 FPS Keyframes:", keyframe_idx)
