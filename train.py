import torch
import random
import math
from env import Env
from dataloader import DataLoader
from agent import KeyframeExtractor
from liveplot import LivePlot
import numpy as np
import sys


class Trainer:
    def __init__(self, train_sample_len=72):
        self.env = Env()
        self.agent = KeyframeExtractor(gamma=1.0, eps=1.0, eps_factor=2000, learning_rate=0.0004,
                                       training_len=train_sample_len, mem_size=100000, batch_size=256, tau=(200, 1.0))

        self.steps = 0
        self.plot = LivePlot()
        self.data_loader = DataLoader()
        self.train_sample_len = train_sample_len

    def train(self, n_episodes):
        prev_results = np.zeros(100, dtype=np.float32)
        prev_results_idx = 0
        best_result = 0.0

        for i_episode in range(1, n_episodes + 1):
            num_keys = random.randint(self.train_sample_len // 12 + 2, self.train_sample_len // 4)
            self.env.new_motion(self.data_loader.sample_train(self.train_sample_len), num_keys)
            env_state, remaining_actions = self.env.get_state()

            state = torch.tensor(env_state, dtype=torch.float)

            done = False
            loss = 0.0
            score = 0.0

            use_eps = random.random() < 0.1 + 0.9 * math.exp(-1 * self.steps / 2000)

            if i_episode % 100 == 0:
                if best_result < prev_results.mean():
                    best_result = prev_results.mean()
                    self.agent.save_models()

            while not done:
                remain = torch.tensor(remaining_actions / state.shape[1]).view(1, 1)
                action = self.agent.select_action(state.unsqueeze(0), remain, self.env.get_keyframes()[1:-1],
                                                  use_eps=use_eps)
                next_state, next_remaining_actions, evals, done, result = self.env.step(action)
                reward = evals[2]

                concat_state = torch.cat([state, remain.view(1, 1).repeat(1, state.shape[1])], dim=0)
                concat_next_state = torch.cat([torch.tensor(next_state),
                                               torch.tensor(next_remaining_actions / state.shape[1]).view(1, 1).repeat(1, state.shape[1])], dim=0)

                self.agent.store_transition(concat_state, action, reward, concat_next_state, done)
                score = evals[1]
                prev_results[prev_results_idx] = score
                prev_results_idx = (prev_results_idx + 1) % 100

                if not done:
                    state = torch.tensor(next_state, dtype=torch.float)
                    remaining_actions = next_remaining_actions
                else:
                    loss = self.agent.optimise_model()

                self.steps += 1
            if not use_eps:
                self.plot.append(i_episode, score, loss)

            keyframe_mask = result == 1
            keyframe_idx = []
            for i in range(keyframe_mask.shape[0]):
                if keyframe_mask[i].item():
                    keyframe_idx.append(i)

            if i_episode % 200 == 0:
                self.plot.draw()

            print(keyframe_idx)
            print("Episode %d, Q-Loss: %.4f, Last Reward: %.4f" % (i_episode, loss, score))

        self.plot.save()
        self.plot.draw(block=True)


def input_error():
    print("Usage: python train.py [NUMBER OF EPISODES]")
    sys.exit()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        try:
            epochs = int(sys.argv[1])
        except ValueError:
            input_error()
    else:
        input_error()

    trainer = Trainer()
    trainer.train(epochs)
