import math

import torch
import random
from model import GKEN
import numpy as np

NUM_BONES = 23


class ReplayMemory(object):
    def __init__(self, capacity, state_size):
        self.capacity = capacity
        self.ctr = 0
        self.full = False

        self.states = np.zeros((self.capacity, *state_size), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, *state_size), dtype=np.float32)

        self.actions = np.zeros(self.capacity, dtype=np.int8)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.bool_)

    def push(self, state, action, reward, next_state, done):
        idx = self.ctr % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        self.ctr += 1

    def sample(self, batch_size):
        total = min(self.ctr, self.capacity)

        batch = np.random.choice(total, batch_size, replace=False)

        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        next_states = self.next_states[batch]
        dones = self.dones[batch]

        return states, actions, rewards, next_states, dones


class KeyframeExtractor(object):
    def __init__(self, gamma, eps, eps_factor, learning_rate, training_len, mem_size, batch_size, tau, save_dir='dqn'):
        self.GAMMA = gamma
        self.eps = eps
        self.EPS_MAX = eps
        self.lr = learning_rate
        self.training_len = training_len
        self.batch_size = batch_size

        self.EPS_MIN = 0.1
        self.EPS_DEC = 1e-3
        self.EPS_FACTOR = eps_factor

        self.TAU = tau
        self.save_dir = save_dir

        self.steps = 0
        self.input_size = NUM_BONES * 6 + 2

        self.memory = ReplayMemory(mem_size, (self.input_size, training_len))

        self.device = torch.device("cuda")

        self.policy_net = GKEN().to(self.device)
        self.target_net = GKEN().to(self.device)
        self.optimiser = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = torch.nn.L1Loss()
        self.target_net.eval()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def sample_memory(self):
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        states = torch.tensor(state)
        actions = torch.tensor(action, dtype=torch.long)
        rewards = torch.tensor(reward)
        next_states = torch.tensor(next_state)
        dones = torch.tensor(done)

        return states, actions, rewards, next_states, dones

    def select_action(self, state, remaining_actions, keyframe_mask, use_eps=True):
        if use_eps is False or random.random() > self.eps:
            pred = self.policy_net.forward(state, remaining_actions)
            for i in range(keyframe_mask.shape[0]):
                if keyframe_mask[i] == 1:
                    pred[:, i] = float("-inf")

            action = torch.argmax(pred).item()
        else:
            valid = []
            for i in range(keyframe_mask.shape[0]):
                if keyframe_mask[i] == 0:
                    valid.append(i)
            action = random.sample(valid, 1)[0]
        return action

    def replace_target(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.TAU[1] * policy_param.data + (1.0 - self.TAU[1]) * target_param.data)
        self.target_net.eval()

    def decrement_epsilon(self):
        self.eps = self.EPS_MIN + (self.EPS_MAX - self.EPS_MIN) * math.exp(-1 * self.steps / self.EPS_FACTOR)

    def decrement_lr(self):
        for g in self.optimiser.param_groups:
            g['lr'] = self.lr / 10 + (self.lr * 9 / 10) * math.exp(-1 * self.steps / self.EPS_FACTOR)

    def optimise_model(self):
        if self.memory.ctr < self.batch_size:
            return 0

        states, actions, rewards, next_states, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        q_pred = self.policy_net(states[:, :-1], states[:, -1, 0])[indices, actions]
        q_next = self.target_net(next_states[:, :-1], next_states[:, -1, 0]).detach()
        q_eval = self.policy_net(next_states[:, :-1], next_states[:, -1, 0])

        max_actions = torch.argmax(q_eval, dim=1)
        q_next[dones] = float(0.0)

        q_target = rewards.to(self.device) + self.GAMMA * q_next[indices, max_actions]
        loss = self.criterion(q_target, q_pred)

        self.optimiser.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimiser.step()

        self.steps += 1
        self.decrement_epsilon()
        self.decrement_lr()

        if self.steps % self.TAU[0] == 0:
            print("Replacing")
            self.replace_target()

        return loss.item()

    def save_models(self):
        torch.save({
            'policy_model_state_dict': self.policy_net.state_dict(),
            'target_model_state_dict': self.target_net.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
        }, "./model.pt")

    def load_models(self, steps=0):
        self.policy_net = GKEN().to(self.device)
        self.target_net = GKEN().to(self.device)
        self.optimiser = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        checkpoint = torch.load("./model.pt")

        self.policy_net.load_state_dict(checkpoint['policy_model_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

        self.target_net.eval()

        self.steps = steps
