import numpy as np
from spatialhelper import CMUSpatialHelper
from interp import lerp


class Env:
    def __init__(self, motion_data=None, n_keys=None):
        self.spatial_helper = CMUSpatialHelper()

        self.original = None
        self.spatial_original = None

        self.original_mean = None
        self.original_std = None
        self.recon = None
        self.spatial_recon = None
        self.keyframes = None

        self.remaining_actions = 0
        self.total_keys = 0
        self.base_score = 0.0

        if motion_data is not None and n_keys is not None:
            self.new_motion(motion_data, n_keys)

    def _normalise(self, x):
        return (x - self.original_mean) / self.original_std

    def _update_recon(self, key=None):
        keyframe_mask = self.keyframes == 1
        keyframe_idx = []
        for i in range(keyframe_mask.shape[0]):
            if keyframe_mask[i].item():
                keyframe_idx.append(i)

        if key is None:
            self.recon = np.zeros(self.original.shape)
            for i in range(len(keyframe_idx) - 1):
                self.recon[:, keyframe_idx[i]:keyframe_idx[i + 1] + 1] = lerp(self.original[:, keyframe_idx[i]],
                                                                              self.original[:, keyframe_idx[i+1]],
                                                                              keyframe_idx[i],
                                                                              keyframe_idx[i+1])
        else:
            idx = keyframe_idx.index(key)
            for i in range(idx - 1, idx + 1):
                self.recon[:, keyframe_idx[i]:keyframe_idx[i + 1] + 1] = lerp(self.original[:, keyframe_idx[i]],
                                                                              self.original[:, keyframe_idx[i+1]],
                                                                              keyframe_idx[i],
                                                                              keyframe_idx[i+1])

        return self.recon

    # Motion data in shape [ATTRIBUTES, FRAMES]
    def new_motion(self, motion_data, n_keys):
        if n_keys < 3:
            raise ValueError("Must provide at least 3 keys")
        if motion_data.shape[1] < 3:
            raise ValueError("Motion must have at least 3 frames")
        if n_keys > motion_data.shape[1]:
            raise ValueError("Motion must have more frames than keys")

        self.original = motion_data

        self.spatial_original = self.spatial_helper.get_spatial_representation(self.original)

        self.original_mean = np.mean(self.spatial_original, (0, 1, 3), keepdims=True)
        self.original_std = np.std(self.spatial_original - self.original_mean)

        self.spatial_original = self._normalise(self.spatial_original)

        self.keyframes = None
        self.keyframes = np.zeros(self.original.shape[1])
        self.keyframes[0] = 1
        self.keyframes[-1] = 1

        self.remaining_actions = n_keys - 2

        self._update_recon()
        self.spatial_recon = self._normalise(self.spatial_helper.get_spatial_representation(self.recon))
        self.base_score = self._get_score() + np.finfo(np.float32).tiny
        self.total_keys = n_keys

    def _get_score(self):
        return np.mean(np.sqrt(np.sum((self.spatial_original - self.spatial_recon) ** 2, axis=2)))

    def step(self, action):
        old_loss = np.arctanh(max(1 - (self._get_score() / self.base_score), -1 + 1e-6))
        self.keyframes[action + 1] = 1
        self._update_recon(action + 1)

        self.remaining_actions -= 1

        done = self.remaining_actions == 0

        self.spatial_recon = self._normalise(self.spatial_helper.get_spatial_representation(self.recon))
        score = self._get_score()
        reward = np.arctanh(max(1 - (score / self.base_score), -1 + 1e-6)) - old_loss

        next_state, _ = self.get_state()

        return next_state, self.remaining_actions, (score, 1 - (score / self.base_score), reward), done, self.keyframes

    def get_state(self):
        state = list()
        state.append(self.spatial_original.reshape(self.spatial_original.shape[0], -1).transpose((1, 0)))
        state.append(self.spatial_recon.reshape(self.spatial_original.shape[0], -1).transpose((1, 0)))
        state.append(np.expand_dims(self.keyframes, 0))

        return np.concatenate(state, 0), self.remaining_actions

    def get_keyframes(self):
        return self.keyframes
