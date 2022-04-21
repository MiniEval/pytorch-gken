# Spatial Representation Helper

import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation

NUM_BONES = 23


class CMUSpatialHelper:

    class Bone:
        def __init__(self):
            self.parent = None
            self.local_rot = None
            self.local_pos = None
            self.global_transform = None

        def set(self, offset, rotation_data, position_data=None, parent=None):
            self.parent = parent
            self.local_rot = rotation_data
            self.local_pos = np.expand_dims(offset, axis=0).repeat(self.local_rot.shape[0], axis=0)
            if position_data is not None:
                self.local_pos += position_data

            self.global_transform = None

        def get_transform_matrix(self):
            rotation = np.deg2rad(self.local_rot)
            location = self.local_pos

            vector = np.zeros((self.local_rot.shape[0], 4, 4), dtype=np.float32)

            r = Rotation.from_euler("ZYX", rotation[:, [2, 1, 0]])
            vector[:, :3, :3] = r.as_matrix()
            vector[:, :3, 3] = location

            vector[:, 3, 3] = 1

            return vector

        def get_global_transform(self):
            if self.parent:
                transform = self.parent.global_transform
            else:
                transform = torch.eye(4).unsqueeze(0).repeat(self.local_rot.shape[0], 1, 1)

            transform_matrix = torch.from_numpy(self.get_transform_matrix()).float()
            self.global_transform = torch.bmm(transform, transform_matrix)

            return self.global_transform

        def get_global_position(self):
            global_transform = self.get_global_transform()

            position_vector = torch.zeros(global_transform.shape[0], 4, 1, dtype=torch.float32)
            position_vector[:, 3] = 1

            return torch.bmm(global_transform, position_vector).numpy()

    def __init__(self):
        self.CMU_OFFSETS = np.zeros((23, 3), dtype=np.float32)
        self.CMU_OFFSETS[2] = np.array([0.02827, 2.03559, -0.19338])
        self.CMU_OFFSETS[3] = np.array([0.05672, 2.04885, -0.04275])
        self.CMU_OFFSETS[5] = np.array([-0.05417, 1.74624, 0.17202])
        self.CMU_OFFSETS[6] = np.array([0.10407, 1.76136, -0.12397])
        self.CMU_OFFSETS[8] = np.array([3.36241, 1.20089, -0.31121])
        self.CMU_OFFSETS[9] = np.array([4.98300, 0, 0])
        self.CMU_OFFSETS[10] = np.array([3.48356, 0, 0])
        self.CMU_OFFSETS[12] = np.array([-3.13660, 1.37405, -0.40465])
        self.CMU_OFFSETS[13] = np.array([-5.24190, 0, 0])
        self.CMU_OFFSETS[14] = np.array([-3.44417, 0, 0])
        self.CMU_OFFSETS[15] = np.array([1.36306, -1.79463, 0.83929])
        self.CMU_OFFSETS[16] = np.array([2.44811, -6.72613, 0])
        self.CMU_OFFSETS[17] = np.array([2.56220, -7.03959, 0])
        self.CMU_OFFSETS[18] = np.array([0.15764, -0.43311, 2.32255])
        self.CMU_OFFSETS[19] = np.array([-1.30552, -1.79463, 0.83929])
        self.CMU_OFFSETS[20] = np.array([-2.54253, -6.98555, 0])
        self.CMU_OFFSETS[21] = np.array([-2.56826, -7.05623, 0])
        self.CMU_OFFSETS[22] = np.array([-0.16473, -0.45259, 2.36315])

        self.bones = [self.Bone() for _ in range(NUM_BONES)]

    # Motion data in shape [ATTRIBUTES, FRAMES]
    def get_spatial_representation(self, motion_data):
        data = np.transpose(motion_data, (1, 0))
        self.bones[0].set(self.CMU_OFFSETS[0], data[:, 3:6], position_data=data[:, :3])
        self.bones[1].set(self.CMU_OFFSETS[1], data[:, 6:9], parent=self.bones[0])
        self.bones[2].set(self.CMU_OFFSETS[2], data[:, 9:12], parent=self.bones[1])
        self.bones[3].set(self.CMU_OFFSETS[3], data[:, 12:15], parent=self.bones[2])
        self.bones[4].set(self.CMU_OFFSETS[4], data[:, 15:18], parent=self.bones[3])
        self.bones[5].set(self.CMU_OFFSETS[5], data[:, 18:21], parent=self.bones[4])
        self.bones[6].set(self.CMU_OFFSETS[6], data[:, 21:24], parent=self.bones[5])
        self.bones[7].set(self.CMU_OFFSETS[7], data[:, 24:27], parent=self.bones[3])
        self.bones[8].set(self.CMU_OFFSETS[8], data[:, 27:30], parent=self.bones[7])
        self.bones[9].set(self.CMU_OFFSETS[9], data[:, 30:33], parent=self.bones[8])
        self.bones[10].set(self.CMU_OFFSETS[10], data[:, 33:36], parent=self.bones[9])
        self.bones[11].set(self.CMU_OFFSETS[11], data[:, 36:39], parent=self.bones[3])
        self.bones[12].set(self.CMU_OFFSETS[12], data[:, 39:42], parent=self.bones[11])
        self.bones[13].set(self.CMU_OFFSETS[13], data[:, 42:45], parent=self.bones[12])
        self.bones[14].set(self.CMU_OFFSETS[14], data[:, 45:48], parent=self.bones[13])
        self.bones[15].set(self.CMU_OFFSETS[15], data[:, 48:51], parent=self.bones[0])
        self.bones[16].set(self.CMU_OFFSETS[16], data[:, 51:54], parent=self.bones[15])
        self.bones[17].set(self.CMU_OFFSETS[17], data[:, 54:57], parent=self.bones[16])
        self.bones[18].set(self.CMU_OFFSETS[18], data[:, 57:60], parent=self.bones[17])
        self.bones[19].set(self.CMU_OFFSETS[19], data[:, 60:63], parent=self.bones[0])
        self.bones[20].set(self.CMU_OFFSETS[20], data[:, 63:66], parent=self.bones[19])
        self.bones[21].set(self.CMU_OFFSETS[21], data[:, 66:69], parent=self.bones[20])
        self.bones[22].set(self.CMU_OFFSETS[22], data[:, 69:72], parent=self.bones[21])

        representation = []

        for bone in self.bones:
            representation.append(bone.get_global_position()[:, :3])

        # Output shape: [FRAMES, BONES, XYZ]
        return np.stack(representation, axis=1)
