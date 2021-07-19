# BVH file parser from https://github.com/20tab/bvh-python

import re
import numpy as np


class BvhNode:

    def __init__(self, value=[], parent=None):
        self.value = value
        self.children = []
        self.parent = parent
        if self.parent:
            self.parent.add_child(self)

    def add_child(self, item):
        item.parent = self
        self.children.append(item)

    def filter(self, key):
        for child in self.children:
            if child.value[0] == key:
                yield child

    def __iter__(self):
        for child in self.children:
            yield child

    def __getitem__(self, key):
        for child in self.children:
            for index, item in enumerate(child.value):
                if item == key:
                    if index + 1 >= len(child.value):
                        return None
                    else:
                        return child.value[index + 1:]
        raise IndexError('key {} not found'.format(key))

    def __repr__(self):
        return str(' '.join(self.value))

    @property
    def name(self):
        return self.value[1]


class Bvh:

    def __init__(self, data):
        self.data = data
        self.root = BvhNode()
        self.frames = []
        self.tokenize()

    def tokenize(self):
        first_round = []
        accumulator = ''
        for char in self.data:
            if char not in ('\n', '\r'):
                accumulator += char
            elif accumulator:
                first_round.append(re.split('\\s+', accumulator.strip()))
                accumulator = ''
        node_stack = [self.root]
        frame_time_found = False
        node = None
        for item in first_round:
            if frame_time_found:
                self.frames.append(item)
                continue
            key = item[0]
            if key == '{':
                node_stack.append(node)
            elif key == '}':
                node_stack.pop()
            else:
                node = BvhNode(item)
                node_stack[-1].add_child(node)
            if item[0] == 'Frame' and item[1] == 'Time:':
                frame_time_found = True

    def search(self, *items):
        found_nodes = []

        def check_children(node):
            if len(node.value) >= len(items):
                failed = False
                for index, item in enumerate(items):
                    if node.value[index] != item:
                        failed = True
                        break
                if not failed:
                    found_nodes.append(node)
            for child in node:
                check_children(child)

        check_children(self.root)
        return found_nodes

    def get_joints(self):
        joints = []

        def iterate_joints(joint):
            joints.append(joint)
            for child in joint.filter('JOINT'):
                iterate_joints(child)

        iterate_joints(next(self.root.filter('ROOT')))
        return joints

    def get_joints_names(self):
        joints = []

        def iterate_joints(joint):
            joints.append(joint.value[1])
            for child in joint.filter('JOINT'):
                iterate_joints(child)

        iterate_joints(next(self.root.filter('ROOT')))
        return joints

    def joint_direct_children(self, name):
        joint = self.get_joint(name)
        return [child for child in joint.filter('JOINT')]

    def get_joint_index(self, name):
        return self.get_joints().index(self.get_joint(name))

    def get_joint(self, name):
        found = self.search('ROOT', name)
        if not found:
            found = self.search('JOINT', name)
        if found:
            return found[0]
        raise LookupError('joint not found')

    def joint_offset(self, name):
        joint = self.get_joint(name)
        offset = joint['OFFSET']
        return (float(offset[0]), float(offset[1]), float(offset[2]))

    def joint_channels(self, name):
        joint = self.get_joint(name)
        return joint['CHANNELS'][1:]

    def get_joint_channels_index(self, joint_name):
        index = 0
        for joint in self.get_joints():
            if joint.value[1] == joint_name:
                return index
            index += int(joint['CHANNELS'][0])
        raise LookupError('joint not found')

    def get_joint_channel_index(self, joint, channel):
        channels = self.joint_channels(joint)
        if channel in channels:
            channel_index = channels.index(channel)
        else:
            channel_index = -1
        return channel_index

    def frame_joint_channel(self, frame_index, joint, channel, value=None):
        joint_index = self.get_joint_channels_index(joint)
        channel_index = self.get_joint_channel_index(joint, channel)
        if channel_index == -1 and value is not None:
            return value
        return float(self.frames[frame_index][joint_index + channel_index])

    def frame_joint_channels(self, frame_index, joint, channels, value=None):
        values = []
        joint_index = self.get_joint_channels_index(joint)
        for channel in channels:
            channel_index = self.get_joint_channel_index(joint, channel)
            if channel_index == -1 and value is not None:
                values.append(value)
            else:
                values.append(
                    float(
                        self.frames[frame_index][joint_index + channel_index]
                    )
                )
        return values

    def frames_joint_channels(self, joint, channels, value=None):
        all_frames = []
        joint_index = self.get_joint_channels_index(joint)
        for frame in self.frames:
            values = []
            for channel in channels:
                channel_index = self.get_joint_channel_index(joint, channel)
                if channel_index == -1 and value is not None:
                    values.append(value)
                else:
                    values.append(
                        float(frame[joint_index + channel_index]))
            all_frames.append(values)
        return all_frames

    def joint_parent(self, name):
        joint = self.get_joint(name)
        if joint.parent == self.root:
            return None
        return joint.parent

    def joint_parent_index(self, name):
        joint = self.get_joint(name)
        if joint.parent == self.root:
            return -1
        return self.get_joints().index(joint.parent)

    @property
    def nframes(self):
        try:
            return int(next(self.root.filter('Frames:')).value[1])
        except StopIteration:
            raise LookupError('number of frames not found')

    @property
    def frame_time(self):
        try:
            return float(next(self.root.filter('Frame')).value[2])
        except StopIteration:
            raise LookupError('frame time not found')


NUM_CHANNELS = 24


def CMU_parse(file, start=1, frame_skip=5):
    mocap = Bvh(file)

    f = 0
    for i in range(start, mocap.nframes, frame_skip):
        f += 1

    motion_data = np.zeros((NUM_CHANNELS, f, 3), dtype=np.float)

    root_channels = ["Xposition", "Yposition", "Zposition"]
    joint_channels = ["Zrotation", "Yrotation", "Xrotation"]

    f = 0

    for i in range(start, mocap.nframes, frame_skip):
        motion_data[0, f] = mocap.frame_joint_channels(i, "Hips", root_channels)
        motion_data[1, f] = mocap.frame_joint_channels(i, "Hips", joint_channels)
        motion_data[2, f] = mocap.frame_joint_channels(i, "LowerBack", joint_channels)
        motion_data[3, f] = mocap.frame_joint_channels(i, "Spine", joint_channels)
        motion_data[4, f] = mocap.frame_joint_channels(i, "Spine1", joint_channels)
        motion_data[5, f] = mocap.frame_joint_channels(i, "Neck", joint_channels)
        motion_data[6, f] = mocap.frame_joint_channels(i, "Neck1", joint_channels)
        motion_data[7, f] = mocap.frame_joint_channels(i, "Head", joint_channels)
        motion_data[8, f] = mocap.frame_joint_channels(i, "LeftShoulder", joint_channels)
        motion_data[9, f] = mocap.frame_joint_channels(i, "LeftArm", joint_channels)
        motion_data[10, f] = mocap.frame_joint_channels(i, "LeftForeArm", joint_channels)
        motion_data[11, f] = mocap.frame_joint_channels(i, "LeftHand", joint_channels)
        motion_data[12, f] = mocap.frame_joint_channels(i, "RightShoulder", joint_channels)
        motion_data[13, f] = mocap.frame_joint_channels(i, "RightArm", joint_channels)
        motion_data[14, f] = mocap.frame_joint_channels(i, "RightForeArm", joint_channels)
        motion_data[15, f] = mocap.frame_joint_channels(i, "RightHand", joint_channels)
        motion_data[16, f] = mocap.frame_joint_channels(i, "LeftUpLeg", joint_channels)
        motion_data[17, f] = mocap.frame_joint_channels(i, "LeftLeg", joint_channels)
        motion_data[18, f] = mocap.frame_joint_channels(i, "LeftFoot", joint_channels)
        motion_data[19, f] = mocap.frame_joint_channels(i, "LeftToeBase", joint_channels)
        motion_data[20, f] = mocap.frame_joint_channels(i, "RightUpLeg", joint_channels)
        motion_data[21, f] = mocap.frame_joint_channels(i, "RightLeg", joint_channels)
        motion_data[22, f] = mocap.frame_joint_channels(i, "RightFoot", joint_channels)
        motion_data[23, f] = mocap.frame_joint_channels(i, "RightToeBase", joint_channels)
        f += 1

    motion_data = motion_data.transpose((0, 2, 1))
    motion_data = np.reshape(motion_data, (3 * NUM_CHANNELS, f))

    return motion_data
