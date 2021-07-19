from torch import nn
import torch
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch


NUM_BONES = 23
device = torch.device("cuda")


class GKEN(nn.Module):
    def __init__(self, n_hidden=128):
        super(GKEN, self).__init__()

        self.CMU_SKELETON = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                           22, 0, 1, 2, 3, 4, 5, 3, 7, 8, 9, 3, 11, 12, 13, 0, 15, 16, 17, 0, 19, 20,
                                           21],
                                          [0, 1, 2, 3, 4, 5, 3, 7, 8, 9, 3, 11, 12, 13, 0, 15, 16, 17, 0, 19, 20, 21, 1,
                                           2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]],
                                         dtype=torch.long, device=torch.device("cpu"))

        self.n_hidden = n_hidden

        self.gconv1 = gnn.GCNConv(7, n_hidden, improved=True, bias=False)
        self.gbn1 = gnn.BatchNorm(n_hidden)
        self.grelu1 = nn.LeakyReLU(inplace=True)
        self.gconv2 = gnn.GCNConv(n_hidden, n_hidden, improved=True, bias=False)
        self.gbn2 = gnn.BatchNorm(n_hidden)
        self.grelu2 = nn.LeakyReLU(inplace=True)
        self.gconv3 = gnn.GCNConv(n_hidden, n_hidden, improved=True, bias=False)
        self.gbn3 = gnn.BatchNorm(n_hidden)
        self.grelu3 = nn.LeakyReLU(inplace=True)

        self.lstm = nn.LSTM(NUM_BONES * n_hidden + 1, 4 * n_hidden, batch_first=True, bidirectional=True, num_layers=8)

        self.fc1 = nn.Conv1d(8 * n_hidden, 8 * n_hidden, 3, padding=1, padding_mode="replicate", bias=False)
        self.bn_fc1 = nn.BatchNorm1d(8 * n_hidden)
        self.relu_fc1 = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Conv1d(8 * n_hidden, 8 * n_hidden, 3, padding=1, padding_mode="replicate", bias=False)
        self.bn_fc2 = nn.BatchNorm1d(8 * n_hidden)
        self.relu_fc2 = nn.LeakyReLU(inplace=True)

        self.out = nn.Conv1d(8 * n_hidden, 1, 1)

    def _create_graph(self, state):
        # Pose data in state is [ATTRIBUTES, FRAMES]

        node_data = torch.cat([state[:-1].t().reshape(-1, 6),
                               state[-1].unsqueeze(1).repeat(1, NUM_BONES).reshape(-1, 1)], dim=1)

        skeleton_edges = torch.arange(state.shape[1], device=torch.device("cpu")).reshape(-1, 1).repeat(2, 1, self.CMU_SKELETON.shape[1]).reshape(2, -1)
        skeleton_edges *= NUM_BONES
        skeleton_edges += self.CMU_SKELETON.repeat(1, state.shape[1])

        return Data(x=node_data, edge_index=skeleton_edges, device=device)

    def forward(self, state_batch, remaining_actions):
        batch_list = torch.unbind(state_batch)
        data_list = []
        for i in range(len(batch_list)):
            data_list.append(self._create_graph(batch_list[i]))

        graph = Batch().from_data_list(data_list).to(device)

        g, edges = graph.x, graph.edge_index

        g = self.grelu1(self.gbn1(self.gconv1(g, edges)))
        g = self.grelu2(self.gbn2(self.gconv2(g, edges)))
        g = self.grelu3(self.gbn3(self.gconv3(g, edges)))

        g_data = g.reshape(state_batch.shape[0], state_batch.shape[2], -1)
        concat = torch.cat([g_data, remaining_actions.view(-1, 1, 1).repeat(1, state_batch.shape[2], 1).to(device)], dim=2)

        h, _ = self.lstm(concat)
        h = self.relu_fc1(self.bn_fc1(self.fc1(h.transpose(1, 2))))
        h = self.relu_fc2(self.bn_fc2(self.fc2(h)))
        out = self.out(h)

        return out.squeeze(1)[:, 1:-1]

