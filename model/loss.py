from importlib_metadata import requires
import torch
import torch.nn as nn
from torch import einsum, positive
import math
import random

def l2_distance(a, b):
    # a: [n, d], b: [m, d]
    # Compute pairwise squared L2 distances between a and b
    a_sq = (a ** 2).sum(dim=1, keepdim=True)  # [n, 1]
    b_sq = (b ** 2).sum(dim=1, keepdim=True).t()  # [1, m]
    dist = a_sq + b_sq - 2 * torch.matmul(a, b.t())  # [n, m]
    dist = torch.clamp(dist, min=0)  # numerical stability
    return dist
class InfoNCEGraph(nn.Module):
    def __init__(self, in_channels=128, out_channels=256, mem_size=512, positive_num=128, negative_num=512, T=0.8, class_num=60, label_all=[]):
        super(InfoNCEGraph, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mem_size = mem_size
        self.positive_num = positive_num
        self.negative_num = negative_num
        self.T = T
        self.trans = nn.Linear(in_channels, out_channels)
        self.Bank = nn.Parameter(
            torch.zeros((mem_size, out_channels)), requires_grad=False
        )
        self.label_all = torch.tensor(label_all)
        nn.init.normal_(self.trans.weight, 0, math.sqrt(2. / class_num))
        nn.init.zeros_(self.trans.bias)
        self.bank_flag = nn.Parameter(
            torch.zeros(len(self.label_all)), requires_grad=False
        ) 
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def l2_distance(self, a, b):
        # a: [n, d], b: [m, d]
        # Compute pairwise squared L2 distances between a and b
        a_sq = (a ** 2).sum(dim=1, keepdim=True)  # [n, 1]
        b_sq = (b ** 2).sum(dim=1, keepdim=True).t()  # [1, m]
        dist = a_sq + b_sq - 2 * torch.matmul(a, b.t())  # [n, m]
        dist = torch.clamp(dist, min=0)  # numerical stability
        return dist
    
    def NN(self, z, Q):
        sims = z @ Q.T
        nn_idxes = sims.argmax(dim=1)  # Top-1 NN indices
        return Q[nn_idxes]
    
    def forward(self, f, graph2_semantic, label, input_index):
        # f: n c label: n
        n, _ = f.size()
        f = self.trans(f)
        f_norm = f.norm(dim=-1, p=2, keepdim=True)
        f_normed = f / f_norm

        graph2_semantic = self.trans(graph2_semantic)
        graph2_semantic_norm = graph2_semantic.norm(dim=-1, p=2, keepdim=True)
        graph2_semantic_normed = graph2_semantic / graph2_semantic_norm

        sematic_guide = self.NN(f_normed, graph2_semantic_normed)

        self.Bank[input_index] = sematic_guide.detach()
        self.bank_flag[input_index] = 1

        all_pairs = einsum('n c, m c -> n m', sematic_guide, self.Bank)
        bank_label = self.label_all.to(label.device)  # mem_size

        positive_mask = (label.view(n, 1) == bank_label.view(1, -1)).view(n, self.mem_size)  # n mem_size
        negative_mask = (1 - positive_mask.float())

        positive_mask = positive_mask * self.bank_flag
        negative_mask = negative_mask * self.bank_flag

        combined_pairs_list = []

        for i in range(n):
            if (positive_mask[i].sum(dim=-1) < self.positive_num) or (negative_mask[i].sum(dim=-1) < self.negative_num * 2):
                continue

            # ===== Positives (hard positives) =====
            positive_pairs = torch.masked_select(all_pairs[i], mask=positive_mask[i].bool()).view(-1)
            positive_pairs_hard = positive_pairs.sort(dim=-1, descending=False)[0][:self.positive_num].view(1, self.positive_num, 1)

            # ===== Negatives pool =====
            negative_pairs = torch.masked_select(all_pairs[i], mask=negative_mask[i].bool()).view(-1)

            # ===== Hard Negatives =====
            negative_pairs_hard = negative_pairs.sort(dim=-1, descending=True)[0][:self.negative_num].view(1, 1, self.negative_num) \
                .expand(-1, self.positive_num, -1)

            # ===== Random Negatives =====
            idx = random.sample(list(range(len(negative_pairs))), k=self.negative_num)
            negative_pairs_random = negative_pairs[idx].view(1, 1, self.negative_num).expand(-1, self.positive_num, -1)

            # ===== Combine (positives + negatives) =====
            combined_pairs_hard2random = torch.cat([positive_pairs_hard, negative_pairs_random], -1).view(self.positive_num, -1)
            combined_pairs_hard2hard = torch.cat([positive_pairs_hard, negative_pairs_hard], -1).view(self.positive_num, -1)

            # Keep whichever combos you want. Here: random + hard (2 blocks)
            combined_pairs = torch.cat([combined_pairs_hard2random, combined_pairs_hard2hard], 0)
            combined_pairs_list.append(combined_pairs)

        if len(combined_pairs_list) == 0:
            return torch.zeros(1, device=f.device)

        combined_pairs = torch.cat(combined_pairs_list, 0)
        combined_label = torch.zeros(combined_pairs.size(0), device=f.device).long()
        loss = self.cross_entropy(combined_pairs / self.T, combined_label)

        return loss
