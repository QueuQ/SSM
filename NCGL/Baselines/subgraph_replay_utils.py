import random
import time
import numpy as np
import torch
import torch as th
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
from dgl.base import DGLError
import dgl.function as fn
import copy

class random_subgraph_sampler(nn.Module):
    def __init__(self,args):
        super().__init__()

    def forward(self, graph, center_node_budget, nei_budget, gnn, ids_per_cls):
        center_nodes_selected = self.node_sampler(ids_per_cls, graph, center_node_budget)
        all_nodes_selected = self.nei_sampler(center_nodes_selected, graph, nei_budget)
        return center_nodes_selected, all_nodes_selected

    def node_sampler(self,ids_per_cls_train, graph, budget, max_ratio_per_cls = 1.0):
        store_ids = []
        for i, ids in enumerate(ids_per_cls_train):
            budget_ = min(budget, int(max_ratio_per_cls * len(ids))) if isinstance(budget, int) else int(
                budget * len(ids))
            store_ids.extend(random.sample(ids, budget_))
        return store_ids

    def nei_sampler(self, center_nodes_selected, graph, nei_budget):
        nodes_selected_current_hop = copy.deepcopy(center_nodes_selected)
        retained_nodes = copy.deepcopy(center_nodes_selected)
        for b in nei_budget:
            # from 1-hop to len(nei_budget)-hop neighbors
            neighbors = graph.in_edges(nodes_selected_current_hop)[0].tolist()
            nodes_selected_current_hop = random.choices(neighbors, k=b)
            retained_nodes.extend(nodes_selected_current_hop)
        return list(set(retained_nodes))


class degree_based_sampler(nn.Module):
    # based on random walk sasmpler01, sample a subgraph based on the degrees of the neighbors
    def __init__(self, args):
        super().__init__()

    def forward(self, graph, center_node_budget, nei_budget, gnn, ids_per_cls, restart=0.0):
        center_nodes_selected = self.node_sampler(ids_per_cls, graph, center_node_budget)
        all_nodes_selected = self.nei_sampler(center_nodes_selected, graph, nei_budget)
        return center_nodes_selected, all_nodes_selected

    def node_sampler(self,ids_per_cls_train, graph, budget, max_ratio_per_cls = 1.0):
        store_ids = []
        for i, ids in enumerate(ids_per_cls_train):
            budget_ = min(budget, int(max_ratio_per_cls * len(ids))) if isinstance(budget, int) else int(
                budget * len(ids))
            store_ids.extend(random.sample(ids, budget_))
        return store_ids

    def nei_sampler(self, center_nodes_selected, graph, nei_budget):
        probs = graph.in_degrees().float()
        nodes_selected_current_hop = copy.deepcopy(center_nodes_selected)
        retained_nodes = copy.deepcopy(center_nodes_selected)
        for b in nei_budget:
            if b==0:
                continue
            # from 1-hop to len(nei_budget)-hop neighbors
            neighbors = list(set(graph.in_edges(nodes_selected_current_hop)[0].tolist()))
            # remove selected nodes
            for n in retained_nodes:
                neighbors.remove(n)
            if len(neighbors)==0:
                continue
            prob = probs[neighbors]
            sampled_neibs_ = torch.multinomial(prob, min(b, len(neighbors)), replacement=False).tolist()
            sampled_neibs = torch.tensor(neighbors)[sampled_neibs_] # map the ids to the original ones
            retained_nodes.extend(sampled_neibs.tolist())
        return list(set(retained_nodes))


