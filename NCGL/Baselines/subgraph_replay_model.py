# compared to subgraph replay, this module removes the inter-task edges when loading stored subgraphs
import torch
import copy
from .subgraph_replay_utils import *
from tqdm import tqdm

samplers = {'random': random_subgraph_sampler, 'd_samp':degree_based_sampler}#, 'random_walk_weighted':random_walk_sampler_weighted}
class NET(torch.nn.Module):
    def __init__(self,
                 model,
                 task_manager,
                 args):
        super(NET, self).__init__()

        self.task_manager = task_manager

        # setup network
        self.net = model
        self.sampler = samplers[args.sgreplay_args['sampler']](args)

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.functional.cross_entropy

        # setup memories
        self.current_task = -1
        self.buffer_c_node = []
        self.buffer_all_nodes = []
        self.aux_g = None

    def forward(self, features):
        output = self.net(features)
        return output


    def observe_class_IL_batch(self, args, g, dataloader, features, labels, t, train_ids, ids_per_cls, dataset):
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        self.net.train()
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        size_g_train = labels.shape[0]

        if t!=self.current_task:
            # store data from the current task
            self.current_task = t
            old_ids = g.ndata['_ID'].cpu()
            c_nodes_sampled, nbs_sampled= self.sampler(g, args.sgreplay_args['c_node_budget'], args.sgreplay_args['nei_budget'], self.net, ids_per_cls_train)
            self.buffer_c_node.extend(old_ids[c_nodes_sampled])
            self.buffer_all_nodes.append(old_ids[nbs_sampled].tolist())

        if t>0:
            aux_g, aux_ids_per_cls, _ = dataset.get_graph(node_ids=self.buffer_all_nodes, remove_edges=False)
            old_ids_aux = aux_g.ndata['_ID']
            aux_train_ids = [(old_ids_aux == i).nonzero().squeeze().item() + size_g_train for i in self.buffer_c_node]
            train_ids.extend(aux_train_ids)
            g = dgl.batch([g, aux_g])
            #g = dgl.batch([g, aux_g.to(device='cuda:{}'.format(args.gpu))])

        dataloader = dgl.dataloading.NodeDataLoader(g, train_ids, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False)

        # train in batches
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(torch.device('cuda')) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))

            output_predictions,_ = self.net.forward_batch(blocks, input_features)
            if args.classifier_increase:
                loss = self.ce(output_predictions[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            else:
                loss = self.ce(output_predictions, output_labels, weight=self.aux_loss_w_)
            loss.backward()
            self.opt.step()


    def observe(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        self.net.train()
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        size_g_train = g.num_nodes()

        if t!=self.current_task:
            # if new tasks come, store data from the current task
            self.current_task = t
            old_ids = g.ndata['_ID'].cpu()
            #size_g_train = old_ids.shape[0] # size of the current subgraph
            c_nodes_sampled, nbs_sampled= self.sampler(g, args.sgreplay_args['c_node_budget'], args.sgreplay_args['nei_budget'], self.net, ids_per_cls_train)
            self.buffer_c_node.extend(old_ids[c_nodes_sampled])
            self.buffer_all_nodes.append(old_ids[nbs_sampled].tolist())

        if t>0:
            # if not the first task, load data and calculate aux loss
            aux_g, aux_ids_per_cls, _ = dataset.get_graph(node_ids=self.buffer_all_nodes, remove_edges=False)
            old_ids_aux = aux_g.ndata['_ID']
            aux_train_ids = [(old_ids_aux == i).nonzero().squeeze().item() + size_g_train for i in
                             self.buffer_c_node]
            train_ids.extend(aux_train_ids)
            #self.aux_g = aux_g.to(device='cuda:{}'.format(args.gpu))
            g = dgl.batch([g,aux_g.to(device='cuda:{}'.format(args.gpu))])

        features, labels = g.srcdata['feat'], g.dstdata['label'].squeeze() # self.g.srcdata['feat'], self.g.dstdata['label'].squeeze()
        if args.cls_balance:
            n_per_cls = [(labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))

        output, _ = self.net(g, features) # self.net(self.g, features)
        if args.classifier_increase:
            loss = self.ce(output[train_ids, offset1:offset2], labels[train_ids], weight=loss_w_[offset1: offset2])
        else:
            loss = self.ce(output[train_ids], labels[train_ids], weight=self.aux_loss_w_)
        loss.backward()
        self.opt.step()


    def observe_task_IL(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        if not isinstance(self.aux_g, list):
            self.aux_g = []
            self.aux_loss_w_ = []
        self.net.train()

        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
        output, _ = self.net(g, features)
        output_labels = labels[train_ids]

        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        loss = self.ce(output[train_ids, offset1:offset2], output_labels-offset1, weight=loss_w_[offset1: offset2])

        if t!=self.current_task:
            # store data from the current task
            self.current_task = t
            c_nodes_sampled, nbs_sampled = self.sampler(g, args.sgreplay_args['c_node_budget'], args.sgreplay_args['nei_budget'], self.net, ids_per_cls_train)
            old_ids = g.ndata['_ID'].cpu()
            self.buffer_c_node.extend(old_ids[c_nodes_sampled])
            self.buffer_all_nodes.append(old_ids[nbs_sampled].tolist())
            #self.buffer_node_ids[t] = old_ids[sampled_ids].tolist()
            g, self.ids_per_cls, _ = dataset.get_graph(node_ids=[self.buffer_all_nodes[t]], remove_edges=False)
            self.aux_g.append(g.to(device='cuda:{}'.format(features.get_device())))
            labels = g.dstdata['label'].squeeze()
            if args.cls_balance:
                n_per_cls = [(labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            self.aux_loss_w_.append(loss_w_)

        if t!=0:
            # if not the first task, calculate aux loss with buffered data
            for oldt in range(t):
                o1, o2 = self.task_manager.get_label_offset(oldt - 1)[1], self.task_manager.get_label_offset(oldt)[1]
                aux_g = self.aux_g[oldt]
                aux_features, aux_labels = aux_g.srcdata['feat'], aux_g.dstdata['label'].squeeze()
                output, _ = self.net(aux_g, aux_features)
                loss_aux = self.ce(output[:, o1:o2], aux_labels - o1, weight=self.aux_loss_w_[oldt][offset1: offset2])
                loss += loss_aux

        loss.backward()
        self.opt.step()


    def observe_task_IL_batch(self, args, g, dataloader, features, labels, t, train_ids, ids_per_cls, dataset):
        """
                        The method for learning the given tasks under the task-IL setting with mini-batch training.

                        :param args: Same as the args in __init__().
                        :param g: The graph of the current task.
                        :param dataloader: The data loader for mini-batch training
                        :param features: Node features of the current task.
                        :param labels: Labels of the nodes in the current task.
                        :param t: Index of the current task.
                        :param train_ids: The indices of the nodes participating in the training.
                        :param ids_per_cls: Indices of the nodes in each class (currently not in use).
                        :param dataset: The entire dataset (currently not in use).

                        """
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        if not isinstance(self.aux_g, list):
            self.aux_g = []
            self.aux_loss_w_ = []
        self.net.train()
        # now compute the grad on the current task
        offset1, offset2 = self.task_manager.get_label_offset(t-1)[1], self.task_manager.get_label_offset(t)[1]
        for input_nodes, output_nodes, blocks in dataloader:
            self.net.zero_grad()
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()

            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            output_labels = output_labels - offset1
            output_predictions,_ = self.net.forward_batch(blocks, input_features)
            loss = self.ce(output_predictions[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])

            # sample and store ids from current task
            if t != self.current_task:
                self.current_task = t
                c_nodes_sampled, nbs_sampled = self.sampler(g, args.sgreplay_args['c_node_budget'],
                                                            args.sgreplay_args['nei_budget'], self.net,
                                                            ids_per_cls_train)
                old_ids = g.ndata['_ID'].cpu()

                self.buffer_c_node.extend(old_ids[c_nodes_sampled])
                self.buffer_all_nodes.append(old_ids[nbs_sampled].tolist())
                # self.buffer_node_ids[t] = old_ids[sampled_ids].tolist()
                g, self.ids_per_cls, _ = dataset.get_graph(node_ids=[self.buffer_all_nodes[t]], remove_edges=False)
                self.aux_g.append(g.to(device='cuda:{}'.format(args.gpu)))
                labels = g.dstdata['label'].squeeze()
                if args.cls_balance:
                    n_per_cls = [(labels == j).sum() for j in range(args.n_cls)]
                    loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
                else:
                    loss_w_ = [1. for i in range(args.n_cls)]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                self.aux_loss_w_.append(loss_w_)

            if t != 0:
                # if not the first task, calculate aux loss with buffered data
                for oldt in range(t):
                    o1, o2 = self.task_manager.get_label_offset(oldt - 1)[1], self.task_manager.get_label_offset(oldt)[
                        1]
                    aux_g = self.aux_g[oldt]
                    aux_features, aux_labels = aux_g.srcdata['feat'], aux_g.dstdata['label'].squeeze()
                    output, _ = self.net(aux_g, aux_features)
                    loss_aux = self.ce(output[:, o1:o2], aux_labels - o1,
                                       weight=self.aux_loss_w_[oldt][offset1: offset2])
                    loss += loss_aux
            loss.backward()
            self.opt.step()


