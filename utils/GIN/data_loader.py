import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from sklearn.model_selection import StratifiedKFold
import dgl

def collate(samples):
    # 'samples (graph, label)'
    graphs, labels = map(list, zip(*samples))
    for g in graphs:
        for key in g.node_attr_schemes().keys():
            g.ndata[key] = g.ndata[key].float()
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    return batched_graph, labels

class GraphDataLoader():
    def __init__(self, dataset, batch_size, device,
                 collate_fn=collate, seed=0, shuffle=True,
                 split_name='rand', split_ratio=0.7):

        self.shuffle = shuffle
        self.seed = seed
        self.kwargs = {'pin_memory': True} if device >= 0 else {}

        labels = [l for _, l in dataset]

 
        if split_name == 'rand':
            train_idx, valid_idx = self._split_rand(
                labels, split_ratio, seed, shuffle
            )
        else:
            raise NotImplementedError()

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)


        self.train_loader = DataLoader(
            dataset, sampler=train_sampler,
            batch_size=batch_size, collate_fn=collate_fn, **self.kwargs
        )

        self.valid_loader = DataLoader(
            dataset, sampler=valid_sampler,
            batch_size=batch_size, collate_fn=collate_fn, **self.kwargs
        )

    def train_valid_loader(self):
        return self.train_loader, self.valid_loader


    def _split_rand(self, labels, split_ratio=0.7, seed=0, shuffle=True):
        num_entries = len(labels)
        indices = list(range(num_entries))

        np.random.seed(seed)
        np.random.shuffle(indices)

        split = int(math.floor(split_ratio * num_entries))
        train_idx, valid_idx = indices[:split], indices[split:]

        print(
            'train_set: test_set = %d : %d' % (len(train_idx), len(valid_idx))
        )

        return train_idx, valid_idx

if __name__ == '__main__':
    from Temp.dataset import GINDataset
    dataset = GINDataset(name='PROTEINS', self_loop=True, degree_as_nlabel=False)

    Loader_list = []
  