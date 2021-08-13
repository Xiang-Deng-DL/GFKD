import torch
from torch.utils.data import DataLoader
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

class GraphFullDataLoader():
    def __init__(self, dataset, batch_size, device, collate_fn=collate):

        self.kwargs = {'pin_memory': True} if device >= 0 else {}

        self.train_load = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, **self.kwargs)

    def train_loader(self):
        return self.train_load

  


