import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

class MADMini(nn.Module):
    def __init__(self, in_dim=4, h=64):
        super().__init__()
        def conv():
            mlp = nn.Sequential(nn.Linear(h,h), nn.ReLU(), nn.Linear(h,h))
            return GINEConv(mlp, edge_dim=h)

        self.node = nn.Linear(in_dim, h)
        self.edge = nn.Sequential(nn.Linear(1,h), nn.ReLU(), nn.Linear(h,h))

        self.c0, self.c1, self.c2 = conv(), conv(), conv()
        self.head = nn.Sequential(nn.Linear(3*h, 64), nn.ReLU(), nn.Linear(64,1))

    def forward(self, data):
        x = self.node(data.x)
        e = self.edge(data.edge_attr)

        m0 = (data.edge_type==0); m1 = (data.edge_type==1); m2 = (data.edge_type==2)

        def run(conv, mask):
            ei = data.edge_index[:, mask]
            ea = e[mask]
            h = x if ei.size(1)==0 else conv(x, ei, ea)
            return global_mean_pool(h, data.batch)

        g0 = run(self.c0, m0)
        g1 = run(self.c1, m1)
        g2 = run(self.c2, m2)

        g = torch.cat([g0,g1,g2], dim=1)
        return self.head(g)
