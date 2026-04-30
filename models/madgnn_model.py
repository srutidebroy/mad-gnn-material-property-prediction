import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

class MADMiniV2(nn.Module):
    def __init__(self, in_dim=4, h=128, L=3, dropout=0.1, attn_tau=1.5):
        super().__init__()

        self.dropout = dropout
        self.attn_tau = attn_tau


        self.node = nn.Linear(in_dim, h)
        self.edge = nn.Sequential(
            nn.Linear(1, h),
            nn.ReLU(),
            nn.Linear(h, h)
        )

 
        def make_stack():
            layers = nn.ModuleList()
            for _ in range(L):
                mlp = nn.Sequential(
                    nn.Linear(h, h),
                    nn.ReLU(),
                    nn.Linear(h, h)
                )
                layers.append(GINEConv(mlp, edge_dim=h))
            return layers

        self.branch0 = make_stack()  
        self.branch1 = make_stack()  
        self.branch2 = make_stack()  

        
        self.attn = nn.Sequential(
            nn.Linear(h, h//2),
            nn.ReLU(),
            nn.Linear(h//2, 1)
        )

    
        self.head = nn.Sequential(
            nn.Linear(h, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


    def run_branch(self, x, edge_index, edge_attr, conv_layers):
        h = x
        for conv in conv_layers:
            h_new = conv(h, edge_index, edge_attr)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  
        return h

    def forward(self, data):

        x = self.node(data.x)
        e = self.edge(data.edge_attr)

        m0 = (data.edge_type == 0)
        m1 = (data.edge_type == 1)
        m2 = (data.edge_type == 2)

        def branch(mask, stack):
            if mask.sum() == 0:
                h_nodes = x
            else:
                ei = data.edge_index[:, mask]
                ea = e[mask]
                h_nodes = self.run_branch(x, ei, ea, stack)
            return global_mean_pool(h_nodes, data.batch)

        g0 = branch(m0, self.branch0)
        g1 = branch(m1, self.branch1)
        g2 = branch(m2, self.branch2)
        s0 = self.attn(g0)
        s1 = self.attn(g1)
        s2 = self.attn(g2)

        scores = torch.cat([s0, s1, s2], dim=1)
        alpha = F.softmax(scores / self.attn_tau, dim=1)

        g = (
            alpha[:, 0:1] * g0 +
            alpha[:, 1:2] * g1 +
            alpha[:, 2:3] * g2
        )

        out = self.head(g).view(-1)

        return out, alpha
        import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

class MADMini(nn.Module):
    def __init__(self, in_dim=4, h=64):
        super().__init__()

        def conv():
            mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
            return GINEConv(mlp, edge_dim=h)

        self.node = nn.Linear(in_dim, h)
        self.edge = nn.Sequential(nn.Linear(1, h), nn.ReLU(), nn.Linear(h, h))

        self.c0, self.c1, self.c2 = conv(), conv(), conv()
        self.head = nn.Sequential(nn.Linear(3*h, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, data):
        x = self.node(data.x)
        e = self.edge(data.edge_attr)

        m0 = (data.edge_type == 0)
        m1 = (data.edge_type == 1)
        m2 = (data.edge_type == 2)

        def run(conv_layer, mask):
            ei = data.edge_index[:, mask]
            ea = e[mask]
            h = x if ei.size(1) == 0 else conv_layer(x, ei, ea)
            return global_mean_pool(h, data.batch)

        g0 = run(self.c0, m0)
        g1 = run(self.c1, m1)
        g2 = run(self.c2, m2)

        g = torch.cat([g0, g1, g2], dim=1)
        return self.head(g)
