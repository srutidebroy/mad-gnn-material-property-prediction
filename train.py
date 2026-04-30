import numpy as np, pandas as pd, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool


def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()
print("Device:", device, "| AMP:", use_amp)


CSV_PATH   = "materials_with_structure.csv"
CIF_COL    = "Structure"
TARGET_COL = "Band Gap (eV)"

df = pd.read_csv(CSV_PATH)
df_clean = df[df[TARGET_COL] > 0].reset_index(drop=True)
print("Total:", len(df), " Clean:", len(df_clean), " Removed zeros:", (df[TARGET_COL]==0).sum())


y_mean = float(df_clean[TARGET_COL].mean())
y_std  = float(df_clean[TARGET_COL].std() + 1e-8)

def norm_y(y):   return (y - y_mean) / y_std
def denorm_y(y): return y * y_std + y_mean
def build_dataset(df_in) : 
    ds, bad = [], 0
    for _, row in df_in.iterrows():
        cif = row[CIF_COL]
        y = float(row[TARGET_COL])
        try:
            struct, pos, r_center, site_types, x = preprocess_structure_train(cif)
            data = structure_to_madgnn_graph(struct, pos, r_center, site_types, x)

            if getattr(data, "edge_attr", None) is None:
                raise ValueError("edge_attr missing")
            if data.edge_attr.dim() == 1:
                data.edge_attr = data.edge_attr.view(-1, 1)

            if not hasattr(data, "edge_type"):
                raise ValueError("edge_type missing (data.edge_type)")

            data.y = torch.tensor([norm_y(y)], dtype=torch.float32)
            ds.append(data)

        except Exception:
            bad += 1
            continue
            print("Graphs built:", len(ds), "Failed:", bad)
    return ds

dataset = build_dataset(df_clean)
print("Graphs:", len(dataset))

def split_80_10_10(dataset, seed=42):
    n = len(dataset)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_train = int(0.8*n)
    n_val   = int(0.1*n)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    train_ds = [dataset[i] for i in train_idx]
    val_ds   = [dataset[i] for i in val_idx]
    test_ds  = [dataset[i] for i in test_idx]
    return train_ds, val_ds, test_ds
  train_ds, val_ds, test_ds = split_80_10_10(dataset, seed=42)
print("Split:", len(train_ds), len(val_ds), len(test_ds))


BATCH_SIZE = 64 if torch.cuda.is_available() else 16
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


tmp = train_ds[0]
IN_DIM = int(tmp.x.size(1))
EDGE_DIM_IN = int(tmp.edge_attr.size(1))
print("Detected IN_DIM:", IN_DIM, "| EDGE_DIM_IN:", EDGE_DIM_IN)

class MADAttnStack_MEG(nn.Module):
    def __init__(
        self,
        in_dim,
        edge_in_dim,
        h=160,
        L=3,
        dropout=0.12,
        attn_tau=1.6,
        branch_drop=0.15,        
        min_branch_w=0.08         
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_tau = attn_tau
        self.branch_drop = branch_drop
        self.min_branch_w = min_branch_w


        self.node = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Linear(h, h)
        )
      def edge_mlp():
            return nn.Sequential(
                nn.Linear(edge_in_dim, h),
                nn.ReLU(),
                nn.Linear(h, h)
            )
        self.edge0 = edge_mlp()   
        self.edge1 = edge_mlp()  
        self.edge2 = edge_mlp()  


        def make_conv():
            mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
            return GINEConv(mlp, edge_dim=h)

        self.convs0 = nn.ModuleList([make_conv() for _ in range(L)])
        self.convs1 = nn.ModuleList([make_conv() for _ in range(L)])
        self.convs2 = nn.ModuleList([make_conv() for _ in range(L)])

        self.ln0 = nn.ModuleList([nn.LayerNorm(h) for _ in range(L)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(h) for _ in range(L)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(h) for _ in range(L)])

      
        self.attn_mlp = nn.Sequential(
            nn.Linear(h, h//2),
            nn.ReLU(),
            nn.Linear(h//2, 1)
        )
self.head = nn.Sequential(
            nn.Linear(h, 192), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(192, 96), nn.ReLU(),
            nn.Linear(96, 1)
        )

    def _stack(self, x, ei, ea, convs, lns):
        h = x
        for conv, ln in zip(convs, lns):
            h_new = conv(h, ei, ea)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = ln(h + h_new)   
        return h

    def _branch(self, x, edge_attr_enc, data, mask, convs, lns):
        if mask.sum() == 0:
            h_nodes = x
        else:
            ei = data.edge_index[:, mask]
            ea = edge_attr_enc[mask]
            h_nodes = self._stack(x, ei, ea, convs, lns)
        return global_mean_pool(h_nodes, data.batch)
      def forward(self, data):
        x = self.node(data.x)

        m0 = (data.edge_type == 0)
        m1 = (data.edge_type == 1)
        m2 = (data.edge_type == 2)

        e0 = self.edge0(data.edge_attr)
        e1 = self.edge1(data.edge_attr)
        e2 = self.edge2(data.edge_attr)

        g0 = self._branch(x, e0, data, m0, self.convs0, self.ln0)
        g1 = self._branch(x, e1, data, m1, self.convs1, self.ln1)
        g2 = self._branch(x, e2, data, m2, self.convs2, self.ln2)

        
        if self.training and self.branch_drop > 0:
            if torch.rand(1).item() < self.branch_drop:
                
                if torch.rand(1).item() < 0.5:
                    g1 = torch.zeros_like(g1)
                else:
                    g2 = torch.zeros_like(g2)

        s0 = self.attn_mlp(g0)
        s1 = self.attn_mlp(g1)
        s2 = self.attn_mlp(g2)
        scores = torch.cat([s0, s1, s2], dim=1)
        alpha = F.softmax(scores / self.attn_tau, dim=1)
        g_fused = alpha[:,0:1]*g0 + alpha[:,1:2]*g1 + alpha[:,2:3]*g2
        out = self.head(g_fused).view(-1)
        return out, alpha
HIDDEN_DIM   = 160    
LAYERS_L     = 3 
DROPOUT      = 0.12 
LR           = 4e-4 
WEIGHT_DECAY = 2e-5 
MAX_EPOCHS   = 260  

ATTN_TAU     = 1.6      
BRANCH_DROP  = 0.15     
MIN_BRANCH_W = 0.08     

model = MADAttnStack_MEG(  
    in_dim=IN_DIM,
    edge_in_dim=EDGE_DIM_IN, 
    h=HIDDEN_DIM, 
    L=LAYERS_L, 
    dropout=DROPOUT, 
    attn_tau=ATTN_TAU, 
    branch_drop=BRANCH_DROP,  
    min_branch_w=MIN_BRANCH_W  
).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


loss_main = nn.SmoothL1Loss(beta=1.0)


LAMBDA_ENT = 0.02  
LAMBDA_MIN = 0.08   

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=10, min_lr=1e-6
)

early_patience = 25
early_counter = 0
min_delta = 1e-3  # eV
best_val_mae = float("inf")
best_path = "madgnn_meg_push_best.pt"


scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

8️⃣ TRAINING LOOP

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    tr_losses = []
    attn_means = []

    for b in train_loader:
        b = b.to(device)
        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred, alpha = model(b)


            L0 = loss_main(pred, b.y.view(-1))

            
            eps = 1e-9
            ent = -torch.sum(alpha * torch.log(alpha + eps), dim=1).mean()
            L_ent = -ent  

           
            a_mean = alpha.mean(dim=0)              
            a1 = a_mean[1]; a2 = a_mean[2]
            L_min = F.relu(model.min_branch_w - a1) + F.relu(model.min_branch_w - a2)
 loss = L0 + LAMBDA_ENT * L_ent + LAMBDA_MIN * L_min

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()

        tr_losses.append(float(loss.detach().cpu().item()))
        attn_means.append(alpha.detach().cpu().mean(dim=0).numpy())

    tr = float(np.mean(tr_losses))
attn_avg = np.mean(attn_means, axis=0)

    val_mae = evaluate_mae_eV(val_loader)
    scheduler.step(val_mae)
    cur_lr = opt.param_groups[0]["lr"]

    if val_mae < best_val_mae - min_delta:
        best_val_mae = val_mae
        early_counter = 0
        torch.save(model.state_dict(), best_path)
    else:
        early_counter += 1

    if epoch == 1 or epoch % 20 == 0:
        print(
            f"Epoch {epoch:03d} | LR {cur_lr:.2e} | TrainLoss {tr:.4f} "
            f"| Val MAE(eV) {val_mae:.4f} | Best {best_val_mae:.4f} | ES {early_counter}/{early_patience} "
            f"| attn_mean [loc ion long]=[{attn_avg[0]:.3f} {attn_avg[1]:.3f} {attn_avg[2]:.3f}]"
        )

    if early_counter >= early_patience:
        print(f"Early stopping at epoch {epoch} (no val MAE improvement for {early_patience} epochs)")
        break
      print("Saved best model:", best_path)

