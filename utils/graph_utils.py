def preprocess_structure(struct_text: str):
    struct = Structure.from_str(struct_text, fmt="cif")

    positions = np.array([site.coords for site in struct.sites], dtype=np.float32)
    center = positions.mean(axis=0, keepdims=True)
    r_center = np.linalg.norm(positions - center, axis=1).astype(np.float32)

    site_types = []
    feats = []
    symbols = []   

    for site in struct.sites:
        sym = site.specie.symbol
        symbols.append(sym)
        site_types.append(classify_site_element(sym))
        feats.append(get_node_features(sym))

    x = torch.tensor(np.stack(feats), dtype=torch.float32)
    pos = torch.tensor(positions, dtype=torch.float32)
    r_center = torch.tensor(r_center, dtype=torch.float32)

    return struct, pos, r_center, site_types, x, symbols
  def visualize_graph_with_symbols(
    data,
    title="MAD-GNN Graph (Element Nodes)",
    only_v0_edges=False
):
    edge_index = data.edge_index.detach().cpu().numpy()
    num_nodes = data.x.size(0)
    v0 = num_nodes - 1

    edges = edge_index.T.tolist()
    if only_v0_edges:
        edges = [e for e in edges if e[0] == v0 or e[1] == v0]

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    # Layout from positions
    p = data.pos.detach().cpu().numpy()
    pos_layout = {i: (p[i, 0], p[i, 1]) for i in range(num_nodes)}

    # push V0 slightly outward
    xs, ys = p[:, 0], p[:, 1]
    pos_layout[v0] = (
        xs.mean() + 0.25 * (xs.max() - xs.min() + 1e-6),
        ys.mean() + 0.25 * (ys.max() - ys.min() + 1e-6)
    )

    # Colors
    node_colors = ["skyblue"] * num_nodes
    node_colors[v0] = "black"

    # Labels = element symbols
    labels = {i: data.symbols[i] for i in range(num_nodes)}

    plt.figure(figsize=(9, 7))
    nx.draw(
        G,
        pos_layout,
        with_labels=True,
        labels=labels,
        node_color=node_colors,
        node_size=900,
        arrows=True,
        arrowsize=16,
        width=1.2,
        font_size=11
    )
    plt.title(title)
    plt.axis("off")
    plt.show()
  from pymatgen.core import Structure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def visualize_graph_with_symbols_safe(
    data,
    struct_text,                  
    title="Graph",
    only_v0_edges=False
):
    
    struct = Structure.from_str(struct_text, fmt="cif")
    symbols = [s.specie.symbol for s in struct.sites] + ["V0"]

    edge_index = data.edge_index.detach().cpu().numpy()
    num_nodes = data.x.size(0)
    v0 = num_nodes - 1

    edges = edge_index.T.tolist()
    if only_v0_edges:
        edges = [e for e in edges if (e[0] == v0 or e[1] == v0)]

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

   
    p = data.pos.detach().cpu().numpy()
    pos_layout = {i: (p[i, 0], p[i, 1]) for i in range(num_nodes)}


    xs, ys = p[:, 0], p[:, 1]
    pos_layout[v0] = (xs.mean() + 0.25*(xs.max()-xs.min()+1e-6),
                      ys.mean() + 0.25*(ys.max()-ys.min()+1e-6))

    
    node_colors = ["skyblue"] * num_nodes
    node_colors[v0] = "black"

    labels = {i: symbols[i] for i in range(num_nodes)}

    plt.figure(figsize=(9, 7))
    nx.draw(G, pos_layout, with_labels=True, labels=labels,
            node_color=node_colors, node_size=900,
            arrows=True, arrowsize=16, width=1.2, font_size=11)
    plt.title(title)
    plt.axis("off")
    plt.show()
  row = df.iloc[0]
struct_text = row["Structure"]   # your dataset column

visualize_graph_with_symbols_safe(
    data, struct_text,
    title="Full Graph (Local + Ionic + V0)",
    only_v0_edges=False
)

visualize_graph_with_symbols_safe(
    data, struct_text,
    title="Only Long-Range via V0",
    only_v0_edges=True
)
