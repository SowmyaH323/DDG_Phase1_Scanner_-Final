# utils_ddg.py
import numpy as np
from Bio.PDB import PDBParser
import torch
import torch.nn as nn
import io

# --- Amino acids and simple physicochemical props (same idea as notebook) ---
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_PROPS = {
    'A':[1.8,88.6,0],'C':[2.5,108.5,0],'D':[-3.5,111.1,-1],
    'E':[-3.5,138.4,-1],'F':[2.8,189.9,0],'G':[-0.4,60.1,0],
    'H':[-3.2,153.2,1],'I':[4.5,166.7,0],'K':[-3.9,168.6,1],
    'L':[3.8,166.7,0],'M':[1.9,162.9,0],'N':[-3.5,114.1,0],
    'P':[-1.6,112.7,0],'Q':[-3.5,143.8,0],'R':[-4.5,173.4,1],
    'S':[-0.8,89.0,0],'T':[-0.7,116.1,0],'V':[4.2,140.0,0],
    'W':[-0.9,227.8,0],'Y':[-1.3,193.6,0]
}

def aa_props(a):
    return AA_PROPS.get(a, [0.0, 0.0, 0])

# --- FASTA parsing ---
def load_fasta_str(raw: str) -> str:
    """Take pasted FASTA text and return sequence (ignores header lines)."""
    seq = ""
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        seq += line
    return seq

# --- Numeric features (must match training) ---
def build_features_single(wt, mt, pos,
                          dataset="unknown",
                          mtype="thermal_stability"):
    wt_p = aa_props(wt)
    mt_p = aa_props(mt)
    diff_p = [mt_p[i]-wt_p[i] for i in range(3)]

    flag_skempi  = 1 if dataset == "skempi" else 0
    flag_thermo  = 1 if dataset == "thermomut" else 0
    flag_binding = 1 if mtype  == "binding" else 0
    flag_thermal = 1 if mtype  == "thermal_stability" else 0

    feat = [
        wt_p[0], wt_p[1], wt_p[2],
        mt_p[0], mt_p[1], mt_p[2],
        diff_p[0], diff_p[1], diff_p[2],
        pos,
        flag_skempi, flag_thermo,
        flag_binding, flag_thermal
    ]
    return np.array([feat], dtype=np.float32)  # shape [1,14]

# --- Contact map from uploaded PDB bytes ---
def build_contact_map_from_pdb_bytes(pdb_bytes: bytes,
                                     chain_id="A",
                                     cutoff=8.0):
    """Build CA-CA contact map from uploaded PDB file (as bytes)."""
    parser = PDBParser(QUIET=True)
    handle = io.StringIO(pdb_bytes.decode(errors="ignore"))
    structure = parser.get_structure("prot", handle)
    model = next(structure.get_models())
    chain = model[chain_id]

    coords = []
    for res in chain:
        if "CA" in res:
            coords.append(res["CA"].get_coord())
    coords = np.array(coords, dtype=np.float32)
    n = coords.shape[0]

    M = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        diff = coords[i] - coords
        d2 = np.sum(diff*diff, axis=1)
        mask = d2 <= cutoff**2
        M[i, mask] = 1.0
    M = np.maximum(M, M.T)
    return M

# --- Resize to 128x128 (same as training) ---
FIXED_SIZE = 128

def to_fixed_128(M):
    n = M.shape[0]
    cut = min(n, FIXED_SIZE)
    out = np.zeros((FIXED_SIZE, FIXED_SIZE), dtype=np.float32)
    out[:cut, :cut] = M[:cut, :cut]
    return out

# --- Normalized adjacency for GNN ---
def normalize_adj(M):
    A = (M != 0).astype(np.float32)
    N = A.shape[0]
    A_tilde = A + np.eye(N, dtype=np.float32)
    d = A_tilde.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-8))
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt

# --- SimpleGCN definition (must match training) ---
class SimpleGCN(nn.Module):
    def __init__(self, in_node_feats=1, num_global_feats=14,
                 hidden=32, dropout=0.3):
        super().__init__()
        self.lin1 = nn.Linear(in_node_feats, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden + num_global_feats, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, A_hat, X_node, x_global):
        # X_node is dummy node feature [N,1] of ones
        H = self.act(self.lin1(X_node))
        H = self.act(self.lin2(H))
        H = A_hat @ H
        g = H.mean(dim=0)
        z = torch.cat([g, x_global], dim=-1)
        z = self.act(self.fc1(z))
        z = self.drop(z)
        z = self.act(self.fc2(z))
        return self.out(z).squeeze(-1)