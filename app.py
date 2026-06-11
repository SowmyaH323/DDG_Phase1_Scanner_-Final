# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
import xgboost as xgb
import tensorflow as tf

from utils_ddg import (
    AA_LIST, load_fasta_str, build_features_single,
    build_contact_map_from_pdb_bytes, to_fixed_128,
    normalize_adj, SimpleGCN
)

st.set_page_config(page_title="ΔΔG Mutation Scanner", layout="wide")

# ---------- Load models once ----------
@st.cache_resource
def load_models():
    xgb_path = os.path.join("models", "xgb_ddg_model.json")
    cnn_path = os.path.join("models", "cnn_struct_model.keras")
    gnn_path = os.path.join("models", "gnn_struct_model_light.pt")

    # XGB
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_path)

    # CNN
    cnn_model = tf.keras.models.load_model(cnn_path)

    # GNN
    device = torch.device("cpu")
    gnn_model = SimpleGCN().to(device)
    state_dict = torch.load(gnn_path, map_location=device)
    gnn_model.load_state_dict(state_dict)
    gnn_model.eval()

    return xgb_model, cnn_model, gnn_model, device

xgb_model, cnn_model, gnn_model, device = load_models()

st.title("ΔΔG Mutation Scanner – Ensemble (XGB + CNN + GNN)")

# ---------- Sidebar: inputs ----------
st.sidebar.header("Protein inputs")

fasta_text = st.sidebar.text_area(
    "Paste FASTA sequence (WT protein)",
    height=180,
    help="Use the wild-type protein sequence. '>' header lines are allowed."
)

pdb_file = st.sidebar.file_uploader(
    "Upload PDB file (matching the same protein)",
    type=["pdb"],
    help="Optional but required for CNN+GNN. Chain must match the ID below."
)

chain_id = st.sidebar.text_input("PDB chain ID", value="A")

# parse sequence
seq = load_fasta_str(fasta_text) if fasta_text else ""

# Precompute contact map + tensors if PDB provided
M4 = None
A_hat_t = None
deg_t = None
node_feat_t = None

if pdb_file is not None:
    pdb_bytes = pdb_file.read()
    M = build_contact_map_from_pdb_bytes(pdb_bytes, chain_id=chain_id)
    A_hat = normalize_adj(M)
    deg = A_hat.sum(axis=1, keepdims=True).astype(np.float32)

    A_hat_t = torch.from_numpy(A_hat).float().to(device)
    deg_t   = torch.from_numpy(deg).float().to(device)
    node_feat_t = torch.ones((A_hat.shape[0], 1), dtype=torch.float32, device=device)

    M128 = to_fixed_128(M)
    M4   = M128[np.newaxis, ..., np.newaxis]  # shape [1,128,128,1]

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["Single mutation", "19-AA scan"])

# ---------- TAB 1: single mutation ----------
with tab1:
    st.subheader("Single mutation prediction")

    if not seq:
        st.info("Paste the FASTA sequence in the sidebar to enable this tab.")
    else:
        max_pos = len(seq)
        col1, col2 = st.columns(2)

        with col1:
            pos = st.number_input(
                "Position (1-based, in FASTA numbering)",
                min_value=1,
                max_value=max_pos,
                value=1,
                step=1
            )
            wt_default = seq[pos-1]
        with col2:
            wt = st.text_input("Wild-type residue", value=wt_default, max_chars=1).upper()
            mt = st.selectbox(
                "Mutant residue",
                [a for a in AA_LIST if a != wt]
            )

        if st.button("Predict ΔΔG for this mutation"):
            X_num = build_features_single(wt=wt, mt=mt, pos=int(pos))

            # XGB
            xgb_pred = float(xgb_model.predict(X_num)[0])
            preds = [xgb_pred]
            detail = {"XGB": xgb_pred}

            # CNN + GNN only if PDB/contact map available
            if M4 is not None and A_hat_t is not None:
                # CNN
                cnn_pred = float(
                    cnn_model.predict([M4, X_num], verbose=0).reshape(-1)[0]
                )
                preds.append(cnn_pred)
                detail["CNN"] = cnn_pred

                # GNN
                xg_t = torch.from_numpy(X_num[0]).float().to(device)
                with torch.no_grad():
                    gnn_out = gnn_model(A_hat_t, node_feat_t, xg_t)
                    gnn_pred = float(gnn_out.item())
                preds.append(gnn_pred)
                detail["GNN"] = gnn_pred

            ens = float(np.mean(preds))
            std = float(np.std(preds))

            st.markdown("### Results")
            st.write(f"**Ensemble ΔΔG:** {ens:.3f} kcal/mol")
            st.write(f"Ensemble std (XGB/CNN/GNN disagreement): {std:.3f}")
            st.json(detail)
            st.caption("Negative ΔΔG → stabilizing, positive ΔΔG → destabilizing (same sign as training).")

# ---------- TAB 2: 19-AA scan ----------
with tab2:
    st.subheader("19-amino-acid scan")

    if not seq:
        st.info("Paste the FASTA sequence in the sidebar to run scans.")
    else:
        max_pos = len(seq)
        col1, col2 = st.columns(2)
        with col1:
            start_pos = st.number_input("Start position", 1, max_pos, 1)
        with col2:
            end_pos = st.number_input("End position", 1, max_pos, min(max_pos, 50))

        if st.button("Run 19-AA scan for this region"):
            rows = []
            for pos in range(int(start_pos), int(end_pos) + 1):
                wt = seq[pos-1]
                if wt not in AA_LIST:
                    continue

                for mt in AA_LIST:
                    if mt == wt:
                        continue

                    X_num = build_features_single(wt=wt, mt=mt, pos=pos)
                    xgb_pred = float(xgb_model.predict(X_num)[0])
                    preds = [xgb_pred]

                    # CNN + GNN if structural info available
                    if M4 is not None and A_hat_t is not None:
                        cnn_pred = float(
                            cnn_model.predict([M4, X_num], verbose=0).reshape(-1)[0]
                        )
                        preds.append(cnn_pred)

                        xg_t = torch.from_numpy(X_num[0]).float().to(device)
                        with torch.no_grad():
                            gnn_out = gnn_model(A_hat_t, node_feat_t, xg_t)
                            gnn_pred = float(gnn_out.item())
                        preds.append(gnn_pred)
                    else:
                        cnn_pred = np.nan
                        gnn_pred = np.nan

                    ens = float(np.mean(preds))
                    std = float(np.std(preds))

                    rows.append({
                        "pos": pos,
                        "wt": wt,
                        "mt": mt,
                        "XGB_ddG": xgb_pred,
                        "CNN_ddG": cnn_pred,
                        "GNN_ddG": gnn_pred,
                        "ENS_3AVG_ddG": ens,
                        "ensemble_std": std
                    })

            df_scan = pd.DataFrame(rows)
            st.markdown("### Full scan results")
            st.dataframe(df_scan.sort_values("ENS_3AVG_ddG"))

            # Pick top stabilizing mutations
            df_stab = df_scan[df_scan["ENS_3AVG_ddG"] < -0.5].copy()
            df_stab = df_stab[df_stab["ensemble_std"] < 0.7]
            df_top10 = df_stab.sort_values("ENS_3AVG_ddG").head(10)

            st.markdown("### Top 10 stabilizing candidates")
            st.dataframe(df_top10)

            # Download buttons
            csv_full = df_scan.to_csv(index=False).encode()
            st.download_button(
                "Download full scan CSV",
                data=csv_full,
                file_name="scan_results.csv",
                mime="text/csv"
            )

            csv_top = df_top10.to_csv(index=False).encode()
            st.download_button(
                "Download top-10 stabilising CSV",
                data=csv_top,
                file_name="top10_stabilising.csv",
                mime="text/csv"
            )