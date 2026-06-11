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

W_XGB = 0.45
W_CNN = 0.35
W_GNN = 0.20

@st.cache_resource
def load_models():
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    xgb_path = os.path.join(BASE_DIR, "xgb_full.json")
    cnn_path = os.path.join(BASE_DIR, "cnn_ddg_model.keras")
    gnn_path = os.path.join(BASE_DIR, "gnn_scripted.pt")

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_path)

    cnn_model = tf.keras.models.load_model(cnn_path)

    device = torch.device("cpu")
    gnn_model = torch.jit.load(gnn_path, map_location=device)
    gnn_model.eval()

    return xgb_model, cnn_model, gnn_model, device

xgb_model, cnn_model, gnn_model, device = load_models()

st.title("ΔΔG Mutation Scanner – Ensemble (CNN + GNN + XGB)")

st.caption(
    "Negative ΔΔG indicates predicted strengthening of protein–protein binding; "
    "positive ΔΔG indicates predicted weakening of protein–protein binding."
)

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

seq = load_fasta_str(fasta_text) if fasta_text else ""

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
    deg_t = torch.from_numpy(deg).float().to(device)

    node_feat_t = torch.ones((A_hat.shape[0], 49), dtype=torch.float32, device=device)

    M128 = to_fixed_128(M)
    M4 = M128[np.newaxis, ..., np.newaxis]

tab1, tab2 = st.tabs(["Single mutation", "19-AA scan"])

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
            mt = st.selectbox("Mutant residue", [a for a in AA_LIST if a != wt])

        if st.button("Predict ΔΔG for this mutation"):
            X_num = build_features_single(wt=wt, mt=mt, pos=int(pos))

            X_array = np.asarray(X_num)
            xgb_pred = float(xgb_model.predict(X_array[:, :4])[0])

            preds = [xgb_pred]
            detail = {"XGB": xgb_pred}

            if M4 is not None and A_hat_t is not None:
                cnn_pred = float(
                    cnn_model.predict([M4, X_num], verbose=0).reshape(-1)[0]
                )
                preds.append(cnn_pred)
                detail["CNN"] = cnn_pred

                try:
                    X_local_placeholder = torch.ones(
                        (A_hat_t.shape[0], 49),
                        dtype=torch.float32,
                        device=device
                    )

                    gnn_aux_placeholder = torch.zeros(
                        (608,),
                        dtype=torch.float32,
                        device=device
                    )

                    with torch.no_grad():
                        gnn_out = gnn_model(
                            A_hat_t,
                            X_local_placeholder,
                            gnn_aux_placeholder
                        )
                        gnn_pred = float(gnn_out.item())

                    preds.append(gnn_pred)
                    detail["GNN"] = gnn_pred

                except Exception:
                    gnn_pred = np.nan
                    detail["GNN"] = "Unavailable"

            if len(preds) == 3:
                ens = float(W_XGB * xgb_pred + W_CNN * cnn_pred + W_GNN * gnn_pred)
            elif len(preds) == 2:
                ens = float((W_XGB * xgb_pred + W_CNN * cnn_pred) / (W_XGB + W_CNN))
            else:
                ens = float(xgb_pred)

            std = float(np.std(preds))

            st.markdown("### Results")
            st.write(f"**Ensemble ΔΔG:** {ens:.3f} kcal/mol")
            st.write(f"Ensemble std (XGB/CNN/GNN disagreement): {std:.3f}")
            st.json(detail)
            st.caption(
                "Negative ΔΔG → predicted stronger protein–protein binding; "
                "positive ΔΔG → predicted weaker protein–protein binding."
            )

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

                    X_array = np.asarray(X_num)
                    xgb_pred = float(xgb_model.predict(X_array[:, :4])[0])

                    preds = [xgb_pred]

                    if M4 is not None and A_hat_t is not None:
                        cnn_pred = float(
                            cnn_model.predict([M4, X_num], verbose=0).reshape(-1)[0]
                        )
                        preds.append(cnn_pred)

                        try:
                            X_local_placeholder = torch.ones(
                                (A_hat_t.shape[0], 49),
                                dtype=torch.float32,
                                device=device
                            )

                            gnn_aux_placeholder = torch.zeros(
                                (608,),
                                dtype=torch.float32,
                                device=device
                            )

                            with torch.no_grad():
                                gnn_out = gnn_model(
                                    A_hat_t,
                                    X_local_placeholder,
                                    gnn_aux_placeholder
                                )
                                gnn_pred = float(gnn_out.item())

                            preds.append(gnn_pred)

                        except Exception:
                            gnn_pred = np.nan

                    else:
                        cnn_pred = np.nan
                        gnn_pred = np.nan

                    if len(preds) == 3:
                        ens = float(W_XGB * xgb_pred + W_CNN * cnn_pred + W_GNN * gnn_pred)
                    elif len(preds) == 2:
                        ens = float((W_XGB * xgb_pred + W_CNN * cnn_pred) / (W_XGB + W_CNN))
                    else:
                        ens = float(xgb_pred)

                    std = float(np.std(preds))

                    rows.append({
                        "pos": pos,
                        "wt": wt,
                        "mt": mt,
                        "mutation": f"{wt}{pos}{mt}",
                        "XGB_ddG": xgb_pred,
                        "CNN_ddG": cnn_pred,
                        "GNN_ddG": gnn_pred,
                        "Ensemble_ddG": ens,
                        "ensemble_std": std
                    })

            df_scan = pd.DataFrame(rows)
            df_scan = df_scan.sort_values("Ensemble_ddG").reset_index(drop=True)
            df_scan.insert(0, "Rank", range(1, len(df_scan) + 1))

            st.markdown("### Scan Results Summary")
            st.dataframe(df_scan)


