#!/usr/bin/env python3
"""Download and plot 'lm loss' from W&B runs, caching data locally.

Runs sharing the same base name (stripping the final '-<id>' suffix) are
concatenated in order and drawn as a single curve.
"""

import argparse
import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
import wandb


# ── Configuration ────────────────────────────────────────────────────────────

WANDB_ENTITY = "alehc"
WANDB_PROJECT = "opt_v1"

CACHE_DIR = Path(__file__).resolve().parent / ".wandb_cache"


# Each entry can be:
#   "experiment_name"                           → uses default entity/project
#   {"name": "exp", "entity": "X"}              → overrides entity
#   {"name": "exp", "project": "Y"}             → overrides project
#   {"name": "exp", "entity": "X", "project": "Y"} → overrides both

EXPERIMENTS_TO_PLOT = [
    ## baselines
    "110M-n1",
    # with cosine
    # "110M-lr0.004-cos-n1",
    # "110M-lr0.003-cos-n1",
    # # OP
    # "110M-qkRMS-nPre-nFin-pst-ls-n1",
    # ######## muon
    # # "110M-muon_m0.95_urm-n1",
    # # "110M-muon_m0.95-lr0.004-n1",
    # "110M-muon_m0.95_mlr2_urm_nest-n1",
    # # "110M-muon_m0.95_urm_nest-n1",
    # # after fix
    # "110M-muon_h_m0.95_mlr2_urm-n1",
    # ################# ngpt baselines
    # "110M-master_a0-wd0-HSrow1_l2_emb_sh-std0.044-L2Norm-fz-nPre-nFin-pst-ppst-usmr-ss-lsS-qklsS-mlplsG-lgslsS-nw-n1",
    # # with projection
    # "110M-master_a0-wd0-HSrow1_l2_emb_sh_p-std0.044-ngpt-nw-n1",
    # # with cosine
    # "110M-master_a0-wd0-HSrow1_l2_emb_sh-std0.044-ngpt-nw-cos-n1",
    # # "110M-master_a0-wd0-HSrow1_l2_emb_sh-lr0.004-std0.044-ngpt-nw-cos-n1",
    # # after fix
    # "110M-master_h_a0-wd0-HSrow1_l2_it0_emb_sh-std0.044-ngpt-nw-cos-n1",
    # # FOG
    # "110M-master_h_a0-wd0-HSrow1_l2_it0_emb_sh-qkRMS-nPre-nFin-pst-ls-lgsls-n1",
    # "110M-master_h_a0-wd0-HSrow1_l2_it0_emb_sh-qkRMS-nPre-nFin-pst-ls-lgsls-nw-n1",
    # # FOG + cosine
    # "110M-master_h_a0-wd0-HSrow1_l2_it0_emb_sh-qkRMS-nPre-nFin-pst-ls-lgsls-nw-cos-n1",
    # # ngpt without weight normalization, just arch
    # "110M-lr0.004-std0.044-ngpt-n1",
    # # ngpt with colum norm for out projections
    # cos
    {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_emb_sh-std0.044-ngpt-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    # wsd
    # {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_emb_sh-std0.044-ngpt-nw-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    # # ngpt with colum norm for out projections, warmup
    # {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_emb_sh-lr0.004-std0.044-ngpt-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    
    ## baseline with weight normalization
    # cos
    {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_emb_sh-std0.044-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    # wsd
    # {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_emb_sh-std0.044-nw-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    # nOP with cosine and MLP scale
    # {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_emb_sh-gelu-qkRMS-nPre-nFin-lsS-mlplsG-lgslsS-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    ###### nOP with cosine, upscaling embeddings, no mlp scale
    # {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_emb_sh-gelu-qkRMS-nPre-nFin-ls0.083S-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    ###### with 1/sqrt(L) scalig
    # {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_emb_sh-gelu-qkRMS-nPre-nFin-ls0.28S-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    # same as above, but nFOG, so post-norm
    {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_emb_sh-qkRMS-nPre-nFin-pst-ls0.083S-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    # {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_emb_sh1-lr0.004-qkRMS-nPre-nFin-pst-ls0.083S-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    # {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_emb_sh1-lr0.001-qkRMS-nPre-nFin-pst-ls0.083S-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    "110M-muon_h_m0.95_mlr2_urm-n1",

    # matrix norm
    {"name": "110M-master_h_a0-wd0-HSflat22.62_l2_it0_emb-qkRMS-nPre-nFin-pst-lsS-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    # with normlaized updates
    {"name": "110M-master_h_a0-wd0-HSflat22.62_l2_it0_u_emb-lr0.004-qkRMS-nPre-nFin-pst-lsS-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    ################## hypermuon
    # "110M-master_o_b0.9_none-wd0-HSrow1_l2_sh-lr0.004-std0.044-ngpt-nw-n1",
    "110M-master_o_b0.9-wd0-HSrow1_l2_u_sh-lr0.004-std0.044-ngpt-nw-n1",
    # "110M-master_o_b0.9_mlr2_urm-wd0-HSrow1_l2_embNO_sh-std0.044-ngpt-nw-n1",
    # "110M-master_o_b0.9-wd0-HSrow1_l2_u_embNO_sh-lr0.004-std0.044-ngpt-nw-n1",
    # "110M-master_o_b0.9_urm-wd0-HSrow1_l2_embNO_sh-std0.044-ngpt-nw-n1",
    # "110M-master_o_b0.9_mlr2_urm-wd0-HSrow1_l2_sh-std0.044-ngpt-nw-n1",
    # after fix
    # "110M-master_h_o_b0.9-wd0-HSrow1_l2_it0_u_sh-lr0.004-std0.044-ngpt-nw-n1",
    # hypermuon with nFog
    # {"name": "110M-master_h_o_b0.9_mlr2_urm_a0-wd0-HSembed1_l2_it0_emb_sh1-qkRMS-nPre-nFin-pst-ls0.083S-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    # without split heads
    # {"name": "110M-master_h_o_b0.9_mlr2_urm_a0-wd0-HSembed1_l2_it0_emb-qkRMS-nPre-nFin-pst-ls0.083S-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    # nfog with cosine and hypermuon, lr 0.004 and mlr2.
    # forgot embNO here
    {"name": "110M-master_h_o_b0.9_mlr2_urm_a0-wd0-HSembed1_l2_it0_emb-lr0.004-qkRMS-nPre-nFin-pst-ls0.083S-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    

    # ngpt with hypermuon, lr 0.004, --hs-u and mlr2 (best)
    {"name": "110M-master_h_o_b0.9_mlr2_urm-wd0-HSembed1_l2_it0_u-lr0.004-std0.044-ngpt-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    # nfog with hypermuon, fixed embNO and lr 0.008, --hs-u
    {"name": "110M-master_h_o_b0.9_urm_a0-wd0-HSembed1_l2_it0_u_embNO-lr0.008-qkRMS-nPre-nFin-pst-lsS-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},


    # nfog with hypermuon, fixed embNO and lr 0.008, no -hs-u
    {"name": "110M-master_h_o_b0.9_urm_a0-wd0-HSembed1_l2_it0_embNO-lr0.008-qkRMS-nPre-nFin-pst-lsS-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    # nfog with hypermuon, fixed embNO and lr 0.004, no -hs-u
    {"name": "110M-master_h_o_b0.9_urm_a0-wd0-HSembed1_l2_it0_embNO-lr0.004-qkRMS-nPre-nFin-pst-lsS-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    
    {"name": "110M-master_h_o_b0.9_urm_a0-wd0-HSembed1_l2_it0_embNO-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    # nfog with hypermuon, fixed embNO and lr 0.012, no --hs-u
    {"name": "110M-master_h_o_b0.9_urm_a0-wd0-HSembed1_l2_it0_embNO-lr0.012-qkRMS-nPre-nFin-pst-lsS-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    # nfog with adam, hs u, lr 0.012
    {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_u_emb-lr0.012-qkRMS-nPre-nFin-pst-lsS-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    # nfog with adam, hs u, lr 0.016
    {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_u_emb-lr0.016-qkRMS-nPre-nFin-pst-lsS-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    # nfog with hypermuon, fixed embNO and lr 0.016, --hs-u
    {"name": "110M-master_h_o_b0.9_urm_a0-wd0-HSembed1_l2_it0_u_embNO-lr0.016-qkRMS-nPre-nFin-pst-lsS-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    # nfog with hypermuon, fixed embNO and lr 0.016, no --hs-u
    {"name": "110M-master_h_o_b0.9_urm_a0-wd0-HSembed1_l2_it0_embNO-lr0.016-qkRMS-nPre-nFin-pst-lsS-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    
    # nfog with adam, hs u, lr 0.008, matrix norm
    {"name": "110M-master_h_a0-wd0-HSflat22.62_l2_it0_u_emb-lr0.008-qkRMS-nPre-nFin-pst-lsS-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_a0-wd0-HSflat22.62_l2_it0-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    # nfog with hypermuon, lr 0.004, --hs-u and mlr2, and emb O
    {"name": "110M-master_h_o_b0.9_mlr2_urm_a0-wd0-HSembed1_l2_it0_u_emb-lr0.004-qkRMS-nPre-nFin-pst-lsS-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    # best baseline nfog (row/col), no u, lr 0.002, but with no layerscale scale (so just set to 1) to check if using this 1/sqrt(d) actually helps. seems to be the same as before
    {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_emb-qkRMS-nPre-nFin-pst-ls1S-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    # just muon fog with mlr 4

    {"name": "110M-muon_h_m0.95_mlr2_urm-qkRMS-nPre-nFin-pst-lsS-lgslsS-ue-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    {"name": "110M-muon_h_m0.95_mlr4_urm-qkRMS-nPre-nFin-pst-lsS-lgslsS-ue-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    {"name": "110M-muon_h_m0.95_mlr8_urm-qkRMS-nPre-nFin-pst-lsS-lgslsS-ue-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    {"name": "110M-muon_h_m0.95_mlr4_mns10_urm-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    # nfog without hs embed and logit layerscale scale 1
    {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    # with old layerscale scale
    {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0-qkRMS-nPre-nFin-pst-lsS-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    # nfog with hypermuon, lr 0.004, and mlr2, layerscale scale 1, no hs-embed
    {"name": "110M-master_h_o_b0.9_mlr2_urm_a0-wd0-HSembed1_l2_it0-lr0.004-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    # nfog with hypermuon, lr 0.004, and mlr4, layerscale scale 1, no hs-embed
    {"name": "110M-master_h_o_b0.9_mlr4_urm_a0-wd0-HSembed1_l2_it0-lr0.004-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr2_urm_a0-wd0-HSembed1_l2_it0_u_embNO-lr0.004-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_u_emb-lr0.008-qkRMS-nPre-nFin-pst-lsS-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr4_urm_a0-wd0-HSembed1_l2_it0_u-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr2_urm_a0-wd0-HSembed1_l2_it0_u-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},


    {"name": "110M-master_h_o_b0.9_mlr4_urm_a0-wd0-HSembed1_l2_it0_embNO-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr4_pmo1_urm_a0-wd0-HSembed1_l2_it0_embNO-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr2_pmo1_urm_a0-wd0-HSembed1_l2_it0_embNO-lr0.004-qkRMS-nPre-nFin-pst-lsS-lgsls1-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr2_pmo1_urm_a0-wd0-HSembed1_l2_it0_embNO-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr4_pmo1_urm_a0-wd0-HSembed1_l2_it0-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr2_pmo1_urm_a0-wd0-HSembed1_l2_it0-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr2_pmo1_urm_a0-wd0-HSembed1_l2_it0-lr0.004-qkRMS-nPre-nFin-pst-lsS-lgsls1-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},


    {"name": "110M-master_h_o_b0.9_mlr4_urm_a0-wd0-HSembed1_l2_it0_u_embNO-lr0.004-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr4_urm_a0-wd0-HSembed1_l2_it0_u-lr0.004-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_a0-wd0-HScol1_l2_it0-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr4_none_a0-wd0-HSembed1_l2_it0-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr4_mns10_none_a0-wd0-HSembed1_l2_it0-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr4_none_a0-wd0-HSembed1_l2_it0-lr0.004-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr8_none_a0-wd0-HSembed1_l2_it0-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr4_urm_a0-wd0-HSembed1_l2_it0-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr4_urm_a0-wd0-HSembed1_l2_it0_u_embNO-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr8_none_a0-wd0-HSembed1_l2_it0-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr4_urm_a0-wd0-HSembed1_l2_it0_u-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    {"name": "110M-master_h_o_b0.9_mlr4_urm_a0-wd0-HSembed1_l2_it0_u-lr0.001-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    
    {"name": "110M-master_h_o_b0.9_mlr4_shsc_a0-wd0-HSembed1_l2_it0-lr0.004-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr8_shsc_a0-wd0-HSembed1_l2_it0-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr4_shsc_a0-wd0-HSembed1_l2_it0-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr4_shsc_a0-wd0-HSembed1_l2_it0_embNO-lr0.004-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr8_shsc_a0-wd0-HSembed1_l2_it0_embNO-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},


    {"name": "110M-master_h_o_b0.9_mlr8_shsc_a0-wd0-HSembed1_l2_it0-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_o_b0.9_mlr4_shsc_a0-wd0-HSembed1_l2_it0_embNO-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},


    {"name": "110M-master_h_o_b0.9_mlr4_pmo1_none_a0-wd0-HSembed1_l2_it0-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_u-lr0.008-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},


    {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_emb-lr0.003-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    {"name": "110M1-master_h_a0-wd0-HSflat1_l2_it0_emb-lr0.003-elr-qkRMS-pst-png-ls1S0.044-lgsls1S-ue-nw-cos-n2", "entity": "epfl-relay", "project": "megatron_opt_v1"},

    {"name": "110M1-master_h_a0-wd0-HSembed1_l2_it0_emb-lr0.003-elr-qkRMS-nPre-nFin-pst-png-ls1S0.044-lgsls1S-ue-nw-cos-n2", "entity": "epfl-relay", "project": "megatron_opt_v1"},


    ### 390m
    # nFog
    # {"name": "390M-master_h_a0-wd0-HSembed1_l2_it0_emb_sh-qkRMS-nPre-nFin-pst-ls0.0625S-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    # # baseline
    # {"name": "390M-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
]


PRESETS = {
    "default": EXPERIMENTS_TO_PLOT,
    "wsdsweep": [
        # True llama adam baseline for reference.
        "110M-n1",

        # Our main master to beat.
        #{"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_emb_sh-qkRMS-nPre-nFin-pst-ls0.083S-lgslsS-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
        {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_emb-lr0.003-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-cos-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},

        # WSD shapes.
        {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_emb-lr0.002-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
        {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_emb-lr0.003-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
        {"name": "110M-master_h_a0-wd0-HSembed1_l2_it0_emb-lr0.004-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"},
    ] + [
        {"name": f"110M-master_h_a0-wd0-HSembed1_l2_it0_emb-lr{lr}-elr-qkRMS-nPre-nFin-pst-lsS-lgsls1S-ue-nw-WSD{wsd}-n1", "entity": "epfl-relay", "project": "megatron_opt_v1"}
        for wsd in ["exponential", "linear", "cosine", "minus_cbrt", "minus_cbcrt", "power2", "power3", "sqrt_pow2"]
        for lr in [0.002, 0.003, 0.004]
    ]
}

EXPERIMENTS_TO_PLOT = PRESETS["default"]
# EXPERIMENTS_TO_PLOT = PRESETS["wsdsweep"]

METRIC_KEY = "lm loss"
STEP_KEY = "consumed-tokens"

# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_experiment(exp, default_entity: str, default_project: str) -> tuple[str, str, str]:
    """Parse an experiment entry into (entity, project, name).

    Accepts either a plain string (uses defaults) or a dict with optional
    'entity' and 'project' keys that override the defaults.
    """
    if isinstance(exp, dict):
        name = exp["name"]
        entity = exp.get("entity", default_entity)
        project = exp.get("project", default_project)
    else:
        name = str(exp)
        entity = default_entity
        project = default_project
    return entity, project, name


def strip_run_suffix(name: str) -> str:
    """Remove a trailing job/run-id suffix like '-1570020' or '-j1552378'."""
    return re.sub(r"-j?\d{5,}$", "", name)


def cache_path_for(run_id: str) -> Path:
    return CACHE_DIR / f"{run_id}.json"


def fetch_run_history(run, keys: list[str]) -> list[dict]:
    """Download the requested keys from a run, using a local JSON cache."""
    cached = cache_path_for(run.id)
    if cached.exists():
        with open(cached) as f:
            return json.load(f)

    rows = []
    for row in run.scan_history(keys=keys, page_size=5000):
        rows.append({k: row.get(k) for k in keys})

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cached, "w") as f:
        json.dump(rows, f)
    return rows


def smooth_and_subsample(
    steps: np.ndarray, vals: np.ndarray, window: int = 50, max_points: int = 2000
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a moving-average smooth, then uniformly subsample to at most max_points."""
    if len(vals) < 2:
        return steps, vals

    smoothed = uniform_filter1d(vals.astype(float), size=min(window, len(vals)), mode="nearest")

    if len(smoothed) > max_points:
        idx = np.linspace(0, len(smoothed) - 1, max_points, dtype=int)
        return steps[idx], smoothed[idx]
    return steps, smoothed


def scrape_from_logs(group: list[wandb.apis.public.runs.Run], step_key: str, keys: list[str], tail: int) -> list[dict]:
    def extract(log_path : Path) -> dict[int, dict[str, float]]:
        res = {}
        with open(log_path) as f:
            for line in f:
                if all(key in line for key in keys + [log_step_key]):
                    #print(line)
                    it = int(re.match(fr".*{log_step_key} +(\d+)/ *{niters}.*", line).group(1))
                    row = {
                        key: float(re.match(fr".*{key}: +(\d+\.\d+E(\+|-)\d+).*", line).group(1))  # Only works for lm loss lmao.
                        for key in keys
                    }
                    res[it] = row
        return res

    # Sort in reverse chronological order to scrape as few logs as needed.
    jobids = {run: int(re.match(r".*-j?(\d{5,})$", run.name).group(1))
              for run in group}
    group = sorted(group, reverse=True, key=lambda run: jobids[run])
    cfg = group[0].config
    assert step_key == "consumed-tokens"
    step_key_mult = cfg["global_batch_size"] * cfg["seq_length"]
    niters = cfg["train_iters"]
    log_step_key = "iteration"

    #"/iopsstor/scratch/cscs/ahernnde/opts/logs/v1/110M-master_h_a0-wd0-HSembed1_l2_it0_emb-lr0.002-elr-qkRMS-nPre-nFin-pst-ls0.083S0.044-lgsls1S1-up22.62-nw-WSDpower3/checkpoints"
    result = []
    it = niters
    for run in group:
        ckpt_path = Path(cfg["save"])
        log_path = ckpt_path.parent/"slurmlogs"/f"{jobids[run]}.out"
        metrics = extract(log_path)
        while it in metrics and len(result) < tail:
            result.append({step_key: it * step_key_mult, **metrics[it]})
            it -= 1
    if len(result) == tail:
        return list(reversed(result))
    raise ValueError(f"Longs not long enough, found {len(result)} / {tail} entries")





# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity", default=WANDB_ENTITY, help="W&B entity (user or team)"
    )
    parser.add_argument("--project", default=WANDB_PROJECT, help="W&B project name")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Base experiment names to plot (overrides EXPERIMENTS_TO_PLOT)",
    )
    parser.add_argument(
        "--metric", default=METRIC_KEY, help="Metric key to download (default: 'lm loss')"
    )
    parser.add_argument(
        "--step-key", default=STEP_KEY, help="Step key (default: 'iteration')"
    )
    parser.add_argument(
        "--clear-cache", action="store_true", help="Delete cached data and re-download"
    )
    parser.add_argument(
        "--output", default=None, help="Save figure to this path instead of showing"
    )
    parser.add_argument(
        "--smooth", type=int, default=50,
        help="Moving-average window size (set to 1 to disable smoothing)",
    )
    parser.add_argument(
        "--max-points", type=int, default=2000,
        help="Max data points per curve after subsampling (default: 2000)",
    )
    parser.add_argument(
        "--tail", type=int, default=100,
        help="Number of final raw iterations to average for the ranking table (default: 100)",
    )
    parser.add_argument(
        "--scrape-from-logs", action="store_true",
        help="When set, the loss of the --tail will be extracted directly from the raw log files",
    )
    args = parser.parse_args()

    raw_experiments = args.experiments if args.experiments else EXPERIMENTS_TO_PLOT
    if not raw_experiments:
        parser.error(
            "No experiments specified. Pass --experiments or edit EXPERIMENTS_TO_PLOT."
        )

    if args.clear_cache and CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print(f"Cleared cache at {CACHE_DIR}")

    # Parse each experiment entry into (entity, project, name) and group by
    # (entity, project) so we issue one W&B query per project.
    parsed_experiments: list[tuple[str, str, str]] = []
    by_project: dict[tuple[str, str], list[str]] = defaultdict(list)
    for exp in raw_experiments:
        entity, project, name = parse_experiment(exp, args.entity, args.project)
        parsed_experiments.append((entity, project, name))
        by_project[(entity, project)].append(name)

    api = wandb.Api()

    # Fetch runs for each (entity, project) group and collect into a unified map.
    experiment_runs: OrderedDict[str, list] = OrderedDict()
    for entity, project, name in parsed_experiments:
        experiment_runs[name] = []

    for (entity, project), names in by_project.items():
        path = f"{entity}/{project}"
        name_regex = "|".join(re.escape(n) for n in names)
        filters = {"display_name": {"$regex": f"^({name_regex})(-j?\\d{{5,}})?$"}}
        runs = api.runs(path, filters=filters, per_page=1000)

        for run in runs:
            base = strip_run_suffix(run.name)
            if base in experiment_runs:
                experiment_runs[base].append(run)
            else:
                print(f"  [warn] Run '{run.name}' matched filter but base '{base}' is unknown")

    for name in experiment_runs:
        experiment_runs[name].sort(key=lambda r: r.name)

    keys = [args.step_key, args.metric]

    # ── Load & plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ranking: list[tuple[str, float, int]] = []  # (name, tail_mean, n_points)
    steps_found: list[tuple[str, list[int]]] = []  # (name, list[step])

    exp_pbar = tqdm(experiment_runs.items(), desc="Experiments", unit="exp")
    for exp_name, run_group in exp_pbar:
        exp_pbar.set_postfix_str(exp_name[:40])
        if not run_group:
            tqdm.write(f"  [skip] No runs found for '{exp_name}'")
            continue

        all_steps, all_vals = [], []
        for run in tqdm(run_group, desc=f"  Runs ({exp_name[:30]})", unit="run", leave=False):
            tqdm.write(f"  Loading {run.name} ({run.id}) …")
            rows = fetch_run_history(run, keys)
            for row in rows:
                s, v = row.get(args.step_key), row.get(args.metric)
                if s is not None and v is not None:
                    all_steps.append(s)
                    all_vals.append(v)

        if not all_steps:
            tqdm.write(f"  [skip] No data for '{exp_name}'")
            continue

        order = np.argsort(all_steps)
        all_steps = np.array(all_steps)[order]
        all_vals = np.array(all_vals)[order]

        if args.scrape_from_logs:
            assert keys[0] == args.step_key
            try:
                rows = scrape_from_logs(run_group, args.step_key, keys[1:], args.tail)
                steps_tail = [row[args.step_key] for row in rows]
                vals_tail = [row[args.metric] for row in rows]
                assert len(steps_tail) == len(vals_tail) == args.tail, f"{steps_tail, vals_tail, args.tail}"
                all_steps[-args.tail:] = steps_tail
                all_vals[-args.tail:] = vals_tail
            except PermissionError:
                print("PermissionError:", exp_name)

        # Deduplicate: for repeated steps keep the value from the latest run.
        # Runs are sorted by name so later restarts come last — keep last occurrence.
        _, unique_idx = np.unique(all_steps[::-1], return_index=True)
        unique_idx = len(all_steps) - 1 - unique_idx  # map back to original order
        unique_idx.sort()
        all_steps = all_steps[unique_idx]
        all_vals = all_vals[unique_idx]
        steps_found.append((exp_name, all_steps))

        tail_k = min(args.tail, len(all_vals))
        tail_mean = float(np.mean(all_vals[-tail_k:]))
        ranking.append((exp_name, tail_mean, len(all_vals)))

        plot_steps, plot_vals = smooth_and_subsample(
            all_steps, all_vals, window=args.smooth, max_points=args.max_points
        )
        ax.plot(plot_steps, plot_vals, linewidth=1.2, label=exp_name)

    tails = {name: tuple(steps)[-args.tail:] for name, steps in steps_found}
    if len(set(tails.values())) > 1:
        raise ValueError(f"Not all tails have same x axis:" + "\n".join(f"{name}: {tail}" for name, tail in tails.items()))

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(args.metric, fontsize=12)
    if args.metric == "lm loss":
        ax.set_ylim(2.5, 3.5)
    else:
        ax.set_ylim(0, 2)
    ax.set_title(args.metric, fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout()

    # ── Ranking table ─────────────────────────────────────────────────
    if ranking:
        ranking.sort(key=lambda x: x[1])
        max_name = max(len(r[0]) for r in ranking)
        print(f"\n{'─' * (max_name + 30)}")
        print(f"  Ranking by mean loss over last {args.tail} raw iterations")
        print(f"{'─' * (max_name + 30)}")
        for i, (name, loss, n) in enumerate(ranking, 1):
            print(f"  {i:>2}.  {name:<{max_name}}   {loss:.4f}   ({n:,} pts)")
        print(f"{'─' * (max_name + 30)}")

    if args.output:
        fig.savefig(args.output, dpi=200)
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
