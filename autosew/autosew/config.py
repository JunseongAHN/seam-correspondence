"""AutoSew reproduction config.

Every field marked [PAPER] is explicitly stated in arXiv 2602.22052 (main or supp).
Every field marked [GAP] is NOT specified in the paper; the default is our decision
(SuperGlue conventions where applicable). Flip flags for ablation, do not silently edit.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class AutoSewConfig:
    # ---------------- features ----------------
    scale_div: float = 100.0          # [PAPER] "scale all geometric values by 1/100" (GCD units = cm -> ~meters)
    curvature_frame: str = "abs"      # [GAP] "abs": control points in panel-local absolute coords (/100),
                                      #       "rel": GarmentCode edge-relative coords as stored in JSON
    curvature_type_norm: bool = False # [GAP] False: k_t raw in {0..5} (paper-literal "22 raw features");
                                      #       True: k_t/5
    curvature_encoding: str = "tagged"  # [PAPER-vs-INDUSTRIAL] "tagged": dims 7..17 = k_t plus the ten
                                      # type-dependent parameter slots, exactly as the paper specifies.
                                      # "sagitta": dims 7.. = a signed sagitta profile (see curves.py).
                                      # The tagged union needs an exact k_t, which a DXF polyline cannot
                                      # supply -- k_t has to be refitted and flips, and eleven dimensions
                                      # change meaning at once. "sagitta" is the industrial-input track.
    sagitta_samples: int = 11         # [GAP] K for curvature_encoding="sagitta". 11 keeps in_dim at 24.
    arc_features: bool = False        # [INDUSTRIAL] append arclength/100 and arclength/chord.
                                      # The paper's dim 4 is the CHORD, but what a seam actually
                                      # matches is the arc -- the fabric edge you sew along.  On
                                      # GCD the arc agrees between the two sides of a stitch far
                                      # more often than the chord does, and only on the seam types
                                      # the model is worst at: collar/torso 66.7% -> 100% within
                                      # 10%, sleeve/torso 36.9% -> 48.2%, skirt/torso 52.7% ->
                                      # 61.0%, while same-part seams are unchanged at 85.7%.
    panel_id_mode: str = "index_norm" # [GAP] encoding of panel ID u. "index_norm": panel position in file /
                                      #       max_panels_norm; "index_raw": raw int; "random_norm": per-sample
                                      #       shuffled ids (augmentation; breaks GCD generation-order semantics)
    max_panels_norm: float = 37.0     # [PAPER-adjacent] GCD.v2 max panels per pattern = 37 (used only for *_norm)
    edge_count_minmax: tuple = (2.0, 40.0)  # [GAP] min-max bounds for per-panel edge count N_e -> [0,1].
                                      # Paper says min-max scaling but not the bounds; recompute from train
                                      # set with dataset.compute_stats() and override here.

    # ---------------- graph ----------------
    graph_connectivity: str = "panel_cycle"  # [GAP-leaning-PAPER] "linked according to the 2D topology of the
                                      # pattern" -> adjacent edges within each panel (disjoint cycles, NO
                                      # cross-panel links; global coupling is left to the OT layer).

    # ---------------- GNN ----------------
    in_dim: int = 24                  # [PAPER]
    hidden_dim: int = 512             # [PAPER] "5 hidden layers, each with 512 neurons"
    out_dim: int = 128                # [PAPER] D = 128
    num_layers: int = 5               # [PAPER] L = 5
    aggregator: str = "mean"          # [PAPER] (max also reported in Table 3)
    final_activation: str = "relu"    # [GAP] eq.(1) applies sigma every layer and f = h^L -> literal reading
                                      # is ReLU on the last layer too. "none" = SuperGlue-style linear head.
    l2_normalize: bool = False        # [GAP] paper silent; False = raw inner products
    layer_scheme: str = "last128"     # [GAP] "last128": dims 24->512x4->128 (5 SAGE layers total);
                                      # "proj": 24->512x5 then a separate Linear 512->128

    # ---------------- Sinkhorn / OT ----------------
    sinkhorn_iters: int = 100         # [PAPER] T = 100
    dustbin_init: float = 1.0         # [GAP] init of learnable z (SuperGlue uses 1.0)
    score_scale: str = "none"         # [GAP] "none": C = <f_i,f_j> raw (paper eq.3); "rsqrt_d": C/sqrt(D)
                                      # if logits explode / loss NaNs
    neg_inf: float = -1e9             # masking value (self-diagonal, padding)
    symmetric_updates: bool = False   # [GAP] False: standard alternating row/col updates (SuperGlue),
                                      # asymmetry mopped up by P' = (P + P^T)/2 (paper eq.6).
                                      # True: tie u = v each iter (fully symmetric variant).
    marginal_mode: str = "superglue"  # [GAP] real edges capacity 1, dustbin capacity M, total mass 2M,
                                      # renormalized so each real row is ~a probability distribution
                                      # (required for tau_multi to be a probability threshold, cf. §5.2)

    # ---------------- loss ----------------
    loss_both_directions: bool = True # [GAP] supervise (i,j) and (j,i) (+ (i,bin),(bin,i) for unstitched);
                                      # False: upper-triangle only
    loss_reduction: str = "mean"      # [GAP] mean over supervised entries

    # ---------------- inference ----------------
    tau_multi: float = 0.4            # [PAPER]
    hard_mode: str = "union"          # [GAP] row-wise selections merged as a union of unordered pairs;
                                      # "mutual": keep only pairs selected from both rows
    gsp_strict: bool = True           # [GAP] GSP = exact set equality (pred == GT). Paper text is
                                      # recall-only ("all GT correspondences predicted correctly");
                                      # metrics.py reports both variants.

    # ---------------- training ----------------
    lr: float = 1e-3                  # [PAPER]
    epochs: int = 18                  # [PAPER]
    optimizer: str = "adam"           # [GAP] paper silent
    batch_size: int = 16              # [GAP] paper silent
    weight_decay: float = 0.0         # [GAP]
    grad_clip: float = 0.0            # [GAP] 0 = off; set 1.0 if spikes
    seed: int = 0

    def to_dict(self):
        d = asdict(self)
        d["edge_count_minmax"] = list(d["edge_count_minmax"])
        return d


# curvature type ids [GAP: paper gives "{0...5}: straight, circular arcs, quadratic Bezier,
# cubic Bezier and B-Splines" -- 5 names for 6 slots; we fix this mapping]
KT_STRAIGHT = 0
KT_CIRCLE = 1
KT_QUADRATIC = 2
KT_CUBIC = 3
KT_BSPLINE = 4
KT_UNKNOWN = 5
