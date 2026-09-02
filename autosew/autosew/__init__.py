"""AutoSew reproduction (arXiv 2602.22052). torch-dependent modules are imported lazily
so that parsing/validation (numpy-only) works on machines without PyTorch."""
from .config import AutoSewConfig
from .gcd_parser import parse_specification, Pattern
from .features import pattern_to_tensors, F_DIM

try:  # optional at import time; training/eval require torch
    from .model import AutoSewGNN
    from .sinkhorn import log_assignment, build_supervision, nll_loss
    from .metrics import MetricAccumulator, evaluate_batch, hard_assign_single
except ModuleNotFoundError:  # torch missing
    pass
