# VF2-Prob, Jan Hladěna, FIM UHK
import math

# ---------------------------------------------------------------------------
# Calibrated nonlinear node/edge compatibilities (see paper, Sec. "Model").
#
# Node compatibility is a logistic function of the type/label agreement; edge
# compatibility is a logistic function of edge-label agreement and the
# (continuous) edge-existence probability via its logit. All parameters are
# fixed once ("calibrated") and reused across all experiments. Both return a
# value in (0,1]; an epsilon floor keeps log-scores finite.
# ---------------------------------------------------------------------------

# node logistic: s_v = sigma(w0 + w_lab * 1[label match])
_NODE_W0 = -3.0          # mismatch prior  -> sigma(-3.0) ~ 0.047
_NODE_W_LABEL = 6.0      # match           -> sigma( 3.0) ~ 0.953

# edge logistic: s_e = sigma(e0 + e_lab * 1[label match] + e_p * logit(p))
_EDGE_W0 = -1.0
_EDGE_W_LABEL = 3.0
_EDGE_W_PROB = 2.0

_EPS = 1e-3


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def node_compat_label(q_lab: int, u_lab: int, eps: float = _EPS) -> float:
    """Calibrated logistic node compatibility over label/type agreement."""
    lab = 1.0 if q_lab == u_lab else 0.0
    z = _NODE_W0 + _NODE_W_LABEL * lab
    return max(eps, min(1.0, _sigmoid(z)))


def edge_compat_prob(label_match: bool, p_exist: float, eps: float = _EPS) -> float:
    """Calibrated logistic edge compatibility over label agreement and the
    continuous edge-existence probability (through its logit)."""
    lab = 1.0 if label_match else 0.0
    p = min(1.0 - 1e-6, max(1e-6, float(p_exist)))
    logit_p = math.log(p / (1.0 - p))
    z = _EDGE_W0 + _EDGE_W_LABEL * lab + _EDGE_W_PROB * logit_p
    return max(eps, min(1.0, _sigmoid(z)))


def safe_log(x: float) -> float:
    return math.log(max(1e-12, min(1.0, x)))
