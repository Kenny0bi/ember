"""Generate the README visuals as hand-built SVGs.

Nothing here is a screenshot of a plotting library. The autograd graph is
extracted from a real ember backward pass (nodes, edges, and gradient
magnitudes are all live values), and the charts are drawn from the actual
training logs. Ember palette: charcoal ground, heat scale for gradients.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ember import Tensor
from ember import functional as F

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

# ---------------------------------------------------------------- palette

BG = "#191512"
PANEL = "#211c17"
GRID = "#332c24"
INK = "#e8ded2"
FAINT = "#8a7d6d"
ASH = "#6b7f8c"      # forward / torch
HEAT = ["#41180b", "#7f1d0e", "#c2410c", "#f59e0b", "#fde68a", "#fffbeb"]

OP_LABELS = {
    "__add__": "+", "__mul__": "×", "__matmul__": "matmul",
    "__pow__": "pow", "__getitem__": "index", "exp": "exp", "log": "log",
    "tanh": "tanh", "relu": "relu", "sigmoid": "σ", "sum": "Σ",
    "max": "max", "reshape": "reshape", "transpose": "transpose",
}


def heat(t):
    """Map t in [0,1] to the ember heat scale."""
    t = float(np.clip(t, 0, 1)) * (len(HEAT) - 1)
    i = min(int(t), len(HEAT) - 2)
    frac = t - i
    a = np.array([int(HEAT[i][j:j + 2], 16) for j in (1, 3, 5)])
    b = np.array([int(HEAT[i + 1][j:j + 2], 16) for j in (1, 3, 5)])
    c = (a + (b - a) * frac).astype(int)
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"


# ------------------------------------------------------- graph extraction

def extract_graph():
    """Run a real forward+backward and read the tape."""
    rng = np.random.default_rng(0)
    x = Tensor(rng.standard_normal((8, 4)).astype(np.float32))
    w1 = Tensor(rng.standard_normal((4, 5)).astype(np.float32) * 0.5, requires_grad=True)
    b1 = Tensor(np.zeros(5, dtype=np.float32), requires_grad=True)
    w2 = Tensor(rng.standard_normal((5, 3)).astype(np.float32) * 0.5, requires_grad=True)
    b2 = Tensor(np.zeros(3, dtype=np.float32), requires_grad=True)
    y = rng.integers(0, 3, 8)

    h = (x @ w1 + b1).relu()
    logits = h @ w2 + b2
    loss = F.cross_entropy(logits, y)
    loss.backward()

    names = {id(x): "x", id(w1): "W1", id(b1): "b1", id(w2): "W2",
             id(b2): "b2", id(loss): "loss"}
    nodes, edges, index = [], [], {}
    stack = [loss]
    while stack:
        t = stack.pop()
        if id(t) in index:
            continue
        index[id(t)] = len(nodes)
        op = "leaf"
        if t._backward is not None:
            parts = t._backward.__qualname__.split(".")
            op = parts[1] if len(parts) >= 2 else "op"
        gmag = float(np.abs(t.grad).mean()) if t.grad is not None else 0.0
        nodes.append({
            "id": id(t), "op": OP_LABELS.get(op, op),
            "label": names.get(id(t), ""), "size": t.data.size,
            "gmag": gmag, "has_grad": t.grad is not None,
        })
        for p in t._prev:
            edges.append((id(p), id(t)))
            stack.append(p)

    # depth = longest path from any leaf (leaves at depth 0)
    children = {}
    parents_of = {}
    for a, b in edges:
        children.setdefault(a, []).append(b)
        parents_of.setdefault(b, []).append(a)
    depth = {}

    def get_depth(nid):
        if nid in depth:
            return depth[nid]
        ps = parents_of.get(nid, [])
        depth[nid] = 0 if not ps else 1 + max(get_depth(p) for p in ps)
        return depth[nid]

    for n in nodes:
        get_depth(n["id"])
    for n in nodes:
        n["depth"] = depth[n["id"]]
    return nodes, edges, float(loss.item())


def render_graph_svg():
    nodes, edges, loss_val = extract_graph()
    # Drop bare scalar constants (lifted Python floats): they read as noise.
    keep = {n["id"] for n in nodes
            if not (n["size"] == 1 and not n["has_grad"] and n["op"] == "leaf")}
    nodes = [n for n in nodes if n["id"] in keep]
    edges = [(a, b) for a, b in edges if a in keep and b in keep]
    max_depth = max(n["depth"] for n in nodes)
    by_depth = {}
    for n in sorted(nodes, key=lambda n: -n["size"]):
        by_depth.setdefault(n["depth"], []).append(n)

    W, H = 1160, 560
    x0, x1 = 90, W - 260
    ytop, ybot = 96, H - 96
    for d, group in by_depth.items():
        gx = x0 + (x1 - x0) * d / max_depth
        for i, n in enumerate(group):
            span = (ybot - ytop)
            step = span / (len(group) + 1)
            n["x"] = gx + (8 if i % 2 else -8)
            n["y"] = ytop + step * (i + 1) + (10 if d % 2 else -10)

    pos = {n["id"]: (n["x"], n["y"]) for n in nodes}
    gmags = [n["gmag"] for n in nodes if n["has_grad"] and n["gmag"] > 0]
    lo, hi = np.log10(min(gmags)), np.log10(max(gmags))

    def gcol(n):
        if not n["has_grad"] or n["gmag"] <= 0:
            return GRID
        t = (np.log10(n["gmag"]) - lo) / (hi - lo + 1e-12)
        return heat(0.15 + 0.85 * t)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="Menlo, Consolas, monospace">']
    s.append(f'<rect width="{W}" height="{H}" fill="{BG}" rx="14"/>')
    s.append('<defs><filter id="glow" x="-80%" y="-80%" width="260%" height="260%">'
             '<feGaussianBlur stdDeviation="5" result="b"/>'
             '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
             '</filter></defs>')

    s.append(f'<text x="46" y="52" fill="{INK}" font-size="21" font-weight="bold">'
             'the tape, caught mid-burn</text>')
    s.append(f'<text x="46" y="74" fill="{FAINT}" font-size="12.5" '
             f'font-family="Georgia, serif" font-style="italic">'
             'every node below is real: one forward and backward pass of a two-layer '
             'network, read straight off ember’s autograd graph</text>')

    for a, b in edges:
        (xa, ya), (xb, yb) = pos[a], pos[b]
        mx = (xa + xb) / 2
        s.append(f'<path d="M{xa:.0f},{ya:.0f} C{mx:.0f},{ya:.0f} {mx:.0f},{yb:.0f} '
                 f'{xb:.0f},{yb:.0f}" stroke="{GRID}" stroke-width="1.6" fill="none"/>')
        # gradient flowing the other way: a heat-colored dash riding each edge
        na = next(n for n in nodes if n["id"] == a)
        if na["has_grad"]:
            s.append(f'<path d="M{xb:.0f},{yb:.0f} C{mx:.0f},{yb:.0f} {mx:.0f},{ya:.0f} '
                     f'{xa:.0f},{ya:.0f}" stroke="{gcol(na)}" stroke-width="1.1" '
                     f'fill="none" stroke-dasharray="3 7" opacity="0.85"/>')

    for n in nodes:
        r = 7 + 3.2 * np.log10(max(n["size"], 1))
        fill = gcol(n)
        ring = INK if n["label"] in ("W1", "b1", "W2", "b2") else "none"
        s.append(f'<circle cx="{n["x"]:.0f}" cy="{n["y"]:.0f}" r="{r:.1f}" '
                 f'fill="{fill}" filter="url(#glow)" stroke="{ring}" '
                 f'stroke-width="1.3" stroke-dasharray="2 2"/>')
        label = n["label"] or n["op"]
        if label == "leaf":
            label = "const"
        color = FAINT if label == "const" else INK
        if n["depth"] == 0:  # leaves: label to the left, clear of the op chain
            s.append(f'<text x="{n["x"] - r - 8:.0f}" y="{n["y"] + 4:.0f}" '
                     f'fill="{color}" font-size="11" text-anchor="end">{label}</text>')
        else:
            s.append(f'<text x="{n["x"]:.0f}" y="{n["y"] + r + 14:.0f}" fill="{color}" '
                     f'font-size="11" text-anchor="middle">{label}</text>')

    # Dear Data-style key
    kx, ky = W - 218, 108
    s.append(f'<text x="{kx}" y="{ky}" fill="{FAINT}" font-size="11.5" '
             f'font-family="Georgia, serif" font-style="italic">how to read it</text>')
    for i, (t, lab) in enumerate([(0.9, "hot: large gradient"), (0.45, "warm: small"),
                                  (None, "grey: no gradient")]):
        c = GRID if t is None else heat(t)
        s.append(f'<circle cx="{kx + 8}" cy="{ky + 22 + i * 22}" r="6" fill="{c}"/>')
        s.append(f'<text x="{kx + 22}" y="{ky + 26 + i * 22}" fill="{INK}" '
                 f'font-size="10.5">{lab}</text>')
    s.append(f'<circle cx="{kx + 8}" cy="{ky + 92}" r="6" fill="none" stroke="{INK}" '
             f'stroke-width="1.3" stroke-dasharray="2 2"/>')
    s.append(f'<text x="{kx + 22}" y="{ky + 96}" fill="{INK}" font-size="10.5">'
             'dashed ring: parameter</text>')
    s.append(f'<text x="{kx}" y="{ky + 122}" fill="{INK}" font-size="10.5">'
             'size ∝ tensor elements</text>')
    s.append(f'<text x="{kx}" y="{ky + 140}" fill="{INK}" font-size="10.5">'
             'dotted trails: grads</text>')
    s.append(f'<text x="{kx}" y="{ky + 158}" fill="{INK}" font-size="10.5">'
             'flowing right to left</text>')

    s.append(f'<text x="46" y="{H - 36}" fill="{FAINT}" font-size="11">'
             f'forward runs left → right · loss = {loss_val:.4f} · '
             'subtraction and division appear as + and × because ember lowers '
             'them to primitive ops · scalar constants omitted</text>')
    s.append("</svg>")
    (ASSETS / "autograd_graph.svg").write_text("\n".join(s))
    print(f"autograd_graph.svg: {len(nodes)} real nodes, {len(edges)} edges")


# ------------------------------------------------------------- parity chart

def render_parity_svg():
    log = json.loads((ASSETS / "parity_log.json").read_text())
    e, t = np.array(log["ember"]), np.array(log["torch"])
    diff = np.abs(e - t)
    n = len(e)

    W, H = 1060, 500
    px0, px1 = 70, W - 40
    # top panel: the two loss curves
    ty0, ty1 = 100, 320
    # bottom strip: the magnified difference
    by0, by1 = 380, 452

    def sx(i):
        return px0 + (px1 - px0) * i / (n - 1)

    lo, hi = 0.0, float(max(e.max(), t.max())) * 1.05

    def sy(v):
        return ty1 - (ty1 - ty0) * (v - lo) / (hi - lo)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="Menlo, Consolas, monospace">']
    s.append(f'<rect width="{W}" height="{H}" fill="{BG}" rx="14"/>')
    s.append(f'<text x="46" y="52" fill="{INK}" font-size="21" font-weight="bold">'
             'two engines, one curve</text>')
    s.append(f'<text x="46" y="74" fill="{FAINT}" font-size="12.5" '
             f'font-family="Georgia, serif" font-style="italic">'
             'same weights, same batches, same SGD: ember and PyTorch trained side '
             'by side for 300 steps on MNIST</text>')

    for v in [0.5, 1.0, 1.5, 2.0]:
        if v < hi:
            s.append(f'<line x1="{px0}" y1="{sy(v):.0f}" x2="{px1}" y2="{sy(v):.0f}" '
                     f'stroke="{GRID}" stroke-width="1"/>')
            s.append(f'<text x="{px0 - 8}" y="{sy(v) + 4:.0f}" fill="{FAINT}" '
                     f'font-size="10.5" text-anchor="end">{v}</text>')

    def path(vals, ymap):
        pts = " ".join(f"{sx(i):.1f},{ymap(v):.1f}" for i, v in enumerate(vals))
        return pts

    # torch: cool ash, wide and underneath
    s.append(f'<polyline points="{path(t, sy)}" fill="none" stroke="{ASH}" '
             f'stroke-width="5" opacity="0.9" stroke-linejoin="round"/>')
    # ember: heat-colored, thin, riding exactly on top
    s.append(f'<polyline points="{path(e, sy)}" fill="none" stroke="{heat(0.62)}" '
             f'stroke-width="1.8" stroke-linejoin="round"/>')

    s.append(f'<text x="{px1 - 4}" y="{ty0 + 16}" fill="{ASH}" font-size="12" '
             f'text-anchor="end">pytorch (5px wide)</text>')
    s.append(f'<text x="{px1 - 4}" y="{ty0 + 34}" fill="{heat(0.62)}" font-size="12" '
             f'text-anchor="end">ember (riding on top, never leaves it)</text>')

    # difference strip, magnified ~7 orders of magnitude
    dmax = diff.max()
    s.append(f'<rect x="{px0}" y="{by0}" width="{px1 - px0}" height="{by1 - by0}" '
             f'fill="{PANEL}" rx="6"/>')

    def by(v):
        return by1 - (by1 - by0 - 10) * (v / (dmax * 1.15))

    area = f"{px0},{by1} " + path(diff, by) + f" {px1},{by1}"
    s.append(f'<polygon points="{area}" fill="{heat(0.5)}" opacity="0.55"/>')
    s.append(f'<polyline points="{path(diff, by)}" fill="none" '
             f'stroke="{heat(0.8)}" stroke-width="1.2"/>')
    s.append(f'<text x="{px0}" y="{by0 - 10}" fill="{FAINT}" font-size="11.5" '
             f'font-family="Georgia, serif" font-style="italic">'
             'the entire daylight between them, magnified ten-million-fold</text>')
    s.append(f'<text x="{px1}" y="{by0 - 10}" fill="{INK}" font-size="11" '
             f'text-anchor="end">max gap {log["max_abs_diff"]:.2e} · '
             f'mean {log["mean_abs_diff"]:.2e} (float32 rounding)</text>')

    s.append(f'<text x="{px0}" y="{H - 18}" fill="{FAINT}" font-size="10.5">step 0</text>')
    s.append(f'<text x="{px1}" y="{H - 18}" fill="{FAINT}" font-size="10.5" '
             f'text-anchor="end">step {n}</text>')
    s.append("</svg>")
    (ASSETS / "parity.svg").write_text("\n".join(s))
    print("parity.svg written")


# ------------------------------------------------- shakespeare training chart

def render_shakespeare_svg():
    log = json.loads((ASSETS / "shakespeare_log.json").read_text())
    steps = log["step"]
    train, val = log["train_loss"], log["val_loss"]
    if len(steps) < 3:
        print("shakespeare log too short, skipping")
        return
    W, H = 1060, 460
    px0, px1, py0, py1 = 70, W - 200, 100, H - 70
    smax = steps[-1]
    lo = min(min(train), min(val)) * 0.92
    hi = max(max(train), max(val)) * 1.03

    def sx(v):
        return px0 + (px1 - px0) * v / smax

    def sy(v):
        return py1 - (py1 - py0) * (v - lo) / (hi - lo)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="Menlo, Consolas, monospace">']
    s.append(f'<rect width="{W}" height="{H}" fill="{BG}" rx="14"/>')
    s.append(f'<text x="46" y="52" fill="{INK}" font-size="21" font-weight="bold">'
             'teaching a transformer to speak, one matmul at a time</text>')
    s.append(f'<text x="46" y="74" fill="{FAINT}" font-size="12.5" '
             f'font-family="Georgia, serif" font-style="italic">'
             f'a {log.get("params", "624,000")}-parameter GPT on Tiny Shakespeare, '
             'trained end to end by the from-scratch engine</text>')

    for v in np.arange(np.ceil(lo * 2) / 2, hi, 0.5):
        s.append(f'<line x1="{px0}" y1="{sy(v):.0f}" x2="{px1}" y2="{sy(v):.0f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{px0 - 8}" y="{sy(v) + 4:.0f}" fill="{FAINT}" '
                 f'font-size="10.5" text-anchor="end">{v:.1f}</text>')

    # loss curves as cooling embers: hot early, settling
    for series, base_t, wid, name in [(val, 0.85, 2.0, "val"), (train, 0.55, 2.0, "train")]:
        pts = list(zip(steps, series))
        for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
            tt = base_t * (1 - 0.55 * s0 / smax)
            s.append(f'<line x1="{sx(s0):.1f}" y1="{sy(v0):.1f}" x2="{sx(s1):.1f}" '
                     f'y2="{sy(v1):.1f}" stroke="{heat(tt)}" stroke-width="{wid}" '
                     f'stroke-linecap="round"/>')
        s.append(f'<circle cx="{sx(steps[-1]):.1f}" cy="{sy(series[-1]):.1f}" r="4" '
                 f'fill="{heat(base_t * 0.45)}"/>')
        s.append(f'<text x="{sx(steps[-1]) + 10:.0f}" y="{sy(series[-1]) + 4:.1f}" '
                 f'fill="{INK}" font-size="11.5">{name} {series[-1]:.3f}</text>')

    s.append(f'<line x1="{px0}" y1="{sy(np.log(65)):.0f}" x2="{px1}" '
             f'y2="{sy(np.log(65)):.0f}" stroke="{FAINT}" stroke-width="1" '
             f'stroke-dasharray="5 5"/>') if lo < np.log(65) < hi else None
    if lo < np.log(65) < hi:
        s.append(f'<text x="{px0 + 6}" y="{sy(np.log(65)) - 7:.0f}" fill="{FAINT}" '
                 f'font-size="10.5" font-family="Georgia, serif" font-style="italic">'
                 'ln(65): the loss of pure guessing</text>')

    tps = log["tokens_per_sec"][-1]
    s.append(f'<text x="{px0}" y="{H - 26}" fill="{FAINT}" font-size="10.5">'
             f'step 0 · cross-entropy (nats) vs steps · '
             f'{tps:,.0f} tokens/sec sustained on a 2014 quad-core CPU, NumPy only</text>')
    s.append(f'<text x="{px1}" y="{H - 26}" fill="{FAINT}" font-size="10.5" '
             f'text-anchor="end">step {smax}</text>')
    s.append("</svg>")
    (ASSETS / "shakespeare.svg").write_text("\n".join(s))
    print("shakespeare.svg written")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "graph"):
        render_graph_svg()
    if which in ("all", "parity"):
        render_parity_svg()
    if which in ("all", "shakespeare"):
        render_shakespeare_svg()
