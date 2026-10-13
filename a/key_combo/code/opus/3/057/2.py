from pathlib import Path
import importlib.util
import random

import numpy as np


def _load_common():
    path = Path(__file__).with_name("_common.py")
    spec = importlib.util.spec_from_file_location("_arc057_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load_common()


def _crop_key(g, comp):
    r0, c0, r1, c1 = U.bbox(comp)
    crop = g[r0:r1 + 1, c0:c1 + 1].copy()
    mask = np.zeros_like(crop)
    for r, c in comp:
        mask[r - r0, c - c0] = int(g[r, c])
    return tuple(map(tuple, mask.tolist()))


def main(grid: np.ndarray) -> np.ndarray:
    """Erase repeated sprite pairs and frame the unique sprite once per pair.

    Input structure:
    The grid contains several small colored sprites. Most form identical
    same-color pairs, while exactly one yellow sprite has no match.

    Rule:
    Group 4-connected sprites by their cropped pixel pattern. The singleton
    group is the anomaly. Each repeated pair contributes one nested frame,
    ordered by the pair's average center column. Repeated sprites are erased;
    the anomaly remains, surrounded by one-cell rectangular frames colored by
    those ordered pairs.

    Key insight:
    The repeated objects are not output objects; they supply frame count, order,
    and colors for decorating the anomaly.
    """
    g = np.array(grid, copy=True)
    out = g.copy()
    groups = {}
    entries = []
    for col, comp in U.color_components(g):
        key = _crop_key(g, comp)
        entry = {"color": col, "comp": comp, "bbox": U.bbox(comp), "key": key}
        entries.append(entry)
        groups.setdefault(key, []).append(entry)
    anomaly = next(v[0] for v in groups.values() if len(v) == 1)
    pairs = [v[:2] for v in groups.values() if len(v) >= 2]
    for pair in pairs:
        for e in pair:
            for r, c in e["comp"]:
                out[r, c] = 0
    pairs.sort(key=lambda pair: (
        sum((e["bbox"][1] + e["bbox"][3]) / 2 for e in pair) / 2,
        sum((e["bbox"][0] + e["bbox"][2]) / 2 for e in pair) / 2,
    ))
    r0, c0, r1, c1 = anomaly["bbox"]
    for k, pair in enumerate(pairs, start=1):
        col = pair[0]["color"]
        rr0, cc0, rr1, cc1 = r0 - k, c0 - k, r1 + k, c1 + k
        if rr0 < 0 or cc0 < 0 or rr1 >= out.shape[0] or cc1 >= out.shape[1]:
            continue
        out[rr0, cc0:cc1 + 1] = col
        out[rr1, cc0:cc1 + 1] = col
        out[rr0:rr1 + 1, cc0] = col
        out[rr0:rr1 + 1, cc1] = col
    return out


def _put(g, r, c, color, pts):
    for dr, dc in pts:
        g[r + dr, c + dc] = color


def generate_input() -> np.ndarray:
    g = np.zeros((15, 15), dtype=int)
    shapes = [[(0, 0), (1, 0), (1, 1)], [(0, 1), (1, 0), (1, 1)], [(0, 0), (0, 1), (1, 1)]]
    locs = [(1, 1), (1, 5), (5, 1), (5, 6), (9, 1), (9, 6)]
    random.shuffle(locs)
    colors = [1, 2, 3]
    n_pairs = random.randint(2, 3)
    for i in range(n_pairs):
        shape = shapes[i]
        color = colors[i]
        _put(g, *locs[2 * i], color, shape)
        _put(g, *locs[2 * i + 1], color, shape)
    anomaly_shape = [(0, 0), (0, 1), (1, 0), (2, 0)]
    _put(g, 7, 10, 4, anomaly_shape)
    return g
