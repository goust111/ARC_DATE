from __future__ import annotations

import random
from collections import Counter

import numpy as np

import os
import sys
from pathlib import Path


_arc_utils_path = None
for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
    for _cand_dir in (_p / "new" / "solvers", _p / "solvers"):
        _cand = _cand_dir / "arc_utils.py"
        if _cand.exists():
            _arc_utils_path = str(_cand_dir)
            break
    if _arc_utils_path is not None:
        break
if _arc_utils_path and _arc_utils_path not in sys.path:
    sys.path.insert(0, _arc_utils_path)

from arc_utils import get_bg, find_components



def _is_symmetric(marker_cells: set[tuple[int, int]], width: int) -> bool:
    for r, c in marker_cells:
        if (r, width - 1 - c) not in marker_cells:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape
    bg = int(get_bg(g))

    all_colors = [int(v) for v in g.flatten() if int(v) != bg]
    if not all_colors:
        return g.copy()
    marker_color = sorted(Counter(all_colors).items(), key=lambda x: (x[1], x[0]))[0][0]

    comps = find_components(
        np.where(g != bg, 1, 0).astype(int),
        bg=0,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )

    sprites = []
    for comp in comps:
        if comp["area"] <= 1:
            continue
        coords = np.array(comp["coords"], dtype=int)
        marker_mask = g[coords[:, 0], coords[:, 1]] == marker_color
        marker_count = int(marker_mask.sum())
        sprites.append({"coords": coords, "bbox": comp["bbox"], "marker_count": marker_count, "marker_mask": marker_mask})

    if not sprites:
        return g.copy()

    selected = max(sprites, key=lambda s: (s["marker_count"], s["coords"].shape[0]))
    r0, c0, r1, c1 = selected["bbox"]

    patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    non_marker = patch[patch != marker_color]
    base = [int(v) for v in non_marker if int(v) != bg]
    main_color = int(max(set(base), key=base.count)) if base else bg
    patch[patch == marker_color] = main_color

    marker_rel = {(int(r) - r0, int(c) - c0) for (r, c) in selected["coords"][selected["marker_mask"]]}
    if not _is_symmetric(marker_rel, patch.shape[1]):
        patch = np.fliplr(patch)

    # isolated marker pixels as anchors
    anchor_comps = find_components(
        np.where(g == marker_color, 1, 0).astype(int),
        bg=0,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    anchors = [(int(comp["coords"][0, 0]), int(comp["coords"][0, 1])) for comp in anchor_comps if comp["area"] == 1]

    out = np.full_like(g, bg)
    ph, pw = patch.shape
    for ar, ac in anchors:
        top = ar - ph // 2
        left = ac - pw // 2
        r0o = max(0, top)
        c0o = max(0, left)
        r1o = min(H, top + ph)
        c1o = min(W, left + pw)
        if r0o >= r1o or c0o >= c1o:
            continue
        sr0 = r0o - top
        sc0 = c0o - left
        out[r0o:r1o, c0o:c1o] = patch[sr0 : sr0 + (r1o - r0o), sc0 : sc0 + (c1o - c0o)]

    return out


def generate_input() -> np.ndarray:
    H = 15
    W = 15
    bg = 0
    marker_color = 9
    primary_colors = [1, 2, 3, 4, 6, 7]

    for _ in range(300):
        g = np.full((H, W), bg, dtype=int)
        occupied = np.zeros((H, W), dtype=bool)
        sprite_count = random.randint(3, 4)
        marker_counts = random.sample(range(1, 6), sprite_count)

        placed = []
        top_candidates = list(range(1, H - 4))
        left_candidates = list(range(1, W - 4))
        for i in range(sprite_count):
            color = primary_colors[i % len(primary_colors)]
            ok = False
            for _p in range(500):
                tr = random.choice(top_candidates)
                lc = random.choice(left_candidates)
                if occupied[tr : tr + 4, lc : lc + 4].any():
                    continue
                g[tr : tr + 4, lc : lc + 4] = color
                occupied[tr : tr + 4, lc : lc + 4] = True
                # markers inside sprite
                all_cells = [(r, c) for r in range(tr, tr + 4) for c in range(lc, lc + 4)]
                for rr, cc in random.sample(all_cells, marker_counts[i]):
                    g[rr, cc] = marker_color
                placed.append((tr, lc))
                ok = True
                break
            if not ok:
                break
        if len(placed) != sprite_count:
            continue

        anchors_needed = random.randint(2, 3)
        anchors = [(r, c) for r in range(H) for c in range(W) if g[r, c] == bg]
        random.shuffle(anchors)
        count = 0
        for r, c in anchors:
            if any(
                0 <= rr < H and 0 <= cc < W and g[rr, cc] != bg
                for rr in (r - 1, r, r + 1)
                for cc in (c - 1, c, c + 1)
            ):
                continue
            g[r, c] = marker_color
            count += 1
            if count >= anchors_needed:
                break

        if count < anchors_needed:
            continue
        try:
            _ = main(g)
            return g
        except Exception:
            continue

    raise RuntimeError("failed to generate valid puzzle-5 input")
