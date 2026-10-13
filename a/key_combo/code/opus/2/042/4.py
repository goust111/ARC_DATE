from __future__ import annotations

import random
from collections import Counter
from itertools import combinations

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



def _diagonal_segments(grid: np.ndarray, bg: int) -> list[dict]:
    comps = find_components(
        grid,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        colors=[5],
        return_format="dict",
    )
    segs = []
    for comp in comps:
        coords = np.array(comp["coords"], dtype=int)
        rs = coords[:, 0]
        cs = coords[:, 1]
        sums = rs + cs
        if np.all(sums == sums[0]):
            d = rs - cs
            segs.append(
                {
                    "coords": [tuple(map(int, p)) for p in coords],
                    "s": int(sums[0]),
                    "d_min": int(d.min()),
                    "d_max": int(d.max()),
                }
            )
    return segs


def _pair_segments(segs: list[dict]) -> list[tuple[int, int]]:
    used = set[int]()
    pairs: list[tuple[int, int]] = []
    for i in range(len(segs)):
        if i in used:
            continue
        best = None
        best_dist = 10_000
        for j in range(i + 1, len(segs)):
            if j in used:
                continue
            if segs[i]["s"] == segs[j]["s"]:
                continue
            if min(segs[i]["d_max"], segs[j]["d_max"]) < max(segs[i]["d_min"], segs[j]["d_min"]):
                continue
            d = abs(segs[i]["s"] - segs[j]["s"])
            if d < best_dist:
                best_dist = d
                best = j
        if best is not None:
            used.add(i)
            used.add(best)
            pairs.append((i, best))
    return pairs


def _region_cells(H: int, W: int, s1: int, s2: int, d_min: int, d_max: int) -> list[tuple[int, int]]:
    s_lo = min(s1, s2)
    s_hi = max(s1, s2)
    if s_hi - s_lo <= 1:
        return []

    out = []
    for s in range(s_lo + 1, s_hi):
        for d in range(d_min, d_max + 1):
            if (s + d) % 2 != 0:
                continue
            r = (s + d) // 2
            c = s - r
            if 0 <= r < H and 0 <= c < W:
                out.append((r, c))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape
    bg = int(get_bg(g))

    out = g.copy()
    segs = _diagonal_segments(g, bg)
    if len(segs) < 2:
        return out

    pairs = _pair_segments(segs)
    if not pairs:
        return out

    for i, j in pairs:
        s1, s2 = segs[i]["s"], segs[j]["s"]
        d_min = max(segs[i]["d_min"], segs[j]["d_min"])
        d_max = min(segs[i]["d_max"], segs[j]["d_max"])
        if d_min > d_max:
            continue

        cells = _region_cells(H, W, s1, s2, d_min, d_max)
        if not cells:
            continue

        colors = [int(g[r, c]) for r, c in cells if g[r, c] != bg and g[r, c] != 5]
        if not colors:
            continue
        fill_color = Counter(colors).most_common(1)[0][0]
        for r, c in cells:
            if g[r, c] == bg:
                out[r, c] = fill_color

    return out


def _candidate_segments(H: int, W: int) -> list[dict]:
    out = []
    for s in range(H + W - 1):
        rmin = max(0, s - (W - 1))
        rmax = min(H - 1, s)
        length_max = rmax - rmin + 1
        for length in range(3, min(6, length_max) + 1):
            for start in range(rmin, rmax - length + 2):
                dvals = [2 * (start + i) - s for i in range(length)]
                coords = [(start + i, s - (start + i)) for i in range(length)]
                out.append(
                    {
                        "s": s,
                        "start": start,
                        "length": length,
                        "d_min": min(dvals),
                        "d_max": max(dvals),
                        "coords": coords,
                    }
                )
    return out


def _random_parallelogram_region(
    seg1: dict, seg2: dict, H: int, W: int
) -> list[tuple[int, int]]:
    d_min = max(seg1["d_min"], seg2["d_min"])
    d_max = min(seg1["d_max"], seg2["d_max"])
    return _region_cells(H, W, seg1["s"], seg2["s"], d_min, d_max)


def generate_input() -> np.ndarray:
    H = random.randint(14, 18)
    W = random.randint(14, 18)
    bg = 0
    for _ in range(200):
        g = np.full((H, W), bg, dtype=int)
        seg_candidates = _candidate_segments(H, W)
        if not seg_candidates:
            continue

        used = set[tuple[int, int]]()
        pairs: list[tuple[dict, dict]] = []
        pair_count = random.randint(2, 3)
        ok = True

        for _pair in range(pair_count):
            s1 = random.choice(seg_candidates)
            if any(c in used for c in s1["coords"]):
                continue

            candidates = [
                s2
                for s2 in seg_candidates
                if s2["s"] != s1["s"]
                and s2["d_max"] >= s1["d_min"]
                and s2["d_min"] <= s1["d_max"]
                and not any(c in used for c in s2["coords"])
            ]
            if not candidates:
                ok = False
                break

            s2 = random.choice(candidates)
            pair = (s1, s2)
            for r, c in s1["coords"]:
                used.add((r, c))
            for r, c in s2["coords"]:
                used.add((r, c))
            pairs.append(pair)

        if not ok or len(pairs) != pair_count:
            continue

        for a, b in pairs:
            for r, c in a["coords"]:
                g[r, c] = 5
            for r, c in b["coords"]:
                g[r, c] = 5

        # keep at most one region empty
        empty_idx = random.randint(-1, pair_count - 1)

        for idx, (a, b) in enumerate(pairs):
            region = _random_parallelogram_region(a, b, H, W)
            if idx == empty_idx:
                continue
            candidates = [cell for cell in region if g[cell] == bg and all(g[r2, c2] != 5 for r2, c2 in _eight_neighbors(*cell, H, W))]
            if not candidates:
                continue
            noise_n = random.randint(2, 4)
            picks = random.sample(candidates, min(noise_n, len(candidates)))

            fill_color = random.choice([1, 2, 3, 4])
            major = random.randint(max(1, len(picks) // 2 + 1), len(picks))
            for t, (r, c) in enumerate(picks):
                if t < major:
                    g[r, c] = fill_color
                else:
                    g[r, c] = random.choice([x for x in [1, 2, 3, 4] if x != fill_color])

        try:
            if main(g).ndim != 2:
                continue
        except Exception:
            continue
        return g

    raise RuntimeError("failed to generate valid puzzle-4 input")


def _eight_neighbors(r: int, c: int, H: int, W: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W:
                out.append((rr, cc))
    return out
