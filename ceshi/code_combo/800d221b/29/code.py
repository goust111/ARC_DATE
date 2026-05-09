# Combo variant of 800d221b (index 29)
# combo_spec: D1/0+D10/1+D17/0
# Name: 少数色+删除短分支+孤立路径干扰
#
# Design analysis:
# - Concrete behavior trace:
#   In the original task, every non-hub P component is treated as a real branch and recolored to the touched
#   seed's majority color. Here that would be wrong in three places at once. A short true branch from the hub to
#   a seed must first be recognized as valid, then recolored to that seed's minority color, then deleted because
#   its component is below the fixed length threshold. Meanwhile a disconnected P squiggle of similar or even
#   shorter size must survive untouched because it never connects hub and seed. The fused rule is therefore:
#   filter by connectivity-to-hub, recolor by minority, then delete only short valid branches.
# - Complexity calibration:
#   The layout keeps four seeds and only two seed colors, so the minority-vs-majority inversion is visible without
#   introducing extra color-mapping ambiguity. Every sample contains repeated support for both surviving and deleted
#   valid branches, plus an isolated P distractor that stays P in the output; this prevents the combo from
#   collapsing into either the original task or a trivial global "delete short P chains" rule.
# - Hidden-assumption audit:
#   The original solver assumes every P component outside the hub is meaningful. That fails once isolated noise is
#   added, so the new solver classifies components by whether they touch both a seed interior and the hub boundary.
#   The original center-color logic also assumes all recolored branches remain present; after adding deletion, the
#   hub center must be chosen only from surviving recolored branches. Finally, the old branch color source was the
#   majority color of the seed; here it is explicitly the minority color computed from the seed interior histogram.
# - Placement legality:
#   The generator uses a fixed legal base layout: a 3x3 hub in the center, four edge-touching framed seed
#   rectangles, four non-overlapping orthogonal branches, and disconnected noise paths in empty corners. The whole
#   scene is then rotated, which preserves non-overlap, edge-touching seeds, and the separation between true
#   branches and isolated noise.
# - Visual observability and cleanliness:
#   The key visual contrast is deliberate: one real short branch disappears, while at least one isolated short P
#   segment remains. Surviving branches recolor to the less frequent seed color, so the output visibly contradicts
#   the original majority-based behavior. Because the scene uses only one path color, two seed colors, and a single
#   central hub, the grid stays readable rather than cluttered.
# - New solver assumptions guaranteed by the generator:
#   Valid branch components touch exactly one seed frame and exactly one hub side after the hub square is removed;
#   isolated noise touches neither. At least one valid branch is shorter than the threshold, at least two valid
#   branches survive deletion, and among the survivors one minority color has a strict hub-adjacent majority so the
#   hub center is deterministic.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, label_mask


GRID_SIZE = 23
HUB_SIZE = 3
SHORT_BRANCH_THRESHOLD = 20
GEN_OUTER_RETRIES = 200

SEED_AREAS = {
    "N": (0, 9, 2, 13),
    "E": (4, 20, 8, 22),
    "S": (20, 12, 22, 16),
    "W": (13, 0, 17, 2),
}
BRANCH_POLYLINES = {
    "N": [(9, 11), (3, 11)],
    "E": [(11, 13), (11, 18), (6, 18), (6, 19)],
    "S": [(13, 11), (18, 11), (18, 14), (19, 14)],
    "W": [(11, 9), (11, 4), (15, 4), (15, 3)],
}
NOISE_POLYLINES = [
    [(6, 1), (6, 3), (4, 3)],
    [(17, 19), (19, 19), (19, 21)],
]
MAJORITY_PATTERNS = [
    {"N": 0, "E": 0, "S": 1, "W": 0},
    {"N": 1, "E": 0, "S": 1, "W": 1},
    {"N": 0, "E": 1, "S": 0, "W": 0},
    {"N": 1, "E": 1, "S": 0, "W": 1},
]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _comp_deg_score(mask: np.ndarray) -> float:
    H, W = mask.shape
    comps = components_mask(mask, connectivity=4)
    if not comps:
        return -1e9
    best_pts = max(comps, key=lambda coords: int(coords.shape[0]))
    best_sz = int(best_pts.shape[0])

    m = np.zeros(mask.shape, dtype=bool)
    m[best_pts[:, 0], best_pts[:, 1]] = True

    deg2 = 0
    deg4 = 0
    for r, c in best_pts:
        r = int(r)
        c = int(c)
        d = 0
        if r > 0 and bool(m[r - 1, c]):
            d += 1
        if r + 1 < H and bool(m[r + 1, c]):
            d += 1
        if c > 0 and bool(m[r, c - 1]):
            d += 1
        if c + 1 < W and bool(m[r, c + 1]):
            d += 1
        if d == 2:
            deg2 += 1
        elif d == 4:
            deg4 += 1
    frac2 = deg2 / max(1, best_sz)
    frac4 = deg4 / max(1, best_sz)
    return float((frac2 - frac4) * best_sz)


def _infer_path_color(g: np.ndarray, bg: int) -> int:
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors, "grid must contain non-background colors"
    return int(max(colors, key=lambda c: _comp_deg_score(g == int(c))))


def _largest_p_square(g: np.ndarray, P: int) -> tuple[int, int, int]:
    H, W = g.shape
    mP = g == int(P)
    ps = mP.astype(int).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
        s = int(ps[int(r1), int(c1)])
        if r0 > 0:
            s -= int(ps[int(r0 - 1), int(c1)])
        if c0 > 0:
            s -= int(ps[int(r1), int(c0 - 1)])
        if r0 > 0 and c0 > 0:
            s += int(ps[int(r0 - 1), int(c0 - 1)])
        return int(s)

    best = None
    for k in range(2, int(min(H, W)) + 1):
        found = None
        for r0 in range(0, int(H - k + 1)):
            r1 = int(r0 + k - 1)
            for c0 in range(0, int(W - k + 1)):
                c1 = int(c0 + k - 1)
                if rect_sum(int(r0), int(c0), int(r1), int(c1)) == int(k * k):
                    found = (int(k), int(r0), int(c0))
        if found is not None:
            best = found
    assert best is not None, "hub square not found"
    return tuple(map(int, best))


def _ordered_most_common(counts: Counter[int], *, pick_max: bool) -> int:
    items = list(counts.items())
    assert items
    if pick_max:
        return int(max(items, key=lambda kv: (int(kv[1]), -int(kv[0])))[0])
    return int(min(items, key=lambda kv: (int(kv[1]), int(kv[0])))[0])


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for idx, ((r0, c0), (r1, c1)) in enumerate(zip(points, points[1:])):
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            seg = [(int(r0), int(c)) for c in range(int(c0), int(c1) + step, step)]
        elif c0 == c1:
            step = 1 if r1 >= r0 else -1
            seg = [(int(r), int(c0)) for r in range(int(r0), int(r1) + step, step)]
        else:
            raise ValueError("non-orthogonal polyline")
        if idx:
            seg = seg[1:]
        out.extend(seg)
    return out


def _paint_cells(g: np.ndarray, cells: list[tuple[int, int]] | set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        g[int(r), int(c)] = int(color)


def _place_seed_border(g: np.ndarray, P: int, ir0: int, ic0: int, ir1: int, ic1: int) -> set[tuple[int, int]]:
    H, W = g.shape
    border_cells: set[tuple[int, int]] = set()
    for cond, cells in [
        (ir0 > 0, [(ir0 - 1, c) for c in range(ic0, ic1 + 1)]),
        (ir1 < H - 1, [(ir1 + 1, c) for c in range(ic0, ic1 + 1)]),
        (ic0 > 0, [(r, ic0 - 1) for r in range(ir0, ir1 + 1)]),
        (ic1 < W - 1, [(r, ic1 + 1) for r in range(ir0, ir1 + 1)]),
    ]:
        if cond:
            for rc in cells:
                border_cells.add((int(rc[0]), int(rc[1])))
    for cond, rc in [
        (ir0 > 0 and ic0 > 0, (ir0 - 1, ic0 - 1)),
        (ir0 > 0 and ic1 < W - 1, (ir0 - 1, ic1 + 1)),
        (ir1 < H - 1 and ic0 > 0, (ir1 + 1, ic0 - 1)),
        (ir1 < H - 1 and ic1 < W - 1, (ir1 + 1, ic1 + 1)),
    ]:
        if cond:
            border_cells.add((int(rc[0]), int(rc[1])))
    _paint_cells(g, border_cells, P)
    return border_cells


def _fill_seed_interior(
    g: np.ndarray,
    bbox: tuple[int, int, int, int],
    majority: int,
    minority: int,
    rng: random.Random,
) -> None:
    ir0, ic0, ir1, ic1 = map(int, bbox)
    cells = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
    rng.shuffle(cells)
    area = len(cells)
    n_minor = rng.randint(3, 5)
    n_minor = min(n_minor, (area - 1) // 2)
    assert n_minor >= 1
    for r, c in cells[:n_minor]:
        g[int(r), int(c)] = int(minority)
    for r, c in cells[n_minor:]:
        g[int(r), int(c)] = int(majority)


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    P = int(_infer_path_color(g, bg))

    k, r0, c0 = _largest_p_square(g, P)
    r1, c1 = int(r0 + k - 1), int(c0 + k - 1)
    hub = np.zeros((H, W), dtype=bool)
    hub[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True

    seed_mask = (g != int(bg)) & (g != int(P))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    seed_major: dict[int, int] = {}
    seed_minor: dict[int, int] = {}
    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        assert int(pts.shape[0]) >= 4, "seed interior too small"
        rn, cn = pts.min(axis=0)
        rx, cx = pts.max(axis=0)
        rn, cn, rx, cx = map(int, (rn, cn, rx, cx))
        assert int(pts.shape[0]) == int((rx - rn + 1) * (cx - cn + 1)), "seed must be rectangular"
        if rn > 0:
            assert all(int(g[rn - 1, c]) == int(P) for c in range(cn, cx + 1)), "seed missing top border"
        if rx < H - 1:
            assert all(int(g[rx + 1, c]) == int(P) for c in range(cn, cx + 1)), "seed missing bottom border"
        if cn > 0:
            assert all(int(g[r, cn - 1]) == int(P) for r in range(rn, rx + 1)), "seed missing left border"
        if cx < W - 1:
            assert all(int(g[r, cx + 1]) == int(P) for r in range(rn, rx + 1)), "seed missing right border"
        vals = Counter(int(g[int(r), int(c)]) for r, c in pts)
        assert len(vals) == 2, "seed interior must use exactly two colors"
        seed_major[int(sid)] = _ordered_most_common(vals, pick_max=True)
        seed_minor[int(sid)] = _ordered_most_common(vals, pick_max=False)

    branch_mask = (g == int(P)) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)
    valid_branches: dict[int, dict] = {}
    noise_ids: list[int] = []

    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touches = Counter()
        hub_touch = 0
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in DIRS:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    sid = int(seed_lab[rr, cc])
                    if sid != 0:
                        touches[int(sid)] += 1
                    if bool(hub[rr, cc]):
                        hub_touch += 1
        if touches and hub_touch:
            sid = _ordered_most_common(touches, pick_max=True)
            valid_branches[int(bid)] = {
                "cells": [(int(r), int(c)) for r, c in pts],
                "seed_id": int(sid),
                "length": int(pts.shape[0]),
            }
        else:
            noise_ids.append(int(bid))

    assert valid_branches, "expected at least one valid branch"
    return {
        "grid": g,
        "bg": int(bg),
        "P": int(P),
        "hub": hub,
        "hub_box": (int(k), int(r0), int(c0)),
        "seed_lab": seed_lab,
        "seed_major": seed_major,
        "seed_minor": seed_minor,
        "branch_lab": branch_lab,
        "valid_branches": valid_branches,
        "noise_ids": noise_ids,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-path network while ignoring disconnected P-colored distractors.

    Input structure:
    - bg is the most frequent color.
    - One non-bg color P forms a 3x3 solid hub, several framed branches that reach seed interiors, and extra
      isolated P path segments that do not touch the hub.
    - Seed interiors are rectangular non-P regions made from exactly two colors and wrapped by P on the non-edge
      sides.

    Rule:
    - Infer P and locate the largest solid P square as the hub.
    - Among the non-hub P components, keep only components that touch both the hub and a seed interior; leave the
      disconnected P distractors unchanged.
    - Recolor each valid branch component to the minority color of the seed interior it touches most, then delete
      any such component whose length is below the fixed threshold. Finally recolor the hub center to the most
      common surviving branch color adjacent to the hub.

    Key insight:
    - The non-hub P pixels are not all equivalent: first separate true hub-to-seed branches from isolated noise,
      then apply the counterintuitive minority-color recoloring, and only after that erase the short true branches.
    """
    info = _analyze_scene(grid)
    g = info["grid"]
    bg = int(info["bg"])
    hub = info["hub"]
    branch_lab = info["branch_lab"]
    valid_branches = info["valid_branches"]
    seed_minor = info["seed_minor"]

    out = g.copy()
    branch_color: dict[int, int] = {}
    deleted: set[int] = set()

    for bid, branch in valid_branches.items():
        col = int(seed_minor[int(branch["seed_id"])])
        branch_color[int(bid)] = int(col)
        for r, c in branch["cells"]:
            out[int(r), int(c)] = int(col)

    for bid, branch in valid_branches.items():
        if int(branch["length"]) < int(SHORT_BRANCH_THRESHOLD):
            deleted.add(int(bid))
            for r, c in branch["cells"]:
                out[int(r), int(c)] = int(bg)

    touch_cols = Counter()
    H, W = g.shape
    for r in range(H):
        for c in range(W):
            if not bool(hub[int(r), int(c)]):
                continue
            for dr, dc in DIRS:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    bid = int(branch_lab[rr, cc])
                    if bid in valid_branches and bid not in deleted:
                        touch_cols[int(branch_color[int(bid)])] += 1

    assert touch_cols, "hub must touch at least one surviving branch"
    k, r0, c0 = info["hub_box"]
    cr = int(r0 + k // 2)
    cc = int(c0 + k // 2)
    out[int(cr), int(cc)] = int(_ordered_most_common(touch_cols, pick_max=True))
    return out


def _build_case(rng: random.Random) -> np.ndarray:
    H = W = GRID_SIZE
    bg, P, A, B = rng.sample(list(range(10)), 4)
    g = np.full((H, W), int(bg), dtype=int)

    hub_r0 = hub_c0 = 10
    g[hub_r0 : hub_r0 + HUB_SIZE, hub_c0 : hub_c0 + HUB_SIZE] = int(P)

    for name, bbox in SEED_AREAS.items():
        ir0, ic0, ir1, ic1 = map(int, bbox)
        _place_seed_border(g, int(P), ir0, ic0, ir1, ic1)

    for poly in BRANCH_POLYLINES.values():
        cells = _polyline_cells(poly)
        _paint_cells(g, cells, int(P))

    for poly in NOISE_POLYLINES:
        _paint_cells(g, _polyline_cells(poly), int(P))

    pattern = rng.choice(MAJORITY_PATTERNS)
    palette = [int(A), int(B)]
    for name, bbox in SEED_AREAS.items():
        majority = int(palette[int(pattern[name])])
        minority = int(B if majority == int(A) else A)
        _fill_seed_interior(g, bbox, majority, minority, rng)

    rot_k = rng.randint(0, 3)
    if rot_k:
        g = np.rot90(g, k=rot_k)
    return g


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    info = _analyze_scene(grid)
    short_valid = 0
    long_valid = 0
    minority_flip = 0
    for branch in info["valid_branches"].values():
        sid = int(branch["seed_id"])
        if int(branch["length"]) < int(SHORT_BRANCH_THRESHOLD):
            short_valid += 1
        else:
            long_valid += 1
        if int(info["seed_major"][sid]) != int(info["seed_minor"][sid]):
            minority_flip += 1
    return {
        "noise_components": int(len(info["noise_ids"])),
        "short_valid_branches": int(short_valid),
        "long_valid_branches": int(long_valid),
        "minority_flip_branches": int(minority_flip),
    }


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        rng = random.Random(random.randrange(1 << 30))
        g = _build_case(rng)
        try:
            out = main(g)
            metrics = _fusion_metrics(g)
        except Exception:
            continue
        if metrics["noise_components"] < 1:
            continue
        if metrics["short_valid_branches"] < 1:
            continue
        if metrics["long_valid_branches"] < 2:
            continue
        if metrics["minority_flip_branches"] < 3:
            continue
        if np.array_equal(g, out):
            continue
        return g
    raise RuntimeError(f"generate_input failed after {GEN_OUTER_RETRIES} attempts")
