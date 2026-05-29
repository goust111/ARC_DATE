# Combo variant of 62593bfd (index 12)
# combo_spec: D2/0+D8/2
# Name: 方向反转 + 双背景色

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import crop_bbox, find_components, paste

# Design analysis for the combo variant
# ------------------------------------------------------------
# a) Concrete behavior and trace
# - Base task logic is unchanged up to D4-canonical grouping: each 8-connected object is
#   matched with others by shape, ignoring color and orientation.
# - The combo changes two places at once:
#   1. D2/0 reverses the destination mapping:
#      shared shapes (freq >= 2) now go to the BOTTOM; unique shapes go to the TOP.
#   2. D8/2 replaces the single background with a split background:
#      rows [0, H//2) use top_bg and rows [H//2, H) use bottom_bg.
# - Small trace: H=20, split=10, top_bg=1, bottom_bg=8, with groups A,A,B.
#   If A has height 4 and B has height 3, then original output was:
#     A -> row 0 on a uniform background, B -> row 17.
#   In the combo output it becomes:
#     B -> row 0 on the top-bg band, A/A -> row 16 on the bottom-bg band.
#   The visible difference is therefore not only "which category goes up/down", but also
#   that the destination band color reveals that the top and bottom zones are distinct.
#
# b) Complexity calibration
# - The generator still enforces at least one duplicated canonical group and at least one
#   unique shape, so the direction reversal is always observable.
# - Background colors are limited to two stable bands with a fixed middle split. This is
#   enough structure for humans to infer the dual-background rule without introducing many
#   free variables.
# - Objects remain 4-6 isolated monochromatic components with distinct colors, so repeated
#   shapes are easy to track across examples without color becoming the grouping key.
#
# c) Hidden-assumption audit of main()
# - Original `get_bg()` is no longer valid because "most common color" would only recover
#   one of the two backgrounds. We instead infer backgrounds separately from the top and
#   bottom halves, assuming the split is at H//2 and each half remains background-dominant.
# - Original `find_components(..., bg=bg)` assumed a single background color. Here we
#   normalize both background colors to a single temporary color before component finding.
# - Original transparent pasting with `transparent=bg` assumed each extracted patch had a
#   single background color inside its bbox. With two backgrounds that may fail if a bbox
#   straddles the split, so we paste by explicit object mask instead of transparent color.
# - The D4 canonical-shape logic itself is unaffected because it depends only on object
#   coordinates relative to the bbox, not on background color.
#
# d) Placement legality
# - Output-first generation is preserved. Objects are first placed in the combo output on
#   the appropriate edge band, with c0 fixed and the destination band determined by the
#   reversed shared/unique rule.
# - Both output and input placement use a stricter legality check than the seed: no object
#   overlap and no 8-neighbor touching. This preserves the assumption used by
#   `find_components(..., monochromatic=False, connectivity=8)` that each object remains an
#   isolated component even when different colors are adjacent.
# - Input scattering keeps objects entirely within one half and away from the top edge,
#   bottom edge, and the split row, so the examples visibly require motion and leave the
#   dual-background divider readable.
#
# e) Visual observability
# - A human can infer the rule from 2-3 examples because repeated shapes consistently end
#   up on the bottom band and one-off shapes on the top band, contradicting the original
#   "repeated = top" expectation.
# - The two background colors make the upper/lower zones explicit in both input and output.
#   The rule is therefore visible in ordinary pixel structure rather than hidden numeric
#   properties such as palette ids or object order.
#
# f) Visual cleanliness
# - The background is only a clean two-band partition; there are no extra markers.
# - Objects are isolated with a one-cell moat under 8-connectivity, so the scene remains
#   legible and components do not visually fuse.
#
# g) Color inference
# - Colors are preserved per object and are not remapped. The only color-level rule is the
#   persistent top-band vs bottom-band background split, which is structurally anchored by
#   the midline and therefore inferable from few examples.
#
# h) Quantitative anti-degeneracy check used after implementation
# - Counterfactual 1: compare the combo output to an "original-direction" solver on the
#   same input. Because every sample has both shared and unique shapes, outputs must differ.
# - Counterfactual 2: compare the combo output to a solver that keeps the reversed direction
#   but collapses the grid to a single background color. Generated samples must still differ,
#   confirming that the dual-background signal is active rather than decorative.
#
# i) New assumptions the generator must guarantee
# - top_bg != bottom_bg.
# - Object colors avoid both background colors.
# - Each half stays background-dominant so per-half majority recovers the intended bg.
# - Shape heights stay small enough that top-edge outputs fit within the upper half and
#   bottom-edge outputs fit within the lower half; with H in [19, 30] and bbox height <= 5,
#   this remains safely true.


def _canonical_shape(coords):
    """Canonical shape key invariant under the D4 dihedral group."""
    pts = [(int(r), int(c)) for r, c in coords]

    def _norm(ps):
        mr = min(r for r, c in ps)
        mc = min(c for r, c in ps)
        return tuple(sorted((r - mr, c - mc) for r, c in ps))

    cur = list(pts)
    best = None
    for _ in range(4):
        n = _norm(cur)
        best = n if best is None or n < best else best
        n = _norm([(-r, c) for r, c in cur])
        best = n if n < best else best
        cur = [(c, -r) for r, c in cur]
    return best


def _mask_from_coords(coords, shape):
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr, cc = int(r), int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


def _split_row(H: int) -> int:
    return int(H) // 2


def _make_dual_bg_grid(H: int, W: int, top_bg: int, bottom_bg: int) -> np.ndarray:
    split = _split_row(H)
    out = np.full((int(H), int(W)), int(bottom_bg), dtype=int)
    out[:split, :] = int(top_bg)
    return out


def _infer_backgrounds(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    split = _split_row(H)
    top_bg = int(Counter(map(int, g[:split, :].flatten())).most_common(1)[0][0])
    bottom_bg = int(Counter(map(int, g[split:, :].flatten())).most_common(1)[0][0])
    assert top_bg != bottom_bg, "expected distinct top/bottom background colors"
    return split, top_bg, bottom_bg


def _normalize_backgrounds(grid: np.ndarray, top_bg: int, bottom_bg: int) -> np.ndarray:
    g = np.array(grid, copy=True, dtype=int)
    g[g == int(bottom_bg)] = int(top_bg)
    return g


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group objects by D4-canonical shape, but under a two-band background:
    - unique shapes move to the TOP edge on the top background band
    - shared shapes move to the BOTTOM edge on the bottom background band
    - c0 is preserved
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    _split, top_bg, bottom_bg = _infer_backgrounds(g)

    normalized = _normalize_backgrounds(g, top_bg, bottom_bg)
    comps = find_components(
        normalized, bg=top_bg, connectivity=8, monochromatic=False, return_format="dict"
    )
    assert comps

    extracted = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        mask = _mask_from_coords(rel, patch.shape)
        extracted.append(
            {
                "c0": int(c0),
                "patch": patch,
                "mask": mask,
                "shape_key": _canonical_shape(rel),
            }
        )

    freq = Counter(d["shape_key"] for d in extracted)
    out = _make_dual_bg_grid(H, W, top_bg, bottom_bg)

    for d in extracted:
        patch = d["patch"]
        h, _w = map(int, patch.shape)
        dest_r0 = int(H - h) if int(freq[d["shape_key"]]) >= 2 else 0
        paste(out, patch, dest_r0, int(d["c0"]), mask=d["mask"])

    return out


def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly area pixels."""
    deltas = (
        [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
        if connectivity == 8
        else [(-1, 0), (1, 0), (0, -1), (0, 1)]
    )
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells:
                    continue
                all_r = [rr for rr, _ in cells] + [nr]
                all_c = [cc for _, cc in cells] + [nc]
                if (
                    max(all_r) - min(all_r) < max_dim
                    and max(all_c) - min(all_c) < max_dim
                ):
                    frontier.append((nr, nc))
        if not frontier:
            raise ValueError("cannot grow shape")
        cells.add(random.choice(frontier))
    mr = min(r for r, _c in cells)
    mc = min(c for _r, c in cells)
    return [(r - mr, c - mc) for r, c in cells]


def _transform_shape(coords, rot, flip):
    pts = list(coords)
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(-r, c) for r, c in pts]
    mr = min(r for r, _c in pts)
    mc = min(c for _r, c in pts)
    return [(r - mr, c - mc) for r, c in pts]


def _coords_to_patch(coords, color):
    mr = max(r for r, _c in coords)
    mc = max(c for _r, c in coords)
    patch = np.zeros((mr + 1, mc + 1), dtype=int)
    mask = np.zeros_like(patch, dtype=bool)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
        mask[int(r), int(c)] = True
    return patch, mask


def _placement_ok(
    grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, bg_colors: set[int]
) -> bool:
    h, w = mask.shape
    H, W = grid.shape
    r0 = int(r0)
    c0 = int(c0)
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False

    coords = np.argwhere(mask)
    abs_cells = {(r0 + int(r), c0 + int(c)) for r, c in coords}
    for rr, cc in abs_cells:
        if int(grid[rr, cc]) not in bg_colors:
            return False
    for rr, cc in abs_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in abs_cells:
                    continue
                if int(grid[nr, nc]) not in bg_colors:
                    return False
    return True


def _input_rows(H: int, h: int) -> list[int]:
    split = _split_row(H)
    top_rows = list(range(1, split - h))
    bottom_rows = list(range(split + 1, H - h))
    rows = top_rows + bottom_rows
    if not rows:
        raise ValueError("no legal interior rows for split-background scatter")
    return rows


_CONFIGS = [
    ([2], 2),
    ([2], 3),
    ([3], 1),
    ([3], 2),
    ([3], 3),
    ([2, 2], 1),
    ([2, 2], 2),
]


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    top_bg, bottom_bg = random.sample([0, 1, 5, 8, 9], 2)
    bg_colors = {int(top_bg), int(bottom_bg)}
    avail = [c for c in range(10) if c not in bg_colors]
    random.shuffle(avail)
    ci = 0

    group_sizes, n_unique = random.choice(_CONFIGS)

    objs = []
    canon_used: set = set()

    for gsz in group_sizes:
        area = random.randint(3, 12)
        base = _grow_shape(area, 8)
        canon = _canonical_shape(base)
        if canon in canon_used:
            raise ValueError("duplicate canonical")
        canon_used.add(canon)
        for _ in range(gsz):
            transformed = _transform_shape(
                base, random.randint(0, 3), random.choice([False, True])
            )
            objs.append({"coords": transformed, "color": avail[ci], "shared": True})
            ci += 1

    for _ in range(n_unique):
        for _attempt in range(200):
            area = random.randint(3, 12)
            shape = _grow_shape(area, 8)
            canon = _canonical_shape(shape)
            if canon not in canon_used:
                canon_used.add(canon)
                objs.append({"coords": shape, "color": avail[ci], "shared": False})
                ci += 1
                break
        else:
            raise ValueError("unique shape not found")

    out = _make_dual_bg_grid(H, W, top_bg, bottom_bg)
    for obj in objs:
        patch, mask = _coords_to_patch(obj["coords"], obj["color"])
        h, w = patch.shape
        dest_r0 = int(H - h) if obj["shared"] else 0
        placed = False
        for _attempt in range(300):
            c0 = random.randint(0, W - w)
            if _placement_ok(out, mask, dest_r0, c0, bg_colors):
                paste(out, patch, dest_r0, c0, mask=mask)
                obj["c0"] = int(c0)
                obj["patch"] = patch
                obj["mask"] = mask
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    inp = _make_dual_bg_grid(H, W, top_bg, bottom_bg)
    for obj in objs:
        patch = obj["patch"]
        mask = obj["mask"]
        h, _w = patch.shape
        c0 = int(obj["c0"])
        rows = _input_rows(H, h)
        placed = False
        for _attempt in range(300):
            r0 = random.choice(rows)
            if _placement_ok(inp, mask, r0, c0, bg_colors):
                paste(inp, patch, r0, c0, mask=mask)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo variant")
