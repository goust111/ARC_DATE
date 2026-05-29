# Combo variant of 4a21e3da (index 17)
# combo_spec: D11/0+D12/0
# Name: 多独立精灵+多色精灵
#
# Design analysis
# - Concrete behavior:
#   Original 4a21e3da assumes one connected, single-color sprite. A border control
#   pixel emits one split ray; fg cells on the split stay, and each side piece
#   slides to its corner.
#   In this combo, the input instead contains 2-3 disconnected sprites near the
#   center, each with a distinct foreground color. The same ray cuts every sprite
#   in parallel. Each sprite is processed independently, but colors are preserved,
#   so a corner can end up containing several differently colored fragments.
# - Small mental example:
#   With one top control and a vertical split column, a green sprite and an orange
#   sprite both straddle that column. Their split-column pixels stay in place.
#   The left halves of both sprites slide to the top-left corner; the right halves
#   of both sprites slide to the top-right corner. Unlike the original task, the
#   output corner now visibly contains multiple colors from multiple independent
#   objects.
# - Complexity calibration:
#   I keep the solver general enough to handle multiple connected components and
#   preserve per-pixel colors, but the generator uses one split line at a time.
#   This is deliberate: forcing 2-3 disconnected sprites to all cross both a row
#   split and a column split makes the center visually noisy and frequently causes
#   accidental touching. One split line still fully expresses both fusion axes:
#   multiple independent sprites and color preservation in the moved pieces.
# - Old implicit assumptions that break:
#   1. "there is exactly one fg color" -> no longer true; ray detection must use
#      all non-bg, non-control pixels.
#   2. "all fg cells form one connected component" -> no longer true; movement
#      must be computed component-by-component.
#   3. "moving one region to one corner cannot collide with another region" ->
#      no longer true by default; the generator now reserves disjoint corner
#      slots so same-corner multi-color fragments remain simultaneously visible.
# - Placement legality:
#   The generator builds canonical corner-slot shapes first, then places matching
#   input pieces back near the split line with fixed gaps between sprites. This
#   guarantees no overlap, no out-of-bounds placement, and exactly 2-3 disconnected
#   foreground components.
# - Visual observability:
#   Every sample has 2-3 colors and 2-3 disconnected sprites in the input. In the
#   output, both target corners contain several colors at once. A human can see
#   that the rule is "apply the same cut-and-slide to each sprite independently,
#   while keeping colors unchanged", not a hidden numeric-color remapping.
# - New solver assumptions and generator support:
#   The solver assumes at most one active split row and at most one active split
#   column, matching the original family. The generator guarantees that foreground
#   sprites stay away from the outer border, so the control color remains the only
#   non-background border color.

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import colors_present, find_components, get_bg


def _foreground_mask(grid: np.ndarray, bg: int, ctrl: int) -> np.ndarray:
    return (grid != int(bg)) & (grid != int(ctrl))


def _split_info(ctrl_pts: list[tuple[int, int]], h: int, w: int) -> tuple[set[str], int | None, int | None]:
    edges: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        if r == 0:
            edges.add("top")
            c_split = c
        elif r == h - 1:
            edges.add("bottom")
            c_split = c
        elif c == 0:
            edges.add("left")
            r_split = r
        elif c == w - 1:
            edges.add("right")
            r_split = r
    return edges, c_split, r_split


def _touches_edges(hs: str, vs: str) -> set[str]:
    touched: set[str] = set()
    if hs in ("T", "A"):
        touched.add("top")
    if hs in ("B", "A"):
        touched.add("bottom")
    if vs in ("L", "A"):
        touched.add("left")
    if vs in ("R", "A"):
        touched.add("right")
    return touched


def _pick_dest(hs: str, vs: str, edges: set[str], r_split: int | None, c_split: int | None) -> tuple[str, str]:
    if r_split is not None:
        dh = hs
    elif "top" in edges and "bottom" not in edges:
        dh = "T"
    elif "bottom" in edges and "top" not in edges:
        dh = "B"
    else:
        dh = "T"

    if c_split is not None:
        dv = vs
    elif "left" in edges and "right" not in edges:
        dv = "L"
    elif "right" in edges and "left" not in edges:
        dv = "R"
    else:
        dv = "L"
    return dh, dv


def main(grid: np.ndarray) -> np.ndarray:
    """
    Multi-sprite, multi-color variant of the original cut-and-slide rule.

    Differences from the seed task:
    - There can be 2-3 independent sprites instead of one.
    - Several foreground colors can appear, and colors are preserved.
    - Each connected sprite is partitioned and moved independently.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    border_cols = set(map(int, g[0, :])) | set(map(int, g[h - 1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, w - 1]))
    border_cols.discard(bg)
    assert border_cols
    ctrl = int(sorted(border_cols)[0])

    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == ctrl)]
    assert ctrl_pts
    edges, c_split, r_split = _split_info(ctrl_pts, h, w)

    fg_mask = _foreground_mask(g, bg, ctrl)
    out = np.full((h, w), bg, dtype=int)

    # Stage 1: draw rays using all foreground colors, not a single fg color.
    for r, c in ctrl_pts:
        if r == 0:
            rows = np.argwhere(fg_mask[:, c]).flatten()
            if rows.size:
                stop = int(rows.max())
                for rr in range(0, stop + 1):
                    if g[rr, c] == bg:
                        out[rr, c] = ctrl
        elif r == h - 1:
            rows = np.argwhere(fg_mask[:, c]).flatten()
            if rows.size:
                stop = int(rows.min())
                for rr in range(h - 1, stop - 1, -1):
                    if g[rr, c] == bg:
                        out[rr, c] = ctrl
        elif c == 0:
            cols = np.argwhere(fg_mask[r, :]).flatten()
            if cols.size:
                stop = int(cols.max())
                for cc in range(0, stop + 1):
                    if g[r, cc] == bg:
                        out[r, cc] = ctrl
        elif c == w - 1:
            cols = np.argwhere(fg_mask[r, :]).flatten()
            if cols.size:
                stop = int(cols.min())
                for cc in range(w - 1, stop - 1, -1):
                    if g[r, cc] == bg:
                        out[r, cc] = ctrl

    for r, c in ctrl_pts:
        out[r, c] = ctrl

    # Split-line pixels stay in place, keeping their own colors.
    for r, c in np.argwhere(fg_mask):
        rr = int(r)
        cc = int(c)
        if (c_split is not None and cc == c_split) or (r_split is not None and rr == r_split):
            out[rr, cc] = int(g[rr, cc])

    # Stage 2: move each connected sprite independently, preserving colors.
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, colors=colors_present(g, exclude={bg, ctrl}), return_format="dict")
    for comp in comps:
        regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for r, c in comp["coords"]:
            rr = int(r)
            cc = int(c)
            if (c_split is not None and cc == c_split) or (r_split is not None and rr == r_split):
                continue
            hs = "A"
            vs = "A"
            if r_split is not None:
                hs = "T" if rr < r_split else "B"
            if c_split is not None:
                vs = "L" if cc < c_split else "R"
            regions.setdefault((hs, vs), []).append((rr, cc))

        for (hs, vs), pts in regions.items():
            if not (_touches_edges(hs, vs) & edges):
                continue

            dh, dv = _pick_dest(hs, vs, edges, r_split, c_split)
            rs = [r for r, _ in pts]
            cs = [c for _, c in pts]
            rmin, rmax = min(rs), max(rs)
            cmin, cmax = min(cs), max(cs)
            dr = -rmin if dh == "T" else (h - 1 - rmax)
            dc = -cmin if dv == "L" else (w - 1 - cmax)

            for r, c in pts:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = int(g[r, c])

    return out


COMMON_TL_SLOTS = [
    {(0, 0), (0, 1), (1, 0)},
    {(0, 2), (1, 2), (1, 1), (2, 1), (2, 0)},
    {(0, 3), (0, 4), (1, 4), (2, 4), (3, 4), (3, 3), (3, 2), (3, 1), (3, 0)},
]

COMMON_TR_SLOTS = [
    {(0, 4), (0, 3), (1, 4)},
    {(0, 2), (1, 2), (1, 3), (2, 3), (2, 4)},
    {(0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)},
]


def _normalize_shape(cells: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0 = min(rs)
    c0 = min(cs)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _shape_hw(shape: frozenset[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _place_shape(grid: np.ndarray, shape: frozenset[tuple[int, int]], top: int, left: int, color: int) -> None:
    for r, c in shape:
        grid[top + r, left + c] = color


def _orientation_transform(grid: np.ndarray, mode: str) -> np.ndarray:
    if mode == "top":
        return np.array(grid, copy=True)
    if mode == "bottom":
        return np.flipud(grid)
    if mode == "left":
        return grid.T
    if mode == "right":
        return np.fliplr(grid.T)
    raise ValueError(f"unknown orientation {mode}")


def _build_canonical_input() -> np.ndarray:
    h = w = random.choice([19, 20])
    bg = 1
    ctrl = 2
    palette = [3, 4, 5, 6, 7, 8, 9]

    n_sprites = random.choice([2, 3])
    slot_ids = sorted(random.sample([0, 1, 2], n_sprites), key=lambda i: (_shape_hw(_normalize_shape(COMMON_TL_SLOTS[i]))[0], i))
    colors = random.sample(palette, n_sprites)

    left_shapes = [_normalize_shape(COMMON_TL_SLOTS[i]) for i in slot_ids]
    right_shapes = [_normalize_shape(COMMON_TR_SLOTS[i]) for i in slot_ids]
    heights = [max(_shape_hw(ls)[0], _shape_hw(rs)[0]) for ls, rs in zip(left_shapes, right_shapes)]

    total_span = sum(heights) + (n_sprites - 1)
    top0 = random.randint(4, h - 5 - total_span + 1)
    c_split = 9 if w == 19 else random.choice([9, 10])

    inp = np.full((h, w), bg, dtype=int)
    cur_top = top0
    for color, left_shape, right_shape, height in zip(colors, left_shapes, right_shapes, heights):
        lh, lw = _shape_hw(left_shape)
        rh, rw = _shape_hw(right_shape)
        left_col = c_split - lw
        right_col = c_split + 1

        _place_shape(inp, left_shape, cur_top, left_col, color)
        _place_shape(inp, right_shape, cur_top, right_col, color)
        inp[cur_top, c_split] = color

        cur_top += height + 1

    inp[0, c_split] = ctrl
    return inp


def _input_metrics(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(bg)
    ctrl = int(sorted(border_cols)[0])
    fg_cols = colors_present(g, exclude={bg, ctrl})
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, colors=fg_cols, return_format="dict")
    return len(comps), len(fg_cols), len({tuple(sorted(set(int(g[r, c]) for r, c in comp["coords"]))) for comp in comps})


def generate_input() -> np.ndarray:
    for _ in range(50):
        canonical = _build_canonical_input()
        inp = _orientation_transform(canonical, random.choice(["top", "bottom", "left", "right"]))
        n_comp, n_cols, n_color_signatures = _input_metrics(inp)
        if 2 <= n_comp <= 3 and n_cols == n_comp and n_color_signatures == n_comp:
            main(inp)
            return inp
    raise RuntimeError("Failed to generate valid combo input")
