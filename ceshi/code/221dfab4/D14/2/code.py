# Variant of 221dfab4
# Dimension: D14 — 边缘选择与旋转对称性 (meta)
# Change: 黄色线段可位于四条边缘中的任意一条，推进方向自动朝对侧
#   → 线段位于角落区域，向两个方向推进
#
# Design analysis for the D14 variant
# a) Concrete behavior and trace
#    Original task: one straight yellow border segment creates one family of stripes,
#    either horizontal or vertical. Example: a top-edge segment on columns 5..8 paints
#    rows 0,2,4,... with colors Y,Y,G while only the green rows recolor the full row.
#    Variant: the seed is an L-shaped yellow border component in one corner, e.g.
#    top-left = row 0 on cols 0..4 union col 0 on rows 0..3. The horizontal arm still
#    generates horizontal stripes moving inward, and the vertical arm simultaneously
#    generates vertical stripes moving inward. The result is an overlaid L-like stripe
#    system. The main flow change therefore happens exactly where the original solver
#    chose a single orientation and a single direction; that branch must be replaced by
#    extracting two arms and running both propagation plans together.
# b) Complexity calibration
#    If the variant only moved the seed "near a corner" but still kept one direction,
#    many samples would look like the original task. Using an actual corner L marker and
#    requiring both a horizontal and a vertical propagation family makes the change
#    visually obvious. Independent arm lengths prevent the corner seed from collapsing
#    into a symmetric square cue.
# c) Hidden assumptions audit
#    - `if r0 == r1` / `elif c0 == c1` in the original solver assumes the yellow marker is
#      a single straight segment. That fails here because the seed spans both one row and
#      one column.
#    - `dir1 = 1 if start_pos == 0 else -1` assumes only one inward direction exists.
#      The variant needs one row direction and one column direction.
#    - The original in-place painting order is harmless because there is only one stripe
#      family. With two families, naive sequential painting would let one family's gaps
#      erase the other family's stripes. The new code separates green floods, gap masks,
#      and stripe masks so crossings are deterministic.
# d) Placement legality
#    Generator guarantees exactly one yellow border component shaped like a corner L,
#    with both arms touching the same corner cell. Foreground blobs are placed away from
#    the border so they cannot merge with the seed. Samples are rejected unless the
#    output shows green flood effects caused by both the horizontal and vertical stripe
#    families, which keeps the puzzle faithful to the intended rule.
# e) Visual observability
#    A human should be able to infer "corner L seed => two inward stripe systems" from a
#    few examples because the output now contains both repeated rows and repeated columns.
#    The generator rejects cases where one direction becomes visually inert, so the
#    variant does not regress to the original one-axis behavior.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path(__file__).resolve().parents[5] / "arc2_solver_select"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite

Y = 4
G = 3
CYCLE = (Y, Y, G)


def _extract_corner_seed(grid: np.ndarray) -> dict[str, object] | None:
    """Return the corner, arm spans, and inward directions for the yellow L marker."""
    g = np.array(grid, dtype=int)
    H, W = g.shape

    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = (g == Y) & border
    if not bool(smask.any()):
        return None

    lab, n = label_mask(smask, connectivity=4)
    if int(n) <= 0:
        return None

    best_id = max(range(1, int(n) + 1), key=lambda cid: int((lab == int(cid)).sum()))
    comp = lab == int(best_id)

    corners = []
    if comp[0, 0]:
        corners.append("tl")
    if comp[0, W - 1]:
        corners.append("tr")
    if comp[H - 1, 0]:
        corners.append("bl")
    if comp[H - 1, W - 1]:
        corners.append("br")
    if len(corners) != 1:
        return None

    corner = corners[0]
    if corner == "tl":
        row_edge, col_edge, row_dir, col_dir = 0, 0, 1, 1
        h_vals = np.where(comp[0, :])[0]
        v_vals = np.where(comp[:, 0])[0]
        if len(h_vals) == 0 or len(v_vals) == 0 or int(h_vals[0]) != 0 or int(v_vals[0]) != 0:
            return None
    elif corner == "tr":
        row_edge, col_edge, row_dir, col_dir = 0, W - 1, 1, -1
        h_vals = np.where(comp[0, :])[0]
        v_vals = np.where(comp[:, W - 1])[0]
        if len(h_vals) == 0 or len(v_vals) == 0 or int(h_vals[-1]) != W - 1 or int(v_vals[0]) != 0:
            return None
    elif corner == "bl":
        row_edge, col_edge, row_dir, col_dir = H - 1, 0, -1, 1
        h_vals = np.where(comp[H - 1, :])[0]
        v_vals = np.where(comp[:, 0])[0]
        if len(h_vals) == 0 or len(v_vals) == 0 or int(h_vals[0]) != 0 or int(v_vals[-1]) != H - 1:
            return None
    else:
        row_edge, col_edge, row_dir, col_dir = H - 1, W - 1, -1, -1
        h_vals = np.where(comp[H - 1, :])[0]
        v_vals = np.where(comp[:, W - 1])[0]
        if len(h_vals) == 0 or len(v_vals) == 0 or int(h_vals[-1]) != W - 1 or int(v_vals[-1]) != H - 1:
            return None

    if np.any(np.diff(h_vals) != 1) or np.any(np.diff(v_vals) != 1):
        return None

    h_span = list(range(int(h_vals[0]), int(h_vals[-1]) + 1))
    v_span = list(range(int(v_vals[0]), int(v_vals[-1]) + 1))
    expected = len(h_span) + len(v_span) - 1
    if int(comp.sum()) != int(expected):
        return None

    return {
        "corner": corner,
        "row_edge": int(row_edge),
        "col_edge": int(col_edge),
        "row_dir": int(row_dir),
        "col_dir": int(col_dir),
        "h_span": h_span,
        "v_span": v_span,
    }


def _build_plan(start: int, direction: int, limit: int) -> tuple[dict[int, int], set[int]]:
    """Map stripe positions to colors and return the forced-gap positions."""
    stripes: dict[int, int] = {}
    gaps: set[int] = set()
    step = 0
    while True:
        pos = int(start) + int(direction) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break
        stripes[int(pos)] = int(CYCLE[int(step) % 3])
        gap = int(start) + int(direction) * (2 * int(step) + 1)
        if 0 <= gap < int(limit):
            gaps.add(int(gap))
        step += 1
    return stripes, gaps


def _variant_metrics(grid: np.ndarray, out: np.ndarray | None = None) -> dict[str, int] | None:
    info = _extract_corner_seed(grid)
    if info is None:
        return None

    g = np.array(grid, dtype=int)
    if out is None:
        out = main(g)
    out = np.array(out, dtype=int)
    bg = int(get_bg(g))

    row_stripes, _ = _build_plan(int(info["row_edge"]), int(info["row_dir"]), g.shape[0])
    col_stripes, _ = _build_plan(int(info["col_edge"]), int(info["col_dir"]), g.shape[1])
    green_rows = {r for r, color in row_stripes.items() if int(color) == G}
    green_cols = {c for c, color in col_stripes.items() if int(color) == G}
    h_span = set(int(c) for c in info["h_span"])
    v_span = set(int(r) for r in info["v_span"])

    row_flood_hits = 0
    for r in green_rows:
        for c in range(g.shape[1]):
            if c in h_span:
                continue
            if int(g[r, c]) not in (bg, Y) and int(out[r, c]) == G:
                row_flood_hits += 1

    col_flood_hits = 0
    for c in green_cols:
        for r in range(g.shape[0]):
            if r in v_span:
                continue
            if int(g[r, c]) not in (bg, Y) and int(out[r, c]) == G:
                col_flood_hits += 1

    return {
        "h_len": len(h_span),
        "v_len": len(v_span),
        "green_rows": len(green_rows),
        "green_cols": len(green_cols),
        "row_flood_hits": int(row_flood_hits),
        "col_flood_hits": int(col_flood_hits),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate two overlaid striped families from a yellow corner-L seed.

    - The top/bottom border arm creates horizontal stripes moving inward.
    - The left/right border arm creates vertical stripes moving inward.
    - Both families use the original Y,Y,G cycle with 1-cell gaps.
    - Green rows/columns recolor existing non-background pixels across the full
      row/column, while segment painting remains confined to the arm spans.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    info = _extract_corner_seed(g)
    if info is None:
        return out

    h_span = np.array(info["h_span"], dtype=int)
    v_span = np.array(info["v_span"], dtype=int)

    row_stripes, gap_rows = _build_plan(int(info["row_edge"]), int(info["row_dir"]), H)
    col_stripes, gap_cols = _build_plan(int(info["col_edge"]), int(info["col_dir"]), W)

    green_rows = [r for r, color in row_stripes.items() if int(color) == G]
    green_cols = [c for c, color in col_stripes.items() if int(color) == G]

    for r in green_rows:
        mask = out[int(r), :] != int(bg)
        out[int(r), mask] = int(G)
    for c in green_cols:
        mask = out[:, int(c)] != int(bg)
        out[mask, int(c)] = int(G)

    clear_mask = np.zeros((H, W), dtype=bool)
    for r in gap_rows:
        clear_mask[int(r), h_span] = True
    for c in gap_cols:
        clear_mask[v_span, int(c)] = True
    out[clear_mask] = int(bg)

    yellow_mask = np.zeros((H, W), dtype=bool)
    green_mask = np.zeros((H, W), dtype=bool)

    for r, color in row_stripes.items():
        target = green_mask if int(color) == G else yellow_mask
        target[int(r), h_span] = True
    for c, color in col_stripes.items():
        target = green_mask if int(color) == G else yellow_mask
        target[v_span, int(c)] = True

    out[yellow_mask] = int(Y)
    out[green_mask] = int(G)
    return out


def _make_l_seed(grid: np.ndarray, corner: str, h_len: int, v_len: int) -> None:
    H, W = grid.shape
    if corner == "tl":
        grid[0, :h_len] = Y
        grid[:v_len, 0] = Y
    elif corner == "tr":
        grid[0, W - h_len :] = Y
        grid[:v_len, W - 1] = Y
    elif corner == "bl":
        grid[H - 1, :h_len] = Y
        grid[H - v_len :, 0] = Y
    else:
        grid[H - 1, W - h_len :] = Y
        grid[H - v_len :, W - 1] = Y


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg = rng.sample(available, 2)

    grid = np.full((H, W), bg, dtype=int)
    corner = rng.choice(["tl", "tr", "bl", "br"])
    h_len = rng.randint(3, 8)
    v_len = rng.randint(3, 8)
    if h_len >= W or v_len >= H:
        return None
    if h_len == v_len:
        if h_len < 8:
            h_len += 1
        else:
            v_len -= 1

    _make_l_seed(grid, corner, h_len, v_len)

    canvas = grid.copy()
    n_blobs = rng.randint(6, 10)

    for _ in range(n_blobs):
        bh = rng.randint(2, 10)
        bw = rng.randint(2, 10)
        density = rng.uniform(0.4, 0.85)

        try:
            sprite = random_sprite(
                bh,
                bw,
                density=density,
                symmetry="not_symmetric",
                color_palette=[fg],
                connectivity=4,
                background=bg,
            )
        except (AssertionError, RecursionError):
            continue

        mask = sprite != bg
        rows_any = np.any(mask, axis=1)
        cols_any = np.any(mask, axis=0)
        if not rows_any.any():
            continue
        r0, r1 = np.where(rows_any)[0][[0, -1]]
        c0, c1 = np.where(cols_any)[0][[0, -1]]
        sprite = sprite[r0 : r1 + 1, c0 : c1 + 1]

        sh, sw = sprite.shape
        pix = int(np.sum(sprite != bg))
        if pix < 5 or pix > 67:
            continue

        try:
            pr, pc = random_free_location(canvas, sprite, transparent=bg, border=1, padding=1)
        except ValueError:
            continue

        for dr in range(sh):
            for dc in range(sw):
                if sprite[dr, dc] != bg:
                    canvas[pr + dr, pc + dc] = sprite[dr, dc]

    fg_mask = (canvas != bg) & (canvas != Y)
    _, n_placed = label_mask(fg_mask, connectivity=4)
    if n_placed < 5:
        return None

    inp = canvas.copy()
    out = main(inp)

    if int(get_bg(inp)) != int(bg):
        return None
    if not np.array_equal(main(inp), out):
        return None

    metrics = _variant_metrics(inp, out)
    if metrics is None:
        return None
    if metrics["h_len"] < 3 or metrics["v_len"] < 3:
        return None
    if metrics["row_flood_hits"] <= 0 or metrics["col_flood_hits"] <= 0:
        return None

    return inp


def generate_input():
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
