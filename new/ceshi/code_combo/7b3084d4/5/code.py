# Combo variant of 7b3084d4 (index 5)
# combo_spec: D3/0+D7/2
# Name: 标记编码非正方形输出尺寸

# Design analysis for the fused variant:
# - Concrete behavior shift:
#   Original: a single grey pixel marks the anchor, and the solver infers the final
#   rectangle only from total area, preferring near-squares.
#   Fused: the anchor carries a 2- or 3-pixel grey line at its corner. After rotating
#   the anchor into its canonical top-left pose, a short horizontal line means a mildly
#   wide rectangle, a long horizontal line means a more stretched wide rectangle, and
#   the vertical counterparts encode the tall versions. Example: if the marker becomes
#   {(0,0),(0,1),(0,2)}, the target ratio is 2:3, so area 48 implies scale 4 and the
#   solver must build an 8x12 rectangle instead of guessing among all factor pairs.
# - Complexity calibration:
#   The ratio code uses only four visible states:
#   2-wide, 3-wide, 2-tall, 3-tall. This is enough to make the non-square change obvious
#   without introducing too many latent classes for humans to infer from 2-4 examples.
# - Hidden assumptions broken and replaced:
#   The old solver assumed a singleton marker and square-first dimension search. Those
#   assumptions are false here, so marker extraction now accepts a 2/3-cell component,
#   and dimension inference is driven by the decoded marker footprint instead of `_dims`.
#   The anchor search still tests rotations, but only the canonical one keeps the marker
#   line flush with the output's top-left corner without sending other anchor cells
#   negative, so the generator guarantees the new assumption is satisfiable.
# - Placement legality:
#   The generator still works output-first. It builds a valid 2x2 split of a non-square
#   rectangle with mirrored tab intrusions, verifies each colored region is 8-connected,
#   recolors the anchor corner cells into the marker line, rotates pieces independently,
#   and scatters them on a 20x20 canvas with 8-neighborhood padding.
# - Visual observability and cleanliness:
#   The only new visual signal is a tiny grey line at the anchor corner. Its orientation
#   and length are easy to see, outputs are always visibly rectangular, and no extra
#   overlays are added elsewhere on the board, so the fused rule remains readable.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg as _bg_color, is_contiguous, neighbors8, random_free_location


MARKER_COLOR = 5
ANCHOR_COLOR = 7
COLOR_POOL = [1, 2, 3, 4, 6, 8, 9]

CANVAS_H, CANVAS_W = 20, 20
PLACEMENT_PADDING = 1
MAX_ATTEMPTS = 400
TAB_NAMES = ["H_left", "H_right", "V_top", "V_bot"]

TL, TR, BL, BR = 0, 1, 2, 3

# Marker footprint at the output's top-left corner -> (height_ratio, width_ratio).
MARKER_TO_RATIO: dict[frozenset[tuple[int, int]], tuple[int, int]] = {
    frozenset({(0, 0), (0, 1)}): (3, 4),
    frozenset({(0, 0), (0, 1), (0, 2)}): (2, 3),
    frozenset({(0, 0), (1, 0)}): (4, 3),
    frozenset({(0, 0), (1, 0), (2, 0)}): (3, 2),
}

RATIO_OPTIONS = [
    {"marker": [(0, 0), (0, 1)], "ratio": (3, 4)},
    {"marker": [(0, 0), (0, 1), (0, 2)], "ratio": (2, 3)},
    {"marker": [(0, 0), (1, 0)], "ratio": (4, 3)},
    {"marker": [(0, 0), (1, 0), (2, 0)], "ratio": (3, 2)},
]
_ratio_option_order: list[int] = []


def _next_ratio_option() -> dict[str, object]:
    """
    Cycle through the four visible marker classes in random order.

    This keeps consecutive generated examples diverse enough for humans to infer
    that line orientation controls wide-vs-tall and line length controls how
    stretched the rectangle is, instead of relying on lucky sampling.
    """
    global _ratio_option_order
    if not _ratio_option_order:
        _ratio_option_order = list(range(len(RATIO_OPTIONS)))
        random.shuffle(_ratio_option_order)
    return RATIO_OPTIONS[_ratio_option_order.pop()]


def _rotate_cells(cells: list[tuple[int, int, int]], k: int) -> list[tuple[int, int, int]]:
    pts = [(r, c, col) for r, c, col in cells]
    for _ in range(k % 4):
        pts = [(c, -r, col) for r, c, col in pts]
    min_r = min(r for r, _, _ in pts)
    min_c = min(c for _, c, _ in pts)
    pts = [(r - min_r, c - min_c, col) for r, c, col in pts]
    pts.sort()
    return pts


def _variants(cells: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    outs: list[list[tuple[int, int, int]]] = []
    for k in range(4):
        rot = _rotate_cells(cells, k)
        key = tuple(rot)
        if key not in seen:
            seen.add(key)
            outs.append(rot)
    return outs


def _marker_component_is_valid(marker_cells: list[tuple[int, int]]) -> bool:
    if len(marker_cells) not in (2, 3):
        return False
    seen = {marker_cells[0]}
    stack = [marker_cells[0]]
    all_cells = set(marker_cells)
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors8(r, c):
            if (nr, nc) in all_cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(all_cells)


def _extract_pieces(g: np.ndarray) -> tuple[list[dict], int]:
    bg = _bg_color(g)
    colors = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    if not colors:
        return [], bg

    marker_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER_COLOR)]
    assert marker_cells, "marker color 5 must appear"
    assert _marker_component_is_valid(marker_cells), "marker must be one 2- or 3-cell connected component"

    neigh = set()
    for mr, mc in marker_cells:
        for nr, nc in neighbors8(mr, mc):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                val = int(g[nr, nc])
                if val != bg and val != MARKER_COLOR:
                    neigh.add(val)
    assert len(neigh) == 1, "marker must touch exactly one anchor color"
    anchor_color = next(iter(neigh))

    pieces: list[dict] = []
    used = {MARKER_COLOR, anchor_color}

    coords = [(int(r), int(c), int(anchor_color)) for r, c in np.argwhere(g == anchor_color)]
    coords.extend((mr, mc, MARKER_COLOR) for mr, mc in marker_cells)
    min_r = min(r for r, _, _ in coords)
    min_c = min(c for _, c, _ in coords)
    coords = [(r - min_r, c - min_c, col) for r, c, col in coords]
    pieces.append({"name": "anchor", "cells": coords, "marker_color": MARKER_COLOR})

    for col in colors:
        if col in used:
            continue
        pts = [(int(r), int(c), int(col)) for r, c in np.argwhere(g == col)]
        min_r = min(r for r, _, _ in pts)
        min_c = min(c for _, c, _ in pts)
        pts = [(r - min_r, c - min_c, color) for r, c, color in pts]
        pieces.append({"name": f"c{col}", "cells": pts})

    return pieces, bg


def _decode_ratio(marker_points: list[tuple[int, int]]) -> tuple[int, int] | None:
    min_r = min(r for r, _ in marker_points)
    min_c = min(c for _, c in marker_points)
    footprint = frozenset((r - min_r, c - min_c) for r, c in marker_points)
    return MARKER_TO_RATIO.get(footprint)


def _dims_from_ratio(area: int, ratio_hw: tuple[int, int]) -> tuple[int, int] | None:
    rh, rw = ratio_hw
    unit = rh * rw
    if area % unit != 0:
        return None
    scale_sq = area // unit
    scale = int(scale_sq**0.5)
    if scale * scale != scale_sq:
        return None
    return rh * scale, rw * scale


def _placeable(board: np.ndarray, placement: list[tuple[int, int, int]]) -> bool:
    return all(board[r, c] == -1 for r, c, _ in placement)


def _apply(board: np.ndarray, placement: list[tuple[int, int, int]], val: int | None = None) -> None:
    if val is None:
        for r, c, col in placement:
            board[r, c] = col
    else:
        for r, c, _ in placement:
            board[r, c] = val


def _all_placements_for_variant(
    variant: list[tuple[int, int, int]], H: int, W: int
) -> list[list[tuple[int, int, int]]]:
    max_r = max(r for r, _, _ in variant)
    max_c = max(c for _, c, _ in variant)
    ph, pw = max_r + 1, max_c + 1
    outs: list[list[tuple[int, int, int]]] = []
    for tr in range(H - ph + 1):
        for tc in range(W - pw + 1):
            outs.append([(r + tr, c + tc, col) for r, c, col in variant])
    return outs


def _solve_packing(pieces: list[dict]) -> np.ndarray | None:
    area = sum(len(p["cells"]) for p in pieces)
    piece_vars = [_variants(p["cells"]) for p in pieces]

    anchor_idx = next((i for i, p in enumerate(pieces) if p["name"] == "anchor"), None)
    assert anchor_idx is not None, "an anchor piece is required"
    marker_col = pieces[anchor_idx]["marker_color"]
    remaining = [i for i in range(len(pieces)) if i != anchor_idx]

    for var in piece_vars[anchor_idx]:
        marker_pts = [(r, c) for r, c, col in var if col == marker_col]
        ratio_hw = _decode_ratio(marker_pts)
        if ratio_hw is None:
            continue
        dims = _dims_from_ratio(area, ratio_hw)
        if dims is None:
            continue
        H, W = dims

        min_mr = min(r for r, _ in marker_pts)
        min_mc = min(c for _, c in marker_pts)
        placed_anchor = [(r - min_mr, c - min_mc, col) for r, c, col in var]
        if any(r < 0 or c < 0 for r, c, _ in placed_anchor):
            continue
        if any(r >= H or c >= W for r, c, _ in placed_anchor):
            continue

        board = np.full((H, W), -1, dtype=int)
        if not _placeable(board, placed_anchor):
            continue
        _apply(board, placed_anchor)

        def backtrack(rem: list[int]) -> bool:
            if not rem:
                return True

            best_pi = None
            best_options = None
            best_count = None
            for pi in rem:
                opts = []
                for piece_var in piece_vars[pi]:
                    for placement in _all_placements_for_variant(piece_var, H, W):
                        if _placeable(board, placement):
                            opts.append(placement)
                if not opts:
                    return False
                if best_count is None or len(opts) < best_count:
                    best_count = len(opts)
                    best_pi = pi
                    best_options = opts
                    if best_count == 1:
                        break

            assert best_pi is not None and best_options is not None
            next_rem = [x for x in rem if x != best_pi]
            for placement in best_options:
                _apply(board, placement)
                if backtrack(next_rem):
                    return True
                _apply(board, placement, -1)
            return False

        if backtrack(remaining):
            return board

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble four scattered colored puzzle pieces into one anchored rectangle whose aspect ratio is encoded by the anchor mark.

    Input structure:
    - Background is the most common color.
    - There are four separated piece colors, and the anchor piece also contains a grey 2- or 3-cell line at one corner.
    - The grey line is the only multi-cell marker component and touches only the anchor color.

    Rule:
    - Find the anchor by locating the grey marker component and merge that marker with its adjacent colored piece.
    - Try anchor rotations, but only keep orientations where the grey line can sit flush at the output's top-left corner.
    - Decode the target rectangle ratio from the top-left marker footprint:
      short horizontal -> 3:4, long horizontal -> 2:3, short vertical -> 4:3, long vertical -> 3:2.
    - Use total area plus that ratio to determine the exact output size, then pack the other three rotatable pieces to tile the rectangle with no overlaps or gaps.

    Key insight:
    - The marker no longer just says "this is the anchor"; its visible line length and direction tell you whether the final board must be wide or tall before you do the packing.
    """
    g = np.asarray(grid, dtype=int)
    pieces, _bg = _extract_pieces(g)
    assert pieces, "grid has no non-background pieces"
    solved = _solve_packing(pieces)
    assert solved is not None, "decoded ratio and pieces must tile one valid rectangle"
    return solved.astype(int)


def _build_ownership(A: int, B: int, p_row: int, p_col: int, tab_specs: dict[str, tuple[int, str] | None]) -> np.ndarray | None:
    H, W = 2 * A, 2 * B
    own = np.zeros((H, W), dtype=int)
    own[:A, :B] = TL
    own[:A, B:] = TR
    own[A:, :B] = BL
    own[A:, B:] = BR

    for kind, col, up_owner, down_owner in [
        ("H_left", p_col, BL, TL),
        ("H_right", W - 1 - p_col, BR, TR),
    ]:
        spec = tab_specs.get(kind)
        if spec is None:
            continue
        depth, direction = spec
        if direction == "up":
            own[A - depth : A, col] = up_owner
        else:
            own[A : A + depth, col] = down_owner

    for kind, row, left_owner, right_owner in [
        ("V_top", p_row, TR, TL),
        ("V_bot", H - 1 - p_row, BR, BL),
    ]:
        spec = tab_specs.get(kind)
        if spec is None:
            continue
        depth, direction = spec
        if direction == "left":
            own[row, B - depth : B] = left_owner
        else:
            own[row, B : B + depth] = right_owner

    for q in (TL, TR, BL, BR):
        mask = (own == q).astype(int)
        if mask.sum() == 0 or not is_contiguous(mask, background=0, connectivity=8):
            return None
    return own


def _extract_piece(
    own: np.ndarray,
    q: int,
    color: int,
    marker_cells: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int, int]]:
    rows, cols = np.where(own == q)
    min_r, min_c = int(rows.min()), int(cols.min())
    cells: list[tuple[int, int, int]] = []
    for r, c in zip(rows, cols):
        rr, cc = int(r), int(c)
        col = MARKER_COLOR if marker_cells is not None and (rr, cc) in marker_cells else color
        cells.append((rr - min_r, cc - min_c, col))
    return cells


def _cells_to_patch(cells: list[tuple[int, int, int]]) -> np.ndarray:
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c, col in cells:
        patch[r, c] = col
    return patch


def _random_tab_specs(A: int, B: int) -> dict[str, tuple[int, str] | None]:
    specs: dict[str, tuple[int, str] | None] = {}
    for name in TAB_NAMES:
        if name.startswith("H"):
            depth = random.randint(1, A - 1)
            direction = random.choice(["up", "down"])
        else:
            depth = random.randint(1, B - 1)
            direction = random.choice(["left", "right"])
        specs[name] = (depth, direction)
    return specs


def _scatter_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=int)
    order = list(range(len(patches)))
    random.shuffle(order)
    for i in order:
        patch = patches[i]
        try:
            r0, c0 = random_free_location(
                canvas,
                patch,
                transparent=0,
                padding=PLACEMENT_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        region = canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        mask = patch != 0
        region[mask] = patch[mask]
    return canvas


def _gen_attempt(option: dict[str, object]) -> tuple[np.ndarray, np.ndarray, dict[str, object]] | None:
    ratio_h, ratio_w = option["ratio"]
    scale = random.choice([2, 4])
    H, W = ratio_h * scale, ratio_w * scale
    A, B = H // 2, W // 2
    if min(A, B) < 2:
        return None

    p_row = random.randint(1, A - 1)
    p_col = random.randint(1, B - 1)
    own = _build_ownership(A, B, p_row, p_col, _random_tab_specs(A, B))
    if own is None:
        return None

    pool = COLOR_POOL.copy()
    random.shuffle(pool)
    quad_colors = {TL: ANCHOR_COLOR, TR: pool[0], BL: pool[1], BR: pool[2]}

    output = np.zeros((H, W), dtype=int)
    for q, color in quad_colors.items():
        output[own == q] = color

    marker_cells = set(option["marker"])
    for r, c in marker_cells:
        output[r, c] = MARKER_COLOR

    anchor_cells = _extract_piece(own, TL, ANCHOR_COLOR, marker_cells=marker_cells)
    other_cells = [_extract_piece(own, q, quad_colors[q]) for q in (TR, BL, BR)]
    rotated_cells = [_rotate_cells(anchor_cells, random.randint(0, 3))]
    rotated_cells.extend(_rotate_cells(cells, random.randint(0, 3)) for cells in other_cells)

    canvas = _scatter_patches([_cells_to_patch(cells) for cells in rotated_cells])
    if canvas is None:
        return None

    try:
        out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(out, output):
        return None

    meta = {
        "ratio": option["ratio"],
        "marker_size": len(option["marker"]),
        "output_shape": (H, W),
        "area": H * W,
    }
    return canvas, output, meta


def generate_input() -> np.ndarray:
    option = _next_ratio_option()
    for _ in range(MAX_ATTEMPTS):
        sample = _gen_attempt(option)
        if sample is not None:
            return sample[0]
    raise RuntimeError("generate_input: exhausted attempts")
