from __future__ import annotations

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste
import random


def _edge_mask(shape: tuple[int, int], side: str) -> np.ndarray:
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    if side == "top":
        mask[0, :] = True
    elif side == "bottom":
        mask[H - 1, :] = True
    elif side == "left":
        mask[:, 0] = True
    elif side == "right":
        mask[:, W - 1] = True
    else:
        raise ValueError(f"unknown side: {side}")
    return mask


def _find_wall(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], np.ndarray]:
    specs = [
        ("top", (-1, 0), g[0, :]),
        ("bottom", (1, 0), g[-1, :]),
        ("left", (0, -1), g[:, 0]),
        ("right", (0, 1), g[:, -1]),
    ]
    candidates = []
    for side, direction, line in specs:
        color = int(line[0])
        if color == bg or not np.all(line == color):
            continue
        mask = _edge_mask(g.shape, side)
        if np.array_equal(g == color, mask):
            candidates.append((color, direction, mask))

    assert len(candidates) == 1, "expected exactly one complete colored edge wall"
    return candidates[0]


def _settle_key(comp: dict, dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    if dr == -1:
        return (r0, c0, c1, r1)
    if dr == 1:
        return (-r1, c0, c1, r0)
    if dc == -1:
        return (c0, r0, r1, c1)
    assert dc == 1
    return (-c1, r0, r1, c0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide every colored domino straight toward the full red edge wall until it is blocked.

    Input structure
    - The background is empty space.
    - One non-background color forms a complete one-cell-thick edge wall and appears nowhere else.
    - Every other foreground object is a monochromatic 4-connected domino, either horizontal or vertical.

    Rule
    - Keep the edge wall fixed and clear the other foreground objects from their original positions.
    - The wall side gives the gravity direction: objects move up, down, left, or right toward that wall.
    - Settle dominoes nearest the wall first; each domino slides rigidly along the gravity direction until the next step would hit the wall, another settled domino, or the grid boundary.

    Key insight
    - The full edge wall is the only anchor; the output is the gravity-packed state of all remaining dominoes against that wall.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    wall_color, (dr, dc), wall_mask = _find_wall(g, bg)
    moving_colors = colors_present(g, exclude={bg, wall_color})
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=moving_colors)

    assert comps, "expected at least one moving domino"
    for comp in comps:
        assert comp["area"] == 2, "expected every moving object to be a domino"

    out = np.full_like(g, bg)
    out[wall_mask] = wall_color
    occupied = wall_mask.copy()
    H, W = g.shape
    step = np.array([dr, dc], dtype=int)

    for comp in sorted(comps, key=lambda item: _settle_key(item, dr, dc)):
        coords = comp["coords"].astype(int).copy()
        color = int(comp["color"])

        while True:
            nxt = coords + step
            in_bounds = (
                np.all(0 <= nxt[:, 0])
                and np.all(nxt[:, 0] < H)
                and np.all(0 <= nxt[:, 1])
                and np.all(nxt[:, 1] < W)
            )
            if not in_bounds or np.any(occupied[nxt[:, 0], nxt[:, 1]]):
                break
            coords = nxt

        assert not np.any(occupied[coords[:, 0], coords[:, 1]]), "settled domino collision"
        out[coords[:, 0], coords[:, 1]] = color
        occupied[coords[:, 0], coords[:, 1]] = True

    return out

# --------------- generator (o5x) ---------------

# Canvas and wall bounds.
_GEN_BG = 0
_GEN_WALL_COLOR = 2
_GEN_ROW_BOUNDS = (5, 10)
_GEN_COL_BOUNDS = (5, 11)
_GEN_WALL_THICKNESS = 1
_GEN_SIDES = ("top", "bottom", "left", "right")

# Domino count, colors, and orientation requirements.
_GEN_DOMINO_AREA = 2
_GEN_DOMINO_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
_GEN_COUNTS = (5, 6, 7, 8, 9, 10, 11)
_GEN_COUNT_WEIGHTS = (1, 2, 3, 5, 10, 18, 28)
_GEN_ORIENTATIONS = ("h", "v")
_GEN_ORIENTATION_STEPS = {"h": (0, 1), "v": (1, 0)}

# Layout feasibility and retry limits.
_GEN_MIN_SLIDE_BACK = 1
_GEN_ROOMY_MIN_H_PLUS_W = {9: 17, 10: 18, 11: 18}
_GEN_RETRY_LIMIT = 300
_GEN_BLOCK_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def _gen_transform_grid(g: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return g.copy()
    if side == "bottom":
        return np.flipud(g).copy()
    if side == "left":
        return np.rot90(g, 1).copy()
    if side == "right":
        return np.rot90(g, -1).copy()
    raise ValueError(side)


def _gen_wall_grid_top(H: int, W: int) -> np.ndarray:
    grid = np.full((H, W), _GEN_BG, dtype=int)
    paste(grid, np.full((_GEN_WALL_THICKNESS, W), _GEN_WALL_COLOR, dtype=int), 0, 0)
    return grid


def _gen_cells_key_top(cells: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return (min(rows), min(cols), max(cols), max(rows))


def _gen_candidate_ranges_top(H: int, W: int, orient: str) -> tuple[range, range]:
    dr, dc = _GEN_ORIENTATION_STEPS[orient]
    rows = range(_GEN_WALL_THICKNESS, H - _GEN_WALL_THICKNESS - dr)
    cols = range(0, W - dc)
    return rows, cols


def _gen_candidate_is_supported_top(
    cells: tuple[tuple[int, int], ...], occupied: np.ndarray
) -> bool:
    own = set(cells)
    for r, c in cells:
        nr = r - 1
        if (nr, c) not in own and occupied[nr, c]:
            return True
    return False


def _gen_stable_candidates_top(
    H: int, W: int, occupied: np.ndarray, allowed: tuple[str, ...]
) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
    candidates = []
    for orient in _GEN_ORIENTATIONS:
        if orient not in allowed:
            continue
        dr, dc = _GEN_ORIENTATION_STEPS[orient]
        row_range, col_range = _gen_candidate_ranges_top(H, W, orient)
        for r in row_range:
            for c in col_range:
                cells = ((r, c), (r + dr, c + dc))
                if any(occupied[rr, cc] for rr, cc in cells):
                    continue
                if _gen_candidate_is_supported_top(cells, occupied):
                    candidates.append((orient, cells))
    random.shuffle(candidates)
    return candidates


def _gen_paste_domino(
    grid: np.ndarray, cells: tuple[tuple[int, int], ...], color: int
) -> None:
    r0, c0, _, _ = _gen_cells_key_top(cells)
    if cells[0][0] == cells[1][0]:
        patch = np.full((1, 2), color, dtype=int)
    else:
        patch = np.full((2, 1), color, dtype=int)
    paste(grid, patch, r0, c0)


def _gen_domino_colors(count: int) -> list[int]:
    palette = list(_GEN_DOMINO_COLORS)
    random.shuffle(palette)
    colors = [palette[i % len(palette)] for i in range(count)]
    random.shuffle(colors)
    return colors


def _gen_make_output_top(H: int, W: int, count: int) -> tuple[np.ndarray, list[dict]] | None:
    out = _gen_wall_grid_top(H, W)
    occupied = out != _GEN_BG
    colors = _gen_domino_colors(count)

    dominoes = []
    missing = set(_GEN_ORIENTATIONS)
    for i in range(count):
        remaining = count - i
        allowed = tuple(missing) if remaining <= len(missing) else _GEN_ORIENTATIONS
        candidates = _gen_stable_candidates_top(H, W, occupied, allowed)
        if not candidates:
            return None

        orient, cells = candidates[0]
        color = int(colors[i])
        _gen_paste_domino(out, cells, color)
        for r, c in cells:
            occupied[r, c] = True
        dominoes.append({"cells": cells, "color": color, "orient": orient})
        missing.discard(orient)

    if missing:
        return None
    return out, dominoes


def _gen_mark_blocked_top(
    blocked: np.ndarray, cells: tuple[tuple[int, int], ...]
) -> None:
    H, W = blocked.shape
    for r, c in cells:
        for dr, dc in _GEN_BLOCK_OFFSETS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                blocked[rr, cc] = True


def _gen_input_cells_ok_top(
    cells: tuple[tuple[int, int], ...], blocked: np.ndarray
) -> bool:
    return all(not blocked[r, c] for r, c in cells)


def _gen_slide_choices_top(
    H: int,
    out_cells: tuple[tuple[int, int], ...],
    blocked: np.ndarray,
    prev_key: tuple[int, int, int, int] | None,
) -> list[tuple[int, tuple[tuple[int, int], ...], tuple[int, int, int, int]]]:
    max_dist = H - _GEN_WALL_THICKNESS - max(r for r, _ in out_cells)
    choices = []
    for dist in range(_GEN_MIN_SLIDE_BACK, max_dist + 1):
        cells = tuple((r + dist, c) for r, c in out_cells)
        key = _gen_cells_key_top(cells)
        if prev_key is not None and key <= prev_key:
            continue
        if _gen_input_cells_ok_top(cells, blocked):
            choices.append((dist, cells, key))
    random.shuffle(choices)
    return choices


def _gen_slide_back_top(H: int, W: int, dominoes: list[dict]) -> list[dict] | None:
    order = sorted(range(len(dominoes)), key=lambda i: _gen_cells_key_top(dominoes[i]["cells"]))
    assigned = [None] * len(dominoes)

    def rec(pos: int, blocked: np.ndarray, prev_key: tuple[int, int, int, int] | None) -> bool:
        if pos == len(order):
            return True

        idx = order[pos]
        out_cells = dominoes[idx]["cells"]
        choices = _gen_slide_choices_top(H, out_cells, blocked, prev_key)

        for _, cells, key in choices:
            next_blocked = blocked.copy()
            _gen_mark_blocked_top(next_blocked, cells)
            assigned[idx] = {**dominoes[idx], "cells": cells}
            if rec(pos + 1, next_blocked, key):
                return True
            assigned[idx] = None
        return False

    if not rec(0, np.zeros((H, W), dtype=bool), None):
        return None
    return assigned


def _gen_input_capacity_top(H: int, W: int) -> int:
    usable_h = H - _GEN_DOMINO_AREA
    horizontal = ((usable_h + 1) // 2) * ((W + 1) // 3)
    vertical = ((usable_h + 1) // 3) * ((W + 1) // 2)
    return max(horizontal, vertical)


def _gen_canonical_shape(final_H: int, final_W: int, side: str) -> tuple[int, int]:
    if side in ("top", "bottom"):
        return final_H, final_W
    return final_W, final_H


def _gen_feasible_sizes(count: int, side: str) -> list[tuple[int, int, int, int]]:
    sizes = []
    min_rows, max_rows = _GEN_ROW_BOUNDS
    min_cols, max_cols = _GEN_COL_BOUNDS
    for final_H in range(min_rows, max_rows + 1):
        for final_W in range(min_cols, max_cols + 1):
            H, W = _gen_canonical_shape(final_H, final_W, side)
            wall_len = W
            bg_count = final_H * final_W - wall_len - _GEN_DOMINO_AREA * count
            if bg_count <= wall_len:
                continue
            if _gen_input_capacity_top(H, W) < count:
                continue
            sizes.append((final_H, final_W, H, W))
    return sizes


def _gen_choose_size(count: int, side: str) -> tuple[int, int, int, int] | None:
    sizes = _gen_feasible_sizes(count, side)
    if not sizes:
        return None

    min_sum = _GEN_ROOMY_MIN_H_PLUS_W.get(count)
    if min_sum is not None:
        roomy = [item for item in sizes if item[2] + item[3] >= min_sum]
        if roomy:
            sizes = roomy
    return random.choice(sizes)


def _gen_render_top(H: int, W: int, dominoes: list[dict]) -> np.ndarray:
    grid = _gen_wall_grid_top(H, W)
    for dom in dominoes:
        _gen_paste_domino(grid, dom["cells"], int(dom["color"]))
    return grid


def _gen_attempt() -> np.ndarray | None:
    count = random.choices(_GEN_COUNTS, weights=_GEN_COUNT_WEIGHTS, k=1)[0]
    side = random.choice(_GEN_SIDES)
    size = _gen_choose_size(count, side)
    if size is None:
        return None
    final_H, final_W, H, W = size

    built = _gen_make_output_top(H, W, count)
    if built is None:
        return None
    out_top, out_dominoes = built

    in_dominoes = _gen_slide_back_top(H, W, out_dominoes)
    if in_dominoes is None:
        return None

    inp_top = _gen_render_top(H, W, in_dominoes)
    inp = _gen_transform_grid(inp_top, side)
    out = _gen_transform_grid(out_top, side)
    if inp.shape != (final_H, final_W):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 6ad5bdfd input")
