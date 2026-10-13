from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)

# Solver and generator geometry.
SEG_LEN_MIN, SEG_LEN_MAX = 2, 4
ARM_GAP_MIN, ARM_GAP_MAX = 1, 6
MIN_GRID_SIDE, MAX_GRID_SIDE = 10, 30
PAIR_COUNT_MAX = 4
PLACEMENT_PADDING = 1
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400

# Branches: where the vertical segment sits relative to the horizontal one,
# and where the horizontal segment sits relative to the vertical one.
BRANCHES = (
    ("left", "up"),
    ("left", "down"),
    ("right", "up"),
    ("right", "down"),
)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        pts: list[tuple[int, int]] = []

        while stack:
            cr, cc = stack.pop()
            pts.append((int(cr), int(cc)))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = int(cr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        comps.append(np.array(pts, dtype=int))

    return comps


def _segment_from_coords(coords: np.ndarray, color: int) -> dict[str, int | str]:
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])

    if rows.size == 1:
        row = int(rows[0])
        c0 = int(cols.min())
        c1 = int(cols.max())
        assert coords.shape[0] == c1 - c0 + 1, "horizontal segment must be contiguous"
        return {
            "orientation": "horizontal",
            "color": int(color),
            "row": row,
            "c0": c0,
            "c1": c1,
        }

    if cols.size == 1:
        col = int(cols[0])
        r0 = int(rows.min())
        r1 = int(rows.max())
        assert coords.shape[0] == r1 - r0 + 1, "vertical segment must be contiguous"
        return {
            "orientation": "vertical",
            "color": int(color),
            "col": col,
            "r0": r0,
            "r1": r1,
        }

    raise AssertionError("each component must be a straight line segment")


def _draw_connection(out: np.ndarray, horizontal: dict[str, int | str], vertical: dict[str, int | str]) -> None:
    row = int(horizontal["row"])
    c0 = int(horizontal["c0"])
    c1 = int(horizontal["c1"])
    col = int(vertical["col"])
    r0 = int(vertical["r0"])
    r1 = int(vertical["r1"])
    color = int(horizontal["color"])

    left_dist = abs(col - c0)
    right_dist = abs(col - c1)
    assert left_dist != right_dist, "nearest horizontal endpoint must be unique"
    anchor_c = c0 if left_dist < right_dist else c1

    top_dist = abs(row - r0)
    bottom_dist = abs(row - r1)
    assert top_dist != bottom_dist, "nearest vertical endpoint must be unique"
    anchor_r = r0 if top_dist < bottom_dist else r1

    out[min(row, anchor_r) : max(row, anchor_r) + 1, anchor_c] = color
    out[anchor_r, min(col, anchor_c) : max(col, anchor_c) + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color horizontal/vertical segment pair with an L-shaped path of that color.

    Input structure:
    - The background is black.
    - The foreground consists of short one-pixel-thick monochrome line segments.
    - Each foreground color appears in exactly two separate segments: one horizontal and one vertical.

    Rule:
    - For each color, identify its horizontal segment and vertical segment.
    - Choose the endpoint of the horizontal segment that is closer to the vertical segment,
      and choose the endpoint of the vertical segment that is closer to the horizontal segment.
    - Extend those two endpoints toward each other along the missing orthogonal directions so
      they meet at a single corner, then fill the whole L-shaped path in that color.

    Key insight:
    - The connector is not an arbitrary shortest path: each segment contributes the arm that is
      orthogonal to its own orientation, so the chosen endpoints uniquely determine the bend.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    out = g.copy()

    colors = [int(color) for color in np.unique(g) if int(color) != int(bg)]
    for color in colors:
        comps = _components(g == int(color))
        assert len(comps) == 2, "each color must form exactly two components"
        segs = [_segment_from_coords(comp, color) for comp in comps]

        horizontals = [seg for seg in segs if seg["orientation"] == "horizontal"]
        verticals = [seg for seg in segs if seg["orientation"] == "vertical"]
        assert len(horizontals) == 1 and len(verticals) == 1, "expected one horizontal and one vertical segment"

        _draw_connection(out, horizontals[0], verticals[0])

    return out


def _build_pair(branch: tuple[str, str], color: int) -> dict[str, object]:
    x_rel, y_rel = branch
    h_len = random.randint(SEG_LEN_MIN, SEG_LEN_MAX)
    v_len = random.randint(SEG_LEN_MIN, SEG_LEN_MAX)
    dx = random.randint(ARM_GAP_MIN, ARM_GAP_MAX)
    dy = random.randint(ARM_GAP_MIN, ARM_GAP_MAX)

    height = v_len + dy
    width = h_len + dx

    if y_rel == "up":
        anchor_r = dy
        h_row = 0
        v_rows = range(anchor_r, anchor_r + v_len)
    else:
        anchor_r = v_len - 1
        h_row = anchor_r + dy
        v_rows = range(0, v_len)

    if x_rel == "left":
        anchor_c = dx
        v_col = 0
        h_cols = range(anchor_c, anchor_c + h_len)
    else:
        anchor_c = h_len - 1
        v_col = anchor_c + dx
        h_cols = range(0, h_len)

    input_cells = {(h_row, c) for c in h_cols}
    input_cells.update((r, v_col) for r in v_rows)

    output_cells = set(input_cells)
    output_cells.update((r, anchor_c) for r in range(min(h_row, anchor_r), max(h_row, anchor_r) + 1))
    output_cells.update((anchor_r, c) for c in range(min(v_col, anchor_c), max(v_col, anchor_c) + 1))

    return {
        "color": int(color),
        "height": int(height),
        "width": int(width),
        "input_cells": sorted(input_cells),
        "output_cells": sorted(output_cells),
    }


def _placement_ok(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    r0 = max(0, top - PLACEMENT_PADDING)
    c0 = max(0, left - PLACEMENT_PADDING)
    r1 = min(occupied.shape[0], top + height + PLACEMENT_PADDING)
    c1 = min(occupied.shape[1], left + width + PLACEMENT_PADDING)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_placement(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    r0 = max(0, top - PLACEMENT_PADDING)
    c0 = max(0, left - PLACEMENT_PADDING)
    r1 = min(occupied.shape[0], top + height + PLACEMENT_PADDING)
    c1 = min(occupied.shape[1], left + width + PLACEMENT_PADDING)
    occupied[r0:r1, c0:c1] = True


def _paint_pair(grid: np.ndarray, pair: dict[str, object], top: int, left: int, key: str) -> None:
    color = int(pair["color"])
    for r, c in pair[key]:
        grid[top + int(r), left + int(c)] = color


def _max_pair_count(height: int, width: int) -> int:
    area_limited = max(1, (height * width) // 80)
    span_limited = max(1, min(height, width) // 7)
    return min(PAIR_COUNT_MAX, area_limited, span_limited, len(PALETTE))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_GRID_SIDE, MAX_GRID_SIDE)
    width = random.randint(MIN_GRID_SIDE, MAX_GRID_SIDE)
    max_pairs = _max_pair_count(height, width)
    pair_count = random.randint(1, max_pairs)

    colors = random.sample(PALETTE, k=pair_count)
    branches = random.sample(BRANCHES, k=pair_count)
    pairs = [_build_pair(branch, color) for branch, color in zip(branches, colors)]
    random.shuffle(pairs)

    inp = np.full((height, width), BLACK, dtype=int)
    expected = np.full((height, width), BLACK, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    for pair in pairs:
        ph = int(pair["height"])
        pw = int(pair["width"])
        if ph > height or pw > width:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(0, height - ph)
            left = random.randint(0, width - pw)
            if not _placement_ok(occupied, top, left, ph, pw):
                continue
            _paint_pair(inp, pair, top, left, "input_cells")
            _paint_pair(expected, pair, top, left, "output_cells")
            _mark_placement(occupied, top, left, ph, pw)
            placed = True
            break
        if not placed:
            return None

    if int(np.count_nonzero(inp == BLACK)) <= int(np.count_nonzero(inp != BLACK)):
        return None

    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
