from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_RETRIES = 300
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int).ravel(), return_counts=True)
    assert vals.size, "expected at least one value"
    return int(vals[int(np.argmax(counts))])


def _edge_palette(g: np.ndarray) -> dict[str, int]:
    h, w = g.shape
    assert h >= 4 and w >= 4, "grid is too small for four edge palettes"
    return {
        "up": _mode(g[0, 1:-1]),
        "down": _mode(g[-1, 1:-1]),
        "left": _mode(g[1:-1, 0]),
        "right": _mode(g[1:-1, -1]),
    }


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue

            seen[sr, sc] = True
            stack = [(sr, sc)]
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in DIRS4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and bool(mask[rr, cc]):
                        seen[rr, cc] = True
                        stack.append((rr, cc))

            comps.append(np.array(cells, dtype=int))
    return comps


def _touches_outer_background(coords: np.ndarray, h: int, w: int) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any(rows <= 1)
        or np.any(rows >= h - 2)
        or np.any(cols <= 1)
        or np.any(cols >= w - 2)
    )


def _is_grey(g: np.ndarray, r: int, c: int) -> bool:
    return 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and int(g[r, c]) == GREY


def _window_direction(g: np.ndarray, coords: np.ndarray) -> str | None:
    cells = {(int(r), int(c)) for r, c in coords}
    candidates: list[str] = []
    specs = {
        "up": ((1, 0), ((-1, 0), (0, -1), (0, 1))),
        "down": ((-1, 0), ((1, 0), (0, -1), (0, 1))),
        "left": ((0, 1), ((0, -1), (-1, 0), (1, 0))),
        "right": ((0, -1), ((0, 1), (-1, 0), (1, 0))),
    }

    for r, c in cells:
        for name, (inward, grey_offsets) in specs.items():
            ir, ic = r + inward[0], c + inward[1]
            if (ir, ic) not in cells:
                continue
            if all(_is_grey(g, r + dr, c + dc) for dr, dc in grey_offsets):
                candidates.append(name)

    counts = Counter(candidates)
    if len(counts) == 1:
        return next(iter(counts))
    if counts:
        top_two = counts.most_common(2)
        if len(top_two) == 1 or top_two[0][1] > top_two[1][1]:
            return str(top_two[0][0])

    return _window_direction_from_bbox(cells)


def _window_direction_from_bbox(cells: set[tuple[int, int]]) -> str | None:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    candidates: list[str] = []
    top = [(r, c) for r, c in cells if r == r0]
    bottom = [(r, c) for r, c in cells if r == r1]
    left = [(r, c) for r, c in cells if c == c0]
    right = [(r, c) for r, c in cells if c == c1]

    if len(top) == 1 and (top[0][0] + 1, top[0][1]) in cells:
        candidates.append("up")
    if len(bottom) == 1 and (bottom[0][0] - 1, bottom[0][1]) in cells:
        candidates.append("down")
    if len(left) == 1 and (left[0][0], left[0][1] + 1) in cells:
        candidates.append("left")
    if len(right) == 1 and (right[0][0], right[0][1] - 1) in cells:
        candidates.append("right")

    return candidates[0] if len(candidates) == 1 else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-walled chamber from the edge palette faced by its window gap.

    Input structure:
    - The background is black, and a central grey wall network encloses several black chamber interiors.
    - Every chamber has exactly one one-pixel black window cut into its grey boundary on the top, bottom, left, or right side.
    - The top row, bottom row, left column, and right column each carry a distinct single-color palette strip.

    Rule:
    - Read the four palette colors from the corresponding outer edge strips.
    - Find black chamber components inside the grey network, excluding the exterior black background around the network.
    - For each chamber, locate the single window pixel: it is black, touches the chamber inward, and has grey wall pixels on the other three sides.
    - Fill that chamber with the top, bottom, left, or right edge color according to whether the window faces up, down, left, or right.
    - Preserve all grey wall pixels and all outer palette strips.

    Key insight:
    - The missing grey wall pixel is a directional window; its facing direction selects which outer edge palette supplies the chamber fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    palette = _edge_palette(g)
    out = g.copy()
    filled = 0

    for comp in _components(g == BG):
        if _touches_outer_background(comp, h, w):
            continue
        direction = _window_direction(g, comp)
        if direction is None:
            continue
        out[comp[:, 0], comp[:, 1]] = palette[direction]
        filled += 1

    assert filled, "expected at least one windowed chamber"
    return out


def _paint_palette(g: np.ndarray, colors: dict[str, int]) -> None:
    g[0, :] = colors["up"]
    g[-1, :] = colors["down"]
    g[1:-1, 0] = colors["left"]
    g[1:-1, -1] = colors["right"]


def _carve_chamber(
    g: np.ndarray,
    cell_r0: int,
    cell_c0: int,
    cell_h: int,
    cell_w: int,
    direction: str,
    rng: random.Random,
) -> list[tuple[int, int]]:
    max_inner_h = cell_h - 4
    max_inner_w = cell_w - 4
    assert max_inner_h >= 2 and max_inner_w >= 2

    inner_h = rng.randint(2, max_inner_h)
    inner_w = rng.randint(2, max_inner_w)
    ir0 = cell_r0 + rng.randint(2, cell_h - inner_h - 2)
    ic0 = cell_c0 + rng.randint(2, cell_w - inner_w - 2)
    ir1 = ir0 + inner_h - 1
    ic1 = ic0 + inner_w - 1

    cells = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
    for r, c in cells:
        g[r, c] = BG

    if direction == "up":
        wr, wc = ir0 - 1, rng.randint(ic0, ic1)
    elif direction == "down":
        wr, wc = ir1 + 1, rng.randint(ic0, ic1)
    elif direction == "left":
        wr, wc = rng.randint(ir0, ir1), ic0 - 1
    else:
        wr, wc = rng.randint(ir0, ir1), ic1 + 1

    g[wr, wc] = BG
    cells.append((wr, wc))
    return cells


def _punch_outer_dents(
    g: np.ndarray,
    block_r0: int,
    block_c0: int,
    block_h: int,
    block_w: int,
    rng: random.Random,
) -> None:
    block_r1 = block_r0 + block_h - 1
    block_c1 = block_c0 + block_w - 1

    def clear_enough(r: int, c: int) -> bool:
        if int(g[r, c]) != GREY:
            return False
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if block_r0 <= rr <= block_r1 and block_c0 <= cc <= block_c1 and int(g[rr, cc]) == BG:
                return False
        return True

    for _ in range(rng.randint(0, 8)):
        side = rng.choice(("top", "bottom", "left", "right"))
        if side == "top":
            r, c = block_r0, rng.randint(block_c0 + 1, block_c1 - 1)
        elif side == "bottom":
            r, c = block_r1, rng.randint(block_c0 + 1, block_c1 - 1)
        elif side == "left":
            r, c = rng.randint(block_r0 + 1, block_r1 - 1), block_c0
        else:
            r, c = rng.randint(block_r0 + 1, block_r1 - 1), block_c1

        if clear_enough(r, c):
            g[r, c] = BG


def _sample_layout(rng: random.Random) -> tuple[list[int], list[int], tuple[int, int, int, int]] | None:
    rows = rng.choice((2, 2, 3))
    cols = rng.choice((2, 3, 3))
    max_cell_h = 7 if rows == 3 else 9
    max_cell_w = 7 if cols == 3 else 9
    row_heights = [rng.randint(6, max_cell_h) for _ in range(rows)]
    col_widths = [rng.randint(6, max_cell_w) for _ in range(cols)]
    top_margin = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)
    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)

    height = 2 + top_margin + sum(row_heights) + bottom_margin
    width = 2 + left_margin + sum(col_widths) + right_margin
    if height > 30 or width > 30:
        return None
    return row_heights, col_widths, (height, width, top_margin, left_margin)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layout = _sample_layout(rng)
    if layout is None:
        return None
    row_heights, col_widths, placement = layout
    height, width, top_margin, left_margin = placement

    colors = list(PALETTE_COLORS)
    rng.shuffle(colors)
    palette = {"up": colors[0], "down": colors[1], "left": colors[2], "right": colors[3]}

    grid = np.full((height, width), BG, dtype=int)
    _paint_palette(grid, palette)

    block_r0 = 1 + top_margin
    block_c0 = 1 + left_margin
    block_h = sum(row_heights)
    block_w = sum(col_widths)
    grid[block_r0 : block_r0 + block_h, block_c0 : block_c0 + block_w] = GREY

    cell_origins: list[tuple[int, int, int, int]] = []
    r = block_r0
    for rh in row_heights:
        c = block_c0
        for cw in col_widths:
            cell_origins.append((r, c, rh, cw))
            c += cw
        r += rh

    chamber_count = rng.randint(4, len(cell_origins))
    chosen = rng.sample(range(len(cell_origins)), chamber_count)
    directions = ["up", "down", "left", "right"]
    directions.extend(rng.choice(("up", "down", "left", "right")) for _ in range(chamber_count - 4))
    rng.shuffle(directions)

    chambers: list[tuple[list[tuple[int, int]], str]] = []
    for idx, direction in zip(chosen, directions):
        cells = _carve_chamber(grid, *cell_origins[idx], direction, rng)
        chambers.append((cells, direction))

    _punch_outer_dents(grid, block_r0, block_c0, block_h, block_w, rng)

    expected = grid.copy()
    for cells, direction in chambers:
        for r, c in cells:
            expected[r, c] = palette[direction]

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")
