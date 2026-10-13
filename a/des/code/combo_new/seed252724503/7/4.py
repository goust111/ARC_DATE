from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
MAX_GRID_SIDE = 30
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 250
LEGEND_SEARCH_ROWS = 4
SIDES = ("top", "bottom", "left", "right")


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    limit = min(LEGEND_SEARCH_ROWS, h - 1)
    for r in range(limit):
        cols = [c for c in range(w) if int(g[r, c]) != BG or int(g[r + 1, c]) != BG]
        if len(cols) < 2:
            continue
        if any((int(g[r, c]) == BG) != (int(g[r + 1, c]) == BG) for c in cols):
            continue
        if any(cols[i + 1] == cols[i] + 1 for i in range(len(cols) - 1)):
            continue

        mapping: dict[int, int] = {}
        ok = True
        for c in cols:
            border = int(g[r, c])
            fill = int(g[r + 1, c])
            if border == BG or fill == BG:
                ok = False
                break
            if border in mapping and mapping[border] != fill:
                ok = False
                break
            mapping[border] = fill
        if not ok or len(mapping) < 2:
            continue

        legend_mask = np.zeros_like(g, dtype=bool)
        for c in cols:
            legend_mask[r, c] = True
            legend_mask[r + 1, c] = True
        return mapping, legend_mask

    raise AssertionError("could not find the top legend")


def _consecutive(values: list[int]) -> bool:
    return all(values[i + 1] == values[i] + 1 for i in range(len(values) - 1))


def _parse_frame(grid: np.ndarray, cells: list[tuple[int, int]]) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    color = int(g[cells[0]])
    assert all(int(g[r, c]) == color for r, c in cells), "frame component must be monochrome"

    r0, c0, r1, c1 = _bbox(cells)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height >= 4 and width >= 4, "frame bbox must be at least 4x4"

    for r, c in cells:
        assert r in (r0, r1) or c in (c0, c1), "component must stay on its bbox perimeter"
    assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == BG), "frame interior must be empty"
    assert int(g[r0, c0]) == color and int(g[r0, c1]) == color
    assert int(g[r1, c0]) == color and int(g[r1, c1]) == color

    top_missing = [c for c in range(c0 + 1, c1) if int(g[r0, c]) == BG]
    bottom_missing = [c for c in range(c0 + 1, c1) if int(g[r1, c]) == BG]
    left_missing = [r for r in range(r0 + 1, r1) if int(g[r, c0]) == BG]
    right_missing = [r for r in range(r0 + 1, r1) if int(g[r, c1]) == BG]

    gaps = []
    if top_missing:
        gaps.append(("top", top_missing))
    if bottom_missing:
        gaps.append(("bottom", bottom_missing))
    if left_missing:
        gaps.append(("left", left_missing))
    if right_missing:
        gaps.append(("right", right_missing))
    assert len(gaps) == 1, "each frame must have exactly one missing edge segment"

    side, positions = gaps[0]
    assert len(positions) in (1, 2), "gap width must be 1 or 2"
    assert _consecutive(positions), "gap must be contiguous"

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "positions": positions,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top legend to choose each frame's fill color, then draw a stripe inside the frame from its border gap.

    Input structure:
    - The background is black.
    - Near the top edge there is a horizontal legend made of 1x2 vertical color pairs, separated by black columns.
    - Each legend pair uses its top pixel as a frame color and its bottom pixel as the matching stripe color.
    - Elsewhere the grid contains several one-pixel-thick rectangular frames with black interiors, and each frame has exactly one 1-2 pixel gap on one side.

    Rule:
    - Read the legend as a border-color to fill-color mapping.
    - For each rectangular frame, identify its bounding box and the side and width of its unique gap.
    - If the gap is on the top or bottom edge, fill the interior cells in the same column span as the gap.
    - If the gap is on the left or right edge, fill the interior cells in the same row span as the gap.
    - Leave the legend and the border pixels unchanged.

    Key insight:
    - The task combines two independent inferences: the top legend determines which color to draw, and the gap location determines the stripe's orientation and width.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mapping, legend_mask = _find_legend(g)
    frame_mask = (g != BG) & ~legend_mask
    components = _components(frame_mask)
    assert components, "expected at least one frame component"

    out = g.copy()
    for comp in components:
        frame = _parse_frame(g, comp)
        color = int(frame["color"])
        assert color in mapping, "frame color must appear in the legend"
        fill = int(mapping[color])
        r0, c0, r1, c1 = frame["bbox"]
        side = str(frame["side"])
        positions = list(frame["positions"])

        if side in ("top", "bottom"):
            for c in positions:
                out[r0 + 1 : r1, c] = fill
        else:
            for r in positions:
                out[r, c0 + 1 : c1] = fill

    return out


def _side_schedule(count: int, rng: random.Random) -> list[str]:
    if count == 1:
        return [rng.choice(SIDES)]

    sides = [rng.choice(("top", "bottom")), rng.choice(("left", "right"))]
    while len(sides) < count:
        sides.append(SIDES[(len(sides) - 2) % len(SIDES)])
    rng.shuffle(sides)
    return sides


def _can_place(occupied: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - 1)
    rr1 = min(h, r0 + height + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(w, c0 + width + 1)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    h, w = occupied.shape
    rr0 = max(0, r0 - 1)
    rr1 = min(h, r0 + height + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(w, c0 + width + 1)
    occupied[rr0:rr1, cc0:cc1] = True


def _draw_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    height: int,
    width: int,
    color: int,
    side: str,
    gap_width: int,
    rng: random.Random,
) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if side == "top":
        start = rng.randint(c0 + 1, c1 - gap_width)
        grid[r0, start : start + gap_width] = BG
    elif side == "bottom":
        start = rng.randint(c0 + 1, c1 - gap_width)
        grid[r1, start : start + gap_width] = BG
    elif side == "left":
        start = rng.randint(r0 + 1, r1 - gap_width)
        grid[start : start + gap_width, c0] = BG
    else:
        start = rng.randint(r0 + 1, r1 - gap_width)
        grid[start : start + gap_width, c1] = BG


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.randint(2, 4)
    pair_count = rng.randint(object_count, 4)

    min_side = 16 if object_count == 2 else 19 if object_count == 3 else 22
    height = rng.randint(min_side, MAX_GRID_SIDE)
    width = rng.randint(min_side, MAX_GRID_SIDE)

    legend_width = 2 * pair_count - 1
    if width < legend_width + 2 or height < 8:
        return None

    color_pool = list(COLORS)
    rng.shuffle(color_pool)
    border_colors = color_pool[:pair_count]
    fill_colors = color_pool[pair_count : 2 * pair_count]
    if len(fill_colors) < pair_count:
        return None

    grid = np.full((height, width), BG, dtype=int)
    legend_start = rng.randint(0, width - legend_width)
    for idx, (border, fill) in enumerate(zip(border_colors, fill_colors)):
        col = legend_start + 2 * idx
        grid[0, col] = border
        grid[1, col] = fill

    specs: list[dict[str, object]] = []
    active_borders = rng.sample(border_colors, object_count)
    side_schedule = _side_schedule(object_count, rng)
    max_dim = 9 if object_count == 2 else 8 if object_count == 3 else 7
    for border, side in zip(active_borders, side_schedule):
        h_cap = min(max_dim, height - 5)
        w_cap = min(max_dim, width - 2)
        if h_cap < 4 or w_cap < 4:
            return None
        frame_h = rng.randint(4, h_cap)
        frame_w = rng.randint(4, w_cap)
        gap_width = rng.choice((1, 2))
        specs.append(
            {
                "color": border,
                "side": side,
                "height": frame_h,
                "width": frame_w,
                "gap_width": gap_width,
            }
        )

    specs.sort(key=lambda spec: int(spec["height"]) * int(spec["width"]), reverse=True)

    occupied = np.zeros((height, width), dtype=bool)
    occupied[:4, :] = True
    placements: list[dict[str, object]] = []
    for spec in specs:
        frame_h = int(spec["height"])
        frame_w = int(spec["width"])
        row_min = 4
        row_max = height - frame_h - 1
        col_min = 1
        col_max = width - frame_w - 1
        if row_max < row_min or col_max < col_min:
            return None

        placed = False
        for _ in range(PLACE_ATTEMPTS):
            r0 = rng.randint(row_min, row_max)
            c0 = rng.randint(col_min, col_max)
            if not _can_place(occupied, r0, c0, frame_h, frame_w):
                continue
            _mark_occupied(occupied, r0, c0, frame_h, frame_w)
            placed_spec = dict(spec)
            placed_spec["r0"] = r0
            placed_spec["c0"] = c0
            placements.append(placed_spec)
            placed = True
            break
        if not placed:
            return None

    for spec in placements:
        _draw_frame(
            grid,
            int(spec["r0"]),
            int(spec["c0"]),
            int(spec["height"]),
            int(spec["width"]),
            int(spec["color"]),
            str(spec["side"]),
            int(spec["gap_width"]),
            rng,
        )

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
