from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import components_mask, get_bg


BG = 0
GUIDE = 5
OBJECT = 7
MARKER_COLORS = (1, 2, 3, 4)
GEN_ATTEMPTS = 400
NEIGHBORS8 = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
]


def _line_span(center_row: int, length: int) -> tuple[int, int]:
    start = int(center_row) - (int(length) - 1) // 2
    end = int(center_row) + int(length) // 2
    return start, end


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_marker_blocks(g: np.ndarray) -> tuple[tuple[int, int, np.ndarray], tuple[int, int, np.ndarray]]:
    marker_mask = np.isin(g, MARKER_COLORS)
    comps = components_mask(marker_mask, connectivity=4)
    assert len(comps) == 2, "expected exactly two marker blocks"

    blocks: list[tuple[int, int, np.ndarray]] = []
    for comp in comps:
        assert int(comp.shape[0]) == 4, "each marker block must have four cells"
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        assert r1 == r0 + 1 and c1 == c0 + 1, "each marker block must be 2x2"
        block = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert sorted(map(int, block.ravel())) == list(MARKER_COLORS), "marker block must use 1,2,3,4 once each"
        blocks.append((r0, c0, block))

    blocks.sort(key=lambda item: item[1])
    return blocks[0], blocks[1]


def _parse_input(g: np.ndarray) -> dict[str, object]:
    h, w = map(int, g.shape)
    assert get_bg(g) == BG, "background must be black"

    left_marker, right_marker = _extract_marker_blocks(g)
    left_r0, left_c0, left_block = left_marker
    right_r0, right_c0, right_block = right_marker

    guide_mask = g == GUIDE
    col_counts = np.count_nonzero(guide_mask, axis=0)
    bar_cols = np.flatnonzero(col_counts == h).tolist()
    assert len(bar_cols) == 2, "expected exactly two full-height guide bars"
    old_left_bar, old_right_bar = map(int, sorted(bar_cols))
    assert (old_left_bar + old_right_bar) % 2 == 0, "guide bars must have an integral midpoint"
    center_col = (old_left_bar + old_right_bar) // 2

    row_counts = np.count_nonzero(guide_mask, axis=1)
    max_row_count = int(row_counts.max())
    center_rows = np.flatnonzero(row_counts == max_row_count).tolist()
    assert len(center_rows) == 1, "expected a unique horizontal guide row"
    center_row = int(center_rows[0])
    assert max_row_count >= w - 4, "horizontal guide row must span the width except marker overlaps"

    object_components = components_mask(g == OBJECT, connectivity=8)
    assert len(object_components) == 1, "expected one 8-connected orange object"
    object_coords = object_components[0]
    area = int(object_coords.shape[0])
    assert 4 <= area <= 10, "object area out of range"

    rows = object_coords[:, 0]
    cols = object_coords[:, 1]
    assert np.all(cols > old_left_bar + 1) and np.all(cols < old_right_bar - 1), "object must stay away from the guide bars"
    assert np.all(rows < center_row - 1) or np.all(rows > center_row + 1), "object must stay in one half-plane"

    assert left_c0 == 0 and right_c0 == w - 2, "markers must anchor the left and right ends"
    assert left_r0 == center_row - 1 and right_r0 == center_row - 1, "markers must straddle the guide row"

    return {
        "shape": (h, w),
        "center_row": center_row,
        "center_col": center_col,
        "old_bars": (old_left_bar, old_right_bar),
        "area": area,
        "left_marker": left_block,
        "right_marker": right_block,
    }


def _fill_side_regions(
    out: np.ndarray,
    *,
    row_lo: int,
    row_hi: int,
    left_bar_col: int,
    right_bar_col: int,
    marker: np.ndarray,
) -> None:
    assert row_lo <= row_hi, "fill region must be non-empty"
    left_cols = np.arange(0, left_bar_col, dtype=int)
    right_cols = np.arange(right_bar_col + 1, out.shape[1], dtype=int)
    assert left_cols.size > 0 and right_cols.size > 0, "side wings must exist"

    mid_row = (row_lo + row_hi) // 2
    upper_rows = np.arange(row_lo, mid_row + 1, dtype=int)
    lower_rows = np.arange(mid_row + 1, row_hi + 1, dtype=int)
    assert upper_rows.size > 0 and lower_rows.size > 0, "side region must split into four non-empty quadrants"

    out[np.ix_(lower_rows, right_cols)] = int(marker[0, 0])
    out[np.ix_(lower_rows, left_cols)] = int(marker[0, 1])
    out[np.ix_(upper_rows, right_cols)] = int(marker[1, 0])
    out[np.ix_(upper_rows, left_cols)] = int(marker[1, 1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the interior orange object into a centered vertical line, move the guide bars inward, and use the two marker blocks to diagonally recolor the upper and lower side regions.

    Input structure:
    - Black background with one grey guide structure: two full-height vertical bars and one full-width horizontal bar.
    - A single 8-connected orange object lies strictly between the two vertical bars and entirely above or below the horizontal bar.
    - Two 2x2 marker blocks sit at the left and right ends of the horizontal bar, each using the four colors 1,2,3,4 exactly once.

    Rule:
    - Count the orange object pixels and erase the original object and the original pair of vertical guide bars.
    - Draw a centered vertical orange line on the horizontal guide row midpoint; for even lengths the line extends one cell farther downward than upward.
    - Rebuild the two grey vertical bars immediately to the left and right of the orange line.
    - Above and below the new line, split the left and right side wings into a logical 2x2 layout and fill the diagonally opposite quadrant colors from the left marker block (upper side) and the right marker block (lower side).

    Key insight:
    - The object area determines the compressed line length, and only after the guide bars are rebuilt does the final pair of side regions become well-defined enough for the diagonal marker mapping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    scene = _parse_input(g)
    h, w = scene["shape"]
    center_row = int(scene["center_row"])
    center_col = int(scene["center_col"])
    length = int(scene["area"])
    left_block = np.asarray(scene["left_marker"], dtype=int)
    right_block = np.asarray(scene["right_marker"], dtype=int)

    start_row, end_row = _line_span(center_row, length)
    assert 2 <= start_row <= center_row <= end_row <= h - 3, "compressed line must leave room above and below"
    new_left_bar = center_col - 1
    new_right_bar = center_col + 1
    assert 1 <= new_left_bar < center_col < new_right_bar <= w - 2, "new guide bars must stay inside the grid"

    out = np.full_like(g, BG)
    out[center_row, :] = GUIDE
    out[:, new_left_bar] = GUIDE
    out[:, new_right_bar] = GUIDE
    out[start_row : end_row + 1, center_col] = OBJECT

    _fill_side_regions(
        out,
        row_lo=0,
        row_hi=start_row - 1,
        left_bar_col=new_left_bar,
        right_bar_col=new_right_bar,
        marker=left_block,
    )
    _fill_side_regions(
        out,
        row_lo=end_row + 1,
        row_hi=h - 1,
        left_bar_col=new_left_bar,
        right_bar_col=new_right_bar,
        marker=right_block,
    )

    out[center_row, :] = GUIDE
    out[:, new_left_bar] = GUIDE
    out[:, new_right_bar] = GUIDE
    out[start_row : end_row + 1, center_col] = OBJECT
    out[center_row - 1 : center_row + 1, 0:2] = left_block
    out[center_row - 1 : center_row + 1, w - 2 : w] = right_block
    return out


def _grow_blob(
    area: int,
    row_range: tuple[int, int],
    col_range: tuple[int, int],
) -> list[tuple[int, int]]:
    row_lo, row_hi = row_range
    col_lo, col_hi = col_range
    assert row_lo <= row_hi and col_lo <= col_hi

    seed = (random.randint(row_lo, row_hi), random.randint(col_lo, col_hi))
    coords = {seed}
    frontier = {
        (seed[0] + dr, seed[1] + dc)
        for dr, dc in NEIGHBORS8
        if row_lo <= seed[0] + dr <= row_hi and col_lo <= seed[1] + dc <= col_hi
    }

    while len(coords) < area:
        assert frontier, "blob frontier exhausted"
        options = list(frontier)
        weights = []
        for r, c in options:
            touching = sum((r + dr, c + dc) in coords for dr, dc in NEIGHBORS8)
            spread = min(abs(r - rr) + abs(c - cc) for rr, cc in coords)
            weights.append((touching + 1) * (spread + 1))
        chosen = random.choices(options, weights=weights, k=1)[0]
        frontier.remove(chosen)
        coords.add(chosen)
        for dr, dc in NEIGHBORS8:
            nxt = (chosen[0] + dr, chosen[1] + dc)
            if row_lo <= nxt[0] <= row_hi and col_lo <= nxt[1] <= col_hi and nxt not in coords:
                frontier.add(nxt)

    return sorted(coords)


def _sample_canvas() -> tuple[int, int, int, int, int, int]:
    h = random.choice([13, 15, 17, 19, 21])
    w = random.choice([15, 17, 19, 21, 23])
    area = random.randint(4, 10)
    half_gap = random.choice([4, 5])

    center_row_lo = 2 + (area - 1) // 2
    center_row_hi = h - 3 - area // 2
    if center_row_lo > center_row_hi:
        raise AssertionError("no legal center row")
    center_row = random.randint(center_row_lo, center_row_hi)

    center_col = random.randint(half_gap + 2, w - half_gap - 3)
    old_left_bar = center_col - half_gap
    old_right_bar = center_col + half_gap
    assert old_left_bar >= 2 and old_right_bar <= w - 3
    return h, w, center_row, center_col, area, half_gap


def _place_structure(
    h: int,
    w: int,
    center_row: int,
    center_col: int,
    area: int,
    half_gap: int,
) -> tuple[np.ndarray, dict[str, object]]:
    grid = np.full((h, w), BG, dtype=int)
    old_left_bar = center_col - half_gap
    old_right_bar = center_col + half_gap
    assert old_left_bar >= 2 and old_right_bar <= w - 3

    grid[:, old_left_bar] = GUIDE
    grid[:, old_right_bar] = GUIDE
    grid[center_row, :] = GUIDE

    left_block = np.array(random.sample(MARKER_COLORS, 4), dtype=int).reshape(2, 2)
    right_block = np.array(random.sample(MARKER_COLORS, 4), dtype=int).reshape(2, 2)
    grid[center_row - 1 : center_row + 1, 0:2] = left_block
    grid[center_row - 1 : center_row + 1, w - 2 : w] = right_block

    start_row, end_row = _line_span(center_row, area)
    top_range = (1, center_row - 2)
    bottom_range = (center_row + 2, h - 2)
    col_range = (old_left_bar + 2, old_right_bar - 2)
    assert top_range[0] <= top_range[1]
    assert bottom_range[0] <= bottom_range[1]
    assert col_range[0] <= col_range[1]

    side = random.choice(["top", "bottom"])
    row_range = top_range if side == "top" else bottom_range
    blob = _grow_blob(area, row_range, col_range)
    rows, cols = zip(*blob)
    grid[rows, cols] = OBJECT

    return grid, {
        "left_block": left_block,
        "right_block": right_block,
        "line_span": (start_row, end_row),
        "new_bars": (center_col - 1, center_col + 1),
    }


def _validate_output(inp: np.ndarray, out: np.ndarray) -> None:
    scene = _parse_input(inp)
    h, w = scene["shape"]
    center_row = int(scene["center_row"])
    center_col = int(scene["center_col"])
    area = int(scene["area"])
    left_block = np.asarray(scene["left_marker"], dtype=int)
    right_block = np.asarray(scene["right_marker"], dtype=int)

    start_row, end_row = _line_span(center_row, area)
    new_left_bar = center_col - 1
    new_right_bar = center_col + 1

    assert out.shape == inp.shape
    assert np.all(out[:, new_left_bar] == GUIDE)
    assert np.all(out[:, new_right_bar] == GUIDE)
    center_row_view = out[center_row].copy()
    center_row_view[center_col] = GUIDE
    assert np.all(center_row_view[2 : w - 2] == GUIDE)
    assert np.all(out[start_row : end_row + 1, center_col] == OBJECT)
    assert np.count_nonzero(out == OBJECT) == area
    assert np.array_equal(out[center_row - 1 : center_row + 1, 0:2], left_block)
    assert np.array_equal(out[center_row - 1 : center_row + 1, w - 2 : w], right_block)

    top_mid = (start_row - 1) // 2
    bottom_mid = (end_row + 1 + h - 1) // 2
    assert np.all(out[0 : top_mid + 1, new_right_bar + 1 : w] == int(left_block[1, 0]))
    assert np.all(out[0 : top_mid + 1, 0:new_left_bar] == int(left_block[1, 1]))
    assert np.all(out[top_mid + 1 : start_row, new_right_bar + 1 : w] == int(left_block[0, 0]))
    assert np.all(out[top_mid + 1 : start_row, 0:new_left_bar] == int(left_block[0, 1]))

    assert np.all(out[end_row + 1 : bottom_mid + 1, new_right_bar + 1 : w] == int(right_block[1, 0]))
    assert np.all(out[end_row + 1 : bottom_mid + 1, 0:new_left_bar] == int(right_block[1, 1]))
    assert np.all(out[bottom_mid + 1 : h, new_right_bar + 1 : w] == int(right_block[0, 0]))
    assert np.all(out[bottom_mid + 1 : h, 0:new_left_bar] == int(right_block[0, 1]))


def _gen_attempt() -> np.ndarray | None:
    try:
        h, w, center_row, center_col, area, half_gap = _sample_canvas()
        grid, _meta = _place_structure(h, w, center_row, center_col, area, half_gap)
        out = main(grid)
        _validate_output(grid, out)
        return grid
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
