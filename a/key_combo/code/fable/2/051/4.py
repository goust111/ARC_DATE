from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg


BG = Color.BLACK
BLUE = Color.BLUE
PALETTE = [
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
MAX_GRID = 30
MAX_RETRIES = 400


def _connect_blue_markers(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    height, width = g.shape

    for r in range(height):
        cols = [int(c) for c in np.flatnonzero(g[r] == BLUE)]
        for c0, c1 in zip(cols, cols[1:]):
            if c1 > c0 + 1:
                assert np.all(g[r, c0 + 1 : c1] == BG)
                out[r, c0 + 1 : c1] = BLUE

    for c in range(width):
        rows = [int(r) for r in np.flatnonzero(g[:, c] == BLUE)]
        for r0, r1 in zip(rows, rows[1:]):
            if r1 > r0 + 1:
                assert np.all(g[r0 + 1 : r1, c] == BG)
                out[r0 + 1 : r1, c] = BLUE

    return out


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    height, width = map(int, shape)
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any(rows == 0)
        or np.any(rows == height - 1)
        or np.any(cols == 0)
        or np.any(cols == width - 1)
    )


def _find_chambers(connected: np.ndarray) -> list[dict]:
    chambers: list[dict] = []
    for coords in components_mask(connected != BLUE, connectivity=4):
        if _touches_edge(coords, connected.shape):
            continue

        rows = coords[:, 0]
        cols = coords[:, 1]
        vals = connected[rows, cols]
        color_mask = vals != BG
        assert int(color_mask.sum()) == 1

        control = coords[color_mask][0]
        chamber = {
            "coords": coords,
            "area": int(np.count_nonzero(vals == BG)),
            "color": int(vals[color_mask][0]),
            "control": (int(control[0]), int(control[1])),
        }
        assert chamber["area"] > 0
        chambers.append(chamber)

    assert chambers
    areas = [int(chamber["area"]) for chamber in chambers]
    assert len(set(areas)) == len(areas)
    return chambers


def _fill_from_order(connected: np.ndarray, ordered: list[dict], *, self_fill: bool = False) -> np.ndarray:
    out = connected.copy()
    count = len(ordered)
    for idx, chamber in enumerate(ordered):
        fill_color = chamber["color"] if self_fill else ordered[(idx - 1) % count]["color"]
        rows = chamber["coords"][:, 0]
        cols = chamber["coords"][:, 1]
        mask = connected[rows, cols] == BG
        out[rows[mask], cols[mask]] = fill_color
    return out


def _cycle_map(ordered: list[dict]) -> dict[int, int]:
    count = len(ordered)
    return {
        int(ordered[idx]["color"]): int(ordered[(idx - 1) % count]["color"])
        for idx in range(count)
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect aligned blue markers into chambers, then recolor each chamber with the previous chamber's control color in area order.

    Input structure:
    - The background is black.
    - Blue marker pixels are placed so that some share rows and some share columns.
    - Other non-black, non-blue colors appear only as distinct single control pixels.

    Rule:
    - First connect each row's consecutive blue markers with blue on the background cells strictly between them, and do the same for each column.
    - In the connected blue network, treat every 4-connected non-blue region that does not touch the grid edge as a chamber; each such chamber contains exactly one colored control pixel.
    - Sort chambers by their background-cell area from small to large, cyclically shift the chamber colors backward by one, and fill each chamber's background cells with the previous chamber's control color while preserving every control pixel in place.

    Key insight:
    - The fill targets are not present in the input at all; they only emerge after the blue connections create enclosed non-edge regions, and the color for each chamber comes from a different chamber chosen by area order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BG

    for color in np.unique(g):
        value = int(color)
        if value not in (BG, BLUE):
            assert int(np.count_nonzero(g == value)) == 1

    connected = _connect_blue_markers(g)
    chambers = _find_chambers(connected)
    ordered = sorted(chambers, key=lambda chamber: (int(chamber["area"]), chamber["control"]))
    return _fill_from_order(connected, ordered)


def _sample_non_monotone(count: int) -> list[int]:
    values = list(range(2, 7))
    for _ in range(80):
        seq = random.sample(values, count)
        if seq != sorted(seq) and seq != sorted(seq, reverse=True):
            return seq
    raise RuntimeError("failed to sample a non-monotone sequence")


def _choose_layout() -> tuple[list[int], list[int]]:
    branch = random.random()
    if branch < 0.4:
        for _ in range(120):
            row_heights = random.sample([2, 3, 4, 5], 2)
            col_widths = random.sample([2, 3, 4, 5, 6], 2)
            areas = [rh * cw for rh in row_heights for cw in col_widths]
            if len(set(areas)) != 4:
                continue
            row_major = [
                {"color": idx, "area": area, "control": (idx, 0)}
                for idx, area in enumerate(areas)
            ]
            area_order = sorted(row_major, key=lambda chamber: chamber["area"])
            if _cycle_map(area_order) != _cycle_map(row_major):
                return row_heights, col_widths
        raise RuntimeError("failed to sample a 2x2 layout")
    if branch < 0.7:
        return [random.randint(3, 6)], _sample_non_monotone(random.choice([3, 4, 5]))
    return _sample_non_monotone(random.choice([3, 4, 5])), [random.randint(3, 6)]


def _pick_margins(inner: int) -> tuple[int, int]:
    remain = MAX_GRID - inner
    assert remain >= 2
    low = 1
    high = min(3, remain - 1)
    first = random.randint(low, high)
    second = random.randint(low, min(3, remain - first))
    return first, second


def _line_positions(start: int, spans: list[int]) -> list[int]:
    pos = [int(start)]
    cur = int(start)
    for span in spans:
        cur += int(span) + 1
        pos.append(cur)
    return pos


def _render_expected(
    grid: np.ndarray,
    line_rows: list[int],
    line_cols: list[int],
    chambers: list[dict],
    ordered: list[dict],
    *,
    self_fill: bool = False,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    top, bottom = int(line_rows[0]), int(line_rows[-1])
    left, right = int(line_cols[0]), int(line_cols[-1])

    for r in line_rows:
        out[int(r), left : right + 1] = BLUE
    for c in line_cols:
        out[top : bottom + 1, int(c)] = BLUE

    count = len(ordered)
    for idx, chamber in enumerate(ordered):
        fill_color = chamber["color"] if self_fill else ordered[(idx - 1) % count]["color"]
        sub = out[chamber["r0"] : chamber["r1"], chamber["c0"] : chamber["c1"]]
        sub[sub == BG] = fill_color

    return out


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    row_heights, col_widths = _choose_layout()
    inner_h = sum(row_heights) + len(row_heights) + 1
    inner_w = sum(col_widths) + len(col_widths) + 1
    assert inner_h <= MAX_GRID - 2 and inner_w <= MAX_GRID - 2

    top, bottom = _pick_margins(inner_h)
    left, right = _pick_margins(inner_w)
    grid = np.full((top + inner_h + bottom, left + inner_w + right), BG, dtype=int)

    line_rows = _line_positions(top, row_heights)
    line_cols = _line_positions(left, col_widths)
    for r in line_rows:
        for c in line_cols:
            grid[int(r), int(c)] = BLUE

    chamber_count = len(row_heights) * len(col_widths)
    color_pool = random.sample(PALETTE, chamber_count + 2)
    chamber_colors = color_pool[:chamber_count]
    distractor_colors = color_pool[chamber_count:]

    chambers: list[dict] = []
    color_idx = 0
    for i, height in enumerate(row_heights):
        for j, width in enumerate(col_widths):
            r0 = line_rows[i] + 1
            r1 = line_rows[i + 1]
            c0 = line_cols[j] + 1
            c1 = line_cols[j + 1]
            slots = [(r, c) for r in range(r0, r1) for c in range(c0, c1)]
            control = random.choice(slots)
            color = chamber_colors[color_idx]
            color_idx += 1
            grid[control] = color
            chambers.append(
                {
                    "r0": int(r0),
                    "r1": int(r1),
                    "c0": int(c0),
                    "c1": int(c1),
                    "area": int(height * width - 1),
                    "color": int(color),
                    "control": (int(control[0]), int(control[1])),
                }
            )

    top_edge, bottom_edge = line_rows[0], line_rows[-1]
    left_edge, right_edge = line_cols[0], line_cols[-1]
    open_slots = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if not (top_edge <= r <= bottom_edge and left_edge <= c <= right_edge)
    ]
    for color, pos in zip(distractor_colors, random.sample(open_slots, len(distractor_colors))):
        grid[pos] = int(color)

    ordered = sorted(chambers, key=lambda chamber: (int(chamber["area"]), chamber["control"]))
    expected = _render_expected(grid, line_rows, line_cols, chambers, ordered)

    row_major = sorted(chambers, key=lambda chamber: chamber["control"])
    descending = sorted(chambers, key=lambda chamber: (-int(chamber["area"]), chamber["control"]))
    assert _cycle_map(ordered) != _cycle_map(row_major)
    assert _cycle_map(ordered) != _cycle_map(descending)
    assert not np.array_equal(expected, _render_expected(grid, line_rows, line_cols, chambers, ordered, self_fill=True))
    assert not np.array_equal(expected, _render_expected(grid, line_rows, line_cols, chambers, row_major))
    assert not np.array_equal(expected, _render_expected(grid, line_rows, line_cols, chambers, descending))
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        try:
            grid, expected = _build_instance()
            actual = main(grid)
            if not np.array_equal(actual, expected):
                continue

            connected = _connect_blue_markers(grid)
            chambers = _find_chambers(connected)
            if len(chambers) < 3:
                continue
            if int(np.count_nonzero((grid != BG) & (grid != BLUE))) != len(chambers) + 2:
                continue
            if len(set(int(chamber["area"]) for chamber in chambers)) != len(chambers):
                continue
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
