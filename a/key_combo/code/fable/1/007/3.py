from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
YELLOW = 4
INPUT_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
MIN_SEG_LEN = 2
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _interval_distance(x: int, lo: int, hi: int) -> int:
    if lo <= x <= hi:
        return 0
    if x < lo:
        return lo - x
    return x - hi


def _extract_segments(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    horizontals: list[dict] = []
    verticals: list[dict] = []

    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert YELLOW not in colors, "yellow is reserved for tie cells"

    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

        if r0 == r1:
            assert coords.shape[0] == c1 - c0 + 1
            assert c1 - c0 + 1 >= MIN_SEG_LEN
            assert np.all(g[r0, c0 : c1 + 1] == color)
            horizontals.append({"color": color, "row": r0, "c0": c0, "c1": c1})
        elif c0 == c1:
            assert coords.shape[0] == r1 - r0 + 1
            assert r1 - r0 + 1 >= MIN_SEG_LEN
            assert np.all(g[r0 : r1 + 1, c0] == color)
            verticals.append({"color": color, "col": c0, "r0": r0, "r1": r1})
        else:
            raise AssertionError("each color must form a one-cell-thick straight bar")

    horizontals.sort(key=lambda seg: seg["row"])
    verticals.sort(key=lambda seg: seg["col"])
    assert horizontals and verticals
    assert len({seg["row"] for seg in horizontals}) == len(horizontals)
    assert len({seg["col"] for seg in verticals}) == len(verticals)
    return bg, horizontals, verticals


def _solve_from_segments(
    shape: tuple[int, int],
    bg: int,
    horizontals: list[dict],
    verticals: list[dict],
    mode: str,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for seg in horizontals:
        out[seg["row"], :] = seg["color"]
    for seg in verticals:
        out[:, seg["col"]] = seg["color"]

    for hseg in horizontals:
        row = hseg["row"]
        for vseg in verticals:
            col = vseg["col"]
            on_h = hseg["c0"] <= col <= hseg["c1"]
            on_v = vseg["r0"] <= row <= vseg["r1"]
            assert not (on_h and on_v), "input bars must not overlap"

            if on_h:
                out[row, col] = hseg["color"]
                continue
            if on_v:
                out[row, col] = vseg["color"]
                continue

            dh = _interval_distance(col, hseg["c0"], hseg["c1"])
            dv = _interval_distance(row, vseg["r0"], vseg["r1"])

            if mode == "correct":
                out[row, col] = hseg["color"] if dh < dv else vseg["color"] if dv < dh else YELLOW
            elif mode == "membership_only":
                out[row, col] = YELLOW
            elif mode == "vertical_priority":
                out[row, col] = vseg["color"]
            elif mode == "no_tie":
                out[row, col] = hseg["color"] if dh <= dv else vseg["color"]
            else:
                raise ValueError(f"unknown mode: {mode}")

    return out


def _crossing_stats(horizontals: list[dict], verticals: list[dict]) -> Counter:
    stats: Counter = Counter()
    for hseg in horizontals:
        row = hseg["row"]
        for vseg in verticals:
            col = vseg["col"]
            on_h = hseg["c0"] <= col <= hseg["c1"]
            on_v = vseg["r0"] <= row <= vseg["r1"]
            assert not (on_h and on_v)
            if on_h:
                stats["zero_h"] += 1
                continue
            if on_v:
                stats["zero_v"] += 1
                continue

            dh = _interval_distance(col, hseg["c0"], hseg["c1"])
            dv = _interval_distance(row, vseg["r0"], vseg["r1"])
            if dh == dv:
                stats["tie"] += 1
            elif dh < dv:
                stats["pos_h"] += 1
            else:
                stats["pos_v"] += 1
    return stats


def _has_color_contact(grid: np.ndarray, bg: int) -> bool:
    g = np.asarray(grid, dtype=int)
    down = (g[:-1, :] != bg) & (g[1:, :] != bg) & (g[:-1, :] != g[1:, :])
    right = (g[:, :-1] != bg) & (g[:, 1:] != bg) & (g[:, :-1] != g[:, 1:])
    return bool(down.any() or right.any())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each colored short bar to its full row or column, then let the nearer source bar win each crossing.

    Input structure:
    - A single background contains several one-cell-thick colored short bars.
    - Each foreground color appears exactly once, as either one horizontal segment or one vertical segment.
    - Horizontal bars occupy distinct rows, vertical bars occupy distinct columns, and yellow does not appear.

    Rule:
    - Extend every horizontal bar across its whole row in its own color.
    - Extend every vertical bar across its whole column in its own color.
    - At each horizontal/vertical crossing, compare the crossing cell's distance to the original horizontal segment and to the original vertical segment, using distance 0 for cells already inside that source segment.
    - The closer source color fills the crossing; if the two distances are equal, recolor that crossing yellow.
    - Original short-bar cells keep their original color.

    Key insight:
    - Crossings are resolved by two competing one-dimensional distance fields anchored on the original short segments, not by a fixed draw order or by bar orientation alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, horizontals, verticals = _extract_segments(g)
    return _solve_from_segments(g.shape, bg, horizontals, verticals, mode="correct")


def _render_input(shape: tuple[int, int], horizontals: list[dict], verticals: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for seg in horizontals:
        row = seg["row"]
        c0 = seg["c0"]
        c1 = seg["c1"]
        assert np.all(grid[row, c0 : c1 + 1] == BG)
        grid[row, c0 : c1 + 1] = seg["color"]
    for seg in verticals:
        col = seg["col"]
        r0 = seg["r0"]
        r1 = seg["r1"]
        assert np.all(grid[r0 : r1 + 1, col] == BG)
        grid[r0 : r1 + 1, col] = seg["color"]
    return grid


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int)
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = out.T
    return np.array(out, copy=True)


def _build_canonical_case() -> np.ndarray:
    tie_dist = random.randint(1, 3)
    h0_len = random.randint(3, 6)
    h0_start = random.randint(1, 4)
    h0_end = h0_start + h0_len - 1
    v0_col = random.randint(h0_start, h0_end)
    v2_col = h0_end + tie_dist
    h_win_dist = tie_dist + random.randint(1, 3)
    v1_col = h0_end + h_win_dist

    h1_gap = random.randint(2, 4)
    h1_len = random.randint(3, 5)
    h1_start = v1_col + h1_gap
    h1_end = h1_start + h1_len - 1

    v2_len = random.randint(2, 4)
    h0_row = tie_dist + v2_len + random.randint(2, 4)

    row_gap_min = max(h_win_dist + 3, 6)
    row_gap_max = min(11, row_gap_min + 3)
    row_gap = random.randint(row_gap_min, row_gap_max)
    h1_row = h0_row + row_gap

    v0_top_dist = random.randint(2, min(4, row_gap - 3))
    v0_bottom_dist = random.randint(1, min(3, row_gap - v0_top_dist - 1))
    v0_r0 = h0_row + v0_top_dist
    v0_r1 = h1_row - v0_bottom_dist

    v1_up = random.randint(0, min(2, row_gap - h_win_dist - 1))
    v1_down = random.randint(1, 3)
    v1_r0 = h1_row - v1_up
    v1_r1 = h1_row + v1_down

    v2_r1 = h0_row - tie_dist
    v2_r0 = v2_r1 - v2_len + 1

    right_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)
    width = h1_end + right_margin + 1
    height = v1_r1 + bottom_margin + 1
    assert width <= 30 and height <= 30

    colors = random.sample(INPUT_COLORS, 5)
    horizontals = [
        {"color": colors[0], "row": h0_row, "c0": h0_start, "c1": h0_end},
        {"color": colors[1], "row": h1_row, "c0": h1_start, "c1": h1_end},
    ]
    verticals = [
        {"color": colors[2], "col": v0_col, "r0": v0_r0, "r1": v0_r1},
        {"color": colors[3], "col": v1_col, "r0": v1_r0, "r1": v1_r1},
        {"color": colors[4], "col": v2_col, "r0": v2_r0, "r1": v2_r1},
    ]
    return _render_input((height, width), horizontals, verticals)


def _validate_generated_grid(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    bg, horizontals, verticals = _extract_segments(g)
    assert bg == BG
    assert not _has_color_contact(g, bg), "different-color bars should stay visually separated"

    for seg in horizontals:
        assert seg["c1"] - seg["c0"] + 1 < g.shape[1]
    for seg in verticals:
        assert seg["r1"] - seg["r0"] + 1 < g.shape[0]

    stats = _crossing_stats(horizontals, verticals)
    assert stats["zero_h"] >= 1
    assert stats["zero_v"] >= 1
    assert stats["pos_h"] >= 1
    assert stats["pos_v"] >= 1
    assert stats["tie"] >= 1

    out = main(g)
    assert out.shape == g.shape
    assert np.count_nonzero(out != bg) > np.count_nonzero(g != bg)
    assert np.any(out == YELLOW)

    membership_only = _solve_from_segments(g.shape, bg, horizontals, verticals, mode="membership_only")
    vertical_priority = _solve_from_segments(g.shape, bg, horizontals, verticals, mode="vertical_priority")
    no_tie = _solve_from_segments(g.shape, bg, horizontals, verticals, mode="no_tie")
    assert not np.array_equal(out, membership_only)
    assert not np.array_equal(out, vertical_priority)
    assert not np.array_equal(out, no_tie)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _random_transform(_build_canonical_case())
        try:
            _validate_generated_grid(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid distance-priority cross grid")
