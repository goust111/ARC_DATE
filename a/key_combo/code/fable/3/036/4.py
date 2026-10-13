import random
from collections import Counter
from typing import Iterable

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


INPUT_COLORS = (
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)
BRANCH_SEQUENCE = ("tie", "horizontal_wins", "vertical_wins")
COUNT_OPTIONS = (
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
)
SIZE_RANGE = (18, 24)
OUTER_MARGIN = 5
ROW_COL_GAP = 3
LINE_LENGTHS = (2, 3, 4)
MAX_START_GAP = 4
GENERATION_RETRIES = 300
_GEN_CALL_COUNT = 0


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: Iterable[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _pick_spaced_values(low: int, high: int, count: int, gap: int) -> list[int] | None:
    values = list(range(int(low), int(high) + 1))
    if count > len(values):
        return None
    for _ in range(200):
        chosen = sorted(random.sample(values, count))
        if all(b - a >= gap for a, b in zip(chosen, chosen[1:], strict=False)):
            return chosen
    return None


def _horizontal_options(side: str, min_col: int, max_col: int, width: int) -> list[tuple[int, int]]:
    options = []
    for length in LINE_LENGTHS:
        if side == "left":
            max_gap = min(min_col - length + 1, MAX_START_GAP)
        else:
            max_gap = min(width - max_col - length, MAX_START_GAP)
        for gap in range(1, max_gap + 1):
            options.append((length, gap))
    return options


def _vertical_options(side: str, min_row: int, max_row: int, height: int) -> list[tuple[int, int]]:
    options = []
    for length in LINE_LENGTHS:
        if side == "top":
            max_gap = min(min_row - length + 1, MAX_START_GAP)
        else:
            max_gap = min(height - max_row - length, MAX_START_GAP)
        for gap in range(1, max_gap + 1):
            options.append((length, gap))
    return options


def _make_horizontal(
    row: int,
    side: str,
    length: int,
    gap: int,
    min_col: int,
    max_col: int,
    color: int,
) -> dict:
    if side == "left":
        c1 = min_col - gap
        c0 = c1 - length + 1
    else:
        c0 = max_col + gap
        c1 = c0 + length - 1
    return {
        "kind": "H",
        "row": int(row),
        "c0": int(c0),
        "c1": int(c1),
        "color": int(color),
        "side": side,
    }


def _make_vertical(
    col: int,
    side: str,
    length: int,
    gap: int,
    min_row: int,
    max_row: int,
    color: int,
) -> dict:
    if side == "top":
        r1 = min_row - gap
        r0 = r1 - length + 1
    else:
        r0 = max_row + gap
        r1 = r0 + length - 1
    return {
        "kind": "V",
        "col": int(col),
        "r0": int(r0),
        "r1": int(r1),
        "color": int(color),
        "side": side,
    }


def _render_segments(shape: tuple[int, int], segments: list[dict]) -> np.ndarray:
    grid = np.full(shape, Color.BLACK, dtype=int)
    for seg in segments:
        if seg["kind"] == "H":
            grid[seg["row"], seg["c0"] : seg["c1"] + 1] = seg["color"]
        else:
            grid[seg["r0"] : seg["r1"] + 1, seg["col"]] = seg["color"]
    return grid


def _extract_segments(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    segments = []
    for color in colors_present(g, exclude={Color.BLACK}):
        coords = np.argwhere(g == color)
        assert coords.shape[0] >= 2, "segments must have visible orientation"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if int(r0) == int(r1):
            cols = sorted(int(c) for c in coords[:, 1])
            expected = list(range(int(c0), int(c1) + 1))
            assert cols == expected, "horizontal segment must be contiguous"
            segments.append(
                {
                    "kind": "H",
                    "row": int(r0),
                    "c0": int(c0),
                    "c1": int(c1),
                    "color": int(color),
                }
            )
        else:
            assert int(c0) == int(c1), "segment must be horizontal or vertical"
            rows = sorted(int(r) for r in coords[:, 0])
            expected = list(range(int(r0), int(r1) + 1))
            assert rows == expected, "vertical segment must be contiguous"
            segments.append(
                {
                    "kind": "V",
                    "col": int(c0),
                    "r0": int(r0),
                    "r1": int(r1),
                    "color": int(color),
                }
            )

    assert segments, "expected at least one segment"
    rows = [seg["row"] for seg in segments if seg["kind"] == "H"]
    cols = [seg["col"] for seg in segments if seg["kind"] == "V"]
    assert rows and cols, "expected both horizontal and vertical segments"
    return segments


def _initial_endpoints(shape: tuple[int, int], segments: list[dict]) -> list[dict]:
    height, width = shape
    endpoints = []
    for seg in segments:
        if seg["kind"] == "H":
            endpoints.append(
                {
                    "r": seg["row"],
                    "c": seg["c0"],
                    "dr": 0,
                    "dc": -1,
                    "color": seg["color"],
                    "active": seg["c0"] > 0,
                }
            )
            endpoints.append(
                {
                    "r": seg["row"],
                    "c": seg["c1"],
                    "dr": 0,
                    "dc": 1,
                    "color": seg["color"],
                    "active": seg["c1"] < width - 1,
                }
            )
        else:
            endpoints.append(
                {
                    "r": seg["r0"],
                    "c": seg["col"],
                    "dr": -1,
                    "dc": 0,
                    "color": seg["color"],
                    "active": seg["r0"] > 0,
                }
            )
            endpoints.append(
                {
                    "r": seg["r1"],
                    "c": seg["col"],
                    "dr": 1,
                    "dc": 0,
                    "color": seg["color"],
                    "active": seg["r1"] < height - 1,
                }
            )
    return endpoints


def _simulate_growth(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    segments = _extract_segments(g)
    out = g.copy()
    height, width = out.shape
    endpoints = _initial_endpoints(out.shape, segments)
    stats = {
        "tie_cells": 0,
        "occupied_stops": 0,
        "boundary_stops": 0,
    }

    while any(endpoint["active"] for endpoint in endpoints):
        proposals: dict[tuple[int, int], list[int]] = {}
        for idx, endpoint in enumerate(endpoints):
            if not endpoint["active"]:
                continue
            nr = endpoint["r"] + endpoint["dr"]
            nc = endpoint["c"] + endpoint["dc"]
            if not (0 <= nr < height and 0 <= nc < width):
                endpoint["active"] = False
                continue
            if out[nr, nc] != Color.BLACK:
                endpoint["active"] = False
                stats["occupied_stops"] += 1
                continue
            proposals.setdefault((nr, nc), []).append(idx)

        if not proposals:
            break

        for (r, c), entrants in proposals.items():
            if len(entrants) >= 2:
                out[r, c] = Color.YELLOW
                stats["tie_cells"] += 1
                for idx in entrants:
                    endpoints[idx]["r"] = r
                    endpoints[idx]["c"] = c
                    endpoints[idx]["active"] = False
                continue

            idx = entrants[0]
            endpoint = endpoints[idx]
            out[r, c] = endpoint["color"]
            endpoint["r"] = r
            endpoint["c"] = c
            if r in {0, height - 1} or c in {0, width - 1}:
                endpoint["active"] = False
                stats["boundary_stops"] += 1

    stats["changed_cells"] = int(np.count_nonzero(out != g))
    stats["non_black"] = int(np.count_nonzero(out != Color.BLACK))
    return out, stats


def _forced_gap_pairs(
    branch: str,
    h_options: list[tuple[int, int]],
    v_options: list[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs = []
    for h_option in h_options:
        for v_option in v_options:
            _h_len, h_gap = h_option
            _v_len, v_gap = v_option
            if branch == "tie" and h_gap == v_gap:
                pairs.append((h_option, v_option))
            elif branch == "horizontal_wins" and h_gap < v_gap:
                pairs.append((h_option, v_option))
            elif branch == "vertical_wins" and v_gap < h_gap:
                pairs.append((h_option, v_option))
    return pairs


def _forced_prev_cell(row: int, col: int, side: str, kind: str) -> tuple[int, int]:
    if kind == "H":
        return (row, col - 1) if side == "left" else (row, col + 1)
    return (row - 1, col) if side == "top" else (row + 1, col)


def _forced_next_cell(row: int, col: int, side: str, kind: str) -> tuple[int, int]:
    if kind == "H":
        return (row, col + 1) if side == "left" else (row, col - 1)
    return (row + 1, col) if side == "top" else (row - 1, col)


def _branch_ok(
    branch: str,
    out: np.ndarray,
    row: int,
    col: int,
    h_color: int,
    v_color: int,
    h_side: str,
    v_side: str,
) -> bool:
    if branch == "tie":
        if int(out[row, col]) != Color.YELLOW:
            return False
        if int(out[_forced_prev_cell(row, col, h_side, "H")]) != h_color:
            return False
        if int(out[_forced_prev_cell(row, col, v_side, "V")]) != v_color:
            return False
        return True

    if branch == "horizontal_wins":
        if int(out[row, col]) != h_color:
            return False
        if int(out[_forced_prev_cell(row, col, v_side, "V")]) != v_color:
            return False
        if int(out[_forced_next_cell(row, col, h_side, "H")]) != h_color:
            return False
        return True

    if int(out[row, col]) != v_color:
        return False
    if int(out[_forced_prev_cell(row, col, h_side, "H")]) != h_color:
        return False
    if int(out[_forced_next_cell(row, col, v_side, "V")]) != v_color:
        return False
    return True


def _gen_attempt(branch: str) -> np.ndarray | None:
    height = random.randint(*SIZE_RANGE)
    width = random.randint(*SIZE_RANGE)
    nh, nv = random.choice(COUNT_OPTIONS)

    rows = _pick_spaced_values(OUTER_MARGIN, height - OUTER_MARGIN - 1, nh, ROW_COL_GAP)
    cols = _pick_spaced_values(OUTER_MARGIN, width - OUTER_MARGIN - 1, nv, ROW_COL_GAP)
    if rows is None or cols is None:
        return None

    min_row, max_row = rows[0], rows[-1]
    min_col, max_col = cols[0], cols[-1]

    h_sides = [random.choice(("left", "right")) for _ in range(nh)]
    v_sides = [random.choice(("top", "bottom")) for _ in range(nv)]
    forced_h_side = random.choice(("left", "right"))
    forced_v_side = random.choice(("top", "bottom"))
    forced_h_idx = 0 if forced_v_side == "top" else nh - 1
    forced_v_idx = 0 if forced_h_side == "left" else nv - 1
    h_sides[forced_h_idx] = forced_h_side
    v_sides[forced_v_idx] = forced_v_side

    forced_h_options = _horizontal_options(forced_h_side, min_col, max_col, width)
    forced_v_options = _vertical_options(forced_v_side, min_row, max_row, height)
    forced_pairs = _forced_gap_pairs(branch, forced_h_options, forced_v_options)
    if not forced_pairs:
        return None
    (forced_h_len, forced_h_gap), (forced_v_len, forced_v_gap) = random.choice(forced_pairs)

    total_segments = nh + nv
    colors = random.sample(INPUT_COLORS, total_segments)
    horizontals = []
    verticals = []

    for idx, row in enumerate(rows):
        side = h_sides[idx]
        if idx == forced_h_idx:
            length, gap = forced_h_len, forced_h_gap
        else:
            options = _horizontal_options(side, min_col, max_col, width)
            if not options:
                return None
            length, gap = random.choice(options)
        horizontals.append(
            _make_horizontal(row, side, length, gap, min_col, max_col, colors[idx])
        )

    for idx, col in enumerate(cols):
        side = v_sides[idx]
        if idx == forced_v_idx:
            length, gap = forced_v_len, forced_v_gap
        else:
            options = _vertical_options(side, min_row, max_row, height)
            if not options:
                return None
            length, gap = random.choice(options)
        verticals.append(
            _make_vertical(col, side, length, gap, min_row, max_row, colors[nh + idx])
        )

    inp = _render_segments((height, width), horizontals + verticals)
    out, stats = _simulate_growth(inp)

    forced_row = rows[forced_h_idx]
    forced_col = cols[forced_v_idx]
    h_color = horizontals[forced_h_idx]["color"]
    v_color = verticals[forced_v_idx]["color"]

    if not _branch_ok(
        branch,
        out,
        forced_row,
        forced_col,
        h_color,
        v_color,
        forced_h_side,
        forced_v_side,
    ):
        return None
    if branch == "tie" and stats["tie_cells"] < 1:
        return None
    if stats["occupied_stops"] + stats["tie_cells"] < 1:
        return None
    if stats["changed_cells"] < 5:
        return None
    if stats["non_black"] >= inp.size * 0.45:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow every horizontal and vertical segment outward until races at crossings determine who survives.

    Input structure:
    - The background is black.
    - Several non-black, one-cell-thick line segments appear; each segment is monochrome,
      contiguous, and either horizontal or vertical.
    - Some segments are horizontal, some are vertical, and their future extensions can cross.

    Rule:
    - Every segment grows from both endpoints simultaneously, one cell per step, along its own axis.
    - If an endpoint would enter an already non-black cell, that endpoint stops without repainting.
    - If two endpoints enter the same background cell in the same step, paint that cell yellow and stop them both.
    - An endpoint that reaches the grid border after moving also stops; keep stepping until all endpoints stop.

    Key insight:
    - The output is the result of a synchronous race, not a full row/column extension: each crossing is decided by
      whichever endpoint arrives first, while exact ties leave a yellow collision mark.
    """
    return _simulate_growth(np.asarray(grid, dtype=int))[0]


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    branch = BRANCH_SEQUENCE[_GEN_CALL_COUNT % len(BRANCH_SEQUENCE)]
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(branch)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError(f"failed to generate a valid input for branch={branch}")
