from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
MARKER_COLORS = (2, 3, 4, 6, 7, 9)
GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as (N,2) coord arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_cups(grid: np.ndarray) -> list[list[dict[str, int]]]:
    """Detect the six blue upward-open cups and return them as two row groups."""
    g = np.asarray(grid, dtype=int)
    cups: list[dict[str, int]] = []

    for coords in _components(g == BLUE):
        r0, c0, r1, c1 = _bbox(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 4 or r0 == 0:
            continue

        view = g[r0 : r1 + 1, c0 : c1 + 1]
        expected_blue = np.zeros((h, w), dtype=bool)
        expected_blue[:, 0] = True
        expected_blue[:, -1] = True
        expected_blue[-1, :] = True
        if not np.array_equal(view == BLUE, expected_blue):
            continue

        inner = view[:-1, 1:-1]
        if inner.size == 0 or np.any(inner == BLUE):
            continue
        marker_mask = (inner != BLACK) & (inner != BLUE)
        if int(marker_mask.sum()) != 1:
            continue
        marker_color = int(inner[marker_mask][0])
        if marker_color not in MARKER_COLORS:
            continue
        if not np.all(inner[~marker_mask] == BLACK):
            continue

        cups.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "cap_r": int(r0 - 1),
                "color": int(marker_color),
            }
        )

    assert len(cups) == 6, "expected exactly six cups"
    cups.sort(key=lambda cup: (cup["r0"], cup["c0"]))
    rows = [sorted(cups[:3], key=lambda cup: cup["c0"]), sorted(cups[3:], key=lambda cup: cup["c0"])]
    assert max(cup["r1"] for cup in rows[0]) < min(cup["r0"] for cup in rows[1]), "cup rows must be separated"
    for row in rows:
        assert len({cup["cap_r"] for cup in row}) == 1, "all cups in a row must share the same cap row"
    return rows


def _fill_and_cap(work: np.ndarray, cup: dict[str, int]) -> None:
    color = int(cup["color"])
    r0, c0, r1, c1 = cup["r0"], cup["c0"], cup["r1"], cup["c1"]
    work[r0:r1, c0 + 1 : c1] = color
    work[cup["cap_r"], c0 + 1 : c1] = color


def _bridge_row(work: np.ndarray, row: list[dict[str, int]]) -> None:
    colors = [int(cup["color"]) for cup in row]
    counts = Counter(colors)
    shared = [color for color, count in counts.items() if count == 2]
    assert len(shared) == 1 and len(counts) == 2, "each row must contain one duplicated marker color"

    pair = [idx for idx, color in enumerate(colors) if color == shared[0]]
    assert pair[1] == pair[0] + 1, "the same-color pair must be adjacent in its row"

    left = row[pair[0]]
    right = row[pair[1]]
    assert left["cap_r"] == right["cap_r"]

    gap_r = int(left["cap_r"])
    gap_c0 = int(left["c1"] + 1)
    gap_c1 = int(right["c0"] - 1)
    assert gap_c0 <= gap_c1, "bridge gap must be non-empty"
    assert np.all(work[gap_r, gap_c0 : gap_c1 + 1] != BLUE), "bridge row must not cross blue outline pixels"
    work[gap_r, gap_c0 : gap_c1 + 1] = GREY


def _cup_bridged(work: np.ndarray, cup: dict[str, int]) -> bool:
    r = int(cup["cap_r"])
    left_touch = cup["c0"] > 0 and int(work[r, cup["c0"] - 1]) == GREY
    right_touch = cup["c1"] + 1 < work.shape[1] and int(work[r, cup["c1"] + 1]) == GREY
    return bool(left_touch or right_touch)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each cup from its marker, bridge the same-color pair in each row, then compress the cup array into a 2x3 summary where bridged cups become grey.

    Input structure:
    - The grid has a black background and exactly six blue cup outlines arranged as a 2x3 array.
    - Every cup is open at the top, closed on the sides and bottom, and contains exactly one non-black non-blue marker pixel.
    - In each row of three cups, exactly one adjacent pair shares the same marker color.

    Rule:
    - For each cup, fill its interior with its marker color and add a one-row cap of the same color immediately above the opening.
    - In each row, find the unique same-color cup pair and draw a grey one-cell-high bridge across the horizontal gap between them at cap height.
    - Compress the six cups to a 2x3 output grid in row-major cup order: a bridged cup becomes grey, otherwise it keeps its fill color.

    Key insight:
    - The marker color only determines the initial fill; the final summary depends on whether a cup becomes part of the grey bridge topology, which overrides its color in the compressed output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC grid"

    rows = _parse_cups(g)
    work = g.copy()
    for row in rows:
        for cup in row:
            _fill_and_cap(work, cup)
    for row in rows:
        _bridge_row(work, row)

    summary = np.zeros((2, 3), dtype=int)
    for r, row in enumerate(rows):
        for c, cup in enumerate(row):
            summary[r, c] = GREY if _cup_bridged(work, cup) else int(cup["color"])
    return summary


def _draw_cup(grid: np.ndarray, top: int, left: int, height: int, width: int, marker_color: int) -> None:
    grid[top : top + height, left] = BLUE
    grid[top : top + height, left + width - 1] = BLUE
    grid[top + height - 1, left : left + width] = BLUE

    marker_r = random.randint(top, top + height - 2)
    marker_c = random.randint(left + 1, left + width - 2)
    grid[marker_r, marker_c] = int(marker_color)


def _row_colors(pair_start: int, pair_color: int, lone_color: int) -> list[int]:
    colors = [None, None, None]
    colors[pair_start] = pair_color
    colors[pair_start + 1] = pair_color
    colors[2 if pair_start == 0 else 0] = lone_color
    return [int(color) for color in colors]


def _sample_color_rows() -> list[list[int]]:
    pair_starts = [random.randint(0, 1), random.randint(0, 1)]
    mode = random.choice(
        [
            "same_pair_color",
            "top_pair_is_bottom_lone",
            "bottom_pair_is_top_lone",
            "cross_repeat_lones",
            "independent",
        ]
    )

    top_pair = random.choice(MARKER_COLORS)
    if mode == "same_pair_color":
        bottom_pair = top_pair
    else:
        bottom_pair = random.choice([color for color in MARKER_COLORS if color != top_pair])

    top_lone_choices = [color for color in MARKER_COLORS if color != top_pair]
    bottom_lone_choices = [color for color in MARKER_COLORS if color != bottom_pair]

    if mode == "top_pair_is_bottom_lone":
        bottom_lone = top_pair
    else:
        bottom_lone = random.choice(bottom_lone_choices)

    if mode == "bottom_pair_is_top_lone":
        top_lone = bottom_pair
    elif mode == "cross_repeat_lones":
        common = random.choice([color for color in MARKER_COLORS if color not in (top_pair, bottom_pair)])
        top_lone = common
        bottom_lone = common
    else:
        top_lone = random.choice(top_lone_choices)

    return [
        _row_colors(pair_starts[0], top_pair, top_lone),
        _row_colors(pair_starts[1], bottom_pair, bottom_lone),
    ]


def _expected_summary(color_rows: list[list[int]]) -> np.ndarray:
    out = np.zeros((2, 3), dtype=int)
    for r, row in enumerate(color_rows):
        counts = Counter(row)
        shared = next(color for color, count in counts.items() if count == 2)
        for c, color in enumerate(row):
            out[r, c] = GREY if color == shared else int(color)
    return out


def _validate_generated(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        rows = _parse_cups(grid)
        actual = main(grid)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False

    if actual.shape != (2, 3):
        return False
    if int(np.sum(actual == GREY)) != 4:
        return False

    for row in rows:
        colors = [cup["color"] for cup in row]
        counts = Counter(colors)
        if sorted(counts.values()) != [1, 2]:
            return False
        pair = [idx for idx, color in enumerate(colors) if counts[color] == 2]
        if pair[1] != pair[0] + 1:
            return False
        left, right = row[pair[0]], row[pair[1]]
        bridge_band = grid[left["cap_r"], left["c1"] + 1 : right["c0"]]
        if bridge_band.size == 0 or not np.all(bridge_band == BLACK):
            return False
    return True


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    cup_height = random.randint(3, 4)
    widths = [random.randint(4, 5) for _ in range(3)]
    h_gaps = [random.randint(3, 5) for _ in range(2)]
    top_margin = random.randint(2, 4)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    v_gap = random.randint(3, 5)

    total_w = left_margin + sum(widths) + sum(h_gaps) + right_margin
    total_h = top_margin + cup_height + v_gap + cup_height + bottom_margin
    if total_w > 30 or total_h > 30:
        return None

    grid = np.zeros((total_h, total_w), dtype=int)
    color_rows = _sample_color_rows()
    expected = _expected_summary(color_rows)

    x_positions = [left_margin]
    for idx in range(2):
        x_positions.append(x_positions[-1] + widths[idx] + h_gaps[idx])

    y_positions = [top_margin, top_margin + cup_height + v_gap]
    for r in range(2):
        for c in range(3):
            _draw_cup(grid, y_positions[r], x_positions[c], cup_height, widths[c], color_rows[r][c])

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        attempt = _generate_attempt()
        if attempt is None:
            continue
        grid, expected = attempt
        if _validate_generated(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid cup-bridge-compress input")
