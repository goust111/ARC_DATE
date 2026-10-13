from __future__ import annotations

from collections import Counter
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8


MARKER_COLORS = (Color.PINK, Color.ORANGE, Color.TEAL)
RANK_COLORS = (Color.BLUE, Color.YELLOW, Color.RED)
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
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
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_nonmonotone(values: list[int]) -> bool:
    return values != sorted(values) and values != sorted(values, reverse=True)


def _sample_spaced_rows(height: int, count: int, min_gap: int) -> list[int]:
    rows = list(range(height))
    valid = [
        combo
        for combo in __import__("itertools").combinations(rows, count)
        if all(combo[i + 1] - combo[i] >= min_gap for i in range(count - 1))
    ]
    if not valid:
        raise ValueError("no valid row placement")
    return list(random.choice(valid))


def _parse_instance(grid: np.ndarray) -> tuple[int, list[dict], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)

    comps = _find_components(g != bg)
    bars: list[dict] = []
    targets: dict[int, int] = {}

    for coords in comps:
        r0, c0, r1, c1 = _bbox_coords(coords)
        colors = {int(g[r, c]) for r, c in coords}
        if coords.shape[0] == 1:
            r, c = map(int, coords[0])
            assert c == w - 1, "singleton foreground pixels must be right-edge targets"
            color = int(g[r, c])
            assert color in MARKER_COLORS, "unexpected target color"
            assert color not in targets, "target colors must be unique"
            targets[color] = r
            continue

        assert colors & set(MARKER_COLORS), "every bar component must include one connector marker"
        marker_cells = [(int(r), int(c)) for r, c in coords if int(g[r, c]) in MARKER_COLORS]
        grey_cells = [(int(r), int(c)) for r, c in coords if int(g[r, c]) == Color.GREY]
        assert len(marker_cells) == 1, "each bar must have exactly one colored marker"
        assert grey_cells, "bar body must contain grey cells"

        marker_r, marker_c = marker_cells[0]
        marker_color = int(g[marker_r, marker_c])
        grey_rows = {r for r, _c in grey_cells}
        assert len(grey_rows) == 1, "bar body must be horizontal"
        body_r = next(iter(grey_rows))
        assert marker_r == body_r, "marker must be aligned with its bar"

        grey_cols = sorted(c for _r, c in grey_cells)
        assert grey_cols == list(range(grey_cols[0], grey_cols[-1] + 1)), "bar body must be contiguous"
        assert marker_c == grey_cols[0] - 1, "marker must touch the left end of the bar"
        assert c0 == marker_c and c1 == grey_cols[-1], "component bbox must span marker and full bar"

        bars.append(
            {
                "row": body_r,
                "length": len(grey_cells),
                "marker_color": marker_color,
                "body_cols": grey_cols,
            }
        )

    assert len(bars) == 3, "expected exactly three marked bars"
    assert len(targets) == 3, "expected exactly three right-edge targets"
    assert {bar["marker_color"] for bar in bars} == set(MARKER_COLORS) == set(targets), "marker/target colors must match one-to-one"

    lengths = [int(bar["length"]) for bar in bars]
    assert len(set(lengths)) == 3, "bar lengths must be distinct"
    return bg, bars, targets


def _expected_output(bg: int, bars: list[dict], targets: dict[int, int]) -> np.ndarray:
    ranked = sorted(bars, key=lambda bar: int(bar["length"]), reverse=True)
    color_by_marker = {
        int(bar["marker_color"]): rank_color
        for bar, rank_color in zip(ranked, RANK_COLORS)
    }

    min_row = min(int(row) for row in targets.values())
    max_row = max(int(row) for row in targets.values())
    width = max(int(bar["length"]) for bar in bars)
    out = np.full((max_row - min_row + 1, width), bg, dtype=int)

    for bar in bars:
        marker_color = int(bar["marker_color"])
        row = int(targets[marker_color]) - min_row
        length = int(bar["length"])
        out[row, :length] = int(color_by_marker[marker_color])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor the three marked gray bars by length rank, dock them to same-colored right-edge targets, and crop the assembled result.

    Input structure:
    - The background is black, with exactly three horizontal gray one-cell-thick bars in the left part of the grid.
    - Each bar has a single colored marker immediately to the left of its leftmost gray cell; the three marker colors are magenta, orange, and teal.
    - The rightmost column contains exactly three single-pixel targets using the same three colors in an arbitrary top-to-bottom order.

    Rule:
    - Rank the three gray bars by length: longest becomes blue, middle becomes yellow, shortest becomes red.
    - Match each bar to the right-edge target that has the same marker color.
    - Translate the whole marked bar so the marker would sit immediately left of its matched target, then erase the marker and the target pixel.
    - Return the tight crop containing only the translated recolored bars on the background.

    Key insight:
    - Length rank determines each bar's final color, but marker-color docking determines its final row, so the output order cannot be inferred from length alone.
    """
    bg, bars, targets = _parse_instance(np.asarray(grid, dtype=int))
    return _expected_output(bg, bars, targets)


def _build_grid(
    height: int,
    width: int,
    bars_by_row: list[tuple[int, int, int, int]],
    targets_by_row: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((height, width), Color.BLACK, dtype=int)
    for row, start_col, length, marker_color in bars_by_row:
        grid[row, start_col - 1] = int(marker_color)
        grid[row, start_col : start_col + length] = Color.GREY
    for row, color in targets_by_row:
        grid[row, width - 1] = int(color)
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        bg, bars, targets = _parse_instance(grid)
    except AssertionError:
        return False

    bars_by_input_row = sorted(bars, key=lambda bar: int(bar["row"]))
    input_lengths = [int(bar["length"]) for bar in bars_by_input_row]
    if not _is_nonmonotone(input_lengths):
        return False

    target_color_order = [color for color, _row in sorted(targets.items(), key=lambda item: int(item[1]))]
    input_color_order = [int(bar["marker_color"]) for bar in bars_by_input_row]
    if target_color_order == input_color_order:
        return False

    length_by_color = {int(bar["marker_color"]): int(bar["length"]) for bar in bars}
    target_lengths = [length_by_color[color] for color in target_color_order]
    if not _is_nonmonotone(target_lengths):
        return False

    expected = _expected_output(bg, bars, targets)
    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    colors = {int(v) for v in np.unique(solved) if int(v) != bg}
    if colors != set(RANK_COLORS):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(8, 13)
    width = random.randint(20, 28)
    max_length = min(9, width // 2 - 2)
    if max_length < 5:
        return None

    lengths = random.sample(list(range(3, max_length + 1)), 3)
    input_rows = _sample_spaced_rows(height, 3, min_gap=2)
    target_rows = _sample_spaced_rows(height, 3, min_gap=2)

    length_order = lengths[:]
    random.shuffle(length_order)
    if not _is_nonmonotone(length_order):
        return None

    marker_order = list(MARKER_COLORS)
    random.shuffle(marker_order)
    bars_by_row = []
    for row, length, marker_color in zip(input_rows, length_order, marker_order):
        max_start = min(width // 2, width - length - 5)
        if max_start < 2:
            return None
        start_col = random.randint(2, max_start)
        bars_by_row.append((row, start_col, length, marker_color))

    input_color_order = [marker_color for _row, _start_col, _length, marker_color in bars_by_row]
    length_by_color = {marker_color: length for _row, _start_col, length, marker_color in bars_by_row}

    target_color_order = list(MARKER_COLORS)
    random.shuffle(target_color_order)
    if target_color_order == input_color_order:
        return None
    if not _is_nonmonotone([length_by_color[color] for color in target_color_order]):
        return None

    targets_by_row = list(zip(target_rows, target_color_order))
    grid = _build_grid(height, width, bars_by_row, targets_by_row)
    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
