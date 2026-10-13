from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
RED = 2
MARKER_COLORS = (1, 4, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _dominant_foreground_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in np.asarray(grid).flat if int(v) != BG)
    assert counts, "expected foreground pixels"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected unique dominant foreground color"
    return int(ranked[0][0])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _mirror_complete(grid: np.ndarray, color: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    axis = w // 2
    out = g.copy()
    for r, c in np.argwhere(g == color):
        mc = 2 * axis - int(c)
        if 0 <= mc < w and out[int(r), mc] in (BG, color):
            out[int(r), mc] = color
    return out


def _marker_pairs(grid: np.ndarray, red_color: int) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    pairs: list[tuple[int, int, int, int]] = []
    for color in sorted(set(int(v) for v in np.unique(g)) - {BG, int(red_color)}):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), f"color {color} must appear exactly twice"
        rows = {int(v) for v in coords[:, 0]}
        assert len(rows) == 1, f"color {color} markers must share a row"
        row = int(coords[0, 0])
        c1, c2 = sorted(int(v) for v in coords[:, 1])
        pairs.append((color, row, c1, c2))
    pairs.sort(key=lambda item: (item[1], item[2], item[0]))
    return pairs


def _bridged_pairs(base_grid: np.ndarray, completed_grid: np.ndarray, red_color: int) -> list[tuple[int, int, int, int, bool]]:
    pairs = _marker_pairs(base_grid, red_color)
    result: list[tuple[int, int, int, int, bool]] = []
    for color, row, c1, c2 in pairs:
        seg = completed_grid[row, c1 : c2 + 1]
        bridged = bool(np.all((seg == red_color) | (seg == color)))
        result.append((color, row, c1, c2, bridged))
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror-complete the dominant red object, then color only marker pairs whose gap becomes fully bridged.

    Input structure:
    - The grid has black background, one uniquely dominant red foreground object, and 3-5 marker pairs.
    - Each marker pair consists of exactly two same-colored pixels on the same row, with no overlap with the red object.
    - The red object is 4-connected and horizontally asymmetric, biased to one side of the vertical center axis.

    Rule:
    - Identify the uniquely most frequent foreground color as the red object.
    - Reflect every red pixel across the grid's vertical centerline, adding red at the mirror cell unless that cell already holds a marker.
    - For each same-colored marker pair, inspect the entire row segment between the two markers after completion.
      If every cell in that closed segment is either red or that pair's marker color, recolor the whole segment to the marker color.
      Otherwise leave the pair unchanged.

    Key insight:
    - Symmetry completion is not the final goal; it creates red bridge segments that determine which marker pairs are eligible to become horizontal colored lines.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    red_color = _dominant_foreground_color(g)
    completed = _mirror_complete(g, red_color)
    out = completed.copy()

    for color, row, c1, c2, bridged in _bridged_pairs(g, completed, red_color):
        if bridged:
            out[row, c1 : c2 + 1] = color
    return out


def _random_half_widths(active_h: int, axis: int) -> list[int]:
    widths = [random.randint(1, axis - 1)]
    for _ in range(active_h - 1):
        step = random.choice((-2, -1, 0, 1, 2))
        widths.append(max(1, min(axis - 1, widths[-1] + step)))
    if all(w == axis - 1 for w in widths):
        widths[random.randrange(active_h)] = axis - 2
    return widths


def _build_symmetric_red_mask(h: int, w: int, active_rows: list[int], widths: list[int]) -> np.ndarray:
    axis = w // 2
    mask = np.zeros((h, w), dtype=bool)
    for row, width in zip(active_rows, widths):
        mask[row, axis - width : axis + width + 1] = True
    return mask


def _place_marker_pairs(
    output_grid: np.ndarray,
    active_rows: list[int],
    widths: list[int],
    pair_count: int,
    unbridged_count: int,
) -> None:
    h, w = output_grid.shape
    axis = w // 2
    row_to_width = dict(zip(active_rows, widths))
    narrow_rows = [row for row in active_rows if row_to_width[row] <= axis - 2]
    assert len(narrow_rows) >= unbridged_count

    unbridged_rows = random.sample(narrow_rows, unbridged_count)
    remaining_rows = [row for row in active_rows if row not in unbridged_rows]
    bridged_rows = random.sample(remaining_rows, pair_count - unbridged_count)
    layouts = [("bridged", row) for row in bridged_rows] + [("unbridged", row) for row in unbridged_rows]
    random.shuffle(layouts)
    colors = random.sample(MARKER_COLORS, pair_count)

    for color, (kind, row) in zip(colors, layouts):
        width = row_to_width[row]
        left_red = axis - width
        right_red = axis + width
        if kind == "bridged":
            left_marker = left_red - 1
            right_marker = right_red + 1
        else:
            options = []
            if left_red >= 2:
                options.append((left_red - 2, right_red + 1))
            if right_red <= w - 3:
                options.append((left_red - 1, right_red + 2))
            if left_red >= 2 and right_red <= w - 3:
                options.append((left_red - 2, right_red + 2))
            left_marker, right_marker = random.choice(options)

        assert 0 <= left_marker < right_marker < w
        assert output_grid[row, left_marker] == BG and output_grid[row, right_marker] == BG
        output_grid[row, left_marker] = color
        output_grid[row, right_marker] = color


def _derive_input_from_output(output_grid: np.ndarray) -> np.ndarray:
    inp = output_grid.copy()
    _, w = inp.shape
    axis = w // 2
    inp[(inp == RED) & (np.arange(w)[None, :] > axis)] = BG
    if random.random() < 0.5:
        inp = np.fliplr(inp).copy()
    return inp


def _validate_generated(input_grid: np.ndarray, expected_output: np.ndarray) -> bool:
    if input_grid.ndim != 2 or expected_output.ndim != 2:
        return False
    if input_grid.shape != expected_output.shape:
        return False
    if max(input_grid.shape) > 30:
        return False
    if not np.array_equal(main(input_grid), expected_output):
        return False

    red_color = _dominant_foreground_color(input_grid)
    if red_color != RED:
        return False

    red_mask = input_grid == RED
    comps = _components4(red_mask)
    if len(comps) != 1:
        return False

    if np.array_equal(red_mask, np.fliplr(red_mask)):
        return False

    completed = _mirror_complete(input_grid, RED)
    before_flags = {item[:4]: item[4] for item in _bridged_pairs(input_grid, input_grid, RED)}
    after_flags = {item[:4]: item[4] for item in _bridged_pairs(input_grid, completed, RED)}
    if len(after_flags) < 3 or len(after_flags) > 5:
        return False
    if not any(after_flags[key] and not before_flags[key] for key in after_flags):
        return False
    if not any(not after_flags[key] for key in after_flags):
        return False

    if np.array_equal(completed, expected_output):
        return False

    for color, row, c1, c2 in _marker_pairs(input_grid, RED):
        segment = input_grid[row, c1 + 1 : c2]
        if np.any(np.isin(segment, MARKER_COLORS)):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.choice((11, 13, 15))
    w = random.choice((11, 13, 15, 17))
    axis = w // 2

    active_h = random.randint(5, h - 2)
    top = random.randint(1, h - active_h - 1)
    active_rows = list(range(top, top + active_h))
    widths = _random_half_widths(active_h, axis)
    narrow_count = sum(width <= axis - 2 for width in widths)

    pair_count = random.randint(3, min(5, active_h))
    max_unbridged = min(pair_count - 1, narrow_count)
    if max_unbridged < 1:
        return None
    unbridged_count = random.randint(1, max_unbridged)

    completed_grid = np.zeros((h, w), dtype=int)
    completed_grid[_build_symmetric_red_mask(h, w, active_rows, widths)] = RED
    _place_marker_pairs(completed_grid, active_rows, widths, pair_count, unbridged_count)

    input_grid = _derive_input_from_output(completed_grid)
    expected_output = main(input_grid)
    if not _validate_generated(input_grid, expected_output):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
