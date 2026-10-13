import random
from collections import deque

import numpy as np


BLACK = 0
GRAY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
GEN_ATTEMPTS = 400


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == GRAY))]
    assert len(rows) == 1, "expected exactly one full gray separator row"
    sep = int(rows[0])
    assert 0 < sep < grid.shape[0] - 1, "separator row must have space above and below"
    return sep


def _color_counts_above_separator(grid: np.ndarray, sep: int) -> dict[int, int]:
    upper = grid[:sep, :]
    counts: dict[int, int] = {}
    for color in np.unique(upper):
        value = int(color)
        if value in (BLACK, GRAY):
            continue
        counts[value] = int(np.count_nonzero(upper == value))
    assert 3 <= len(counts) <= 4, "expected 3 or 4 foreground colors above the separator"
    assert len(set(counts.values())) == len(counts), "foreground color counts must be distinct"
    return counts


def _compact_strip(counts: dict[int, int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    ordered = sorted(counts.items(), key=lambda item: item[1])
    values: list[int] = []
    for idx, (color, count) in enumerate(ordered):
        values.extend([int(color)] * int(count))
        if idx != len(ordered) - 1:
            values.append(BLACK)
    strip = np.array(values, dtype=int)
    assert strip.size > 0
    return strip, ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten each color's scattered pixels into a centered sorted strip and repeat that strip below the gray separator.

    Input structure:
    - The canvas has a black background and exactly one full gray separator row.
    - Above the separator are 3 or 4 distinct non-black, non-gray foreground colors;
      each color may form one irregular 4-connected blob or several separated blobs.
    - The row immediately above the separator is the display row for the straightened strip,
      and rows below the separator start empty.

    Rule:
    - Count the total number of pixels of each foreground color above the separator.
    - Turn each color into a horizontal segment whose length equals that count, sort the
      segments from shortest to longest, and concatenate them with one black cell between
      adjacent segments to make a compact strip.
    - Erase every pixel above the separator except the row immediately above it, and place
      the compact strip centered on that row.
    - Fill every row below the separator by repeating the compact strip from its first cell
      across the full width, keeping the strip's internal black gap cells black.

    Key insight:
    - Only each color's pixel count matters: it determines both the segment length and the
      left-to-right order in the strip, and that same strip becomes the repeating fill template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)

    sep = _find_separator_row(g)
    counts = _color_counts_above_separator(g, sep)
    strip, _ordered = _compact_strip(counts)
    assert strip.size <= width, "straightened strip must fit in the grid width"

    out = np.zeros((height, width), dtype=int)
    out[sep, :] = GRAY

    start = (width - int(strip.size)) // 2
    out[sep - 1, start : start + int(strip.size)] = strip

    repeated = strip[np.arange(width) % int(strip.size)]
    for row in range(sep + 1, height):
        out[row, :] = repeated
    return out


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            count += 1
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(int(r), int(c))])
            while queue:
                cr, cc = queue.popleft()
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                        if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                            seen[nr, nc] = True
                            queue.append((nr, nc))
    return count


def _random_connected_shape(size: int) -> list[tuple[int, int]]:
    if size == 1:
        return [(0, 0)]

    for _ in range(120):
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        while len(cells) < size:
            nr, nc = random.choice(tuple(frontier))
            frontier.remove((nr, nc))
            cells.add((nr, nc))
            for dr, dc in DIRS4:
                nxt = (nr + dr, nc + dc)
                if nxt not in cells:
                    frontier.add(nxt)

        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        norm = sorted((r - min_r, c - min_c) for r, c in cells)
        rows = {r for r, _ in norm}
        cols = {c for _, c in norm}
        if size >= 3 and (len(rows) == 1 or len(cols) == 1):
            continue
        return norm

    side = (size + 1) // 2
    coords = []
    for idx in range(size):
        coords.append((idx // side, idx % side))
    return coords


def _component_sizes(total: int, force_split: bool) -> list[int]:
    if not force_split:
        return [total]
    if total <= 4:
        return [2, total - 2]
    left = random.randint(2, total - 2)
    return [left, total - left]


def _place_shape(
    grid: np.ndarray,
    occupied: np.ndarray,
    shape: list[tuple[int, int]],
    color: int,
) -> list[tuple[int, int]] | None:
    upper_h, width = occupied.shape
    shape_h = 1 + max(r for r, _ in shape)
    shape_w = 1 + max(c for _, c in shape)
    if shape_h > upper_h or shape_w > width:
        return None

    for _ in range(180):
        r0 = random.randint(0, upper_h - shape_h)
        c0 = random.randint(0, width - shape_w)
        placed = [(r0 + r, c0 + c) for r, c in shape]

        ok = True
        for r, c in placed:
            if bool(occupied[r, c]):
                ok = False
                break
            for dr, dc in DIRS8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < upper_h and 0 <= nc < width and bool(occupied[nr, nc]):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        for r, c in placed:
            occupied[r, c] = True
            grid[r, c] = int(color)
        return placed
    return None


def _centroid_order(grid: np.ndarray, colors: list[int], top_rows: int) -> list[int]:
    order = []
    for color in colors:
        coords = np.argwhere(grid[:top_rows, :] == color)
        assert len(coords) > 0
        mean_c = float(coords[:, 1].mean())
        min_c = int(coords[:, 1].min())
        min_r = int(coords[:, 0].min())
        order.append((mean_c, min_c, min_r, int(color)))
    order.sort()
    return [color for _mean_c, _min_c, _min_r, color in order]


def _scan_order(grid: np.ndarray, top_rows: int) -> list[int]:
    seen: set[int] = set()
    order: list[int] = []
    for r in range(top_rows):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color in (BLACK, GRAY) or color in seen:
                continue
            seen.add(color)
            order.append(color)
    return order


def _relocate_same_color_pixel(grid: np.ndarray) -> np.ndarray:
    moved = grid.copy()
    sep = _find_separator_row(moved)
    draw_rows = sep - 1
    colors = [int(color) for color in np.unique(moved[:sep, :]) if int(color) not in (BLACK, GRAY)]
    random.shuffle(colors)

    for color in colors:
        source = np.argwhere(moved[:draw_rows, :] == color)
        empty = np.argwhere(moved[:draw_rows, :] == BLACK)
        if len(source) == 0 or len(empty) == 0:
            continue
        src_r, src_c = map(int, random.choice(source))
        dst_r, dst_c = map(int, random.choice(empty))
        moved[src_r, src_c] = BLACK
        moved[dst_r, dst_c] = color
        return moved
    raise AssertionError("expected at least one movable foreground pixel")


def _validate_output(grid: np.ndarray, out: np.ndarray) -> None:
    sep = _find_separator_row(grid)
    counts = _color_counts_above_separator(grid, sep)
    strip, ordered = _compact_strip(counts)
    height, width = grid.shape
    start = (width - int(strip.size)) // 2
    repeated = strip[np.arange(width) % int(strip.size)]

    assert np.all(out[: sep - 1, :] == BLACK)
    assert np.all(out[sep, :] == GRAY)

    expected_seq = np.zeros(width, dtype=int)
    expected_seq[start : start + int(strip.size)] = strip
    assert np.array_equal(out[sep - 1, :], expected_seq)

    for row in range(sep + 1, height):
        assert np.array_equal(out[row, :], repeated)

    count_order = [color for color, _count in ordered]
    centroid_order = _centroid_order(grid, list(counts), sep - 1)
    scan_order = _scan_order(grid, sep - 1)
    assert count_order != sorted(count_order), "reject numeric-color-order shortcut"
    assert count_order != centroid_order, "reject left-to-right shortcut"
    assert count_order != scan_order, "reject row-major first-seen shortcut"

    component_counts = [_component_count(grid[: sep - 1, :] == color) for color in counts]
    assert any(value > 1 for value in component_counts), "need at least one disconnected color group"
    assert any(value == 1 for value in component_counts), "need at least one single connected color group"

    moved = _relocate_same_color_pixel(grid)
    assert np.array_equal(main(moved), out), "output should depend on color counts, not exact shape"


def _generate_instance() -> np.ndarray | None:
    n_colors = random.choice([3, 4])
    available_counts = list(range(3, 8))
    sampled_counts = random.sample(available_counts, n_colors)
    pattern_len = sum(sampled_counts) + (n_colors - 1)

    max_extra_margin = min(3, (30 - pattern_len) // 2)
    if max_extra_margin < 0:
        return None
    if max_extra_margin == 0:
        extra_margin = 0
    elif random.random() < 0.15:
        extra_margin = 0
    else:
        extra_margin = random.randint(1, max_extra_margin)

    width = pattern_len + 2 * extra_margin
    draw_rows = random.randint(5, 8)
    if sum(sampled_counts) > int(draw_rows * width * 0.42):
        return None
    bottom_rows = random.randint(3, 6)
    height = draw_rows + 2 + bottom_rows
    if height > 30:
        return None

    colors = random.sample(PALETTE, n_colors)
    random.shuffle(sampled_counts)
    color_to_count = {color: count for color, count in zip(colors, sampled_counts)}

    count_order = [color for color, _count in sorted(color_to_count.items(), key=lambda item: item[1])]
    if count_order == sorted(count_order):
        return None

    split_candidates = [color for color, count in color_to_count.items() if count >= 4]
    if not split_candidates:
        return None
    split_color = random.choice(split_candidates)

    grid = np.zeros((height, width), dtype=int)
    sep = draw_rows + 1
    grid[sep, :] = GRAY
    occupied = np.zeros((draw_rows, width), dtype=bool)

    placement_order = sorted(color_to_count, key=lambda color: color_to_count[color], reverse=True)
    for color in placement_order:
        force_split = color == split_color
        sizes = _component_sizes(color_to_count[color], force_split)
        sizes.sort(reverse=True)
        for size in sizes:
            shape = _random_connected_shape(size)
            placed = _place_shape(grid, occupied, shape, color)
            if placed is None:
                return None

    if not np.all(grid[sep + 1 :, :] == BLACK):
        return None

    try:
        out = main(grid)
        _validate_output(grid, out)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
