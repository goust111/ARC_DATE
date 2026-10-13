from __future__ import annotations

from collections import Counter

import numpy as np


BACKGROUND = 0
COLOR_PRIORITY = (1, 2, 3, 4, 5, 6, 7, 8, 9)
COLOR_RANK = {color: rank for rank, color in enumerate(COLOR_PRIORITY)}
RNG = np.random.default_rng()

GEN_RETRIES = 400
ASSIGNMENT_RETRIES = 200


def _most_common_color(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_edge(coords: np.ndarray, height: int, width: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == height - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == width - 1)
    )


def _choose_fill_color(border_colors: list[int] | tuple[int, int, int, int]) -> int:
    counts = Counter(int(color) for color in border_colors)
    min_count = min(counts.values())
    candidates = [color for color, count in counts.items() if count == min_count]
    return min(candidates, key=lambda color: COLOR_RANK[int(color)])


def _infer_line_colors(
    grid: np.ndarray,
    bg: int,
) -> tuple[list[int], list[int], dict[int, int], dict[int, int]]:
    height, width = grid.shape
    horizontal_rows = [int(r) for r in np.flatnonzero(np.all(grid != bg, axis=1))]
    vertical_cols = [int(c) for c in np.flatnonzero(np.all(grid != bg, axis=0))]
    assert len(horizontal_rows) >= 2 and len(vertical_cols) >= 2, "expected full-span horizontal and vertical lines"

    vertical_set = set(vertical_cols)
    horizontal_set = set(horizontal_rows)
    non_vertical_cols = np.array([c for c in range(width) if c not in vertical_set], dtype=int)
    non_horizontal_rows = np.array([r for r in range(height) if r not in horizontal_set], dtype=int)

    row_colors: dict[int, int] = {}
    for row in horizontal_rows:
        samples = grid[row, non_vertical_cols] if non_vertical_cols.size else grid[row, :]
        row_colors[row] = _most_common_color(samples)

    col_colors: dict[int, int] = {}
    for col in vertical_cols:
        samples = grid[non_horizontal_rows, col] if non_horizontal_rows.size else grid[:, col]
        col_colors[col] = _most_common_color(samples)

    return horizontal_rows, vertical_cols, row_colors, col_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed rectangle with the least frequent color among its four boundary lines.

    Input structure:
    - The most common color is the background.
    - Several one-pixel-wide horizontal and vertical colored lines span the full grid and cross each other.
    - The lines split the background into rectangular enclosed regions; intersections may visibly show only one of the two crossing colors.

    Rule:
    - Find every 4-connected background component that does not touch the grid edge.
    - Recover the color of each full-span horizontal and vertical line from its non-intersection cells.
    - For each enclosed background rectangle, read the colors of its top, bottom, left, and right boundary lines.
    - Fill the rectangle with the color whose boundary count is smallest; ties are broken by the visible priority order blue, red, green, yellow, gray, magenta, orange, azure, maroon.

    Key insight:
    - Even when intersections are overwritten, each real line still owns an entire row or column of non-background pixels, so the hidden boundary colors can be reconstructed and then applied region by region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape

    bg = _most_common_color(g.ravel())
    horizontal_rows, vertical_cols, row_colors, col_colors = _infer_line_colors(g, bg)
    _ = horizontal_rows, vertical_cols

    out = g.copy()
    for coords in _components4(g == bg):
        if _touches_edge(coords, height, width):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == bg), "enclosed components must be rectangular"
        fill_color = _choose_fill_color(
            [row_colors[r0 - 1], row_colors[r1 + 1], col_colors[c0 - 1], col_colors[c1 + 1]]
        )
        out[r0 : r1 + 1, c0 : c1 + 1] = fill_color

    return out


def _sample_axis(line_count: int) -> tuple[int, list[int]]:
    min_dim = max(14, 2 * line_count - 1)
    dim = int(RNG.integers(min_dim, 31))
    extras = dim - (2 * line_count - 1)
    buckets = np.zeros(line_count + 1, dtype=int)
    for _ in range(extras):
        buckets[int(RNG.integers(0, line_count + 1))] += 1

    positions = [int(buckets[0])]
    for extra_gap in buckets[1:-1]:
        positions.append(positions[-1] + 2 + int(extra_gap))

    assert len(positions) == line_count
    assert positions[-1] < dim
    return dim, positions


def _sample_color_counts(color_count: int, total_lines: int) -> list[int]:
    extras = total_lines - 2 * color_count
    counts = [2] * color_count
    while extras > 0:
        idx = int(RNG.integers(0, color_count))
        if counts[idx] < 4:
            counts[idx] += 1
            extras -= 1
    return counts


def _draw_lines(
    height: int,
    width: int,
    row_positions: list[int],
    row_colors: list[int],
    col_positions: list[int],
    col_colors: list[int],
) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    ops = [("h", row, color) for row, color in zip(row_positions, row_colors)] + [
        ("v", col, color) for col, color in zip(col_positions, col_colors)
    ]
    RNG.shuffle(ops)

    for kind, index, color in ops:
        if kind == "h":
            grid[int(index), :] = int(color)
        else:
            grid[:, int(index)] = int(color)
    return grid


def _expected_output(
    grid: np.ndarray,
    row_positions: list[int],
    row_colors: list[int],
    col_positions: list[int],
    col_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    for i in range(len(row_positions) - 1):
        for j in range(len(col_positions) - 1):
            r0, r1 = row_positions[i] + 1, row_positions[i + 1]
            c0, c1 = col_positions[j] + 1, col_positions[j + 1]
            fill_color = _choose_fill_color(
                [row_colors[i], row_colors[i + 1], col_colors[j], col_colors[j + 1]]
            )
            out[r0:r1, c0:c1] = fill_color
    return out


def _branch_profile(row_colors: list[int], col_colors: list[int]) -> tuple[bool, bool, int]:
    saw_unique = False
    saw_tie = False
    fill_colors: set[int] = set()
    for i in range(len(row_colors) - 1):
        for j in range(len(col_colors) - 1):
            border_colors = [row_colors[i], row_colors[i + 1], col_colors[j], col_colors[j + 1]]
            counts = Counter(border_colors)
            min_count = min(counts.values())
            rare_count = sum(count == min_count for count in counts.values())
            saw_unique |= rare_count == 1
            saw_tie |= rare_count > 1
            fill_colors.add(_choose_fill_color(border_colors))
    return saw_unique, saw_tie, len(fill_colors)


def _sample_line_colors(horizontal_count: int, vertical_count: int) -> tuple[list[int], list[int]] | None:
    total_lines = horizontal_count + vertical_count
    color_count = int(RNG.choice((3, 4)))
    if not (2 * color_count <= total_lines <= 4 * color_count):
        return None

    chosen_colors = [int(c) for c in RNG.choice(COLOR_PRIORITY, size=color_count, replace=False)]
    counts = _sample_color_counts(color_count, total_lines)

    for _ in range(ASSIGNMENT_RETRIES):
        slots: list[int] = []
        for color, count in zip(chosen_colors, counts):
            slots.extend([color] * count)
        RNG.shuffle(slots)

        row_colors = [int(color) for color in slots[:horizontal_count]]
        col_colors = [int(color) for color in slots[horizontal_count:]]
        if len(set(row_colors)) < 2 or len(set(col_colors)) < 2:
            continue

        saw_unique, saw_tie, fill_variety = _branch_profile(row_colors, col_colors)
        if saw_unique and saw_tie and fill_variety >= 2:
            return row_colors, col_colors

    return None


def _gen_attempt() -> np.ndarray | None:
    horizontal_count = int(RNG.integers(3, 6))
    vertical_count = int(RNG.integers(3, 6))
    sampled = _sample_line_colors(horizontal_count, vertical_count)
    if sampled is None:
        return None
    row_colors, col_colors = sampled

    height, row_positions = _sample_axis(horizontal_count)
    width, col_positions = _sample_axis(vertical_count)
    grid = _draw_lines(height, width, row_positions, row_colors, col_positions, col_colors)

    if _most_common_color(grid.ravel()) != BACKGROUND:
        return None

    expected = _expected_output(grid, row_positions, row_colors, col_positions, col_colors)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
