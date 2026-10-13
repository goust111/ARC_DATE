from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
RETRY_LIMIT = 400
QUAD_HEIGHT_RANGE = (7, 10)
QUAD_WIDTH_RANGE = (7, 10)
UNIQUE_COLOR_PATTERNS = (
    (1, 1, 1, 1),
    (2, 1, 1, 1),
    (2, 2, 1, 1),
    (3, 1, 1, 1),
)
SHARED_SUBSET_OPTIONS = {
    1: (
        ((0, 1, 2, 3),),
    ),
    2: (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
        ((0, 1, 2), (1, 2, 3)),
        ((0, 1, 3), (0, 2, 3)),
        ((0, 1, 2, 3), (0, 2)),
    ),
    3: (
        ((0, 1), (1, 2), (2, 3)),
        ((0, 1, 2), (0, 2, 3), (1, 3)),
        ((0, 1, 2, 3), (0, 1), (2, 3)),
        ((0, 1), (0, 2), (0, 3)),
    ),
}
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _find_separator_cross(g: np.ndarray) -> tuple[int, int, int]:
    h, w = g.shape
    solid_rows = [
        (r, int(g[r, 0]))
        for r in range(h)
        if bool(np.all(g[r, :] == int(g[r, 0])))
    ]
    solid_cols = [
        (c, int(g[0, c]))
        for c in range(w)
        if bool(np.all(g[:, c] == int(g[0, c])))
    ]

    candidates = [
        (r, c, row_color)
        for r, row_color in solid_rows
        for c, col_color in solid_cols
        if row_color == col_color and 2 * r + 1 == h and 2 * c + 1 == w
    ]
    assert len(candidates) == 1, "expected one centered separator cross"
    return candidates[0]


def _quadrant_slices(sep_row: int, sep_col: int, h: int, w: int) -> list[tuple[slice, slice]]:
    return [
        (slice(0, sep_row), slice(0, sep_col)),
        (slice(0, sep_row), slice(sep_col + 1, w)),
        (slice(sep_row + 1, h), slice(0, sep_col)),
        (slice(sep_row + 1, h), slice(sep_col + 1, w)),
    ]


def _background_color(g: np.ndarray, sep_row: int, sep_col: int) -> int:
    mask = np.ones_like(g, dtype=bool)
    mask[sep_row, :] = False
    mask[:, sep_col] = False
    counts = Counter(int(v) for v in g[mask].reshape(-1))
    assert counts, "expected non-separator cells"
    return int(counts.most_common(1)[0][0])


def _first_occurrence(quad: np.ndarray, color: int) -> tuple[int, int]:
    hits = np.argwhere(quad == color)
    assert hits.size > 0, "color must appear in its quadrant"
    return int(hits[0, 0]), int(hits[0, 1])


def _replacement_color(quad: np.ndarray, unique_colors: list[int]) -> int:
    counts = {int(color): int(np.count_nonzero(quad == int(color))) for color in unique_colors}
    max_count = max(counts.values())
    candidates = [color for color, count in counts.items() if count == max_count]
    if len(candidates) == 1:
        return int(candidates[0])
    return min(candidates, key=lambda color: _first_occurrence(quad, int(color)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each quadrant's shared-color pixels with that quadrant's strongest unique color.

    The grid is split by a centered solid cross into four equal quadrants. Foreground
    colors that appear in multiple quadrants are "shared"; foreground colors that
    appear in only one quadrant are "unique". In each quadrant, every shared-color
    pixel is recolored to that quadrant's most frequent unique color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 3, "expected a non-empty 2D grid"

    h, w = map(int, g.shape)
    sep_row, sep_col, separator = _find_separator_cross(g)
    bg = _background_color(g, sep_row, sep_col)
    quadrants = _quadrant_slices(sep_row, sep_col, h, w)

    color_presence = Counter()
    quadrant_color_sets: list[set[int]] = []
    for rows, cols in quadrants:
        quad = g[rows, cols]
        colors = {
            int(color)
            for color in np.unique(quad)
            if int(color) not in {bg, separator}
        }
        quadrant_color_sets.append(colors)
        color_presence.update(colors)

    out = g.copy()
    for (rows, cols), colors in zip(quadrants, quadrant_color_sets, strict=False):
        if not colors:
            continue
        unique_colors = [color for color in colors if color_presence[color] == 1]
        shared_colors = [color for color in colors if color_presence[color] >= 2]
        if not shared_colors:
            continue
        assert unique_colors, "each changed quadrant must have at least one unique color"

        patch = out[rows, cols]
        fill_color = _replacement_color(g[rows, cols], unique_colors)
        patch[np.isin(patch, shared_colors)] = int(fill_color)

    return out


def _sample_scattered_cells(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(80):
        shuffled = cells[:]
        random.shuffle(shuffled)
        chosen: list[tuple[int, int]] = []
        blocked: set[tuple[int, int]] = set()
        for r, c in shuffled:
            if (r, c) in blocked:
                continue
            chosen.append((r, c))
            if len(chosen) == count:
                return chosen
            blocked.add((r, c))
            for dr, dc in DIR4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    blocked.add((rr, cc))
    return None


def _unique_pixel_counts(color_count: int) -> list[int]:
    if color_count == 1:
        return [random.randint(3, 6)]
    winner = random.randint(4, 6)
    others = random.sample(range(1, winner), color_count - 1)
    counts = [winner] + others
    random.shuffle(counts)
    return counts


def _build_quadrant_specs(
    shared_subsets: tuple[tuple[int, ...], ...],
    unique_color_counts: list[int],
    palette: list[int],
) -> tuple[list[dict[int, int]], list[set[int]], list[set[int]]]:
    idx = 0
    specs = [dict() for _ in range(4)]
    unique_sets = [set() for _ in range(4)]
    shared_sets = [set() for _ in range(4)]

    for quad_idx, color_count in enumerate(unique_color_counts):
        colors = palette[idx : idx + color_count]
        idx += color_count
        for color, pixels in zip(colors, _unique_pixel_counts(color_count), strict=False):
            specs[quad_idx][int(color)] = int(pixels)
            unique_sets[quad_idx].add(int(color))

    shared_colors = palette[idx:]
    assert len(shared_colors) == len(shared_subsets)
    for color, subset in zip(shared_colors, shared_subsets, strict=False):
        for quad_idx in subset:
            specs[int(quad_idx)][int(color)] = random.randint(2, 4)
            shared_sets[int(quad_idx)].add(int(color))

    return specs, unique_sets, shared_sets


def _render_quadrant(
    spec: dict[int, int],
    unique_colors: set[int],
    shared_colors: set[int],
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    total_pixels = sum(spec.values())
    coords = _sample_scattered_cells(height, width, total_pixels)
    if coords is None:
        return None

    patch = np.full((height, width), BG, dtype=int)
    expected = patch.copy()
    replacement = _replacement_color_from_spec(spec, unique_colors)

    color_order = list(spec.items())
    random.shuffle(color_order)
    offset = 0
    for color, count in color_order:
        taken = coords[offset : offset + count]
        offset += count
        for r, c in taken:
            patch[r, c] = int(color)
            expected[r, c] = int(replacement if color in shared_colors else color)

    return patch, expected


def _replacement_color_from_spec(spec: dict[int, int], unique_colors: set[int]) -> int:
    unique_counts = {int(color): int(spec[int(color)]) for color in unique_colors}
    max_count = max(unique_counts.values())
    winners = [color for color, count in unique_counts.items() if count == max_count]
    assert len(winners) == 1, "generator keeps a unique dominant unique color"
    return int(winners[0])


def _expected_grid(
    quadrants: list[np.ndarray],
    quadrant_outputs: list[np.ndarray],
    sep_color: int,
) -> tuple[np.ndarray, np.ndarray]:
    qh, qw = quadrants[0].shape
    grid = np.full((2 * qh + 1, 2 * qw + 1), BG, dtype=int)
    expected = grid.copy()
    grid[qh, :] = sep_color
    grid[:, qw] = sep_color
    expected[qh, :] = sep_color
    expected[:, qw] = sep_color

    slices = _quadrant_slices(qh, qw, grid.shape[0], grid.shape[1])
    for (rows, cols), patch, out_patch in zip(slices, quadrants, quadrant_outputs, strict=False):
        grid[rows, cols] = patch
        expected[rows, cols] = out_patch

    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    qh = random.randint(*QUAD_HEIGHT_RANGE)
    qw = random.randint(*QUAD_WIDTH_RANGE)
    sep_color = random.randint(1, 9)

    unique_color_counts = list(random.choice(UNIQUE_COLOR_PATTERNS))
    random.shuffle(unique_color_counts)

    usable_palette = [color for color in range(10) if color not in {BG, sep_color}]
    max_shared = min(3, len(usable_palette) - sum(unique_color_counts))
    if max_shared < 1:
        return None

    shared_count = random.randint(1, max_shared)
    shared_subsets = random.choice(SHARED_SUBSET_OPTIONS[shared_count])
    total_colors = sum(unique_color_counts) + shared_count
    palette = random.sample(usable_palette, total_colors)

    specs, unique_sets, shared_sets = _build_quadrant_specs(shared_subsets, unique_color_counts, palette)
    quadrants: list[np.ndarray] = []
    quadrant_outputs: list[np.ndarray] = []
    for spec, unique_colors, shared_colors in zip(specs, unique_sets, shared_sets, strict=False):
        if not shared_colors:
            return None
        rendered = _render_quadrant(spec, unique_colors, shared_colors, qh, qw)
        if rendered is None:
            return None
        patch, out_patch = rendered
        quadrants.append(patch)
        quadrant_outputs.append(out_patch)

    grid, expected = _expected_grid(quadrants, quadrant_outputs, sep_color)

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {RETRY_LIMIT} attempts")
