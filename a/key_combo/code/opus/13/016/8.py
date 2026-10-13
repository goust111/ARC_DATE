from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np


MARKER_COLOR = 4
BACKGROUND = 0
PATTERN_COLORS = np.array([1, 2, 3, 5, 6, 7, 8, 9], dtype=int)
NOISE_COLORS = np.array([1, 2, 3, 5, 5, 5, 6, 7, 8, 9], dtype=int)
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    components = []

    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _marker_pairs_for_color(grid: np.ndarray, color: int) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] | None:
    comps = _find_color_components(grid, color)
    if not comps or len(comps) % 2 != 0:
        return None

    by_row_span: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if comp["area"] != 4 or r1 - r0 != 1 or c1 - c0 != 1:
            return None
        by_row_span.setdefault((r0, r1), []).append(comp["bbox"])

    pairs = []
    gap_widths = set()
    for row_span in sorted(by_row_span):
        blocks = sorted(by_row_span[row_span], key=lambda b: b[1])
        if len(blocks) != 2:
            return None
        left, right = blocks
        gap_width = int(right[1] - left[3] - 1)
        if gap_width <= 0:
            return None
        pairs.append((left, right))
        gap_widths.add(gap_width)

    if not pairs or len(gap_widths) != 1:
        return None
    return pairs


def _paired_marker_blocks(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    bg = _get_bg(grid)
    colors = _colors_present(grid, exclude={bg})
    ordered = ([MARKER_COLOR] if MARKER_COLOR in colors else []) + [c for c in colors if c != MARKER_COLOR]

    candidates: list[tuple[int, list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]]] = []
    for color in ordered:
        pairs = _marker_pairs_for_color(grid, color)
        if pairs is not None:
            candidates.append((color, pairs))

    assert candidates, "expected paired 2x2 marker blocks"
    if any(color == MARKER_COLOR for color, _pairs in candidates):
        matches = [pairs for color, pairs in candidates if color == MARKER_COLOR]
        assert len(matches) == 1
        return matches[0]

    assert len(candidates) == 1, "expected a unique marker color"
    return candidates[0][1]


def _extract_stacked(grid: np.ndarray) -> np.ndarray:
    stripes = []
    widths = set()
    for left, right in _paired_marker_blocks(grid):
        stripe = grid[left[0] : left[2] + 1, left[3] + 1 : right[1]]
        assert stripe.shape[0] == 2 and stripe.shape[1] > 0
        stripes.append(stripe)
        widths.add(int(stripe.shape[1]))
    assert stripes and len(widths) == 1
    return np.vstack(stripes)


def _is_lr_symmetric(grid: np.ndarray) -> bool:
    return bool(np.array_equal(grid, np.fliplr(grid)))


def _swap_right_half(grid: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    right_start = (out.shape[1] + 1) // 2
    right = out[:, right_start:].copy()
    mask_a = right == int(color_a)
    mask_b = right == int(color_b)
    right[mask_a] = int(color_b)
    right[mask_b] = int(color_a)
    out[:, right_start:] = right
    return out


def _recover_symmetric_pattern(stacked: np.ndarray) -> np.ndarray:
    colors = [c for c in _colors_present(stacked, exclude={BACKGROUND})]
    solutions = []
    for color_a, color_b in combinations(colors, 2):
        candidate = _swap_right_half(stacked, color_a, color_b)
        if _is_lr_symmetric(candidate):
            solutions.append(candidate)
    assert len(solutions) == 1, "expected exactly one swap pair that restores symmetry"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract and stack the interiors between paired marker blocks, then undo the unknown right-half color swap to restore mirror symmetry.

    Input structure:
    - The grid has a black background with 2-3 row-aligned pairs of 2x2 marker blocks.
    - Each pair flanks a 2-row interior stripe; unrelated noise may appear elsewhere and must be ignored.
    - Stacking those interior stripes from top to bottom forms a wider pattern whose right half has one hidden two-color swap.

    Rule:
    - Identify the paired 2x2 marker blocks and crop the rectangle strictly between each pair.
    - Stack those rectangles vertically in top-to-bottom order to obtain pattern P.
    - Find the unique pair of colors such that swapping only those two colors in the right half of P makes P perfectly left-right mirror symmetric.
    - Output that repaired symmetric stacked pattern.

    Key insight:
    - The markers tell you what to extract, but symmetry tells you which two colors were swapped: the correct answer is the only right-half recoloring that makes the stacked pattern mirror itself.
    """
    g = np.asarray(grid, dtype=int)
    stacked = _extract_stacked(g)
    return _recover_symmetric_pattern(stacked)


def _row_starts(rng: np.random.Generator, height: int, num_pairs: int) -> list[int] | None:
    gaps = rng.integers(1, 4, size=max(0, num_pairs - 1))
    used = 2 * num_pairs + int(gaps.sum())
    slack = height - used
    if slack < 0:
        return None
    row = int(rng.integers(0, slack + 1))
    starts = [row]
    for gap in gaps:
        row += 2 + int(gap)
        starts.append(row)
    return starts


def _make_left_half(
    rng: np.random.Generator,
    height: int,
    half_width: int,
    palette: np.ndarray,
    swap_pair: tuple[int, int],
) -> np.ndarray | None:
    for _ in range(200):
        left = np.zeros((height, half_width), dtype=int)

        stamp_count = int(rng.integers(max(5, half_width + 1), max(8, height * half_width) + 1))
        for _ in range(stamp_count):
            color = int(rng.choice(palette))
            sh = int(rng.integers(1, min(2, height) + 1))
            sw = int(rng.integers(1, min(2, half_width) + 1))
            r0 = int(rng.integers(0, height - sh + 1))
            c0 = int(rng.integers(0, half_width - sw + 1))
            left[r0 : r0 + sh, c0 : c0 + sw] = color

        hole_count = int(rng.integers(0, max(2, height * half_width // 5) + 1))
        if hole_count:
            holes = rng.choice(height * half_width, size=hole_count, replace=False)
            left.flat[holes] = 0

        density = float(np.count_nonzero(left)) / float(left.size)
        if not (0.45 <= density <= 0.85):
            continue

        used_colors = _colors_present(left, exclude={0})
        if len(used_colors) < 3:
            continue

        count_map = {color: int(np.sum(left == color)) for color in used_colors}
        a, b = map(int, swap_pair)
        if count_map.get(a, 0) < 2 or count_map.get(b, 0) < 2:
            continue

        other_counts = [count_map[color] for color in used_colors if color not in {a, b}]
        if not other_counts or max(other_counts) < 2:
            continue

        if any(np.count_nonzero(left[2 * i : 2 * i + 2]) < 2 for i in range(height // 2)):
            continue

        if int(np.count_nonzero(left == 0)) == 0:
            continue

        return left
    return None


def _gen_stacked_pair(
    rng: np.random.Generator, num_pairs: int, gap_width: int
) -> tuple[np.ndarray, np.ndarray] | None:
    height = 2 * num_pairs
    half_width = gap_width // 2

    for _ in range(200):
        palette_size = int(rng.integers(3, 5))
        palette = rng.choice(PATTERN_COLORS, size=palette_size, replace=False)
        swap_colors = tuple(map(int, rng.choice(palette, size=2, replace=False)))
        left = _make_left_half(rng, height, half_width, palette, swap_colors)
        if left is None:
            continue

        expected = np.hstack([left, np.fliplr(left)])
        corrupted = _swap_right_half(expected, swap_colors[0], swap_colors[1])
        if np.array_equal(corrupted, expected):
            continue

        try:
            solved = _recover_symmetric_pattern(corrupted)
        except AssertionError:
            continue
        if np.array_equal(solved, expected):
            return corrupted, expected
    return None


def _add_noise(rng: np.random.Generator, grid: np.ndarray, protected: np.ndarray) -> None:
    free = np.argwhere(~protected)
    if len(free) == 0:
        return
    density = float(rng.uniform(0.04, 0.12))
    noise_count = int(round(len(free) * density))
    noise_count = max(2, min(noise_count, len(free)))
    picks = free[rng.choice(len(free), size=noise_count, replace=False)]
    colors = rng.choice(NOISE_COLORS, size=noise_count, replace=True)
    grid[picks[:, 0], picks[:, 1]] = colors


def _build_input_from_stacked(
    rng: np.random.Generator, stacked: np.ndarray, num_pairs: int, gap_width: int
) -> np.ndarray | None:
    min_height = 2 * num_pairs + max(1, num_pairs - 1) + 2
    max_height = min(20, min_height + 6)
    height = int(rng.integers(min_height, max_height + 1))
    row_starts = _row_starts(rng, height, num_pairs)
    if row_starts is None:
        return None

    min_width = max(gap_width + 8, gap_width + 4 + num_pairs)
    max_width = min(28, min_width + 8)
    if min_width > max_width:
        return None
    width = int(rng.integers(min_width, max_width + 1))

    max_left = width - (gap_width + 4)
    left_choices = np.arange(max_left + 1)
    if len(left_choices) < num_pairs:
        return None
    left_cols = rng.choice(left_choices, size=num_pairs, replace=False)

    grid = np.zeros((height, width), dtype=int)
    protected = np.zeros((height, width), dtype=bool)
    for i, (row, left_col) in enumerate(zip(row_starts, left_cols)):
        stripe = stacked[2 * i : 2 * i + 2]
        right_col = int(left_col + 2 + gap_width)

        grid[row : row + 2, left_col : left_col + 2] = MARKER_COLOR
        grid[row : row + 2, left_col + 2 : right_col] = stripe
        grid[row : row + 2, right_col : right_col + 2] = MARKER_COLOR
        protected[row : row + 2, left_col : right_col + 2] = True

    _add_noise(rng, grid, protected)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    num_pairs = int(rng.integers(2, 4))
    gap_width = int(rng.choice([6, 8, 10]))

    stacked_pair = _gen_stacked_pair(rng, num_pairs, gap_width)
    if stacked_pair is None:
        return None
    corrupted, expected = stacked_pair

    grid = _build_input_from_stacked(rng, corrupted, num_pairs, gap_width)
    if grid is None:
        return None

    counts = np.bincount(grid.ravel(), minlength=10)
    if counts[BACKGROUND] <= max(counts[1:]):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
