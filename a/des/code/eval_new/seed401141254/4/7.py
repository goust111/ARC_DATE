from __future__ import annotations

import random

import numpy as np


BG = 8
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)
MAX_ATTEMPTS = 400
GRID_MIN = 9
GRID_MAX = 18
MAX_ACTIVE_RAILS = 4
MAX_SINGLETONS = 3
TRUNCATION_MODES = ("none", "neg", "pos", "both")


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _schedule_paint(
    paint: dict[tuple[int, int], int],
    seed_mask: np.ndarray,
    r: int,
    c: int,
    color: int,
) -> None:
    if seed_mask[r, c]:
        return
    key = (int(r), int(c))
    prev = paint.get(key)
    if prev is None or prev == int(color):
        paint[key] = int(color)
        return
    raise AssertionError("conflicting stripe colors on overlapping rails")


def _grow_row(
    paint: dict[tuple[int, int], int],
    seed_mask: np.ndarray,
    row: int,
    left_c: int,
    right_c: int,
    left_color: int,
    right_color: int,
    width: int,
) -> None:
    gap = right_c - left_c - 1
    for step in range(1, gap + 1):
        c = left_c - step
        if c < 0:
            break
        _schedule_paint(paint, seed_mask, row, c, right_color)
    for step in range(1, gap + 1):
        c = right_c + step
        if c >= width:
            break
        _schedule_paint(paint, seed_mask, row, c, left_color)


def _grow_col(
    paint: dict[tuple[int, int], int],
    seed_mask: np.ndarray,
    col: int,
    top_r: int,
    bottom_r: int,
    top_color: int,
    bottom_color: int,
    height: int,
) -> None:
    gap = bottom_r - top_r - 1
    for step in range(1, gap + 1):
        r = top_r - step
        if r < 0:
            break
        _schedule_paint(paint, seed_mask, r, col, bottom_color)
    for step in range(1, gap + 1):
        r = bottom_r + step
        if r >= height:
            break
        _schedule_paint(paint, seed_mask, r, col, top_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each two-seed row or column into outward stripes whose color and length come from the opposite seed.

    Input structure:
    - The background is azure.
    - The foreground consists of scattered single-pixel colored seeds.
    - Some rows or columns contain exactly two seeds; all other rows and columns
      contain at most one seed.

    Rule:
    - Treat every row or column with exactly two seeds as an active rail.
    - On an active rail, each seed grows away from its partner along that rail.
    - The grown stripe uses the partner seed's color and has length equal to the
      number of empty cells between the two seeds; growth truncates at the grid
      boundary.
    - Original seed pixels keep their own colors, and rows or columns with fewer
      than two seeds remain unchanged.

    Key insight:
    - A paired seed does not describe itself: its color becomes the other seed's
      stripe color, while the pair's gap length and reversed direction jointly
      determine how far the visible band grows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    bg = _background_color(g)
    seed_mask = g != bg

    out = g.copy()
    paint: dict[tuple[int, int], int] = {}

    for r in range(height):
        cols = np.flatnonzero(seed_mask[r])
        assert len(cols) <= 2, "rows may contain at most two seeds"
        if len(cols) != 2:
            continue
        left_c, right_c = map(int, cols)
        _grow_row(
            paint,
            seed_mask,
            r,
            left_c,
            right_c,
            int(g[r, left_c]),
            int(g[r, right_c]),
            width,
        )

    for c in range(width):
        rows = np.flatnonzero(seed_mask[:, c])
        assert len(rows) <= 2, "columns may contain at most two seeds"
        if len(rows) != 2:
            continue
        top_r, bottom_r = map(int, rows)
        _grow_col(
            paint,
            seed_mask,
            c,
            top_r,
            bottom_r,
            int(g[top_r, c]),
            int(g[bottom_r, c]),
            height,
        )

    for (r, c), color in paint.items():
        out[r, c] = color
    return out


def _truncation_mode(a: int, b: int, length: int) -> str | None:
    gap = b - a - 1
    if gap <= 0:
        return None
    neg_trunc = a < gap
    pos_trunc = length - 1 - b < gap
    if neg_trunc and pos_trunc:
        return "both"
    if neg_trunc:
        return "neg"
    if pos_trunc:
        return "pos"
    return "none"


def _pair_candidates(length: int, mode: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for a in range(length):
        for b in range(a + 2, length):
            if _truncation_mode(a, b, length) == mode:
                pairs.append((a, b))
    return pairs


def _choose_modes(count: int) -> list[str]:
    if count == 1:
        return [random.choice(TRUNCATION_MODES)]
    modes = [random.choice(("none",)), random.choice(("neg", "pos", "both"))]
    while len(modes) < count:
        modes.append(random.choice(TRUNCATION_MODES))
    random.shuffle(modes)
    return modes


def _choose_same_color_flags(count: int) -> list[bool]:
    if count == 1:
        return [bool(random.getrandbits(1))]
    flags = [True, False]
    while len(flags) < count:
        flags.append(bool(random.getrandbits(1)))
    random.shuffle(flags)
    return flags


def _select_disjoint_pairs(length: int, modes: list[str]) -> list[tuple[int, int]] | None:
    buckets = {mode: _pair_candidates(length, mode) for mode in TRUNCATION_MODES}
    if any(not buckets[mode] for mode in modes):
        return None

    order = sorted(range(len(modes)), key=lambda i: len(buckets[modes[i]]))
    for mode in TRUNCATION_MODES:
        random.shuffle(buckets[mode])

    chosen: list[tuple[int, int] | None] = [None] * len(modes)
    used: set[int] = set()

    def backtrack(k: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        for a, b in buckets[modes[idx]]:
            if a in used or b in used:
                continue
            used.add(a)
            used.add(b)
            chosen[idx] = (a, b)
            if backtrack(k + 1):
                return True
            chosen[idx] = None
            used.remove(a)
            used.remove(b)
        return False

    if not backtrack(0):
        return None
    return [pair for pair in chosen if pair is not None]


def _paint_expected_from_rows(
    expected: np.ndarray,
    rails: list[tuple[int, int, int, int, int]],
) -> None:
    width = expected.shape[1]
    for row, left_c, right_c, left_color, right_color in rails:
        gap = right_c - left_c - 1
        for step in range(1, gap + 1):
            c = left_c - step
            if c < 0:
                break
            expected[row, c] = right_color
        for step in range(1, gap + 1):
            c = right_c + step
            if c >= width:
                break
            expected[row, c] = left_color


def _paint_expected_from_cols(
    expected: np.ndarray,
    rails: list[tuple[int, int, int, int, int]],
) -> None:
    height = expected.shape[0]
    for col, top_r, bottom_r, top_color, bottom_color in rails:
        gap = bottom_r - top_r - 1
        for step in range(1, gap + 1):
            r = top_r - step
            if r < 0:
                break
            expected[r, col] = bottom_color
        for step in range(1, gap + 1):
            r = bottom_r + step
            if r >= height:
                break
            expected[r, col] = top_color


def _sample_scene_palette() -> list[int]:
    palette_size = random.randint(3, 6)
    return random.sample(list(PALETTE), palette_size)


def _render_row_scene(height: int, width: int) -> np.ndarray | None:
    max_active = min(MAX_ACTIVE_RAILS, height, width // 2)
    if max_active < 2:
        return None
    active_count = random.randint(2, max_active)
    max_singletons = min(MAX_SINGLETONS, height - active_count, width - 2 * active_count)
    singleton_count = random.randint(0, max(0, max_singletons))

    modes = _choose_modes(active_count)
    pairs = _select_disjoint_pairs(width, modes)
    if pairs is None:
        return None

    active_rows = random.sample(range(height), active_count)
    remaining_rows = [r for r in range(height) if r not in active_rows]
    singleton_rows = random.sample(remaining_rows, singleton_count)

    used_cols = {c for pair in pairs for c in pair}
    remaining_cols = [c for c in range(width) if c not in used_cols]
    singleton_cols = random.sample(remaining_cols, singleton_count)

    palette = _sample_scene_palette()
    same_flags = _choose_same_color_flags(active_count)

    grid = np.full((height, width), BG, dtype=int)
    expected = grid.copy()
    rails: list[tuple[int, int, int, int, int]] = []

    for row, (left_c, right_c), same_color in zip(active_rows, pairs, same_flags):
        left_color = random.choice(palette)
        right_color = left_color if same_color else random.choice([c for c in palette if c != left_color])
        grid[row, left_c] = left_color
        grid[row, right_c] = right_color
        expected[row, left_c] = left_color
        expected[row, right_c] = right_color
        rails.append((row, left_c, right_c, left_color, right_color))

    for row, col in zip(singleton_rows, singleton_cols):
        color = random.choice(palette)
        grid[row, col] = color
        expected[row, col] = color

    _paint_expected_from_rows(expected, rails)
    return grid if np.array_equal(main(grid), expected) else None


def _render_col_scene(height: int, width: int) -> np.ndarray | None:
    max_active = min(MAX_ACTIVE_RAILS, height // 2, width)
    if max_active < 2:
        return None
    active_count = random.randint(2, max_active)
    max_singletons = min(MAX_SINGLETONS, height - 2 * active_count, width - active_count)
    singleton_count = random.randint(0, max(0, max_singletons))

    modes = _choose_modes(active_count)
    pairs = _select_disjoint_pairs(height, modes)
    if pairs is None:
        return None

    active_cols = random.sample(range(width), active_count)
    remaining_cols = [c for c in range(width) if c not in active_cols]
    singleton_cols = random.sample(remaining_cols, singleton_count)

    used_rows = {r for pair in pairs for r in pair}
    remaining_rows = [r for r in range(height) if r not in used_rows]
    singleton_rows = random.sample(remaining_rows, singleton_count)

    palette = _sample_scene_palette()
    same_flags = _choose_same_color_flags(active_count)

    grid = np.full((height, width), BG, dtype=int)
    expected = grid.copy()
    rails: list[tuple[int, int, int, int, int]] = []

    for col, (top_r, bottom_r), same_color in zip(active_cols, pairs, same_flags):
        top_color = random.choice(palette)
        bottom_color = top_color if same_color else random.choice([c for c in palette if c != top_color])
        grid[top_r, col] = top_color
        grid[bottom_r, col] = bottom_color
        expected[top_r, col] = top_color
        expected[bottom_r, col] = bottom_color
        rails.append((col, top_r, bottom_r, top_color, bottom_color))

    for row, col in zip(singleton_rows, singleton_cols):
        color = random.choice(palette)
        grid[row, col] = color
        expected[row, col] = color

    _paint_expected_from_cols(expected, rails)
    return grid if np.array_equal(main(grid), expected) else None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    if bool(random.getrandbits(1)):
        return _render_row_scene(height, width)
    return _render_col_scene(height, width)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
