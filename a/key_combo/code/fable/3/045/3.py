from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _has_horizontal_pair(cols: list[int]) -> bool:
    return any(b == a + 1 for a, b in zip(cols, cols[1:]))


def _find_bars(g: np.ndarray, bg: int) -> list[tuple[int, int, set[int]]]:
    bars: list[tuple[int, int, set[int]]] = []
    for color in _colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        rows = sorted({int(r) for r, _c in coords})
        if len(rows) != 1:
            continue
        cols = sorted({int(c) for _r, c in coords})
        if len(cols) >= 2 and _has_horizontal_pair(cols):
            bars.append((rows[0], int(color), set(cols)))

    assert bars, "expected at least one horizontal sieve bar"
    bar_rows = [row for row, _color, _cols in bars]
    assert len(set(bar_rows)) == len(bar_rows), "expected sieve bars on distinct rows"
    return sorted(bars)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop scattered source pixels through gapped horizontal sieve bars and recolor each settled stack by its catcher.

    Input structure:
    - A background canvas contains several one-pixel-thick horizontal sieve bars on distinct rows.
    - Each sieve bar has its own color and contains background-colored gaps in some columns.
    - The only remaining foreground color appears as isolated source pixels above the sieve stack.

    Rule:
    - Clear every source pixel from its original position while keeping all sieve bars unchanged.
    - In each column, process source pixels from lower to higher; each pixel falls straight down until the next cell below it is a sieve pixel or an already settled pixel.
    - A settled pixel is painted with the color of that supporting pixel, so a whole vertical stack inherits the color of the bottom sieve bar that caught it.
    - A source pixel that reaches the bottom without finding any support is erased.

    Key insight:
    - Gaps are vertical channels, not damaged bar pixels: the first non-gap sieve cell in the source column determines the stack position and its final color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = _background(g)

    bars = _find_bars(g, bg)
    bar_colors = {color for _row, color, _cols in bars}
    source_colors = [color for color in _colors_present(g, exclude={bg}) if color not in bar_colors]
    assert len(source_colors) == 1, "expected exactly one source-pixel color"
    source_color = int(source_colors[0])

    out = g.copy()
    out[g == source_color] = bg

    source_coords = np.argwhere(g == source_color)
    assert source_coords.size > 0, "expected at least one source pixel"
    for c_raw in sorted({int(c) for _r, c in source_coords}):
        rows = sorted((int(r) for r, c in source_coords if int(c) == c_raw), reverse=True)
        for r in rows:
            support_row = None
            for rr in range(r + 1, height):
                if int(out[rr, c_raw]) != bg:
                    support_row = rr
                    break
            if support_row is None:
                continue

            land_row = support_row - 1
            assert 0 <= land_row < height, "source has no room to settle above its support"
            assert int(out[land_row, c_raw]) == bg, "landing cell must be empty"
            out[land_row, c_raw] = int(out[support_row, c_raw])

    assert out.shape == (height, width)
    return out


def _spaced_sample(columns: list[int], count: int, gap: int = 2) -> list[int] | None:
    for _ in range(100):
        picked = sorted(random.sample(columns, count))
        if all(b - a >= gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _catcher_index(col: int, gap_sets: list[set[int]]) -> int | None:
    for idx, gaps in enumerate(gap_sets):
        if col not in gaps:
            return idx
    return None


def _render_bars(
    shape: tuple[int, int],
    bar_rows: list[int],
    bar_colors: list[int],
    gap_sets: list[set[int]],
) -> np.ndarray:
    height, width = shape
    grid = np.full((height, width), BG, dtype=int)
    for row, color, gaps in zip(bar_rows, bar_colors, gap_sets):
        for col in range(width):
            if col not in gaps:
                grid[row, col] = color
    return grid


def _source_free(grid: np.ndarray, r: int, c: int, source_color: int) -> bool:
    if int(grid[r, c]) != BG:
        return False
    r0, r1 = max(0, r - 1), min(grid.shape[0], r + 2)
    c0, c1 = max(0, c - 1), min(grid.shape[1], c + 2)
    return not np.any(grid[r0:r1, c0:c1] == source_color)


def _place_sources(
    grid: np.ndarray,
    source_counts: dict[int, int],
    first_bar_row: int,
    source_color: int,
) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int]] = []
    items = list(source_counts.items())
    random.shuffle(items)
    source_rows = list(range(0, first_bar_row - 1))
    if not source_rows:
        return None

    for col, count in items:
        for _ in range(count):
            rows = source_rows.copy()
            random.shuffle(rows)
            placed = False
            for row in rows:
                if _source_free(grid, row, col, source_color):
                    grid[row, col] = source_color
                    placements.append((row, col))
                    placed = True
                    break
            if not placed:
                return None
    return placements


def _expected_from_specs(
    shape: tuple[int, int],
    bar_rows: list[int],
    bar_colors: list[int],
    gap_sets: list[set[int]],
    source_counts: dict[int, int],
) -> np.ndarray | None:
    out = _render_bars(shape, bar_rows, bar_colors, gap_sets)
    for col, count in source_counts.items():
        catcher = _catcher_index(col, gap_sets)
        if catcher is None:
            continue
        catch_row = bar_rows[catcher]
        if count >= catch_row + 1:
            return None
        for k in range(1, count + 1):
            row = catch_row - k
            if int(out[row, col]) != BG:
                return None
            out[row, col] = bar_colors[catcher]
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    gap_sets: list[set[int]],
    source_counts: dict[int, int],
) -> bool:
    catchers = {col: _catcher_index(col, gap_sets) for col in source_counts}
    caught = [idx for idx in catchers.values() if idx is not None]
    if not caught:
        return False
    if 0 not in caught:
        return False
    if not any(idx is not None and idx > 0 for idx in catchers.values()):
        return False
    if not any(idx is None for idx in catchers.values()):
        return False
    if not any(source_counts[col] > 1 and catchers[col] is not None for col in source_counts):
        return False
    if len(set(caught)) < 2:
        return False
    if np.array_equal(inp, expected):
        return False
    return True


def _build_gap_sets(width: int, n_bars: int, required: dict[int, int | None]) -> list[set[int]] | None:
    gap_sets = [set() for _ in range(n_bars)]
    for col, catcher in required.items():
        if catcher is None:
            for gaps in gap_sets:
                gaps.add(col)
        else:
            for idx in range(catcher):
                gap_sets[idx].add(col)

    protected = set(required)
    available = [col for col in range(1, width - 1) if col not in protected]
    max_gaps = max(3, min(6, width // 3))
    for gaps in gap_sets:
        target = random.randint(2, max_gaps)
        choices = available.copy()
        random.shuffle(choices)
        for col in choices:
            if len(gaps) >= target:
                break
            gaps.add(col)
        if len(gaps) < 1 or len(gaps) >= width - 2:
            return None

    if len({tuple(sorted(gaps)) for gaps in gap_sets}) != n_bars:
        return None
    return gap_sets


def _choose_extra_sources(
    width: int,
    source_counts: dict[int, int],
    gap_sets: list[set[int]],
) -> None:
    extra_count = random.randint(0, 3)
    for _ in range(extra_count):
        candidates = [
            col
            for col in range(1, width - 1)
            if col not in source_counts
            and all(abs(col - used) >= 2 for used in source_counts)
            and _catcher_index(col, gap_sets) is not None
        ]
        if not candidates:
            return
        col = random.choice(candidates)
        source_counts[col] = 1 if random.random() < 0.7 else 2


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(2, 5)
    width = random.randint(10, 24)

    first_bar_row = random.randint(7, 9)
    bar_rows = [first_bar_row]
    for _ in range(n_bars - 1):
        bar_rows.append(bar_rows[-1] + random.randint(4, 5))
    height = bar_rows[-1] + 1 + random.randint(2, 4)
    if height > 30:
        return None

    needed_cols = 4 if n_bars == 2 else 5
    special = _spaced_sample(list(range(1, width - 1)), needed_cols, gap=2)
    if special is None:
        return None
    random.shuffle(special)

    top_col = special[0]
    stack_col = special[1]
    leak_col = special[2]
    lower_col = special[3]

    stack_catcher = random.randint(1, n_bars - 1)
    lower_options = [idx for idx in range(1, n_bars) if idx != stack_catcher]
    lower_catcher = random.choice(lower_options) if lower_options else stack_catcher

    required: dict[int, int | None] = {
        top_col: 0,
        stack_col: stack_catcher,
        lower_col: lower_catcher,
        leak_col: None,
    }
    if n_bars >= 3 and needed_cols == 5:
        required[special[4]] = random.randint(1, n_bars - 1)

    gap_sets = _build_gap_sets(width, n_bars, required)
    if gap_sets is None:
        return None

    colors = random.sample(PALETTE, n_bars + 1)
    source_color = colors[0]
    bar_colors = colors[1:]

    source_counts: dict[int, int] = {
        top_col: 1,
        stack_col: random.randint(2, 3),
        lower_col: 1,
        leak_col: random.randint(1, 2),
    }
    for col in required:
        if col not in source_counts and required[col] is not None:
            source_counts[col] = 1
    _choose_extra_sources(width, source_counts, gap_sets)

    expected = _expected_from_specs((height, width), bar_rows, bar_colors, gap_sets, source_counts)
    if expected is None:
        return None

    inp = _render_bars((height, width), bar_rows, bar_colors, gap_sets)
    placements = _place_sources(inp, source_counts, first_bar_row, source_color)
    if placements is None:
        return None

    if len(placements) != sum(source_counts.values()):
        return None
    if not _semantic_checks(inp, expected, gap_sets, source_counts):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layered sieve-drop input")
