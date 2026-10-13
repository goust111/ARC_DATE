from __future__ import annotations

from collections import Counter

import numpy as np


BACKGROUND = 0
YELLOW = 4
INPUT_PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 600


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def border_path(height: int, width: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(width):
        coords.append((0, c))
    for r in range(1, height - 1):
        coords.append((r, width - 1))
    for c in range(width - 1, -1, -1):
        coords.append((height - 1, c))
    for r in range(height - 2, 0, -1):
        coords.append((r, 0))
    return coords


def cyclic_runs(values: list[int]) -> list[tuple[int, int, int]]:
    assert values
    runs: list[tuple[int, int, int]] = []
    start = 0
    color = int(values[0])
    for idx in range(1, len(values)):
        value = int(values[idx])
        if value != color:
            runs.append((start, idx - start, color))
            start = idx
            color = value
    runs.append((start, len(values) - start, color))
    if len(runs) > 1 and runs[0][2] == runs[-1][2]:
        last_start, last_len, last_color = runs.pop()
        _first_start, first_len, _first_color = runs[0]
        runs[0] = (last_start, last_len + first_len, last_color)
    return runs


def periodic_cover(perimeter: int, start: int, length: int) -> np.ndarray:
    indices = np.arange(perimeter, dtype=int)
    return ((indices - int(start)) % (2 * int(length))) < int(length)


def render_from_segments(
    shape: tuple[int, int],
    bg: int,
    seed_a: tuple[int, int, int],
    seed_b: tuple[int, int, int],
) -> np.ndarray:
    height, width = map(int, shape)
    path = border_path(height, width)
    perimeter = len(path)
    start_a, len_a, color_a = seed_a
    start_b, len_b, color_b = seed_b
    cover_a = periodic_cover(perimeter, start_a, len_a)
    cover_b = periodic_cover(perimeter, start_b, len_b)

    out = np.full((height, width), int(bg), dtype=int)
    for idx, (r, c) in enumerate(path):
        a = bool(cover_a[idx])
        b = bool(cover_b[idx])
        if a and b:
            out[r, c] = YELLOW
        elif a:
            out[r, c] = int(color_a)
        elif b:
            out[r, c] = int(color_b)
    return out


def extract_seed_segments(grid: np.ndarray) -> tuple[int, tuple[int, int, int], tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert 2 <= height <= 30 and 2 <= width <= 30, "grid dimensions must be within ARC limits"

    bg = get_bg(g)
    assert bg != YELLOW, "background must differ from the overlap color"
    if height > 2 and width > 2:
        assert np.all(g[1:-1, 1:-1] == bg), "the canvas interior must be pure background"

    path = border_path(height, width)
    border_values = [int(g[r, c]) for r, c in path]
    runs = cyclic_runs(border_values)
    non_bg_runs = sorted((run for run in runs if run[2] != bg), key=lambda item: item[0])

    assert len(non_bg_runs) == 2, "expected exactly two foreground border segments"
    seed_a, seed_b = non_bg_runs
    start_a, len_a, color_a = seed_a
    start_b, len_b, color_b = seed_b
    assert len_a != len_b, "the two seed segment lengths must differ"
    assert color_a != color_b, "the two seed segment colors must differ"
    assert color_a != YELLOW and color_b != YELLOW, "yellow is reserved for overlap cells"
    assert all(value in {bg, color_a, color_b} for value in border_values), "border may only contain background and the two seed colors"

    perimeter = len(path)
    assert perimeter % (2 * len_a) == 0 and perimeter % (2 * len_b) == 0, "segment lengths must define valid periods"
    return bg, seed_a, seed_b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the two seed border segments into independent border waves and color their overlap yellow.

    Input structure:
    - The canvas interior is entirely background.
    - Only the outer rectangular border is relevant, and it contains exactly two continuous foreground segments.
    - The two segments have different colors and different lengths, and each length defines a valid half-duty cyclic period on the same border loop.

    Rule:
    - Unroll the outer border clockwise into one cyclic path.
    - For each seed segment, keep its clockwise start phase and repeat an on-block of that segment's length every other block around the loop, so its period is twice its own length.
    - Paint cells covered by both repeated waves yellow, cells covered by exactly one wave with that wave's seed color, and leave all remaining cells as background.

    Key insight:
    - The input gives two short border intervals, but the output is determined by treating each one as a phase-anchored square wave on the same perimeter cycle and then combining the two waves pointwise.
    """
    g = np.asarray(grid, dtype=int)
    bg, seed_a, seed_b = extract_seed_segments(g)
    return render_from_segments(g.shape, bg, seed_a, seed_b)


def interval_cells(start: int, length: int, perimeter: int) -> set[int]:
    return {(int(start) + step) % int(perimeter) for step in range(int(length))}


def corner_indices(height: int, width: int) -> list[int]:
    return [0, width - 1, width + height - 2, 2 * width + height - 3]


def segment_crosses_corner(start: int, length: int, height: int, width: int) -> bool:
    cells = interval_cells(start, length, 2 * height + 2 * width - 4)
    return any(corner in cells for corner in corner_indices(height, width))


def render_input_from_spec(
    shape: tuple[int, int],
    bg: int,
    seed_a: tuple[int, int, int],
    seed_b: tuple[int, int, int],
) -> np.ndarray:
    height, width = map(int, shape)
    out = np.full((height, width), int(bg), dtype=int)
    path = border_path(height, width)
    for start, length, color in (seed_a, seed_b):
        for offset in range(int(length)):
            r, c = path[(int(start) + offset) % len(path)]
            out[r, c] = int(color)
    return out


def choose_dimensions() -> tuple[int, int]:
    while True:
        height = int(RNG.integers(6, 19))
        width = int(RNG.integers(6, 19))
        half = height + width - 2
        divisors = [d for d in range(2, half + 1) if half % d == 0]
        if len(divisors) >= 2:
            return height, width


def candidate_lengths(height: int, width: int) -> list[int]:
    half = height + width - 2
    perimeter = 2 * half
    return [d for d in range(2, half + 1) if half % d == 0 and 2 * d <= perimeter and d < perimeter]


def choose_length_pair(height: int, width: int) -> tuple[int, int]:
    lengths = candidate_lengths(height, width)
    valid_pairs = [(a, b) for a in lengths for b in lengths if a != b and a + b <= (2 * height + 2 * width - 4) - 2]
    assert valid_pairs
    idx = int(RNG.integers(len(valid_pairs)))
    return valid_pairs[idx]


def interesting_starts(length: int, height: int, width: int) -> list[int]:
    perimeter = 2 * height + 2 * width - 4
    starts: set[int] = set()
    for corner in corner_indices(height, width):
        for delta in range(length):
            starts.add((corner - delta) % perimeter)
    return sorted(starts)


def choose_segment_start(
    length: int,
    height: int,
    width: int,
    occupied: set[int],
    force_interesting: bool,
) -> int | None:
    perimeter = 2 * height + 2 * width - 4
    pool = interesting_starts(length, height, width) if force_interesting else list(range(perimeter))
    if not pool:
        return None
    perm = RNG.permutation(len(pool))
    for raw_idx in perm:
        start = int(pool[int(raw_idx)])
        cells = interval_cells(start, length, perimeter)
        if occupied & cells:
            continue
        return start
    if force_interesting:
        return choose_segment_start(length, height, width, occupied, force_interesting=False)
    return None


def output_color_counts(output: np.ndarray, bg: int, color_a: int, color_b: int) -> dict[int, int]:
    unique, counts = np.unique(output, return_counts=True)
    mapping = {int(color): int(count) for color, count in zip(unique, counts)}
    for color in (bg, color_a, color_b, YELLOW):
        mapping.setdefault(int(color), 0)
    return mapping


def spec_is_good(
    shape: tuple[int, int],
    bg: int,
    seed_a: tuple[int, int, int],
    seed_b: tuple[int, int, int],
) -> bool:
    color_a = int(seed_a[2])
    color_b = int(seed_b[2])
    out = render_from_segments(shape, bg, seed_a, seed_b)
    counts = output_color_counts(out, bg, color_a, color_b)
    if counts[YELLOW] == 0 or counts[color_a] == 0 or counts[color_b] == 0 or counts[bg] == 0:
        return False
    inp = render_input_from_spec(shape, bg, seed_a, seed_b)
    if np.array_equal(inp, out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height, width = choose_dimensions()
        perimeter = 2 * height + 2 * width - 4
        len_a, len_b = choose_length_pair(height, width)
        color_a, color_b = RNG.choice(INPUT_PALETTE, size=2, replace=False)

        occupied: set[int] = set()
        start_a = choose_segment_start(
            len_a,
            height,
            width,
            occupied,
            force_interesting=bool(RNG.random() < 0.7),
        )
        if start_a is None:
            continue
        occupied |= interval_cells(start_a, len_a, perimeter)

        start_b = choose_segment_start(
            len_b,
            height,
            width,
            occupied,
            force_interesting=bool(RNG.random() < 0.7),
        )
        if start_b is None:
            continue

        seed_a = (start_a, len_a, int(color_a))
        seed_b = (start_b, len_b, int(color_b))
        if not spec_is_good((height, width), BACKGROUND, seed_a, seed_b):
            continue

        # Keep wrap-around and cross-corner examples in the mix, but do not force them every time.
        interesting = segment_crosses_corner(start_a, len_a, height, width) or segment_crosses_corner(start_b, len_b, height, width)
        if not interesting and RNG.random() < 0.25:
            continue
        return render_input_from_spec((height, width), BACKGROUND, seed_a, seed_b)

    raise RuntimeError("failed to generate a valid border-wave interference input")
