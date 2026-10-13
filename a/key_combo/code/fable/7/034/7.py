from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 300

# Cycling these two genuinely visible branches makes every first group of three
# generated examples cover all color counts and both possible balanced shapes.
_generation_index = 0
_seen_outputs: set[tuple[tuple[int, int], bytes]] = set()


def _clockwise_border(h: int, w: int) -> list[tuple[int, int]]:
    """Border coordinates starting at upper-left, initially moving right."""
    path = [(0, c) for c in range(w)]
    path.extend((r, w - 1) for r in range(1, h))
    path.extend((h - 1, c) for c in range(w - 2, -1, -1))
    path.extend((r, 0) for r in range(h - 2, 0, -1))
    return path


def _balanced_dimensions(perimeter: int) -> tuple[int, int]:
    """Return the unique most balanced orientation with height <= width."""
    assert perimeter >= 4 and perimeter % 2 == 0
    side_sum = perimeter // 2 + 2
    h = side_sum // 2
    w = side_sum - h
    assert 2 * h + 2 * w - 4 == perimeter and 2 <= h <= w
    return h, w


def main(grid: np.ndarray) -> np.ndarray:
    """Melt all foreground pixels into a frequency-ordered, most-balanced hollow frame.

    Input structure:
    - A black background contains scattered monochromatic foreground objects in several colors.
    - Object shapes and positions are arbitrary; each color has a distinct total pixel count.
    - The even foreground total is the perimeter of at least one one-cell-thick rectangular frame.

    Rule:
    - Count all pixels of each foreground color and choose, with height no greater than width,
      the perimeter-matching rectangle whose height and width are closest.
    - Starting at the upper-left corner and walking clockwise (first moving right), fill one
      continuous segment per color in strictly decreasing order of total pixel count.
    - Leave every cell strictly inside the frame black.

    Key insight:
    - The objects are only stores of colored material: their combined amount determines the
      balanced frame, while frequency rank determines the clockwise segment order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert np.all((0 <= g) & (g <= 9)), "expected ARC palette values"

    colors, raw_counts = np.unique(g[g != BACKGROUND], return_counts=True)
    assert len(colors) >= 2, "at least two foreground colors are required"
    counts = [int(n) for n in raw_counts]
    assert len(set(counts)) == len(counts), "foreground color totals must be distinct"

    total = sum(counts)
    h, w = _balanced_dimensions(total)
    assert h <= 30 and w <= 30

    ranked = sorted(
        ((int(n), int(color)) for color, n in zip(colors, raw_counts)),
        reverse=True,
    )
    sequence: list[int] = []
    for count, color in ranked:
        sequence.extend([color] * count)

    path = _clockwise_border(h, w)
    assert len(path) == total == len(sequence)
    out = np.zeros((h, w), dtype=int)
    for (r, c), color in zip(path, sequence):
        out[r, c] = color
    return out


def _random_polyomino(size: int) -> set[tuple[int, int]]:
    """Grow a connected, irregular polyomino and normalize its coordinates."""
    cells = {(0, 0)}
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while len(cells) < size:
        r, c = random.choice(tuple(cells))
        dr, dc = random.choice(directions)
        cells.add((r + dr, c + dc))
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _split_count(total: int, parts: int) -> list[int]:
    """Random positive composition, avoiding one dominant whole-color object."""
    assert total >= parts
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = [b - a for a, b in zip([0] + cuts, cuts + [total])]
    random.shuffle(values)
    return values


def _place_object(
    grid: np.ndarray, shape: set[tuple[int, int]], color: int
) -> bool:
    """Place with a one-cell Chebyshev moat, keeping object identities clear."""
    H, W = grid.shape
    ph = max(r for r, _ in shape) + 1
    pw = max(c for _, c in shape) + 1
    candidates = [
        (r0, c0)
        for r0 in range(1, H - ph)
        for c0 in range(1, W - pw)
    ]
    random.shuffle(candidates)
    occupied = set(map(tuple, np.argwhere(grid != BACKGROUND)))
    for r0, c0 in candidates:
        cells = {(r0 + r, c0 + c) for r, c in shape}
        if any(
            (r + dr, c + dc) in occupied
            for r, c in cells
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
        ):
            continue
        for r, c in cells:
            grid[r, c] = color
        return True
    return False


def _component_sizes(mask: np.ndarray) -> list[int]:
    """Small self-contained 4-connected component counter for validation."""
    unseen = set(map(tuple, np.argwhere(mask)))
    sizes: list[int] = []
    while unseen:
        stack = [unseen.pop()]
        size = 0
        while stack:
            r, c = stack.pop()
            size += 1
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
        sizes.append(size)
    return sizes


def _semantic_check(grid: np.ndarray, counts_by_color: dict[int, int]) -> np.ndarray:
    """Check the stated semantics independently of generator construction."""
    observed = {
        int(color): int(np.count_nonzero(grid == color))
        for color in np.unique(grid)
        if int(color) != BACKGROUND
    }
    assert observed == counts_by_color
    assert len(set(observed.values())) == len(observed)
    assert BACKGROUND == 0 and np.count_nonzero(grid == 0) > sum(observed.values())

    # Every total must require aggregation over at least two visible objects.
    for color, total in observed.items():
        parts = _component_sizes(grid == color)
        assert len(parts) >= 2 and max(parts) < total

    total = sum(observed.values())
    candidates = [
        (abs(h - w), h, w)
        for h in range(2, total + 1)
        for w in range(h, total + 1)
        if 2 * h + 2 * w - 4 == total
    ]
    best_gap = min(item[0] for item in candidates)
    best = [(h, w) for gap, h, w in candidates if gap == best_gap]
    assert len(best) == 1

    out = main(grid)
    assert out.shape == best[0]
    h, w = out.shape
    if h > 2 and w > 2:
        assert np.all(out[1:-1, 1:-1] == BACKGROUND)
    expected = []
    for color, count in sorted(observed.items(), key=lambda item: item[1], reverse=True):
        expected.extend([color] * count)
    actual = [int(out[r, c]) for r, c in _clockwise_border(h, w)]
    assert actual == expected
    assert {c: actual.count(c) for c in observed} == observed
    return out


def _gen_attempt(index: int) -> tuple[np.ndarray, np.ndarray] | None:
    color_count = (3, 4, 5)[index % 3]
    # Alternating parity covers square and one-column-wider balanced frames.
    side_sums = [s for s in range(10, 21) if s % 2 == index % 2]
    random.shuffle(side_sums)

    for side_sum in side_sums:
        perimeter = 2 * side_sum - 4
        choices = list(combinations(range(2, 14), color_count))
        choices = [values for values in choices if sum(values) == perimeter]
        random.shuffle(choices)
        for values in choices[:30]:
            colors = random.sample(PALETTE, color_count)
            random.shuffle(colors)
            counts_by_color = dict(zip(colors, values))
            rank_colors = [
                c for c, _ in sorted(counts_by_color.items(), key=lambda item: item[1], reverse=True)
            ]
            # Color IDs carry no ordering information, visibly ruling out both ID directions.
            if rank_colors in (sorted(colors), sorted(colors, reverse=True)):
                continue

            H = random.randint(19, 27)
            W = random.randint(19, 27)
            grid = np.zeros((H, W), dtype=int)
            objects: list[tuple[int, int]] = []
            for color, total in counts_by_color.items():
                part_count = 2 if total < 7 or random.random() < 0.65 else 3
                for size in _split_count(total, part_count):
                    objects.append((color, size))
            random.shuffle(objects)

            if not all(_place_object(grid, _random_polyomino(size), color) for color, size in objects):
                continue
            try:
                out = _semantic_check(grid, counts_by_color)
            except AssertionError:
                continue
            signature = (tuple(map(int, out.shape)), out.tobytes())
            if signature in _seen_outputs:
                continue
            return grid, out
    return None


def generate_input() -> np.ndarray:
    """Generate a diverse, non-degenerate valid instance of the puzzle."""
    global _generation_index
    for _ in range(MAX_ATTEMPTS):
        generated = _gen_attempt(_generation_index)
        if generated is None:
            continue
        grid, out = generated
        _seen_outputs.add((tuple(map(int, out.shape)), out.tobytes()))
        _generation_index += 1
        return grid
    raise RuntimeError("failed to generate a valid balanced-frame input")
