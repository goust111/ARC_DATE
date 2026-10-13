import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PALETTE = tuple(c for c in range(10) if c not in (BLACK, GREY, MAROON))
MAX_GENERATION_ATTEMPTS = 200


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    rows = [r for r in range(h) if np.all(grid[r, :] == GREY)]
    cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert len(rows) == 1, "expected exactly one full grey row"
    assert len(cols) == 1, "expected exactly one full grey column"
    row = rows[0]
    col = cols[0]
    assert 0 < row < h - 1 and 0 < col < w - 1, "grey cross must split four nonempty quadrants"
    return row, col


def _dominant_quadrant_color(region: np.ndarray) -> int:
    values = region[(region != BLACK) & (region != GREY) & (region != MAROON)]
    assert values.size > 0, "quadrant must contain at least one payload color"
    counts = Counter(map(int, values))
    color, count = counts.most_common(1)[0]
    tied = [k for k, v in counts.items() if v == count]
    assert len(tied) == 1, "quadrant dominant color must be unique"
    return color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    split_r, split_c = _find_cross(g)
    quadrants = {
        "tl": g[:split_r, :split_c],
        "tr": g[:split_r, split_c + 1 :],
        "bl": g[split_r + 1 :, :split_c],
        "br": g[split_r + 1 :, split_c + 1 :],
    }
    colors = {name: _dominant_quadrant_color(region) for name, region in quadrants.items()}

    maroon_coords = np.argwhere(g == MAROON)
    assert len(maroon_coords) > 0, "expected maroon direction markers"
    rows = {int(r) for r, _ in maroon_coords}
    cols = {int(c) for _, c in maroon_coords}

    horizontal = len(rows) == 1
    vertical = len(cols) == 1
    assert horizontal != vertical, "maroon markers must encode exactly one orientation"

    if horizontal:
        return np.array([[colors["tl"], colors["tr"], colors["bl"], colors["br"]]], dtype=int)
    return np.array([[colors["tl"]], [colors["bl"]], [colors["tr"]], [colors["br"]]], dtype=int)


def _make_quadrant(
    height: int,
    width: int,
    main_color: int,
    noise_colors: list[int],
    reserved: set[tuple[int, int]],
) -> np.ndarray:
    area = height * width
    region = np.full((height, width), BLACK, dtype=int)
    free = [(r, c) for r in range(height) for c in range(width) if (r, c) not in reserved]
    if len(free) < 4:
        raise ValueError("quadrant too constrained")

    random.shuffle(free)
    max_noise_per_color = max(1, min(3, len(free) // 8))
    noise_counts = [random.randint(0, max_noise_per_color) for _ in noise_colors]
    total_noise = sum(noise_counts)

    min_main = max(total_noise + 2, area // 3)
    max_main = max(min_main, len(free) - max(1, len(free) // 5))
    if min_main > len(free):
        raise ValueError("not enough space for a dominant color")
    main_count = random.randint(min_main, min(max_main, len(free)))

    idx = 0
    for _ in range(main_count):
        r, c = free[idx]
        region[r, c] = main_color
        idx += 1

    for color, count in zip(noise_colors, noise_counts):
        for _ in range(count):
            if idx >= len(free):
                break
            r, c = free[idx]
            region[r, c] = color
            idx += 1

    return region


def _marker_positions(length: int) -> list[int]:
    if length < 3:
        raise ValueError("need at least three cells for a periodic marker set")
    choices = []
    for step in range(1, length):
        for start in range(step):
            pos = list(range(start, length, step))
            if len(pos) >= 3:
                choices.append(pos)
    if not choices:
        raise ValueError("no valid periodic marker positions")
    return random.choice(choices)


def _gen_attempt() -> np.ndarray | None:
    top = random.randint(3, 8)
    bottom = random.randint(3, 8)
    left = random.randint(3, 8)
    right = random.randint(3, 8)
    h = top + 1 + bottom
    w = left + 1 + right

    grid = np.full((h, w), BLACK, dtype=int)
    split_r = top
    split_c = left
    grid[split_r, :] = GREY
    grid[:, split_c] = GREY

    main_colors = random.sample(PALETTE, 4)
    orientation = random.choice(["horizontal", "vertical"])

    quadrant_reserved = {
        "tl": set(),
        "tr": set(),
        "bl": set(),
        "br": set(),
    }

    if orientation == "horizontal":
        marker_row = random.choice([r for r in range(h) if r != split_r])
        positions = _marker_positions(w - 1)
        actual_cols = [p if p < split_c else p + 1 for p in positions]
        for col in actual_cols:
            grid[marker_row, col] = MAROON
            if marker_row < split_r and col < split_c:
                quadrant_reserved["tl"].add((marker_row, col))
            elif marker_row < split_r:
                quadrant_reserved["tr"].add((marker_row, col - split_c - 1))
            elif col < split_c:
                quadrant_reserved["bl"].add((marker_row - split_r - 1, col))
            else:
                quadrant_reserved["br"].add((marker_row - split_r - 1, col - split_c - 1))
    else:
        marker_col = random.choice([c for c in range(w) if c != split_c])
        positions = _marker_positions(h - 1)
        actual_rows = [p if p < split_r else p + 1 for p in positions]
        for row in actual_rows:
            grid[row, marker_col] = MAROON
            if row < split_r and marker_col < split_c:
                quadrant_reserved["tl"].add((row, marker_col))
            elif row < split_r:
                quadrant_reserved["tr"].add((row, marker_col - split_c - 1))
            elif marker_col < split_c:
                quadrant_reserved["bl"].add((row - split_r - 1, marker_col))
            else:
                quadrant_reserved["br"].add((row - split_r - 1, marker_col - split_c - 1))

    other_colors = [c for c in PALETTE if c not in main_colors]
    random.shuffle(other_colors)
    quadrants = {
        "tl": _make_quadrant(top, left, main_colors[0], other_colors[:2], quadrant_reserved["tl"]),
        "tr": _make_quadrant(top, right, main_colors[1], other_colors[1:4], quadrant_reserved["tr"]),
        "bl": _make_quadrant(bottom, left, main_colors[2], other_colors[2:5], quadrant_reserved["bl"]),
        "br": _make_quadrant(bottom, right, main_colors[3], other_colors[:3], quadrant_reserved["br"]),
    }

    grid[:split_r, :split_c] = quadrants["tl"]
    grid[:split_r, split_c + 1 :] = quadrants["tr"]
    grid[split_r + 1 :, :split_c] = quadrants["bl"]
    grid[split_r + 1 :, split_c + 1 :] = quadrants["br"]

    if orientation == "horizontal":
        grid[marker_row, actual_cols] = MAROON
    else:
        grid[actual_rows, marker_col] = MAROON

    try:
        out = main(grid)
    except AssertionError:
        return None

    expected_shape = (1, 4) if orientation == "horizontal" else (4, 1)
    if out.shape != expected_shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
