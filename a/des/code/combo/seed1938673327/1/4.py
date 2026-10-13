from __future__ import annotations

import random

import numpy as np


BACKGROUND_COLOR = 7
GRAY_COLOR = 5
SEED_COLOR_POOL = (0, 1, 2, 3, 4, 6, 8, 9)

HEIGHT_RANGE = (6, 12)
WIDTH_RANGE = (8, 18)
ACTIVE_ROW_RANGE = (2, 6)
MAX_SEEDS_PER_ROW = 5
MAX_GENERATION_ATTEMPTS = 400


def _row_direction(cols: np.ndarray, width: int) -> int:
    midline = (width - 1) / 2.0
    leftmost = int(cols.min())
    rightmost = int(cols.max())
    if rightmost < midline:
        return 1
    if leftmost > midline:
        return -1
    raise AssertionError("foreground pixels in a non-empty row must lie strictly on one side of the midline")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each seed row into a same-row gradient band whose direction comes from the side of the midline.

    Input structure
    - The grid has an orange background.
    - Some rows contain one or more same-colored non-background pixels.
    - Every non-empty row keeps all of its foreground pixels entirely in the left half or entirely in the right half.

    Rule
    - Left-half rows extend to the right; right-half rows extend to the left.
    - Let N be the count of foreground pixels in the row.
    - Starting from the seed pixel nearest the midline, extend N new pixels in that direction.
    - Odd-numbered extension steps use the row's seed color; even-numbered steps use gray.
    - Original seed pixels remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, width = map(int, g.shape)
    bg = BACKGROUND_COLOR
    out = g.copy()

    for row in range(height):
        cols = np.flatnonzero(g[row] != bg)
        if cols.size == 0:
            continue

        colors = {int(v) for v in g[row, cols]}
        assert len(colors) == 1, "each non-empty row must use a single foreground color"
        seed_color = colors.pop()
        direction = _row_direction(cols, width)
        seed_count = int(cols.size)
        anchor = int(cols.max()) if direction == 1 else int(cols.min())

        targets = anchor + direction * np.arange(1, seed_count + 1)
        assert np.all((0 <= targets) & (targets < width)), "extension must remain inside the grid"
        assert np.all(g[row, targets] == bg), "extension cells must be background in the input"

        for step, col in enumerate(targets, start=1):
            out[row, int(col)] = seed_color if step % 2 == 1 else GRAY_COLOR

    return out


def _side_columns(width: int, side: str) -> list[int]:
    midline = (width - 1) / 2.0
    if side == "left":
        return [col for col in range(width) if col < midline]
    if side == "right":
        return [col for col in range(width) if col > midline]
    raise ValueError(f"unknown side: {side}")


def _feasible_seed_counts(width: int, side: str) -> list[int]:
    feasible = []
    for seed_count in range(1, MAX_SEEDS_PER_ROW + 1):
        if _sample_seed_columns(width, side, seed_count) is not None:
            feasible.append(seed_count)
    return feasible


def _sample_seed_columns(width: int, side: str, seed_count: int) -> list[int] | None:
    side_cols = _side_columns(width, side)
    if len(side_cols) < seed_count:
        return None

    if side == "left":
        anchors = [col for col in side_cols if col >= seed_count - 1 and col + seed_count < width]
        if not anchors:
            return None
        anchor = random.choice(anchors)
        others = random.sample(list(range(anchor)), seed_count - 1)
        cols = others + [anchor]
        return sorted(cols)

    anchors = [
        col
        for col in side_cols
        if col - seed_count >= 0 and (width - 1 - col) >= seed_count - 1
    ]
    if not anchors:
        return None
    anchor = random.choice(anchors)
    others = random.sample(list(range(anchor + 1, width)), seed_count - 1)
    cols = [anchor] + others
    return sorted(cols)


def _apply_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        anchor = max(spec["cols"]) if spec["side"] == "left" else min(spec["cols"])
        step = 1 if spec["side"] == "left" else -1
        for k in range(1, len(spec["cols"]) + 1):
            out[spec["row"], anchor + step * k] = spec["color"] if k % 2 == 1 else GRAY_COLOR
    return out


def _sample_specs(height: int, width: int) -> list[dict] | None:
    active_max = min(ACTIVE_ROW_RANGE[1], height - 1)
    if active_max < ACTIVE_ROW_RANGE[0]:
        return None

    feasible_left = _feasible_seed_counts(width, "left")
    feasible_right = _feasible_seed_counts(width, "right")
    odd_choices = [n for n in sorted(set(feasible_left) & set(feasible_right)) if n % 2 == 1]
    even_choices = [n for n in sorted(set(feasible_left) & set(feasible_right)) if n % 2 == 0]
    if not odd_choices or not even_choices:
        return None

    active_rows = random.randint(ACTIVE_ROW_RANGE[0], active_max)
    row_ids = sorted(random.sample(range(height), active_rows))

    left_count = active_rows // 2
    right_count = active_rows // 2
    if active_rows % 2 == 1:
        if random.random() < 0.5:
            left_count += 1
        else:
            right_count += 1

    sides = ["left"] * left_count + ["right"] * right_count
    random.shuffle(sides)

    seed_counts = [random.choice(odd_choices), random.choice(even_choices)]
    common_choices = sorted(set(feasible_left) & set(feasible_right))
    while len(seed_counts) < active_rows:
        seed_counts.append(random.choice(common_choices))
    random.shuffle(seed_counts)

    specs = []
    for row, side, seed_count in zip(row_ids, sides, seed_counts):
        cols = _sample_seed_columns(width, side, seed_count)
        if cols is None:
            return None
        specs.append(
            {
                "row": row,
                "side": side,
                "cols": cols,
                "color": random.choice(SEED_COLOR_POOL),
            }
        )
    return specs


def _render_input(height: int, width: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    for spec in specs:
        grid[spec["row"], spec["cols"]] = spec["color"]
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    specs = _sample_specs(height, width)
    if specs is None:
        return None

    grid = _render_input(height, width, specs)
    expected = _apply_specs(grid, specs)

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
    raise RuntimeError("failed to generate a valid input grid")
