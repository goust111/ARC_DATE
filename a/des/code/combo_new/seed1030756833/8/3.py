from __future__ import annotations

from collections import Counter
import random

import numpy as np


COLOR_COUNT = 10
RECT_HEIGHT_RANGE = (4, 6)
RECT_WIDTH_RANGE = (4, 6)
OBJECT_COUNT_RANGE = (3, 6)
GRID_LIMIT = 30
GEN_RETRIES = 500
PLACE_RETRIES = 800
TIE_BRANCH_PROB = 0.5


def _most_common_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    max_count = int(counts.max())
    assert int(np.count_nonzero(counts == max_count)) == 1, "background must be uniquely most common"
    return int(vals[int(np.argmax(counts))])


def _block_records(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue

        coords = np.argwhere(g == color)
        assert coords.size, "foreground color must have pixels"
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()

        seen = set(map(int, np.unique(patch)))
        assert seen.issubset({bg, color}), "each rectangle bbox may contain only its color and background holes"

        top = patch[0, :]
        bottom = patch[-1, :]
        left = patch[:, 0]
        right = patch[:, -1]
        assert np.all(top == color) and np.all(bottom == color), "rectangle border must remain visible"
        assert np.all(left == color) and np.all(right == color), "rectangle border must remain visible"

        records.append(
            {
                "color": color,
                "hole_count": int(np.count_nonzero(patch == bg)),
                "pos": (r0, c0),
                "patch": patch,
            }
        )

    assert records, "expected at least one foreground rectangle"
    shapes = {tuple(map(int, rec["patch"].shape)) for rec in records}
    assert len(shapes) == 1, "all rectangles must have the same size"
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort holed color rectangles by their number of background holes and pack them into one row.

    Input structure:
    - The uniquely most common color is the background.
    - Several separated same-size foreground rectangles appear on the background.
    - Each rectangle has a unique foreground color and a visible solid border.
    - Background-colored pixels inside a rectangle are holes in that rectangle.

    Rule:
    - For each foreground color, take its bounding box as the rectangle patch.
    - Count the background pixels inside that patch.
    - Sort rectangles from fewest holes to most holes; ties are broken by the
      rectangle's row-major input position.
    - Concatenate the sorted rectangle patches left-to-right with no gap.

    Key insight:
    - The colors only identify whole rectangles; the ordering signal is the
      counted number of internal background holes, followed by position for ties.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    records = _block_records(g, bg)
    records.sort(key=lambda rec: (int(rec["hole_count"]), tuple(rec["pos"])))
    return np.concatenate([rec["patch"] for rec in records], axis=1)


def _patch(color: int, bg: int, height: int, width: int, hole_count: int) -> np.ndarray:
    patch = np.full((height, width), int(color), dtype=int)
    inner = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    assert 0 <= hole_count <= len(inner), "hole count must fit inside the visible border"
    for r, c in random.sample(inner, hole_count):
        patch[r, c] = int(bg)
    return patch


def _hole_counts(count: int, max_holes: int) -> list[int]:
    values = list(range(max_holes + 1))
    if random.random() < TIE_BRANCH_PROB and count >= 3:
        duplicate = random.choice(values)
        others = [v for v in values if v != duplicate]
        random.shuffle(others)
        result = [duplicate, duplicate] + others[: count - 2]
        random.shuffle(result)
        return result

    result = random.sample(values, count)
    random.shuffle(result)
    return result


def _layout_shape(rect_h: int, rect_w: int, count: int) -> tuple[int, int] | None:
    options = []
    for rows in range(1, count + 1):
        cols = (count + rows - 1) // rows
        min_h = rows * rect_h + rows + 1
        min_w = cols * rect_w + cols + 1
        if min_h <= GRID_LIMIT and min_w <= GRID_LIMIT:
            options.append((min_h, min_w))
    if not options:
        return None

    min_h, min_w = random.choice(options)
    return random.randint(min_h, GRID_LIMIT), random.randint(min_w, GRID_LIMIT)


def _place_patches(
    grid: np.ndarray,
    patches: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    height, width = map(int, grid.shape)
    rect_h, rect_w = map(int, patches[0].shape)
    reserved = np.zeros((height, width), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    order = list(range(len(patches)))
    random.shuffle(order)

    for idx in order:
        placed = False
        for _ in range(PLACE_RETRIES):
            r0 = random.randint(1, height - rect_h - 1)
            c0 = random.randint(1, width - rect_w - 1)
            rr0, rr1 = r0 - 1, r0 + rect_h + 1
            cc0, cc1 = c0 - 1, c0 + rect_w + 1
            if bool(reserved[rr0:rr1, cc0:cc1].any()):
                continue

            grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = patches[idx]
            reserved[rr0:rr1, cc0:cc1] = True
            positions[idx] = (int(r0), int(c0))
            placed = True
            break
        if not placed:
            return None

    assert all(pos is not None for pos in positions), "all patches must be placed"
    return [pos for pos in positions if pos is not None]


def _expected_from_specs(
    patches: list[np.ndarray],
    positions: list[tuple[int, int]],
    hole_counts: list[int],
) -> np.ndarray:
    order = sorted(range(len(patches)), key=lambda i: (int(hole_counts[i]), positions[i]))
    return np.concatenate([patches[i] for i in order], axis=1)


def _input_position_order(positions: list[tuple[int, int]]) -> list[int]:
    return sorted(range(len(positions)), key=lambda i: positions[i])


def _gen_attempt() -> np.ndarray | None:
    rect_h = random.randint(*RECT_HEIGHT_RANGE)
    rect_w = random.randint(*RECT_WIDTH_RANGE)
    max_holes = (rect_h - 2) * (rect_w - 2)
    max_count = min(OBJECT_COUNT_RANGE[1], GRID_LIMIT // rect_w, max_holes + 1, COLOR_COUNT - 1)
    if max_count < OBJECT_COUNT_RANGE[0]:
        return None
    count = random.randint(OBJECT_COUNT_RANGE[0], max_count)

    shape = _layout_shape(rect_h, rect_w, count)
    if shape is None:
        return None
    height, width = shape

    bg = random.randrange(COLOR_COUNT)
    colors = random.sample([c for c in range(COLOR_COUNT) if c != bg], count)
    hole_counts = _hole_counts(count, max_holes)
    patches = [_patch(color, bg, rect_h, rect_w, holes) for color, holes in zip(colors, hole_counts)]

    grid = np.full((height, width), int(bg), dtype=int)
    positions = _place_patches(grid, patches)
    if positions is None:
        return None

    expected = _expected_from_specs(patches, positions, hole_counts)
    rule_order = sorted(range(count), key=lambda i: (int(hole_counts[i]), positions[i]))
    if rule_order == _input_position_order(positions):
        return None

    counts = Counter(map(int, grid.ravel()))
    if counts[bg] <= max(n for color, n in counts.items() if color != bg):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed1030756833/8/3 input")
