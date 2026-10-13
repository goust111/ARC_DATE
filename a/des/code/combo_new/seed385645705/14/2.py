from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


MAX_ATTEMPTS = 300
MIN_BOARD_SIZE = 4
MAX_BOARD_SIZE = 6
REGION_COUNT_CHOICES = (4, 5)
PALETTE = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _other_color(colors: tuple[int, int], phase: int) -> int:
    a, b = colors
    if phase == a:
        return b
    if phase == b:
        return a
    raise ValueError("phase marker color must be one of the checkerboard colors")


def _expected_checkerboard(size: int, phase: int, other: int) -> np.ndarray:
    rr, cc = np.indices((size, size))
    return np.where((rr + cc) % 2 == 0, phase, other)


def _is_stripe_mask(mask: np.ndarray) -> bool:
    nonempty_rows = [tuple(bool(v) for v in row) for row in mask if np.any(row)]
    nonempty_cols = [tuple(bool(v) for v in col) for col in mask.T if np.any(col)]
    if not nonempty_rows or not nonempty_cols:
        return False
    return len(set(nonempty_rows)) == 1 and len(set(nonempty_cols)) == 1


def _is_region_candidate(
    grid: np.ndarray, bg: int, colors: tuple[int, int], marker_r: int, marker_c: int, size: int
) -> bool:
    h, w = grid.shape
    r0 = marker_r + 2
    c0 = marker_c + 2
    if size < MIN_BOARD_SIZE:
        return False
    if r0 + size >= h or c0 + size >= w:
        return False

    phase = int(grid[marker_r, marker_c])
    if phase not in colors:
        return False
    other = _other_color(colors, phase)

    if int(grid[r0, c0]) != phase:
        return False
    if np.any(grid[marker_r + 1, marker_c : c0 + size + 1] != bg):
        return False
    if np.any(grid[marker_r : r0 + size + 1, marker_c + 1] != bg):
        return False
    if np.any(grid[r0 + size, c0 : c0 + size] != bg):
        return False
    if np.any(grid[r0 : r0 + size, c0 + size] != bg):
        return False

    window = grid[r0 : r0 + size, c0 : c0 + size]
    if not np.isin(window, (bg, colors[0], colors[1])).all():
        return False

    visible = window != bg
    if not visible[0, 0]:
        return False
    if not visible[0].any() or not visible[:, 0].any():
        return False
    if not visible[-1].any() or not visible[:, -1].any():
        return False
    if not _is_stripe_mask(visible):
        return False

    expected = _expected_checkerboard(size, phase, other)
    if np.any(window[visible] != expected[visible]):
        return False

    return True


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _detect_regions(grid: np.ndarray, bg: int, colors: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    by_size: dict[int, list[tuple[int, int, int, int, tuple[int, int, int, int]]]] = defaultdict(list)

    for marker_r, marker_c in np.argwhere(grid != bg):
        marker_r = int(marker_r)
        marker_c = int(marker_c)
        if marker_r + 3 >= h or marker_c + 3 >= w:
            continue
        if marker_r > 0 and int(grid[marker_r - 1, marker_c]) != bg:
            continue
        if marker_c > 0 and int(grid[marker_r, marker_c - 1]) != bg:
            continue
        if int(grid[marker_r + 1, marker_c]) != bg or int(grid[marker_r, marker_c + 1]) != bg:
            continue
        if int(grid[marker_r + 1, marker_c + 1]) != bg:
            continue

        max_size = min(h - marker_r - 3, w - marker_c - 3)
        for size in range(MIN_BOARD_SIZE, max_size + 1):
            if not _is_region_candidate(grid, bg, colors, marker_r, marker_c, size):
                continue
            bbox = (marker_r, marker_c, marker_r + size + 2, marker_c + size + 2)
            by_size[size].append((marker_r, marker_c, size, int(grid[marker_r, marker_c]), bbox))

    scored: list[tuple[int, int, list[tuple[int, int, int, int]]]] = []
    for size, candidates in by_size.items():
        chosen: list[tuple[int, int, int, int]] = []
        chosen_boxes: list[tuple[int, int, int, int]] = []
        for marker_r, marker_c, _size, phase, bbox in sorted(candidates, key=lambda x: (x[0], x[1])):
            if any(_bbox_overlap(bbox, box) for box in chosen_boxes):
                continue
            chosen.append((marker_r, marker_c, size, phase))
            chosen_boxes.append(bbox)
        scored.append((len(chosen), size, chosen))

    assert scored, "failed to find any checkerboard regions"
    scored.sort(key=lambda item: (item[0], item[1]))
    count, _size, regions = scored[-1]
    assert count >= 2, "expected multiple regions sharing the same checkerboard rule"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """Restore each striped checkerboard square by using the shared two-color phase markers.

    Input structure:
    - The background is the most common color.
    - Several equal-size square checkerboards are scattered on the canvas. Each board uses the same two colors, but some whole rows and/or columns inside a board are replaced by background, creating missing stripes.
    - Every board has a single-pixel marker placed just above-left of it; the marker color is one of the two checkerboard colors and indicates the board's top-left phase.

    Rule:
    - Detect every square board and read its phase from the marker.
    - The visible cells already agree with a two-color checkerboard, and the background cells inside a board form whole missing rows and columns.
    - Repaint each board as a complete checkerboard of its original size, using the marker color for the top-left cell and alternating with the other shared color. Leave the markers and background elsewhere unchanged.

    Key insight:
    - All boards share the same two-color checker pattern; the only per-board difference is the phase marker, which tells whether the restored pattern should keep or swap the two colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = tuple(int(v) for v in np.unique(g) if int(v) != bg)
    assert len(colors) == 2, "expected exactly two non-background checkerboard colors"

    regions = _detect_regions(g, bg, colors)
    out = g.copy()
    for marker_r, marker_c, size, phase in regions:
        r0 = marker_r + 2
        c0 = marker_c + 2
        other = _other_color(colors, phase)
        out[r0 : r0 + size, c0 : c0 + size] = _expected_checkerboard(size, phase, other)
    return out


def _sample_missing_subset(indices: list[int], rng: random.Random) -> list[int]:
    max_size = min(2, len(indices))
    size = rng.randint(1, max_size)
    return sorted(rng.sample(indices, size))


def _sample_disjoint_nonempty_subsets(indices: list[int], rng: random.Random) -> tuple[list[int], list[int]]:
    shuffled = indices[:]
    rng.shuffle(shuffled)
    left = [shuffled[0]]
    right = [shuffled[1]]
    for value in shuffled[2:]:
        target = rng.choice((0, 1, 2))
        if target == 1 and len(left) < 2:
            left.append(value)
        elif target == 2 and len(right) < 2:
            right.append(value)
    return sorted(left), sorted(right)


def _choose_layout(count: int, size: int, rng: random.Random) -> list[tuple[int, int]] | None:
    options: list[tuple[list[tuple[int, int]], int, int]] = []
    for rows in range(1, count + 1):
        cols = (count + rows - 1) // rows
        for row_gap in range(0, 3):
            for col_gap in range(0, 3):
                for top in range(1, 4):
                    for left in range(1, 4):
                        for bottom in range(1, 4):
                            for right in range(1, 4):
                                row_pitch = size + 3 + row_gap
                                col_pitch = size + 3 + col_gap
                                height = top + (rows - 1) * row_pitch + size + 3 + bottom
                                width = left + (cols - 1) * col_pitch + size + 3 + right
                                if height > 30 or width > 30:
                                    continue
                                slots = [
                                    (top + rr * row_pitch, left + cc * col_pitch)
                                    for rr in range(rows)
                                    for cc in range(cols)
                                ]
                                options.append((slots, height, width))
    if not options:
        return None

    slots, _height, _width = rng.choice(options)
    chosen = rng.sample(slots, count)
    chosen.sort()
    return chosen


def _paint_region(
    grid: np.ndarray,
    marker_r: int,
    marker_c: int,
    size: int,
    phase: int,
    other: int,
    missing_rows: list[int],
    missing_cols: list[int],
) -> None:
    r0 = marker_r + 2
    c0 = marker_c + 2
    grid[marker_r, marker_c] = phase
    pattern = _expected_checkerboard(size, phase, other)
    for rr in range(size):
        for cc in range(size):
            if rr in missing_rows or cc in missing_cols:
                continue
            grid[r0 + rr, c0 + cc] = int(pattern[rr, cc])


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice(PALETTE)
    colors = tuple(rng.sample([c for c in PALETTE if c != bg], 2))

    size = rng.randint(MIN_BOARD_SIZE, MAX_BOARD_SIZE)
    interior = list(range(1, size - 1))
    if len(interior) < 2:
        return None

    count = rng.choice(REGION_COUNT_CHOICES)
    positions = _choose_layout(count, size, rng)
    if positions is None:
        return None

    row_miss_a, row_miss_b = _sample_disjoint_nonempty_subsets(interior, rng)
    col_miss_a, col_miss_b = _sample_disjoint_nonempty_subsets(interior, rng)
    phase_a, phase_b = colors if rng.random() < 0.5 else colors[::-1]

    specs = [
        {"missing_rows": row_miss_a, "missing_cols": [], "phase": phase_a},
        {"missing_rows": row_miss_b, "missing_cols": [], "phase": phase_b},
        {"missing_rows": [], "missing_cols": col_miss_a, "phase": phase_b},
        {"missing_rows": [], "missing_cols": col_miss_b, "phase": phase_a},
    ]
    if count == 5:
        mode = rng.choice(("rows", "cols", "both"))
        specs.append(
            {
                "missing_rows": _sample_missing_subset(interior, rng) if mode in ("rows", "both") else [],
                "missing_cols": _sample_missing_subset(interior, rng) if mode in ("cols", "both") else [],
                "phase": rng.choice(colors),
            }
        )
    rng.shuffle(specs)

    height = max(marker_r + size + 4 for marker_r, _marker_c in positions)
    width = max(marker_c + size + 4 for _marker_r, marker_c in positions)
    grid = np.full((height, width), bg, dtype=int)

    other_by_phase = {phase: _other_color(colors, phase) for phase in colors}
    for (marker_r, marker_c), spec in zip(positions, specs):
        _paint_region(
            grid,
            marker_r,
            marker_c,
            size,
            int(spec["phase"]),
            other_by_phase[int(spec["phase"])],
            list(spec["missing_rows"]),
            list(spec["missing_cols"]),
        )

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid checkerboard-stripe puzzle after {MAX_ATTEMPTS} attempts")
