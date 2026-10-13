from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
RED = 2
TEAL = 8
SPRITE_SIZE = 4
MACRO_SIDE = 3
PATTERN_SIZE = SPRITE_SIZE * MACRO_SIDE
PALETTE = (1, 3, 4, 6, 7, 9)
RNG = np.random.default_rng()
GEN_RETRY_LIMIT = 400
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def main(grid: np.ndarray) -> np.ndarray:
    """Count each color's singleton markers, rotate the control sprite by that count, and stamp it into matching macro-cells.

    Input structure:
    - The background is black.
    - One upper-left 4x4 control sprite is completely filled with red and teal pixels, with all four quarter-turns distinct.
    - One upper-right 3x3 coarse template is made of 4x4 macro-cells whose colors are black or 2-3 non-reserved colors.
    - The lower half contains isolated singleton markers in those same template colors; marker positions are not otherwise meaningful.

    Rule:
    - For each template color, count its singleton markers and reduce the count modulo 4.
    - Rotate the 4x4 control sprite counterclockwise by 90 degrees times that count.
    - Replace every macro-cell of that color with the rotated sprite, keeping teal pixels teal and recoloring red pixels to the macro-cell's color.
    - Leave black macro-cells black, preserve the original control sprite in place, and clear all marker pixels to background.

    Key insight:
    - The singleton colors identify which template cells to rewrite, while the singleton counts identify the rotation of a separate reusable sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    control_bbox, control = _find_control(g)
    pattern_colors = _pattern_colors(g)
    assert 2 <= len(pattern_colors) <= 3, "expected 2-3 template colors"

    marker_counts, pattern_bbox = _find_pattern_and_markers(g, pattern_colors)
    macro_grid = _read_macro_grid(g, pattern_bbox, control.shape[0], pattern_colors)

    return _render_output(g.shape, control_bbox, control, pattern_bbox, macro_grid, marker_counts)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        built = _generate_attempt()
        if built is not None:
            return built
    raise RuntimeError("failed to generate a valid input")


def _generate_attempt() -> np.ndarray | None:
    color_count = int(RNG.choice([2, 3], p=[0.45, 0.55]))
    colors = tuple(int(c) for c in RNG.choice(PALETTE, size=color_count, replace=False))
    marker_counts = _sample_marker_counts(color_count)
    count_by_color = {color: count for color, count in zip(colors, marker_counts, strict=True)}

    sprite = _sample_sprite()
    macro_grid = _sample_macro_grid(colors)
    if macro_grid is None:
        return None

    top_pad = int(RNG.integers(1, 4))
    left_pad = int(RNG.integers(1, 4))
    gap = int(RNG.integers(2, 6))
    right_pad = int(RNG.integers(1, 5))
    marker_gap = int(RNG.integers(2, 5))
    marker_band = int(RNG.integers(4, 8))
    bottom_pad = int(RNG.integers(1, 4))

    control_r0 = top_pad
    control_c0 = left_pad
    pattern_r0 = top_pad
    pattern_c0 = control_c0 + SPRITE_SIZE + gap

    height = pattern_r0 + PATTERN_SIZE + marker_gap + marker_band + bottom_pad
    width = pattern_c0 + PATTERN_SIZE + right_pad
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    _paste(grid, sprite, control_r0, control_c0)
    _paint_macro_template(grid, macro_grid, pattern_r0, pattern_c0)

    occupied = {(r, c) for r in range(height) for c in range(width) if grid[r, c] != BG}
    start_row = pattern_r0 + PATTERN_SIZE + marker_gap
    if not _place_markers(grid, occupied, colors, count_by_color, start_row):
        return None

    control_bbox = (control_r0, control_c0, control_r0 + SPRITE_SIZE - 1, control_c0 + SPRITE_SIZE - 1)
    pattern_bbox = (pattern_r0, pattern_c0, pattern_r0 + PATTERN_SIZE - 1, pattern_c0 + PATTERN_SIZE - 1)
    expected = _render_output(grid.shape, control_bbox, sprite, pattern_bbox, macro_grid, count_by_color)

    if not np.array_equal(main(grid), expected):
        return None
    if np.array_equal(grid, expected):
        return None
    if not _semantic_checks(grid, expected, control_bbox, sprite, pattern_bbox, macro_grid, count_by_color):
        return None
    return grid


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    control_bbox: tuple[int, int, int, int],
    sprite: np.ndarray,
    pattern_bbox: tuple[int, int, int, int],
    macro_grid: np.ndarray,
    count_by_color: dict[int, int],
) -> bool:
    rotations = {np.rot90(sprite, k).tobytes() for k in range(4)}
    if len(rotations) != 4:
        return False

    colors = [int(color) for color in np.unique(macro_grid) if int(color) != BG]
    counts = [count_by_color[color] % 4 for color in colors]
    if len(set(counts)) != len(counts):
        return False
    if len({count for count in counts if count != 0}) == 0:
        return False
    if not any(count == 0 for count in counts):
        # Bias toward exposing the zero-rotation branch in examples.
        if RNG.random() < 0.5:
            return False

    for color in colors:
        if not np.any(macro_grid == color):
            return False
        alt_counts = dict(count_by_color)
        alt_counts[color] = (alt_counts[color] + 1) % 4
        alt = _render_output(grid.shape, control_bbox, sprite, pattern_bbox, macro_grid, alt_counts)
        if np.array_equal(alt, expected):
            return False

    unrotated = {color: 0 for color in colors}
    if all(count_by_color[color] % 4 == 0 for color in colors):
        return False
    if np.array_equal(_render_output(grid.shape, control_bbox, sprite, pattern_bbox, macro_grid, unrotated), expected):
        return False

    return True


def _sample_marker_counts(color_count: int) -> list[int]:
    counts_pool = np.array([0, 1, 2, 3], dtype=int)
    for _ in range(200):
        if color_count == 2 and RNG.random() < 0.65:
            chosen = [0, int(RNG.choice([1, 2, 3]))]
            RNG.shuffle(chosen)
            return chosen
        chosen = RNG.choice(counts_pool, size=color_count, replace=False)
        if color_count == 3 and 0 not in chosen and RNG.random() < 0.7:
            continue
        return [int(x) for x in chosen]
    return [int(x) for x in RNG.choice(counts_pool, size=color_count, replace=False)]


def _sample_sprite() -> np.ndarray:
    for _ in range(400):
        teal_total = int(RNG.integers(6, 11))
        mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
        mask.flat[RNG.choice(SPRITE_SIZE * SPRITE_SIZE, size=teal_total, replace=False)] = True
        sprite = np.where(mask, TEAL, RED)
        if len({np.rot90(sprite, k).tobytes() for k in range(4)}) != 4:
            continue
        if len(np.unique(sprite, axis=0)) < 3:
            continue
        if len(np.unique(sprite, axis=1)) < 3:
            continue
        return sprite.astype(int)
    raise RuntimeError("failed to sample a non-symmetric sprite")


def _sample_macro_grid(colors: tuple[int, ...]) -> np.ndarray | None:
    cell_positions = [(r, c) for r in range(MACRO_SIDE) for c in range(MACRO_SIDE)]
    for _ in range(400):
        counts = RNG.integers(1, 4, size=len(colors))
        occupied_total = int(counts.sum())
        if occupied_total < 4 or occupied_total > 9:
            continue
        if len(colors) == 2 and occupied_total > 6:
            continue
        if len(colors) == 3 and occupied_total == 9 and RNG.random() < 0.5:
            continue

        chosen_indices = RNG.choice(len(cell_positions), size=occupied_total, replace=False)
        occupied = [cell_positions[int(i)] for i in chosen_indices]
        rows = {r for r, _ in occupied}
        cols = {c for _, c in occupied}
        if rows != {0, 1, 2} and (0 not in rows or 2 not in rows):
            continue
        if cols != {0, 1, 2} and (0 not in cols or 2 not in cols):
            continue
        if 0 not in rows or 2 not in rows or 0 not in cols or 2 not in cols:
            continue

        assignment = np.zeros((MACRO_SIDE, MACRO_SIDE), dtype=int)
        bag: list[int] = []
        for color, count in zip(colors, counts, strict=True):
            bag.extend([int(color)] * int(count))
        RNG.shuffle(bag)
        for (r, c), color in zip(occupied, bag, strict=True):
            assignment[r, c] = int(color)

        if len(set(int(x) for x in bag)) != len(colors):
            continue
        if len({tuple(row) for row in assignment.tolist()}) < 2:
            continue
        return assignment
    return None


def _paint_macro_template(grid: np.ndarray, macro_grid: np.ndarray, r0: int, c0: int) -> None:
    for mr in range(MACRO_SIDE):
        for mc in range(MACRO_SIDE):
            color = int(macro_grid[mr, mc])
            if color == BG:
                continue
            grid[
                r0 + mr * SPRITE_SIZE : r0 + (mr + 1) * SPRITE_SIZE,
                c0 + mc * SPRITE_SIZE : c0 + (mc + 1) * SPRITE_SIZE,
            ] = color


def _place_markers(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    colors: tuple[int, ...],
    count_by_color: dict[int, int],
    start_row: int,
) -> bool:
    h, w = grid.shape
    blocked = set(occupied)
    for r, c in list(occupied):
        for dr, dc in NEIGHBORS8:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                blocked.add((rr, cc))

    requests = [(color, idx) for color in colors for idx in range(count_by_color[color])]
    RNG.shuffle(requests)
    for color, _ in requests:
        candidates = []
        for r in range(start_row, h - 1):
            for c in range(1, w - 1):
                if (r, c) in blocked:
                    continue
                candidates.append((r, c))
        if not candidates:
            return False
        pick = candidates[int(RNG.integers(len(candidates)))]
        pr, pc = pick
        grid[pr, pc] = int(color)
        blocked.add(pick)
        for dr, dc in NEIGHBORS8:
            rr = pr + dr
            cc = pc + dc
            if 0 <= rr < h and 0 <= cc < w:
                blocked.add((rr, cc))
    return True


def _find_control(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    mask = (grid == RED) | (grid == TEAL)
    candidates = []
    for coords in _components(mask, connectivity=4):
        bbox = _bbox(coords)
        patch = _crop(grid, bbox)
        if patch.shape != (SPRITE_SIZE, SPRITE_SIZE):
            continue
        if coords.shape[0] != SPRITE_SIZE * SPRITE_SIZE:
            continue
        colors = set(int(v) for v in np.unique(patch))
        if colors != {RED, TEAL}:
            continue
        candidates.append((bbox, patch))
    assert len(candidates) == 1, f"expected one control sprite, got {len(candidates)}"
    return candidates[0]


def _pattern_colors(grid: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) not in {BG, RED, TEAL})


def _find_pattern_and_markers(
    grid: np.ndarray,
    pattern_colors: list[int],
) -> tuple[dict[int, int], tuple[int, int, int, int]]:
    marker_counts = {int(color): 0 for color in pattern_colors}
    pattern_cells: list[tuple[int, int]] = []
    for color in pattern_colors:
        for coords in _components(grid == int(color), connectivity=4):
            if coords.shape[0] == 1:
                marker_counts[int(color)] += 1
            else:
                pattern_cells.extend((int(r), int(c)) for r, c in coords)
    assert pattern_cells, "expected coarse template cells"
    pattern_bbox = _bbox(np.array(pattern_cells, dtype=int))
    return marker_counts, pattern_bbox


def _read_macro_grid(
    grid: np.ndarray,
    pattern_bbox: tuple[int, int, int, int],
    cell_size: int,
    pattern_colors: list[int],
) -> np.ndarray:
    pattern = _crop(grid, pattern_bbox)
    assert pattern.shape == (cell_size * MACRO_SIDE, cell_size * MACRO_SIDE), pattern.shape
    allowed = set(pattern_colors) | {BG}
    macro = np.zeros((MACRO_SIDE, MACRO_SIDE), dtype=int)
    for mr in range(MACRO_SIDE):
        for mc in range(MACRO_SIDE):
            cell = pattern[
                mr * cell_size : (mr + 1) * cell_size,
                mc * cell_size : (mc + 1) * cell_size,
            ]
            values = np.unique(cell)
            assert values.size == 1, f"macro-cell {(mr, mc)} is not uniform"
            color = int(values[0])
            assert color in allowed, f"unexpected macro color {color}"
            macro[mr, mc] = color
    return macro


def _render_output(
    shape: tuple[int, int],
    control_bbox: tuple[int, int, int, int],
    control: np.ndarray,
    pattern_bbox: tuple[int, int, int, int],
    macro_grid: np.ndarray,
    marker_counts: dict[int, int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    control_r0, control_c0, _, _ = control_bbox
    _paste(out, control, control_r0, control_c0)

    pattern_r0, pattern_c0, _, _ = pattern_bbox
    for mr in range(MACRO_SIDE):
        for mc in range(MACRO_SIDE):
            color = int(macro_grid[mr, mc])
            if color == BG:
                continue
            turns = int(marker_counts[color]) % 4
            rotated = np.rot90(control, turns)
            stamp = np.where(rotated == TEAL, TEAL, color).astype(int)
            _paste(out, stamp, pattern_r0 + mr * SPRITE_SIZE, pattern_c0 + mc * SPRITE_SIZE)
    return out


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neighbors = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in neighbors:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch

