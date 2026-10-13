from __future__ import annotations

import random

import numpy as np


BG = 0
QUAD_COLORS = (1, 2, 3, 4)
CROSS = 6
MARKER = 7

QUAD_SIZE = 8
GRID_SIZE = 2 * QUAD_SIZE + 1
MINI_SIZE = QUAD_SIZE // 2
CENTER_OFFSET = 2
MAX_GENERATION_ATTEMPTS = 400

QUADRANT_SLICES = (
    (slice(0, QUAD_SIZE), slice(0, QUAD_SIZE)),
    (slice(0, QUAD_SIZE), slice(QUAD_SIZE + 1, GRID_SIZE)),
    (slice(QUAD_SIZE + 1, GRID_SIZE), slice(0, QUAD_SIZE)),
    (slice(QUAD_SIZE + 1, GRID_SIZE), slice(QUAD_SIZE + 1, GRID_SIZE)),
)


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < QUAD_SIZE and 0 <= nc < QUAD_SIZE:
            out.append((nr, nc))
    return out


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _in_center(cell: tuple[int, int]) -> bool:
    r, c = cell
    return CENTER_OFFSET <= r < CENTER_OFFSET + MINI_SIZE and CENTER_OFFSET <= c < CENTER_OFFSET + MINI_SIZE


def _micro_neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < MINI_SIZE and 0 <= nc < MINI_SIZE:
            out.append((nr, nc))
    return out


def _block_cells(micro: tuple[int, int]) -> set[tuple[int, int]]:
    r0 = 2 * micro[0]
    c0 = 2 * micro[1]
    return {(r0 + dr, c0 + dc) for dr in (0, 1) for dc in (0, 1)}


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    path = [start]
    r, c = start
    while (r, c) != end:
        options: list[tuple[int, int]] = []
        if r < end[0]:
            options.append((r + 1, c))
        elif r > end[0]:
            options.append((r - 1, c))
        if c < end[1]:
            options.append((r, c + 1))
        elif c > end[1]:
            options.append((r, c - 1))
        r, c = random.choice(options)
        path.append((r, c))
    return path


def _sample_growth_box(points: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _bbox(points)
    min_h = r1 - r0 + 1
    min_w = c1 - c0 + 1
    box_h = random.randint(min_h, 6)
    box_w = random.randint(min_w, 6)
    row_lo = max(0, r1 - box_h + 1)
    row_hi = min(r0, QUAD_SIZE - box_h)
    col_lo = max(0, c1 - box_w + 1)
    col_hi = min(c0, QUAD_SIZE - box_w)
    box_r0 = random.randint(row_lo, row_hi)
    box_c0 = random.randint(col_lo, col_hi)
    return box_r0, box_c0, box_h, box_w


def _grow_connected_shape(
    seed: set[tuple[int, int]],
    target_area: int,
    forbidden: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    if forbidden is None:
        forbidden = set()
    if seed & forbidden:
        return None

    box_r0, box_c0, box_h, box_w = _sample_growth_box(seed)
    box_cells = {
        (r, c)
        for r in range(box_r0, box_r0 + box_h)
        for c in range(box_c0, box_c0 + box_w)
        if (r, c) not in forbidden
    }
    if not seed <= box_cells:
        return None

    cells = set(seed)
    while len(cells) < target_area:
        frontier = {
            nbr
            for cell in cells
            for nbr in _neighbors4(cell)
            if nbr in box_cells and nbr not in cells
        }
        if not frontier:
            return None
        cells.add(random.choice(sorted(frontier)))

    br0, bc0, br1, bc1 = _bbox(cells)
    if br1 - br0 + 1 > 6 or bc1 - bc0 + 1 > 6:
        return None
    return cells


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((QUAD_SIZE, QUAD_SIZE), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _main_overlap_anchor(center_focus: tuple[int, int]) -> tuple[int, int]:
    r, c = center_focus
    return random.choice(((1, c), (6, c), (r, 1), (r, 6)))


def _outer_neighbor_options(cell: tuple[int, int]) -> list[tuple[int, int]]:
    return [nbr for nbr in _neighbors4(cell) if not _in_center(nbr)]


def _make_main_shape(
    center_focus: tuple[int, int],
    overlap_focus: tuple[int, int],
    unique_anchor: tuple[int, int],
) -> np.ndarray | None:
    seed = set(_random_path(unique_anchor, overlap_focus))
    seed.update(_random_path(overlap_focus, center_focus))
    if len(seed) > 10:
        return None
    target_area = random.randint(max(4, len(seed)), 10)
    cells = _grow_connected_shape(seed, target_area)
    if cells is None:
        return None
    return _cells_to_mask(cells)


def _secondary_seed(
    shared_micro: tuple[int, int],
    unique_micro: tuple[int, int],
    use_top_left: bool,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    sr, sc = 2 * shared_micro[0], 2 * shared_micro[1]
    ur, uc = 2 * unique_micro[0], 2 * unique_micro[1]
    dr = unique_micro[0] - shared_micro[0]
    dc = unique_micro[1] - shared_micro[1]
    shared_top_left = (sr, sc)

    if abs(dr) + abs(dc) != 1:
        return None

    if dr == 0 and dc == 1:
        if use_top_left:
            seed = {(sr, sc), (sr, sc + 1), (ur, uc)}
            return seed, set()
        seed = {(sr + 1, sc + 1), (ur + 1, uc)}
        return seed, {shared_top_left}
    if dr == 0 and dc == -1:
        if use_top_left:
            seed = {(sr, sc), (ur, uc + 1)}
            return seed, set()
        seed = {(sr + 1, sc), (ur + 1, uc + 1)}
        return seed, {shared_top_left}
    if dr == 1 and dc == 0:
        if use_top_left:
            seed = {(sr, sc), (sr + 1, sc), (ur, uc)}
            return seed, set()
        seed = {(sr + 1, sc + 1), (ur, uc + 1)}
        return seed, {shared_top_left}
    if dr == -1 and dc == 0:
        if use_top_left:
            seed = {(sr, sc), (ur + 1, uc)}
            return seed, set()
        seed = {(sr, sc + 1), (ur + 1, uc + 1)}
        return seed, {shared_top_left}
    return None


def _make_secondary_shape(
    shared_micro: tuple[int, int],
    unique_micro: tuple[int, int],
    use_top_left: bool,
    blocked_micro: tuple[int, int],
) -> np.ndarray | None:
    seed_info = _secondary_seed(shared_micro, unique_micro, use_top_left)
    if seed_info is None:
        return None
    seed, forbidden = seed_info
    forbidden = set(forbidden)
    forbidden.update(_block_cells(blocked_micro))
    if not use_top_left:
        forbidden.discard(next(iter(seed)))
    target_area = random.randint(max(4, len(seed)), 10)
    cells = _grow_connected_shape(seed, target_area, forbidden=forbidden)
    if cells is None:
        return None
    return _cells_to_mask(cells)


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    rows = np.where(np.all(grid == CROSS, axis=1))[0]
    cols = np.where(np.all(grid == CROSS, axis=0))[0]
    assert len(rows) == 1 and len(cols) == 1, "expected one full magenta separator row and column"
    cross_r = int(rows[0])
    cross_c = int(cols[0])
    assert cross_r == QUAD_SIZE and cross_c == QUAD_SIZE, "separator cross must split four 8x8 quadrants"
    return cross_r, cross_c


def _shrink_panel(panel: np.ndarray) -> np.ndarray:
    mini = np.zeros((MINI_SIZE, MINI_SIZE), dtype=int)
    for mr in range(MINI_SIZE):
        for mc in range(MINI_SIZE):
            block = panel[2 * mr : 2 * mr + 2, 2 * mc : 2 * mc + 2]
            if int(block[0, 0]) != BG:
                mini[mr, mc] = int(block[0, 0])
                continue
            nonzero = block[block != BG]
            if nonzero.size:
                mini[mr, mc] = int(nonzero[0])
    return mini


def _shrink_mask(mask: np.ndarray) -> np.ndarray:
    mini = np.zeros((MINI_SIZE, MINI_SIZE), dtype=bool)
    for mr in range(MINI_SIZE):
        for mc in range(MINI_SIZE):
            block = mask[2 * mr : 2 * mr + 2, 2 * mc : 2 * mc + 2]
            mini[mr, mc] = bool(block[0, 0]) or bool(block.any())
    return mini


def _compose_output(masks: list[np.ndarray], main_idxs: tuple[int, int], colors: list[int]) -> tuple[np.ndarray, np.ndarray]:
    base = np.zeros((QUAD_SIZE, QUAD_SIZE), dtype=int)
    for idx in sorted(main_idxs):
        base[masks[idx]] = colors[idx]

    final = base.copy()
    center = final[CENTER_OFFSET : CENTER_OFFSET + MINI_SIZE, CENTER_OFFSET : CENTER_OFFSET + MINI_SIZE]
    secondary_idxs = tuple(idx for idx in range(4) if idx not in main_idxs)
    for idx in sorted(secondary_idxs):
        mini_mask = _shrink_mask(masks[idx])
        center[mini_mask] = colors[idx]
    return base, final


def _marker_position(mask: np.ndarray) -> tuple[int, int] | None:
    shape_cells = np.argwhere(mask)
    candidates: list[tuple[int, int, int]] = []
    for r in range(QUAD_SIZE):
        for c in range(QUAD_SIZE):
            if mask[r, c]:
                continue
            distance = min(abs(r - int(sr)) + abs(c - int(sc)) for sr, sc in shape_cells)
            candidates.append((distance, r, c))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    top = candidates[: min(8, len(candidates))]
    _, r, c = random.choice(top)
    return r, c


def _build_input_grid(masks: list[np.ndarray], main_idxs: tuple[int, int], colors: list[int]) -> np.ndarray | None:
    grid = np.full((GRID_SIZE, GRID_SIZE), CROSS, dtype=int)
    for idx, (rows, cols) in enumerate(QUADRANT_SLICES):
        panel = np.zeros((QUAD_SIZE, QUAD_SIZE), dtype=int)
        panel[masks[idx]] = colors[idx]
        if idx in main_idxs:
            marker = _marker_position(masks[idx])
            if marker is None:
                return None
            panel[marker] = MARKER
        grid[rows, cols] = panel
    return grid


def _quadrant_panels(grid: np.ndarray) -> list[np.ndarray]:
    _find_cross(grid)
    return [grid[rows, cols].copy() for rows, cols in QUADRANT_SLICES]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the marked quadrants at full size, then embed centered miniatures from the unmarked quadrants.

    Input structure:
    - A 17x17 grid is split by a full magenta row and column into four 8x8 quadrants.
    - Each quadrant contains a black background and one connected monochrome shape in a distinct color from blue, red, green, yellow.
    - Exactly two diagonally opposite quadrants also contain one orange marker pixel on background cells; the marker is not part of the shape.

    Rule:
    - Remove the magenta separator and erase the orange markers back to black.
    - Treat the two marked quadrants as full-size layers and overlay them on one 8x8 canvas in quadrant-index order TL, TR, BL, BR, where later quadrants overwrite earlier ones on conflicts.
    - Shrink each unmarked quadrant from 8x8 to 4x4 by scanning each 2x2 block: keep the top-left pixel if it is non-black, otherwise keep any non-black pixel from that block if one exists, otherwise keep black; then overlay both miniatures onto the center 4x4 of the canvas with the same quadrant-index priority.

    Key insight:
    - The orange markers are only selectors: they decide which two quadrants stay full-size and which two are compressed, while all overwrite priority comes from the original quadrant positions rather than from color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 17x17 grid"

    panels = _quadrant_panels(g)
    cleaned: list[np.ndarray] = []
    marker_idxs: list[int] = []
    fg_colors: list[int] = []

    for idx, panel in enumerate(panels):
        marker_coords = np.argwhere(panel == MARKER)
        assert marker_coords.shape[0] in (0, 1), "each quadrant may contain at most one marker"
        cleaned_panel = panel.copy()
        if marker_coords.shape[0] == 1:
            marker_idxs.append(idx)
            mr, mc = map(int, marker_coords[0])
            cleaned_panel[mr, mc] = BG
        unique_fg = sorted(int(v) for v in np.unique(cleaned_panel) if int(v) != BG)
        assert len(unique_fg) == 1, "each quadrant must contain exactly one non-background shape color"
        fg_colors.append(unique_fg[0])
        cleaned.append(cleaned_panel)

    assert len(marker_idxs) == 2, "expected exactly two marked quadrants"
    assert set(marker_idxs) in ({0, 3}, {1, 2}), "marked quadrants must be diagonal"
    assert sorted(fg_colors) == list(QUAD_COLORS), "quadrant colors must be blue, red, green, yellow once each"

    out = np.zeros((QUAD_SIZE, QUAD_SIZE), dtype=int)
    for idx in sorted(marker_idxs):
        mask = cleaned[idx] != BG
        out[mask] = cleaned[idx][mask]

    center = out[CENTER_OFFSET : CENTER_OFFSET + MINI_SIZE, CENTER_OFFSET : CENTER_OFFSET + MINI_SIZE]
    for idx in range(4):
        if idx in marker_idxs:
            continue
        mini = _shrink_panel(cleaned[idx])
        mask = mini != BG
        center[mask] = mini[mask]

    return out


def _input_semantics_ok(
    masks: list[np.ndarray],
    main_idxs: tuple[int, int],
    colors: list[int],
    center_focus: tuple[int, int],
) -> bool:
    if set(main_idxs) not in ({0, 3}, {1, 2}):
        return False

    for mask in masks:
        area = int(mask.sum())
        if not 4 <= area <= 10:
            return False
        r0, c0, r1, c1 = _bbox({tuple(map(int, rc)) for rc in np.argwhere(mask)})
        if r1 - r0 + 1 > 6 or c1 - c0 + 1 > 6:
            return False

    main_a, main_b = main_idxs
    overlap = masks[main_a] & masks[main_b]
    if not overlap.any():
        return False
    if not np.any(overlap & ~_center_mask8()):
        return False
    if not np.any(masks[main_a] & ~masks[main_b]):
        return False
    if not np.any(masks[main_b] & ~masks[main_a]):
        return False

    secondary_idxs = tuple(idx for idx in range(4) if idx not in main_idxs)
    mini_a = _shrink_mask(masks[secondary_idxs[0]])
    mini_b = _shrink_mask(masks[secondary_idxs[1]])
    if not (mini_a & mini_b).any():
        return False
    if not np.any(mini_a & ~mini_b):
        return False
    if not np.any(mini_b & ~mini_a):
        return False

    if not mini_a[center_focus[0] - CENTER_OFFSET, center_focus[1] - CENTER_OFFSET]:
        return False
    if not mini_b[center_focus[0] - CENTER_OFFSET, center_focus[1] - CENTER_OFFSET]:
        return False

    top_left_used = False
    fallback_used = False
    for idx in secondary_idxs:
        mask = masks[idx]
        for mr in range(MINI_SIZE):
            for mc in range(MINI_SIZE):
                block = mask[2 * mr : 2 * mr + 2, 2 * mc : 2 * mc + 2]
                if not block.any():
                    continue
                if block[0, 0]:
                    top_left_used = True
                else:
                    fallback_used = True
    if not top_left_used or not fallback_used:
        return False

    base, final = _compose_output(masks, main_idxs, colors)
    if base[center_focus] == BG:
        return False
    if final[center_focus] == base[center_focus]:
        return False
    if np.array_equal(base, final):
        return False
    for color in colors:
        if not np.any(final == color):
            return False
    return True


def _center_mask8() -> np.ndarray:
    mask = np.zeros((QUAD_SIZE, QUAD_SIZE), dtype=bool)
    mask[CENTER_OFFSET : CENTER_OFFSET + MINI_SIZE, CENTER_OFFSET : CENTER_OFFSET + MINI_SIZE] = True
    return mask


def _gen_attempt() -> np.ndarray | None:
    colors = list(QUAD_COLORS)
    random.shuffle(colors)
    main_idxs = random.choice(((0, 3), (1, 2)))

    center_focus = (random.randint(2, 5), random.randint(2, 5))
    overlap_focus = _main_overlap_anchor(center_focus)
    unique_options = _outer_neighbor_options(overlap_focus)
    if len(unique_options) < 2:
        return None
    unique_a, unique_b = random.sample(unique_options, 2)

    masks: list[np.ndarray] = [np.zeros((QUAD_SIZE, QUAD_SIZE), dtype=bool) for _ in range(4)]
    masks[main_idxs[0]] = _make_main_shape(center_focus, overlap_focus, unique_a)
    masks[main_idxs[1]] = _make_main_shape(center_focus, overlap_focus, unique_b)
    if any(mask is None for mask in (masks[main_idxs[0]], masks[main_idxs[1]])):
        return None

    shared_micro = (center_focus[0] - CENTER_OFFSET, center_focus[1] - CENTER_OFFSET)
    micro_neighbors = _micro_neighbors(shared_micro)
    if len(micro_neighbors) < 2:
        return None
    unique_micro_a, unique_micro_b = random.sample(micro_neighbors, 2)

    secondary_idxs = [idx for idx in range(4) if idx not in main_idxs]
    if random.random() < 0.5:
        top_left_idx, fallback_idx = secondary_idxs
        top_left_micro, fallback_micro = unique_micro_a, unique_micro_b
    else:
        top_left_idx, fallback_idx = secondary_idxs[::-1]
        top_left_micro, fallback_micro = unique_micro_a, unique_micro_b

    masks[top_left_idx] = _make_secondary_shape(shared_micro, top_left_micro, True, fallback_micro)
    masks[fallback_idx] = _make_secondary_shape(shared_micro, fallback_micro, False, top_left_micro)
    if any(mask is None for mask in (masks[top_left_idx], masks[fallback_idx])):
        return None

    if not _input_semantics_ok(masks, main_idxs, colors, center_focus):
        return None

    expected_base, expected_out = _compose_output(masks, main_idxs, colors)
    if np.array_equal(expected_base, expected_out):
        return None

    grid = _build_input_grid(masks, main_idxs, colors)
    if grid is None:
        return None
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
