from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, components_mask, get_bg


def _nearest_squared_distance(source: np.ndarray, target: np.ndarray) -> int:
    diffs = source[:, None, :] - target[None, :, :]
    return int((diffs * diffs).sum(axis=2).min())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the grey object with the color of the marker set closest to it, erasing all markers.

    Input structure
    - The grid has a black background with one 4-connected grey object.
    - Every other foreground color appears as one or more isolated marker pixels.

    Rule
    - For each marker color, measure the smallest squared Euclidean distance from any marker pixel of that color to any grey-object pixel.
    - The unique marker color with the smallest such distance becomes the output color.
    - Return a blank background grid with the grey object's mask painted in that selected color.

    Key insight
    - The scattered pixels are competing color labels; only the label with the closest approach to the grey object survives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    assert bg == Color.BLACK, "expected black background"

    grey_mask = g == Color.GREY
    grey_components = components_mask(grey_mask, connectivity=4)
    assert len(grey_components) == 1, "expected exactly one grey object"
    grey_coords = np.argwhere(grey_mask)
    assert grey_coords.size > 0, "expected grey object pixels"

    scores: list[tuple[int, int]] = []
    for color in colors_present(g, exclude={bg, Color.GREY}):
        marker_mask = g == color
        marker_components = components_mask(marker_mask, connectivity=4)
        assert marker_components, "expected marker pixels for every candidate color"
        assert all(comp.shape[0] == 1 for comp in marker_components), "candidate colors must form isolated markers"

        marker_coords = np.argwhere(marker_mask)
        scores.append((_nearest_squared_distance(marker_coords, grey_coords), int(color)))

    assert scores, "expected at least one marker color"
    best_distance = min(distance for distance, _color in scores)
    winning_colors = [color for distance, color in scores if distance == best_distance]
    assert len(winning_colors) == 1, "expected a unique closest marker color"

    out = np.full_like(g, bg)
    out[grey_mask] = winning_colors[0]
    return out

# --------------- generator (o5x) ---------------

GRID_SIZE = 10
GRID_SHAPE = (GRID_SIZE, GRID_SIZE)
ORTHOGONAL_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Grey object size, symmetry axes, and growth retries.
GREY_AREA_MIN = 4
GREY_AREA_MAX = 13
GREY_MASK_ATTEMPTS = 25
SYMMETRIC_GROW_MAX_DEPTH = 8
CELL_CENTER_AXIS_CHOICES = [8, 10]
CENTER_BETWEEN_CELLS_AXIS = 9
DOUBLE_CENTER_AXIS_PROB = 0.4

# Marker color/count sampling and distance constraints.
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MARKER_COLOR_COUNT_MIN = 2
MARKER_COLOR_COUNT_MAX = 3
MARKERS_PER_COLOR_MIN = 1
MARKERS_PER_COLOR_MAX = 6
MARKER_TOTAL_MAX = 14
MARKER_COUNT_ATTEMPTS = 50
GREY_ADJACENT_DISTANCE = 1
WINNER_DISTANCE_MIN = 2
LOSER_DISTANCE_GAP = 2

GENERATE_ATTEMPTS = 300
Cell = tuple[int, int]
MarkerCells = list[tuple[int, list[Cell]]]


def _gen_reflect_orbit(r: int, c: int, axis_r2: int, axis_c2: int) -> frozenset[Cell]:
    rr = {int(r), int(axis_r2 - r)}
    cc = {int(c), int(axis_c2 - c)}
    return frozenset((a, b) for a in rr for b in cc if 0 <= a < GRID_SIZE and 0 <= b < GRID_SIZE)


def _gen_mask_from_coords(coords: set[Cell]) -> np.ndarray:
    mask = np.zeros(GRID_SHAPE, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _gen_connected(coords: set[Cell]) -> bool:
    if not coords:
        return False
    return len(components_mask(_gen_mask_from_coords(coords), connectivity=4)) == 1


def _gen_orthogonally_convex(mask: np.ndarray) -> bool:
    for r in range(GRID_SIZE):
        cols = np.flatnonzero(mask[r])
        if cols.size and cols.size != cols[-1] - cols[0] + 1:
            return False
    for c in range(GRID_SIZE):
        rows = np.flatnonzero(mask[:, c])
        if rows.size and rows.size != rows[-1] - rows[0] + 1:
            return False
    return True


def _gen_frontier_orbits(coords: set[Cell], axis_r2: int, axis_c2: int, remaining: int) -> list[frozenset[Cell]]:
    selected = frozenset(coords)
    out = []
    seen = set()
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            orb = _gen_reflect_orbit(r, c, axis_r2, axis_c2)
            if not orb or orb <= selected or orb in seen:
                continue
            new_cells = set(orb) - coords
            if len(new_cells) > remaining:
                continue
            touches = any((rr - 1, cc) in coords or (rr + 1, cc) in coords or (rr, cc - 1) in coords or (rr, cc + 1) in coords for rr, cc in new_cells)
            if touches and _gen_connected(coords | new_cells):
                seen.add(orb)
                out.append(frozenset(new_cells))
    return out


def _gen_grow_symmetric(coords: set[Cell], axis_r2: int, axis_c2: int, target: int, depth: int = 0) -> set[Cell] | None:
    remaining = target - len(coords)
    if remaining == 0:
        return coords if _gen_connected(coords) else None
    if remaining < 0 or depth > SYMMETRIC_GROW_MAX_DEPTH:
        return None

    orbits = _gen_frontier_orbits(coords, axis_r2, axis_c2, remaining)
    order = np.random.permutation(len(orbits)) if orbits else []
    for idx in order:
        grown = _gen_grow_symmetric(coords | set(orbits[int(idx)]), axis_r2, axis_c2, target, depth + 1)
        if grown is not None:
            return grown
    return None


def _gen_axis_pair_for_area(target: int) -> tuple[int, int]:
    if target % 2 == 1:
        return int(np.random.choice(CELL_CENTER_AXIS_CHOICES)), int(np.random.choice(CELL_CENTER_AXIS_CHOICES))
    if target % 4 == 0 and np.random.random() < DOUBLE_CENTER_AXIS_PROB:
        return CENTER_BETWEEN_CELLS_AXIS, CENTER_BETWEEN_CELLS_AXIS
    if np.random.random() < 0.5:
        return int(np.random.choice(CELL_CENTER_AXIS_CHOICES)), CENTER_BETWEEN_CELLS_AXIS
    return CENTER_BETWEEN_CELLS_AXIS, int(np.random.choice(CELL_CENTER_AXIS_CHOICES))


def _gen_grey_mask() -> np.ndarray | None:
    target = int(np.random.randint(GREY_AREA_MIN, GREY_AREA_MAX + 1))
    for _ in range(GREY_MASK_ATTEMPTS):
        axis_r2, axis_c2 = _gen_axis_pair_for_area(target)
        seed = set(_gen_reflect_orbit(axis_r2 // 2, axis_c2 // 2, axis_r2, axis_c2))
        if len(seed) > target:
            continue
        coords = _gen_grow_symmetric(seed, axis_r2, axis_c2, target)
        if coords is None:
            continue
        mask = _gen_mask_from_coords(coords)
        if len(components_mask(mask, connectivity=4)) == 1 and _gen_orthogonally_convex(mask):
            return mask
    return None


def _gen_squared_distances(grey_mask: np.ndarray) -> np.ndarray:
    grey = np.argwhere(grey_mask)
    rr, cc = np.indices(GRID_SHAPE)
    diffs_r = rr[:, :, None] - grey[:, 0]
    diffs_c = cc[:, :, None] - grey[:, 1]
    return (diffs_r * diffs_r + diffs_c * diffs_c).min(axis=2)


def _gen_marker_counts(color_count: int) -> list[int]:
    for _ in range(MARKER_COUNT_ATTEMPTS):
        counts = [int(np.random.randint(MARKERS_PER_COLOR_MIN, MARKERS_PER_COLOR_MAX + 1)) for _ in range(color_count)]
        if sum(counts) <= MARKER_TOTAL_MAX:
            return counts
    total = int(np.random.randint(color_count, min(MARKER_TOTAL_MAX, MARKERS_PER_COLOR_MAX * color_count) + 1))
    counts = [MARKERS_PER_COLOR_MIN] * color_count
    for _ in range(total - color_count):
        choices = [i for i, n in enumerate(counts) if n < MARKERS_PER_COLOR_MAX]
        counts[int(np.random.choice(choices))] += 1
    return counts


def _gen_pick_cells(candidates: list[Cell], count: int, occupied: set[Cell], *, first_pool: list[Cell] | None = None) -> list[Cell] | None:
    picked: list[Cell] = []
    blocked = set(occupied)

    if first_pool is not None:
        pool = [cell for cell in first_pool if cell not in blocked]
        if not pool:
            return None
        first = pool[int(np.random.randint(len(pool)))]
        picked.append(first)
        blocked.add(first)

    pool = [cell for cell in candidates if cell not in blocked]
    order = np.random.permutation(len(pool)) if pool else []
    for idx in order:
        if len(picked) == count:
            break
        cell = pool[int(idx)]
        r, c = cell
        if any(abs(r - pr) + abs(c - pc) == 1 for pr, pc in picked):
            continue
        if any((r + dr, c + dc) in occupied for dr, dc in ORTHOGONAL_NEIGHBORS):
            continue
        picked.append(cell)
        blocked.add(cell)

    return picked if len(picked) == count else None


def _gen_marker_plan() -> tuple[list[int], int, int, list[int]]:
    """Sample marker colors, the winning color index, and per-color marker counts."""
    color_count = int(np.random.randint(MARKER_COLOR_COUNT_MIN, MARKER_COLOR_COUNT_MAX + 1))
    color_order = list(np.random.choice(MARKER_COLORS, size=color_count, replace=False))
    winner_idx = int(np.random.randint(color_count))
    winner = int(color_order[winner_idx])
    counts = _gen_marker_counts(color_count)
    return color_order, winner_idx, winner, counts


def _gen_distance_layout(grey_mask: np.ndarray, counts: list[int], winner_idx: int) -> tuple[np.ndarray, set[Cell], list[Cell], int] | None:
    """Return the distance field, blocked cells, free cells, and sampled winning distance."""
    dists = _gen_squared_distances(grey_mask)
    blocked_by_grey = set(map(tuple, np.argwhere(grey_mask | (dists == GREY_ADJACENT_DISTANCE))))
    available = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if (r, c) not in blocked_by_grey]
    dist_values = sorted({int(dists[r, c]) for r, c in available if int(dists[r, c]) >= WINNER_DISTANCE_MIN})
    loser_total = sum(n for i, n in enumerate(counts) if i != winner_idx)
    viable = [
        d
        for d in dist_values
        if any(int(dists[r, c]) == d for r, c in available)
        and sum(1 for r, c in available if int(dists[r, c]) >= d) >= counts[winner_idx]
        and sum(1 for r, c in available if int(dists[r, c]) >= d + LOSER_DISTANCE_GAP) >= loser_total
    ]
    if not viable:
        return None
    d_win = int(viable[int(np.random.randint(len(viable)))])
    return dists, blocked_by_grey, available, d_win


def _gen_place_markers(
    color_order: list[int], counts: list[int], winner_idx: int, dists: np.ndarray, available: list[Cell], d_win: int, occupied: set[Cell]
) -> MarkerCells | None:
    """Assign marker cells while keeping the winner uniquely closest to grey."""
    marker_cells: MarkerCells = []
    for idx, color in enumerate(color_order):
        if idx == winner_idx:
            candidates = [(r, c) for r, c in available if int(dists[r, c]) >= d_win]
            first_pool = [(r, c) for r, c in candidates if int(dists[r, c]) == d_win]
        else:
            candidates = [(r, c) for r, c in available if int(dists[r, c]) >= d_win + LOSER_DISTANCE_GAP]
            first_pool = None
        cells = _gen_pick_cells(candidates, counts[idx], occupied, first_pool=first_pool)
        if cells is None:
            return None
        for cell in cells:
            occupied.add(cell)
        marker_cells.append((int(color), cells))
    return marker_cells


def _gen_render_input(grey_mask: np.ndarray, marker_cells: MarkerCells) -> np.ndarray:
    inp = np.zeros(GRID_SHAPE, dtype=int)
    inp[grey_mask] = Color.GREY
    for color, cells in marker_cells:
        for r, c in cells:
            inp[r, c] = color
    return inp


def _gen_attempt() -> np.ndarray | None:
    grey_mask = _gen_grey_mask()
    if grey_mask is None:
        return None

    color_order, winner_idx, winner, counts = _gen_marker_plan()
    out = np.zeros(GRID_SHAPE, dtype=int)
    out[grey_mask] = winner

    layout = _gen_distance_layout(grey_mask, counts, winner_idx)
    if layout is None:
        return None
    dists, blocked_by_grey, available, d_win = layout

    marker_cells = _gen_place_markers(color_order, counts, winner_idx, dists, available, d_win, set(blocked_by_grey))
    if marker_cells is None:
        return None
    inp = _gen_render_input(grey_mask, marker_cells)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 6df30ad6 input")
