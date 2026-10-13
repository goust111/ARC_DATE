from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

import numpy as np


BACKGROUND = 0
ARC_COLORS = tuple(range(1, 10))
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
PERP = {
    (-1, 0): ((0, -1), (0, 1)),
    (1, 0): ((0, -1), (0, 1)),
    (0, -1): ((-1, 0), (1, 0)),
    (0, 1): ((-1, 0), (1, 0)),
}
GEN_ATTEMPTS = 900


@dataclass(frozen=True)
class Ring:
    center: tuple[int, int]
    gaps: tuple[tuple[int, int], tuple[int, int]]
    color: int


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return int(a[0] + b[0]), int(a[1] + b[1])


def _neg(d: tuple[int, int]) -> tuple[int, int]:
    return -int(d[0]), -int(d[1])


def _ring_solid_cells(center: tuple[int, int], gaps: tuple[tuple[int, int], tuple[int, int]]) -> set[tuple[int, int]]:
    cr, cc = center
    gap_set = set(gaps)
    cells: set[tuple[int, int]] = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if abs(dr) + abs(dc) == 1 and (dr, dc) in gap_set:
                continue
            cells.add((cr + dr, cc + dc))
    return cells


def _ring_gap_cells(ring: Ring) -> dict[tuple[int, int], tuple[int, int]]:
    cr, cc = ring.center
    return {(cr + dr, cc + dc): (dr, dc) for dr, dc in ring.gaps}


def _ring_bbox_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = center
    return {(cr + dr, cc + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _find_source(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], tuple[int, int], int]:
    h, w = map(int, grid.shape)
    edge_cells: list[tuple[int, int]] = []
    for c in range(w):
        if int(grid[0, c]) != bg:
            edge_cells.append((0, c))
        if int(grid[h - 1, c]) != bg:
            edge_cells.append((h - 1, c))
    for r in range(1, h - 1):
        if int(grid[r, 0]) != bg:
            edge_cells.append((r, 0))
        if int(grid[r, w - 1]) != bg:
            edge_cells.append((r, w - 1))

    assert len(edge_cells) == 1, "expected one non-background source pixel on the grid edge"
    r, c = edge_cells[0]
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    assert sum((on_top, on_bottom, on_left, on_right)) == 1, "source cannot sit on a corner"

    if on_top:
        direction = (1, 0)
    elif on_bottom:
        direction = (-1, 0)
    elif on_left:
        direction = (0, 1)
    else:
        direction = (0, -1)
    return (int(r), int(c)), direction, int(grid[r, c])


def _find_rings(grid: np.ndarray, bg: int) -> list[Ring]:
    h, w = map(int, grid.shape)
    rings: list[Ring] = []
    used_solid: set[tuple[int, int]] = set()

    for cr in range(1, h - 1):
        for cc in range(1, w - 1):
            if int(grid[cr, cc]) != bg:
                continue

            corner_values = [
                int(grid[cr - 1, cc - 1]),
                int(grid[cr - 1, cc + 1]),
                int(grid[cr + 1, cc - 1]),
                int(grid[cr + 1, cc + 1]),
            ]
            if any(v == bg for v in corner_values):
                continue
            if len(set(corner_values)) != 1:
                continue
            color = corner_values[0]

            gaps: list[tuple[int, int]] = []
            ok = True
            for side in DIRS:
                sr, sc = cr + side[0], cc + side[1]
                value = int(grid[sr, sc])
                if value == bg:
                    gaps.append(side)
                elif value != color:
                    ok = False
                    break
            if not ok or len(gaps) != 2:
                continue

            ring = Ring((int(cr), int(cc)), (gaps[0], gaps[1]), int(color))
            solid = _ring_solid_cells(ring.center, ring.gaps)
            assert not (solid & used_solid), "ring frames must be disjoint"
            used_solid |= solid
            rings.append(ring)

    assert rings, "expected at least one two-gap 3x3 marker ring"
    return rings


def _trace(
    grid: np.ndarray,
    source: tuple[int, int],
    direction: tuple[int, int],
    source_color: int,
    rings: list[Ring],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    bg = _most_common_color(grid)

    gap_map: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for idx, ring in enumerate(rings):
        for cell, normal in _ring_gap_cells(ring).items():
            assert cell not in gap_map, "ring gaps must be unique"
            gap_map[cell] = (idx, normal)

    pos = source
    d = direction
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for _ in range(grid.size * 8 + 20):
        state = (pos, d)
        assert state not in seen, "light path must not loop"
        seen.add(state)

        nr, nc = pos[0] + d[0], pos[1] + d[1]
        if not _in_bounds(grid.shape, nr, nc):
            return out.astype(int)

        cell = (int(nr), int(nc))
        value = int(grid[cell])
        if value != bg:
            return out.astype(int)

        if cell in gap_map:
            ring_idx, entry_normal = gap_map[cell]
            assert entry_normal == _neg(d), "ray reached a ring gap from the wrong side"
            ring = rings[ring_idx]
            center = ring.center
            assert _add(cell, d) == center, "entry gap must lead directly to the ring center"

            exit_normals = [normal for normal in ring.gaps if normal != entry_normal]
            assert len(exit_normals) == 1, "each ring must have one exit gap after entry"
            exit_dir = exit_normals[0]
            exit_cell = _add(center, exit_dir)
            assert int(grid[center]) == bg and int(grid[exit_cell]) == bg

            out[cell] = source_color
            out[center] = source_color
            out[exit_cell] = source_color
            pos = exit_cell
            d = exit_dir
            continue

        out[cell] = source_color
        pos = cell

    raise AssertionError("light path exceeded the finite tracing limit")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the edge source ray through two-gap square rings and paint the full bent light path.

    Input structure:
    - The background is the dominant color, with exactly one non-background source pixel on a non-corner edge.
    - Several separated 3x3 hollow square rings sit away from the border.
    - Each ring has a single-color solid frame except for exactly two missing side-midpoint gap cells;
      ring centers and gap cells are background.

    Rule:
    - Send the ray from the source perpendicular to its edge and paint every traversed background cell
      with the source color.
    - When the ray enters a ring through the gap facing it, paint that gap, the ring center, and the
      other gap; then continue outward in the other gap's side-normal direction.
    - Stop without repainting if the next cell is a solid ring frame; stop when the next step leaves
      the grid.  Ring frame pixels keep their original color.

    Key insight:
    - A double-notched 3x3 ring is a directional pipe: the entry gap determines the ring, and the
      other gap determines the next ray direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"

    bg = _most_common_color(g)
    source, direction, source_color = _find_source(g, bg)
    rings = _find_rings(g, bg)
    return _trace(g, source, direction, source_color, rings)


def _segment_to_ring(
    pos: tuple[int, int],
    d: tuple[int, int],
    center: tuple[int, int],
) -> list[tuple[int, int]]:
    entry = (center[0] - d[0], center[1] - d[1])
    cells: list[tuple[int, int]] = []
    r, c = pos
    while (r, c) != entry:
        r += d[0]
        c += d[1]
        cells.append((int(r), int(c)))
    cells.append(center)
    return cells


def _distance_to_entry(pos: tuple[int, int], d: tuple[int, int], center: tuple[int, int]) -> int:
    entry = (center[0] - d[0], center[1] - d[1])
    return abs(entry[0] - pos[0]) + abs(entry[1] - pos[1])


def _center_candidates(
    h: int,
    w: int,
    pos: tuple[int, int],
    d: tuple[int, int],
    existing: list[Ring],
    min_steps: int,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for cr in range(3, h - 3):
        for cc in range(3, w - 3):
            center = (cr, cc)
            if d[0] != 0 and cc != pos[1]:
                continue
            if d[1] != 0 and cr != pos[0]:
                continue
            if (cr - pos[0]) * d[0] + (cc - pos[1]) * d[1] <= 0:
                continue
            if _distance_to_entry(pos, d, center) < min_steps:
                continue
            if any(max(abs(cr - r.center[0]), abs(cc - r.center[1])) < 5 for r in existing):
                continue
            candidates.append(center)
    return candidates


def _ray_cells_until_edge(shape: tuple[int, int], pos: tuple[int, int], d: tuple[int, int]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = pos
    while True:
        r += d[0]
        c += d[1]
        if not _in_bounds(shape, r, c):
            return cells
        cells.append((int(r), int(c)))


def _render_grid(
    shape: tuple[int, int],
    source: tuple[int, int],
    source_color: int,
    rings: list[Ring],
) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    grid[source] = int(source_color)
    for ring in rings:
        for r, c in _ring_solid_cells(ring.center, ring.gaps):
            grid[r, c] = int(ring.color)
    return grid


def _planned_trace(
    grid: np.ndarray,
    source: tuple[int, int],
    direction: tuple[int, int],
    source_color: int,
    rings: list[Ring],
) -> np.ndarray:
    return _trace(grid, source, direction, source_color, rings)


def _straight_ray_output(grid: np.ndarray, source: tuple[int, int], d: tuple[int, int], color: int) -> np.ndarray:
    out = grid.copy()
    for r, c in _ray_cells_until_edge(grid.shape, source, d):
        if int(grid[r, c]) != BACKGROUND:
            break
        out[r, c] = int(color)
    return out


def _painted_cells(grid: np.ndarray, out: np.ndarray, source_color: int) -> set[tuple[int, int]]:
    coords = np.argwhere((grid != out) & (out == int(source_color)))
    return {(int(r), int(c)) for r, c in coords}


def _choose_source(rng: random.Random, h: int, w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    side = rng.choice(("top", "bottom", "left", "right"))
    if side == "top":
        return (0, rng.randrange(4, w - 4)), (1, 0)
    if side == "bottom":
        return (h - 1, rng.randrange(4, w - 4)), (-1, 0)
    if side == "left":
        return (rng.randrange(4, h - 4), 0), (0, 1)
    return (rng.randrange(4, h - 4), w - 1), (0, -1)


def _try_add_active_ring(
    rng: random.Random,
    h: int,
    w: int,
    pos: tuple[int, int],
    d: tuple[int, int],
    rings: list[Ring],
    path_cells: set[tuple[int, int]],
    ring_colors: list[int],
) -> tuple[Ring, tuple[int, int], tuple[int, int], set[tuple[int, int]]] | None:
    centers = _center_candidates(h, w, pos, d, rings, min_steps=3)
    rng.shuffle(centers)
    for center in centers[:40]:
        out_dirs = list(PERP[d])
        rng.shuffle(out_dirs)
        for out_dir in out_dirs:
            gaps = (_neg(d), out_dir)
            ring = Ring(center, gaps, rng.choice(ring_colors))
            exit_cell = _add(center, out_dir)
            if not _in_bounds((h, w), exit_cell[0] + out_dir[0], exit_cell[1] + out_dir[1]):
                continue

            new_path = set(_segment_to_ring(pos, d, center))
            new_path.add(exit_cell)
            if path_cells & new_path:
                continue

            solids = _ring_solid_cells(center, gaps)
            if solids & path_cells:
                continue
            if any(cell in solids for cell in new_path):
                continue

            return ring, exit_cell, out_dir, new_path
    return None


def _try_add_blocker(
    rng: random.Random,
    h: int,
    w: int,
    pos: tuple[int, int],
    d: tuple[int, int],
    rings: list[Ring],
    path_cells: set[tuple[int, int]],
    ring_colors: list[int],
) -> tuple[Ring, list[tuple[int, int]], tuple[int, int]] | None:
    centers = _center_candidates(h, w, pos, d, rings, min_steps=3)
    rng.shuffle(centers)
    forbidden_gap = _neg(d)
    existing_ring_area = set().union(*(_ring_bbox_cells(ring.center) for ring in rings)) if rings else set()
    for center in centers[:60]:
        gap_options = [side for side in DIRS if side != forbidden_gap]
        rng.shuffle(gap_options)
        for i in range(len(gap_options)):
            for j in range(i + 1, len(gap_options)):
                gaps = (gap_options[i], gap_options[j])
                ring = Ring(center, gaps, rng.choice(ring_colors))
                hit_cell = (center[0] - d[0], center[1] - d[1])
                segment = []
                r, c = pos
                while (r, c) != hit_cell:
                    r += d[0]
                    c += d[1]
                    segment.append((int(r), int(c)))
                if any(cell in path_cells for cell in segment[:-1]):
                    continue
                if any(cell in existing_ring_area for cell in segment[:-1]):
                    continue
                if hit_cell not in _ring_solid_cells(center, gaps):
                    continue
                return ring, segment[:-1], hit_cell
    return None


def _add_decoy_ring(
    rng: random.Random,
    h: int,
    w: int,
    rings: list[Ring],
    path_cells: set[tuple[int, int]],
    ring_colors: list[int],
) -> Ring | None:
    centers = [(r, c) for r in range(3, h - 3) for c in range(3, w - 3)]
    rng.shuffle(centers)
    for center in centers[:160]:
        if any(max(abs(center[0] - ring.center[0]), abs(center[1] - ring.center[1])) < 5 for ring in rings):
            continue
        gap_list = list(DIRS)
        rng.shuffle(gap_list)
        gaps = (gap_list[0], gap_list[1])
        solid = _ring_solid_cells(center, gaps)
        gap_cells = set(_ring_gap_cells(Ring(center, gaps, 1)).keys())
        if solid & path_cells:
            continue
        if gap_cells & path_cells:
            continue
        return Ring(center, gaps, rng.choice(ring_colors))
    return None


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randrange(17, 31)
    w = rng.randrange(17, 31)
    source, direction = _choose_source(rng, h, w)
    source_color = rng.choice(ARC_COLORS)
    ring_colors = [color for color in ARC_COLORS if color != source_color]
    rng.shuffle(ring_colors)
    ring_colors = ring_colors[: rng.randrange(1, 4)]

    active_count = rng.randrange(2, 5)
    rings: list[Ring] = []
    active_rings: list[Ring] = []
    path_cells: set[tuple[int, int]] = {source}
    pos = source
    d = direction

    for _ in range(active_count):
        added = _try_add_active_ring(rng, h, w, pos, d, rings, path_cells, ring_colors)
        if added is None:
            return None
        ring, pos, d, new_path = added
        rings.append(ring)
        active_rings.append(ring)
        path_cells |= new_path

    hit_branch = rng.random() < 0.45
    blocker: Ring | None = None
    blocker_before: list[tuple[int, int]] = []
    blocker_hit: tuple[int, int] | None = None
    if hit_branch:
        blocker_result = _try_add_blocker(rng, h, w, pos, d, rings, path_cells, ring_colors)
        if blocker_result is not None:
            blocker, blocker_before, blocker_hit = blocker_result
            rings.append(blocker)
            path_cells |= set(blocker_before)
        else:
            hit_branch = False

    if not hit_branch:
        edge_cells = set(_ray_cells_until_edge((h, w), pos, d))
        if len(edge_cells) < 2 or path_cells & edge_cells:
            return None
        path_cells |= edge_cells

    decoy_target = rng.randrange(1, 3)
    decoys = 0
    for _ in range(decoy_target):
        decoy = _add_decoy_ring(rng, h, w, rings, path_cells, ring_colors)
        if decoy is None:
            return None
        rings.append(decoy)
        decoys += 1

    grid = _render_grid((h, w), source, source_color, rings)
    expected = _planned_trace(grid, source, direction, source_color, rings)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    changed = _painted_cells(grid, expected, source_color)
    if len(changed) < 12:
        return None
    if np.array_equal(expected, _straight_ray_output(grid, source, direction, source_color)):
        return None
    for ring in active_rings:
        if ring.center not in changed:
            return None
        for gap_cell in _ring_gap_cells(ring):
            if gap_cell not in changed:
                return None
    if decoys < 1:
        return None
    if hit_branch and blocker is not None and blocker_hit is not None:
        if int(expected[blocker_hit]) != int(grid[blocker_hit]):
            return None
        if any(int(expected[cell]) != int(source_color) for cell in blocker_before):
            return None
    if not any(ring.gaps[1] in PERP[_neg(ring.gaps[0])] for ring in active_rings):
        return None

    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert expected.ndim == 2 and max(expected.shape) <= 30
    return grid.astype(int)


def generate_input() -> np.ndarray:
    """Generate a random non-degenerate gap-guided light-ring input."""
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gap-light-ring input")
