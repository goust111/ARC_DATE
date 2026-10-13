from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

OFFSETS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FULL_2X2 = {(0, 0), (0, 1), (1, 0), (1, 1)}
MISSING_TO_ROTATION = {
    (0, 0): 0,  # missing top-left
    (0, 1): 1,  # missing top-right
    (1, 1): 2,  # missing bottom-right
    (1, 0): 3,  # missing bottom-left
}

MAX_GENERATION_ATTEMPTS = 500
SOURCE_ATTEMPTS = 200
TARGET_ATTEMPTS = 200
EXTRA_TARGET_ATTEMPTS = 120
TRIANGLE_ATTEMPTS = 80
SOURCE_DIMENSIONS = [(h, w) for h in range(2, 5) for w in range(2, 5) if h * w >= 5]


Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r + dr, c + dc) for dr, dc in OFFSETS4]


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[Coord] | tuple[Coord, ...] | set[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[Coord] | tuple[Coord, ...] | set[Coord]) -> tuple[Coord, ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _rotate_once(offset: Coord) -> Coord:
    dr, dc = offset
    return dc, -dr


def _rotate_offsets(offsets: list[Coord] | tuple[Coord, ...], turns: int) -> tuple[Coord, ...]:
    out = list(offsets)
    for _ in range(turns % 4):
        out = [_rotate_once(offset) for offset in out]
    return tuple(sorted((int(dr), int(dc)) for dr, dc in out))


def _decode_triangle_rotation(cells: list[Coord] | tuple[Coord, ...]) -> int:
    assert len(cells) == 3, "each grey triangle must have exactly 3 cells"
    r0, c0, r1, c1 = _bbox(cells)
    assert r1 - r0 == 1 and c1 - c0 == 1, "grey triangle must occupy a 2x2 box"
    present = {(r - r0, c - c0) for r, c in cells}
    missing = next(iter(FULL_2X2 - present))
    return MISSING_TO_ROTATION[missing]


def _find_anchor(grid: np.ndarray) -> Coord:
    coords = np.argwhere(grid == RED)
    assert len(coords) == 1, "expected exactly one red anchor"
    return int(coords[0, 0]), int(coords[0, 1])


def _extract_source_offsets(grid: np.ndarray, anchor: Coord) -> tuple[Coord, ...]:
    comps = _components(grid == BLUE)
    assert len(comps) == 1, "expected one 4-connected blue source mask"
    comp = comps[0]
    assert 5 <= len(comp) <= 8, "source mask size must be 5..8"
    r0, c0, r1, c1 = _bbox(comp)
    assert r1 - r0 + 1 <= 4 and c1 - c0 + 1 <= 4, "source bbox must be at most 4x4"
    ar, ac = anchor
    offsets = tuple(sorted((r - ar, c - ac) for r, c in comp))
    assert len({_normalize(_rotate_offsets(offsets, k)) for k in range(4)}) == 4, "rotations must stay distinguishable"
    return offsets


def _targets(grid: np.ndarray) -> list[Coord]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREEN)]
    coords.sort()
    assert 3 <= len(coords) <= 4, "expected 3 or 4 green targets"
    return coords


def _triangle_components(grid: np.ndarray) -> list[list[Coord]]:
    comps = _components(grid == GREY)
    assert len(comps) >= 3, "expected one grey triangle per target"
    for comp in comps:
        assert len(comp) == 3, "grey triangle must have 3 cells"
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 == 1 and c1 - c0 == 1, "grey triangle must be an L in a 2x2 box"
    return comps


def _pair_targets_with_rotations(grid: np.ndarray, targets: list[Coord]) -> list[int]:
    triangles = _triangle_components(grid)
    rotations: list[int] = []
    used: set[int] = set()
    for target in targets:
        matches = []
        for idx, comp in enumerate(triangles):
            if idx in used:
                continue
            if any(abs(r - target[0]) + abs(c - target[1]) == 1 for r, c in comp):
                matches.append(idx)
        assert len(matches) == 1, "each target must touch exactly one grey triangle"
        idx = matches[0]
        used.add(idx)
        rotations.append(_decode_triangle_rotation(triangles[idx]))
    assert len(used) == len(targets), "targets and triangles must match one-to-one"
    return rotations


def _stamp_cells(shape: tuple[int, int], target: Coord, offsets: list[Coord] | tuple[Coord, ...]) -> set[Coord]:
    cells = {(target[0] + dr, target[1] + dc) for dr, dc in offsets}
    assert all(_in_bounds(shape, r, c) for r, c in cells), "stamp must stay in bounds"
    return cells


def _render_output(
    shape: tuple[int, int],
    targets: list[Coord],
    rotated_offsets: list[tuple[Coord, ...]],
    *,
    background: int = BLACK,
    overlap_recolors: bool = True,
) -> np.ndarray:
    out = np.full(shape, background, dtype=int)
    target_set = set(targets)
    cover_counts: dict[Coord, int] = {}
    for target, offsets in zip(targets, rotated_offsets):
        for cell in _stamp_cells(shape, target, offsets):
            cover_counts[cell] = cover_counts.get(cell, 0) + 1
    for (r, c), count in cover_counts.items():
        if (r, c) in target_set:
            continue
        out[r, c] = YELLOW if overlap_recolors and count >= 2 else BLUE
    for r, c in targets:
        out[r, c] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate-and-stamp the blue source mask at each green target, with grey L-triangles encoding the rotation and overlaps turning yellow.

    Input structure:
    - A black background contains one 4-connected blue source mask and a single red anchor pixel adjacent to it.
    - There are 3 or 4 green target pixels, each touching exactly one grey 3-cell L-shape occupying a 2x2 box.
    - The missing corner of each grey L-shape encodes one of the four quarter-turn rotations.

    Rule:
    - Read the blue source mask as offsets relative to the red anchor.
    - For each green target, decode the neighbouring grey L-shape: missing top-left = 0 degrees, top-right = 90 clockwise, bottom-right = 180, bottom-left = 270 clockwise.
    - Rotate the source offsets by that amount, erase the original source, anchor, and grey triangles, then stamp the rotated blue mask around the target.
    - If two or more stamps cover the same non-target cell, recolor that overlap cell yellow instead of blue; every green target cell itself stays green.

    Key insight:
    - The red anchor defines the source coordinate frame, while each grey L-shape does not mark a destination by itself but tells you exactly how the same source mask must be rotated before stamping there.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    anchor = _find_anchor(g)
    source_offsets = _extract_source_offsets(g, anchor)
    targets = _targets(g)
    rotations = _pair_targets_with_rotations(g, targets)
    rotated_offsets = [_rotate_offsets(source_offsets, turns) for turns in rotations]
    return _render_output(g.shape, targets, rotated_offsets)


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _grow_connected_shape(h: int, w: int, area: int) -> list[Coord] | None:
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return sorted(cells)


def _anchor_candidates(shape_cells: list[Coord]) -> list[Coord]:
    r0, c0, r1, c1 = _bbox(shape_cells)
    shape_set = set(shape_cells)
    candidates = set()
    for r in range(r0, r1 + 1):
        for c in (c0 - 1, c1 + 1):
            if (r, c) not in shape_set and any((nr, nc) in shape_set for nr, nc in _neighbors4(r, c)):
                candidates.add((r, c))
    for c in range(c0, c1 + 1):
        for r in (r0 - 1, r1 + 1):
            if (r, c) not in shape_set and any((nr, nc) in shape_set for nr, nc in _neighbors4(r, c)):
                candidates.add((r, c))
    return sorted(candidates)


def _overlap_specs(offsets: tuple[Coord, ...]) -> list[tuple[int, int, int, int]]:
    rotated = [set(_rotate_offsets(offsets, k)) for k in range(4)]
    specs: list[tuple[int, int, int, int]] = []
    for k1 in range(4):
        for k2 in range(4):
            if k1 == k2:
                continue
            for dr in range(-8, 9):
                for dc in range(-8, 9):
                    if dr == 0 and dc == 0:
                        continue
                    if abs(dr) + abs(dc) < 6:
                        continue
                    shifted = {(r + dr, c + dc) for r, c in rotated[k2]}
                    overlap = len(rotated[k1] & shifted)
                    if 1 <= overlap <= 3:
                        specs.append((k1, k2, dr, dc))
    return specs


def _generate_source_offsets() -> tuple[Coord, ...]:
    for _ in range(SOURCE_ATTEMPTS):
        h, w = random.choice(SOURCE_DIMENSIONS)
        area = random.randint(5, min(8, h * w))
        shape_cells = _grow_connected_shape(h, w, area)
        if shape_cells is None:
            continue
        candidates = _anchor_candidates(shape_cells)
        random.shuffle(candidates)
        for anchor in candidates:
            offsets = tuple(sorted((r - anchor[0], c - anchor[1]) for r, c in shape_cells))
            if len({_normalize(_rotate_offsets(offsets, k)) for k in range(4)}) < 4:
                continue
            if _overlap_specs(offsets):
                return offsets
    raise RuntimeError("failed to generate a non-degenerate source mask")


def _triangle_layouts() -> dict[int, list[tuple[Coord, ...]]]:
    layouts: dict[int, list[tuple[Coord, ...]]] = {}
    for missing, rotation in MISSING_TO_ROTATION.items():
        tri = FULL_2X2 - {missing}
        rels = set()
        for tr in range(-1, 3):
            for tc in range(-1, 3):
                if (tr, tc) in FULL_2X2:
                    continue
                if not any(abs(r - tr) + abs(c - tc) == 1 for r, c in tri):
                    continue
                rel = tuple(sorted((r - tr, c - tc) for r, c in tri))
                rels.add(rel)
        layouts[rotation] = sorted(rels)
    return layouts


TRIANGLE_LAYOUTS = _triangle_layouts()


def _target_stamp_ok(stamp: set[Coord], target: Coord, placed_targets: list[Coord], placed_stamps: list[set[Coord]]) -> bool:
    if any(target in other_stamp for other_stamp in placed_stamps):
        return False
    if any(other_target in stamp for other_target in placed_targets):
        return False
    if any(stamp & other_stamp for other_stamp in placed_stamps):
        return False
    return True


def _place_targets(
    shape: tuple[int, int],
    source_offsets: tuple[Coord, ...],
    overlap_specs: list[tuple[int, int, int, int]],
) -> tuple[list[Coord], list[int]]:
    rotated = [_rotate_offsets(source_offsets, k) for k in range(4)]
    for _ in range(TARGET_ATTEMPTS):
        target_count = random.choice([3, 3, 4])
        k1, k2, dr, dc = random.choice(overlap_specs)
        target1 = (random.randrange(shape[0]), random.randrange(shape[1]))
        target2 = (target1[0] + dr, target1[1] + dc)
        if not _in_bounds(shape, *target2):
            continue
        if _manhattan(target1, target2) < 6:
            continue
        try:
            stamp1 = _stamp_cells(shape, target1, rotated[k1])
            stamp2 = _stamp_cells(shape, target2, rotated[k2])
        except AssertionError:
            continue
        overlap = stamp1 & stamp2
        if not (1 <= len(overlap) <= 3):
            continue
        if target1 in stamp2 or target2 in stamp1:
            continue
        targets = [target1, target2]
        rotations = [k1, k2]
        stamps = [stamp1, stamp2]
        ok = True
        while len(targets) < target_count:
            placed = False
            for _ in range(EXTRA_TARGET_ATTEMPTS):
                target = (random.randrange(shape[0]), random.randrange(shape[1]))
                if any(_manhattan(target, other) < 6 for other in targets):
                    continue
                rotation = random.randrange(4)
                try:
                    stamp = _stamp_cells(shape, target, rotated[rotation])
                except AssertionError:
                    continue
                if not _target_stamp_ok(stamp, target, targets, stamps):
                    continue
                targets.append(target)
                rotations.append(rotation)
                stamps.append(stamp)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok and len(set(rotations)) >= 2:
            return targets, rotations
    raise RuntimeError("failed to place target markers")


def _place_triangles(shape: tuple[int, int], targets: list[Coord], rotations: list[int]) -> list[set[Coord]]:
    for _ in range(TRIANGLE_ATTEMPTS):
        occupied = set(targets)
        occupied_grey: set[Coord] = set()
        triangles: list[set[Coord]] = []
        ok = True
        for target, rotation in zip(targets, rotations):
            layouts = TRIANGLE_LAYOUTS[rotation][:]
            random.shuffle(layouts)
            chosen: set[Coord] | None = None
            for rel in layouts:
                cells = {(target[0] + dr, target[1] + dc) for dr, dc in rel}
                if not all(_in_bounds(shape, r, c) for r, c in cells):
                    continue
                if cells & occupied:
                    continue
                if any((nr, nc) in occupied_grey for r, c in cells for nr, nc in _neighbors4(r, c)):
                    continue
                chosen = cells
                break
            if chosen is None:
                ok = False
                break
            triangles.append(chosen)
            occupied |= chosen
            occupied_grey |= chosen
        if ok:
            return triangles
    raise RuntimeError("failed to place grey triangles")


def _source_fit_positions(shape: tuple[int, int], offsets: tuple[Coord, ...]) -> list[Coord]:
    rows = [0, *(dr for dr, _ in offsets)]
    cols = [0, *(dc for _, dc in offsets)]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    positions = []
    for ar in range(-min_r, shape[0] - max_r):
        for ac in range(-min_c, shape[1] - max_c):
            positions.append((ar, ac))
    return positions


def _place_source(
    shape: tuple[int, int],
    source_offsets: tuple[Coord, ...],
    blocked: set[Coord],
) -> tuple[Coord, set[Coord]]:
    candidates = _source_fit_positions(shape, source_offsets)
    candidates.sort(
        key=lambda p: (
            min(p[0], p[1], shape[0] - 1 - p[0], shape[1] - 1 - p[1]),
            random.random(),
        )
    )
    for anchor in candidates:
        blue_cells = {(anchor[0] + dr, anchor[1] + dc) for dr, dc in source_offsets}
        if blue_cells & blocked or anchor in blocked:
            continue
        if any(any(_manhattan(cell, other) <= 1 for other in blocked) for cell in blue_cells):
            continue
        if any(_manhattan(anchor, other) <= 1 for other in blocked):
            continue
        return anchor, blue_cells
    raise RuntimeError("failed to place source mask")


def _build_grid(
    shape: tuple[int, int],
    anchor: Coord,
    blue_cells: set[Coord],
    targets: list[Coord],
    triangles: list[set[Coord]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for r, c in blue_cells:
        grid[r, c] = BLUE
    grid[anchor] = RED
    for r, c in targets:
        grid[r, c] = GREEN
    for tri in triangles:
        for r, c in tri:
            grid[r, c] = GREY
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        anchor = _find_anchor(grid)
        source_offsets = _extract_source_offsets(grid, anchor)
        targets = _targets(grid)
        rotations = _pair_targets_with_rotations(grid, targets)
    except AssertionError:
        return False

    rotated = [_rotate_offsets(source_offsets, turns) for turns in rotations]
    stamps = [_stamp_cells(grid.shape, target, offsets) for target, offsets in zip(targets, rotated)]
    overlaps = [len(stamps[i] & stamps[j]) for i in range(len(stamps)) for j in range(i + 1, len(stamps))]
    if not any(1 <= overlap <= 3 for overlap in overlaps):
        return False
    if any(overlap > 3 for overlap in overlaps):
        return False
    if len(set(rotations)) < 2:
        return False
    if any(_manhattan(targets[i], targets[j]) < 6 for i in range(len(targets)) for j in range(i + 1, len(targets))):
        return False

    try:
        actual = _render_output(grid.shape, targets, rotated)
        no_rotation = _render_output(grid.shape, targets, [_rotate_offsets(source_offsets, 0) for _ in targets])
    except AssertionError:
        return False
    if np.array_equal(actual, no_rotation):
        return False
    try:
        no_overlap = _render_output(grid.shape, targets, rotated, overlap_recolors=False)
    except AssertionError:
        return False
    if np.array_equal(actual, no_overlap):
        return False

    blue_neighbors = 0
    blue_set = {(r, c) for r, c in np.argwhere(grid == BLUE)}
    for r, c in blue_set:
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(grid.shape, nr, nc):
                continue
            value = int(grid[nr, nc])
            if value not in (BLACK, BLUE):
                blue_neighbors += 1
                if (nr, nc) != anchor:
                    return False
    if blue_neighbors == 0:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        shape = (random.randint(15, 17), random.randint(15, 17))
        source_offsets = _generate_source_offsets()
        overlap_specs = _overlap_specs(source_offsets)
        if not overlap_specs:
            continue
        try:
            targets, rotations = _place_targets(shape, source_offsets, overlap_specs)
            triangles = _place_triangles(shape, targets, rotations)
            blocked = set(targets)
            for tri in triangles:
                blocked |= tri
            anchor, blue_cells = _place_source(shape, source_offsets, blocked)
            grid = _build_grid(shape, anchor, blue_cells, targets, triangles)
        except (AssertionError, RuntimeError):
            continue
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
