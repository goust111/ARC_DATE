from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALLOWED_OFFSETS = {
    (dr, dc)
    for dr in range(-5, 6)
    for dc in range(-5, 6)
    if 2 <= abs(dr) + abs(dc) <= 5
}

# These source shapes were pre-validated to yield both overlap branches after
# four-way rotation: at least one multi-pixel overlap component and at least one
# isolated overlap pixel. Small (<=8) connected shapes cannot realize both
# branches simultaneously under the stated geometry, so the generator uses the
# smallest verified family above that threshold.
SOURCE_SHAPES: list[tuple[tuple[int, int], ...]] = [
    ((-3, -1), (-2, -1), (-2, 0), (-2, 1), (-1, -2), (-1, -1), (0, -3), (0, -2), (1, -3)),
    ((-3, 2), (-2, 0), (-2, 1), (-2, 2), (-1, 1), (-1, 2), (0, 2), (1, 2), (1, 3), (2, 3)),
    ((-3, 0), (-2, -1), (-2, 0), (-2, 1), (-2, 3), (-1, -1), (-1, 1), (-1, 2), (-1, 3), (0, 3)),
    ((-1, 2), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2), (2, 0), (2, 1), (3, 1), (4, 0), (4, 1)),
    ((-2, 3), (-1, 2), (-1, 3), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (3, 2)),
    ((-1, -3), (0, -3), (0, -2), (1, -3), (1, -2), (1, -1), (2, -3), (2, -2), (2, -1), (2, 0), (2, 1), (3, -1)),
    ((-3, -2), (-3, -1), (-2, -3), (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-1, -2), (-1, 1), (-1, 2), (-1, 3), (-1, 4)),
    ((-3, 1), (-2, 1), (-1, 1), (-1, 2), (0, 2), (0, 3), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 2)),
    ((-4, 1), (-3, 0), (-3, 1), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, -1), (-1, 1), (-1, 2), (-1, 3), (0, 3)),
    ((-1, 1), (-1, 2), (0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (3, -1), (3, 0), (3, 1)),
    ((1, -1), (1, 1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (2, 3), (3, -2), (3, -1), (3, 0), (4, -1)),
    ((-1, 3), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, -1), (2, 0), (2, 1), (2, 2), (3, 1)),
]


def _rotate_offset_cw(dr: int, dc: int, turns: int) -> tuple[int, int]:
    turns %= 4
    for _ in range(turns):
        dr, dc = dc, -dr
    return dr, dc


def _transform_offsets(
    offsets: set[tuple[int, int]],
    turns: int,
    flip_h: bool,
) -> set[tuple[int, int]]:
    out = set()
    for dr, dc in offsets:
        if flip_h:
            dc = -dc
        out.add(_rotate_offset_cw(dr, dc, turns))
    return out


def _components(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    comps: list[list[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for start in cells:
        if start in seen:
            continue
        q: deque[tuple[int, int]] = deque([start])
        seen.add(start)
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in DIRS4:
                nxt = (r + dr, c + dc)
                if nxt in cells and nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(comp)
    return comps


def _is_connected(offsets: set[tuple[int, int]]) -> bool:
    if not offsets:
        return False
    return len(_components(offsets)) == 1


def _coverage_counts(offsets: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for dr, dc in offsets:
        for turns in range(4):
            rr, cc = _rotate_offset_cw(dr, dc, turns)
            counts[(rr, cc)] = counts.get((rr, cc), 0) + 1
    return counts


def _has_required_overlap(offsets: set[tuple[int, int]]) -> bool:
    if not _is_connected(offsets):
        return False
    if _transform_offsets(offsets, 1, False) == offsets:
        return False
    counts = _coverage_counts(offsets)
    overlap = {cell for cell, count in counts.items() if count >= 2}
    if not overlap:
        return False
    comp_sizes = [len(comp) for comp in _components(overlap)]
    return any(size > 1 for size in comp_sizes) and any(size == 1 for size in comp_sizes) and any(
        count == 1 for count in counts.values()
    )


def _candidate_additions(offsets: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in offsets:
        for ddr, ddc in DIRS4:
            nxt = (dr + ddr, dc + ddc)
            if nxt in ALLOWED_OFFSETS and nxt not in offsets:
                out.append(nxt)
    random.shuffle(out)
    dedup: list[tuple[int, int]] = []
    seen = set()
    for cell in out:
        if cell not in seen:
            seen.add(cell)
            dedup.append(cell)
    return dedup


def _removable_leaves(offsets: set[tuple[int, int]]) -> list[tuple[int, int]]:
    leaves: list[tuple[int, int]] = []
    for cell in offsets:
        degree = sum((cell[0] + dr, cell[1] + dc) in offsets for dr, dc in DIRS4)
        if degree <= 1 and len(offsets) > 1:
            trial = set(offsets)
            trial.remove(cell)
            if _is_connected(trial):
                leaves.append(cell)
    random.shuffle(leaves)
    return leaves


def _mutate_shape(offsets: set[tuple[int, int]]) -> set[tuple[int, int]]:
    current = set(offsets)
    for _ in range(random.randint(0, 3)):
        leaves = _removable_leaves(current)
        adds = _candidate_additions(current)
        changed = False
        for remove_cell in leaves:
            for add_cell in adds:
                if add_cell == remove_cell:
                    continue
                trial = set(current)
                trial.remove(remove_cell)
                trial.add(add_cell)
                if _has_required_overlap(trial):
                    current = trial
                    changed = True
                    break
            if changed:
                break
    return current


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the blue source shape around the gray pivot, overlay the four copies, and recolor overlaps by 4-connectivity.

    Input structure:
    - The grid is square with black background.
    - Exactly one gray pixel marks the rotation pivot.
    - A single irregular blue 4-connected shape sits away from the pivot; all other
      cells are background.

    Rule:
    - Use the gray pixel as the center and rotate the blue shape by 0, 90, 180, and
      270 degrees clockwise.
    - Overlay the four copies and count how many copies cover each non-pivot cell.
    - Keep count-1 cells blue; among cells with count >= 2, recolor 4-connected
      overlap components of area > 1 to red and singleton overlap cells to yellow.
    - Leave the pivot gray and the background black.

    Key insight:
    - The final colors are determined by overlap connectivity after rotation, not by
      the original blue shape alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, GREY}), "unexpected colors in input"

    pivot_coords = np.argwhere(g == GREY)
    assert len(pivot_coords) == 1, "expected exactly one gray pivot"
    pr, pc = map(int, pivot_coords[0])

    source_coords = np.argwhere(g == BLUE)
    assert len(source_coords) > 0, "expected a blue source shape"
    offsets = {(int(r) - pr, int(c) - pc) for r, c in source_coords}

    counts: dict[tuple[int, int], int] = {}
    for dr, dc in offsets:
        for turns in range(4):
            rr, cc = _rotate_offset_cw(dr, dc, turns)
            cell = (pr + rr, pc + cc)
            counts[cell] = counts.get(cell, 0) + 1

    out = np.full_like(g, BLACK)
    out[pr, pc] = GREY

    overlap_cells = {cell for cell, count in counts.items() if count >= 2}
    for cell, count in counts.items():
        if count == 1:
            out[cell] = BLUE

    for comp in _components(overlap_cells):
        color = RED if len(comp) > 1 else YELLOW
        for r, c in comp:
            out[r, c] = color

    return out


def _build_input_from_offsets(
    n: int,
    pivot: tuple[int, int],
    offsets: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((n, n), BLACK, dtype=int)
    pr, pc = pivot
    grid[pr, pc] = GREY
    for dr, dc in offsets:
        grid[pr + dr, pc + dc] = BLUE
    return grid


def _center_band(n: int) -> list[int]:
    center = n // 2
    low = max(5, center - 1)
    high = min(n - 6, center + 1)
    if low > high:
        return [center]
    return list(range(low, high + 1))


def _sample_offsets() -> set[tuple[int, int]]:
    base = set(random.choice(SOURCE_SHAPES))
    transformed = _transform_offsets(base, random.randrange(4), bool(random.getrandbits(1)))
    return _mutate_shape(transformed)


def generate_input() -> np.ndarray:
    for _ in range(400):
        n = random.choice([13, 15, 17, 19])
        rows = _center_band(n)
        cols = _center_band(n)
        pivot = (random.choice(rows), random.choice(cols))
        offsets = _sample_offsets()

        if not offsets <= ALLOWED_OFFSETS:
            continue
        if not _has_required_overlap(offsets):
            continue

        grid = _build_input_from_offsets(n, pivot, offsets)
        out = main(grid)
        colors = set(map(int, np.unique(out)))
        if not {BLUE, RED, YELLOW, GREY}.issubset(colors):
            continue
        return grid

    raise RuntimeError("failed to generate a valid rotation-overlap ARC input")
