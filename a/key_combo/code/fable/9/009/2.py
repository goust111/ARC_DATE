from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]

BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                neigh = _neighbors4 if connectivity == 4 else _neighbors8
                for nr, nc in neigh(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _background_components(occupied: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    h, w = occupied.shape
    seen = np.zeros_like(occupied, dtype=bool)
    holes: list[np.ndarray] = []
    exterior = np.zeros_like(occupied, dtype=bool)

    for r in range(h):
        for c in range(w):
            if occupied[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            touches_border = r in (0, h - 1) or c in (0, w - 1)
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not occupied[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        if nr in (0, h - 1) or nc in (0, w - 1):
                            touches_border = True
                        q.append((nr, nc))
            comp = np.array(coords, dtype=int)
            if touches_border:
                exterior[comp[:, 0], comp[:, 1]] = True
            else:
                holes.append(comp)
    return holes, exterior


def _hole_wall_counts(hole: np.ndarray, left_occ: np.ndarray, right_occ: np.ndarray) -> tuple[int, int]:
    h, w = left_occ.shape
    wall_cells: set[Coord] = set()
    for r, c in hole:
        for nr, nc in _neighbors8(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and (left_occ[nr, nc] or right_occ[nr, nc]):
                wall_cells.add((nr, nc))
    left_count = sum(1 for r, c in wall_cells if left_occ[r, c])
    right_count = sum(1 for r, c in wall_cells if right_occ[r, c])
    return left_count, right_count


def _solve(left: np.ndarray, right: np.ndarray, left_color: int, right_color: int) -> np.ndarray:
    left_occ = left != BG
    right_occ = right != BG
    assert not np.any(left_occ & right_occ), "ownership must stay unambiguous"

    union = left_occ | right_occ
    holes, _exterior = _background_components(union)
    out = np.full(left.shape, BG, dtype=int)
    for hole in holes:
        left_count, right_count = _hole_wall_counts(hole, left_occ, right_occ)
        assert left_count != right_count, "every hole must have a strict wall-majority owner"
        out[hole[:, 0], hole[:, 1]] = left_color if left_count > right_count else right_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay the two aligned panels as one wall and fill only emergent holes with their wall-majority owner color.

    Input structure:
    - The input width is split into two equal side-by-side panels with black background.
    - The left and right panels each use exactly one distinct foreground color, and their occupied cells do not overlap when aligned.
    - Each panel alone contains only open wall fragments, but their occupancy union forms one or more enclosed background holes.

    Rule:
    - Split the grid into equal left and right panels and align them by row and column.
    - Treat the union of their occupied cells as walls; boundary-connected union background stays black.
    - For each enclosed hole, collect the union wall cells that are 8-neighbors of that hole and count how many come from the left panel versus the right panel.
    - Fill the whole hole with the foreground color of the panel owning a strict majority of those adjacent wall cells, and keep every wall cell black in the output.

    Key insight:
    - No panel is sufficient on its own: enclosure appears only after transparent union, and each hole's fill color is determined locally by the ownership majority of the surrounding combined wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert w % 2 == 0, "expected two equal-width panels"

    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    assert bg == BG, "expected black background"

    mid = w // 2
    left = g[:, :mid]
    right = g[:, mid:]
    left_colors = sorted(set(map(int, left.ravel())) - {bg})
    right_colors = sorted(set(map(int, right.ravel())) - {bg})
    assert len(left_colors) == 1, "expected one foreground color in the left panel"
    assert len(right_colors) == 1, "expected one foreground color in the right panel"
    assert left_colors[0] != right_colors[0], "panel colors must differ"

    return _solve(left, right, left_colors[0], right_colors[0])


def _rectangle_perimeter(height: int, width: int) -> tuple[list[Coord], list[Coord], list[Coord]]:
    cells: list[Coord] = []
    for c in range(width):
        cells.append((0, c))
    for r in range(1, height - 1):
        cells.append((r, width - 1))
    for c in range(width - 1, -1, -1):
        cells.append((height - 1, c))
    for r in range(height - 2, 0, -1):
        cells.append((r, 0))

    corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
    corner_set = set(corners)
    noncorners = [cell for cell in cells if cell not in corner_set]
    return cells, noncorners, corners


def _partition_perimeter(
    height: int,
    width: int,
    majority_owner: str,
    mode: str,
    rng: random.Random,
) -> tuple[set[Coord], set[Coord], bool]:
    perimeter, noncorners, corners = _rectangle_perimeter(height, width)
    left_cut, right_cut = rng.sample(noncorners, 2)
    left_cells: set[Coord] = set()
    right_cells: set[Coord] = set()

    if mode == "corner_sensitive":
        half_noncorner = len(noncorners) // 2
        if majority_owner == "left":
            right_noncorners = {left_cut}
            pool = [cell for cell in noncorners if cell not in {left_cut, right_cut}]
            right_noncorners.update(rng.sample(pool, half_noncorner - 1))
            right_cells = right_noncorners
            left_cells = set(noncorners) - right_noncorners
            left_cells.update(corners)
        else:
            left_noncorners = {right_cut}
            pool = [cell for cell in noncorners if cell not in {left_cut, right_cut}]
            left_noncorners.update(rng.sample(pool, half_noncorner - 1))
            left_cells = left_noncorners
            right_cells = set(noncorners) - left_noncorners
            right_cells.update(corners)
        return left_cells, right_cells, True

    total = len(perimeter)
    max_minority = (total - 1) // 2
    minority_total = rng.randint(2, max_minority)
    if majority_owner == "left":
        right_cells = {left_cut}
        pool = [cell for cell in perimeter if cell not in {left_cut, right_cut}]
        right_cells.update(rng.sample(pool, minority_total - 1))
        left_cells = set(perimeter) - right_cells
    else:
        left_cells = {right_cut}
        pool = [cell for cell in perimeter if cell not in {left_cut, right_cut}]
        left_cells.update(rng.sample(pool, minority_total - 1))
        right_cells = set(perimeter) - left_cells
    return left_cells, right_cells, False


def _place_rectangles(h: int, w: int, count: int, rng: random.Random):
    rectangles = []
    for _ in range(120):
        rectangles.clear()
        inflated = np.zeros((h, w), dtype=bool)
        ok = True
        for _idx in range(count):
            rect_h = rng.randint(4, min(8, h - 2))
            rect_w = rng.randint(4, min(8, w - 2))
            spots: list[tuple[int, int]] = []
            for r0 in range(1, h - rect_h):
                for c0 in range(1, w - rect_w):
                    r1 = r0 + rect_h - 1
                    c1 = c0 + rect_w - 1
                    rr0 = max(0, r0 - 2)
                    cc0 = max(0, c0 - 2)
                    rr1 = min(h - 1, r1 + 2)
                    cc1 = min(w - 1, c1 + 2)
                    if not inflated[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                        spots.append((r0, c0))
            if not spots:
                ok = False
                break
            r0, c0 = rng.choice(spots)
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1
            rectangles.append((r0, c0, rect_h, rect_w))
            rr0 = max(0, r0 - 2)
            cc0 = max(0, c0 - 2)
            rr1 = min(h - 1, r1 + 2)
            cc1 = min(w - 1, c1 + 2)
            inflated[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        if ok:
            return rectangles
    return None


def _background_hole_count(occupied: np.ndarray) -> int:
    holes, _exterior = _background_components(occupied)
    return len(holes)


def _solve_with_4_neighbor_owner(left: np.ndarray, right: np.ndarray, left_color: int, right_color: int) -> np.ndarray:
    left_occ = left != BG
    right_occ = right != BG
    union = left_occ | right_occ
    holes, _exterior = _background_components(union)
    out = np.full(left.shape, BG, dtype=int)
    h, w = left.shape
    for hole in holes:
        wall_cells: set[Coord] = set()
        for r, c in hole:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and union[nr, nc]:
                    wall_cells.add((nr, nc))
        left_count = sum(1 for r, c in wall_cells if left_occ[r, c])
        right_count = sum(1 for r, c in wall_cells if right_occ[r, c])
        if left_count == right_count:
            return out
        out[hole[:, 0], hole[:, 1]] = left_color if left_count > right_count else right_color
    return out


def _build_instance(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(11, 16)
    w = rng.randint(11, 15)
    hole_count = rng.choices([1, 2, 3], weights=[2, 4, 3], k=1)[0]
    rectangles = _place_rectangles(h, w, hole_count, rng)
    if rectangles is None:
        return None

    left_color, right_color = rng.sample(PALETTE, 2)
    left = np.full((h, w), BG, dtype=int)
    right = np.full((h, w), BG, dtype=int)

    if hole_count >= 2 and rng.random() < 0.7:
        owners = ["left", "right"]
        while len(owners) < hole_count:
            owners.append(rng.choice(["left", "right"]))
        rng.shuffle(owners)
    else:
        owners = [rng.choice(["left", "right"]) for _ in range(hole_count)]

    modes = []
    for i in range(hole_count):
        if hole_count == 1:
            mode = "corner_sensitive"
        else:
            mode = rng.choice(["corner_sensitive", "strong_majority"])
        modes.append(mode)

    corner_sensitive_used = False
    for (r0, c0, rect_h, rect_w), owner, mode in zip(rectangles, owners, modes):
        left_cells, right_cells, used_corner_sensitive = _partition_perimeter(rect_h, rect_w, owner, mode, rng)
        corner_sensitive_used = corner_sensitive_used or used_corner_sensitive
        for rr, cc in left_cells:
            left[r0 + rr, c0 + cc] = left_color
        for rr, cc in right_cells:
            right[r0 + rr, c0 + cc] = right_color

    if np.any((left != BG) & (right != BG)):
        return None

    left_occ = left != BG
    right_occ = right != BG
    union = left_occ | right_occ

    if _background_hole_count(left_occ) != 0 or _background_hole_count(right_occ) != 0:
        return None

    holes, exterior = _background_components(union)
    if len(holes) != hole_count:
        return None
    if not exterior.any():
        return None

    owner_set = set()
    for hole in holes:
        left_count, right_count = _hole_wall_counts(hole, left_occ, right_occ)
        if left_count == right_count:
            return None
        owner_set.add("left" if left_count > right_count else "right")

    out = _solve(left, right, left_color, right_color)
    if np.count_nonzero(out) == 0:
        return None

    both_empty = ~(union)
    if np.array_equal(out != BG, both_empty):
        return None

    if hole_count == 1:
        if np.array_equal(out, _solve_with_4_neighbor_owner(left, right, left_color, right_color)):
            return None
    else:
        if len(owner_set) == 1 and not corner_sensitive_used:
            return None

    grid = np.concatenate([left, right], axis=1)
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_instance(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid emergent-union enclosure instance")
