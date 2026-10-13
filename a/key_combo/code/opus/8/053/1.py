from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


Coord = tuple[int, int]
Object = dict[str, object]

BG = 0
CORE = 5
PALETTE = [1, 2, 3, 4, 7]
AREA_CHOICES = [3, 4, 5, 6, 7, 8]
GRID_SIDE_RANGE = (13, 16)
OBJECT_COUNT_RANGE = (3, 5)
MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _normalize(cells: set[Coord]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _transform_coords(coords: tuple[Coord, ...], rot: int, flip: bool) -> tuple[Coord, ...]:
    arr = np.array(coords, dtype=int)
    if flip:
        arr[:, 1] *= -1
    for _ in range(rot % 4):
        arr = np.stack([-arr[:, 1], arr[:, 0]], axis=1)
    cells = {(int(r), int(c)) for r, c in arr}
    return _normalize(cells)


def _shape_from_area(area: int) -> tuple[Coord, ...]:
    for _ in range(300):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if -4 <= nr <= 4 and -4 <= nc <= 4 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            cells.add(random.choice(tuple(frontier)))

        coords = _normalize(cells)
        if random.random() < 0.5:
            coords = _transform_coords(coords, random.randrange(4), True)
        else:
            coords = _transform_coords(coords, random.randrange(4), False)

        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        h = max(rows) + 1
        w = max(cols) + 1
        if not (2 <= h <= 4 and 2 <= w <= 4):
            continue
        if area == h * w:
            continue
        if len(coords) != area:
            continue
        return coords
    raise RuntimeError(f"failed to build irregular shape of area {area}")


def _extract_objects(grid: np.ndarray) -> list[Object]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[Object] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords.sort()
            objects.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "area": len(coords),
                }
            )
    return objects


def _halo_claims(grid: np.ndarray, objects: list[Object]) -> dict[Coord, list[int]]:
    h, w = grid.shape
    claims: dict[Coord, list[int]] = defaultdict(list)
    for idx, obj in enumerate(objects):
        for r, c in obj["coords"]:
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    if int(grid[rr, cc]) == BG:
                        claims[(rr, cc)].append(idx)
    return claims


def _min_chebyshev(coords_a: tuple[Coord, ...], coords_b: tuple[Coord, ...]) -> int:
    best = 10**9
    for ra, ca in coords_a:
        for rb, cb in coords_b:
            best = min(best, max(abs(ra - rb), abs(ca - cb)))
    return best


def _winner_for_cell(objects: list[Object], claimants: list[int]) -> int:
    unique_claimants = sorted(set(claimants))
    areas = [int(objects[idx]["area"]) for idx in unique_claimants]
    assert len(set(areas)) == len(areas), "areas must be distinct for deterministic arbitration"
    return max(unique_claimants, key=lambda idx: int(objects[idx]["area"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each object by one halo cell, arbitrate halo conflicts by object area, and grey-mark the original cores.

    Input structure:
    - A black background contains several 4-connected monochrome foreground objects.
    - Each object is a small irregular shape, and different objects are spatially separated.
    - Objects may have distinct one-step halo candidates that overlap on background cells.

    Rule:
    - Identify every 4-connected foreground object and record its pixel area.
    - For each object, mark every black cell in the Chebyshev-distance-1 neighborhood of any object pixel as that object's candidate halo.
    - If a background cell is claimed by multiple objects, assign it to the unique claimant with the largest area; otherwise keep the sole claimant's color.
    - After all halo cells are colored, recolor every original object pixel to grey.

    Key insight:
    - The result depends on combining local halo geometry with a global per-object area comparison, because only the larger claimant owns each disputed halo cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _extract_objects(g)
    if not objects:
        return g.copy()

    areas = [int(obj["area"]) for obj in objects]
    assert len(set(areas)) == len(areas), "foreground component areas must be unique"

    claims = _halo_claims(g, objects)
    out = g.copy()
    for cell, claimants in claims.items():
        winner = _winner_for_cell(objects, claimants)
        out[cell] = int(objects[winner]["color"])

    for obj in objects:
        for r, c in obj["coords"]:
            out[r, c] = CORE
    return out


def _candidate_positions(shape: tuple[Coord, ...], h: int, w: int) -> list[tuple[Coord, ...]]:
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1
    positions: list[tuple[Coord, ...]] = []
    for r0 in range(h - shape_h + 1):
        for c0 in range(w - shape_w + 1):
            positions.append(tuple(sorted((r0 + r, c0 + c) for r, c in shape)))
    random.shuffle(positions)
    return positions


def _place_objects(shapes: list[tuple[Coord, ...]], h: int, w: int) -> list[tuple[Coord, ...]] | None:
    placed: list[tuple[Coord, ...]] = []
    candidate_lists = [_candidate_positions(shape, h, w) for shape in shapes]

    def backtrack(i: int) -> bool:
        if i == len(shapes):
            return True
        for coords in candidate_lists[i]:
            ok = True
            for other in placed:
                dist = _min_chebyshev(coords, other)
                if dist < 2 or dist > 4:
                    ok = False
                    break
            if not ok:
                continue
            placed.append(coords)
            if backtrack(i + 1):
                return True
            placed.pop()
        return False

    return placed if backtrack(0) else None


def _render_input(objects: list[Object], h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["coords"]:
            grid[r, c] = color
    return grid


def _build_objects(h: int, w: int) -> list[Object] | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    areas = random.sample(AREA_CHOICES, count)
    random.shuffle(areas)
    colors = random.sample(PALETTE, count)
    shapes = [_shape_from_area(area) for area in areas]

    order = list(range(count))
    random.shuffle(order)
    ordered_shapes = [shapes[i] for i in order]
    ordered_areas = [areas[i] for i in order]
    ordered_colors = [colors[i] for i in order]
    placed = _place_objects(ordered_shapes, h, w)
    if placed is None:
        return None

    return [
        {
            "color": ordered_colors[i],
            "coords": placed[i],
            "area": ordered_areas[i],
        }
        for i in range(count)
    ]


def _semantic_check(inp: np.ndarray) -> bool:
    objects = _extract_objects(inp)
    if not (OBJECT_COUNT_RANGE[0] <= len(objects) <= OBJECT_COUNT_RANGE[1]):
        return False

    colors = [int(obj["color"]) for obj in objects]
    areas = [int(obj["area"]) for obj in objects]
    if len(set(colors)) != len(colors) or len(set(areas)) != len(areas):
        return False
    if any(color not in PALETTE for color in colors):
        return False
    if any(not (3 <= area <= 8) for area in areas):
        return False

    for obj in objects:
        coords = tuple(obj["coords"])
        norm = _normalize(set(coords))
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if int(obj["area"]) == h * w:
            return False

    pairwise_distances = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            dist = _min_chebyshev(tuple(objects[i]["coords"]), tuple(objects[j]["coords"]))
            pairwise_distances.append(dist)
            if not (2 <= dist <= 4):
                return False
    if not pairwise_distances or 2 not in pairwise_distances:
        return False

    claims = _halo_claims(inp, objects)
    conflict_cells = {cell: claimants for cell, claimants in claims.items() if len(set(claimants)) > 1}
    uncontested = [cell for cell, claimants in claims.items() if len(claimants) == 1]
    if len(conflict_cells) < 2 or not uncontested:
        return False

    contested_winners = {_winner_for_cell(objects, claimants) for claimants in conflict_cells.values()}
    if len(contested_winners) < 2:
        return False

    owner_halo_counts = [0] * len(objects)
    for claimants in claims.values():
        owner_halo_counts[_winner_for_cell(objects, claimants)] += 1
    if any(count == 0 for count in owner_halo_counts):
        return False

    out = main(inp)
    if np.array_equal(inp, out):
        return False
    if int(np.count_nonzero(out == CORE)) != sum(areas):
        return False

    for obj in objects:
        color = int(obj["color"])
        coords = tuple(obj["coords"])
        for r, c in coords:
            if int(out[r, c]) != CORE:
                return False
        halo_count = 0
        for cell, claimants in claims.items():
            if _winner_for_cell(objects, claimants) == objects.index(obj):
                if int(out[cell]) != color:
                    return False
                halo_count += 1
        if halo_count == 0:
            return False

    if len(set(map(int, out.flatten()))) < 4:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        h = random.randint(*GRID_SIDE_RANGE)
        w = random.randint(*GRID_SIDE_RANGE)
        objects = _build_objects(h, w)
        if objects is None:
            continue
        inp = _render_input(objects, h, w)
        if _semantic_check(inp):
            return inp
    raise RuntimeError("failed to generate a valid halo-arbitration ARC input")
