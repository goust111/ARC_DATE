from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 1
PATTERN_COLORS = (2, 3, 4)
CENTER = 5
MARKER = 6

GEN_ATTEMPTS = 400
GRID_RANGE = (17, 23)
TARGET_REGION_RANGE = (3, 5)
MIN_REGION_SPAN = 5
MAX_PATTERNS = 3  # bounded by the three visible pattern colors


Coord = tuple[int, int]
Region = tuple[int, int, int, int]  # r0, r1, c0, c1 inclusive


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _flood_mask(grid: np.ndarray, starts: list[Coord], passable) -> np.ndarray:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[Coord] = deque()
    for r, c in starts:
        if _in_bounds(grid.shape, r, c) and not seen[r, c] and passable(r, c):
            seen[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and passable(nr, nc):
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _connected_components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[Coord] = []
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_marker(grid: np.ndarray) -> Coord:
    pts = np.argwhere(grid == MARKER)
    assert len(pts) == 1, "expected exactly one marker"
    r, c = map(int, pts[0])
    h, w = grid.shape
    assert (r, c) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}, "marker must be in a corner"
    return r, c


def _region_from_seed(grid: np.ndarray, seed: Coord) -> set[Coord]:
    rr, cc = seed
    region = _flood_mask(grid, [(rr, cc)], lambda r, c: int(grid[r, c]) != WALL)
    return {tuple(map(int, rc)) for rc in np.argwhere(region)}


def _center_is_reachable(center: Coord, reachable_bg: np.ndarray) -> bool:
    r, c = center
    if reachable_bg[r, c]:
        return True
    return any(
        _in_bounds(reachable_bg.shape, nr, nc) and reachable_bg[nr, nc]
        for nr, nc in _neighbors4(r, c)
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete only the center-symmetric half-patterns that lie in the corner-marker's reachable region system.

    Input structure:
    - The grid uses black background, blue 1-cell-thick walls, one magenta corner marker, and several gray center markers.
    - Each gray center belongs to a single wall-delimited region and has one same-colored half-pattern in that region.
    - Every half-pattern is positioned on one side of its center so its 180-degree copy also stays inside the same region.

    Rule:
    - Flood-fill from the magenta corner marker through black cells only, treating blue cells as walls.
    - For each gray center, inspect the wall-delimited region containing it. If the center is on or 4-neighbor-adjacent to the reachable background, keep that region's colored half-pattern and add its 180-degree point reflection around the gray center.
    - If the gray center is not reachable in that sense, erase the center and all colored pattern pixels in its region. Finally erase the magenta marker while leaving the blue walls unchanged.

    Key insight:
    - Reachability is the gate: the same kind of half-pattern is completed in reachable regions but deleted in unreachable ones, so flood-fill must be solved before the symmetry step.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    mr, mc = _find_marker(g)
    reachable_bg = _flood_mask(
        g,
        [(mr, mc)],
        lambda r, c: int(g[r, c]) in (BG, MARKER),
    )

    centers = [tuple(map(int, rc)) for rc in np.argwhere(g == CENTER)]
    for cr, cc in centers:
        region = _region_from_seed(g, (cr, cc))
        pattern_cells = [
            (r, c)
            for (r, c) in region
            if int(g[r, c]) not in (BG, WALL, CENTER, MARKER)
        ]
        if _center_is_reachable((cr, cc), reachable_bg):
            for r, c in pattern_cells:
                rr = 2 * cr - r
                rc = 2 * cc - c
                assert 0 <= rr < h and 0 <= rc < w
                if out[rr, rc] == BG:
                    out[rr, rc] = g[r, c]
                else:
                    assert out[rr, rc] == g[r, c], "reflected cell must stay background or same color"
        else:
            out[cr, cc] = BG
            for r, c in pattern_cells:
                out[r, c] = BG

    out[mr, mc] = BG
    return out


def _region_dims(region: Region) -> tuple[int, int]:
    r0, r1, c0, c1 = region
    return r1 - r0 + 1, c1 - c0 + 1


def _contains(region: Region, cell: Coord) -> bool:
    r0, r1, c0, c1 = region
    r, c = cell
    return r0 <= r <= r1 and c0 <= c <= c1


def _split_regions(h: int, w: int, target_count: int) -> tuple[list[Region], set[Coord]] | None:
    regions: list[Region] = [(0, h - 1, 0, w - 1)]
    walls: set[Coord] = set()
    while len(regions) < target_count:
        splittable: list[tuple[int, str]] = []
        for idx, region in enumerate(regions):
            r0, r1, c0, c1 = region
            if r1 - r0 + 1 >= 2 * MIN_REGION_SPAN + 1:
                splittable.append((idx, "H"))
            if c1 - c0 + 1 >= 2 * MIN_REGION_SPAN + 1:
                splittable.append((idx, "V"))
        if not splittable:
            return None
        idx, orient = random.choice(splittable)
        r0, r1, c0, c1 = regions.pop(idx)
        if orient == "H":
            split_r = random.randint(r0 + MIN_REGION_SPAN, r1 - MIN_REGION_SPAN)
            for c in range(c0, c1 + 1):
                walls.add((split_r, c))
            regions.append((r0, split_r - 1, c0, c1))
            regions.append((split_r + 1, r1, c0, c1))
        else:
            split_c = random.randint(c0 + MIN_REGION_SPAN, c1 - MIN_REGION_SPAN)
            for r in range(r0, r1 + 1):
                walls.add((r, split_c))
            regions.append((r0, r1, c0, split_c - 1))
            regions.append((r0, r1, split_c + 1, c1))
    return regions, walls


def _grow_connected_shape(allowed: set[Coord], seed: Coord, size: int) -> set[Coord] | None:
    if seed not in allowed:
        return None
    shape = {seed}
    while len(shape) < size:
        frontier = {
            nbr
            for cell in shape
            for nbr in _neighbors4(*cell)
            if nbr in allowed and nbr not in shape
        }
        if not frontier:
            return None
        shape.add(random.choice(tuple(frontier)))
    return shape


def _place_pattern_in_region(
    grid: np.ndarray,
    region: Region,
    color: int,
    forbidden: set[Coord],
) -> dict[str, object] | None:
    r0, r1, c0, c1 = region
    orientations = ["left", "right", "up", "down"]
    for _ in range(160):
        orient = random.choice(orientations)
        cr = random.randint(r0 + 1, r1 - 1)
        cc = random.randint(c0 + 1, c1 - 1)
        center = (cr, cc)
        if center in forbidden or int(grid[cr, cc]) != BG:
            continue

        allowed: set[Coord] = set()
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r, c) == center or (r, c) in forbidden:
                    continue
                if orient == "left" and not (c < cc):
                    continue
                if orient == "right" and not (c > cc):
                    continue
                if orient == "up" and not (r < cr):
                    continue
                if orient == "down" and not (r > cr):
                    continue
                rr = 2 * cr - r
                rc = 2 * cc - c
                if not _contains(region, (rr, rc)):
                    continue
                if (rr, rc) in forbidden or (rr, rc) == center:
                    continue
                allowed.add((r, c))

        seed_map = {
            "left": (cr, cc - 1),
            "right": (cr, cc + 1),
            "up": (cr - 1, cc),
            "down": (cr + 1, cc),
        }
        seed = seed_map[orient]
        if seed not in allowed:
            continue
        max_size = min(8, len(allowed))
        if max_size < 3:
            continue
        target_size = random.randint(3, max_size)
        shape = _grow_connected_shape(allowed, seed, target_size)
        if shape is None:
            continue

        mirrors = {(2 * cr - r, 2 * cc - c) for r, c in shape}
        if shape & mirrors:
            continue
        if any(int(grid[r, c]) != BG for r, c in shape | mirrors):
            continue

        grid[cr, cc] = CENTER
        for r, c in shape:
            grid[r, c] = color
        return {
            "center": center,
            "color": color,
            "source": shape,
            "mirrors": mirrors,
            "orientation": orient,
            "region": region,
        }
    return None


def _count_regions_without_walls(grid: np.ndarray) -> int:
    mask = grid != WALL
    return len(_connected_components(mask))


def _validate_generated(
    grid: np.ndarray,
    regions: list[Region],
    marker: Coord,
    patterns: list[dict[str, object]],
    reachable_region: Region,
) -> bool:
    if max(grid.shape) > 30:
        return False
    if _count_regions_without_walls(grid) != len(regions):
        return False
    if len(patterns) < 2 or len(patterns) > MAX_PATTERNS:
        return False
    centers = [item["center"] for item in patterns]
    if len(set(centers)) != len(centers):
        return False
    if reachable_region not in [item["region"] for item in patterns]:
        return False
    if sum(item["region"] != reachable_region for item in patterns) < 1:
        return False

    reachable_bg = _flood_mask(
        grid,
        [marker],
        lambda r, c: int(grid[r, c]) in (BG, MARKER),
    )
    for item in patterns:
        center = item["center"]
        source = item["source"]
        mirrors = item["mirrors"]
        region = item["region"]
        if len(source) < 3 or len(source) > 8:
            return False
        source_mask = np.zeros(grid.shape, dtype=bool)
        for r, c in source:
            source_mask[r, c] = True
        if len(_connected_components(source_mask)) != 1:
            return False
        actual_reachable = _center_is_reachable(center, reachable_bg)
        if (region == reachable_region) != actual_reachable:
            return False
        color = int(item["color"])
        for r, c in source:
            if int(grid[r, c]) != color:
                return False
        for r, c in mirrors:
            if int(grid[r, c]) != BG:
                return False
        region_cells = _region_from_seed(grid, center)
        region_colors = {
            (r, c)
            for (r, c) in region_cells
            if int(grid[r, c]) not in (BG, WALL, CENTER, MARKER)
        }
        if region_colors != set(source):
            return False

    expected = grid.copy()
    for item in patterns:
        center = item["center"]
        source = item["source"]
        mirrors = item["mirrors"]
        region = item["region"]
        color = int(item["color"])
        cr, cc = center
        if region == reachable_region:
            for r, c in mirrors:
                expected[r, c] = color
        else:
            expected[cr, cc] = BG
            for r, c in source:
                expected[r, c] = BG
    expected[marker] = BG
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    added = int(np.sum((grid == BG) & (actual != BG)))
    erased = int(np.sum((grid != BG) & (actual == BG)))
    return added > 0 and erased > 1


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    target_regions = random.randint(*TARGET_REGION_RANGE)
    split_result = _split_regions(h, w, target_regions)
    if split_result is None:
        return None
    regions, walls = split_result

    grid = np.full((h, w), BG, dtype=int)
    for r, c in walls:
        grid[r, c] = WALL

    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    marker = random.choice(corners)
    reachable_region = next(region for region in regions if _contains(region, marker))
    grid[marker] = MARKER

    usable_regions = [region for region in regions if min(_region_dims(region)) >= 5]
    if len(usable_regions) < 2 or reachable_region not in usable_regions:
        return None

    pattern_count = random.randint(2, min(MAX_PATTERNS, len(usable_regions)))
    other_regions = [region for region in usable_regions if region != reachable_region]
    chosen_regions = [reachable_region] + random.sample(other_regions, pattern_count - 1)
    random.shuffle(chosen_regions)
    colors = random.sample(PATTERN_COLORS, len(chosen_regions))

    forbidden = {marker}
    patterns: list[dict[str, object]] = []
    for region, color in zip(chosen_regions, colors):
        item = _place_pattern_in_region(grid, region, color, forbidden)
        if item is None:
            return None
        forbidden.add(item["center"])  # type: ignore[arg-type]
        forbidden |= set(item["source"])  # type: ignore[arg-type]
        forbidden |= set(item["mirrors"])  # type: ignore[arg-type]
        patterns.append(item)

    if not _validate_generated(grid, regions, marker, patterns, reachable_region):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for reachability-gated point symmetry task")
