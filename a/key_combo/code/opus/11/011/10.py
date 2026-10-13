from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
FILL_COLORS = (2, 3, 4)

GRID_SIZE_RANGE = (20, 27)
PLACEMENT_GAP = 2
GENERATE_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _enclosed_regions(
    wall_coords: np.ndarray, box: tuple[int, int, int, int]
) -> list[np.ndarray]:
    r0, c0, r1, c1 = map(int, box)
    h, w = r1 - r0 + 1, c1 - c0 + 1

    wall = np.zeros((h, w), dtype=bool)
    local = wall_coords - np.array([r0, c0], dtype=int)
    wall[local[:, 0], local[:, 1]] = True
    passable = ~wall

    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for rr in range(h):
        for cc in (0, w - 1):
            if passable[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))
    for cc in range(w):
        for rr in (0, h - 1):
            if passable[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    enclosed = passable & ~outside
    return [coords + np.array([r0, c0], dtype=int) for coords in _components_mask(enclosed)]


def _component_info(grid: np.ndarray) -> list[dict]:
    infos: list[dict] = []
    for coords in _components_mask(grid == BLUE):
        box = _bbox(coords)
        regions = _enclosed_regions(coords, box)
        solid_regions: list[np.ndarray] = []
        nested = False
        for region in regions:
            values = grid[region[:, 0], region[:, 1]]
            if np.all(values == BG):
                solid_regions.append(region)
            else:
                nested = True
        infos.append(
            {
                "coords": coords,
                "bbox": box,
                "regions": regions,
                "solid_regions": solid_regions,
                "is_open": len(regions) == 0,
                "is_nested_closed": len(regions) > 0 and nested,
            }
        )
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only solid enclosed blue regions with area-ranked horizontal stripes.

    Input structure:
    - The background is black.
    - Blue pixels form several disjoint 4-connected objects: some are closed boundaries, some are open curves, and some closed boundaries contain smaller blue structures inside.
    - Solid closed boundaries surround only black interior cells, while nested closed boundaries surround an interior that already contains blue pixels.

    Rule:
    - Find each blue 4-connected component and test whether it encloses any region disconnected from the outside of its bbox.
    - Keep only the enclosed regions whose entire interior is black; closed boundaries containing nested blue structure are not fill targets.
    - Sort those solid enclosed regions by area from small to large and assign fill colors cyclically: red, green, yellow, then repeat.
    - Inside each selected region, color every other row starting from the region's top interior row; the alternating rows stay black.

    Key insight:
    - The decision is purely topological: a region must be both closed and free of nested foreground, and the fill color depends on interior area order rather than position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    targets: list[np.ndarray] = []
    for comp in _component_info(g):
        targets.extend(comp["solid_regions"])

    for index, region in enumerate(sorted(targets, key=lambda coords: int(coords.shape[0]))):
        color = FILL_COLORS[index % len(FILL_COLORS)]
        top = int(region[:, 0].min())
        rows = region[:, 0]
        cols = region[:, 1]
        paint = ((rows - top) % 2) == 0
        out[rows[paint], cols[paint]] = color

    return out


def _rect_loop(interior_h: int, interior_w: int) -> np.ndarray:
    patch = np.zeros((interior_h + 2, interior_w + 2), dtype=int)
    patch[0, :] = BLUE
    patch[-1, :] = BLUE
    patch[:, 0] = BLUE
    patch[:, -1] = BLUE
    return patch


def _open_patch() -> np.ndarray:
    for _ in range(80):
        h = random.randint(4, 8)
        w = random.randint(5, 10)
        cells: set[tuple[int, int]] = set()
        jogs = random.choice((1, 2))
        if jogs == 1:
            r0, r1 = random.sample(range(h), 2)
            c0 = random.randint(1, w - 2)
            points = [(r0, 0), (r0, c0), (r1, c0), (r1, w - 1)]
        else:
            rows = random.sample(range(h), 3)
            r0, r1, r2 = rows
            c0 = random.randint(1, w - 3)
            c1 = random.randint(c0 + 1, w - 2)
            points = [(r0, 0), (r0, c0), (r1, c0), (r1, c1), (r2, c1), (r2, w - 1)]
        for (r0, c0), (r1, c1) in zip(points, points[1:]):
            if r0 == r1:
                lo, hi = sorted((c0, c1))
                for cc in range(lo, hi + 1):
                    cells.add((r0, cc))
            else:
                lo, hi = sorted((r0, r1))
                for rr in range(lo, hi + 1):
                    cells.add((rr, c0))

        patch = np.zeros((h, w), dtype=int)
        for r, c in cells:
            patch[r, c] = BLUE
        if random.random() < 0.5:
            patch = patch.T
        if random.random() < 0.5:
            patch = np.flipud(patch)
        if random.random() < 0.5:
            patch = np.fliplr(patch)

        coords = np.argwhere(patch == BLUE)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        patch = patch[r0 : r1 + 1, c0 : c1 + 1]

        info = _component_info(patch)
        if len(info) == 1 and info[0]["is_open"]:
            return patch
    raise RuntimeError("failed to generate open patch")


def _random_transform(patch: np.ndarray) -> np.ndarray:
    out = np.rot90(patch, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _sample_unique_loop(used_areas: set[int], interior_range: tuple[int, int]) -> tuple[np.ndarray, int]:
    low, high = interior_range
    candidates = [
        (ih, iw)
        for ih in range(low, high + 1)
        for iw in range(low, high + 1)
        if ih * iw not in used_areas
    ]
    if not candidates:
        raise RuntimeError("exhausted unique loop areas")
    ih, iw = random.choice(candidates)
    return _random_transform(_rect_loop(ih, iw)), ih * iw


def _nested_group(kind: str, used_areas: set[int]) -> tuple[np.ndarray, set[int]]:
    for _ in range(80):
        if kind == "inner_solid":
            inner_patch, area = _sample_unique_loop(used_areas, (2, 4))
            local_used = {area}
        else:
            inner_patch = _open_patch()
            local_used = set()

        ih, iw = inner_patch.shape
        outer_h = random.randint(max(ih + 4, 8), max(ih + 6, 11))
        outer_w = random.randint(max(iw + 4, 8), max(iw + 6, 11))
        patch = _rect_loop(outer_h, outer_w)

        max_r = patch.shape[0] - ih - 2
        max_c = patch.shape[1] - iw - 2
        if max_r < 2 or max_c < 2:
            continue
        r0 = random.randint(2, max_r)
        c0 = random.randint(2, max_c)
        view = patch[r0 : r0 + ih, c0 : c0 + iw]
        if np.any((view == BLUE) & (inner_patch == BLUE)):
            continue
        view[inner_patch == BLUE] = BLUE
        return _random_transform(patch), local_used
    raise RuntimeError("failed to build nested group")


def _place_units(units: list[np.ndarray], height: int, width: int) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    order = sorted(units, key=lambda patch: int(patch.shape[0] * patch.shape[1]), reverse=True)

    for patch in order:
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(height - ph + 1):
            for c0 in range(width - pw + 1):
                rr0 = max(0, r0 - PLACEMENT_GAP)
                cc0 = max(0, c0 - PLACEMENT_GAP)
                rr1 = min(height, r0 + ph + PLACEMENT_GAP)
                cc1 = min(width, c0 + pw + PLACEMENT_GAP)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        mask = patch == BLUE
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = BLUE
        occupied[max(0, r0 - PLACEMENT_GAP) : min(height, r0 + ph + PLACEMENT_GAP),
                 max(0, c0 - PLACEMENT_GAP) : min(width, c0 + pw + PLACEMENT_GAP)] = True

    return grid


def _summary(grid: np.ndarray) -> dict:
    infos = _component_info(grid)
    solid_areas = [int(region.shape[0]) for info in infos for region in info["solid_regions"]]
    return {
        "components": len(infos),
        "open_count": sum(1 for info in infos if info["is_open"]),
        "nested_closed_count": sum(1 for info in infos if info["is_nested_closed"]),
        "solid_count": len(solid_areas),
        "solid_areas": solid_areas,
    }


def _generate_candidate() -> np.ndarray | None:
    branch = random.choice(
        ("inner_solid_two_extra", "inner_open_three_extra", "inner_solid_three_extra")
    )
    used_areas: set[int] = set()
    units: list[np.ndarray] = []

    if branch == "inner_solid_two_extra":
        nested, nested_areas = _nested_group("inner_solid", used_areas)
        extra_solids = 2
    elif branch == "inner_open_three_extra":
        nested, nested_areas = _nested_group("inner_open", used_areas)
        extra_solids = 3
    else:
        nested, nested_areas = _nested_group("inner_solid", used_areas)
        extra_solids = 3

    used_areas |= nested_areas
    units.append(nested)
    units.append(_random_transform(_open_patch()))

    for _ in range(extra_solids):
        loop, area = _sample_unique_loop(used_areas, (2, 6))
        used_areas.add(area)
        units.append(loop)

    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    for _ in range(PLACEMENT_ATTEMPTS):
        grid = _place_units([_random_transform(unit) for unit in units], height, width)
        if grid is None:
            continue
        summary = _summary(grid)
        if not (4 <= summary["components"] <= 6):
            continue
        if summary["open_count"] < 1:
            continue
        if summary["nested_closed_count"] < 1:
            continue
        if summary["solid_count"] < 3:
            continue
        if len(set(summary["solid_areas"])) != len(summary["solid_areas"]):
            continue
        out = main(grid)
        if np.array_equal(out, grid):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
