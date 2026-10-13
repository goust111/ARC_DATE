from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

Coord = tuple[int, int]

PALETTE = tuple(range(10))
GRID_SIZES = tuple(range(11, 30, 2))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    """Most common color, matching arc_utils.get_bg()."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(grid: np.ndarray, bg: int | None = None) -> list[dict]:
    """
    Minimal monochromatic 4-connected component finder using arc_utils conventions.
    Returns dicts with color, coords, bbox, and area.
    """
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = _get_bg(g)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    components: list[dict] = []

    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return components


def _min_chebyshev_distance(coords: np.ndarray, center: Coord) -> int:
    cr, cc = center
    return min(max(abs(int(r) - cr), abs(int(c) - cc)) for r, c in coords)


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape
    assert H == W and H % 2 == 1, "expected an odd-sized square grid"
    assert H <= 30 and W <= 30, "grid must fit ARC size limits"

    bg = _get_bg(g)
    components = _find_components(g, bg=bg)
    assert len(components) >= 2, "expected several foreground components"

    colors = [comp["color"] for comp in components]
    sizes = [comp["area"] for comp in components]
    assert len(set(colors)) == len(colors), "each component must have a distinct color"
    assert len(set(sizes)) == len(sizes), "component sizes must be distinct"
    assert np.count_nonzero(g == bg) > g.size // 2, "background must dominate"

    max_radius = H // 2
    assert all(1 <= size <= max_radius for size in sizes), "every size must map to a valid ring radius"

    center = (H // 2, W // 2)
    distances = [_min_chebyshev_distance(comp["coords"], center) for comp in components]
    return {
        "grid": g,
        "bg": bg,
        "components": components,
        "sizes": sizes,
        "center": center,
        "max_radius": max_radius,
        "min_distances": distances,
    }


def _paint_square_ring(grid: np.ndarray, center: Coord, radius: int, color: int) -> None:
    cr, cc = center
    r0, r1 = cr - radius, cr + radius
    c0, c1 = cc - radius, cc + radius
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Turn each colored scatter-cluster's pixel count into a same-colored concentric square ring around the grid center.

    Input structure:
    - The input is an odd-sized square grid with a dominant background color.
    - Several 4-connected monochrome components appear in distinct colors, and all component sizes are different.
    - Component locations are intentionally unrelated to the ring they will produce; some sit at center distances matching another component's size.

    Rule:
    - Erase the whole input to background first.
    - For each component with area k, paint every cell whose Chebyshev distance from the grid center is exactly k using that component's color.
    - Leave radii with no matching component as background, and keep the center cell as background.

    Key insight:
    - The components are not moved by position; their pixel counts are reinterpreted as radii, converting scattered shapes into ordered concentric square rings.
    """
    info = _analyze_grid(input_grid)
    out = np.full_like(info["grid"], info["bg"])
    for comp in sorted(info["components"], key=lambda comp: comp["area"]):
        _paint_square_ring(out, info["center"], int(comp["area"]), int(comp["color"]))
    return out


def _normalize_cells(cells: set[Coord] | list[Coord] | tuple[Coord, ...]) -> tuple[Coord, ...]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _cells_to_mask(cells: tuple[Coord, ...]) -> np.ndarray:
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _all_transforms(cells: tuple[Coord, ...]) -> list[tuple[Coord, ...]]:
    mask = _cells_to_mask(cells)
    variants: dict[tuple[Coord, ...], tuple[Coord, ...]] = {}
    for rot in range(4):
        rotated = np.rot90(mask, rot)
        for arr in (rotated, np.fliplr(rotated)):
            coords = tuple(map(tuple, np.argwhere(arr)))
            norm = _normalize_cells(coords)
            variants[norm] = norm
    return list(variants.values())


def _random_growth_shape(area: int) -> tuple[Coord, ...]:
    cells: set[Coord] = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))
    return _normalize_cells(cells)


def _row_major_shape(area: int) -> tuple[Coord, ...]:
    height = random.randint(1, max(1, int(np.sqrt(area))))
    width = (area + height - 1) // height
    cells = []
    remaining = area
    for r in range(height):
        take = min(width, remaining)
        for c in range(take):
            cells.append((r, c))
        remaining -= take
        if remaining == 0:
            break
    return _normalize_cells(cells)


def _line_shape(area: int) -> tuple[Coord, ...]:
    return tuple((0, c) for c in range(area))


def _make_shape(area: int) -> tuple[Coord, ...]:
    if area == 1:
        return ((0, 0),)
    style = random.choices(
        population=("growth", "row-major", "line"),
        weights=(6, 3, 2),
        k=1,
    )[0]
    if style == "line":
        return _line_shape(area)
    if style == "row-major":
        return _row_major_shape(area)
    return _random_growth_shape(area)


def _expanded_8_neighbors(cells: list[Coord], n: int) -> set[Coord]:
    blocked: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    blocked.add((rr, cc))
    return blocked


def _candidate_positions(
    shape: tuple[Coord, ...],
    n: int,
    center: Coord,
    desired_min_distance: int,
    blocked: set[Coord],
) -> list[list[Coord]]:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    placements: list[list[Coord]] = []
    for r0 in range(n - h + 1):
        for c0 in range(n - w + 1):
            placed = [(r0 + r, c0 + c) for r, c in shape]
            if center in placed:
                continue
            if any(cell in blocked for cell in placed):
                continue
            dist = min(max(abs(r - center[0]), abs(c - center[1])) for r, c in placed)
            if dist != desired_min_distance:
                continue
            placements.append(placed)
    random.shuffle(placements)
    return placements


def _choose_sizes(radius: int, count: int) -> list[int]:
    sizes: set[int] = set()
    if random.random() < 0.6:
        sizes.add(1)
    if radius >= 3 and random.random() < 0.45:
        sizes.add(radius)
    while len(sizes) < count:
        sizes.add(random.randint(1, radius))
    while len(sizes) > count:
        sizes.remove(random.choice(tuple(sizes)))
    return sorted(sizes)


def _choose_min_distances(sizes: list[int], radius: int) -> list[int]:
    distances = []
    upper = max(1, radius - 1)
    for size in sizes:
        options = [d for d in range(1, upper + 1) if d != size]
        distances.append(random.choice(options or [upper]))

    if len(sizes) >= 2:
        candidates = [i for i in range(len(sizes)) if any(other < radius for j, other in enumerate(sizes) if j != i)]
        forced_idx = random.choice(candidates or list(range(len(sizes))))
        other_sizes = [other for j, other in enumerate(sizes) if j != forced_idx and other < radius]
        if not other_sizes:
            other_sizes = [other for j, other in enumerate(sizes) if j != forced_idx]
        distances[forced_idx] = random.choice(other_sizes)
    return distances


def _build_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    radius = n // 2
    max_count = 3 if radius <= 6 else 4 if radius <= 10 else 5
    count = random.randint(2, min(max_count, radius))
    sizes = _choose_sizes(radius, count)

    bg = random.choice(PALETTE)
    colors = random.sample([c for c in PALETTE if c != bg], count)
    min_distances = _choose_min_distances(sizes, radius)

    specs = [
        {
            "size": size,
            "color": color,
            "min_distance": min_distance,
            "shape": _make_shape(size),
        }
        for size, color, min_distance in zip(sizes, colors, min_distances)
    ]

    grid = np.full((n, n), bg, dtype=int)
    center = (n // 2, n // 2)
    blocked: set[Coord] = set()
    placed_specs = []

    for spec in sorted(specs, key=lambda item: item["size"], reverse=True):
        shapes = _all_transforms(spec["shape"])
        random.shuffle(shapes)
        placement = None
        for shape in shapes:
            candidates = _candidate_positions(shape, n, center, spec["min_distance"], blocked)
            if candidates:
                placement = random.choice(candidates)
                spec["shape"] = shape
                break
        if placement is None:
            return None
        for r, c in placement:
            grid[r, c] = spec["color"]
        blocked |= _expanded_8_neighbors(placement, n)
        placed_specs.append(spec | {"placement": placement})

    info = _analyze_grid(grid)
    if set(info["sizes"]) != set(sizes):
        return None

    size_set = set(info["sizes"])
    if not any(
        dist in (size_set - {comp["area"]})
        for comp, dist in zip(info["components"], info["min_distances"])
    ):
        return None

    if all(dist == comp["area"] for comp, dist in zip(info["components"], info["min_distances"])):
        return None

    if len(info["components"]) != count:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid quantity-to-ring input")
