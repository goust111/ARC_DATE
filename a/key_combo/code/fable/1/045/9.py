from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 1200
MIN_SIDE = 10
MAX_SIDE = 22
MIN_PATH_LEN = 8
MIN_REGION_AREA = 5
MAX_SEEDS = 3


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.asarray(comp, dtype=int))
    return comps


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    out: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg)}):
        for coords in _components_from_mask(g == int(color)):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _is_arrow_component(coords: np.ndarray) -> bool:
    if int(coords.shape[0]) != 3:
        return False
    r0, c0, r1, c1 = _bbox_coords(coords)
    if (r1 - r0, c1 - c0) != (1, 1):
        return False
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    return {tuple(map(int, cell)) for cell in coords} <= corners


def _arrow_start_and_direction(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    pts = {tuple(map(int, cell)) for cell in coords}
    r0, c0, r1, c1 = _bbox_coords(coords)
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    missing = [p for p in corners if p not in pts]
    assert len(missing) == 1, "arrow must be a 2x2 L shape with one missing corner"
    mr, mc = missing[0]
    dr = 1 if mr == r0 else -1
    dc = 1 if mc == c0 else -1

    degrees = {p: 0 for p in pts}
    for r, c in pts:
        for dr4, dc4 in NEIGHBORS4:
            if (r + dr4, c + dc4) in pts:
                degrees[(r, c)] += 1
    start = max(degrees, key=degrees.get)
    return (int(start[0]), int(start[1])), (int(dr), int(dc))


def _locate_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_monochrome_components(g, bg)
    arrows = [comp for comp in comps if _is_arrow_component(comp["coords"])]
    assert len(arrows) == 1, "expected exactly one L-shaped arrow"
    arrow = arrows[0]
    seeds = [comp for comp in comps if comp is not arrow]
    assert all(int(comp["area"]) == 1 for comp in seeds), "all non-arrow foreground objects must be isolated seed pixels"
    start, direction = _arrow_start_and_direction(arrow["coords"])
    return {
        "bg": int(bg),
        "arrow_color": int(arrow["color"]),
        "arrow_cells": [tuple(map(int, cell)) for cell in arrow["coords"]],
        "start": start,
        "direction": direction,
        "seeds": seeds,
    }


def _simulate_ray(
    shape: tuple[int, int],
    arrow_cells: list[tuple[int, int]],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> dict:
    h, w = map(int, shape)
    dr, dc = map(int, direction)
    cur = (int(start[0]), int(start[1]))
    barrier = set(arrow_cells)
    path: list[tuple[int, int]] = []
    side_bounces = 0
    corner_bounces = 0

    while True:
        nr = cur[0] + dr
        nc = cur[1] + dc
        hit_row = not (0 <= nr < h)
        hit_col = not (0 <= nc < w)

        if hit_row or hit_col:
            if hit_row:
                dr = -dr
            if hit_col:
                dc = -dc
            if hit_row and hit_col:
                corner_bounces += 1
            else:
                side_bounces += 1
            nr = cur[0] + dr
            nc = cur[1] + dc

        if (nr, nc) in barrier:
            break

        barrier.add((int(nr), int(nc)))
        path.append((int(nr), int(nc)))
        cur = (int(nr), int(nc))

    return {
        "path": path,
        "barrier": barrier,
        "side_bounces": int(side_bounces),
        "corner_bounces": int(corner_bounces),
    }


def _walkable_components(shape: tuple[int, int], barrier: set[tuple[int, int]]) -> list[np.ndarray]:
    h, w = map(int, shape)
    walkable = np.ones((h, w), dtype=bool)
    for r, c in barrier:
        walkable[int(r), int(c)] = False
    return _components_from_mask(walkable)


def _paint_coords(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[int(r), int(c)] = int(color)


def _component_seed_colors(grid: np.ndarray, bg: int, component: np.ndarray) -> set[int]:
    return {int(grid[r, c]) for r, c in component if int(grid[r, c]) != int(bg)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the arrow's bouncing diagonal fence and flood each enclosed territory by its seed color.

    Input structure:
    - A black-dominant background contains exactly one monochrome 3-pixel L-shaped
      arrow in a 2x2 bbox with one missing corner.
    - The remaining foreground objects are isolated single-pixel colored seeds.
    - There are no interior obstacles; only the grid boundary can reflect the ray.

    Rule:
    - The arrow launches a diagonal ray away from its missing corner, starting from
      the arrow's right-angle pixel.
    - Each visited cell becomes the arrow color; when the next step would leave the
      grid, the corresponding axis reflects, and a corner hit reflects both axes.
    - The ray stops as soon as its next step would enter the arrow itself or any
      already painted path cell.
    - Treat the arrow and full path as barriers, split all remaining cells into
      4-connected regions, and recolor every region containing seed pixels with that
      region's unique seed color. Regions with no seed stay black.

    Key insight:
    - The diagonal path matters because it partitions 4-connected space; the seeds do
      not spread along the ray, they merely declare which final territory gets filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    info = _locate_scene(g)
    sim = _simulate_ray(g.shape, info["arrow_cells"], info["start"], info["direction"])
    out = np.full_like(g, info["bg"])
    _paint_coords(out, info["arrow_cells"], info["arrow_color"])
    _paint_coords(out, sim["path"], info["arrow_color"])

    for component in _walkable_components(g.shape, sim["barrier"]):
        colors = _component_seed_colors(g, info["bg"], component)
        if not colors:
            continue
        assert len(colors) == 1, "each fenced territory must contain at most one seed color"
        fill_color = next(iter(colors))
        out[component[:, 0], component[:, 1]] = int(fill_color)

    return out


def _make_arrow_cells(r0: int, c0: int, missing_index: int) -> list[tuple[int, int]]:
    corners = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
    missing = corners[int(missing_index)]
    return [cell for cell in corners if cell != missing]


def _seed_candidates(
    component: np.ndarray,
    arrow_set: set[tuple[int, int]],
    chosen: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    pts = {tuple(map(int, cell)) for cell in component}
    strong: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []
    for cell in pts:
        if any((cell[0] + dr, cell[1] + dc) in arrow_set or (cell[0] + dr, cell[1] + dc) in chosen for dr, dc in NEIGHBORS4):
            continue
        degree = sum((cell[0] + dr, cell[1] + dc) in pts for dr, dc in NEIGHBORS4)
        fallback.append(cell)
        if degree >= 3:
            strong.append(cell)
    return strong or fallback


def _verify_semantics(grid: np.ndarray, want_corner: bool) -> bool:
    info = _locate_scene(grid)
    sim = _simulate_ray(grid.shape, info["arrow_cells"], info["start"], info["direction"])
    if len(sim["path"]) < MIN_PATH_LEN:
        return False
    if want_corner:
        if sim["corner_bounces"] == 0:
            return False
    else:
        if sim["corner_bounces"] != 0 or sim["side_bounces"] == 0:
            return False

    components = _walkable_components(grid.shape, sim["barrier"])
    comp_of: dict[tuple[int, int], int] = {}
    for idx, component in enumerate(components):
        for r, c in component:
            comp_of[(int(r), int(c))] = int(idx)

    region_colors: dict[int, set[int]] = {}
    for seed in info["seeds"]:
        cell = tuple(map(int, seed["coords"][0]))
        if cell in sim["barrier"]:
            return False
        idx = comp_of[cell]
        region_colors.setdefault(idx, set()).add(int(seed["color"]))
    if not region_colors or any(len(colors) != 1 for colors in region_colors.values()):
        return False
    if len(region_colors) >= len(components):
        return False

    out = main(grid)
    for r, c in info["arrow_cells"]:
        if int(out[r, c]) != info["arrow_color"]:
            return False
    for r, c in sim["path"]:
        if int(out[r, c]) != info["arrow_color"]:
            return False

    seeded_regions = 0
    unseeded_regions = 0
    for idx, component in enumerate(components):
        values = {int(out[r, c]) for r, c in component}
        if idx in region_colors:
            seeded_regions += 1
            if values != region_colors[idx]:
                return False
        else:
            unseeded_regions += 1
            if values != {info["bg"]}:
                return False
    return seeded_regions >= 1 and unseeded_regions >= 1


def _sample_seed_count(rng: np.random.Generator, seedable_count: int) -> int:
    if seedable_count == 2:
        return 1 if float(rng.random()) < 0.75 else 2
    options = [1, 2, min(MAX_SEEDS, seedable_count)]
    probs = np.array([0.2, 0.55, 0.25], dtype=float)
    return int(rng.choice(options, p=probs / probs.sum()))


def _gen_attempt(rng: np.random.Generator, want_corner: bool) -> np.ndarray | None:
    h = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    w = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    arrow_color = int(rng.choice(PALETTE))
    r0 = int(rng.integers(0, h - 1))
    c0 = int(rng.integers(0, w - 1))
    arrow_cells = _make_arrow_cells(r0, c0, int(rng.integers(0, 4)))
    arrow_coords = np.asarray(arrow_cells, dtype=int)
    start, direction = _arrow_start_and_direction(arrow_coords)
    sim = _simulate_ray((h, w), arrow_cells, start, direction)

    if len(sim["path"]) < MIN_PATH_LEN or len(sim["path"]) > (h * w) // 2:
        return None
    if want_corner:
        if sim["corner_bounces"] == 0:
            return None
    else:
        if sim["corner_bounces"] != 0 or sim["side_bounces"] == 0:
            return None

    components = _walkable_components((h, w), sim["barrier"])
    usable = [component for component in components if int(component.shape[0]) >= MIN_REGION_AREA]
    if len(usable) < 2:
        return None

    arrow_set = set(arrow_cells)
    seedable: list[tuple[np.ndarray, list[tuple[int, int]]]] = []
    for component in usable:
        candidates = _seed_candidates(component, arrow_set, set())
        if candidates:
            seedable.append((component, candidates))
    if len(seedable) < 2:
        return None

    seed_count = _sample_seed_count(rng, len(seedable))
    perm = rng.permutation(len(seedable))
    selected = [seedable[int(i)] for i in perm[:seed_count]]
    if len(selected) >= len(components):
        return None

    seed_palette = [color for color in PALETTE if color != arrow_color]
    if len(seed_palette) < seed_count:
        return None
    seed_colors = list(map(int, rng.choice(seed_palette, size=seed_count, replace=False)))

    grid = np.full((h, w), BG, dtype=int)
    _paint_coords(grid, arrow_cells, arrow_color)

    chosen: set[tuple[int, int]] = set()
    for (component, _base_candidates), seed_color in zip(selected, seed_colors):
        candidates = _seed_candidates(component, arrow_set, chosen)
        if not candidates:
            return None
        cell = candidates[int(rng.integers(len(candidates)))]
        grid[cell] = int(seed_color)
        chosen.add(cell)

    return grid if _verify_semantics(grid, want_corner) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    want_corner = bool(float(rng.random()) < 0.45)
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng, want_corner)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_ATTEMPTS} attempts")
