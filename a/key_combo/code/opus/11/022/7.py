from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
PALETTE = (1, 2, 3, 4)
MAX_GENERATION_ATTEMPTS = 400
MIN_GRID_SIZE = 25
MAX_GRID_SIZE = 29
MIN_BBOX_GAP = 8


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_center_round(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return _round_half_up((r0 + r1) / 2.0), _round_half_up((c0 + c1) / 2.0)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _bbox_gap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> int:
    r0a, c0a, r1a, c1a = b1
    r0b, c0b, r1b, c1b = b2
    dr = max(0, r0b - r1a - 1, r0a - r1b - 1)
    dc = max(0, c0b - c1a - 1, c0a - c1b - 1)
    if dr == 0:
        return int(dc)
    if dc == 0:
        return int(dr)
    return int(dr + dc)


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _centroid_round(coords: np.ndarray) -> tuple[int, int]:
    mean_r, mean_c = coords.mean(axis=0)
    return _round_half_up(mean_r), _round_half_up(mean_c)


def _diamond_outline(shape: tuple[int, int], center: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    h, w = shape
    cr, cc = center
    pts: set[tuple[int, int]] = set()
    for dr in range(-radius, radius + 1):
        dc = radius - abs(dr)
        for sign in (-1, 1):
            rr = cr + dr
            cc2 = cc + sign * dc
            if 0 <= rr < h and 0 <= cc2 < w:
                pts.add((int(rr), int(cc2)))
    return pts


def _render_components(
    shape: tuple[int, int],
    components: list[dict],
    *,
    center_mode: str = "centroid",
    radius_mode: str = "area",
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    ordered = sorted(
        components,
        key=lambda comp: (-int(comp["area"]), int(comp["color"]), int(comp["bbox"][0]), int(comp["bbox"][1])),
    )
    for comp in ordered:
        if center_mode == "centroid":
            center = _centroid_round(comp["coords"])
        elif center_mode == "bbox":
            center = _bbox_center_round(comp["bbox"])
        else:
            raise ValueError(f"unknown center_mode: {center_mode}")

        if radius_mode == "area":
            radius = int(comp["area"])
        elif radius_mode == "bbox_area":
            radius = _bbox_area(comp["bbox"])
        else:
            raise ValueError(f"unknown radius_mode: {radius_mode}")

        for r, c in _diamond_outline(shape, center, radius):
            out[r, c] = int(comp["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each compact foreground component with a same-color diamond outline whose Manhattan radius equals the component area.

    Input structure:
    - The grid has a black background.
    - The foreground contains 3-4 separated monochromatic 4-connected objects in distinct colors.
    - Each object is a small compact irregular shape, and objects are spaced far enough apart that their derived diamonds do not overlap.

    Rule:
    - Detect every non-black 4-connected component and record its color and pixel coordinates.
    - For each component, compute its area and its centroid by averaging row and column coordinates, then round each coordinate to the nearest integer with half-up rounding.
    - Clear the canvas to black and draw only the component's color on the diamond outline defined by |r-cr| + |c-cc| = area.
    - Draw diamonds from larger area to smaller area; the original component pixels are not preserved.

    Key insight:
    - The input object disappears, but its statistical signature survives: area becomes the diamond radius and the rounded centroid becomes the new geometric center.
    """
    g = np.asarray(grid, dtype=int)
    return _render_components(g.shape, _find_components(g), center_mode="centroid", radius_mode="area")


def _normalize_shape(cells: set[tuple[int, int]]) -> np.ndarray:
    arr = np.array(sorted(cells), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _unique_orientations(base: list[tuple[int, int]]) -> list[np.ndarray]:
    arr = np.array(base, dtype=int)
    seen = set()
    out = []
    for swap in (False, True):
        cur0 = arr[:, ::-1] if swap else arr.copy()
        for flip_r in (False, True):
            for flip_c in (False, True):
                cur = cur0.copy()
                if flip_r:
                    cur[:, 0] = int(cur[:, 0].max()) - cur[:, 0]
                if flip_c:
                    cur[:, 1] = int(cur[:, 1].max()) - cur[:, 1]
                norm = _normalize_shape(set(map(tuple, cur.tolist())))
                key = tuple(map(tuple, norm.tolist()))
                if key not in seen:
                    seen.add(key)
                    out.append(norm)
    return out


_BASE_SHAPES = {
    3: [
        [(0, 0), (1, 0), (1, 1)],
    ],
    4: [
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
    ],
    5: [
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
        [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
    ],
}

_SHAPES_BY_AREA = {
    area: [shape for base in bases for shape in _unique_orientations(base)]
    for area, bases in _BASE_SHAPES.items()
}


def _shape_stats(shape: np.ndarray) -> dict:
    bbox = _bbox_from_coords(shape)
    center = _centroid_round(shape)
    mean_r, mean_c = shape.mean(axis=0)
    return {
        "shape": shape,
        "bbox": bbox,
        "bbox_area": _bbox_area(bbox),
        "bbox_center": _bbox_center_round(bbox),
        "centroid": center,
        "mean_r": float(mean_r),
        "mean_c": float(mean_c),
        "half_row": abs(float(mean_r) - int(mean_r) - 0.5) < 1e-9,
        "half_col": abs(float(mean_c) - int(mean_c) - 0.5) < 1e-9,
    }


_SHAPE_STATS_BY_AREA = {
    area: [_shape_stats(shape) for shape in shapes]
    for area, shapes in _SHAPES_BY_AREA.items()
}


def _candidate_shape_variants(area: int, mode: str) -> list[dict]:
    stats = list(_SHAPE_STATS_BY_AREA[area])
    if mode == "half_row":
        return [s for s in stats if s["half_row"]]
    if mode == "half_col":
        return [s for s in stats if s["half_col"]]
    if mode == "bbox_mismatch":
        return [s for s in stats if s["centroid"] != s["bbox_center"]]
    return stats


def _zone_center(zone: str, area: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    inward_r = random.randint(0, 1)
    inward_c = random.randint(0, 1)
    if zone[0] == "t":
        cr = area + inward_r
    else:
        cr = h - 1 - area - inward_r
    if zone[1] == "l":
        cc = area + inward_c
    else:
        cc = w - 1 - area - inward_c
    return int(cr), int(cc)


def _placements_for_center(
    shape: np.ndarray, grid_shape: tuple[int, int], center: tuple[int, int]
) -> list[np.ndarray]:
    h, w = grid_shape
    sh = int(shape[:, 0].max()) + 1
    sw = int(shape[:, 1].max()) + 1
    mean_r, mean_c = shape.mean(axis=0)
    target_r, target_c = center
    placements = []
    r_min = max(0, target_r - sh - 1)
    r_max = min(h - sh, target_r + 1)
    c_min = max(0, target_c - sw - 1)
    c_max = min(w - sw, target_c + 1)
    for r0 in range(r_min, r_max + 1):
        if _round_half_up(r0 + mean_r) != target_r:
            continue
        for c0 in range(c_min, c_max + 1):
            if _round_half_up(c0 + mean_c) != target_c:
                continue
            coords = shape + np.array([r0, c0], dtype=int)
            placements.append(coords)
    random.shuffle(placements)
    return placements


def _components_overlap(coords1: np.ndarray, coords2: np.ndarray) -> bool:
    s1 = {tuple(map(int, rc)) for rc in coords1.tolist()}
    s2 = {tuple(map(int, rc)) for rc in coords2.tolist()}
    return bool(s1 & s2)


def _build_component(color: int, coords: np.ndarray) -> dict:
    return {
        "color": int(color),
        "coords": np.array(coords, dtype=int),
        "bbox": _bbox_from_coords(coords),
        "area": int(coords.shape[0]),
    }


def _make_input_grid(shape: tuple[int, int], components: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for comp in components:
        coords = comp["coords"]
        grid[coords[:, 0], coords[:, 1]] = int(comp["color"])
    return grid


def _validate_instance(grid: np.ndarray) -> None:
    comps = _find_components(grid)
    assert 3 <= len(comps) <= 4
    colors = [int(comp["color"]) for comp in comps]
    assert len(set(colors)) == len(colors)
    assert set(colors).issubset(set(PALETTE))

    areas = [int(comp["area"]) for comp in comps]
    assert all(3 <= area <= 5 for area in areas)
    assert len(set(areas)) >= 2

    bbox_mismatch_found = False
    diamond_sets = []
    for comp in comps:
        bbox = comp["bbox"]
        bbox_area = _bbox_area(bbox)
        assert bbox_area > int(comp["area"])
        center = _centroid_round(comp["coords"])
        bbox_center = _bbox_center_round(bbox)
        bbox_mismatch_found = bbox_mismatch_found or center != bbox_center
        diamond = _diamond_outline(grid.shape, center, int(comp["area"]))
        assert diamond
        assert len(diamond) == 4 * int(comp["area"])
        diamond_sets.append(diamond)

    assert bbox_mismatch_found

    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            assert _bbox_gap(comps[i]["bbox"], comps[j]["bbox"]) >= MIN_BBOX_GAP
            assert not (diamond_sets[i] & diamond_sets[j])

    out = main(grid)
    alt_center = _render_components(grid.shape, comps, center_mode="bbox", radius_mode="area")
    alt_radius = _render_components(grid.shape, comps, center_mode="centroid", radius_mode="bbox_area")
    assert not np.array_equal(out, alt_center)
    assert not np.array_equal(out, alt_radius)
    assert np.count_nonzero(out) == sum(4 * int(comp["area"]) for comp in comps)


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    w = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    count = random.choice((3, 4))
    zones = random.sample(["tl", "tr", "bl", "br"], count)
    colors = random.sample(PALETTE, count)

    areas = [random.randint(3, 5) for _ in range(count)]
    if len(set(areas)) == 1:
        areas[random.randrange(count)] = random.choice([a for a in (3, 4, 5) if a != areas[0]])

    special_modes = ["half_row", "half_col", "bbox_mismatch"]
    special_index = random.randrange(count)
    special_mode = random.choice(special_modes)
    if special_mode in {"half_row", "half_col"}:
        areas[special_index] = 4

    components: list[dict] = []
    for idx, (zone, color, area) in enumerate(zip(zones, colors, areas)):
        mode = special_mode if idx == special_index else "any"
        variants = _candidate_shape_variants(area, mode)
        if not variants:
            return None
        random.shuffle(variants)

        placed = None
        for variant in variants:
            center = _zone_center(zone, area, (h, w))
            placements = _placements_for_center(variant["shape"], (h, w), center)
            for coords in placements:
                comp = _build_component(color, coords)
                if any(_components_overlap(comp["coords"], prev["coords"]) for prev in components):
                    continue
                if any(_bbox_gap(comp["bbox"], prev["bbox"]) < MIN_BBOX_GAP for prev in components):
                    continue
                diamond = _diamond_outline((h, w), _centroid_round(comp["coords"]), area)
                if len(diamond) != 4 * area:
                    continue
                if any(diamond & _diamond_outline((h, w), _centroid_round(prev["coords"]), int(prev["area"])) for prev in components):
                    continue
                placed = comp
                break
            if placed is not None:
                break
        if placed is None:
            return None
        components.append(placed)

    if len(components) != count:
        return None

    grid = _make_input_grid((h, w), components)
    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance for puzzle 7")
