from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GRID_MIN = 22
GRID_MAX = 30
TEMPLATE_SIDE_MIN = 12
TEMPLATE_SIDE_MAX = 17
TEMPLATE_MARGIN_MIN = 4
GEN_RETRIES = 400
HOLE_COUNT_CHOICES = (2, 3, 4)
SHAPE_POOL = (
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (1, 4),
    (4, 1),
)
QUADRANTS = ("tl", "tr", "bl", "br")


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = np.asarray(coords, dtype=int)
    mins = pts.min(axis=0)
    rel = pts - mins
    order = np.lexsort((rel[:, 1], rel[:, 0]))
    return tuple(tuple(int(v) for v in rel[i]) for i in order)


def _enclosed_holes(component_mask: np.ndarray) -> list[np.ndarray]:
    coords = np.argwhere(component_mask)
    assert len(coords) > 0
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    patch = component_mask[r0 : r1 + 1, c0 : c1 + 1]
    bg_patch = ~patch
    h, w = map(int, patch.shape)

    outside = np.zeros_like(bg_patch, dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if bg_patch[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg_patch[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        cr, cc = q.popleft()
        for nr, nc in _neighbors4(cr, cc, h, w):
            if bg_patch[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    holes_mask = bg_patch & ~outside
    holes = _components(holes_mask)
    offset = np.array([r0, c0], dtype=int)
    return [hole + offset for hole in holes]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(np.bincount(g.ravel()).argmax())
    comps = _components(g != bg)
    assert len(comps) >= 2, "expected one template and at least one detached fragment"

    areas = [len(comp) for comp in comps]
    template_index = int(np.argmax(areas))
    assert areas[template_index] > max(a for i, a in enumerate(areas) if i != template_index)
    template = comps[template_index]

    template_mask = np.zeros_like(g, dtype=bool)
    template_mask[template[:, 0], template[:, 1]] = True
    holes = _enclosed_holes(template_mask)
    assert holes, "template must contain at least one enclosed hole"

    holes_by_shape: dict[tuple[tuple[int, int], ...], tuple[np.ndarray, int]] = {}
    for hole in holes:
        shape = _normalize(hole)
        assert shape not in holes_by_shape, "hole shapes must be unique"
        border_colors = set()
        for r, c in hole:
            for nr, nc in _neighbors4(int(r), int(c), *g.shape):
                if template_mask[nr, nc]:
                    border_colors.add(int(g[nr, nc]))
        assert len(border_colors) == 1, "each hole must belong to one template region"
        holes_by_shape[shape] = (hole, border_colors.pop())

    out = g.copy()
    matched_shapes = set()
    for i, comp in enumerate(comps):
        if i == template_index:
            continue
        shape = _normalize(comp)
        assert shape in holes_by_shape, "every detached fragment must match a hole shape"
        assert shape not in matched_shapes, "fragment shapes must also be unique"
        matched_shapes.add(shape)
        out[comp[:, 0], comp[:, 1]] = bg

    assert matched_shapes == set(holes_by_shape), "holes and fragments must match one-to-one"
    for hole, color in holes_by_shape.values():
        out[hole[:, 0], hole[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered shape-key fragments to the central template's missing holes and restore each hole with its region color.

    Input structure:
    - The background is black.
    - A single large multicolor template sits near the center and forms one connected non-background object.
    - Several template regions contain enclosed background holes, while detached monochrome fragments with matching shapes sit in the outer corners.

    Rule:
    - Identify the largest connected non-background component as the template object.
    - Find every enclosed background hole inside that template and determine the unique template color touching each hole.
    - Match each detached fragment to a hole by normalized shape, erase the fragment from its corner, and fill the matched hole with the touching template color.

    Key insight:
    - Fragment colors are distractors: only the fragment shapes matter, while the surrounding template region determines the output color.
    """
    return _solve(grid)


def _random_composition(total: int, parts: int, min_size: int) -> list[int]:
    sizes = [min_size] * parts
    extra = total - parts * min_size
    for _ in range(extra):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    return sizes


def _make_regions(height: int, width: int) -> list[dict[str, int]]:
    branch = random.choice(("v3", "h3", "left_l", "top_l", "quad4"))
    regions: list[dict[str, int]] = []

    if branch == "v3":
        x = 0
        for stripe_w in _random_composition(width, 3, 4):
            regions.append({"r0": 0, "r1": height, "c0": x, "c1": x + stripe_w})
            x += stripe_w
    elif branch == "h3":
        y = 0
        for stripe_h in _random_composition(height, 3, 4):
            regions.append({"r0": y, "r1": y + stripe_h, "c0": 0, "c1": width})
            y += stripe_h
    elif branch == "left_l":
        left_w = random.randint(4, width - 8)
        split_r = random.randint(4, height - 4)
        regions.extend(
            [
                {"r0": 0, "r1": height, "c0": 0, "c1": left_w},
                {"r0": 0, "r1": split_r, "c0": left_w, "c1": width},
                {"r0": split_r, "r1": height, "c0": left_w, "c1": width},
            ]
        )
    elif branch == "top_l":
        top_h = random.randint(4, height - 8)
        split_c = random.randint(4, width - 4)
        regions.extend(
            [
                {"r0": 0, "r1": top_h, "c0": 0, "c1": width},
                {"r0": top_h, "r1": height, "c0": 0, "c1": split_c},
                {"r0": top_h, "r1": height, "c0": split_c, "c1": width},
            ]
        )
    else:
        split_r = random.randint(4, height - 4)
        split_c = random.randint(4, width - 4)
        regions.extend(
            [
                {"r0": 0, "r1": split_r, "c0": 0, "c1": split_c},
                {"r0": 0, "r1": split_r, "c0": split_c, "c1": width},
                {"r0": split_r, "r1": height, "c0": 0, "c1": split_c},
                {"r0": split_r, "r1": height, "c0": split_c, "c1": width},
            ]
        )

    colors = random.sample(range(1, 10), len(regions))
    for region, color in zip(regions, colors):
        region["color"] = color
    return regions


def _place_holes(regions: list[dict[str, int]], height: int, width: int):
    occupied = np.zeros((height, width), dtype=bool)
    hole_count = random.choice(HOLE_COUNT_CHOICES)
    remaining_shapes = list(SHAPE_POOL)
    random.shuffle(remaining_shapes)
    holes: list[dict[str, int]] = []

    for _ in range(hole_count):
        candidates = []
        for shape in remaining_shapes:
            shape_h, shape_w = shape
            for idx, region in enumerate(regions):
                if region["r1"] - region["r0"] < shape_h + 2:
                    continue
                if region["c1"] - region["c0"] < shape_w + 2:
                    continue
                for r in range(region["r0"] + 1, region["r1"] - shape_h):
                    for c in range(region["c0"] + 1, region["c1"] - shape_w):
                        if occupied[r - 1 : r + shape_h + 1, c - 1 : c + shape_w + 1].any():
                            continue
                        candidates.append((shape, idx, r, c))
        if not candidates:
            return None

        shape, region_idx, r, c = random.choice(candidates)
        shape_h, shape_w = shape
        remaining_shapes.remove(shape)
        occupied[r - 1 : r + shape_h + 1, c - 1 : c + shape_w + 1] = True
        holes.append(
            {
                "r": r,
                "c": c,
                "h": shape_h,
                "w": shape_w,
                "region_idx": region_idx,
            }
        )

    return holes


def _corner_zones(grid_h: int, grid_w: int, top: int, left: int, template_h: int, template_w: int):
    bottom = top + template_h
    right = left + template_w
    return {
        "tl": (0, top, 0, left),
        "tr": (0, top, right, grid_w),
        "bl": (bottom, grid_h, 0, left),
        "br": (bottom, grid_h, right, grid_w),
    }


def _place_fragments(
    grid: np.ndarray,
    holes: list[dict[str, int]],
    template_colors: set[int],
    top: int,
    left: int,
    template_h: int,
    template_w: int,
):
    zones = _corner_zones(*grid.shape, top, left, template_h, template_w)
    quadrants = random.sample(QUADRANTS, len(holes))
    fragment_colors = random.sample([c for c in range(1, 10) if c not in template_colors], len(holes))
    placements = []

    for hole, quadrant, color in zip(holes, quadrants, fragment_colors):
        zone_r0, zone_r1, zone_c0, zone_c1 = zones[quadrant]
        shape_h, shape_w = hole["h"], hole["w"]
        candidates = []
        for r in range(zone_r0, zone_r1 - shape_h + 1):
            for c in range(zone_c0, zone_c1 - shape_w + 1):
                rr0 = max(0, r - 1)
                rr1 = min(grid.shape[0], r + shape_h + 1)
                cc0 = max(0, c - 1)
                cc1 = min(grid.shape[1], c + shape_w + 1)
                if np.any(grid[rr0:rr1, cc0:cc1] != BG):
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        grid[r : r + shape_h, c : c + shape_w] = color
        placements.append({"r": r, "c": c, "h": shape_h, "w": shape_w, "color": color})
    return placements


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(GRID_MIN, GRID_MAX)
    grid_w = random.randint(GRID_MIN, GRID_MAX)
    max_template_h = min(TEMPLATE_SIDE_MAX, grid_h - 2 * TEMPLATE_MARGIN_MIN)
    max_template_w = min(TEMPLATE_SIDE_MAX, grid_w - 2 * TEMPLATE_MARGIN_MIN)
    if max_template_h < TEMPLATE_SIDE_MIN or max_template_w < TEMPLATE_SIDE_MIN:
        return None

    template_h = random.randint(TEMPLATE_SIDE_MIN, max_template_h)
    template_w = random.randint(TEMPLATE_SIDE_MIN, max_template_w)
    top = random.randint(TEMPLATE_MARGIN_MIN, grid_h - template_h - TEMPLATE_MARGIN_MIN)
    left = random.randint(TEMPLATE_MARGIN_MIN, grid_w - template_w - TEMPLATE_MARGIN_MIN)

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    regions = _make_regions(template_h, template_w)
    template_colors = set()
    for region in regions:
        color = region["color"]
        template_colors.add(color)
        grid[top + region["r0"] : top + region["r1"], left + region["c0"] : left + region["c1"]] = color

    holes = _place_holes(regions, template_h, template_w)
    if holes is None:
        return None

    for hole in holes:
        r, c, h, w = hole["r"], hole["c"], hole["h"], hole["w"]
        grid[top + r : top + r + h, left + c : left + c + w] = BG

    placements = _place_fragments(grid, holes, template_colors, top, left, template_h, template_w)
    if placements is None:
        return None

    # Validate the intended one-to-one matching explicitly before accepting.
    expected = grid.copy()
    for placement in placements:
        expected[placement["r"] : placement["r"] + placement["h"], placement["c"] : placement["c"] + placement["w"]] = BG
    for hole in holes:
        region = regions[hole["region_idx"]]
        expected[top + hole["r"] : top + hole["r"] + hole["h"], left + hole["c"] : left + hole["c"] + hole["w"]] = region["color"]

    solved = _solve(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
