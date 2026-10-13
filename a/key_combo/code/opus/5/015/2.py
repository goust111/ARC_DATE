from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 7
BLACK = 0
OUTLINE_COLORS = (1, 2, 3, 4)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

# Small 4-connected black blotches used by the generator.
BASE_SHAPES = [
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (0, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    dirs = DIR4 if int(connectivity) == 4 else DIR8
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _diamond_outline_coords(shape: tuple[int, int], cr: int, cc: int, radius: int) -> list[tuple[int, int]]:
    h, w = shape
    if radius == 0:
        return [(cr, cc)] if 0 <= cr < h and 0 <= cc < w else []
    pts = []
    for r in range(max(0, cr - radius), min(h, cr + radius + 1)):
        span = radius - abs(r - cr)
        c1 = cc - span
        c2 = cc + span
        if 0 <= c1 < w:
            pts.append((r, c1))
        if c2 != c1 and 0 <= c2 < w:
            pts.append((r, c2))
    return pts


def _diamond_filled_coords(shape: tuple[int, int], cr: int, cc: int, radius: int) -> set[tuple[int, int]]:
    h, w = shape
    pts = set()
    for r in range(max(0, cr - radius), min(h, cr + radius + 1)):
        span = radius - abs(r - cr)
        for c in range(max(0, cc - span), min(w, cc + span + 1)):
            pts.add((r, c))
    return pts


def _strict_interior_mask(shape: tuple[int, int], cr: int, cc: int, radius: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if radius <= 0:
        return mask
    for r in range(max(0, cr - radius + 1), min(h, cr + radius)):
        span = radius - 1 - abs(r - cr)
        for c in range(max(0, cc - span), min(w, cc + span + 1)):
            mask[r, c] = True
    return mask


def _spot_allowed_rel_cells(radius: int) -> set[tuple[int, int]]:
    if radius < 2:
        return set()
    out = set()
    inner = radius - 2
    for dr in range(-inner, inner + 1):
        span = inner - abs(dr)
        for dc in range(-span, span + 1):
            out.add((dr, dc))
    return out


def _relative_spot_placements(radius: int) -> list[frozenset[tuple[int, int]]]:
    allowed = _spot_allowed_rel_cells(radius)
    placements: set[frozenset[tuple[int, int]]] = set()
    if not allowed:
        return []
    for shape in BASE_SHAPES:
        rows = [r for r, _ in shape]
        cols = [c for _, c in shape]
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                cells = tuple((dr + r, dc + c) for r, c in shape)
                if all(cell in allowed for cell in cells):
                    placements.add(frozenset(cells))
    return list(placements)


def _placements_touch(a: frozenset[tuple[int, int]], b: frozenset[tuple[int, int]]) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return True
    return False


def _sample_relative_components(radius: int, count: int) -> list[list[tuple[int, int]]] | None:
    placements = _relative_spot_placements(radius)
    if count == 0:
        return []
    if not placements:
        return None

    compatible = {i: [] for i in range(len(placements))}
    for i in range(len(placements)):
        for j in range(i + 1, len(placements)):
            if not _placements_touch(placements[i], placements[j]):
                compatible[i].append(j)

    for _ in range(80):
        order = list(range(len(placements)))
        random.shuffle(order)

        def dfs(start_indices: list[int], chosen: list[int]) -> list[int] | None:
            if len(chosen) == count:
                return chosen
            if len(start_indices) + len(chosen) < count:
                return None
            shuffled = start_indices[:]
            random.shuffle(shuffled)
            for idx in shuffled:
                next_allowed = [j for j in compatible[idx] if j in start_indices and j > idx]
                found = dfs(next_allowed, chosen + [idx])
                if found is not None:
                    return found
            return None

        found = dfs(order, [])
        if found is not None:
            return [sorted(list(placements[i])) for i in found]
    return None


def _parse_diamond(coords: np.ndarray) -> tuple[int, int, int]:
    r0, c0, r1, c1 = bbox_coords(coords)
    if (r1 - r0) != (c1 - c0) or ((r1 - r0) % 2) != 0:
        raise AssertionError("diamond bbox must be odd square")
    radius = (r1 - r0) // 2
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    expected = set(_diamond_outline_coords((r1 + 1, c1 + 1), cr, cc, radius))
    actual = {tuple(map(int, cell)) for cell in coords.tolist()}
    if actual != expected:
        raise AssertionError("component is not a full Manhattan diamond outline")
    return cr, cc, radius


def _detect_diamonds(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (bg, BLACK)):
        for comp in _components(g == int(color), connectivity=8):
            cr, cc, radius = _parse_diamond(comp)
            interior = _strict_interior_mask(g.shape, cr, cc, radius)
            objects.append(
                {
                    "color": int(color),
                    "center": (int(cr), int(cc)),
                    "radius": int(radius),
                    "interior": interior,
                }
            )
    return objects


def _render_objects(shape: tuple[int, int], bg: int, objects: list[dict]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for obj in sorted(objects, key=lambda item: (int(item["radius"]), int(item["color"]))):
        cr, cc = obj["center"]
        new_radius = int(obj["new_radius"])
        coords = _diamond_outline_coords(shape, cr, cc, new_radius)
        for r, c in coords:
            out[r, c] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count black interior components inside each diamond outline, then expand or contract the outline and erase the counted blotches.

    Input structure:
    - Orange background contains 2-4 one-pixel-thick Manhattan diamond outlines of distinct non-black colors.
    - Each outline encloses one or more black 4-connected blotches strictly inside the diamond interior.
    - Different diamonds are separated in the input, and the blotches sit on the orange interior background.

    Rule:
    - For each colored diamond, count the black 4-connected components inside its strict interior.
    - Odd counts expand the outline by one Manhattan layer; positive even counts contract it by one layer.
    - If a contraction reaches radius 0, draw a single center pixel of the same color; erase all black blotches.
    - When transformed outlines overlap, the diamond with the larger original radius overwrites the smaller one.

    Key insight:
    - The black marks are not output objects: they are hidden counters whose parity controls whether the surrounding diamond grows or shrinks before the counters disappear.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    transformed = []
    for obj in _detect_diamonds(g):
        count = len(_components((g == BLACK) & obj["interior"], connectivity=4))
        if count == 0:
            new_radius = obj["radius"]
        elif count % 2 == 1:
            new_radius = obj["radius"] + 1
        else:
            new_radius = obj["radius"] - 1
        transformed.append(
            {
                "color": obj["color"],
                "center": obj["center"],
                "radius": obj["radius"],
                "new_radius": int(new_radius),
            }
        )
    return _render_objects(g.shape, bg, transformed)


COUNT_OPTIONS = {
    2: [1],
    3: [1, 2],
    4: [1, 2, 3, 4],
    5: [1, 2, 3, 4, 5],
}


def _sample_radius_and_spots(parity: int, allowed_radii: list[int]) -> tuple[int, int, list[list[tuple[int, int]]]] | None:
    radii = allowed_radii[:]
    random.shuffle(radii)
    for radius in radii:
        counts = [n for n in COUNT_OPTIONS[radius] if n % 2 == parity]
        random.shuffle(counts)
        for count in counts:
            rel_comps = _sample_relative_components(radius, count)
            if rel_comps is not None:
                return int(radius), int(count), rel_comps
    return None


def _sample_center(radius: int, shape: tuple[int, int], existing: list[dict]) -> tuple[int, int] | None:
    h, w = shape
    ext = radius + 1
    rows = list(range(ext, h - ext))
    cols = list(range(ext, w - ext))
    random.shuffle(rows)
    random.shuffle(cols)
    for cr in rows:
        for cc in cols:
            ok = True
            for other in existing:
                dist = abs(cr - other["center"][0]) + abs(cc - other["center"][1])
                min_dist = (radius + 1) + (other["radius"] + 1) + 1
                if dist <= min_dist:
                    ok = False
                    break
            if ok:
                return int(cr), int(cc)
    return None


def _make_object(color: int, parity: int, shape: tuple[int, int], existing: list[dict], radius_choices: list[int]) -> dict | None:
    sampled = _sample_radius_and_spots(parity, radius_choices)
    if sampled is None:
        return None
    radius, count, rel_comps = sampled
    center = _sample_center(radius, shape, existing)
    if center is None:
        return None
    cr, cc = center
    comps = [[(cr + dr, cc + dc) for dr, dc in comp] for comp in rel_comps]
    return {
        "color": int(color),
        "center": (int(cr), int(cc)),
        "radius": int(radius),
        "count": int(count),
        "components": comps,
    }


def _place_overlap_pair(shape: tuple[int, int], colors: list[int]) -> list[dict] | None:
    h, w = shape
    axes = ["h", "v"]
    random.shuffle(axes)
    radius_pairs = [(5, 3), (4, 3), (5, 2), (4, 2)]
    random.shuffle(radius_pairs)
    for axis in axes:
        for big_radius, small_radius in radius_pairs:
            big_components = _sample_relative_components(big_radius, random.choice([1, 3, 5]))
            small_components = _sample_relative_components(small_radius, 1)
            if big_components is None or small_components is None:
                continue
            d = (big_radius + 1) + (small_radius + 1)
            if axis == "h":
                row_min = max(big_radius + 1, small_radius + 1)
                row_max = h - 1 - max(big_radius + 1, small_radius + 1)
                col_min = big_radius + 1
                col_max = w - 1 - (small_radius + 1) - d
                if row_min > row_max or col_min > col_max:
                    continue
                cr = random.randint(row_min, row_max)
                c_big = random.randint(col_min, col_max)
                centers = [(cr, c_big), (cr, c_big + d)]
            else:
                col_min = max(big_radius + 1, small_radius + 1)
                col_max = w - 1 - max(big_radius + 1, small_radius + 1)
                row_min = big_radius + 1
                row_max = h - 1 - (small_radius + 1) - d
                if row_min > row_max or col_min > col_max:
                    continue
                cc = random.randint(col_min, col_max)
                r_big = random.randint(row_min, row_max)
                centers = [(r_big, cc), (r_big + d, cc)]

            objects = []
            for color, radius, center, rel_comps in [
                (colors[0], big_radius, centers[0], big_components),
                (colors[1], small_radius, centers[1], small_components),
            ]:
                cr, cc = center
                objects.append(
                    {
                        "color": int(color),
                        "center": (int(cr), int(cc)),
                        "radius": int(radius),
                        "count": sum(1 for _ in rel_comps),
                        "components": [[(cr + dr, cc + dc) for dr, dc in comp] for comp in rel_comps],
                    }
                )
            if _transformed_overlap(objects[0], objects[1]):
                return objects
    return None


def _transformed_overlap(a: dict, b: dict) -> bool:
    a_pts = set(_diamond_outline_coords((64, 64), 32 + a["center"][0], 32 + a["center"][1], a["radius"] + 1))
    b_pts = set(_diamond_outline_coords((64, 64), 32 + b["center"][0], 32 + b["center"][1], b["radius"] + 1))
    return bool(a_pts & b_pts)


def _render_input(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for obj in objects:
        for r, c in _diamond_outline_coords(shape, obj["center"][0], obj["center"][1], obj["radius"]):
            grid[r, c] = int(obj["color"])
        for comp in obj["components"]:
            for r, c in comp:
                grid[r, c] = BLACK
    return grid


def _expected_from_plan(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    transformed = []
    for obj in objects:
        delta = 1 if (obj["count"] % 2 == 1) else -1
        transformed.append(
            {
                "color": obj["color"],
                "center": obj["center"],
                "radius": obj["radius"],
                "new_radius": obj["radius"] + delta,
            }
        )
    return _render_objects(shape, BG, transformed)


def _validate_plan(grid: np.ndarray, objects: list[dict], expect_overlap: bool) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    colors = [obj["color"] for obj in objects]
    if len(colors) != len(set(colors)):
        return False
    for obj in objects:
        cr, cc = obj["center"]
        radius = obj["radius"]
        if radius < 2:
            return False
        interior = _strict_interior_mask(g.shape, cr, cc, radius)
        comps = _components((g == BLACK) & interior, connectivity=4)
        if len(comps) != obj["count"]:
            return False
        for comp in comps:
            for r, c in comp.tolist():
                if abs(int(r) - cr) + abs(int(c) - cc) > radius - 2:
                    return False
    if expect_overlap and not _transformed_overlap(objects[0], objects[1]):
        return False
    if not np.array_equal(main(g), _expected_from_plan(g.shape, objects)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    expect_overlap = random.random() < 0.3
    h = random.randint(20, 24) if expect_overlap else random.randint(18, 24)
    w = random.randint(20, 24) if expect_overlap else random.randint(18, 24)
    n = random.randint(3, 4) if expect_overlap else random.randint(2, 4)
    colors = random.sample(list(OUTLINE_COLORS), n)

    objects: list[dict] = []
    remaining_colors = colors[:]
    if expect_overlap:
        pair = _place_overlap_pair((h, w), remaining_colors[:2])
        if pair is None:
            return None
        objects.extend(pair)
        remaining_colors = remaining_colors[2:]
        parities = [0]
        while len(parities) < n - 2:
            parities.append(random.randint(0, 1))
    else:
        parities = [0, 1][:n]
        while len(parities) < n:
            parities.append(random.randint(0, 1))
        random.shuffle(parities)

    for color, parity in zip(remaining_colors, parities):
        radius_choices = [2, 3, 4, 5] if parity == 1 else [3, 4, 5]
        obj = _make_object(color, parity, (h, w), objects, radius_choices)
        if obj is None:
            return None
        objects.append(obj)

    grid = _render_input((h, w), objects)
    if not _validate_plan(grid, objects, expect_overlap):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
