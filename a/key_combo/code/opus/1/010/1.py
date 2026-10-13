import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

BACKGROUND = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]

CANVAS_HEIGHT_RANGE = (16, 24)
CANVAS_WIDTH_RANGE = (16, 24)
GENERATION_ATTEMPTS = 400
OBJECT_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR8:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or not mask[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _count_neighbors(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    count = 0
    for dr, dc in DIR8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            count += 1
    return count


def _find_endpoints(mask: np.ndarray) -> list[tuple[int, int]]:
    points = np.argwhere(mask)
    endpoints: list[tuple[int, int]] = []
    for r, c in points:
        rr = int(r)
        cc = int(c)
        if _count_neighbors(mask, rr, cc) == 1:
            endpoints.append((rr, cc))
    endpoints.sort()
    return endpoints


def _find_facing_pairs(mask: np.ndarray, occupied: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    h, w = mask.shape
    endpoints = _find_endpoints(mask)
    endpoint_set = set(endpoints)
    partner_of: dict[tuple[int, int], tuple[int, int]] = {}

    for r, c in endpoints:
        candidates: list[tuple[int, int]] = []
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            gap_len = 0
            while 0 <= nr < h and 0 <= nc < w and not occupied[nr, nc]:
                gap_len += 1
                nr += dr
                nc += dc
            if gap_len == 0:
                continue
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in endpoint_set:
                candidates.append((nr, nc))
        if len(candidates) != 1:
            return None
        partner_of[(r, c)] = candidates[0]

    pairs = set()
    for a, b in partner_of.items():
        if partner_of.get(b) != a:
            return None
        pairs.add(tuple(sorted((a, b))))
    return sorted(pairs, key=lambda pair: pair[0])


def _fill_pairs(mask: np.ndarray, pairs: list[tuple[tuple[int, int], tuple[int, int]]]) -> np.ndarray:
    out = mask.copy()
    for (r1, c1), (r2, c2) in pairs:
        if r1 == r2:
            out[r1, min(c1, c2): max(c1, c2) + 1] = True
        elif c1 == c2:
            out[min(r1, r2): max(r1, r2) + 1, c1] = True
        else:
            raise AssertionError("facing endpoints must share a row or a column")
    return out


def _repair_color(grid: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = grid.copy()
    mask = grid == color
    occupied = grid != bg
    pairs = _find_facing_pairs(mask, occupied)
    assert pairs is not None, f"ambiguous facing endpoints for color {color}"
    for (r1, c1), (r2, c2) in pairs:
        if r1 == r2:
            out[r1, min(c1, c2): max(c1, c2) + 1] = color
        else:
            out[min(r1, r2): max(r1, r2) + 1, c1] = color
    return out


def _row_span_fill(repaired: np.ndarray, bg: int) -> np.ndarray:
    out = repaired.copy()
    inf = repaired.shape[0] + repaired.shape[1] + 10
    owner_r = np.full(repaired.shape, inf, dtype=int)
    owner_c = np.full(repaired.shape, inf, dtype=int)

    components: list[tuple[int, int, int, np.ndarray]] = []
    for color in sorted(int(v) for v in np.unique(repaired) if int(v) != bg):
        for coords in _components_of_mask(repaired == color):
            order = np.lexsort((coords[:, 1], coords[:, 0]))
            anchor = coords[order[0]]
            components.append((int(anchor[0]), int(anchor[1]), color, coords))
    components.sort(key=lambda item: (item[0], item[1]))

    for anchor_r, anchor_c, color, coords in components:
        for r in sorted(set(int(v) for v in coords[:, 0])):
            row_cols = coords[coords[:, 0] == r, 1]
            c0 = int(row_cols.min())
            c1 = int(row_cols.max())
            for c in range(c0, c1 + 1):
                if repaired[r, c] != bg:
                    continue
                if (anchor_r, anchor_c) < (owner_r[r, c], owner_c[r, c]):
                    out[r, c] = color
                    owner_r[r, c] = anchor_r
                    owner_c[r, c] = anchor_c
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each broken monochrome outline into a closed contour, then fill the repaired contour row by row.

    Input structure:
    - The grid is a black-background canvas with 2 or 3 disconnected one-pixel-wide colored outlines.
    - Each color forms a single irregular contour fragment set with exactly 1 or 2 axis-aligned gaps.
    - The two endpoints of each gap are same-color pixels on one row or one column, facing each other through only background cells.

    Rule:
    - For each foreground color, find same-color endpoints that have exactly one same-color 8-neighbor.
    - Pair endpoints that face each other along one row or one column through a pure background gap, and fill the whole gap with that color.
    - After repairing the gaps, treat each repaired color component as a closed outline and, on every occupied row, fill the full span from its leftmost outline pixel to its rightmost outline pixel.
    - If row-span fills from different colors want the same background pixel, keep the color whose repaired contour has the lexicographically smallest anchor pixel (topmost, then leftmost).

    Key insight:
    - The endpoint-gap step is not the final goal: it restores geometric closure so the later row-span fill has well-defined left and right boundaries on every occupied row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    repaired = g.copy()
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        repaired = _repair_color(repaired, color, bg)

    return _row_span_fill(repaired, bg)


def _interp_int(a: int, b: int, steps: int) -> list[int]:
    if steps == 0:
        return [a]
    return [int(round(a + (b - a) * i / steps)) for i in range(steps + 1)]


def _outline_from_solid(solid: np.ndarray) -> np.ndarray:
    h, w = solid.shape
    outline = np.zeros_like(solid, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not solid[r, c]:
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not solid[nr, nc]:
                    outline[r, c] = True
                    break
    return outline


def _row_fill_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for r in range(mask.shape[0]):
        cols = np.where(mask[r])[0]
        if cols.size == 0:
            continue
        out[r, int(cols.min()): int(cols.max()) + 1] = True
    return out


def _generate_solid_shape(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    h = rng.randint(6, 10)
    mid_left = rng.randint(2, h - 3)
    mid_right = rng.randint(2, h - 3)
    top_width = rng.randint(4, 6)
    mid_width = rng.randint(max(top_width, 4) + 1, min(10, max(top_width, 4) + 4))
    bottom_width = rng.randint(4, 6)

    center_top = 0
    center_mid = rng.randint(-3, 3)
    center_bottom = rng.randint(-3, 3)

    left_top = center_top - top_width // 2
    right_top = left_top + top_width - 1
    left_mid = center_mid - mid_width // 2
    right_mid = left_mid + mid_width - 1
    left_bottom = center_bottom - bottom_width // 2
    right_bottom = left_bottom + bottom_width - 1

    left_profile = (
        _interp_int(left_top, left_mid, mid_left)
        + _interp_int(left_mid, left_bottom, h - 1 - mid_left)[1:]
    )
    right_profile = (
        _interp_int(right_top, right_mid, mid_right)
        + _interp_int(right_mid, right_bottom, h - 1 - mid_right)[1:]
    )
    if len(left_profile) != h or len(right_profile) != h:
        return None

    for left, right in zip(left_profile, right_profile):
        if right - left < 3:
            return None

    min_col = min(left_profile)
    max_col = max(right_profile)
    solid = np.zeros((h, max_col - min_col + 1), dtype=bool)
    for r, (left, right) in enumerate(zip(left_profile, right_profile)):
        solid[r, left - min_col: right - min_col + 1] = True

    outline = _outline_from_solid(solid)
    if len(_components_of_mask(outline)) != 1:
        return None
    if not np.array_equal(_row_fill_mask(outline), solid):
        return None
    return solid, outline


def _cells_are_separated(cells_a: tuple[tuple[int, int], ...], cells_b: tuple[tuple[int, int], ...]) -> bool:
    for r1, c1 in cells_a:
        for r2, c2 in cells_b:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _try_gap_combo(
    outline: np.ndarray,
    removed_groups: list[tuple[tuple[int, int], ...]],
    orientations: list[str],
) -> dict | None:
    broken = outline.copy()
    removed_cells = []
    for group in removed_groups:
        removed_cells.extend(group)
        for r, c in group:
            broken[r, c] = False

    pairs = _find_facing_pairs(broken, broken)
    if pairs is None or len(pairs) != len(removed_groups):
        return None
    repaired = _fill_pairs(broken, pairs)
    if not np.array_equal(repaired, outline):
        return None

    pattern = "".join(sorted(orientations))
    return {
        "broken": broken,
        "pattern": pattern,
        "gap_count": len(removed_groups),
        "removed": tuple(sorted(removed_cells)),
    }


def _enumerate_gap_combos(outline: np.ndarray, desired_pattern: str) -> list[dict]:
    h, w = outline.shape
    singles: list[tuple[tuple[tuple[int, int], ...], str]] = []
    seen_single = set()
    need_h = "H" in desired_pattern
    need_v = "V" in desired_pattern

    def add_single(cells: list[tuple[int, int]], orientation: str) -> None:
        key = tuple(sorted(cells))
        if key in seen_single:
            return
        combo = _try_gap_combo(outline, [key], [orientation])
        if combo is None:
            return
        seen_single.add(key)
        singles.append((key, orientation))

    if need_h:
        for r in range(h):
            c = 0
            while c < w:
                if not outline[r, c]:
                    c += 1
                    continue
                start = c
                while c + 1 < w and outline[r, c + 1]:
                    c += 1
                end = c
                run_len = end - start + 1
                if run_len >= 3:
                    max_gap = min(3, run_len - 2)
                    for gap_len in range(1, max_gap + 1):
                        for gap_start in range(start + 1, end - gap_len + 1):
                            add_single([(r, x) for x in range(gap_start, gap_start + gap_len)], "H")
                c += 1

    if need_v:
        for c in range(w):
            r = 0
            while r < h:
                if not outline[r, c]:
                    r += 1
                    continue
                start = r
                while r + 1 < h and outline[r + 1, c]:
                    r += 1
                end = r
                run_len = end - start + 1
                if run_len >= 3:
                    max_gap = min(3, run_len - 2)
                    for gap_len in range(1, max_gap + 1):
                        for gap_start in range(start + 1, end - gap_len + 1):
                            add_single([(x, c) for x in range(gap_start, gap_start + gap_len)], "V")
                r += 1

    combos: list[dict] = []
    if desired_pattern in {"H", "V"}:
        for cells, orientation in singles:
            if orientation == desired_pattern:
                combo = _try_gap_combo(outline, [cells], [orientation])
                if combo is not None:
                    combos.append(combo)
        return combos

    required = sorted(desired_pattern)
    for i, (cells_a, orientation_a) in enumerate(singles):
        for cells_b, orientation_b in singles[i + 1:]:
            if sorted([orientation_a, orientation_b]) != required:
                continue
            if set(cells_a) & set(cells_b):
                continue
            if not _cells_are_separated(cells_a, cells_b):
                continue
            combo = _try_gap_combo(outline, [cells_a, cells_b], [orientation_a, orientation_b])
            if combo is not None:
                combos.append(combo)
    return combos


def _mask_anchor(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask)
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    anchor = coords[order[0]]
    return int(anchor[0]), int(anchor[1])


def _generate_object(rng: random.Random, color: int, desired_pattern: str) -> dict | None:
    for _ in range(OBJECT_ATTEMPTS):
        built = _generate_solid_shape(rng)
        if built is None:
            continue
        solid, outline = built
        combos = _enumerate_gap_combos(outline, desired_pattern)
        if not combos:
            continue
        combo = rng.choice(combos)
        broken = combo["broken"]
        pairs = _find_facing_pairs(broken, broken)
        if pairs is None or len(pairs) != combo["gap_count"]:
            continue
        return {
            "color": color,
            "solid_mask": solid,
            "outline_mask": outline,
            "broken_mask": broken,
            "anchor_local": _mask_anchor(outline),
            "gap_count": combo["gap_count"],
            "pattern": desired_pattern,
        }
    return None


def _sample_patterns(rng: random.Random, object_count: int) -> list[str]:
    if object_count == 2:
        if rng.random() < 0.5:
            patterns = ["H", rng.choice(["VV", "HV"])]
        else:
            patterns = ["V", rng.choice(["HH", "HV"])]
    else:
        patterns = ["H", "V", rng.choice(["HH", "VV", "HV"])]
    rng.shuffle(patterns)
    return patterns


def _can_place(input_occ: np.ndarray, solid_occ: np.ndarray, obj: dict, top: int, left: int) -> bool:
    h, w = input_occ.shape
    broken = obj["broken_mask"]
    solid = obj["solid_mask"]

    broken_points = np.argwhere(broken)
    for r, c in broken_points:
        gr = top + int(r)
        gc = left + int(c)
        if input_occ[gr, gc]:
            return False
        for dr, dc in DIR8:
            nr, nc = gr + dr, gc + dc
            if 0 <= nr < h and 0 <= nc < w and input_occ[nr, nc]:
                return False

    solid_points = np.argwhere(solid)
    for r, c in solid_points:
        if solid_occ[top + int(r), left + int(c)]:
            return False
    return True


def _place_objects(objects: list[dict], height: int, width: int, rng: random.Random) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int] | None] = [None] * len(objects)
    order = sorted(range(len(objects)), key=lambda idx: -int(objects[idx]["solid_mask"].sum()))
    input_occ = np.zeros((height, width), dtype=bool)
    solid_occ = np.zeros((height, width), dtype=bool)

    for idx in order:
        obj = objects[idx]
        ph, pw = obj["broken_mask"].shape
        if ph > height or pw > width:
            return None

        candidates: list[tuple[int, int]] = []
        for _ in range(PLACEMENT_ATTEMPTS):
            top = rng.randint(0, height - ph)
            left = rng.randint(0, width - pw)
            if _can_place(input_occ, solid_occ, obj, top, left):
                candidates.append((top, left))
                if len(candidates) >= 24:
                    break
        if not candidates:
            return None

        top, left = rng.choice(candidates)
        placements[idx] = (top, left)

        rr, cc = np.where(obj["broken_mask"])
        input_occ[top + rr, left + cc] = True
        rr, cc = np.where(obj["solid_mask"])
        solid_occ[top + rr, left + cc] = True

    assert all(p is not None for p in placements)
    return [p for p in placements if p is not None]


def _compose_scene(
    objects: list[dict],
    placements: list[tuple[int, int]],
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), BACKGROUND, dtype=int)
    out = np.full((height, width), BACKGROUND, dtype=int)
    inf = height + width + 10
    owner_r = np.full((height, width), inf, dtype=int)
    owner_c = np.full((height, width), inf, dtype=int)

    solid_items: list[tuple[int, int, int, np.ndarray]] = []
    for obj, (top, left) in zip(objects, placements):
        color = int(obj["color"])
        rr, cc = np.where(obj["broken_mask"])
        inp[top + rr, left + cc] = color

        solid_coords = np.argwhere(obj["solid_mask"]).copy()
        solid_coords[:, 0] += top
        solid_coords[:, 1] += left
        anchor_r = top + int(obj["anchor_local"][0])
        anchor_c = left + int(obj["anchor_local"][1])
        solid_items.append((anchor_r, anchor_c, color, solid_coords))

    solid_items.sort(key=lambda item: (item[0], item[1]))
    for anchor_r, anchor_c, color, coords in solid_items:
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if (anchor_r, anchor_c) < (owner_r[rr, cc], owner_c[rr, cc]):
                out[rr, cc] = color
                owner_r[rr, cc] = anchor_r
                owner_c[rr, cc] = anchor_c
    return inp, out


def _check_input_separation(grid: np.ndarray) -> bool:
    points = np.argwhere(grid != BACKGROUND)
    for r, c in points:
        color = int(grid[r, c])
        for dr, dc in DIR8:
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                other = int(grid[nr, nc])
                if other != BACKGROUND and other != color:
                    return False
    return True


def _generate_case(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    if rng is None:
        rng = random.Random()

    for _ in range(GENERATION_ATTEMPTS):
        height = rng.randint(*CANVAS_HEIGHT_RANGE)
        width = rng.randint(*CANVAS_WIDTH_RANGE)
        object_count = 2 if rng.random() < 0.55 else 3
        patterns = _sample_patterns(rng, object_count)
        colors = rng.sample(COLOR_POOL, object_count)

        objects: list[dict] = []
        ok = True
        for color, pattern in zip(colors, patterns):
            obj = _generate_object(rng, color, pattern)
            if obj is None:
                ok = False
                break
            objects.append(obj)
        if not ok:
            continue

        placements = _place_objects(objects, height, width, rng)
        if placements is None:
            continue

        inp, expected = _compose_scene(objects, placements, height, width)
        if not _check_input_separation(inp):
            continue

        # Independent generation-time semantic checks.
        for obj, (top, left) in zip(objects, placements):
            color = int(obj["color"])
            local_mask = inp == color
            pairs = _find_facing_pairs(local_mask, inp != BACKGROUND)
            if pairs is None or len(pairs) != obj["gap_count"]:
                ok = False
                break
            repaired = _fill_pairs(local_mask, pairs)
            outline = np.zeros_like(inp, dtype=bool)
            rr, cc = np.where(obj["outline_mask"])
            outline[top + rr, left + cc] = True
            if not np.array_equal(repaired, outline):
                ok = False
                break
        if not ok:
            continue

        return inp, expected, {"patterns": tuple(patterns), "colors": tuple(colors)}

    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    return _generate_case()[0]


def _manual_gap_case() -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((8, 12), dtype=int)
    color = 2

    grid[1, 1:4] = color
    grid[1, 6:9] = color
    grid[2:5, 1] = color
    grid[2:5, 8] = color
    grid[5, 1:9] = color

    expected = np.zeros_like(grid)
    expected[1:6, 1:9] = color
    return grid, expected


def _manual_priority_case() -> tuple[np.ndarray, np.ndarray]:
    # This is a direct row-span priority test rather than a full legal puzzle input.
    grid = np.array(
        [
            [0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 2, 2, 0, 0, 0, 0],
            [0, 0, 0, 2, 2, 2, 0, 0, 0, 3],
            [0, 0, 0, 2, 3, 0, 2, 3, 3, 0],
            [0, 0, 0, 0, 0, 3, 3, 3, 3, 3],
            [0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )

    expected = np.array(
        [
            [0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 2, 2, 0, 0, 0, 0],
            [0, 0, 0, 2, 2, 2, 0, 0, 0, 3],
            [0, 0, 0, 2, 3, 2, 2, 3, 3, 0],
            [0, 0, 0, 0, 0, 3, 3, 3, 3, 3],
            [0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    return grid, expected
