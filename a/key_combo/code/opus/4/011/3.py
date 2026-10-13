from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BG = 0
RED = 2
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]

ANGLE_TO_STEP = {
    0: (-1, 0),
    45: (-1, 1),
    90: (0, 1),
    135: (1, 1),
    180: (1, 0),
    225: (1, -1),
    270: (0, -1),
    315: (-1, -1),
}
ANGLE_ORDER = [0, 45, 90, 135, 180, 225, 270, 315]
CARDINAL_ANGLES = [0, 90, 180, 270]
DIAGONAL_ANGLES = [45, 135, 225, 315]

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRY_LIMIT = 400


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def plus_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = center
    return {
        (cr, cc),
        (cr - 1, cc),
        (cr + 1, cc),
        (cr, cc - 1),
        (cr, cc + 1),
    }


def find_plus_center(grid: np.ndarray) -> tuple[int, int]:
    red = np.argwhere(grid == RED)
    assert red.shape == (5, 2), "expected exactly one five-pixel red plus"
    r0, c0, r1, c1 = bbox_coords(red)
    assert (r1 - r0, c1 - c0) == (2, 2), "red plus must have a 3x3 bounding box"
    center = (r0 + 1, c0 + 1)
    assert set(map(tuple, red)) == plus_cells(center), "red pixels must form a plus"
    return center


def extract_regions(grid: np.ndarray) -> list[dict]:
    region_cells = (grid != BG) & (grid != RED)
    comps = components_mask(region_cells)
    regions: list[dict] = []
    seen_colors: set[int] = set()
    for idx, coords in enumerate(comps):
        colors = {int(v) for v in grid[coords[:, 0], coords[:, 1]]}
        assert len(colors) == 1, "each foreground region must be monochrome"
        color = colors.pop()
        assert color not in seen_colors, "moving-region colors must be unique"
        seen_colors.add(color)
        regions.append(
            {
                "id": idx,
                "color": color,
                "coords": coords,
                "coord_set": {tuple(map(int, rc)) for rc in coords},
                "bbox": bbox_coords(coords),
            }
        )
    return regions


def cell_ray_angles(center: tuple[int, int], r: int, c: int) -> set[int]:
    cr, cc = center
    if (r, c) == center:
        return set()
    hits: set[int] = set()
    if c == cc:
        hits.add(0 if r < cr else 180)
    if r == cr:
        hits.add(90 if c > cc else 270)
    dr = r - cr
    dc = c - cc
    if abs(dr) == abs(dc):
        if dr < 0 and dc > 0:
            hits.add(45)
        elif dr > 0 and dc > 0:
            hits.add(135)
        elif dr > 0 and dc < 0:
            hits.add(225)
        elif dr < 0 and dc < 0:
            hits.add(315)
    return hits


def centroid_angle(center: tuple[int, int], coords: np.ndarray) -> float:
    cr, cc = center
    mean_r, mean_c = coords.mean(axis=0)
    return math.degrees(math.atan2(float(mean_c - cc), float(cr - mean_r))) % 360.0


def scan_activations(grid: np.ndarray, center: tuple[int, int], regions: list[dict]) -> dict[int, int]:
    coord_to_region = {}
    for region in regions:
        for cell in region["coord_set"]:
            coord_to_region[cell] = int(region["id"])

    activations: dict[int, int] = {}
    for angle in ANGLE_ORDER:
        dr, dc = ANGLE_TO_STEP[angle]
        r, c = center[0] + dr, center[1] + dc
        while in_bounds(grid.shape, r, c):
            value = int(grid[r, c])
            if value not in (BG, RED):
                region_id = coord_to_region[(r, c)]
                activations[region_id] = min(angle, activations.get(region_id, angle))
            r += dr
            c += dc
    return activations


def transformed_from_regions(grid: np.ndarray, regions: list[dict], activations: dict[int, int]) -> np.ndarray:
    active = [region for region in regions if int(region["id"]) in activations]
    assert len(active) >= 3, "expected at least three activated regions"
    assert len(active) < len(regions), "expected at least one untouched region"

    ordered = sorted(active, key=lambda region: activations[int(region["id"])])
    angles = [activations[int(region["id"])] for region in ordered]
    assert len(set(angles)) == len(angles), "activated regions must have distinct ray angles"

    out = np.array(grid, copy=True)
    for idx, region in enumerate(ordered):
        src = ordered[(idx - 1) % len(ordered)]
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = int(src["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor only the regions hit by the red plus's eight hidden rays, cycling colors by ray direction order.

    Input structure:
    - The grid has black background, one five-pixel red plus, and several detached monochrome 4-connected regions in distinct non-red colors.
    - The red plus provides a unique center from which the eight cardinal and diagonal rays are emitted.
    - Some regions contain at least one pixel on one of those eight ray lines, while at least one other region avoids every ray line.

    Rule:
    - Keep the red plus fixed and treat it as the ray origin.
    - Scan from the plus center along up, up-right, right, down-right, down, down-left, left, and up-left until leaving the grid; whenever a ray crosses a non-background non-red pixel, mark that pixel's whole 4-connected region as activated.
    - Assign each activated region the smallest ray angle that hits it, sort activated regions by those angles, and cyclically shift their colors so each activated region receives the previous activated region's color.
    - Leave every unactivated region unchanged.

    Key insight:
    - The rays are selectors rather than drawn output: they choose which regions participate and also provide the ordering key for the color cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert np.any(g == RED), "expected a red plus"

    center = find_plus_center(g)
    regions = extract_regions(g)
    activations = scan_activations(g, center, regions)
    return transformed_from_regions(g, regions, activations)


def ray_cells(shape: tuple[int, int], center: tuple[int, int], angle: int) -> list[tuple[int, int]]:
    dr, dc = ANGLE_TO_STEP[angle]
    r, c = center[0] + dr, center[1] + dc
    out = []
    while in_bounds(shape, r, c):
        out.append((r, c))
        r += dr
        c += dc
    return out


def all_ray_cells(shape: tuple[int, int], center: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for angle in ANGLE_ORDER:
        out.update(ray_cells(shape, center, angle))
    return out


def region_isolated(
    shape: tuple[int, int],
    cells: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> bool:
    for r, c in cells:
        if not in_bounds(shape, r, c) or (r, c) in blocked:
            return False
        for nr, nc in neighbors4(r, c):
            if in_bounds(shape, nr, nc) and (nr, nc) in blocked and (nr, nc) not in cells:
                return False
    return True


def expand_blocked(shape: tuple[int, int], cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blocked = set(cells)
    for r, c in list(cells):
        for nr, nc in neighbors4(r, c):
            if in_bounds(shape, nr, nc):
                blocked.add((nr, nc))
    return blocked


def choose_center(h: int, w: int) -> tuple[int, int]:
    return (
        h // 2 + random.randint(-1, 1),
        w // 2 + random.randint(-1, 1),
    )


def bias_vectors_for_angle(angle: int) -> list[tuple[int, int]]:
    if angle == 0:
        return [(0, 1), (0, -1), (1, 0)]
    if angle == 45:
        return [(-1, 0), (0, 1), (1, 0)]
    if angle == 90:
        return [(-1, 0), (1, 0), (0, -1)]
    if angle == 135:
        return [(0, 1), (1, 0), (0, -1)]
    if angle == 180:
        return [(0, 1), (0, -1), (-1, 0)]
    if angle == 225:
        return [(1, 0), (0, -1), (-1, 0)]
    if angle == 270:
        return [(-1, 0), (1, 0), (0, 1)]
    return [(0, -1), (-1, 0), (1, 0)]


def grow_region(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    size: int,
    allowed: set[tuple[int, int]],
    preferred: list[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    cells = {anchor}
    pref_index = {vec: idx for idx, vec in enumerate(preferred)}
    while len(cells) < size:
        frontier = []
        for r, c in sorted(cells):
            for nr, nc in neighbors4(r, c):
                if (nr, nc) not in allowed or (nr, nc) in cells:
                    continue
                dr = nr - anchor[0]
                dc = nc - anchor[1]
                score = 0
                for idx, vec in enumerate(preferred):
                    score += (len(preferred) - idx) * (dr * vec[0] + dc * vec[1])
                frontier.append((score, random.random(), (nr, nc)))
        if not frontier:
            return None
        frontier.sort(reverse=True)
        top = frontier[: min(6, len(frontier))]
        cells.add(random.choice(top)[2])
    return cells


def line_distance(center: tuple[int, int], r: int, c: int) -> int:
    cr, cc = center
    vals = [
        abs(r - cr),
        abs(c - cc),
        abs((r - cr) - (c - cc)),
        abs((r - cr) + (c - cc)),
    ]
    return min(vals)


def allowed_patch(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    center: tuple[int, int],
    blocked: set[tuple[int, int]],
    chosen_angle: int | None,
) -> set[tuple[int, int]]:
    h, w = shape
    ar, ac = anchor
    allowed: set[tuple[int, int]] = set()
    for r in range(max(0, ar - 3), min(h, ar + 4)):
        for c in range(max(0, ac - 3), min(w, ac + 4)):
            if (r, c) in blocked:
                continue
            hits = cell_ray_angles(center, r, c)
            if chosen_angle is None:
                if hits:
                    continue
            else:
                if not hits.issubset({chosen_angle}):
                    continue
            if abs(r - center[0]) + abs(c - center[1]) <= 2:
                continue
            allowed.add((r, c))
    return allowed


def build_active_region(
    shape: tuple[int, int],
    center: tuple[int, int],
    angle: int,
    blocked: set[tuple[int, int]],
    require_multi_ray: bool,
) -> set[tuple[int, int]] | None:
    candidates = ray_cells(shape, center, angle)[2:-1]
    random.shuffle(candidates)
    biases = bias_vectors_for_angle(angle)
    for anchor in candidates:
        if anchor in blocked:
            continue
        allowed = allowed_patch(shape, anchor, center, blocked, angle)
        if anchor not in allowed:
            continue
        for _ in range(10):
            size = random.randint(3, 10)
            pref = random.sample(biases, len(biases))
            region = grow_region(shape, anchor, size, allowed, pref)
            if region is None or not region_isolated(shape, region, blocked):
                continue
            ray_hits = sum(1 for cell in region if angle in cell_ray_angles(center, cell[0], cell[1]))
            if require_multi_ray and ray_hits < 2:
                continue
            if ray_hits < 1:
                continue
            return region
    return None


def build_inactive_region(
    shape: tuple[int, int],
    center: tuple[int, int],
    blocked: set[tuple[int, int]],
    prefer_near_ray: bool,
) -> set[tuple[int, int]] | None:
    h, w = shape
    anchors = []
    for r in range(h):
        for c in range(w):
            if (r, c) in blocked:
                continue
            if cell_ray_angles(center, r, c):
                continue
            if abs(r - center[0]) + abs(c - center[1]) <= 2:
                continue
            anchors.append((r, c))
    if prefer_near_ray:
        anchors.sort(key=lambda cell: (line_distance(center, cell[0], cell[1]), random.random()))
    else:
        random.shuffle(anchors)
    for anchor in anchors[:80]:
        allowed = allowed_patch(shape, anchor, center, blocked, None)
        if anchor not in allowed:
            continue
        for _ in range(8):
            size = random.randint(3, 10)
            pref = random.sample([(1, 0), (-1, 0), (0, 1), (0, -1)], 4)
            region = grow_region(shape, anchor, size, allowed, pref)
            if region is None or not region_isolated(shape, region, blocked):
                continue
            if any(cell_ray_angles(center, r, c) for r, c in region):
                continue
            return region
    return None


def expected_from_activation_order(grid: np.ndarray, region_order: list[int]) -> np.ndarray:
    regions = extract_regions(grid)
    region_by_id = {int(region["id"]): region for region in regions}
    out = np.array(grid, copy=True)
    ordered = [region_by_id[region_id] for region_id in region_order]
    for idx, region in enumerate(ordered):
        src = ordered[(idx - 1) % len(ordered)]
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = int(src["color"])
    return out


def semantic_check(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        center = find_plus_center(inp)
        assert set(np.unique(inp)).issubset(set(range(10)))
        regions = extract_regions(inp)
        assert 5 <= len(regions) <= 8
        for region in regions:
            assert 3 <= len(region["coord_set"]) <= 10
            for r, c in region["coord_set"]:
                assert not plus_cells(center) & {tuple(nei) for nei in neighbors4(r, c)}
        activations = scan_activations(inp, center, regions)
        assert 3 <= len(activations) < len(regions)
        assert len(set(activations.values())) == len(activations)

        active_regions = [region for region in regions if int(region["id"]) in activations]
        angles = [activations[int(region["id"])] for region in active_regions]
        assert any(angle in CARDINAL_ANGLES for angle in angles)
        assert any(angle in DIAGONAL_ANGLES for angle in angles)
        assert any(
            min(line_distance(center, r, c) for r, c in region["coord_set"]) == 1
            for region in regions
            if int(region["id"]) not in activations
        )

        actual = transformed_from_regions(inp, regions, activations)
        assert np.array_equal(actual, out)
        assert not np.array_equal(inp, out)

        active_sorted = sorted(active_regions, key=lambda region: activations[int(region["id"])])
        changed = 0
        unchanged = 0
        for region in regions:
            coords = region["coords"]
            before = int(inp[coords[0, 0], coords[0, 1]])
            after_vals = {int(v) for v in out[coords[:, 0], coords[:, 1]]}
            assert len(after_vals) == 1
            after = after_vals.pop()
            if int(region["id"]) in activations:
                assert before != after
                changed += 1
            else:
                assert before == after
                unchanged += 1
        assert changed >= 3 and unchanged >= 1

        ray_order = [int(region["id"]) for region in active_sorted]
        centroid_order = [
            int(region["id"])
            for region in sorted(active_regions, key=lambda region: centroid_angle(center, region["coords"]))
        ]
        assert ray_order != centroid_order

        desc_order = list(reversed(ray_order))
        assert not np.array_equal(expected_from_activation_order(inp, desc_order), out)
        assert plus_cells(center) <= {tuple(map(int, rc)) for rc in np.argwhere(out == RED)}
        return True
    except AssertionError:
        return False


def build_attempt() -> np.ndarray | None:
    h = random.randint(15, 21)
    w = random.randint(15, 21)
    grid = np.zeros((h, w), dtype=int)
    center = choose_center(h, w)
    cr, cc = center
    if cr < 4 or cr > h - 5 or cc < 4 or cc > w - 5:
        return None

    cross = plus_cells(center)
    blocked = expand_blocked(grid.shape, cross)
    for r, c in cross:
        grid[r, c] = RED

    num_active = random.randint(3, 6)
    num_regions = random.randint(max(5, num_active + 1), 8)

    chosen_cardinal = random.choice(CARDINAL_ANGLES)
    chosen_diagonal = random.choice(DIAGONAL_ANGLES)
    remaining = [a for a in ANGLE_ORDER if a not in {chosen_cardinal, chosen_diagonal}]
    active_angles = [chosen_cardinal, chosen_diagonal] + random.sample(remaining, num_active - 2)
    random.shuffle(active_angles)

    regions: list[set[tuple[int, int]]] = []
    colors = random.sample(PALETTE, num_regions)

    multi_ray_done = False
    for angle in active_angles:
        require_multi = (not multi_ray_done) and angle in CARDINAL_ANGLES
        region = build_active_region(grid.shape, center, angle, blocked, require_multi)
        if region is None:
            return None
        if require_multi:
            multi_ray_done = True
        regions.append(region)
        blocked |= expand_blocked(grid.shape, region)

    if not multi_ray_done:
        return None

    inactive_count = num_regions - num_active
    for idx in range(inactive_count):
        region = build_inactive_region(grid.shape, center, blocked, prefer_near_ray=(idx == 0))
        if region is None:
            return None
        regions.append(region)
        blocked |= expand_blocked(grid.shape, region)

    random.shuffle(regions)
    for color, region in zip(colors, regions):
        for r, c in region:
            grid[r, c] = color

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not semantic_check(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
