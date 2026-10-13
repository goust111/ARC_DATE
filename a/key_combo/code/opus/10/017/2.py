from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations
import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]

GEN_RETRIES = 400
GEN_SIZE_MIN = 13
GEN_SIZE_MAX = 19


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _anchor_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    cr, cc = center
    return [(r, c) for r in range(cr - 1, cr + 2) for c in range(cc - 1, cc + 2)]


def _find_anchor(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    anchors: list[tuple[int, int]] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if np.all(grid[r0 : r0 + 3, c0 : c0 + 3] == BLUE):
                anchors.append((r0 + 1, c0 + 1))
    assert len(anchors) == 1, "expected exactly one solid 3x3 blue anchor"
    return anchors[0]


def _direction_from_center(center: tuple[int, int], cell: tuple[int, int]) -> tuple[int, int] | None:
    dr = int(cell[0] - center[0])
    dc = int(cell[1] - center[1])
    if dr == 0 and dc == 0:
        return None
    if dr == 0:
        return 0, 1 if dc > 0 else -1
    if dc == 0:
        return 1 if dr > 0 else -1, 0
    if abs(dr) == abs(dc):
        return 1 if dr > 0 else -1, 1 if dc > 0 else -1
    return None


def _ray_distance(center: tuple[int, int], cell: tuple[int, int]) -> int:
    return max(abs(cell[0] - center[0]), abs(cell[1] - center[1]))


def _reachable_mask(passable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = passable.shape
    out = np.zeros((h, w), dtype=bool)
    if not passable[start]:
        return out
    q: deque[tuple[int, int]] = deque([start])
    out[start] = True
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if _in_bounds((h, w), nr, nc) and passable[nr, nc] and not out[nr, nc]:
                out[nr, nc] = True
                q.append((nr, nc))
    return out


def _component_ids(wall_mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = wall_mask.shape
    comp = np.full((h, w), -1, dtype=int)
    groups: list[list[tuple[int, int]]] = []
    next_id = 0
    for r in range(h):
        for c in range(w):
            if wall_mask[r, c] or comp[r, c] != -1:
                continue
            coords: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp[r, c] = next_id
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds((h, w), nr, nc) and not wall_mask[nr, nc] and comp[nr, nc] == -1:
                        comp[nr, nc] = next_id
                        q.append((nr, nc))
            groups.append(coords)
            next_id += 1
    return comp, groups


def _ray_steps(center: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[int]:
    cr, cc = center
    dr, dc = direction
    steps: list[int] = []
    k = 2
    while _in_bounds(shape, cr + dr * k, cc + dc * k):
        steps.append(k)
        k += 1
    return steps


def _valid_spacing(values: tuple[int, ...], blocked: tuple[int, ...]) -> bool:
    ordered = tuple(sorted(values))
    for a, b in zip(ordered, ordered[1:]):
        if b - a <= 1:
            return False
    for value in ordered:
        for other in blocked:
            if abs(value - other) <= 1:
                return False
    return True


def _pick_spaced(values: list[int], count: int, blocked: tuple[int, ...] = ()) -> list[int] | None:
    if count == 0:
        return []
    unique = sorted(set(values))
    if len(unique) < count:
        return None
    valid = [list(combo) for combo in combinations(unique, count) if _valid_spacing(combo, blocked)]
    if not valid:
        return None
    return random.choice(valid)


def _build_ray_catalog(
    center: tuple[int, int],
    region_ids: np.ndarray,
    anchor_region: int,
) -> dict[tuple[int, int], dict[str, list[int]]]:
    catalog: dict[tuple[int, int], dict[str, list[int]]] = {}
    for direction in DIRS8:
        inside: list[int] = []
        outside: list[int] = []
        for k in _ray_steps(center, direction, region_ids.shape):
            r = center[0] + direction[0] * k
            c = center[1] + direction[1] * k
            region = int(region_ids[r, c])
            if region == -1:
                continue
            if region == anchor_region:
                inside.append(k)
            else:
                outside.append(k)
        catalog[direction] = {"inside": inside, "outside": outside}
    return catalog


def _choose_rays(
    catalog: dict[tuple[int, int], dict[str, list[int]]]
) -> dict[tuple[int, int], dict[str, list[int]]] | None:
    mixed_stack_dirs = [
        direction
        for direction, values in catalog.items()
        if len(values["inside"]) >= 2 and len(values["outside"]) >= 1
    ]
    if not mixed_stack_dirs:
        return None

    primary = random.choice(mixed_stack_dirs)
    chosen: dict[tuple[int, int], dict[str, list[int]]] = {
        direction: {"inside": [], "outside": []} for direction in DIRS8
    }

    primary_inside_count = 3 if len(catalog[primary]["inside"]) >= 3 and random.random() < 0.45 else 2
    primary_inside = _pick_spaced(catalog[primary]["inside"], primary_inside_count)
    if primary_inside is None:
        primary_inside = _pick_spaced(catalog[primary]["inside"], 2)
    if primary_inside is None:
        return None

    primary_outside_goal = 2 if len(catalog[primary]["outside"]) >= 2 and random.random() < 0.4 else 1
    primary_outside = _pick_spaced(
        catalog[primary]["outside"],
        primary_outside_goal,
        blocked=tuple(primary_inside),
    )
    if primary_outside is None:
        primary_outside = _pick_spaced(catalog[primary]["outside"], 1, blocked=tuple(primary_inside))
    if primary_outside is None:
        return None

    chosen[primary]["inside"] = sorted(primary_inside)
    chosen[primary]["outside"] = sorted(primary_outside)

    inside_target = random.randint(3, 5)
    outside_target = random.randint(2, 4)
    inside_total = len(primary_inside)
    outside_total = len(primary_outside)

    extra_inside_dirs = [direction for direction in DIRS8 if direction != primary and catalog[direction]["inside"]]
    random.shuffle(extra_inside_dirs)
    if not extra_inside_dirs:
        return None

    for direction in extra_inside_dirs:
        if inside_total >= inside_target:
            break
        max_count = min(2, len(catalog[direction]["inside"]))
        desired = 2 if max_count == 2 and inside_target - inside_total >= 2 and random.random() < 0.35 else 1
        picked = _pick_spaced(catalog[direction]["inside"], desired)
        if picked is None and desired == 2:
            picked = _pick_spaced(catalog[direction]["inside"], 1)
        if picked is None:
            continue
        chosen[direction]["inside"] = sorted(picked)
        inside_total += len(picked)

    if inside_total < 3:
        return None

    outside_dirs = [direction for direction in DIRS8 if catalog[direction]["outside"]]
    random.shuffle(outside_dirs)
    for direction in outside_dirs:
        if outside_total >= outside_target:
            break
        blocked = tuple(chosen[direction]["inside"] + chosen[direction]["outside"])
        remaining = [k for k in catalog[direction]["outside"] if k not in chosen[direction]["outside"]]
        if not remaining:
            continue
        desired = 1
        if direction != primary and len(remaining) >= 2 and outside_target - outside_total >= 2 and random.random() < 0.25:
            desired = 2
        picked = _pick_spaced(remaining, desired, blocked=blocked)
        if picked is None and desired == 2:
            picked = _pick_spaced(remaining, 1, blocked=blocked)
        if picked is None:
            continue
        chosen[direction]["outside"].extend(sorted(picked))
        chosen[direction]["outside"].sort()
        outside_total += len(picked)

    if outside_total < 2:
        return None
    return chosen


def _gen_vertical_wall(h: int, w: int) -> set[tuple[int, int]]:
    col = random.randint(2, w - 3)
    coords = {(0, col)}
    for r in range(h - 1):
        next_col = max(2, min(w - 3, col + random.choice((-1, 0, 1))))
        step = 1 if next_col > col else -1 if next_col < col else 0
        cur_col = col
        while cur_col != next_col:
            cur_col += step
            coords.add((r, cur_col))
        coords.add((r + 1, next_col))
        col = next_col
    return coords


def _gen_horizontal_wall(h: int, w: int) -> set[tuple[int, int]]:
    row = random.randint(2, h - 3)
    coords = {(row, 0)}
    for c in range(w - 1):
        next_row = max(2, min(h - 3, row + random.choice((-1, 0, 1))))
        step = 1 if next_row > row else -1 if next_row < row else 0
        cur_row = row
        while cur_row != next_row:
            cur_row += step
            coords.add((cur_row, c))
        coords.add((next_row, c + 1))
        row = next_row
    return coords


def _candidate_centers(region_ids: np.ndarray, wall_mask: np.ndarray, region_id: int) -> list[tuple[int, int]]:
    h, w = region_ids.shape
    centers: list[tuple[int, int]] = []
    for cr in range(3, h - 3):
        for cc in range(3, w - 3):
            if not np.all(region_ids[cr - 1 : cr + 2, cc - 1 : cc + 2] == region_id):
                continue
            if wall_mask[cr - 2 : cr + 3, cc - 2 : cc + 3].any():
                continue
            centers.append((cr, cc))
    random.shuffle(centers)
    return centers


def _build_input(
    shape: tuple[int, int],
    wall_mask: np.ndarray,
    center: tuple[int, int],
    chosen: dict[tuple[int, int], dict[str, list[int]]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[wall_mask] = GREY
    for r, c in _anchor_cells(center):
        grid[r, c] = BLUE
    for direction, values in chosen.items():
        for side in ("inside", "outside"):
            for k in values[side]:
                r = center[0] + direction[0] * k
                c = center[1] + direction[1] * k
                grid[r, c] = BLUE
    return grid


def _build_expected(
    shape: tuple[int, int],
    wall_mask: np.ndarray,
    center: tuple[int, int],
    chosen: dict[tuple[int, int], dict[str, list[int]]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[wall_mask] = GREY
    for r, c in _anchor_cells(center):
        out[r, c] = BLUE
    for direction, values in chosen.items():
        keep_count = len(values["inside"])
        for idx in range(keep_count):
            k = idx + 2
            r = center[0] + direction[0] * k
            c = center[1] + direction[1] * k
            out[r, c] = BLUE
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GEN_SIZE_MIN, GEN_SIZE_MAX)
    w = random.randint(GEN_SIZE_MIN, GEN_SIZE_MAX)
    wall_cells = _gen_vertical_wall(h, w) if random.random() < 0.5 else _gen_horizontal_wall(h, w)

    wall_mask = np.zeros((h, w), dtype=bool)
    for r, c in wall_cells:
        wall_mask[r, c] = True

    region_ids, groups = _component_ids(wall_mask)
    if len(groups) != 2:
        return None

    region_order = [0, 1]
    random.shuffle(region_order)
    for anchor_region in region_order:
        for center in _candidate_centers(region_ids, wall_mask, anchor_region):
            catalog = _build_ray_catalog(center, region_ids, anchor_region)
            chosen = _choose_rays(catalog)
            if chosen is None:
                continue
            inp = _build_input((h, w), wall_mask, center, chosen)
            expected = _build_expected((h, w), wall_mask, center, chosen)
            try:
                if np.array_equal(main(inp), expected):
                    return inp
            except AssertionError:
                continue
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Flood from the blue anchor to keep only reachable ray pixels, then compress the survivors against the anchor.
    Input structure:
    - A black background contains one grey wall, one solid blue 3x3 anchor block, and several extra blue single-pixel rays.
    - The grey wall spans between two grid edges and partitions the non-wall cells into two 4-connected regions.
    - Every extra blue pixel is aligned with the anchor center on a row, column, or 45-degree diagonal.

    Rule:
    - Treat grey pixels as the only blocking cells and flood-fill 4-connectedly from the anchor center.
    - Erase every blue ray pixel outside that reachable region, but keep the anchor block itself.
    - For each surviving ray direction, sort the surviving pixels by distance to the anchor and slide them inward so they occupy the consecutive cells starting immediately outside the anchor.
    - Leave the grey wall unchanged and keep all other cells black.

    Key insight:
    - The anchor center is both the flood-fill seed and the compression attractor, so the wall first decides which rays survive and only then determines the packed pattern around the anchor.
    """
    g = np.asarray(grid, dtype=int)
    center = _find_anchor(g)
    passable = g != GREY
    reachable = _reachable_mask(passable, center)

    out = np.zeros_like(g)
    out[g == GREY] = GREY
    for r, c in _anchor_cells(center):
        out[r, c] = BLUE

    kept_by_direction: dict[tuple[int, int], list[int]] = defaultdict(list)
    for raw_r, raw_c in np.argwhere(g == BLUE):
        r = int(raw_r)
        c = int(raw_c)
        if abs(r - center[0]) <= 1 and abs(c - center[1]) <= 1:
            continue
        direction = _direction_from_center(center, (r, c))
        assert direction is not None, "ray pixel must align with the anchor center"
        if reachable[r, c]:
            kept_by_direction[direction].append(_ray_distance(center, (r, c)))

    for direction, distances in kept_by_direction.items():
        distances.sort()
        dr, dc = direction
        for idx, _dist in enumerate(distances):
            step = idx + 2
            rr = center[0] + dr * step
            cc = center[1] + dc * step
            out[rr, cc] = BLUE

    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 2")
