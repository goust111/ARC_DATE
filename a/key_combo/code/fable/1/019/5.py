from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GRAY = 5

_GEN_CALL_COUNT = 0


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if int(connectivity) == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if dr or dc
        ]

    out: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        q = deque([(rr, cc)])
        pts: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            pts.append((cr, cc))
            for dr, dc in steps:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(np.asarray(pts, dtype=int))
    return out


def _flood_from_border(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return outside


def _enclosed_mask_for_wall(
    wall_coords: np.ndarray, box: tuple[int, int, int, int]
) -> np.ndarray:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    wall = np.zeros((h, w), dtype=bool)
    local = wall_coords - np.asarray([r0, c0], dtype=int)
    wall[local[:, 0], local[:, 1]] = True

    passable = ~wall
    outside = _flood_from_border(passable)
    return passable & ~outside


def main(grid: np.ndarray) -> np.ndarray:
    """
    Invert red and black cells by the parity of true closed gray-ring containment, while ignoring broken rings.

    Input structure:
    - The background is black, with scattered red single pixels and one-cell-thick gray rectangular ring-like objects.
    - Gray rings may be nested up to several layers deep.
    - Some gray rings contain a one-cell side gap, so they look almost closed but do not actually enclose space.

    Rule:
    - Keep every gray pixel unchanged.
    - For each 4-connected gray component, test whether its bbox contains a region unreachable from the bbox border without crossing that gray component; only such components count as closed rings.
    - For every non-gray cell, count how many closed gray rings strictly contain it. If that count is odd, swap black and red at that cell; if even, leave it unchanged.

    Key insight:
    - The transformation is driven by nesting-depth parity, but only genuinely closed gray loops contribute depth; a one-cell break cancels enclosure completely.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    gray_mask = g == GRAY
    depth = np.zeros(g.shape, dtype=int)

    for coords in components_mask(gray_mask, connectivity=4):
        box = bbox_coords(coords)
        enclosed = _enclosed_mask_for_wall(coords, box)
        if not enclosed.any():
            continue
        r0, c0, r1, c1 = box
        view = depth[r0 : r1 + 1, c0 : c1 + 1]
        view[enclosed] += 1

    odd = (depth % 2 == 1) & ~gray_mask
    out = g.copy()
    out[odd & (g == BLACK)] = RED
    out[odd & (g == RED)] = BLACK
    return out


def _make_gap(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        return r0, random.randint(c0 + 1, c1 - 1)
    if side == "bottom":
        return r1, random.randint(c0 + 1, c1 - 1)
    if side == "left":
        return random.randint(r0 + 1, r1 - 1), c0
    return random.randint(r0 + 1, r1 - 1), c1


def _build_group(pattern: list[bool]) -> dict:
    depth = len(pattern)
    if depth == 1:
        outer_h = random.randint(7, 11)
        outer_w = random.randint(7, 11)
        offsets = [0]
    else:
        inner_h = random.randint(5, 7)
        inner_w = random.randint(5, 7)
        pads = [random.randint(2, 3) for _ in range(depth - 1)]
        total_pad = sum(pads)
        outer_h = inner_h + 2 * total_pad
        outer_w = inner_w + 2 * total_pad
        offsets = [0]
        acc = 0
        for pad in pads:
            acc += pad
            offsets.append(acc)

    rings = []
    for closed, off in zip(pattern, offsets):
        r0, c0 = off, off
        r1, c1 = outer_h - 1 - off, outer_w - 1 - off
        rings.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "closed": bool(closed),
                "gap": None if closed else _make_gap(r0, c0, r1, c1),
            }
        )
    return {"height": outer_h, "width": outer_w, "rings": rings}


def _random_span(low: int, high: int) -> int:
    if low >= high:
        return int(low)
    return random.randint(int(low), int(high))


def _arrange_groups(groups: list[dict]) -> tuple[int, int, list[tuple[dict, int, int]]] | None:
    order = groups[:]
    random.shuffle(order)
    gap = random.randint(3, 5)

    horiz_w = sum(group["width"] for group in order) + gap * (len(order) - 1)
    horiz_h = max(group["height"] for group in order)
    vert_h = sum(group["height"] for group in order) + gap * (len(order) - 1)
    vert_w = max(group["width"] for group in order)

    options = []
    if horiz_w + 4 <= 30 and horiz_h + 4 <= 30:
        options.append("H")
    if vert_w + 4 <= 30 and vert_h + 4 <= 30:
        options.append("V")
    if not options:
        return None

    orient = random.choice(options)
    placements: list[tuple[dict, int, int]] = []

    if orient == "H":
        h = _random_span(max(horiz_h + 4, 18), 30)
        w = _random_span(max(horiz_w + 4, 18), 30)
        c = random.randint(1, w - horiz_w - 1)
        for group in order:
            r = random.randint(1, h - group["height"] - 1)
            placements.append((group, r, c))
            c += group["width"] + gap
        return h, w, placements

    h = _random_span(max(vert_h + 4, 18), 30)
    w = _random_span(max(vert_w + 4, 18), 30)
    r = random.randint(1, h - vert_h - 1)
    for group in order:
        c = random.randint(1, w - group["width"] - 1)
        placements.append((group, r, c))
        r += group["height"] + gap
    return h, w, placements


def _draw_ring(grid: np.ndarray, spec: dict) -> None:
    r0, c0, r1, c1 = spec["r0"], spec["c0"], spec["r1"], spec["c1"]
    grid[r0, c0 : c1 + 1] = GRAY
    grid[r1, c0 : c1 + 1] = GRAY
    grid[r0 : r1 + 1, c0] = GRAY
    grid[r0 : r1 + 1, c1] = GRAY
    if spec["gap"] is not None:
        gr, gc = spec["gap"]
        grid[gr, gc] = BLACK


def _paint_groups(
    shape: tuple[int, int], placements: list[tuple[dict, int, int]]
) -> tuple[np.ndarray, list[dict]]:
    grid = np.full(shape, BLACK, dtype=int)
    abs_specs: list[dict] = []

    for group, base_r, base_c in placements:
        for rel_spec in group["rings"]:
            spec = {
                "r0": base_r + rel_spec["r0"],
                "c0": base_c + rel_spec["c0"],
                "r1": base_r + rel_spec["r1"],
                "c1": base_c + rel_spec["c1"],
                "closed": rel_spec["closed"],
                "gap": None,
            }
            if rel_spec["gap"] is not None:
                gap_r, gap_c = rel_spec["gap"]
                spec["gap"] = (base_r + gap_r, base_c + gap_c)
            _draw_ring(grid, spec)
            abs_specs.append(spec)
    return grid, abs_specs


def _depth_from_specs(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    depth = np.zeros(shape, dtype=int)
    for spec in specs:
        if not spec["closed"]:
            continue
        if spec["r1"] - spec["r0"] < 2 or spec["c1"] - spec["c0"] < 2:
            continue
        depth[spec["r0"] + 1 : spec["r1"], spec["c0"] + 1 : spec["c1"]] += 1
    return depth


def _expected_output_from_specs(grid: np.ndarray, specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    depth = _depth_from_specs(grid.shape, specs)
    out = grid.copy()
    odd = (depth % 2 == 1) & (grid != GRAY)
    out[odd & (grid == BLACK)] = RED
    out[odd & (grid == RED)] = BLACK
    return out, depth


def _strict_interior_mask(shape: tuple[int, int], spec: dict) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if spec["r1"] - spec["r0"] < 2 or spec["c1"] - spec["c0"] < 2:
        return mask
    mask[spec["r0"] + 1 : spec["r1"], spec["c0"] + 1 : spec["c1"]] = True
    return mask


def _red_can_fit(red_mask: np.ndarray, r: int, c: int) -> bool:
    h, w = red_mask.shape
    for nr in range(max(0, r - 1), min(h, r + 2)):
        for nc in range(max(0, c - 1), min(w, c + 2)):
            if red_mask[nr, nc]:
                return False
    return True


def _place_red(grid: np.ndarray, red_mask: np.ndarray, candidate_mask: np.ndarray) -> bool:
    candidates = np.argwhere(candidate_mask & (grid == BLACK))
    if candidates.size == 0:
        return False
    order = list(range(len(candidates)))
    random.shuffle(order)
    for idx in order:
        r, c = map(int, candidates[idx])
        if _red_can_fit(red_mask, r, c):
            grid[r, c] = RED
            red_mask[r, c] = True
            return True
    return False


def _scenario_patterns(scenario: str) -> list[list[bool]]:
    if scenario == "closed_vs_broken":
        return [[True], [False]]
    if scenario == "broken_middle":
        return [[True, False, True]]
    return [[True, True, True], [False]]


def _add_anchor_reds(grid: np.ndarray, specs: list[dict], depth: np.ndarray, scenario: str) -> bool:
    red_mask = np.zeros(grid.shape, dtype=bool)
    non_gray = grid != GRAY
    broken_specs = [spec for spec in specs if not spec["closed"]]

    required: list[np.ndarray] = []
    if scenario == "closed_vs_broken":
        required.append(non_gray & (depth == 1))
        required.append(_strict_interior_mask(grid.shape, broken_specs[0]) & non_gray & (depth == 0))
        required.append(non_gray & (depth == 0))
    elif scenario == "broken_middle":
        required.append(non_gray & (depth == 1))
        required.append(non_gray & (depth == 2))
        required.append(_strict_interior_mask(grid.shape, broken_specs[0]) & non_gray & (depth == 1))
        required.append(non_gray & (depth == 0))
    else:
        required.append(non_gray & (depth == 1))
        required.append(non_gray & (depth == 2))
        required.append(non_gray & (depth == 3))
        required.append(_strict_interior_mask(grid.shape, broken_specs[0]) & non_gray & (depth == 0))
        required.append(non_gray & (depth == 0))

    for mask in required:
        if not _place_red(grid, red_mask, mask):
            return False

    extra_target = random.randint(2, 6)
    weighted_masks = [
        non_gray & (depth == 0),
        non_gray & (depth == 1),
        non_gray & (depth == 2),
        non_gray & (depth == 3),
    ]
    for _ in range(extra_target):
        random.shuffle(weighted_masks)
        placed = False
        for mask in weighted_masks:
            if _place_red(grid, red_mask, mask):
                placed = True
                break
        if not placed:
            break
    return True


def _red_pixels_are_isolated(grid: np.ndarray) -> bool:
    h, w = grid.shape
    for r, c in np.argwhere(grid == RED):
        rr, cc = int(r), int(c)
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                if (nr != rr or nc != cc) and grid[nr, nc] == RED:
                    return False
    return True


def _validate_semantics(grid: np.ndarray, specs: list[dict], scenario: str) -> bool:
    expected, depth = _expected_output_from_specs(grid, specs)
    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(grid, expected):
        return False
    if not _red_pixels_are_isolated(grid):
        return False
    if not np.any((grid == BLACK) & ((depth % 2) == 1)):
        return False

    broken_specs = [spec for spec in specs if not spec["closed"]]
    if not broken_specs or not any(spec["closed"] for spec in specs):
        return False

    if scenario == "closed_vs_broken":
        if int(depth.max()) != 1:
            return False
        broken_mask = _strict_interior_mask(grid.shape, broken_specs[0]) & (depth == 0)
        if not np.any((grid == RED) & broken_mask):
            return False
        if not np.any((grid == RED) & (depth == 1)):
            return False
    elif scenario == "broken_middle":
        if int(depth.max()) != 2:
            return False
        broken_mask = _strict_interior_mask(grid.shape, broken_specs[0]) & (depth == 1)
        if not np.any((grid == RED) & broken_mask):
            return False
        if not np.any((grid == RED) & (depth == 2)):
            return False
    else:
        if int(depth.max()) != 3:
            return False
        broken_mask = _strict_interior_mask(grid.shape, broken_specs[0]) & (depth == 0)
        if not np.any((grid == RED) & broken_mask):
            return False
        if not np.any((grid == RED) & (depth == 2)):
            return False
        if not np.any((grid == RED) & (depth == 3)):
            return False
    return True


def _attempt_generate(scenario: str) -> np.ndarray | None:
    groups = [_build_group(pattern) for pattern in _scenario_patterns(scenario)]
    arranged = _arrange_groups(groups)
    if arranged is None:
        return None

    h, w, placements = arranged
    grid, specs = _paint_groups((h, w), placements)
    _, depth = _expected_output_from_specs(grid, specs)

    if not _add_anchor_reds(grid, specs, depth, scenario):
        return None
    if not _validate_semantics(grid, specs, scenario):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    scenarios = ["closed_vs_broken", "broken_middle", "triple_plus_broken"]
    if _GEN_CALL_COUNT < len(scenarios):
        scenario = scenarios[_GEN_CALL_COUNT]
    else:
        scenario = random.choice(scenarios)
    _GEN_CALL_COUNT += 1

    for _ in range(300):
        grid = _attempt_generate(scenario)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample for scenario {scenario}")
