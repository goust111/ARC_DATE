from __future__ import annotations

from collections import deque
import math
import random

import numpy as np


BLACK = 0
GREY = 5
ARC_COLORS = tuple(range(10))
OBJECT_COLORS = tuple(c for c in ARC_COLORS if c not in (BLACK, GREY))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def _components(mask: np.ndarray, *, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = NEIGHBORS4 if int(connectivity) == 4 else NEIGHBORS8
    out: list[np.ndarray] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in dirs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    rs = coords[:, 0]
    cs = coords[:, 1]
    return bool((rs == 0).any() or (rs == h - 1).any() or (cs == 0).any() or (cs == w - 1).any())


def _adjacent_nonblack_colors(grid: np.ndarray, coords: np.ndarray) -> set[int]:
    h, w = grid.shape
    colors: set[int] = set()
    coord_set = {tuple(map(int, rc)) for rc in coords}
    for r, c in coord_set:
        for dr, dc in NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in coord_set:
                color = int(grid[rr, cc])
                if color != BLACK:
                    colors.add(color)
    return colors


def _count_hub_holes(grid: np.ndarray) -> int:
    holes = 0
    for comp in _components(grid == BLACK, connectivity=4):
        if _touches_border(comp, grid.shape):
            continue
        if _adjacent_nonblack_colors(grid, comp) == {GREY}:
            holes += 1
    return holes


def _clockwise_angle(point: np.ndarray, center: np.ndarray) -> float:
    dr = float(point[0] - center[0])
    dc = float(point[1] - center[1])
    angle = math.atan2(dc, -dr)
    if angle < 0:
        angle += 2 * math.pi
    return angle


def _peripheral_regions(grid: np.ndarray) -> list[dict]:
    regions: list[dict] = []
    for color in sorted(set(map(int, grid.flat)) - {BLACK, GREY}):
        comps = _components(grid == color, connectivity=4)
        assert len(comps) == 1, "each peripheral color must form one 4-connected region"
        coords = comps[0]
        regions.append(
            {
                "color": int(color),
                "coords": coords,
                "centroid": coords.astype(float).mean(axis=0),
            }
        )
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the colors of regions around a gray hub by the number of hub holes.

    Input structure:
    - The grid has a black background and one 4-connected gray central hub.
    - The gray hub fully encloses one or more black 4-connected holes.
    - Around the hub are several separated monochrome 4-connected regions, and
      no two peripheral regions use the same color.

    Rule:
    - Keep all black pixels and all gray hub pixels fixed.
    - Count the black holes enclosed only by the gray hub; call this count k.
    - Sort the peripheral regions clockwise by each region centroid relative to
      the gray hub centroid.
    - Keep every peripheral shape in place, but move all region colors k steps
      clockwise around that ordered ring.

    Key insight:
    - The shift amount is not a constant; it is encoded by the topology of the
      central gray hub as its number of enclosed black holes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset(set(ARC_COLORS)), "unexpected color id"

    hubs = _components(g == GREY, connectivity=4)
    assert len(hubs) == 1, "expected exactly one gray hub"
    hub_coords = hubs[0]
    hub_center = hub_coords.astype(float).mean(axis=0)

    hole_count = _count_hub_holes(g)
    assert hole_count >= 1, "expected at least one enclosed hub hole"

    regions = _peripheral_regions(g)
    assert len(regions) >= 3, "expected several peripheral regions"
    assert len({region["color"] for region in regions}) == len(regions)

    for region in regions:
        region["angle"] = _clockwise_angle(region["centroid"], hub_center)
        assert not np.allclose(region["centroid"], hub_center), "peripheral region cannot be at the hub center"

    angles = sorted(region["angle"] for region in regions)
    for a, b in zip(angles, angles[1:] + [angles[0] + 2 * math.pi]):
        assert b - a > 1.0e-7, "clockwise region order must be unambiguous"

    ordered = sorted(regions, key=lambda region: region["angle"])
    n = len(ordered)
    shift = hole_count % n
    colors = [int(region["color"]) for region in ordered]

    out = g.copy()
    for index, region in enumerate(ordered):
        new_color = colors[(index - shift) % n]
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = new_color
    return out


# --------------- generator ---------------

GENERATION_RETRIES = 300
GRID_SIDE_RANGE = (25, 30)
HUB_SIDE_OPTIONS = (8, 9, 10, 11)
HUB_JITTER = 1
REGION_COUNT_RANGE = (4, 7)
HUB_GAP_OPTIONS = (2, 3)
HOLE_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2))
SLOT_DIRS = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)

_GEN_CALLS = 0


def _gen_connected_mask() -> np.ndarray | None:
    for _ in range(40):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        max_area = min(8, h * w)
        target = random.randint(3, max_area)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target:
            continue

        coords = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = _bbox(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        return mask
    return None


def _can_place_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    if r0 < 0 or c0 < 0 or r0 + mh > h or c0 + mw > w:
        return False
    rr, cc = np.nonzero(mask)
    for dr, dc in ((0, 0),) + NEIGHBORS8:
        rows = r0 + rr + dr
        cols = c0 + cc + dc
        valid = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
        if not bool(np.all(valid)):
            return False
        if bool(np.any(grid[rows, cols] != BLACK)):
            return False
    return True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> np.ndarray:
    rr, cc = np.nonzero(mask)
    coords = np.column_stack([r0 + rr, c0 + cc]).astype(int)
    grid[coords[:, 0], coords[:, 1]] = int(color)
    return coords


def _place_holes(grid: np.ndarray, hub_box: tuple[int, int, int, int], count: int) -> bool:
    r0, c0, r1, c1 = hub_box
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    blocked = np.zeros((inner_h, inner_w), dtype=bool)

    for _ in range(count):
        placed = False
        shapes = list(HOLE_SHAPES)
        random.shuffle(shapes)
        for hh, hw in shapes:
            if hh > inner_h or hw > inner_w:
                continue
            positions = [
                (rr, cc)
                for rr in range(inner_h - hh + 1)
                for cc in range(inner_w - hw + 1)
            ]
            random.shuffle(positions)
            for rr, cc in positions:
                er0 = max(0, rr - 1)
                ec0 = max(0, cc - 1)
                er1 = min(inner_h, rr + hh + 1)
                ec1 = min(inner_w, cc + hw + 1)
                if blocked[er0:er1, ec0:ec1].any():
                    continue
                blocked[rr : rr + hh, cc : cc + hw] = True
                grid[r0 + 1 + rr : r0 + 1 + rr + hh, c0 + 1 + cc : c0 + 1 + cc + hw] = BLACK
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _slot_origin(
    slot: int,
    mask: np.ndarray,
    hub_box: tuple[int, int, int, int],
    center: tuple[float, float],
    gap: int,
) -> tuple[int, int]:
    r0, c0, r1, c1 = hub_box
    mh, mw = mask.shape
    cr, cc = center
    dr, dc = SLOT_DIRS[slot]

    if dr < 0:
        base_r = r0 - gap - mh
    elif dr > 0:
        base_r = r1 + gap + 1
    else:
        base_r = int(round(cr - (mh - 1) / 2))

    if dc < 0:
        base_c = c0 - gap - mw
    elif dc > 0:
        base_c = c1 + gap + 1
    else:
        base_c = int(round(cc - (mw - 1) / 2))

    return int(base_r), int(base_c)


def _place_regions(
    grid: np.ndarray,
    hub_box: tuple[int, int, int, int],
    hub_center: tuple[float, float],
    count: int,
) -> list[dict] | None:
    selected_slots = sorted(random.sample(range(len(SLOT_DIRS)), count))
    colors = random.sample(list(OBJECT_COLORS), count)
    random.shuffle(colors)
    regions: list[dict] = []

    for slot, color in zip(selected_slots, colors):
        mask = _gen_connected_mask()
        if mask is None:
            return None
        gap = random.choice(HUB_GAP_OPTIONS)
        base_r, base_c = _slot_origin(slot, mask, hub_box, hub_center, gap)

        offsets = [(0, 0)]
        offsets.extend((random.randint(-1, 1), random.randint(-1, 1)) for _ in range(20))
        placed = False
        for jr, jc in offsets:
            r = base_r + jr
            c = base_c + jc
            if not _can_place_mask(grid, mask, r, c):
                continue
            coords = _paint_mask(grid, mask, r, c, color)
            regions.append(
                {
                    "slot": int(slot),
                    "color": int(color),
                    "coords": coords,
                    "centroid": coords.astype(float).mean(axis=0),
                }
            )
            placed = True
            break
        if not placed:
            return None
    return regions


def _render_expected_from_regions(grid: np.ndarray, regions: list[dict], shift: int) -> np.ndarray:
    ordered = sorted(regions, key=lambda region: int(region["slot"]))
    n = len(ordered)
    colors = [int(region["color"]) for region in ordered]
    out = grid.copy()
    for i, region in enumerate(ordered):
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = colors[(i - shift) % n]
    return out


def _gen_semantic_accept(grid: np.ndarray, regions: list[dict], hole_count: int) -> bool:
    if int(_count_hub_holes(grid)) != int(hole_count):
        return False
    if len(_peripheral_regions(grid)) != len(regions):
        return False
    if hole_count % len(regions) == 0:
        return False

    expected = _render_expected_from_regions(grid, regions, hole_count)
    if np.array_equal(expected, grid):
        return False
    if hole_count != 1:
        one_step = _render_expected_from_regions(grid, regions, 1)
        if np.array_equal(expected, one_step):
            return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt(hole_count: int) -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)

    hub_h = random.choice(HUB_SIDE_OPTIONS)
    hub_w = random.choice(HUB_SIDE_OPTIONS)
    hr = h // 2 - hub_h // 2 + random.randint(-HUB_JITTER, HUB_JITTER)
    hc = w // 2 - hub_w // 2 + random.randint(-HUB_JITTER, HUB_JITTER)
    if hr < 6 or hc < 6 or hr + hub_h > h - 6 or hc + hub_w > w - 6:
        return None

    hub_box = (hr, hc, hr + hub_h - 1, hc + hub_w - 1)
    grid[hr : hr + hub_h, hc : hc + hub_w] = GREY
    if not _place_holes(grid, hub_box, hole_count):
        return None

    hub_coords = np.argwhere(grid == GREY)
    hub_center = tuple(map(float, hub_coords.astype(float).mean(axis=0)))
    min_regions = max(REGION_COUNT_RANGE[0], hole_count + 2)
    region_count = random.randint(min_regions, REGION_COUNT_RANGE[1])
    regions = _place_regions(grid, hub_box, hub_center, region_count)
    if regions is None:
        return None

    if not _gen_semantic_accept(grid, regions, hole_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALLS
    preferred_holes = (_GEN_CALLS % 3) + 1
    hole_options = [preferred_holes, *random.sample([1, 2, 3], 3)]
    for hole_count in hole_options:
        for _ in range(GENERATION_RETRIES // len(hole_options)):
            grid = _gen_attempt(int(hole_count))
            if grid is not None:
                _GEN_CALLS += 1
                return grid
    raise RuntimeError("failed to generate a valid fable/3/041/1 input")
