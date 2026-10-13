from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
WALL = 1
SEED_COLORS = tuple(range(2, 10))

DIRECTIONS = ("up", "down", "left", "right")
STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

REGION_COUNT_RANGE = (3, 6)
GRID_MAX = 30
RECT_H_RANGE = (5, 8)
RECT_W_RANGE = (5, 8)
PLACE_TRIES = 120
GEN_TRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for dr, dc in STEPS.values():
                nr = cr + dr
                nc = cc + dc
                if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows, cols = zip(*coords)
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros(sub.shape, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(sub == WALL, expected))


def _contains_bbox(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _primary_direction(bbox: tuple[int, int, int, int], seed: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    sr, sc = seed
    dr2 = 2 * sr - (r0 + r1)
    dc2 = 2 * sc - (c0 + c1)
    assert dr2 != 0 or dc2 != 0, "seed must not sit exactly at the center"
    if abs(dr2) > abs(dc2):
        return "down" if dr2 > 0 else "up"
    return "right" if dc2 > 0 else "left"


def _gap_and_step(
    bbox: tuple[int, int, int, int], seed: tuple[int, int], direction: str
) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    sr, sc = seed
    if direction == "up":
        return (r0, sc), STEPS["up"]
    if direction == "down":
        return (r1, sc), STEPS["down"]
    if direction == "left":
        return (sr, c0), STEPS["left"]
    return (sr, c1), STEPS["right"]


def _parse_regions(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    regions: list[dict[str, object]] = []
    seen_seed_cells = 0
    for comp in _components4(g == WALL):
        box = _bbox(comp)
        assert _is_rect_frame(g, box), "each blue component must be a one-pixel-thick rectangle"
        r0, c0, r1, c1 = box
        assert 0 < r0 < r1 < h - 1 and 0 < c0 < c1 < w - 1, "rectangles must be enclosed"

        inside = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_pts = np.argwhere((inside != BG) & (inside != WALL))
        assert seed_pts.shape[0] == 1, "each rectangle must contain exactly one seed pixel"
        sr = int(seed_pts[0, 0]) + r0 + 1
        sc = int(seed_pts[0, 1]) + c0 + 1
        seen_seed_cells += 1

        regions.append(
            {
                "bbox": box,
                "seed": (sr, sc),
                "color": int(g[sr, sc]),
            }
        )

    assert regions, "expected at least one rectangular region"
    extra_seed_cells = int(np.count_nonzero((g != BG) & (g != WALL)))
    assert seen_seed_cells == extra_seed_cells, "all non-blue foreground cells must be seeds"

    bboxes = [region["bbox"] for region in regions]
    for i, box_a in enumerate(bboxes):
        for j, box_b in enumerate(bboxes):
            if i != j:
                assert not _contains_bbox(box_a, box_b), "rectangles must not be nested"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open one wall gap per rectangular enclosure based on the seed's dominant offset
    from the center, then cast the seed color outward from that gap.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    blue_mask = g == WALL

    for region in _parse_regions(g):
        bbox = region["bbox"]
        seed = region["seed"]
        color = int(region["color"])
        direction = _primary_direction(bbox, seed)
        gap, step = _gap_and_step(bbox, seed, direction)

        out[gap] = BG
        r = gap[0] + step[0]
        c = gap[1] + step[1]
        while _in_bounds(g.shape, r, c) and not blue_mask[r, c]:
            out[r, c] = color
            r += step[0]
            c += step[1]

    return out


def _balanced_direction_pool(count: int) -> list[str]:
    pool = list(DIRECTIONS)
    random.shuffle(pool)
    while len(pool) < count:
        extra = list(DIRECTIONS)
        random.shuffle(extra)
        pool.extend(extra)
    return pool[:count]


def _expanded_bbox(
    bbox: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(shape[0] - 1, r1 + pad),
        min(shape[1] - 1, c1 + pad),
    )


def _bbox_is_free(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = _expanded_bbox(bbox, blocked.shape, pad=pad)
    return not bool(np.any(blocked[r0 : r1 + 1, c0 : c1 + 1]))


def _mark_bbox(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    r0, c0, r1, c1 = _expanded_bbox(bbox, blocked.shape, pad=pad)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _mark_cells(blocked: np.ndarray, cells: Iterable[tuple[int, int]]) -> None:
    for r, c in cells:
        blocked[r, c] = True


def _candidate_seeds(bbox: tuple[int, int, int, int], direction: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    center_r2 = r0 + r1
    center_c2 = c0 + c1
    seeds = []
    for sr in range(r0 + 1, r1):
        for sc in range(c0 + 1, c1):
            if 2 * sr == center_r2 and 2 * sc == center_c2:
                continue
            if _primary_direction(bbox, (sr, sc)) == direction:
                seeds.append((sr, sc))
    return seeds


def _ray_to_edge(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    seed: tuple[int, int],
    direction: str,
    blocked: np.ndarray,
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    gap, step = _gap_and_step(bbox, seed, direction)
    ray: list[tuple[int, int]] = []
    r = gap[0] + step[0]
    c = gap[1] + step[1]
    while _in_bounds(shape, r, c):
        if blocked[r, c]:
            return None
        ray.append((r, c))
        r += step[0]
        c += step[1]
    return (gap, ray) if ray else None


def _random_bbox(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h = random.randint(*RECT_H_RANGE)
    w = random.randint(*RECT_W_RANGE)
    r0 = random.randint(1, shape[0] - h - 1)
    c0 = random.randint(1, shape[1] - w - 1)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _place_single(
    shape: tuple[int, int], blocked: np.ndarray, direction: str, color: int
) -> dict[str, object] | None:
    for _ in range(PLACE_TRIES):
        bbox = _random_bbox(shape)
        if not _bbox_is_free(blocked, bbox, pad=1):
            continue
        seeds = _candidate_seeds(bbox, direction)
        random.shuffle(seeds)
        for seed in seeds:
            gap_and_ray = _ray_to_edge(shape, bbox, seed, direction, blocked)
            if gap_and_ray is None:
                continue
            gap, ray = gap_and_ray
            region = {
                "bbox": bbox,
                "seed": seed,
                "color": color,
                "direction": direction,
                "gap": gap,
                "ray": ray,
                "stop": "edge",
            }
            _mark_bbox(blocked, bbox, pad=1)
            _mark_cells(blocked, ray)
            return region
    return None


def _sample_blocker_bbox(
    shape: tuple[int, int],
    blocked: np.ndarray,
    emitter_bbox: tuple[int, int, int, int],
    seed: tuple[int, int],
    direction: str,
) -> list[tuple[tuple[int, int, int, int], list[tuple[int, int]]]]:
    h, w = shape
    sr, sc = seed
    er0, ec0, er1, ec1 = emitter_bbox
    options: list[tuple[tuple[int, int, int, int], list[tuple[int, int]]]] = []

    for _ in range(PLACE_TRIES):
        bh = random.randint(*RECT_H_RANGE)
        bw = random.randint(*RECT_W_RANGE)
        if direction == "right":
            min_r0 = max(1, sr - bh + 2)
            max_r0 = min(h - bh - 1, sr - 1)
            min_c0 = ec1 + 3
            max_c0 = w - bw - 1
            if min_r0 > max_r0 or min_c0 > max_c0:
                continue
            br0 = random.randint(min_r0, max_r0)
            bc0 = random.randint(min_c0, max_c0)
            bbox = (br0, bc0, br0 + bh - 1, bc0 + bw - 1)
            ray = [(sr, c) for c in range(ec1 + 1, bc0)]
        elif direction == "left":
            min_r0 = max(1, sr - bh + 2)
            max_r0 = min(h - bh - 1, sr - 1)
            min_c0 = 1
            max_c0 = ec0 - bw - 2
            if min_r0 > max_r0 or min_c0 > max_c0:
                continue
            br0 = random.randint(min_r0, max_r0)
            bc0 = random.randint(min_c0, max_c0)
            bbox = (br0, bc0, br0 + bh - 1, bc0 + bw - 1)
            ray = [(sr, c) for c in range(ec0 - 1, bbox[3], -1)]
        elif direction == "down":
            min_c0 = max(1, sc - bw + 2)
            max_c0 = min(w - bw - 1, sc - 1)
            min_r0 = er1 + 3
            max_r0 = h - bh - 1
            if min_c0 > max_c0 or min_r0 > max_r0:
                continue
            br0 = random.randint(min_r0, max_r0)
            bc0 = random.randint(min_c0, max_c0)
            bbox = (br0, bc0, br0 + bh - 1, bc0 + bw - 1)
            ray = [(r, sc) for r in range(er1 + 1, br0)]
        else:
            min_c0 = max(1, sc - bw + 2)
            max_c0 = min(w - bw - 1, sc - 1)
            min_r0 = 1
            max_r0 = er0 - bh - 2
            if min_c0 > max_c0 or min_r0 > max_r0:
                continue
            br0 = random.randint(min_r0, max_r0)
            bc0 = random.randint(min_c0, max_c0)
            bbox = (br0, bc0, br0 + bh - 1, bc0 + bw - 1)
            ray = [(r, sc) for r in range(er0 - 1, bbox[2], -1)]

        if not ray or not _bbox_is_free(blocked, bbox, pad=1):
            continue
        if any(blocked[r, c] for r, c in ray):
            continue
        options.append((bbox, ray))

    random.shuffle(options)
    return options


def _direction_search_order(preferred: str, forbidden: str) -> list[str]:
    order = []
    if preferred != forbidden:
        order.append(preferred)
    rest = [d for d in DIRECTIONS if d not in order and d != forbidden]
    random.shuffle(rest)
    order.extend(rest)
    return order


def _place_pair(
    shape: tuple[int, int],
    blocked: np.ndarray,
    emitter_dir: str,
    blocker_pref_dir: str,
    emitter_color: int,
    blocker_color: int,
) -> tuple[dict[str, object], dict[str, object]] | None:
    for _ in range(PLACE_TRIES):
        emitter_bbox = _random_bbox(shape)
        if not _bbox_is_free(blocked, emitter_bbox, pad=1):
            continue

        seeds = _candidate_seeds(emitter_bbox, emitter_dir)
        random.shuffle(seeds)
        for emitter_seed in seeds:
            blocker_choices = _sample_blocker_bbox(
                shape, blocked, emitter_bbox, emitter_seed, emitter_dir
            )
            for blocker_bbox, emitter_ray in blocker_choices:
                blocker_blocked = blocked.copy()
                _mark_bbox(blocker_blocked, emitter_bbox, pad=1)
                _mark_cells(blocker_blocked, emitter_ray)

                for blocker_dir in _direction_search_order(
                    blocker_pref_dir, OPPOSITE[emitter_dir]
                ):
                    blocker_seeds = _candidate_seeds(blocker_bbox, blocker_dir)
                    random.shuffle(blocker_seeds)
                    for blocker_seed in blocker_seeds:
                        gap_and_ray = _ray_to_edge(
                            shape, blocker_bbox, blocker_seed, blocker_dir, blocker_blocked
                        )
                        if gap_and_ray is None:
                            continue
                        blocker_gap, blocker_ray = gap_and_ray
                        emitter_gap, _ = _gap_and_step(emitter_bbox, emitter_seed, emitter_dir)
                        emitter_region = {
                            "bbox": emitter_bbox,
                            "seed": emitter_seed,
                            "color": emitter_color,
                            "direction": emitter_dir,
                            "gap": emitter_gap,
                            "ray": emitter_ray,
                            "stop": "wall",
                        }
                        blocker_region = {
                            "bbox": blocker_bbox,
                            "seed": blocker_seed,
                            "color": blocker_color,
                            "direction": blocker_dir,
                            "gap": blocker_gap,
                            "ray": blocker_ray,
                            "stop": "edge",
                        }
                        _mark_bbox(blocked, emitter_bbox, pad=1)
                        _mark_bbox(blocked, blocker_bbox, pad=1)
                        _mark_cells(blocked, emitter_ray)
                        _mark_cells(blocked, blocker_ray)
                        return emitter_region, blocker_region
    return None


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL


def _render_input(shape: tuple[int, int], regions: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for region in regions:
        _draw_frame(grid, region["bbox"])
        sr, sc = region["seed"]
        grid[sr, sc] = int(region["color"])
    return grid


def _render_expected(inp: np.ndarray, regions: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for region in regions:
        out[region["gap"]] = BG
        for r, c in region["ray"]:
            out[r, c] = int(region["color"])
    return out


def _choose_pair_count(region_count: int) -> int:
    pair_count = 0
    if region_count >= 4 and random.random() < 0.7:
        pair_count = 1
    if region_count >= 6 and random.random() < 0.35:
        pair_count = 2
    return min(pair_count, region_count // 2)


def _gen_attempt() -> np.ndarray | None:
    region_count = random.randint(*REGION_COUNT_RANGE)
    min_side = 14 + 2 * region_count
    if min_side > GRID_MAX:
        return None
    shape = (
        random.randint(min_side, GRID_MAX),
        random.randint(min_side, GRID_MAX),
    )

    colors = random.sample(SEED_COLORS, region_count)
    directions = _balanced_direction_pool(region_count)
    pair_count = _choose_pair_count(region_count)
    actions = ["pair"] * pair_count + ["single"] * (region_count - 2 * pair_count)
    random.shuffle(actions)

    blocked = np.zeros(shape, dtype=bool)
    regions: list[dict[str, object]] = []
    color_idx = 0
    dir_idx = 0

    for action in actions:
        if action == "single":
            region = _place_single(
                shape,
                blocked,
                directions[dir_idx],
                colors[color_idx],
            )
            if region is None:
                return None
            regions.append(region)
            color_idx += 1
            dir_idx += 1
        else:
            pair = _place_pair(
                shape,
                blocked,
                directions[dir_idx],
                directions[dir_idx + 1],
                colors[color_idx],
                colors[color_idx + 1],
            )
            if pair is None:
                return None
            regions.extend(pair)
            color_idx += 2
            dir_idx += 2

    inp = _render_input(shape, regions)
    expected = _render_expected(inp, regions)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    stop_modes = {region["stop"] for region in regions}
    if pair_count == 0 and "wall" in stop_modes:
        return None
    if pair_count > 0 and "wall" not in stop_modes:
        return None
    if len({_primary_direction(region["bbox"], region["seed"]) for region in regions}) < min(4, region_count):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
