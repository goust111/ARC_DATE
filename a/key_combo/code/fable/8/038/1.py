from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components


BG = int(Color.BLACK)
WATER = int(Color.BLUE)
OIL = int(Color.YELLOW)
WALL = int(Color.GREY)

_MAX_GEN_ATTEMPTS = 500
_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _parse_container(grid: np.ndarray, comp: dict) -> dict:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    assert c1 - c0 >= 2, "container must have at least one interior column"
    assert np.all(grid[r1, c0 : c1 + 1] == WALL), "container floor must span the bbox"

    left_rows = [r for r in range(r0, r1 + 1) if int(grid[r, c0]) == WALL]
    right_rows = [r for r in range(r0, r1 + 1) if int(grid[r, c1]) == WALL]
    assert left_rows and right_rows, "container must have two walls"
    left_top = left_rows[0]
    right_top = right_rows[0]
    assert left_rows == list(range(left_top, r1 + 1)), "left wall must be contiguous"
    assert right_rows == list(range(right_top, r1 + 1)), "right wall must be contiguous"
    assert left_top != right_top, "container walls must have different heights"

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros_like(patch, dtype=bool)
    expected[r1 - r0, :] = True
    expected[left_top - r0 :, 0] = True
    expected[right_top - r0 :, -1] = True
    assert np.array_equal(patch == WALL, expected), "container must be an open-top basin"

    stable_top = max(left_top, right_top)
    fill_cells = [(r, c) for r in range(r1 - 1, stable_top - 1, -1) for c in range(c0 + 1, c1)]
    assert fill_cells, "container must have positive capacity"

    return {
        "bbox": (r0, c0, r1, c1),
        "left_top": left_top,
        "right_top": right_top,
        "high_top": min(left_top, right_top),
        "stable_top": stable_top,
        "capacity": len(fill_cells),
        "fill_cells": fill_cells,
        "interior_cols": list(range(c0 + 1, c1)),
        "overflow_side": -1 if left_top > right_top else 1,
    }


def _container_targets(containers: list[dict]) -> list[int | None]:
    targets: list[int | None] = []
    n = len(containers)
    for idx, container in enumerate(containers):
        if container["overflow_side"] < 0:
            targets.append(idx - 1 if idx > 0 else None)
        else:
            targets.append(idx + 1 if idx + 1 < n else None)
    return targets


def _settle_amounts(total_water: int, total_oil: int, capacity: int) -> tuple[tuple[int, int], tuple[int, int]]:
    keep_water = min(int(total_water), int(capacity))
    keep_oil = min(int(total_oil), max(0, int(capacity) - keep_water))
    spill_water = int(total_water) - keep_water
    spill_oil = int(total_oil) - keep_oil
    return (keep_water, keep_oil), (spill_water, spill_oil)


def _simulate(containers: list[dict], local_counts: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[dict]]:
    n = len(containers)
    targets = _container_targets(containers)
    indegree = [0] * n
    for target in targets:
        if target is not None:
            indegree[target] += 1

    incoming = [[0, 0] for _ in range(n)]
    settled = [(0, 0)] * n
    meta: list[dict] = [None] * n  # type: ignore[assignment]
    queue = [idx for idx, deg in enumerate(indegree) if deg == 0]
    visited = 0

    while queue:
        idx = queue.pop(0)
        visited += 1
        own_water, own_oil = map(int, local_counts[idx])
        inc_water, inc_oil = incoming[idx]
        total_water = own_water + inc_water
        total_oil = own_oil + inc_oil
        kept, spilled = _settle_amounts(total_water, total_oil, containers[idx]["capacity"])
        settled[idx] = kept
        meta[idx] = {
            "own": (own_water, own_oil),
            "incoming": (inc_water, inc_oil),
            "total": (total_water, total_oil),
            "kept": kept,
            "spilled": spilled,
            "target": targets[idx],
        }
        if spilled != (0, 0):
            target = targets[idx]
            assert target is not None, "overflow cannot leave the puzzle boundary"
            incoming[target][0] += spilled[0]
            incoming[target][1] += spilled[1]
        target = targets[idx]
        if target is not None:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    assert visited == n, "overflow graph must be acyclic"
    return settled, meta


def _collect_local_counts(grid: np.ndarray, containers: list[dict]) -> list[tuple[int, int]]:
    col_to_container: dict[int, int] = {}
    for idx, container in enumerate(containers):
        for c in container["interior_cols"]:
            assert c not in col_to_container, "container openings must not overlap"
            col_to_container[c] = idx

    counts = [[0, 0] for _ in containers]
    for r, c in np.argwhere((grid == WATER) | (grid == OIL)):
        c = int(c)
        assert c in col_to_container, "every liquid pixel must be above exactly one opening"
        idx = col_to_container[c]
        if int(grid[r, c]) == WATER:
            counts[idx][0] += 1
        else:
            counts[idx][1] += 1
    return [tuple(pair) for pair in counts]


def _solve_with_details(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[tuple[int, int]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    components = find_components(
        g,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[WALL],
        return_format="dict",
    )
    assert len(components) >= 3, "expected several grey containers"
    containers = sorted((_parse_container(g, comp) for comp in components), key=lambda item: item["bbox"][1])

    for left, right in zip(containers, containers[1:]):
        assert left["bbox"][3] < right["bbox"][1], "containers must be separated left-to-right"

    local_counts = _collect_local_counts(g, containers)
    settled, meta = _simulate(containers, local_counts)

    out = g.copy()
    out[(out == WATER) | (out == OIL)] = BG
    for container, (water_count, oil_count) in zip(containers, settled):
        colors = [WATER] * int(water_count) + [OIL] * int(oil_count)
        assert len(colors) <= container["capacity"]
        for (r, c), color in zip(container["fill_cells"], colors):
            out[r, c] = color

    return out, containers, settled, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop blue water and yellow oil into grey basins, layer them by density, and propagate overflow across the lower rims.

    Input structure:
    - A black background contains several separated grey open-top rectangular basins aligned along the bottom.
    - Each basin has one taller wall and one shorter wall, so its stable fill height is limited by the lower rim.
    - Blue and yellow single-cell droplets appear above the basin openings; each droplet column belongs to exactly one basin.

    Rule:
    - For each basin, count all blue droplets above its opening as water and all yellow droplets as oil.
    - Fill the basin interior from the bottom row upward and from left to right within each row, placing all water first and then oil above it.
    - If the total liquid exceeds the basin capacity below the lower rim, spill the excess into the adjacent basin on that lower side; the spilled portion comes from the top layer first, so oil spills before water.
    - Process such overflows as a chain, merging spilled liquid with the next basin's own droplets, then clear all airborne droplets while keeping the grey walls unchanged.

    Key insight:
    - The solution depends on combining three cues at once: opening-column ownership, water-below-oil stratification inside each basin, and oil-first cascading overflow determined by the shorter wall.
    """
    out, _containers, _settled, _meta = _solve_with_details(grid)
    return out


def _signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _draw_container(grid: np.ndarray, left: int, interior_w: int, bottom: int, left_h: int, right_h: int) -> dict:
    right = left + interior_w + 1
    grid[bottom, left : right + 1] = WALL
    grid[bottom - left_h + 1 : bottom + 1, left] = WALL
    grid[bottom - right_h + 1 : bottom + 1, right] = WALL
    return {
        "bbox": (bottom - max(left_h, right_h) + 1, left, bottom, right),
        "left_h": left_h,
        "right_h": right_h,
    }


def _air_cells(container: dict) -> list[tuple[int, int]]:
    high_top = int(container["high_top"])
    return [(r, c) for r in range(high_top) for c in container["interior_cols"]]


def _place_droplets(grid: np.ndarray, container: dict, water_count: int, oil_count: int) -> None:
    cells = _air_cells(container)
    colors = [WATER] * int(water_count) + [OIL] * int(oil_count)
    assert len(colors) <= len(cells), "not enough air space above the opening"
    for (r, c), color in zip(cells, colors):
        grid[r, c] = color


def _flow_order(n: int, direction: str) -> list[int]:
    return list(range(n)) if direction == "right" else list(range(n - 1, -1, -1))


def _sample_counts(caps: list[int], n: int) -> list[tuple[int, int]] | None:
    cap0, cap1, cap2 = caps[0], caps[1], caps[2]
    if cap0 < 2 or cap1 < 2 or cap2 < 4:
        return None

    keep_water0 = random.randint(1, cap0 - 1)
    keep_oil0 = cap0 - keep_water0
    spill_oil0 = random.randint(1, 3)
    local0 = (keep_water0, keep_oil0 + spill_oil0)

    spill_water1 = random.randint(1, 3)
    own_oil1 = random.randint(1, 3)
    local1 = (cap1 + spill_water1, own_oil1)
    incoming_to_2 = (spill_water1, spill_oil0 + own_oil1)

    free2 = cap2 - sum(incoming_to_2)
    if free2 < 2:
        return None
    own_water2 = random.randint(1, min(3, free2 - 1))
    own_oil2 = random.randint(1, min(3, free2 - own_water2))
    local2 = (own_water2, own_oil2)

    counts = [local0, local1, local2]
    for idx in range(3, n):
        cap = caps[idx]
        if cap == 1:
            counts.append((1, 0))
            continue
        own_water = random.randint(0, min(3, cap))
        own_oil = random.randint(0, min(3, cap - own_water))
        if own_water + own_oil == 0:
            own_water = 1
        counts.append((own_water, own_oil))
    return counts


def _build_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    direction = random.choice(["right", "left"])
    height = random.randint(18, 24)
    bottom = height - 1

    attempts = 0
    while attempts < 60:
        attempts += 1
        interior_ws = [random.randint(2, 3) for _ in range(n)]
        low_hs = [random.randint(2, 4) for _ in range(n)]
        high_deltas = [random.randint(1, 2) for _ in range(n)]
        outer_ws = [w + 2 for w in interior_ws]
        gaps = [random.randint(1, 2) for _ in range(n - 1)]
        left_margin = random.randint(1, 2)
        right_margin = random.randint(1, 2)
        width = left_margin + sum(outer_ws) + sum(gaps) + right_margin
        if width > 30:
            continue

        grid = np.full((height, width), BG, dtype=int)
        raw_specs = []
        c = left_margin
        for idx in range(n):
            low_h = low_hs[idx]
            high_h = low_h + high_deltas[idx]
            if direction == "right":
                left_h, right_h = high_h, low_h
            else:
                left_h, right_h = low_h, high_h
            raw_specs.append(_draw_container(grid, c, interior_ws[idx], bottom, left_h, right_h))
            c += outer_ws[idx]
            if idx + 1 < n:
                c += gaps[idx]

        containers = sorted(
            (_parse_container(grid, comp) for comp in find_components(grid, bg=BG, connectivity=4, monochromatic=True, colors=[WALL])),
            key=lambda item: item["bbox"][1],
        )
        order = _flow_order(n, direction)
        caps_flow = [containers[idx]["capacity"] for idx in order]
        local_flow = _sample_counts(caps_flow, n)
        if local_flow is None:
            continue

        local_lr = [(0, 0)] * n
        for idx, flow_idx in enumerate(order):
            local_lr[flow_idx] = local_flow[idx]

        for idx, container in enumerate(containers):
            if sum(local_lr[idx]) > len(_air_cells(container)):
                break
        else:
            for idx, container in enumerate(containers):
                water_count, oil_count = local_lr[idx]
                _place_droplets(grid, container, water_count, oil_count)

            try:
                out, solved_containers, settled, meta = _solve_with_details(grid)
            except AssertionError:
                continue

            spills = [item["spilled"] for item in meta]
            if sum(1 for spill in spills if spill != (0, 0)) < 2:
                continue
            if not any(spill[1] > 0 and spill[0] == 0 for spill in spills):
                continue
            if not any(spill[1] > 0 and spill[0] > 0 for spill in spills):
                continue
            if not any(item["incoming"] != (0, 0) and item["own"] != (0, 0) for item in meta):
                continue
            if not any(w > 0 and o > 0 for w, o in settled):
                continue
            if np.array_equal(grid, out):
                continue
            if len({tuple(pair) for pair in local_lr}) < 2:
                continue

            out_sig = _signature(out)
            if out_sig in _SEEN_OUTPUT_SIGNATURES:
                continue
            _SEEN_OUTPUT_SIGNATURES.add(out_sig)
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cascading-basin input")
