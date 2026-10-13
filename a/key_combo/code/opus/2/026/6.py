from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
WATER = 1
RED = 2
GREEN = 3
YELLOW = 4
SHELF = 5


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Minimal 4-connected component finder, adapted to arc_utils conventions."""
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _shelf_segments(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], dict[int, list[int]]]:
    """Return shelf segments as (sid, row, c0, c1) plus per-row index lists."""
    h, w = map(int, grid.shape)
    shelves: list[tuple[int, int, int, int]] = []
    row_map: dict[int, list[int]] = {r: [] for r in range(h)}
    sid = 0
    for r in range(h):
        c = 0
        while c < w:
            if int(grid[r, c]) != SHELF:
                c += 1
                continue
            c0 = c
            while c + 1 < w and int(grid[r, c + 1]) == SHELF:
                c += 1
            shelves.append((sid, r, c0, c))
            row_map[r].append(sid)
            sid += 1
            c += 1
    return shelves, row_map


def _source_starts(grid: np.ndarray) -> list[tuple[int, int, int]]:
    """Use bottom-edge cells of each top-touching source component as stream starts."""
    starts: list[tuple[int, int, int]] = []
    source_id = 0
    for comp in _components_mask(grid == WATER):
        if not bool(np.any(comp[:, 0] == 0)):
            continue
        bottom = int(comp[:, 0].max())
        cols = sorted({int(c) for r, c in comp if int(r) == bottom})
        for c in cols:
            starts.append((bottom, c, source_id))
        source_id += 1
    return starts


def _pool_color(area: int) -> int:
    if area <= 2:
        return RED
    if area <= 4:
        return GREEN
    return YELLOW


def _find_shelf_id(shelves: list[tuple[int, int, int, int]], row_map: dict[int, list[int]], row: int, col: int) -> int | None:
    for sid in row_map.get(int(row), []):
        _sid, rr, c0, c1 = shelves[sid]
        if rr == row and c0 <= col <= c1:
            return sid
    return None


def _paint_bottom_row(original: np.ndarray, out: np.ndarray, col: int) -> tuple[int, int]:
    """Spread across the contiguous bottom-row background segment containing col."""
    h, w = map(int, original.shape)
    r = h - 1
    left = int(col)
    while left - 1 >= 0 and int(original[r, left - 1]) == BG:
        left -= 1
    right = int(col)
    while right + 1 < w and int(original[r, right + 1]) == BG:
        right += 1
    out[r, left : right + 1] = WATER
    return left, right


def _simulate(grid: np.ndarray, *, with_meta: bool = False):
    """
    Simulate vertical falling, shelf pooling, and overflow splitting.

    Water spreads on the row just above a shelf across the shelf span, plus one
    unsupported overflow pixel on each in-bounds side. Bottom-row water spreads
    through contiguous background only.
    """
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    out = g.copy()
    shelves, row_map = _shelf_segments(g)
    starts = _source_starts(g)

    meta = {
        "source_depth": defaultdict(int),
        "shelf_incoming": defaultdict(set),
        "bottom_segments": [],
        "starts": starts,
        "shelves": shelves,
    }

    q = deque((r, c, source_id, 0) for r, c, source_id in starts)
    seen: set[tuple[int, int, int]] = set()

    while q:
        r, c, source_id, depth = q.popleft()
        state = (int(r), int(c), int(source_id))
        if state in seen:
            continue
        seen.add(state)

        rr, cc = int(r), int(c)
        if int(g[rr, cc]) != SHELF:
            out[rr, cc] = WATER

        while True:
            if rr == h - 1:
                left, right = _paint_bottom_row(g, out, cc)
                meta["bottom_segments"].append((source_id, left, right))
                break

            if int(g[rr + 1, cc]) == SHELF:
                sid = _find_shelf_id(shelves, row_map, rr + 1, cc)
                if sid is None:
                    break
                _sid, shelf_row, c0, c1 = shelves[sid]
                meta["source_depth"][source_id] = max(meta["source_depth"][source_id], depth + 1)
                meta["shelf_incoming"][sid].add((int(source_id), int(rr), int(cc)))

                out[rr, c0 : c1 + 1] = WATER
                overflow_cols: list[int] = []
                if c0 - 1 >= 0 and int(g[rr, c0 - 1]) != SHELF:
                    out[rr, c0 - 1] = WATER
                    overflow_cols.append(c0 - 1)
                if c1 + 1 < w and int(g[rr, c1 + 1]) != SHELF:
                    out[rr, c1 + 1] = WATER
                    overflow_cols.append(c1 + 1)
                for nc in sorted(set(overflow_cols)):
                    q.append((rr, nc, source_id, depth + 1))
                break

            rr += 1
            if int(g[rr, cc]) != SHELF:
                out[rr, cc] = WATER

    if with_meta:
        return out, meta
    return out


def _collect_pools(original: np.ndarray, water_grid: np.ndarray, shelves: list[tuple[int, int, int, int]]) -> list[dict]:
    """Collect shelf-supported runs and bottom-row runs; overflow pixels are excluded."""
    h, w = map(int, original.shape)
    pools: list[dict] = []

    for sid, shelf_row, c0, c1 in shelves:
        if shelf_row == 0:
            continue
        r = shelf_row - 1
        c = c0
        while c <= c1:
            if int(water_grid[r, c]) != WATER:
                c += 1
                continue
            start = c
            while c + 1 <= c1 and int(water_grid[r, c + 1]) == WATER:
                c += 1
            pools.append(
                {
                    "kind": "shelf",
                    "sid": sid,
                    "row": r,
                    "c0": start,
                    "c1": c,
                    "area": c - start + 1,
                }
            )
            c += 1

    r = h - 1
    c = 0
    while c < w:
        if int(water_grid[r, c]) != WATER:
            c += 1
            continue
        start = c
        while c + 1 < w and int(water_grid[r, c + 1]) == WATER:
            c += 1
        pools.append(
            {
                "kind": "bottom",
                "sid": None,
                "row": r,
                "c0": start,
                "c1": c,
                "area": c - start + 1,
            }
        )
        c += 1

    return pools


def _recolor_pools(water_grid: np.ndarray, pools: list[dict]) -> np.ndarray:
    out = np.asarray(water_grid, dtype=int).copy()
    for pool in pools:
        out[int(pool["row"]), int(pool["c0"]) : int(pool["c1"]) + 1] = _pool_color(int(pool["area"]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate cascading waterfall branches, then recolor each shelf or bottom pool by its horizontal area.

    Input structure:
    - The grid uses black background, blue top-edge source clusters, and gray horizontal shelf segments.
    - Sources touch the top row and may be one or two pixels wide; shelves are disjoint horizontal bars at several heights.

    Rule:
    - Let each source fall vertically through background, painting blue water.
    - When water reaches a shelf, fill the row just above the shelf across the full shelf span and mark one extra blue overflow pixel on each in-bounds end.
    - Resume vertical falling from each overflow pixel; when a stream reaches the bottom row, spread across the contiguous bottom background segment.
    - After the full cascade is finished, identify each shelf-supported horizontal blue run and each bottom-row blue run, measure its length, and recolor lengths 1-2 red, 3-4 green, and 5+ yellow.
    - Blue pixels that are only part of vertical flow or overflow remain blue, and gray shelves stay gray.

    Key insight:
    - Shelf hits create a branching waterfall tree, and only the final horizontal resting segments are recolored, so solving requires both cascade simulation and pool-area counting.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    water = _simulate(g)
    pools = _collect_pools(g, water, _shelf_segments(g)[0])
    return _recolor_pools(water, pools)


def _top_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    c = 0
    w = int(grid.shape[1])
    while c < w:
        if int(grid[0, c]) != WATER:
            c += 1
            continue
        c0 = c
        while c + 1 < w and int(grid[0, c + 1]) == WATER:
            c += 1
        runs.append((c0, c))
        c += 1
    return runs


def _place_source(grid: np.ndarray, start: int, width: int, height: int) -> None:
    grid[:height, start : start + width] = WATER


def _extra_source_start(c0: int, c1: int, width: int, blocked: list[tuple[int, int]]) -> int | None:
    for start in random.sample(list(range(c0, c1 - width + 2)), k=max(0, c1 - c0 - width + 2)):
        end = start + width - 1
        if all(end < b0 - 1 or start > b1 + 1 for b0, b1 in blocked):
            return start
    return None


def _find_pool(pools: list[dict], kind: str, row: int, c0: int, c1: int) -> dict | None:
    for pool in pools:
        if pool["kind"] == kind and int(pool["row"]) == int(row) and int(pool["c0"]) == int(c0) and int(pool["c1"]) == int(c1):
            return pool
    return None


def _validate_instance(grid: np.ndarray, spec: dict) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if not set(map(int, np.unique(g))).issubset({BG, WATER, SHELF}):
        return False

    source_comps = [comp for comp in _components_mask(g == WATER) if bool(np.any(comp[:, 0] == 0))]
    if len(source_comps) != int(spec["source_count"]):
        return False
    if any(int(comp[:, 1].max()) - int(comp[:, 1].min()) + 1 not in (1, 2) for comp in source_comps):
        return False

    water, meta = _simulate(g, with_meta=True)
    shelves = meta["shelves"]
    pools = _collect_pools(g, water, shelves)
    areas = {int(pool["area"]) for pool in pools}
    if len(areas) < 3:
        return False
    if not any(1 <= int(pool["area"]) <= 2 for pool in pools):
        return False
    if not any(3 <= int(pool["area"]) <= 4 for pool in pools):
        return False
    if not any(int(pool["area"]) >= 5 for pool in pools):
        return False
    if max(meta["source_depth"].values(), default=0) < 2:
        return False

    shelf_lookup = {(row, c0, c1): sid for sid, row, c0, c1 in shelves}
    right_sid = shelf_lookup.get((spec["right_row"], spec["right_c0"], spec["right_c1"]))
    if right_sid is None or len(meta["shelf_incoming"].get(right_sid, set())) < 2:
        return False

    root_pool = _find_pool(pools, "shelf", spec["root_row"] - 1, spec["root_c0"], spec["root_c1"])
    left_pool = _find_pool(pools, "shelf", spec["left_row"] - 1, spec["left_c0"], spec["left_c1"])
    right_pool = _find_pool(pools, "shelf", spec["right_row"] - 1, spec["right_c0"], spec["right_c1"])
    bottom_pool = _find_pool(pools, "bottom", g.shape[0] - 1, 0, spec["bottom_gap"] - 1)
    if root_pool is None or int(root_pool["area"]) != int(spec["root_area"]):
        return False
    if left_pool is None or int(left_pool["area"]) != int(spec["left_area"]):
        return False
    if right_pool is None or int(right_pool["area"]) != int(spec["right_area"]):
        return False
    if bottom_pool is None or int(bottom_pool["area"]) != int(spec["bottom_gap"]):
        return False

    out = _recolor_pools(water, pools)
    if not np.any(out == WATER):
        return False
    if not np.all(out[spec["root_row"] - 1, spec["root_c0"] : spec["root_c1"] + 1] == YELLOW):
        return False
    if not np.all(out[spec["left_row"] - 1, spec["left_c0"] : spec["left_c1"] + 1] == GREEN):
        return False
    if not np.all(out[g.shape[0] - 1, : spec["bottom_gap"]] == RED):
        return False
    if not np.any(out == RED) or not np.any(out == GREEN) or not np.any(out == YELLOW):
        return False
    if np.array_equal(out, g):
        return False

    # Visual clarity: keep the scene relatively sparse.
    if np.count_nonzero(g != BG) > int(0.38 * g.size):
        return False

    return True


def _build_candidate() -> tuple[np.ndarray, dict] | None:
    h = random.randint(18, 24)
    w = random.randint(12, 20)

    bottom_gap = random.choice([1, 2])
    left_width = random.choice([3, 4])
    root_width = random.randint(5, 7)
    right_width = random.randint(5, 7)
    bottom_block_len = random.randint(3, min(7, w - bottom_gap))

    left_c0 = 1
    left_c1 = left_c0 + left_width - 1
    src1_width = random.randint(1, 2)
    src2_width = random.randint(1, 2)

    choices: list[tuple[int, list[int]]] = []
    for root_c0 in range(2, left_c1 + 2):
        root_c1 = root_c0 + root_width - 1
        if root_c1 >= w - 3:
            continue
        right_incoming_col = root_c1 + 1
        right_c0_min = max(left_c1 + 3, right_incoming_col + src2_width - right_width + 1)
        right_c0_max = min(right_incoming_col, w - right_width)
        if right_c0_min > right_c0_max:
            continue
        choices.append((root_c0, list(range(right_c0_min, right_c0_max + 1))))
    if not choices:
        return None

    root_c0, right_c0_choices = random.choice(choices)
    root_c1 = root_c0 + root_width - 1
    right_incoming_col = root_c1 + 1
    right_c0 = random.choice(right_c0_choices)
    right_c1 = right_c0 + right_width - 1

    source1_start = random.randint(root_c0 + 1, root_c1 - src1_width)
    source2_low = max(right_incoming_col + 1, right_c0)
    source2_high = right_c1 - src2_width + 1
    if source2_low > source2_high:
        return None
    source2_start = random.randint(source2_low, source2_high)

    root_row = random.randint(4, 6)
    left_row_low = root_row + 4
    left_row_high = min(root_row + 7, h - 6)
    if left_row_low > left_row_high:
        return None
    left_row = random.randint(left_row_low, left_row_high)

    right_row_low = root_row + 3
    right_row_high = min(left_row + 1, h - 6)
    if right_row_low > right_row_high:
        return None
    right_row = random.randint(right_row_low, right_row_high)

    grid = np.zeros((h, w), dtype=int)
    source_specs: list[tuple[int, int, int]] = []
    source_specs.append((source1_start, src1_width, random.randint(1, 2)))
    source_specs.append((source2_start, src2_width, random.randint(1, 2)))

    if random.random() < 0.4:
        src3_width = random.randint(1, 2)
        src3_start = _extra_source_start(right_c0, right_incoming_col - 1, src3_width, [(source1_start, source1_start + src1_width - 1), (source2_start, source2_start + src2_width - 1)])
        if src3_start is not None:
            source_specs.append((src3_start, src3_width, random.randint(1, 2)))

    for start, width, height in source_specs:
        _place_source(grid, start, width, height)

    shelves = [
        (root_row, root_c0, root_c1),
        (left_row, left_c0, left_c1),
        (right_row, right_c0, right_c1),
        (h - 1, bottom_gap, bottom_gap + bottom_block_len - 1),
    ]

    active_cols = {
        0,
        root_c0 - 1,
        root_c1 + 1,
        left_c0 - 1,
        left_c1 + 1,
        right_c0 - 1,
        right_c1 + 1,
    }
    for start, width, _height in source_specs:
        active_cols.update(range(start, start + width))

    extra_rows_forbidden = {root_row, left_row, right_row, h - 1}
    for row, c0, c1 in shelves:
        grid[row, c0 : c1 + 1] = SHELF

    for _ in range(random.randint(0, 2)):
        placed = False
        for _attempt in range(40):
            row = random.randint(2, h - 3)
            if row in extra_rows_forbidden:
                continue
            length = random.randint(3, min(7, w))
            c0 = random.randint(0, w - length)
            c1 = c0 + length - 1
            if any(col in active_cols for col in range(c0, c1 + 1)):
                continue
            if any(row == other_row and not (c1 < other_c0 - 1 or c0 > other_c1 + 1) for other_row, other_c0, other_c1 in shelves):
                continue
            if any(abs(row - other_row) <= 1 and not (c1 < other_c0 - 1 or c0 > other_c1 + 1) for other_row, other_c0, other_c1 in shelves):
                continue
            grid[row, c0 : c1 + 1] = SHELF
            shelves.append((row, c0, c1))
            extra_rows_forbidden.add(row)
            placed = True
            break
        if not placed:
            continue

    spec = {
        "source_count": len(source_specs),
        "root_row": root_row,
        "root_c0": root_c0,
        "root_c1": root_c1,
        "root_area": root_width,
        "left_row": left_row,
        "left_c0": left_c0,
        "left_c1": left_c1,
        "left_area": left_width,
        "right_row": right_row,
        "right_c0": right_c0,
        "right_c1": right_c1,
        "right_area": right_width,
        "bottom_gap": bottom_gap,
    }
    return grid, spec


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_candidate()
        if built is None:
            continue
        grid, spec = built
        if _validate_instance(grid, spec):
            return grid
    raise RuntimeError("failed to generate a valid cascade-pool instance")
