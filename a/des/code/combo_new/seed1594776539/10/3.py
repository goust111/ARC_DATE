import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

SOURCE_COLORS = (BLUE, RED, YELLOW)
MIX_MAP = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
}
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

MAX_GENERATION_ATTEMPTS = 300
PAIR_OPTIONS = [(RED, BLUE), (RED, YELLOW), (BLUE, YELLOW)]
SIDE_NAMES = ("left", "right", "top", "bottom")
SIDE_PAIR_OPTIONS = [
    ("left", "right"),
    ("top", "bottom"),
    ("left", "top"),
    ("left", "bottom"),
    ("right", "top"),
    ("right", "bottom"),
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _cells_from_rect(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _rects_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR4]


def _label_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _solid_3x3_component(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    color: int,
) -> tuple[int, int, int, int] | None:
    if len(coords) != 9:
        return None
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 != 2 or c1 - c0 != 2:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if patch.shape != (3, 3) or not np.all(patch == color):
        return None
    return (r0, c0, r1, c1)


def _source_ports(source: dict) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = source["bbox"]
    return {
        "left": (r0 + 1, c0 - 1),
        "right": (r0 + 1, c1 + 1),
        "top": (r0 - 1, c0 + 1),
        "bottom": (r1 + 1, c0 + 1),
    }


def _pool_ports(pool: dict) -> dict[str, tuple[int, int]]:
    r0, c0 = pool["top_left"]
    return {
        "left": (r0 + 1, c0 - 1),
        "right": (r0 + 1, c0 + 3),
        "top": (r0 - 1, c0 + 1),
        "bottom": (r0 + 3, c0 + 1),
    }


def _detect_sources(grid: np.ndarray) -> list[dict]:
    sources: list[dict] = []
    for color in SOURCE_COLORS:
        for comp in _label_components(grid == color):
            bbox = _solid_3x3_component(grid, comp, color)
            if bbox is None:
                continue
            sources.append({"color": color, "bbox": bbox, "cells": set(comp)})
    assert len(sources) in (4, 6), "expected 4 or 6 source blocks"
    return sources


def _detect_pools(grid: np.ndarray) -> list[dict]:
    pools: list[dict] = []
    used = np.zeros(grid.shape, dtype=bool)
    h, w = grid.shape
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            patch = grid[r0 : r0 + 3, c0 : c0 + 3]
            if patch.shape != (3, 3) or not np.all(patch == GREY):
                continue
            if np.any(used[r0 : r0 + 3, c0 : c0 + 3]):
                continue
            pool = {"top_left": (r0, c0), "cells": {(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)}}
            touching_ports = []
            for side, (pr, pc) in _pool_ports(pool).items():
                if _in_bounds(grid.shape, pr, pc) and grid[pr, pc] == GREY:
                    touching_ports.append(side)
            if len(touching_ports) != 2:
                continue
            pool["touching_sides"] = tuple(touching_ports)
            pools.append(pool)
            used[r0 : r0 + 3, c0 : c0 + 3] = True
    assert len(pools) in (2, 3), "expected 2 or 3 pools"
    return pools


def _trace_path_to_pool(
    grid: np.ndarray,
    start: tuple[int, int],
    pool_by_cell: dict[tuple[int, int], int],
) -> tuple[list[tuple[int, int]], int]:
    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    found: tuple[int, int] | None = None

    while q:
        cell = q.popleft()
        if cell in pool_by_cell:
            found = cell
            break
        r, c = cell
        for nr, nc in _neighbors4(r, c):
            nxt = (nr, nc)
            if not _in_bounds(grid.shape, nr, nc):
                continue
            if grid[nr, nc] != GREY or nxt in prev:
                continue
            prev[nxt] = cell
            q.append(nxt)

    assert found is not None, "every source pipe must reach a pool"
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = found
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path, pool_by_cell[found]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey pipe by its source color and fill each grey pool with the mixed color of its two inputs.

    Input structure:
    - The background is black.
    - Several solid 3x3 source blocks use red, blue, or yellow.
    - Several solid 3x3 grey pool blocks are connected to sources by 4-connected, one-pixel-wide grey pipes.
    - Each source has exactly one outgoing pipe, and each pool receives exactly two pipes.

    Rule:
    - Detect every 3x3 colored source block and its unique adjacent grey pipe port.
    - Follow that grey pipe until it reaches a 3x3 grey pool block, and recolor the pipe cells to the source color.
    - For each pool, read the two source colors that reach it and replace the entire 3x3 pool with their fixed mixed color:
      red+blue -> magenta, red+yellow -> orange, blue+yellow -> green.
    - Keep the source blocks unchanged.

    Key insight:
    - The grid encodes a small connection graph: colored 3x3 blocks are sources, embedded 3x3 grey regions are pools, and solving requires tracing connectivity before applying a learned color-mixing lookup.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sources = _detect_sources(g)
    pools = _detect_pools(g)
    pool_by_cell = {cell: idx for idx, pool in enumerate(pools) for cell in pool["cells"]}

    out = g.copy()
    pool_inputs: dict[int, list[int]] = {idx: [] for idx in range(len(pools))}

    for source in sources:
        ports = _source_ports(source)
        live_ports = []
        for side, cell in ports.items():
            r, c = cell
            if _in_bounds(g.shape, r, c) and g[r, c] == GREY:
                live_ports.append((side, cell))
        assert len(live_ports) == 1, "each source must expose exactly one grey pipe"
        _side, start = live_ports[0]
        path, pool_idx = _trace_path_to_pool(g, start, pool_by_cell)
        for r, c in path:
            if (r, c) not in pools[pool_idx]["cells"]:
                out[r, c] = source["color"]
        pool_inputs[pool_idx].append(source["color"])

    for idx, pool in enumerate(pools):
        colors = pool_inputs[idx]
        assert len(colors) == 2 and len(set(colors)) == 2, "each pool must receive two distinct source colors"
        mixed = MIX_MAP[frozenset(colors)]
        for r, c in pool["cells"]:
            out[r, c] = mixed

    return out


def _line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    assert ac == bc
    step = 1 if br >= ar else -1
    return [(r, ac) for r in range(ar, br + step, step)]


def _polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = _line(points[i], points[i + 1])
        if i:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _choose_pair_types(n_pools: int, rng: random.Random) -> list[tuple[int, int]]:
    if n_pools == 3:
        pairs = list(PAIR_OPTIONS)
        rng.shuffle(pairs)
        return pairs
    pairs = rng.sample(PAIR_OPTIONS, 2)
    rng.shuffle(pairs)
    return pairs


def _place_pools(h: int, w: int, n_pools: int, rng: random.Random) -> list[dict] | None:
    candidates = [(r, c) for r in range(4, h - 6) for c in range(4, w - 6)]
    rng.shuffle(candidates)
    pools: list[dict] = []
    rects: list[tuple[int, int, int, int]] = []
    for r0, c0 in candidates:
        rect = (r0, c0, r0 + 2, c0 + 2)
        if any(_rects_too_close(rect, other, gap=1) for other in rects):
            continue
        rects.append(rect)
        pools.append({"top_left": (r0, c0), "rect": rect, "center": (r0 + 1, c0 + 1)})
        if len(pools) == n_pools:
            return pools
    return None


def _choose_side_pairs(n_pools: int, rng: random.Random) -> list[tuple[str, str]] | None:
    side_counts = {side: 0 for side in SIDE_NAMES}
    pairs: list[tuple[str, str]] = []
    for _ in range(n_pools):
        candidates = [
            pair
            for pair in SIDE_PAIR_OPTIONS
            if side_counts[pair[0]] < 2 and side_counts[pair[1]] < 2
        ]
        if not candidates:
            return None
        pair = rng.choice(candidates)
        pairs.append(pair)
        side_counts[pair[0]] += 1
        side_counts[pair[1]] += 1
    rng.shuffle(pairs)
    return pairs


def _source_spec_from_side(
    side: str,
    axis: int,
    h: int,
    w: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    if side == "left":
        return (axis - 1, 1, axis + 1, 3), (axis, 4)
    if side == "right":
        return (axis - 1, w - 4, axis + 1, w - 2), (axis, w - 5)
    if side == "top":
        return (1, axis - 1, 3, axis + 1), (4, axis)
    return (h - 4, axis - 1, h - 2, axis + 1), (h - 5, axis)


def _choose_source_axis(
    side: str,
    pool_center: tuple[int, int],
    h: int,
    w: int,
    used_rects: list[tuple[int, int, int, int]],
    pool_by_cell: dict[tuple[int, int], int],
    rng: random.Random,
) -> tuple[tuple[int, int, int, int], tuple[int, int]] | None:
    pool_axis = pool_center[0] if side in ("left", "right") else pool_center[1]
    upper = h - 2 if side in ("left", "right") else w - 2
    candidates = list(range(2, upper))
    candidates = [v for v in candidates if abs(v - pool_axis) >= 2]
    if not candidates:
        return None
    preferred = pool_axis + rng.choice([-6, -4, -2, 2, 4, 6])
    rng.shuffle(candidates)
    candidates.sort(key=lambda v: abs(v - preferred))
    for axis in candidates:
        rect, port = _source_spec_from_side(side, axis, h, w)
        if any(_rects_too_close(rect, other, gap=1) for other in used_rects):
            continue
        if _adjacent_pool_ids(port, pool_by_cell):
            continue
        return rect, port
    return None


def _path_candidate_lanes(
    side: str,
    start: tuple[int, int],
    end: tuple[int, int],
    h: int,
    w: int,
    rng: random.Random,
) -> list[int]:
    if side == "left":
        lanes = list(range(start[1], end[1] + 1))
    elif side == "right":
        lanes = list(range(end[1], start[1] + 1))
    elif side == "top":
        lanes = list(range(start[0], end[0] + 1))
    else:
        lanes = list(range(end[0], start[0] + 1))
    rng.shuffle(lanes)
    return lanes


def _build_path_for_lane(
    side: str,
    start: tuple[int, int],
    end: tuple[int, int],
    lane: int,
) -> list[tuple[int, int]]:
    if side in ("left", "right"):
        return _polyline([start, (start[0], lane), (end[0], lane), end])
    return _polyline([start, (lane, start[1]), (lane, end[1]), end])


def _dilate_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        expanded.add((r, c))
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if _in_bounds(shape, nr, nc):
                expanded.add((nr, nc))
    return expanded


def _adjacent_pool_ids(
    cell: tuple[int, int],
    pool_by_cell: dict[tuple[int, int], int],
) -> set[int]:
    hits: set[int] = set()
    r, c = cell
    for nr, nc in _neighbors4(r, c):
        pool_idx = pool_by_cell.get((nr, nc))
        if pool_idx is not None:
            hits.add(pool_idx)
    return hits


def _reconstruct_path(
    prev: dict[tuple[int, int], tuple[int, int] | None],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _route_path_bfs(
    start: tuple[int, int],
    end: tuple[int, int],
    target_pool_idx: int,
    occupied: np.ndarray,
    blocked_near_paths: set[tuple[int, int]],
    reserved_ports: set[tuple[int, int]],
    pool_by_cell: dict[tuple[int, int], int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while q:
        cell = q.popleft()
        if cell == end:
            return _reconstruct_path(prev, end)
        dirs = list(DIR4)
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = cell[0] + dr, cell[1] + dc
            nxt = (nr, nc)
            if not _in_bounds(occupied.shape, nr, nc) or nxt in prev:
                continue
            if occupied[nr, nc]:
                continue
            if nxt in blocked_near_paths and nxt not in (start, end):
                continue
            if nxt in reserved_ports and nxt not in (start, end):
                continue
            adjacent_pool_ids = _adjacent_pool_ids(nxt, pool_by_cell)
            if nxt == end:
                if adjacent_pool_ids != {target_pool_idx}:
                    continue
            elif adjacent_pool_ids:
                continue
            prev[nxt] = cell
            q.append(nxt)
    return None


def _choose_sources_and_routes(
    pools: list[dict],
    pair_types: list[tuple[int, int]],
    side_pairs: list[tuple[str, str]],
    h: int,
    w: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict]] | None:
    sources: list[dict] = []
    block_rects = [pool["rect"] for pool in pools]
    pool_by_cell = {
        cell: idx
        for idx, pool in enumerate(pools)
        for cell in _cells_from_rect(pool["rect"])
    }

    pool_indices = list(range(len(pools)))
    rng.shuffle(pool_indices)
    for pool_idx in pool_indices:
        pool = pools[pool_idx]
        colors = list(pair_types[pool_idx])
        sides = list(side_pairs[pool_idx])
        rng.shuffle(colors)
        for side, color in zip(sides, colors):
            picked = _choose_source_axis(side, pool["center"], h, w, block_rects, pool_by_cell, rng)
            if picked is None:
                return None
            rect, port = picked
            block_rects.append(rect)
            sources.append(
                {
                    "pool_idx": pool_idx,
                    "side": side,
                    "color": color,
                    "rect": rect,
                    "port": port,
                    "cells": _cells_from_rect(rect),
                }
            )

    occupied = np.zeros((h, w), dtype=bool)
    for pool in pools:
        for r, c in _cells_from_rect(pool["rect"]):
            occupied[r, c] = True
    for source in sources:
        for r, c in source["cells"]:
            occupied[r, c] = True

    flat_reserved_ports = {tuple(source["port"]) for source in sources}
    for pool in pools:
        flat_reserved_ports.update(_pool_ports(pool).values())

    routes: list[dict] = []
    buffered_path_cells: set[tuple[int, int]] = set()
    source_order = list(range(len(sources)))
    source_order.sort(
        key=lambda idx: (
            abs(sources[idx]["port"][0] - _pool_ports(pools[sources[idx]["pool_idx"]])[sources[idx]["side"]][0])
            + abs(sources[idx]["port"][1] - _pool_ports(pools[sources[idx]["pool_idx"]])[sources[idx]["side"]][1])
        ),
        reverse=True,
    )

    for idx in source_order:
        source = sources[idx]
        start = tuple(source["port"])
        end = _pool_ports(pools[source["pool_idx"]])[source["side"]]
        chosen_path = _route_path_bfs(
            start,
            end,
            source["pool_idx"],
            occupied,
            buffered_path_cells,
            flat_reserved_ports,
            pool_by_cell,
            rng,
        )
        if chosen_path is None:
            return None
        for r, c in chosen_path:
            occupied[r, c] = True
        buffered_path_cells |= _dilate_cells(set(chosen_path), (h, w))
        routes.append({"source_idx": idx, "path": chosen_path})

    return sources, routes


def _render_input(
    h: int,
    w: int,
    pools: list[dict],
    sources: list[dict],
    routes: list[dict],
) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for pool in pools:
        r0, c0, r1, c1 = pool["rect"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    for route in routes:
        for r, c in route["path"]:
            grid[r, c] = GREY
    for source in sources:
        r0, c0, r1, c1 = source["rect"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = source["color"]
    return grid


def _expected_output(
    grid: np.ndarray,
    pools: list[dict],
    sources: list[dict],
    routes: list[dict],
) -> np.ndarray:
    out = grid.copy()
    by_pool: dict[int, list[int]] = {idx: [] for idx in range(len(pools))}
    for route in routes:
        source = sources[route["source_idx"]]
        by_pool[source["pool_idx"]].append(source["color"])
        for r, c in route["path"]:
            out[r, c] = source["color"]
    for idx, pool in enumerate(pools):
        mixed = MIX_MAP[frozenset(by_pool[idx])]
        r0, c0, r1, c1 = pool["rect"]
        out[r0 : r1 + 1, c0 : c1 + 1] = mixed
    return out


def _dedupe_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for cell in path:
        if cell not in seen:
            out.append(cell)
            seen.add(cell)
    return out


def _choose_side_axis_structured(
    side: str,
    pool_center_row: int,
    h: int,
    w: int,
    used_rects: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[tuple[int, int, int, int], tuple[int, int]] | None:
    offsets = [-3, -2, 2, 3]
    rng.shuffle(offsets)
    for offset in offsets:
        axis = pool_center_row + offset
        if not (2 <= axis <= h - 3):
            continue
        rect, port = _source_spec_from_side(side, axis, h, w)
        if any(_rects_too_close(rect, other, gap=1) for other in used_rects):
            continue
        return rect, port
    return None


def _validate_route_cells(
    path: list[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
    target_pool_idx: int,
    occupied: np.ndarray,
    reserved_ports: set[tuple[int, int]],
    pool_by_cell: dict[tuple[int, int], int],
) -> list[tuple[int, int]] | None:
    unique_path = _dedupe_path(path)
    for pos, (r, c) in enumerate(unique_path):
        if not _in_bounds(occupied.shape, r, c):
            return None
        if occupied[r, c]:
            return None
        if (r, c) in reserved_ports and (r, c) not in (start, end):
            return None
        adjacent_pools = _adjacent_pool_ids((r, c), pool_by_cell)
        if pos == len(unique_path) - 1:
            if adjacent_pools != {target_pool_idx}:
                return None
        elif adjacent_pools:
            return None
    return unique_path


def _transform_grid(grid: np.ndarray, ops: list[str]) -> np.ndarray:
    out = grid
    for op in ops:
        if op == "transpose":
            out = out.T
        elif op == "flipud":
            out = np.flipud(out)
        elif op == "fliplr":
            out = np.fliplr(out)
    return np.array(out, dtype=int, copy=True)


def _structured_horizontal_instance(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n_pools = rng.choice([2, 3])
    if n_pools == 2:
        h = rng.randint(17, 20)
    else:
        h = rng.randint(19, 20)
    w = rng.randint(17, 20)

    c0 = 7
    if w < 17 or h < 17:
        return None

    if n_pools == 2:
        row_choices = [rng.randint(4, 5), rng.randint(h - 7, h - 6)]
    else:
        middle_row = 9 if h == 19 else rng.randint(9, 10)
        row_choices = [rng.randint(4, 5), middle_row, h - 6]

    pools: list[dict] = []
    for row in row_choices:
        rect = (row, c0, row + 2, c0 + 2)
        pools.append({"top_left": (row, c0), "rect": rect, "center": (row + 1, c0 + 1)})

    pair_types = _choose_pair_types(n_pools, rng)
    used_rects = [pool["rect"] for pool in pools]
    sources: list[dict] = []
    pool_by_cell = {
        cell: idx
        for idx, pool in enumerate(pools)
        for cell in _cells_from_rect(pool["rect"])
    }

    left_mid_cols = [4, 5, 6][:n_pools]
    right_mid_cols = [w - 5, w - 6, w - 7][:n_pools]
    rng.shuffle(left_mid_cols)
    rng.shuffle(right_mid_cols)

    for pool_idx, pool in enumerate(pools):
        colors = list(pair_types[pool_idx])
        rng.shuffle(colors)
        left_spec = _choose_side_axis_structured("left", pool["center"][0], h, w, used_rects, rng)
        if left_spec is None:
            return None
        left_rect, left_port = left_spec
        used_rects.append(left_rect)
        sources.append(
            {
                "pool_idx": pool_idx,
                "side": "left",
                "color": colors[0],
                "rect": left_rect,
                "port": left_port,
                "cells": _cells_from_rect(left_rect),
                "mid_col": left_mid_cols[pool_idx],
            }
        )

        right_spec = _choose_side_axis_structured("right", pool["center"][0], h, w, used_rects, rng)
        if right_spec is None:
            return None
        right_rect, right_port = right_spec
        used_rects.append(right_rect)
        sources.append(
            {
                "pool_idx": pool_idx,
                "side": "right",
                "color": colors[1],
                "rect": right_rect,
                "port": right_port,
                "cells": _cells_from_rect(right_rect),
                "mid_col": right_mid_cols[pool_idx],
            }
        )

    occupied = np.zeros((h, w), dtype=bool)
    for pool in pools:
        for r, c in _cells_from_rect(pool["rect"]):
            occupied[r, c] = True
    for source in sources:
        for r, c in source["cells"]:
            occupied[r, c] = True

    reserved_ports = {tuple(source["port"]) for source in sources}
    for pool in pools:
        reserved_ports.update(_pool_ports(pool).values())

    routes: list[dict] = []
    for source in sources:
        pool = pools[source["pool_idx"]]
        end = _pool_ports(pool)[source["side"]]
        if source["side"] == "left":
            raw_path = _polyline(
                [
                    tuple(source["port"]),
                    (source["port"][0], source["mid_col"]),
                    (pool["center"][0], source["mid_col"]),
                    end,
                ]
            )
        else:
            raw_path = _polyline(
                [
                    tuple(source["port"]),
                    (source["port"][0], source["mid_col"]),
                    (pool["center"][0], source["mid_col"]),
                    end,
                ]
            )
        path = _validate_route_cells(
            raw_path,
            tuple(source["port"]),
            end,
            source["pool_idx"],
            occupied,
            reserved_ports,
            pool_by_cell,
        )
        if path is None:
            return None
        for r, c in path:
            occupied[r, c] = True
        routes.append({"source_idx": len(routes), "path": path})

    grid = _render_input(h, w, pools, sources, routes)
    expected = _expected_output(grid, pools, sources, routes)
    ops: list[str] = []
    if rng.random() < 0.5:
        ops.append("transpose")
    if rng.random() < 0.5:
        ops.append("flipud")
    if rng.random() < 0.5:
        ops.append("fliplr")
    return _transform_grid(grid, ops), _transform_grid(expected, ops)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    built = _structured_horizontal_instance(rng)
    if built is None:
        return None
    grid, expected = built
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
