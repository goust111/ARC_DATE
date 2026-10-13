from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

RANK_TO_COLOR = {
    1: BLUE,
    2: RED,
    3: GREEN,
    4: YELLOW,
}

GROUP_LAYOUTS = [
    [3, 2, 1],
    [3, 2, 1, 1],
    [2, 2, 1],
    [2, 2, 1, 1],
    [3, 3, 1],
    [2, 2, 2, 1],
]

GROUP_AREA_OPTIONS = {
    2: [(4, 6), (4, 7), (6, 7), (6, 8), (7, 8)],
    3: [(4, 6, 7), (4, 6, 8), (4, 7, 8), (6, 7, 8)],
}

AREA_POOL = (4, 6, 7, 8)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bridge_decomposition(mask: np.ndarray) -> tuple[list[np.ndarray], dict[tuple[int, int], int], list[np.ndarray]]:
    """Split black cells into cycle-rich hole components and bridge-path components."""
    mask = np.asarray(mask, dtype=bool)
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not cells:
        return [], {}, []

    index_of = {cell: i for i, cell in enumerate(cells)}
    adj = [[] for _ in cells]
    for i, (r, c) in enumerate(cells):
        for nr, nc in _neighbors4(r, c):
            j = index_of.get((nr, nc))
            if j is not None:
                adj[i].append(j)

    tin = [-1] * len(cells)
    low = [0] * len(cells)
    timer = 0
    bridges: set[tuple[int, int]] = set()

    def dfs(v: int, parent: int) -> None:
        nonlocal timer
        tin[v] = low[v] = timer
        timer += 1
        for to in adj[v]:
            if to == parent:
                continue
            if tin[to] != -1:
                low[v] = min(low[v], tin[to])
                continue
            dfs(to, v)
            low[v] = min(low[v], low[to])
            if low[to] > tin[v]:
                bridges.add((v, to) if v < to else (to, v))

    for start in range(len(cells)):
        if tin[start] == -1:
            dfs(start, -1)

    seen = [False] * len(cells)
    holes: list[np.ndarray] = []
    hole_of: dict[tuple[int, int], int] = {}

    for start in range(len(cells)):
        if seen[start]:
            continue
        q = deque([start])
        seen[start] = True
        comp = []
        while q:
            v = q.popleft()
            comp.append(cells[v])
            for to in adj[v]:
                edge = (v, to) if v < to else (to, v)
                if edge in bridges or seen[to]:
                    continue
                seen[to] = True
                q.append(to)
        if len(comp) >= 2:
            hid = len(holes)
            coords = np.array(comp, dtype=int)
            holes.append(coords)
            for cell in comp:
                hole_of[cell] = hid

    bridge_mask = mask.copy()
    for cell in hole_of:
        bridge_mask[cell] = False
    bridge_components = _find_components(bridge_mask)
    return holes, hole_of, bridge_components


def _hole_groups(
    shape: tuple[int, int], holes: list[np.ndarray], hole_of: dict[tuple[int, int], int], bridge_components: list[np.ndarray]
) -> list[list[int]]:
    graph = [set() for _ in holes]
    h, w = shape
    for coords in bridge_components:
        neighbor_holes = set()
        for r, c in coords:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w:
                    hid = hole_of.get((int(nr), int(nc)))
                    if hid is not None:
                        neighbor_holes.add(int(hid))
        if len(neighbor_holes) == 2:
            a, b = sorted(neighbor_holes)
            graph[a].add(b)
            graph[b].add(a)

    groups: list[list[int]] = []
    seen: set[int] = set()
    for start in range(len(holes)):
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        group = []
        while q:
            cur = q.popleft()
            group.append(cur)
            for nxt in sorted(graph[cur]):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        groups.append(group)
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group grey-template holes by black bridge channels, then fill each grouped hole by its within-group area rank.

    Input structure:
    - A grey template covers nearly the whole grid, and every meaningful black cell lies inside that template.
    - Black cells form several enclosed hole-shapes plus some 1-pixel-wide bridge channels between hole edges.
    - Each true hole is a compact 4-connected black region, while bridges stay as thin paths and do not count toward hole area.

    Rule:
    - Separate the compact hole regions from the thin bridge-path cells.
    - Build a hole graph where a bridge path joins the two holes it touches, then take connected components of that graph as groups.
    - For each group with at least two holes, sort its hole areas from small to large and recolor them rank-wise: 1->blue, 2->red, 3->green, 4->yellow.
    - Leave isolated holes black and preserve every bridge-path cell as black.

    Key insight:
    - Color depends on relative area inside each bridge-defined group, not on global area; the bridge topology must be resolved before ranking.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    holes, hole_of, bridge_components = _bridge_decomposition(g == BLACK)
    out = g.copy()

    for group in _hole_groups(g.shape, holes, hole_of, bridge_components):
        if len(group) < 2:
            continue
        ranked = sorted(group, key=lambda hid: int(holes[hid].shape[0]))
        areas = [int(holes[hid].shape[0]) for hid in ranked]
        assert len(areas) == len(set(areas)), "group hole areas must be unique"
        for rank, hid in enumerate(ranked, start=1):
            coords = holes[hid]
            out[coords[:, 0], coords[:, 1]] = RANK_TO_COLOR[rank]
    return out


def _shape(mask_rows: list[str], port_rows: tuple[int, ...]) -> dict:
    mask = np.array([[ch == "1" for ch in row] for row in mask_rows], dtype=bool)
    return {"mask": mask, "port_rows": tuple(int(v) for v in port_rows)}


SHAPES_BY_AREA = {
    4: [
        _shape(["11", "11"], (0, 1)),
    ],
    6: [
        _shape(["111", "111"], (0, 1)),
        _shape(["11", "11", "11"], (0, 1, 2)),
    ],
    7: [
        _shape(["011", "111", "110"], (1,)),
        _shape(["110", "111", "011"], (1,)),
    ],
    8: [
        _shape(["011", "111", "111"], (1, 2)),
        _shape(["110", "111", "111"], (1, 2)),
        _shape(["111", "111", "011"], (0, 1)),
        _shape(["111", "111", "110"], (0, 1)),
        _shape(["1111", "1111"], (0, 1)),
    ],
}


def _has_cross_group_rank_counterexample(groups: list[list[int]]) -> bool:
    ranks_by_area: dict[int, set[int]] = defaultdict(set)
    for areas in groups:
        for rank, area in enumerate(sorted(areas), start=1):
            ranks_by_area[int(area)].add(int(rank))
    return any(len(ranks) >= 2 for ranks in ranks_by_area.values())


def _choose_group_areas(group_sizes: list[int]) -> list[list[int]]:
    for _ in range(200):
        groups = [list(random.choice(GROUP_AREA_OPTIONS[size])) for size in group_sizes]
        if _has_cross_group_rank_counterexample(groups):
            return groups
    raise RuntimeError("failed to choose group areas with a cross-group rank counterexample")


def _choose_isolated_areas(count: int, grouped_areas: list[list[int]]) -> list[int]:
    flat = [area for areas in grouped_areas for area in areas]
    isolated = []
    for _ in range(count):
        if flat and random.random() < 0.7:
            isolated.append(int(random.choice(flat)))
        else:
            isolated.append(int(random.choice(AREA_POOL)))
    return isolated


def _lane_rows(n_lanes: int) -> tuple[int, list[int]]:
    height = 4 * n_lanes + 4 + random.randint(0, 2)
    rows = [3 + 4 * i for i in range(n_lanes)]
    return height, rows


def _instantiate_lane(area_sequence: list[int], lane_row: int) -> dict:
    holes = []
    cursor = random.randint(1, 3)
    for idx, area in enumerate(area_sequence):
        spec = random.choice(SHAPES_BY_AREA[int(area)])
        mask = spec["mask"]
        port_row = int(random.choice(spec["port_rows"]))
        top = int(lane_row - port_row)
        left = int(cursor)
        row_cols = np.where(mask[port_row])[0]
        holes.append(
            {
                "area": int(area),
                "mask": mask.copy(),
                "top": top,
                "left": left,
                "port_row": port_row,
                "lane_row": int(lane_row),
                "left_port": int(row_cols.min()),
                "right_port": int(row_cols.max()),
            }
        )
        cursor += int(mask.shape[1])
        if idx != len(area_sequence) - 1:
            cursor += random.randint(3, 5)
    span = cursor + random.randint(1, 3)
    return {"holes": holes, "span": int(span), "grouped": len(area_sequence) >= 2}


def _render_plan(lanes: list[dict], height: int) -> tuple[np.ndarray, np.ndarray]:
    width = max(int(lane["offset"] + lane["span"]) for lane in lanes) + random.randint(1, 3)
    grid = np.full((height, width), GREY, dtype=int)

    for lane in lanes:
        for hole in lane["holes"]:
            mask = hole["mask"]
            top = int(hole["top"])
            left = int(hole["left"] + lane["offset"])
            rr, cc = np.where(mask)
            coords = np.column_stack((rr + top, cc + left)).astype(int)
            hole["coords"] = coords
            hole["placed_left"] = left
            grid[coords[:, 0], coords[:, 1]] = BLACK

        if lane["grouped"]:
            for left_hole, right_hole in zip(lane["holes"], lane["holes"][1:]):
                row = int(left_hole["top"] + left_hole["port_row"])
                start = int(left_hole["placed_left"] + left_hole["right_port"] + 1)
                stop = int(right_hole["placed_left"] + right_hole["left_port"] - 1)
                assert start <= stop, "bridge must contain at least one path cell"
                grid[row, start : stop + 1] = BLACK

    expected = grid.copy()
    for lane in lanes:
        if not lane["grouped"]:
            continue
        ranked = sorted(lane["holes"], key=lambda hole: int(hole["area"]))
        for rank, hole in enumerate(ranked, start=1):
            coords = hole["coords"]
            expected[coords[:, 0], coords[:, 1]] = RANK_TO_COLOR[rank]

    return grid, expected


def _validate_generated_case(grid: np.ndarray, expected: np.ndarray, lanes: list[dict]) -> bool:
    grouped_lanes = [lane for lane in lanes if lane["grouped"]]
    isolated_lanes = [lane for lane in lanes if not lane["grouped"]]
    total_holes = sum(len(lane["holes"]) for lane in lanes)
    bridge_count = sum(len(lane["holes"]) - 1 for lane in grouped_lanes)
    if not (5 <= total_holes <= 7):
        return False
    if not (2 <= len(grouped_lanes) <= 3):
        return False
    if not (1 <= len(isolated_lanes) <= 2):
        return False
    if not (2 <= bridge_count <= 4):
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    grouped_areas = [[int(hole["area"]) for hole in lane["holes"]] for lane in grouped_lanes]
    if not _has_cross_group_rank_counterexample(grouped_areas):
        return False

    holes, hole_of, bridge_components = _bridge_decomposition(grid == BLACK)
    planned_areas = sorted(int(hole["area"]) for lane in lanes for hole in lane["holes"])
    if sorted(int(coords.shape[0]) for coords in holes) != planned_areas:
        return False

    groups = _hole_groups(grid.shape, holes, hole_of, bridge_components)
    group_sizes = sorted(len(group) for group in groups)
    target_sizes = sorted(len(lane["holes"]) for lane in grouped_lanes + isolated_lanes)
    if group_sizes != target_sizes:
        return False

    if len(bridge_components) != bridge_count:
        return False
    for coords in bridge_components:
        neighbor_holes = set()
        for r, c in coords:
            black_neighbors = 0
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == BLACK:
                    black_neighbors += 1
                    hid = hole_of.get((int(nr), int(nc)))
                    if hid is not None:
                        neighbor_holes.add(int(hid))
            if black_neighbors > 2:
                return False
        if len(neighbor_holes) != 2:
            return False

    return np.array_equal(main(grid), expected)


def _attempt_generate() -> np.ndarray | None:
    layout = list(random.choice(GROUP_LAYOUTS))
    group_sizes = [int(size) for size in layout if size >= 2]
    grouped_areas = _choose_group_areas(group_sizes)
    isolated_areas = _choose_isolated_areas(layout.count(1), grouped_areas)

    grouped_lanes = []
    for areas in grouped_areas:
        area_sequence = list(areas)
        random.shuffle(area_sequence)
        grouped_lanes.append({"areas": area_sequence, "grouped": True})

    isolated_lanes = [{"areas": [int(area)], "grouped": False} for area in isolated_areas]
    lane_specs = grouped_lanes + isolated_lanes
    random.shuffle(lane_specs)

    height, lane_rows = _lane_rows(len(lane_specs))
    lanes = []
    max_span = 0
    for lane_spec, lane_row in zip(lane_specs, lane_rows):
        lane = _instantiate_lane(list(lane_spec["areas"]), int(lane_row))
        lane["offset"] = 0
        lanes.append(lane)
        max_span = max(max_span, int(lane["span"]))

    width = max_span + 4
    if width > 28:
        return None

    for lane in lanes:
        slack = max(0, width - int(lane["span"]) - 1)
        lane["offset"] = random.randint(0, slack)

    grid, expected = _render_plan(lanes, height)
    if _validate_generated_case(grid, expected, lanes):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridge-grouped hole grid")
