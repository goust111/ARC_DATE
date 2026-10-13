from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GEAR_COLORS = (1, 3, 5, 6, 7, 8, 9)

TIP_ORDER = ("up", "right", "down", "left")
TIP_OFFSETS = {
    "up": (-2, 0),
    "right": (0, 2),
    "down": (2, 0),
    "left": (0, -2),
}
BODY_OFFSETS = tuple(
    (dr, dc)
    for dr in range(-1, 2)
    for dc in range(-1, 2)
) + ((0, -2), (0, 2), (-2, 0), (2, 0))
GEAR_OFFSETS = frozenset(BODY_OFFSETS)
BODY_ONLY_OFFSETS = frozenset((dr, dc) for dr, dc in BODY_OFFSETS if (dr, dc) not in TIP_OFFSETS.values())
GEAR_AREA = len(GEAR_OFFSETS)

ARROW_SHAPES = {
    "cw": ((0, 1), (1, 0), (1, 1)),
    "ccw": ((0, 0), (1, 0), (1, 1)),
}
ARROW_BBOX_ANCHOR = {
    "up": (-4, -1),
    "right": (-1, 3),
    "down": (3, -1),
    "left": (-1, -4),
}
DIR_TO_STEP = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

GEN_RETRIES = 300
BOARD_SIDE_RANGE = (3, 5)
CONNECTED_GEAR_RANGE = (2, 5)
ISOLATED_GEAR_RANGE = (0, 2)
MARGIN_RANGE = (2, 5)


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _gear_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr, dc in GEAR_OFFSETS}


def _tip_cell(center: tuple[int, int], direction: str) -> tuple[int, int]:
    r, c = center
    dr, dc = TIP_OFFSETS[direction]
    return r + dr, c + dc


def _find_gears(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    gears: list[dict] = []
    red_cells = set()

    for r in range(2, h - 2):
        for c in range(2, w - 2):
            body = int(g[r, c])
            if body in (BLACK, RED, YELLOW):
                continue

            ok = True
            for dr, dc in BODY_ONLY_OFFSETS:
                if int(g[r + dr, c + dc]) != body:
                    ok = False
                    break
            if not ok:
                continue

            marker_dir = None
            for direction in TIP_ORDER:
                tr, tc = _tip_cell((r, c), direction)
                value = int(g[tr, tc])
                if value == RED:
                    if marker_dir is not None:
                        ok = False
                        break
                    marker_dir = direction
                elif value != body:
                    ok = False
                    break
            if not ok or marker_dir is None:
                continue

            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    rr = r + dr
                    cc = c + dc
                    if (dr, dc) in GEAR_OFFSETS:
                        continue
                    if int(g[rr, cc]) != BLACK:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            gears.append(
                {
                    "center": (r, c),
                    "body_color": body,
                    "marker_dir": marker_dir,
                    "cells": _gear_cells((r, c)),
                }
            )
            red_cells.add(_tip_cell((r, c), marker_dir))

    actual_red = {tuple(map(int, rc)) for rc in np.argwhere(g == RED)}
    assert gears, "expected at least one gear"
    assert actual_red == red_cells, "every red pixel must be a gear marker"
    return gears


def _parse_arrow(g: np.ndarray, gears: list[dict]) -> tuple[int, str]:
    yellow_mask = g == YELLOW
    comps = _components4(yellow_mask)
    assert len(comps) == 1, "expected exactly one yellow arrow"
    coords = comps[0]
    assert len(coords) == 3, "yellow arrow must use exactly three pixels"

    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 == r0 + 1 and c1 == c0 + 1, "yellow arrow must be a 2x2 L-shape"

    present = set(coords)
    missing = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)} - present
    assert len(missing) == 1
    missing_cell = next(iter(missing))
    if missing_cell == (r0, c0):
        drive_rotation = "cw"
    else:
        assert missing_cell == (r0, c1), "unsupported arrow opening"
        drive_rotation = "ccw"

    arrow_cells = present
    distances = []
    for idx, gear in enumerate(gears):
        distance = min(
            abs(ar - gr) + abs(ac - gc)
            for ar, ac in arrow_cells
            for gr, gc in gear["cells"]
        )
        distances.append((distance, idx))

    best = min(distance for distance, _ in distances)
    drivers = [idx for distance, idx in distances if distance == best]
    assert best == 1 and len(drivers) == 1, "yellow arrow must touch exactly one driver gear"
    return drivers[0], drive_rotation


def _meshed(g1: dict, g2: dict, g: np.ndarray) -> bool:
    r1, c1 = g1["center"]
    r2, c2 = g2["center"]
    if r1 == r2 and abs(c1 - c2) == 6:
        left = min(c1, c2)
        return int(g[r1, left + 3]) == BLACK
    if c1 == c2 and abs(r1 - r2) == 6:
        top = min(r1, r2)
        return int(g[top + 3, c1]) == BLACK
    return False


def _adjacency(gears: list[dict], g: np.ndarray) -> list[list[int]]:
    adj = [[] for _ in gears]
    for i in range(len(gears)):
        for j in range(i + 1, len(gears)):
            if _meshed(gears[i], gears[j], g):
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _rotation_step(rotation: str) -> int:
    return 1 if rotation == "cw" else -1


def _propagate_directions(adj: list[list[int]], driver_idx: int, drive_rotation: str, gears: list[dict]) -> list[str]:
    new_dirs = [gear["marker_dir"] for gear in gears]
    depth = {driver_idx: 0}
    queue: deque[int] = deque([driver_idx])

    while queue:
        node = queue.popleft()
        for nxt in adj[node]:
            if nxt not in depth:
                depth[nxt] = depth[node] + 1
                queue.append(nxt)
            else:
                assert (depth[nxt] - depth[node]) % 2 == 1, "gear graph must be bipartite"

    base_step = _rotation_step(drive_rotation)
    for idx, gear in enumerate(gears):
        if idx not in depth:
            continue
        step = base_step if depth[idx] % 2 == 0 else -base_step
        old_idx = TIP_ORDER.index(gear["marker_dir"])
        new_dirs[idx] = TIP_ORDER[(old_idx + step) % 4]
    return new_dirs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate gear markers along the meshing chain driven by the unique yellow arrow.

    Input structure:
    - Black background.
    - Each gear is a 5x5 cross-shaped sprite: a solid 3x3 block plus one tip cell
      at the middle of each side.
    - Exactly one tip of each gear is red, marking orientation.
    - Exactly one yellow 3-cell L-shape touches the driver gear.

    Rule:
    - The yellow arrow tells the driver gear to rotate 90 degrees clockwise or
      counterclockwise.
    - Meshed gears are centered 6 cells apart in a cardinal direction, so their
      facing tooth tips have one black cell between them.
    - Rotation direction alternates at every meshing step along the connected
      transmission component.
    - Only the red marker moves; the yellow arrow is erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    gears = _find_gears(g)
    driver_idx, drive_rotation = _parse_arrow(g, gears)
    adj = _adjacency(gears, g)
    new_dirs = _propagate_directions(adj, driver_idx, drive_rotation, gears)

    out = g.copy()
    out[out == YELLOW] = BLACK

    for gear in gears:
        center = gear["center"]
        body = gear["body_color"]
        for direction in TIP_ORDER:
            tr, tc = _tip_cell(center, direction)
            out[tr, tc] = body

    for gear, direction in zip(gears, new_dirs):
        tr, tc = _tip_cell(gear["center"], direction)
        out[tr, tc] = RED

    return out


def _neighbors4(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _grow_tree(rows: int, cols: int, count: int) -> tuple[set[tuple[int, int]], set[tuple[tuple[int, int], tuple[int, int]]]] | None:
    start = (random.randrange(rows), random.randrange(cols))
    nodes = {start}
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    while len(nodes) < count:
        candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for r in range(rows):
            for c in range(cols):
                cell = (r, c)
                if cell in nodes:
                    continue
                attached = [nbr for nbr in _neighbors4(cell) if nbr in nodes]
                if len(attached) == 1:
                    candidates.append((cell, attached[0]))
        if not candidates:
            return None
        cell, parent = random.choice(candidates)
        nodes.add(cell)
        edges.add(tuple(sorted((cell, parent))))

    return nodes, edges


def _choose_isolated(
    rows: int,
    cols: int,
    occupied: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    chosen: list[tuple[int, int]] = []
    all_nodes = set(occupied)
    for _ in range(count):
        candidates = []
        for r in range(rows):
            for c in range(cols):
                cell = (r, c)
                if cell in all_nodes:
                    continue
                if any(nbr in all_nodes for nbr in _neighbors4(cell)):
                    continue
                candidates.append(cell)
        if not candidates:
            return None
        cell = random.choice(candidates)
        chosen.append(cell)
        all_nodes.add(cell)
    return chosen


def _free_sides(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> list[str]:
    r, c = cell
    sides = []
    for direction, (dr, dc) in DIR_TO_STEP.items():
        if (r + dr, c + dc) not in occupied:
            sides.append(direction)
    return sides


def _render_grid(
    board_rows: int,
    board_cols: int,
    gears: list[dict],
    driver_cell: tuple[int, int],
    arrow_side: str,
    drive_rotation: str,
    draw_arrow: bool,
    top_margin: int,
    left_margin: int,
    bottom_margin: int,
    right_margin: int,
) -> np.ndarray:
    h = top_margin + 5 + 6 * (board_rows - 1) + bottom_margin
    w = left_margin + 5 + 6 * (board_cols - 1) + right_margin
    grid = np.full((h, w), BLACK, dtype=int)

    centers: dict[tuple[int, int], tuple[int, int]] = {}
    for gear in gears:
        gr, gc = gear["cell"]
        center = (top_margin + 2 + 6 * gr, left_margin + 2 + 6 * gc)
        centers[(gr, gc)] = center
        gear["center"] = center
        for dr, dc in GEAR_OFFSETS:
            grid[center[0] + dr, center[1] + dc] = int(gear["body_color"])
        tr, tc = _tip_cell(center, gear["marker_dir"])
        grid[tr, tc] = RED

    if draw_arrow:
        center = centers[driver_cell]
        ar0 = center[0] + ARROW_BBOX_ANCHOR[arrow_side][0]
        ac0 = center[1] + ARROW_BBOX_ANCHOR[arrow_side][1]
        for dr, dc in ARROW_SHAPES[drive_rotation]:
            grid[ar0 + dr, ac0 + dc] = YELLOW

    return grid


def _expected_marker_dirs(
    gears: list[dict],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
    driver_cell: tuple[int, int],
    drive_rotation: str,
) -> dict[tuple[int, int], str]:
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {gear["cell"]: [] for gear in gears}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    depth = {driver_cell: 0}
    queue: deque[tuple[int, int]] = deque([driver_cell])
    while queue:
        node = queue.popleft()
        for nxt in adj[node]:
            if nxt not in depth:
                depth[nxt] = depth[node] + 1
                queue.append(nxt)

    out = {gear["cell"]: gear["marker_dir"] for gear in gears}
    base_step = _rotation_step(drive_rotation)
    for gear in gears:
        cell = gear["cell"]
        if cell not in depth:
            continue
        step = base_step if depth[cell] % 2 == 0 else -base_step
        old_idx = TIP_ORDER.index(gear["marker_dir"])
        out[cell] = TIP_ORDER[(old_idx + step) % 4]
    return out


def _gen_attempt() -> np.ndarray | None:
    board_rows = random.randint(*BOARD_SIDE_RANGE)
    board_cols = random.randint(*BOARD_SIDE_RANGE)

    connected_count = random.randint(CONNECTED_GEAR_RANGE[0], min(CONNECTED_GEAR_RANGE[1], len(GEAR_COLORS)))
    max_isolated = min(ISOLATED_GEAR_RANGE[1], len(GEAR_COLORS) - connected_count)
    isolated_count = random.randint(ISOLATED_GEAR_RANGE[0], max_isolated)

    tree = _grow_tree(board_rows, board_cols, connected_count)
    if tree is None:
        return None
    tree_nodes, tree_edges = tree

    isolated = _choose_isolated(board_rows, board_cols, tree_nodes, isolated_count)
    if isolated is None:
        return None

    all_nodes = set(tree_nodes) | set(isolated)
    driver_candidates = [cell for cell in tree_nodes if _free_sides(cell, all_nodes)]
    if not driver_candidates:
        return None
    driver_cell = random.choice(driver_candidates)
    arrow_side = random.choice(_free_sides(driver_cell, all_nodes))
    drive_rotation = random.choice(("cw", "ccw"))

    colors = random.sample(GEAR_COLORS, len(all_nodes))
    random.shuffle(colors)
    cells = list(tree_nodes) + list(isolated)
    random.shuffle(cells)
    gears = []
    for cell, color in zip(cells, colors):
        gears.append(
            {
                "cell": cell,
                "body_color": int(color),
                "marker_dir": random.choice(TIP_ORDER),
            }
        )

    top_margin = random.randint(*MARGIN_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)

    inp = _render_grid(
        board_rows,
        board_cols,
        [gear.copy() for gear in gears],
        driver_cell,
        arrow_side,
        drive_rotation,
        True,
        top_margin,
        left_margin,
        bottom_margin,
        right_margin,
    )

    expected_dirs = _expected_marker_dirs(gears, tree_edges, driver_cell, drive_rotation)
    out_gears = []
    for gear in gears:
        updated = gear.copy()
        updated["marker_dir"] = expected_dirs[gear["cell"]]
        out_gears.append(updated)

    expected = _render_grid(
        board_rows,
        board_cols,
        [gear.copy() for gear in out_gears],
        driver_cell,
        arrow_side,
        drive_rotation,
        False,
        top_margin,
        left_margin,
        bottom_margin,
        right_margin,
    )

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gear-rotation input")
