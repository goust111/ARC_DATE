# Variant of 28a6681f
# Dimension: D1 — 水流路径机制 (path)
# Change: 重力驱动的水流模拟：水像素先向下直落，被阻挡时通过唯一开放侧的对角方向下滑，支撑后水平滑向最近可落点... → 通道跟踪路径

from __future__ import annotations

from collections import deque
import random
import numpy as np


# Design analysis for the D1 variant
# a) Concrete behavior:
#    The original task moved blue water through open black space by gravity. That assumption
#    made lateral motion a secondary effect after vertical falling. In this variant the open
#    "air" is no longer the transport medium. Instead, color 8 marks a narrow pipe network.
#    Blue water starts packed at the source-side mouth of that pipe, and the output shows the
#    same amount of water translated to the farthest reachable pipe cells, usually in 2-3
#    lower terminal chambers. A concrete trace:
#    - Original: a blue pixel above a basin first drops, then maybe diagonally slides.
#    - Variant: the same blue pixel may move horizontally first, may turn upward inside the
#      network, and cannot leave color-8 cells at all.
#    - Difference: the relevant structure is the 4-connected channel topology, not the
#      vertical emptiness beneath the water.
# b) Complexity calibration:
#    A single straight pipe would make the change visually weak. Each generated sample uses
#    one source reservoir, one trunk, and 2 or 3 lower branches ending in wider chambers.
#    This keeps the rule visible while avoiding too many branch types. Width varies between
#    the source reservoir, trunk, and chambers, so "channel width determines flow capacity"
#    remains observable.
# c) Implicit assumption audit against the old main():
#    - Old input checks required supported water and forbade bg on both horizontal sides.
#      Those assumptions are invalid here because water can sit in a top-edge source mouth.
#    - Old settle() assumed motion choices came from local gravity, diagonal, and slide
#      rules. That entire mechanism must be replaced by graph traversal over corridor cells.
#    - Old output checks assumed stable water must be blocked on left/right/below. That is
#      also invalid because a pipe chamber may have empty pipe cells beside filled pipe cells.
#    New assumptions are explicit instead: water lies on the pipe network, the pipe component
#    touches at least one non-bottom edge cell, and generator construction guarantees that
#    only one such source mouth exists.
# d) Placement legality:
#    The generator creates the corridor first as a tree-shaped set of in-bounds coordinates,
#    then paints solid blocks only onto non-corridor cells. Source mouths touch only one of
#    top/left/right edges; terminal chambers are kept away from left/right edges to avoid
#    ambiguity with the source detector. No rectangles overlap because every paint step skips
#    existing corridor cells.
# e) Visual observability:
#    Humans can infer the rule from 2-4 examples because the orange pipe color is explicit
#    and the water packet always relocates from the edge source into remote bottom chambers,
#    often through long horizontal segments where gravity alone would not explain the motion.
#    Branch count and source side vary, but the underlying transformation is always "follow
#    the same pipe graph away from the source".
# f) Visual cleanliness:
#    The grid stays 10x10 like the original. Corridor cells use a single dedicated color,
#    water never overlaps solids, and wall colors are confined mostly to the lower third so
#    the pipe network remains legible.
# g) Color inferability:
#    Only four semantic roles exist: bg=0, water=1, pipe=8, plus two solid wall colors.
#    No hidden ordering depends on color id; color 8 is always the traversable channel and
#    blue is always the movable fluid.


BG = 0
WATER = 1
PIPE = 8


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _multi_source_dist(mask: np.ndarray, sources: list[tuple[int, int]]) -> np.ndarray:
    h, w = mask.shape
    dist = -np.ones((h, w), dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in sources:
        if bool(mask[r, c]) and int(dist[r, c]) == -1:
            dist[r, c] = 0
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c, h, w):
            if bool(mask[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                q.append((rr, cc))
    return dist


def _rank_cells(
    cells: list[tuple[int, int]],
    dist: np.ndarray,
    reverse: bool,
) -> list[tuple[int, int]]:
    if reverse:
        return sorted(cells, key=lambda p: (int(dist[p]), int(p[0]), int(p[1])), reverse=True)
    return sorted(cells, key=lambda p: (int(dist[p]), int(p[0]), int(p[1])))


def _detect_source_cells(mask: np.ndarray, water_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    edge_cells = [
        (int(r), int(c))
        for r, c in np.argwhere(mask)
        if (int(r) == 0 or int(c) == 0 or int(c) == w - 1) and int(r) != h - 1
    ]
    if not edge_cells:
        return []
    water_cells = [(int(r), int(c)) for r, c in np.argwhere(water_mask)]
    if not water_cells:
        return edge_cells
    water_dist = _multi_source_dist(mask, water_cells)
    best = min(int(water_dist[r, c]) for r, c in edge_cells if int(water_dist[r, c]) >= 0)
    return [(r, c) for r, c in edge_cells if int(water_dist[r, c]) == best]


def _paint_rect(
    grid: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    color: int,
    forbidden: set[tuple[int, int]],
) -> None:
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(grid.shape[0], int(r0 + h))
    cc1 = min(grid.shape[1], int(c0 + w))
    for r in range(rr0, rr1):
        for c in range(cc0, cc1):
            if (r, c) not in forbidden:
                grid[r, c] = int(color)


def _add_hline(cells: set[tuple[int, int]], r: int, c0: int, c1: int) -> None:
    for c in range(min(c0, c1), max(c0, c1) + 1):
        cells.add((int(r), int(c)))


def _add_vline(cells: set[tuple[int, int]], c: int, r0: int, r1: int) -> None:
    for r in range(min(r0, r1), max(r0, r1) + 1):
        cells.add((int(r), int(c)))


def _build_corridor() -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    source_side = random.choice(["top", "left", "right"])
    split_r = random.choice([4, 5])
    split_c = random.choice([4, 5])
    corridor: set[tuple[int, int]] = set()

    chamber_specs = [
        (8, 1, 2, random.choice([2, 3])),
        (7, 4, 3, 2),
        (8, 7, 2, 2),
    ]
    random.shuffle(chamber_specs)
    chamber_specs = sorted(chamber_specs[: random.choice([2, 3])], key=lambda x: x[1])

    if source_side == "top":
        src_w = random.choice([2, 3])
        src_c0 = random.choice([1, 2, 5, 6])
        outlet_c = int(src_c0 + src_w // 2)
        for r in range(0, 2):
            for c in range(src_c0, src_c0 + src_w):
                corridor.add((r, c))
        _add_vline(corridor, outlet_c, 1, split_r)
        _add_hline(corridor, split_r, outlet_c, split_c)
    elif source_side == "left":
        src_h = random.choice([2, 3])
        src_r0 = random.choice([1, 2])
        outlet_r = int(src_r0 + src_h // 2)
        for r in range(src_r0, src_r0 + src_h):
            for c in range(0, 2):
                corridor.add((r, c))
        _add_hline(corridor, outlet_r, 1, split_c)
        _add_vline(corridor, split_c, outlet_r, split_r)
    else:
        src_h = random.choice([2, 3])
        src_r0 = random.choice([1, 2])
        outlet_r = int(src_r0 + src_h // 2)
        for r in range(src_r0, src_r0 + src_h):
            for c in range(8, 10):
                corridor.add((r, c))
        _add_hline(corridor, outlet_r, split_c, 8)
        _add_vline(corridor, split_c, outlet_r, split_r)

    for r0, c0, h, w in chamber_specs:
        entry_c = int(c0 + w // 2)
        _add_hline(corridor, split_r, split_c, entry_c)
        _add_vline(corridor, entry_c, split_r, r0)
        for r in range(r0, r0 + h):
            for c in range(c0, c0 + w):
                corridor.add((r, c))

    # Add one small width-2 bulge on the trunk or on a branch so width is visible.
    widen_candidates = [
        (r, c)
        for r, c in corridor
        if 1 <= r <= 7 and 2 <= c <= 7 and (r + 1, c) in corridor and (r, c + 1) not in corridor
    ]
    random.shuffle(widen_candidates)
    for r, c in widen_candidates[:2]:
        corridor.add((r, c + 1))

    source_cells = []
    for r, c in sorted(corridor):
        if r == 0 or c == 0 or c == 9:
            if r != 9:
                source_cells.append((r, c))
    assert source_cells
    return corridor, source_cells


def _decorate_solids(
    corridor: set[tuple[int, int]],
    wall_left: int,
    wall_right: int,
) -> np.ndarray:
    grid = np.zeros((10, 10), dtype=int)
    for r, c in corridor:
        grid[r, c] = PIPE

    # Bottom masses preserve the original "basin area" look without changing the new path rule.
    _paint_rect(grid, 7, 0, 3, 4, wall_left, corridor)
    _paint_rect(grid, 8, 3, 2, 3, wall_left, corridor)
    _paint_rect(grid, 7, 6, 3, 4, wall_right, corridor)
    _paint_rect(grid, 8, 4, 2, 3, wall_right, corridor)

    # A few mid-level supports make the pipe appear embedded rather than floating.
    for r, c in sorted(corridor):
        if r >= 6:
            continue
        if c + 1 < 10 and (r, c + 1) not in corridor and grid[r, c + 1] == BG and random.random() < 0.18:
            grid[r, c + 1] = wall_right
        if c - 1 >= 0 and (r, c - 1) not in corridor and grid[r, c - 1] == BG and random.random() < 0.18:
            grid[r, c - 1] = wall_left
    return grid


def _make_example() -> tuple[np.ndarray, np.ndarray]:
    wall_colors = random.sample([2, 3, 4, 5, 6, 7, 9], 2)
    corridor, source_cells = _build_corridor()
    base = _decorate_solids(corridor, wall_colors[0], wall_colors[1])

    mask = (base == PIPE)
    dist = _multi_source_dist(mask, source_cells)
    all_cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    assert all(int(dist[r, c]) >= 0 for r, c in all_cells)

    near = _rank_cells(all_cells, dist, reverse=False)
    far = _rank_cells(all_cells, dist, reverse=True)

    max_k = min(10, len(all_cells) - 3)
    min_k = min(5, max_k)
    water_count = random.randint(min_k, max_k)

    inp = base.copy()
    out = base.copy()
    for r, c in near[:water_count]:
        inp[r, c] = WATER
    for r, c in far[:water_count]:
        out[r, c] = WATER

    return inp, out


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Route a packet of water through a prepainted pipe network into remote lower chambers.

    Input structure:
    - bg = 0. Water pixels = color 1. Pipe/channel cells = color 8.
    - Two other colors form static basin-like rock or wall structures around the lower area.
    - Water begins packed near the pipe mouth on the top, left, or right border instead of resting in open air.

    Rule:
    - Treat every pipe or water cell as one 4-connected channel network.
    - Use the channel cells touching a non-bottom border as the source mouth and measure path distance through the network.
    - Clear the input water back to pipe cells, then place the same number of water pixels onto the farthest reachable channel cells.
    - Solid wall colors never move; only the blue packet is translated along the corridor topology.

    Key insight: The blue cells no longer obey gravity in open space. They follow the explicit channel graph, so turns, branches, and wider terminal chambers determine the final arrangement.
    """
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape

    mask = (g == PIPE) | (g == WATER)
    source_cells = _detect_source_cells(mask, g == WATER)

    if check_input:
        assert g.shape == (10, 10), "generator and solver both assume the original 10x10 canvas"
        assert source_cells, "channel must touch a non-bottom edge to define the source mouth"
        water_cells = [(int(r), int(c)) for r, c in np.argwhere(g == WATER)]
        assert water_cells, "input must contain water"
        assert all(bool(mask[r, c]) for r, c in water_cells), "water must lie on the pipe network"

    dist = _multi_source_dist(mask, source_cells)
    channel_cells = [(int(r), int(c)) for r, c in np.argwhere(mask) if int(dist[int(r), int(c)]) >= 0]
    water_count = int(np.sum(g == WATER))
    ranked = _rank_cells(channel_cells, dist, reverse=True)

    out = g.copy()
    out[out == WATER] = PIPE
    for r, c in ranked[:water_count]:
        out[r, c] = WATER

    if check_output:
        assert int(np.sum(out == WATER)) == water_count
        assert all(out[r, c] == WATER for r, c in ranked[:water_count])
        assert source_cells, "source detection must stay valid after transport"

    return out


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp, out = _make_example()
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid corridor-following instance")
