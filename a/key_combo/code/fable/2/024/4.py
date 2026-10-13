from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
DIRS4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8: tuple[Coord, ...] = tuple(
    (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
)
PALETTE = tuple(range(10))


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: list[Coord] | set[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_majority_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    border = list(map(int, grid[0, :]))
    border.extend(map(int, grid[h - 1, :]))
    if h > 2:
        border.extend(map(int, grid[1 : h - 1, 0]))
        border.extend(map(int, grid[1 : h - 1, w - 1]))
    return int(Counter(border).most_common(1)[0][0])


def _components_from_mask(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[Coord]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            cid = len(comps)
            q = deque([(r, c)])
            labels[r, c] = cid
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = cid
                        q.append((nr, nc))
            comps.append(comp)
    return labels, comps


def _solid_seed_component(coords: list[Coord]) -> bool:
    if len(coords) < 4:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    return (r1 - r0 + 1) >= 2 and (c1 - c0 + 1) >= 2 and len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _infer_seed_components_and_valves(
    grid: np.ndarray, bg: int, wall: int
) -> tuple[list[dict[str, object]], np.ndarray]:
    seeds: list[dict[str, object]] = []
    valve_mask = np.zeros_like(grid, dtype=bool)
    for color in sorted(int(x) for x in np.unique(grid) if int(x) not in {bg, wall}):
        for comp in _components_from_mask(grid == int(color)):
            if _solid_seed_component(comp):
                seeds.append({"color": int(color), "coords": comp})
            else:
                for r, c in comp:
                    valve_mask[r, c] = True
    return seeds, valve_mask


def _row_run_lengths(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        c = 0
        while c < w:
            if not mask[r, c]:
                c += 1
                continue
            c0 = c
            while c < w and mask[r, c]:
                c += 1
            out[r, c0:c] = c - c0
    return out


def _col_run_lengths(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=int)
    for c in range(w):
        r = 0
        while r < h:
            if not mask[r, c]:
                r += 1
                continue
            r0 = r
            while r < h and mask[r, c]:
                r += 1
            out[r0:r, c] = r - r0
    return out


def _cells_in_open_2x2(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r in range(h - 1):
        for c in range(w - 1):
            if mask[r : r + 2, c : c + 2].all():
                out[r : r + 2, c : c + 2] = True
    return out


def _pipe_mask(grid: np.ndarray, bg: int, valve_mask: np.ndarray) -> np.ndarray:
    openish = (grid == int(bg)) | valve_mask
    wide = _cells_in_open_2x2(openish)
    hlen = _row_run_lengths(openish)
    vlen = _col_run_lengths(openish)
    return openish & (~wide) & ((hlen >= 3) | (vlen >= 3))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood seed colors through width-one background pipe networks and stop at embedded valve pixels.

    Input structure:
    - One background color dominates the border, and one dense wall color fills the noisy interior field.
    - Inside the wall, width-one background corridors form orthogonal networks; some corridor cells are replaced by lone valve pixels of another color.
    - Each network touches exactly one solid seed rectangle at one endpoint, and there are extra background holes that do not belong to any network.

    Rule:
    - Infer the wall as the largest non-background color, the seed blocks as the remaining solid rectangular components, and the leftover colored singletons as valves.
    - Treat only width-one long straight openings, plus their orthogonal junctions, as pipe cells; include valve pixels in the geometric network but not in the floodable set.
    - From each seed-adjacent pipe mouth, flood through reachable background pipe cells, recolor them to that seed color, and stop whenever a valve blocks further reachability.

    Key insight:
    - The true object is negative space: the blank corridors are the pipes, while the isolated colored valve pixels look like noise unless they are interpreted as blockers embedded inside that empty network.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = _border_majority_color(g)
    non_bg = g[g != int(bg)]
    if non_bg.size == 0:
        return g.copy()
    wall = int(Counter(map(int, non_bg.flat)).most_common(1)[0][0])

    seeds, valve_mask = _infer_seed_components_and_valves(g, int(bg), int(wall))
    pipe = _pipe_mask(g, int(bg), valve_mask)
    labels, _comps = _label_components(pipe)
    passable = pipe & (g == int(bg))

    out = g.copy()
    claimed_components: set[int] = set()
    for seed in seeds:
        color = int(seed["color"])
        starts: set[Coord] = set()
        touched: set[int] = set()
        for r, c in seed["coords"]:
            for nr, nc in _neighbors4(int(r), int(c)):
                if not _in_bounds(g.shape, nr, nc):
                    continue
                cid = int(labels[nr, nc])
                if cid >= 0:
                    touched.add(cid)
                    if passable[nr, nc]:
                        starts.add((nr, nc))
        if not touched:
            continue
        if len(touched) != 1:
            raise ValueError("each seed must touch exactly one pipe network")
        cid = next(iter(touched))
        if cid in claimed_components:
            raise ValueError("pipe network touched by multiple seeds")
        claimed_components.add(cid)
        if not starts:
            continue

        q = deque(starts)
        seen = set(starts)
        while q:
            r, c = q.popleft()
            out[r, c] = int(color)
            for nr, nc in _neighbors4(r, c):
                if (
                    _in_bounds(g.shape, nr, nc)
                    and passable[nr, nc]
                    and int(labels[nr, nc]) == cid
                    and (nr, nc) not in seen
                ):
                    seen.add((nr, nc))
                    q.append((nr, nc))
    return out


def _transform_point(p: Coord, rot: int, mirror: bool) -> Coord:
    r, c = p
    if mirror:
        c = -c
    for _ in range(rot % 4):
        r, c = c, -r
    return int(r), int(c)


def _transform_dir(d: Coord, rot: int, mirror: bool) -> Coord:
    return _transform_point(d, rot, mirror)


def _seed_cells(start: Coord, flow: Coord) -> set[Coord]:
    back = (-flow[0], -flow[1])
    perp_choices = [(-back[1], back[0]), (back[1], -back[0])]
    perp = random.choice(perp_choices)
    depth = random.randint(2, 3)
    height = random.randint(2, 3)
    cells: set[Coord] = set()
    for i in range(1, depth + 1):
        for j in range(height):
            cells.add((start[0] + back[0] * i + perp[0] * j, start[1] + back[1] * i + perp[1] * j))
    return cells


def _template_blocked_branch() -> dict[str, object]:
    trunk = random.randint(11, 15)
    b1 = random.randint(3, 4)
    b2 = random.randint(b1 + 4, trunk - 3)
    valve = random.randint(b1 + 2, b2 - 2)
    l1 = random.randint(3, 5)
    l2 = random.randint(3, 5)
    s1 = random.choice((-1, 1))
    s2 = random.choice((-1, 1))

    cells = {(0, c) for c in range(trunk)}
    cells |= {(s1 * k, b1) for k in range(1, l1 + 1)}
    cells |= {(s2 * k, b2) for k in range(1, l2 + 1)}
    return {
        "kind": "blocked_branch",
        "cells": cells,
        "valves": {(0, valve)},
        "start": (0, 0),
        "flow": (0, 1),
    }


def _template_open_turn() -> dict[str, object]:
    trunk = random.randint(7, 10)
    branch_at = random.randint(2, trunk - 4)
    branch_len = random.randint(3, 5)
    turn_len = random.randint(3, 6)
    branch_side = random.choice((-1, 1))
    turn_side = random.choice((-1, 1))

    cells = {(0, c) for c in range(trunk)}
    cells |= {(branch_side * k, branch_at) for k in range(1, branch_len + 1)}
    cells |= {(turn_side * k, trunk - 1) for k in range(1, turn_len + 1)}
    return {
        "kind": "open_turn",
        "cells": cells,
        "valves": set(),
        "start": (0, 0),
        "flow": (0, 1),
    }


def _materialize_network(template: dict[str, object]) -> dict[str, object]:
    rot = random.randrange(4)
    mirror = bool(random.getrandbits(1))
    cells = {_transform_point(p, rot, mirror) for p in template["cells"]}
    valves = {_transform_point(p, rot, mirror) for p in template["valves"]}
    start = _transform_point(template["start"], rot, mirror)
    flow = _transform_dir(template["flow"], rot, mirror)
    return {
        "kind": template["kind"],
        "cells": cells,
        "valves": valves,
        "start": start,
        "flow": flow,
        "seed": _seed_cells(start, flow),
    }


def _translate_coords(coords: set[Coord], dr: int, dc: int) -> set[Coord]:
    return {(r + dr, c + dc) for r, c in coords}


def _expanded(coords: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    out: set[Coord] = set()
    for r, c in coords:
        for dr, dc in DIRS8 + ((0, 0),):
            rr, cc = r + dr, c + dc
            if _in_bounds(shape, rr, cc):
                out.add((rr, cc))
    return out


def _place_network(
    h: int,
    w: int,
    wall_box: tuple[int, int, int, int],
    blocked: np.ndarray,
    network: dict[str, object],
) -> dict[str, object] | None:
    wr0, wc0, wr1, wc1 = wall_box
    all_cells = set(network["cells"]) | set(network["seed"])
    r0, c0, r1, c1 = _bbox(all_cells)

    row_min = wr0 + 1 - r0
    row_max = wr1 - 1 - r1
    col_min = wc0 + 1 - c0
    col_max = wc1 - 1 - c1
    if row_min > row_max or col_min > col_max:
        return None

    choices = [
        (dr, dc)
        for dr in range(row_min, row_max + 1)
        for dc in range(col_min, col_max + 1)
    ]
    random.shuffle(choices)
    for dr, dc in choices[:250]:
        placed_cells = _translate_coords(set(network["cells"]), dr, dc)
        placed_seed = _translate_coords(set(network["seed"]), dr, dc)
        footprint = _expanded(placed_cells | placed_seed, (h, w))
        if any(blocked[r, c] for r, c in footprint):
            continue
        placed = dict(network)
        placed["cells"] = placed_cells
        placed["valves"] = _translate_coords(set(network["valves"]), dr, dc)
        placed["seed"] = placed_seed
        placed["start"] = (network["start"][0] + dr, network["start"][1] + dc)
        placed["footprint"] = footprint
        return placed
    return None


_BLANK_SHAPES: tuple[set[Coord], ...] = (
    {(0, 0)},
    {(0, 0), (0, 1)},
    {(0, 0), (1, 0)},
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0), (1, 0), (1, 1)},
)


def _place_blank_holes(
    grid: np.ndarray,
    wall_box: tuple[int, int, int, int],
    blocked: np.ndarray,
    count: int,
) -> list[set[Coord]] | None:
    h, w = grid.shape
    wr0, wc0, wr1, wc1 = wall_box
    holes: list[set[Coord]] = []
    for _ in range(count):
        shape = set(random.choice(_BLANK_SHAPES))
        r0, c0, r1, c1 = _bbox(shape)
        row_min = wr0 + 1 - r0
        row_max = wr1 - 1 - r1
        col_min = wc0 + 1 - c0
        col_max = wc1 - 1 - c1
        if row_min > row_max or col_min > col_max:
            return None
        positions = [
            (dr, dc)
            for dr in range(row_min, row_max + 1)
            for dc in range(col_min, col_max + 1)
        ]
        random.shuffle(positions)
        placed = None
        for dr, dc in positions[:200]:
            cells = _translate_coords(shape, dr, dc)
            footprint = _expanded(cells, (h, w))
            if any(blocked[r, c] for r, c in footprint):
                continue
            placed = cells
            for r, c in footprint:
                blocked[r, c] = True
            holes.append(cells)
            break
        if placed is None:
            return None
    return holes


def _reachable_pipe_cells(network: dict[str, object]) -> set[Coord]:
    passable = set(network["cells"]) - set(network["valves"])
    start = network["start"]
    if start not in passable:
        return set()
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in passable and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(22, 30)
    w = random.randint(22, 30)

    bg, wall, valve = random.sample(PALETTE, 3)
    seed_count = random.randint(2, 3)
    seed_colors = random.sample([c for c in PALETTE if c not in {bg, wall, valve}], seed_count)

    top = random.randint(2, 3)
    left = random.randint(2, 3)
    bottom = h - random.randint(3, 4)
    right = w - random.randint(3, 4)
    wall_box = (top, left, bottom, right)
    if bottom - top < 12 or right - left < 12:
        return None

    templates = [_template_blocked_branch(), _template_open_turn()]
    while len(templates) < seed_count:
        templates.append(random.choice((_template_blocked_branch(), _template_open_turn())))
    random.shuffle(templates)
    raw_networks = [_materialize_network(t) for t in templates[:seed_count]]

    blocked = np.zeros((h, w), dtype=bool)
    placed_networks: list[dict[str, object]] = []
    for net, color in zip(raw_networks, seed_colors):
        placed = _place_network(h, w, wall_box, blocked, net)
        if placed is None:
            return None
        placed["seed_color"] = int(color)
        for r, c in placed["footprint"]:
            blocked[r, c] = True
        placed_networks.append(placed)

    hole_count = random.randint(2, 4)
    hole_cells = _place_blank_holes(np.empty((h, w), dtype=int), wall_box, blocked, hole_count)
    if hole_cells is None:
        return None

    inp = np.full((h, w), int(bg), dtype=int)
    wr0, wc0, wr1, wc1 = wall_box
    inp[wr0 : wr1 + 1, wc0 : wc1 + 1] = int(wall)

    for net in placed_networks:
        for r, c in net["cells"]:
            inp[r, c] = int(bg)
        for r, c in net["valves"]:
            inp[r, c] = int(valve)
        for r, c in net["seed"]:
            inp[r, c] = int(net["seed_color"])
    for hole in hole_cells:
        for r, c in hole:
            inp[r, c] = int(bg)

    out = inp.copy()
    blocked_example = False
    open_example = False
    for net in placed_networks:
        reachable = _reachable_pipe_cells(net)
        for r, c in reachable:
            out[r, c] = int(net["seed_color"])
        if net["valves"]:
            if any(cell not in reachable for cell in (set(net["cells"]) - set(net["valves"]))):
                blocked_example = True
        else:
            if reachable == set(net["cells"]):
                open_example = True

    if not blocked_example or not open_example:
        return None
    if np.array_equal(inp, out):
        return None
    for hole in hole_cells:
        if any(out[r, c] != int(bg) for r, c in hole):
            return None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(300):
        case = _build_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid valve-pipe network input")
