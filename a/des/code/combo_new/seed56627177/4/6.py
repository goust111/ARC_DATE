from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
GREY = 5
BULB = 6
SOURCE_COLORS = (1, 2, 3, 4, 7, 8, 9)

Coord = tuple[int, int]


@dataclass(frozen=True)
class Wire:
    source: Coord
    bulb: Coord
    color: int
    orientation: str
    path: tuple[Coord, ...]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _cardinal(pos: Coord) -> list[Coord]:
    r, c = pos
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _wire_path(source: Coord, bulb: Coord) -> tuple[str, tuple[Coord, ...]]:
    sr, sc = source
    br, bc = bulb
    if sr == br:
        step = 1 if bc > sc else -1
        path = tuple((sr, c) for c in range(sc + step, bc, step))
        return "h", path
    if sc == bc:
        step = 1 if br > sr else -1
        path = tuple((r, sc) for r in range(sr + step, br, step))
        return "v", path
    raise AssertionError("wire endpoints must be aligned")


def _make_wire(source: Coord, bulb: Coord, color: int) -> Wire:
    orientation, path = _wire_path(source, bulb)
    assert path, "wire must contain at least one grey cell"
    return Wire(source=source, bulb=bulb, color=int(color), orientation=orientation, path=path)


def _cross_coord(a: Wire, b: Wire) -> Coord | None:
    if a.orientation == b.orientation:
        return None
    h = a if a.orientation == "h" else b
    v = b if a.orientation == "h" else a
    rr = h.source[0]
    cc = v.source[1]
    if (rr, cc) in set(h.path) and (rr, cc) in set(v.path):
        return (rr, cc)
    return None


def _parse_wires(grid: np.ndarray) -> list[Wire]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sources = sorted(
        tuple(map(int, pos))
        for pos in np.argwhere((g != BG) & (g != GREY) & (g != BULB))
    )
    bulbs = {tuple(map(int, pos)) for pos in np.argwhere(g == BULB)}
    assert sources, "expected at least one signal source"
    assert bulbs, "expected at least one bulb"

    wires: list[Wire] = []
    seen_bulbs: set[Coord] = set()
    grey_usage: dict[Coord, list[int]] = {}

    for source in sources:
        sr, sc = source
        color = int(g[sr, sc])
        neighbours = []
        for rr, cc in _cardinal(source):
            if _in_bounds((h, w), rr, cc) and int(g[rr, cc]) == GREY:
                neighbours.append((rr, cc))
        assert len(neighbours) == 1, "each source must touch exactly one grey wire cell"

        nr, nc = neighbours[0]
        dr = nr - sr
        dc = nc - sc
        assert abs(dr) + abs(dc) == 1

        path: list[Coord] = []
        rr, cc = nr, nc
        while _in_bounds((h, w), rr, cc) and int(g[rr, cc]) == GREY:
            path.append((rr, cc))
            rr += dr
            cc += dc

        assert path, "wire must include grey cells"
        assert _in_bounds((h, w), rr, cc), "wire must terminate inside the grid"
        assert int(g[rr, cc]) == BULB, "wire must end at a bulb"

        bulb = (rr, cc)
        seen_bulbs.add(bulb)
        orientation = "h" if dr == 0 else "v"
        wire = Wire(source=source, bulb=bulb, color=color, orientation=orientation, path=tuple(path))
        wires.append(wire)
        for cell in wire.path:
            grey_usage.setdefault(cell, []).append(len(wires) - 1)

    assert seen_bulbs == bulbs, "every bulb must belong to exactly one wire"
    grey_cells = {tuple(map(int, pos)) for pos in np.argwhere(g == GREY)}
    assert grey_cells == set(grey_usage), "all grey cells must belong to parsed wires"

    for cell, owners in grey_usage.items():
        assert 1 <= len(owners) <= 2, "grey cell may belong to one wire or one crossing pair"
        if len(owners) == 2:
            a, b = owners
            assert wires[a].orientation != wires[b].orientation, "overlapping parallel wires are invalid"
            assert cell == _cross_coord(wires[a], wires[b]), "shared grey cells must be orthogonal crossings"

    return wires


def _simulate_wires(wires: list[Wire]) -> tuple[list[bool], dict[Coord, int]]:
    cell_to_wires: dict[Coord, list[int]] = {}
    for idx, wire in enumerate(wires):
        for cell in wire.path:
            cell_to_wires.setdefault(cell, []).append(idx)

    crossings = {cell for cell, owners in cell_to_wires.items() if len(owners) > 1}
    owner: dict[Coord, int] = {}
    alive = [True] * len(wires)
    progress = [0] * len(wires)
    max_steps = max((len(wire.path) for wire in wires), default=0)

    for _time in range(1, max_steps + 1):
        crossing_arrivals: dict[Coord, list[int]] = {}
        straight_moves: list[int] = []

        for idx, wire in enumerate(wires):
            if not alive[idx]:
                continue
            if progress[idx] >= len(wire.path):
                continue
            cell = wire.path[progress[idx]]
            if cell in crossings:
                crossing_arrivals.setdefault(cell, []).append(idx)
            else:
                straight_moves.append(idx)

        for idx in straight_moves:
            progress[idx] += 1

        for cell, arrivals in crossing_arrivals.items():
            if cell in owner:
                for idx in arrivals:
                    alive[idx] = False
                continue

            if len(arrivals) == 1:
                idx = arrivals[0]
                owner[cell] = idx
                progress[idx] += 1
                continue

            winner = next((idx for idx in arrivals if wires[idx].orientation == "h"), arrivals[0])
            owner[cell] = winner
            progress[winner] += 1
            for idx in arrivals:
                if idx != winner:
                    alive[idx] = False

    return alive, owner


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    wires = _parse_wires(g)
    alive, _owner = _simulate_wires(wires)
    out = g.copy()
    for wire, survives in zip(wires, alive):
        if survives:
            br, bc = wire.bulb
            out[br, bc] = wire.color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decide which signal reaches each bulb in a crossing wire network by comparing source-to-crossing distances.

    Input structure:
    - The background is black.
    - Grey pixels form one-cell-wide horizontal or vertical wire segments, and some grey pixels are shared by one horizontal and one vertical wire as crossings.
    - Each wire has one colored source pixel at one endpoint and one magenta bulb pixel at the opposite endpoint.
    - Sources and bulbs are endpoints only; every grey pixel belongs to one straight wire or to one orthogonal crossing.

    Rule:
    - Parse every source-to-bulb straight wire and move each signal one grey pixel per step from its source toward its bulb.
    - At a crossing, the signal that reaches the crossing in fewer grey-pixel steps permanently claims that crossing and the later signal is blocked there.
    - If both signals reach a fresh crossing on the same step, the horizontal signal wins and the vertical signal stops.
    - A surviving signal recolors its bulb to the source color, while a blocked signal leaves its bulb magenta; sources and grey wires stay unchanged.

    Key insight:
    - The decisive quantity is not global wire length but the step count from each source to each crossing, so the network is solved by following each straight path and resolving conflicts in propagation order.
    """
    return _solve(grid)


def _render_grid(shape: tuple[int, int], wires: list[Wire]) -> np.ndarray | None:
    grid = np.full(shape, BG, dtype=int)
    for wire in wires:
        sr, sc = wire.source
        br, bc = wire.bulb
        if grid[sr, sc] != BG or grid[br, bc] != BG:
            return None
        grid[sr, sc] = wire.color
        grid[br, bc] = BULB
        for rr, cc in wire.path:
            if grid[rr, cc] in (BG, GREY):
                grid[rr, cc] = GREY
                continue
            return None
    return grid


def _crossing_count(wire: Wire, others: list[Wire]) -> int:
    return sum(_cross_coord(wire, other) is not None for other in others)


def _wire_with_cross(
    shape: tuple[int, int],
    cell: Coord,
    orientation: str,
    source_side: str,
    source_dist: int,
    other_dist: int,
    color: int,
) -> Wire | None:
    r, c = cell
    h, w = shape
    if orientation == "h":
        if source_side == "left":
            source = (r, c - source_dist)
            bulb = (r, c + other_dist)
        else:
            source = (r, c + source_dist)
            bulb = (r, c - other_dist)
    else:
        if source_side == "top":
            source = (r - source_dist, c)
            bulb = (r + other_dist, c)
        else:
            source = (r + source_dist, c)
            bulb = (r - other_dist, c)

    for rr, cc in (source, bulb):
        if not _in_bounds((h, w), rr, cc):
            return None
    try:
        return _make_wire(source, bulb, color)
    except AssertionError:
        return None


def _random_wire(shape: tuple[int, int], color: int) -> Wire | None:
    h, w = shape
    orientation = random.choice(("h", "v"))
    if orientation == "h":
        if w < 6:
            return None
        row = random.randint(1, h - 2)
        left = random.randint(1, w - 5)
        right = random.randint(left + 3, w - 2)
        if random.random() < 0.5:
            source, bulb = (row, left), (row, right)
        else:
            source, bulb = (row, right), (row, left)
    else:
        if h < 6:
            return None
        col = random.randint(1, w - 2)
        top = random.randint(1, h - 5)
        bottom = random.randint(top + 3, h - 2)
        if random.random() < 0.5:
            source, bulb = (top, col), (bottom, col)
        else:
            source, bulb = (bottom, col), (top, col)
    return _make_wire(source, bulb, color)


def _validated_grid(shape: tuple[int, int], wires: list[Wire]) -> np.ndarray | None:
    grid = _render_grid(shape, wires)
    if grid is None:
        return None
    try:
        parsed = _parse_wires(grid)
    except AssertionError:
        return None
    if len(parsed) != len(wires):
        return None
    return grid


def _mandatory_pair(shape: tuple[int, int], colors: list[int], scenario: str) -> tuple[list[Wire], Coord] | None:
    h, w = shape
    h_source_side = random.choice(("left", "right"))
    v_source_side = random.choice(("top", "bottom"))

    if scenario == "tie_h":
        dist = random.randint(2, 4)
        h_dist = v_dist = dist
    elif scenario == "strict_h":
        h_dist = random.randint(1, 3)
        v_dist = random.randint(h_dist + 1, min(5, h_dist + 3))
    else:
        v_dist = random.randint(1, 3)
        h_dist = random.randint(v_dist + 1, min(5, v_dist + 3))

    h_other = random.randint(3, 7)
    v_other = random.randint(3, 7)

    left_need = h_dist if h_source_side == "left" else h_other
    right_need = h_other if h_source_side == "left" else h_dist
    top_need = v_dist if v_source_side == "top" else v_other
    bottom_need = v_other if v_source_side == "top" else v_dist

    if left_need > w - 1 - right_need or top_need > h - 1 - bottom_need:
        return None

    cross = (
        random.randint(top_need, h - 1 - bottom_need),
        random.randint(left_need, w - 1 - right_need),
    )
    horizontal = _wire_with_cross(shape, cross, "h", h_source_side, h_dist, h_other, colors[0])
    vertical = _wire_with_cross(shape, cross, "v", v_source_side, v_dist, v_other, colors[1])
    if horizontal is None or vertical is None:
        return None
    return [horizontal, vertical], cross


def _add_crossing_wire(
    shape: tuple[int, int],
    wires: list[Wire],
    mandatory_cross: Coord,
    color: int,
) -> list[Wire] | None:
    host_index = random.randrange(2)
    host = wires[host_index]
    mandatory_idx = host.path.index(mandatory_cross)
    later_cells = list(host.path[mandatory_idx + 1 :])
    random.shuffle(later_cells)

    for cell in later_cells:
        orientation = "v" if host.orientation == "h" else "h"
        source_side = random.choice(("top", "bottom")) if orientation == "v" else random.choice(("left", "right"))
        for _ in range(20):
            source_dist = random.randint(1, 4)
            other_dist = random.randint(2, 6)
            extra = _wire_with_cross(shape, cell, orientation, source_side, source_dist, other_dist, color)
            if extra is None:
                continue
            if _crossing_count(extra, wires) != 1:
                continue
            grid = _validated_grid(shape, wires + [extra])
            if grid is not None:
                return wires + [extra]
    return None


def _add_isolated_wires(shape: tuple[int, int], wires: list[Wire], colors: list[int], count: int) -> list[Wire] | None:
    out = list(wires)
    for color in colors[:count]:
        placed = False
        for _ in range(80):
            candidate = _random_wire(shape, color)
            if candidate is None:
                continue
            if _crossing_count(candidate, out) != 0:
                continue
            grid = _validated_grid(shape, out + [candidate])
            if grid is None:
                continue
            out.append(candidate)
            placed = True
            break
        if not placed:
            return None
    return out


def _outcome_matches(
    wires: list[Wire],
    scenario: str,
    mandatory_cross: Coord,
    output: np.ndarray,
) -> bool:
    horizontal = next(wire for wire in wires[:2] if wire.orientation == "h")
    vertical = next(wire for wire in wires[:2] if wire.orientation == "v")
    h_wins = int(output[horizontal.bulb]) == horizontal.color
    v_wins = int(output[vertical.bulb]) == vertical.color

    h_dist = horizontal.path.index(mandatory_cross) + 1
    v_dist = vertical.path.index(mandatory_cross) + 1
    if scenario == "tie_h" and h_dist != v_dist:
        return False
    if scenario == "strict_h" and not (h_dist < v_dist):
        return False
    if scenario == "strict_v" and not (v_dist < h_dist):
        return False
    return h_wins and not v_wins if scenario in ("tie_h", "strict_h") else v_wins and not h_wins


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(16, 28), random.randint(16, 28))
    scenario = random.choice(("strict_h", "strict_v", "tie_h"))
    colors = random.sample(SOURCE_COLORS, k=5)

    pair = _mandatory_pair(shape, colors, scenario)
    if pair is None:
        return None
    wires, mandatory_cross = pair

    if random.random() < 0.6:
        expanded = _add_crossing_wire(shape, wires, mandatory_cross, colors[2])
        if expanded is None:
            return None
        wires = expanded
        next_colors = colors[3:]
    else:
        next_colors = colors[2:]

    isolated_count = 1 + (1 if random.random() < 0.45 else 0)
    wires = _add_isolated_wires(shape, wires, next_colors, isolated_count)
    if wires is None:
        return None

    grid = _validated_grid(shape, wires)
    if grid is None:
        return None

    out = main(grid)
    if not _outcome_matches(wires, scenario, mandatory_cross, out):
        return None

    bulbs_before = np.argwhere(grid == BULB)
    changed = 0
    unchanged = 0
    for r_raw, c_raw in bulbs_before:
        r = int(r_raw)
        c = int(c_raw)
        if int(out[r, c]) == BULB:
            unchanged += 1
        else:
            changed += 1
    if changed == 0 or unchanged == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wire-signal task input")
