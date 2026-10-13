from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np


BG = 0
GREEN = 3
BEAD_COLORS = [1, 2, 4, 5, 6, 7, 8, 9]
MIN_HEIGHT = 3
MAX_GEN_ATTEMPTS = 500

CW_OPEN_DIR = {"up": "right", "right": "down", "down": "left", "left": "up"}
OPEN_SPECS = {
    "up": {"axis": (1, 0), "back": (-1, 0), "perp": (0, 1)},
    "down": {"axis": (-1, 0), "back": (1, 0), "perp": (0, 1)},
    "left": {"axis": (0, 1), "back": (0, -1), "perp": (1, 0)},
    "right": {"axis": (0, -1), "back": (0, 1), "perp": (1, 0)},
}


@dataclass
class Funnel:
    open_dir: str
    tip: tuple[int, int]
    height: int
    closed: bool
    axis: tuple[int, int]
    back: tuple[int, int]
    perp: tuple[int, int]
    wall_cells: tuple[tuple[int, int], ...]
    interior_map: dict[tuple[int, int], tuple[int, int]]


@dataclass
class Bead:
    pos: tuple[int, int]
    color: int


def _add(a: tuple[int, int], b: tuple[int, int], mul: int = 1) -> tuple[int, int]:
    return a[0] + b[0] * mul, a[1] + b[1] * mul


def _combine(
    origin: tuple[int, int],
    v1: tuple[int, int],
    k1: int,
    v2: tuple[int, int],
    k2: int,
) -> tuple[int, int]:
    return origin[0] + v1[0] * k1 + v2[0] * k2, origin[1] + v1[1] * k1 + v2[1] * k2


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(cells)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _halo(cells: Iterable[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                    out.add((rr, cc))
    return out


def _build_funnel(open_dir: str, tip: tuple[int, int], height: int, closed: bool, shape: tuple[int, int]) -> Funnel | None:
    spec = OPEN_SPECS[open_dir]
    axis = spec["axis"]
    back = spec["back"]
    perp = spec["perp"]
    wall_cells: list[tuple[int, int]] = []
    if closed:
        wall_cells.append(tip)
    for k in range(1, height + 1):
        wall_cells.append(_combine(tip, back, k, perp, -k))
        wall_cells.append(_combine(tip, back, k, perp, k))

    interior_map: dict[tuple[int, int], tuple[int, int]] = {}
    for k in range(1, height + 1):
        for m in range(-(k - 1), k):
            cell = _combine(tip, back, k, perp, m)
            interior_map[cell] = (k, m)

    all_cells = set(wall_cells) | set(interior_map) | {tip}
    if not all(_in_bounds(shape, cell) for cell in all_cells):
        return None

    return Funnel(
        open_dir=open_dir,
        tip=tip,
        height=height,
        closed=closed,
        axis=axis,
        back=back,
        perp=perp,
        wall_cells=tuple(wall_cells),
        interior_map=interior_map,
    )


def _signature(funnel: Funnel) -> tuple[int, int, str, int, bool]:
    return funnel.tip[0], funnel.tip[1], funnel.open_dir, funnel.height, funnel.closed


def _height_from_grid(grid: np.ndarray, tip: tuple[int, int], open_dir: str) -> int:
    back = OPEN_SPECS[open_dir]["back"]
    perp = OPEN_SPECS[open_dir]["perp"]
    height = 0
    while True:
        step = height + 1
        left = _combine(tip, back, step, perp, -step)
        right = _combine(tip, back, step, perp, step)
        if not (_in_bounds(grid.shape, left) and _in_bounds(grid.shape, right)):
            break
        if grid[left] != GREEN or grid[right] != GREEN:
            break
        height += 1
    return height


def _detect_funnels(grid: np.ndarray) -> list[Funnel]:
    g = np.asarray(grid, dtype=int)
    candidates: list[Funnel] = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            tip = (r, c)
            for open_dir in OPEN_SPECS:
                for closed in (True, False):
                    if closed and g[tip] != GREEN:
                        continue
                    if not closed and g[tip] == GREEN:
                        continue
                    height = _height_from_grid(g, tip, open_dir)
                    if height < MIN_HEIGHT:
                        continue
                    funnel = _build_funnel(open_dir, tip, height, closed, g.shape)
                    if funnel is None:
                        continue
                    if any(g[cell] == GREEN for cell in funnel.interior_map):
                        continue
                    candidates.append(funnel)

    candidates.sort(key=lambda f: (len(f.wall_cells), len(f.interior_map)), reverse=True)
    used_walls: set[tuple[int, int]] = set()
    used_interior: set[tuple[int, int]] = set()
    funnels: list[Funnel] = []
    for funnel in candidates:
        walls = set(funnel.wall_cells)
        interior = set(funnel.interior_map)
        if walls & used_walls:
            continue
        if interior & used_interior:
            continue
        funnels.append(funnel)
        used_walls |= walls
        used_interior |= interior

    green_cells = set(map(tuple, np.argwhere(g == GREEN)))
    assert used_walls == green_cells, "green pixels must be exactly funnel walls"
    assert funnels, "expected at least one funnel"
    return funnels


def _next_funnels(funnels: list[Funnel], shape: tuple[int, int]) -> list[int | None]:
    cell_to_funnel: dict[tuple[int, int], int] = {}
    for i, funnel in enumerate(funnels):
        for cell in funnel.interior_map:
            assert cell not in cell_to_funnel, "funnel interiors must not overlap"
            cell_to_funnel[cell] = i

    nxt: list[int | None] = [None] * len(funnels)
    for i, funnel in enumerate(funnels):
        if funnel.closed:
            continue
        curr = funnel.tip
        while True:
            curr = _add(curr, funnel.axis)
            if not _in_bounds(shape, curr):
                break
            j = cell_to_funnel.get(curr)
            if j is not None and funnels[j].axis == funnel.axis:
                nxt[i] = j
                break
    return nxt


def _depths(next_funnel: list[int | None]) -> list[int]:
    memo: dict[int, int] = {}

    def visit(i: int) -> int:
        if i in memo:
            return memo[i]
        j = next_funnel[i]
        memo[i] = 0 if j is None else 1 + visit(j)
        return memo[i]

    return [visit(i) for i in range(len(next_funnel))]


def _simulate(
    shape: tuple[int, int],
    funnels: list[Funnel],
    beads: list[Bead],
    *,
    keep_outside: bool,
    trace: bool = False,
) -> tuple[np.ndarray, list[dict] | None]:
    grid = np.full(shape, BG, dtype=int)
    for funnel in funnels:
        for cell in funnel.wall_cells:
            grid[cell] = GREEN
    for bead in beads:
        assert grid[bead.pos] == BG
        grid[bead.pos] = bead.color

    out = grid.copy()
    next_funnel = _next_funnels(funnels, shape)
    depths = _depths(next_funnel)

    cell_to_funnel: dict[tuple[int, int], int] = {}
    for i, funnel in enumerate(funnels):
        for cell in funnel.interior_map:
            cell_to_funnel[cell] = i

    moving: list[tuple[int, int, int, int, int, int]] = []
    occupied: set[tuple[int, int]] = set()
    index_by_pos = {bead.pos: idx for idx, bead in enumerate(beads)}
    traces = [{"start": bead.pos, "color": bead.color, "funnels": [], "final": bead.pos} for bead in beads]

    for bead in beads:
        fidx = cell_to_funnel.get(bead.pos)
        if fidx is None:
            if keep_outside:
                occupied.add(bead.pos)
            continue
        out[bead.pos] = BG
        k, m = funnels[fidx].interior_map[bead.pos]
        moving.append((depths[fidx], k, bead.pos[0], bead.pos[1], fidx, bead.color))

    moving.sort()

    def stop(idx: int, cell: tuple[int, int]) -> None:
        occupied.add(cell)
        out[cell] = traces[idx]["color"]
        traces[idx]["final"] = cell

    for _depth, _k, _r, _c, start_fidx, color in moving:
        current = next(bead.pos for bead in beads if bead.color == color)
        bead_idx = index_by_pos[current]
        funnel_idx = start_fidx
        while True:
            funnel = funnels[funnel_idx]
            if not traces[bead_idx]["funnels"] or traces[bead_idx]["funnels"][-1] != funnel_idx:
                traces[bead_idx]["funnels"].append(funnel_idx)
            while True:
                k, m = funnel.interior_map[current]
                if k == 1:
                    if funnel.closed:
                        stop(bead_idx, current)
                        break
                    current = funnel.tip
                    break

                next_m = min(max(m, -(k - 2)), k - 2)
                nxt = _combine(funnel.tip, funnel.back, k - 1, funnel.perp, next_m)
                if nxt in occupied:
                    stop(bead_idx, current)
                    break
                current = nxt
            if traces[bead_idx]["final"] != traces[bead_idx]["start"]:
                if traces[bead_idx]["final"] in occupied:
                    break
            if funnel.closed:
                break

            while True:
                nxt = _add(current, funnel.axis)
                if not _in_bounds(shape, nxt):
                    stop(bead_idx, current)
                    break
                if nxt in occupied:
                    stop(bead_idx, current)
                    break
                if grid[nxt] == GREEN:
                    stop(bead_idx, current)
                    break
                downstream = cell_to_funnel.get(nxt)
                if downstream is not None and funnels[downstream].axis == funnel.axis:
                    current = nxt
                    funnel_idx = downstream
                    break
                current = nxt
            else:
                raise AssertionError("unreachable")
            if traces[bead_idx]["final"] in occupied:
                break

    return out, traces if trace else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each bead through green V-funnels, letting open tips relay beads onward and closed tips accumulate them.

    Input structure:
    - A black background contains several green 45-degree V-shaped funnels whose openings face up, down, left, or right.
    - Each funnel is either closed by a green tip pixel or open by a one-cell missing tip gap.
    - Colored single-pixel beads may lie inside funnel interiors or elsewhere on the background.

    Rule:
    - Detect every funnel from its paired diagonal green walls and keep all green pixels unchanged.
    - Beads outside all funnels stay where they are. Beads inside funnels are processed from nearest-to-tip to farthest within downstream funnels first.
    - A controlled bead advances one step toward the tip; when the next row or column narrows against a wall, it is deflected diagonally along that wall toward the tip. At a closed tip or an already stopped bead it stops and stacks. At an open tip it passes through the gap, keeps flying straight along the funnel axis, and if it enters another same-direction funnel interior it becomes controlled by that funnel.

    Key insight:
    - The missing or present tip pixel is the gate: local sliding inside one funnel is only the first stage, and the final resting place is determined by downstream relay, wall deflection, and earlier settled beads.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    funnels = _detect_funnels(g)
    beads = [Bead(pos=(int(r), int(c)), color=int(g[r, c])) for r, c in np.argwhere((g != BG) & (g != GREEN))]
    out, _ = _simulate(g.shape, funnels, beads, keep_outside=True, trace=False)
    return out


def _render_scene(shape: tuple[int, int], funnels: list[Funnel], beads: list[Bead]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for funnel in funnels:
        for cell in funnel.wall_cells:
            grid[cell] = GREEN
    for bead in beads:
        assert grid[bead.pos] == BG
        grid[bead.pos] = bead.color
    return grid


def _rotate_coord(coord: tuple[int, int], shape: tuple[int, int], steps_cw: int) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c = coord
    h, w = shape
    for _ in range(steps_cw % 4):
        r, c = c, h - 1 - r
        h, w = w, h
    return (r, c), (h, w)


def _rotate_open_dir(open_dir: str, steps_cw: int) -> str:
    out = open_dir
    for _ in range(steps_cw % 4):
        out = CW_OPEN_DIR[out]
    return out


def _rotate_scene(shape: tuple[int, int], funnels: list[Funnel], beads: list[Bead], steps_cw: int) -> tuple[tuple[int, int], list[Funnel], list[Bead]]:
    new_shape = shape
    rotated_funnels: list[Funnel] = []
    for funnel in funnels:
        new_tip, new_shape = _rotate_coord(funnel.tip, shape, steps_cw)
        rotated = _build_funnel(_rotate_open_dir(funnel.open_dir, steps_cw), new_tip, funnel.height, funnel.closed, new_shape)
        assert rotated is not None
        rotated_funnels.append(rotated)

    rotated_beads: list[Bead] = []
    for bead in beads:
        pos, new_shape = _rotate_coord(bead.pos, shape, steps_cw)
        rotated_beads.append(Bead(pos=pos, color=bead.color))
    return new_shape, rotated_funnels, rotated_beads


def _canonical_chain(shape: tuple[int, int]) -> tuple[list[Funnel], list[tuple[int, int]]]:
    rows, cols = shape
    chain_len = 3 if random.random() < 0.55 else 2
    heights = [random.randint(3, 4) for _ in range(chain_len)]
    gaps = [random.randint(2, 4) for _ in range(chain_len - 1)]
    last_closed = random.random() < 0.5

    total_axis = heights[0]
    for i in range(1, chain_len):
        total_axis += gaps[i - 1] + heights[i]

    lower_margin = 2 if last_closed else 4
    if total_axis + lower_margin + 2 > rows:
        raise ValueError("chain too tall")

    max_h = max(heights)
    chain_c = random.randint(max_h + 2, cols - max_h - 3)
    mouth_row = random.randint(2, rows - total_axis - lower_margin)

    funnels: list[Funnel] = []
    mouth_rows: list[int] = []
    current_mouth = mouth_row
    for i, height in enumerate(heights):
        tip = (current_mouth + height, chain_c)
        closed = last_closed if i == chain_len - 1 else False
        funnel = _build_funnel("up", tip, height, closed, shape)
        assert funnel is not None
        funnels.append(funnel)
        mouth_rows.append(current_mouth)
        if i < chain_len - 1:
            current_mouth = tip[0] + gaps[i]
    return funnels, mouth_rows


def _place_extra_funnel(shape: tuple[int, int], blocked: set[tuple[int, int]]) -> Funnel | None:
    orientations = ["left", "right", "down"]
    for _ in range(150):
        open_dir = random.choice(orientations)
        height = random.randint(3, 4)
        tip = (random.randint(2, shape[0] - 3), random.randint(2, shape[1] - 3))
        funnel = _build_funnel(open_dir, tip, height, True, shape)
        if funnel is None:
            continue
        cells = set(funnel.wall_cells) | set(funnel.interior_map) | {funnel.tip}
        if _halo(cells, shape, 1) & blocked:
            continue
        return funnel
    return None


def _pick_distinct_levels(height: int, count: int, include_one: bool = False) -> list[int]:
    levels = list(range(1, height + 1))
    if include_one:
        levels.remove(1)
        picked = [1]
        picked.extend(random.sample(levels, count - 1))
        return sorted(picked)
    return sorted(random.sample(levels, count))


def _cell_at(funnel: Funnel, k: int, m: int) -> tuple[int, int]:
    return _combine(funnel.tip, funnel.back, k, funnel.perp, m)


def _make_beads(shape: tuple[int, int], funnels: list[Funnel], mouth_rows: list[int]) -> list[Bead]:
    needed = 2 + (len(funnels) - 1) + 1 + 2
    colors = random.sample(BEAD_COLORS, needed)
    beads: list[Bead] = []

    first = funnels[0]
    side_sign = random.choice([-1, 1])
    beads.append(Bead(_cell_at(first, first.height, side_sign * (first.height - 1)), colors.pop()))
    second_level = 2 if first.height == 3 else random.choice([2, 3])
    beads.append(Bead(_cell_at(first, second_level, 0), colors.pop()))

    for funnel in funnels[1:]:
        if funnel.closed:
            beads.append(Bead(_cell_at(funnel, 1, 0), colors.pop()))
        else:
            level = min(2, funnel.height)
            beads.append(Bead(_cell_at(funnel, level, random.choice([-1, 0, 1]) if level > 1 else 0), colors.pop()))

    extra = funnels[-1]
    if extra.open_dir != "up":
        level = extra.height
        beads.append(Bead(_cell_at(extra, level, random.choice([-(level - 1), level - 1])), colors.pop()))

    occupied = {bead.pos for bead in beads}
    blocked = set()
    for funnel, mouth_row in zip(funnels[: len(mouth_rows)], mouth_rows):
        for rr in range(mouth_row, funnel.tip[0] + 1):
            blocked.add((rr, funnel.tip[1]))
    for funnel in funnels:
        blocked |= set(funnel.wall_cells) | set(funnel.interior_map) | {funnel.tip}
    blocked = _halo(blocked, shape, 1)

    candidates = [
        (r, c)
        for r in range(shape[0])
        for c in range(shape[1])
        if (r, c) not in blocked and (r, c) not in occupied
    ]
    random.shuffle(candidates)
    for pos in candidates[:2]:
        beads.append(Bead(pos, colors.pop()))
    return beads


def _scene_semantics(shape: tuple[int, int], funnels: list[Funnel], beads: list[Bead]) -> bool:
    detected = _detect_funnels(_render_scene(shape, funnels, beads))
    if {_signature(f) for f in detected} != {_signature(f) for f in funnels}:
        return False

    cell_to_funnel = {}
    for i, funnel in enumerate(funnels):
        for cell in funnel.interior_map:
            if cell in cell_to_funnel:
                return False
            cell_to_funnel[cell] = i

    inside = [bead for bead in beads if bead.pos in cell_to_funnel]
    outside = [bead for bead in beads if bead.pos not in cell_to_funnel]
    if not inside or not outside:
        return False
    if not any(not funnel.closed for funnel in funnels):
        return False
    if not any(funnel.closed for funnel in funnels):
        return False
    if not any(funnels[cell_to_funnel[bead.pos]].interior_map[bead.pos][1] != 0 for bead in inside):
        return False

    oracle, traces = _simulate(shape, funnels, beads, keep_outside=True, trace=True)
    if traces is None:
        return False
    if not any(len(t["funnels"]) >= 2 for t in traces):
        return False
    if not any(t["start"] == t["final"] for t in traces if t["start"] not in cell_to_funnel):
        return False
    if not any(t["start"] != t["final"] for t in traces if t["start"] in cell_to_funnel):
        return False
    if np.array_equal(_render_scene(shape, funnels, beads), oracle):
        return False
    return True


def _generate_scene() -> tuple[tuple[int, int], list[Funnel], list[Bead]] | None:
    shape = (random.randint(18, 24), random.randint(18, 24))
    try:
        chain, mouth_rows = _canonical_chain(shape)
    except ValueError:
        return None

    blocked = set()
    for funnel, mouth_row in zip(chain, mouth_rows):
        cells = set(funnel.wall_cells) | set(funnel.interior_map) | {funnel.tip}
        blocked |= _halo(cells, shape, 1)
        for rr in range(mouth_row, funnel.tip[0] + 2):
            if 0 <= rr < shape[0]:
                blocked.add((rr, funnel.tip[1]))

    extra = _place_extra_funnel(shape, blocked)
    if extra is None:
        return None

    funnels = chain + [extra]
    beads = _make_beads(shape, funnels, mouth_rows)

    steps_cw = random.randint(0, 3)
    return _rotate_scene(shape, funnels, beads, steps_cw)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        scene = _generate_scene()
        if scene is None:
            continue
        shape, funnels, beads = scene
        if not _scene_semantics(shape, funnels, beads):
            continue
        grid = _render_scene(shape, funnels, beads)
        oracle, _ = _simulate(shape, funnels, beads, keep_outside=True, trace=False)
        if not np.array_equal(main(grid), oracle):
            continue
        return grid
    raise RuntimeError("failed to generate a valid funnel-relay scene")
