from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
NOISE_COLORS = (1, 3, 5, 6, 7, 8, 9)

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HORIZONTAL = "h"
VERTICAL = "v"
LEFT = "left"
RIGHT = "right"
UP = "up"
DOWN = "down"

GEN_GRID_MIN = 18
GEN_GRID_MAX = 30
GEN_DISTANCE_MIN = 4
GEN_DISTANCE_MAX = 8
GEN_RETRIES = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _block_cells(block: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = block
    return {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}


def _row_band(block: tuple[int, int]) -> tuple[int, int]:
    r, _ = block
    return r, r + 1


def _col_band(block: tuple[int, int]) -> tuple[int, int]:
    _, c = block
    return c, c + 1


def _bands_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _black_components(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLACK or (r, c) in seen:
                continue
            comp: set[tuple[int, int]] = set()
            q = deque([(r, c)])
            seen.add((r, c))
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if int(grid[nr, nc]) != BLACK or (nr, nc) in seen:
                        continue
                    seen.add((nr, nc))
                    q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    yellow_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW)]
    if len(yellow_coords) != 1:
        raise AssertionError("expected exactly one yellow ignition marker")
    yellow = yellow_coords[0]

    red = {tuple(map(int, rc)) for rc in np.argwhere(g == RED)}
    blocks: list[tuple[int, int]] = []
    for comp in _black_components(g):
        rows = sorted(r for r, _ in comp)
        cols = sorted(c for _, c in comp)
        if len(comp) != 4 or rows[-1] - rows[0] != 1 or cols[-1] - cols[0] != 1:
            raise AssertionError("every black component must be a solid 2x2 block")
        blocks.append((rows[0], cols[0]))
    if not blocks:
        raise AssertionError("expected at least one black block")
    blocks.sort()

    initial = []
    for idx, block in enumerate(blocks):
        if any(abs(yellow[0] - r) + abs(yellow[1] - c) == 1 for r, c in _block_cells(block)):
            initial.append(idx)
    if len(initial) != 1:
        raise AssertionError("yellow marker must touch exactly one black block")

    return {"grid": g, "yellow": yellow, "red": red, "blocks": blocks, "initial": initial[0]}


def _sweep_from_block(
    block: tuple[int, int],
    shape: tuple[int, int],
    red: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], dict[str, str]]:
    r, c = block
    swept = set(_block_cells(block))
    stop_reason: dict[str, str] = {}

    for cc in range(c - 1, -1, -1):
        seg = ((r, cc), (r + 1, cc))
        if any(cell in red for cell in seg):
            stop_reason[LEFT] = "red"
            break
        swept.update(seg)
    else:
        stop_reason[LEFT] = "edge"

    for cc in range(c + 2, shape[1]):
        seg = ((r, cc), (r + 1, cc))
        if any(cell in red for cell in seg):
            stop_reason[RIGHT] = "red"
            break
        swept.update(seg)
    else:
        stop_reason[RIGHT] = "edge"

    for rr in range(r - 1, -1, -1):
        seg = ((rr, c), (rr, c + 1))
        if any(cell in red for cell in seg):
            stop_reason[UP] = "red"
            break
        swept.update(seg)
    else:
        stop_reason[UP] = "edge"

    for rr in range(r + 2, shape[0]):
        seg = ((rr, c), (rr, c + 1))
        if any(cell in red for cell in seg):
            stop_reason[DOWN] = "red"
            break
        swept.update(seg)
    else:
        stop_reason[DOWN] = "edge"

    return swept, stop_reason


def _propagate(
    blocks: list[tuple[int, int]],
    shape: tuple[int, int],
    red: set[tuple[int, int]],
    initial: int,
) -> dict[str, object]:
    block_sets = [_block_cells(block) for block in blocks]
    sweeps: list[set[tuple[int, int]]] = []
    stops: list[dict[str, str]] = []
    for block in blocks:
        swept, stop_reason = _sweep_from_block(block, shape, red)
        sweeps.append(swept)
        stops.append(stop_reason)

    adjacency = [set() for _ in blocks]
    for i, swept in enumerate(sweeps):
        for j, cells in enumerate(block_sets):
            if i != j and swept & cells:
                adjacency[i].add(j)

    dist = [-1] * len(blocks)
    dist[initial] = 0
    q = deque([initial])
    while q:
        cur = q.popleft()
        for nxt in sorted(adjacency[cur]):
            if dist[nxt] != -1:
                continue
            dist[nxt] = dist[cur] + 1
            q.append(nxt)

    reachable = {idx for idx, d in enumerate(dist) if d >= 0}
    return {
        "sweeps": sweeps,
        "stops": stops,
        "adjacency": adjacency,
        "distance": dist,
        "reachable": reachable,
    }


def _render_output(
    grid: np.ndarray,
    yellow: tuple[int, int],
    red: set[tuple[int, int]],
    reachable: set[int],
    sweeps: list[set[tuple[int, int]]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[yellow] = BLACK
    burned = set()
    for idx in reachable:
        burned.update(sweeps[idx])
    for r, c in burned:
        if (r, c) not in red:
            out[r, c] = BLACK
    return out


def _analyze(grid: np.ndarray, *, ignore_red: bool = False) -> dict[str, object]:
    scene = _parse_scene(grid)
    red = set() if ignore_red else set(scene["red"])
    propagation = _propagate(scene["blocks"], grid.shape, red, int(scene["initial"]))
    output = _render_output(
        scene["grid"],
        tuple(scene["yellow"]),
        set(scene["red"]),
        set(propagation["reachable"]),
        propagation["sweeps"],
    )
    return {**scene, **propagation, "output": output}


def _shift_block(
    block: tuple[int, int],
    axis: str,
    sign: int,
    distance: int,
    offset: int,
) -> tuple[int, int]:
    r, c = block
    if axis == HORIZONTAL:
        return r + offset, c + sign * distance
    return r + sign * distance, c + offset


def _blocks_are_separated(blocks: list[tuple[int, int]]) -> bool:
    cell_sets = [_block_cells(block) for block in blocks]
    for i, cells_i in enumerate(cell_sets):
        for cells_j in cell_sets[i + 1 :]:
            if cells_i & cells_j:
                return False
            for r0, c0 in cells_i:
                for r1, c1 in cells_j:
                    if abs(r0 - r1) + abs(c0 - c1) == 1:
                        return False
    return True


def _barrier_candidates(
    source: tuple[int, int],
    target: tuple[int, int],
    axis: str,
    sign: int,
) -> list[tuple[int, int]]:
    sr, sc = source
    if axis == HORIZONTAL:
        rows = (sr, sr + 1)
        if sign > 0:
            cols = range(sc + 3, target[1])
        else:
            cols = range(target[1] + 2, sc - 1)
        return [(r, c) for c in cols for r in rows]
    cols = (sc, sc + 1)
    if sign > 0:
        rows = range(sr + 3, target[0])
    else:
        rows = range(target[0] + 2, sr - 1)
    return [(r, c) for r in rows for c in cols]


def _adjacent_block_indices(
    cell: tuple[int, int],
    blocks: list[tuple[int, int]],
) -> list[int]:
    out = []
    for idx, block in enumerate(blocks):
        if any(abs(cell[0] - r) + abs(cell[1] - c) == 1 for r, c in _block_cells(block)):
            out.append(idx)
    return out


def _marker_candidates(
    block: tuple[int, int],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    red: set[tuple[int, int]],
    blocks: list[tuple[int, int]],
    initial_idx: int,
) -> list[tuple[int, int]]:
    r, c = block
    segments = [
        ((r, c - 1), (r + 1, c - 1)),
        ((r, c + 2), (r + 1, c + 2)),
        ((r - 1, c), (r - 1, c + 1)),
        ((r + 2, c), (r + 2, c + 1)),
    ]
    candidates = []
    for seg in segments:
        if not all(_in_bounds(shape, rr, cc) for rr, cc in seg):
            continue
        if any(cell in red for cell in seg):
            continue
        for cell in seg:
            if cell in occupied:
                continue
            if _adjacent_block_indices(cell, blocks) != [initial_idx]:
                continue
            candidates.append(cell)
    return sorted(set(candidates))


def _noise_background(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    palette_size = int(rng.integers(3, min(6, len(NOISE_COLORS)) + 1))
    palette = rng.choice(NOISE_COLORS, size=palette_size, replace=False).astype(int)
    base_weight = float(rng.uniform(0.55, 0.72))
    rest = rng.random(palette_size - 1)
    rest /= rest.sum()
    probs = np.concatenate(([base_weight], rest * (1.0 - base_weight)))
    return rng.choice(palette, size=shape, p=probs).astype(int)


def _build_candidate(rng: np.random.Generator) -> np.ndarray | None:
    axis1 = HORIZONTAL if int(rng.integers(0, 2)) == 0 else VERTICAL
    axis2 = VERTICAL if axis1 == HORIZONTAL else HORIZONTAL
    sign1 = -1 if int(rng.integers(0, 2)) == 0 else 1
    sign2 = -1 if int(rng.integers(0, 2)) == 0 else 1

    d1, d2, du1, du2 = [int(rng.integers(GEN_DISTANCE_MIN, GEN_DISTANCE_MAX + 1)) for _ in range(4)]
    off1, off2, offu1, offu2 = [int(rng.integers(-1, 2)) for _ in range(4)]

    b0 = (0, 0)
    b1 = _shift_block(b0, axis1, sign1, d1, off1)
    b2 = _shift_block(b1, axis2, sign2, d2, off2)
    u1 = _shift_block(b0, axis1, -sign1, du1, offu1)
    u2 = _shift_block(b2, axis2, sign2, du2, offu2)
    rel_blocks = [b0, b1, b2, u1, u2]

    if not _blocks_are_separated(rel_blocks):
        return None

    min_r = min(r for r, _ in rel_blocks)
    max_r = max(r for r, _ in rel_blocks)
    min_c = min(c for _, c in rel_blocks)
    max_c = max(c for _, c in rel_blocks)

    need_h = max(GEN_GRID_MIN, max_r - min_r + 4)
    need_w = max(GEN_GRID_MIN, max_c - min_c + 4)
    if need_h > GEN_GRID_MAX or need_w > GEN_GRID_MAX:
        return None

    h = int(rng.integers(need_h, GEN_GRID_MAX + 1))
    w = int(rng.integers(need_w, GEN_GRID_MAX + 1))
    r_shift_lo, r_shift_hi = 1 - min_r, h - 3 - max_r
    c_shift_lo, c_shift_hi = 1 - min_c, w - 3 - max_c
    if r_shift_lo > r_shift_hi or c_shift_lo > c_shift_hi:
        return None

    dr = int(rng.integers(r_shift_lo, r_shift_hi + 1))
    dc = int(rng.integers(c_shift_lo, c_shift_hi + 1))
    blocks = [(r + dr, c + dc) for r, c in rel_blocks]
    if not _blocks_are_separated(blocks):
        return None

    start, mid, end, blocked_a, blocked_b = blocks
    barrier1_choices = _barrier_candidates(start, blocked_a, axis1, -sign1)
    barrier2_choices = _barrier_candidates(end, blocked_b, axis2, sign2)
    if not barrier1_choices or not barrier2_choices:
        return None

    red = {
        barrier1_choices[int(rng.integers(0, len(barrier1_choices)))],
        barrier2_choices[int(rng.integers(0, len(barrier2_choices)))],
    }

    occupied = set(red)
    for block in blocks:
        occupied.update(_block_cells(block))

    marker_options = _marker_candidates(start, (h, w), occupied, red, blocks, 0)
    if not marker_options:
        return None
    marker = marker_options[int(rng.integers(0, len(marker_options)))]

    grid = _noise_background(rng, (h, w))
    for rr, cc in red:
        grid[rr, cc] = RED
    grid[marker] = YELLOW
    for block in blocks:
        for rr, cc in _block_cells(block):
            grid[rr, cc] = BLACK

    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        actual = _analyze(grid)
        no_red = _analyze(grid, ignore_red=True)
    except AssertionError:
        return False

    blocks = actual["blocks"]
    reachable = set(actual["reachable"])
    distances = list(actual["distance"])
    if len(blocks) != 5 or len(reachable) != 3:
        return False
    if max(distances) < 2:
        return False
    one_hop = {idx for idx, d in enumerate(distances) if 0 <= d <= 1}
    if reachable == one_hop:
        return False
    if len(no_red["reachable"]) <= len(reachable):
        return False
    if set(no_red["reachable"]) == reachable:
        return False

    if not any(
        reason == "red"
        for idx in reachable
        for reason in actual["stops"][idx].values()
    ):
        return False

    blocked_due_to_red = set(no_red["reachable"]) - reachable
    if not blocked_due_to_red:
        return False
    if not any(
        _bands_overlap(_row_band(blocks[src]), _row_band(blocks[dst]))
        or _bands_overlap(_col_band(blocks[src]), _col_band(blocks[dst]))
        for src in reachable
        for dst in blocked_due_to_red
    ):
        return False

    out = actual["output"]
    changed_noise = int(
        np.count_nonzero(
            (grid != BLACK) & (grid != RED) & (grid != YELLOW) & (out == BLACK)
        )
    )
    if changed_noise < 12:
        return False
    if not np.all(out[grid == RED] == RED):
        return False
    if np.count_nonzero(grid == YELLOW) != 1 or np.count_nonzero(out == YELLOW) != 0:
        return False
    if np.array_equal(out, grid):
        return False
    if not np.array_equal(main(grid), out):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Ignite the yellow-tagged 2x2 black block and let red-stopped width-2 cross rays chain-burn only the reachable black blocks.

    Input structure:
    - The grid is filled with multicolor noise and contains several isolated solid 2x2 black blocks.
    - Red pixels act as firebreak cells that can lie inside the future sweep lanes.
    - Exactly one yellow pixel is 4-neighbor-adjacent to one black block and marks the initial ignition block.

    Rule:
    - Find the unique black block touched by the yellow marker and ignite it first.
    - Every ignited 2x2 block sweeps outward along its two rows to the left and right, and along its two columns upward and downward; each step examines the next 2-cell segment in that direction and stops before it if either cell is red.
    - Every non-red cell visited by these width-2 rays becomes black, and any still-unignited black block touched on any of its four cells is ignited and emits its own rays.
    - Iterate this contact ignition to a fixpoint, erase the yellow marker to black, keep all red cells unchanged, and leave untouched noise and untouched black blocks as they were.

    Key insight:
    - The result is determined by reachability in a graph of 2x2 black blocks whose width-2 row and column rays are clipped by red cells, so both the relay chain and the red stoppages are essential.
    """
    analysis = _analyze(np.asarray(grid, dtype=int))
    return analysis["output"]


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _build_candidate(rng)
        if grid is not None and _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid point-ignition chain input")
