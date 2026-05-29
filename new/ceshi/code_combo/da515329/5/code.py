# Combo variant of da515329 (index 5)
# combo_spec: D2/1+D18/0
# Name: 2像素间距 + 固定段长

import random

import numpy as np

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_PITCH = 3
_SPACING = 2
_COLOR = 8


# Design analysis for the fused variant
# a) Concrete behavior:
#    Original da515329 used a 2/1 step alternation to create an expanding spiral:
#    after a turn the path advanced only 1 cell, then straight stretches used 2,
#    so every later rectangle side became longer by (line width + gap) = 2.
#    Here that growth is removed. A tip first launches outward by 3 cells, then
#    every later move is also exactly 3 cells. Because 3 = 1 painted cell + 2-cell
#    corridor, the agent now walks on a coarse lattice and leaves visible 2-pixel
#    hallways. The repeated equal macro-steps create a Greek-key / "hui" meander
#    instead of a continuously expanding spiral.
# b) Complexity calibration:
#    Only two spatial parameters change at once: the corridor widens from 1 to 2,
#    and the segment length stops growing. The source object, launch sites, and
#    clockwise turn preference stay recognizable, so the variant is still traceable
#    from a few ARC examples while remaining visually distinct from the original.
# c) Implicit assumptions removed from the original solver:
#    - The old parity cleanup depended on 1-pixel spacing; it is invalid here.
#    - The old p-state ("1 right after a turn, else 2") encoded growth and must go.
#    - If source pixels participate in the new 2-pixel spacing check, the wider
#      corridor blocks the launch immediately. The fused rule therefore treats the
#      original cross as a seed/anchor and enforces spacing only against generated
#      path cells.
# d) Placement legality:
#    The generator keeps every arm tip at least 4 cells from the border so the
#    mandatory 3-cell launch fits. Candidate inputs are accepted only if the solver
#    yields enough generated segments and a non-degenerate fill ratio.
# e) Visual observability:
#    Across samples, humans can see two stable cues: parallel path lanes are always
#    separated by two black cells, and every newly drawn straight stroke has the
#    same 3-cell macro length. This is directly visible without relying on color
#    value arithmetic or hidden ordering.
# f) Visual cleanliness:
#    The solver uses one-cell-wide teal paths on a 3-cell lattice, so overlapping
#    layers are avoided. The generator filters out nearly empty placements.
# g) Generator/solver contract:
#    main() assumes a single 4-armed teal cross with equal arm lengths and an empty
#    background. generate_input() constructs exactly that object and validates the
#    resulting fused pattern before returning.


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _extract_cross(grid: np.ndarray):
    h, w = grid.shape
    row_counts = np.sum(grid == _COLOR, axis=1)
    col_counts = np.sum(grid == _COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    arms = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells = []
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == _COLOR:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((d, cells))
    arm_len = max((len(cells) for _, cells in arms), default=0)
    return cr, cc, arms, arm_len


def _solve_with_stats(grid: np.ndarray):
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    cr, cc, arms, arm_len = _extract_cross(grid)
    source = set(map(tuple, np.argwhere(grid == _COLOR)))
    path = set()

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos, d, k=1):
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    def seg_cells(cur, d):
        return [add(cur, d, t) for t in range(1, _PITCH + 1)]

    def spacing_ok(cells, prev) -> bool:
        new_cells = set(cells)
        for r, c in cells:
            for dr, dc in _DIRS_URDL:
                for dist in range(1, _SPACING + 1):
                    nr, nc = r + dr * dist, c + dc * dist
                    if not inb(nr, nc):
                        break
                    if (nr, nc) == prev or (nr, nc) in new_cells or (nr, nc) in source:
                        continue
                    if (nr, nc) in path:
                        return False
        return True

    def can_move(cur, d) -> bool:
        cells = seg_cells(cur, d)
        if not all(inb(r, c) and out[r, c] == 0 for r, c in cells):
            return False
        return spacing_ok(cells, cur)

    def do_move(agent, d) -> None:
        cells = seg_cells(agent["cur"], d)
        for r, c in cells:
            out[r, c] = _COLOR
            path.add((r, c))
        agent["cur"] = cells[-1]
        agent["dir"] = d

    stats = {
        "arm_len": arm_len,
        "moves": 0,
        "turn_right": 0,
        "turn_left": 0,
        "straight": 0,
        "launches": 0,
        "generated_cells": 0,
    }

    agents = []
    for outward, cells in arms:
        tip = cells[-1]
        launch_cells = seg_cells(tip, outward)
        cur = tip
        if all(inb(r, c) and out[r, c] == 0 for r, c in launch_cells):
            for r, c in launch_cells:
                out[r, c] = _COLOR
                path.add((r, c))
            cur = launch_cells[-1]
            stats["moves"] += 1
            stats["straight"] += 1
            stats["launches"] += 1
        agents.append({"cur": cur, "dir": _right(outward), "done": False})

    guard = 0
    changed = True
    while changed and guard < 20000:
        guard += 1
        changed = False
        for agent in agents:
            if agent["done"]:
                continue
            cur = agent["cur"]
            old_dir = agent["dir"]
            dirs = [_right(old_dir), old_dir]
            if not is_border(*cur):
                dirs.append(_left(old_dir))
            moved = False
            for nd in dirs:
                if can_move(cur, nd):
                    do_move(agent, nd)
                    stats["moves"] += 1
                    if nd == _right(old_dir):
                        stats["turn_right"] += 1
                    elif nd == _left(old_dir):
                        stats["turn_left"] += 1
                    else:
                        stats["straight"] += 1
                    changed = True
                    moved = True
                    break
            if not moved:
                agent["done"] = True
        if all(agent["done"] for agent in agents):
            break

    stats["generated_cells"] = len(path)
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch four wide-gap clockwise meanders from the tips of a teal cross.

    Input structure:
    - bg = 0 (black). A single teal (8) plus sign with a hollow center.
    - The four equal-length arm tips are the only launch sites.

    Rule:
    - Each tip first extends 3 cells outward, then continues with fixed 3-cell
      segments while following the priority right turn > straight > left.
    - Every new segment must keep a 2-pixel orthogonal corridor from previously
      generated path cells; the original cross acts only as the seed anchor.
    - Because every macro-step has the same length, the four paths form repeated
      square-meander / "hui" motifs instead of an ever-growing spiral.

    Key insight:
    - Reinterpret the drawing on a coarse lattice whose edge length is 3 pixels.
      The 3-pixel edge simultaneously enforces the wider spacing and removes the
      old layer-by-layer growth, so the visual switches from dense spiral shells
      to equal-step Greek-key patterns.
    """

    out, _ = _solve_with_stats(grid)
    return out


# --------------- generator (cx) ---------------

ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
SIZE_PAD_MIN, SIZE_PAD_MAX = 8, 12
TIP_GAP_MIN = 4
FILL_LO, FILL_HI = 0.14, 0.34
MIN_MOVES = 12
MIN_RIGHT_TURNS = 4
MAX_OUTER_RETRIES = 400


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = _COLOR
    grid[cr + offsets, cc] = _COLOR
    grid[cr, cc - offsets] = _COLOR
    grid[cr, cc + offsets] = _COLOR
    return grid


def _candidate_sizes(arm: int):
    lo = max(12, 4 * arm + SIZE_PAD_MIN)
    hi = max(lo, 4 * arm + SIZE_PAD_MAX)
    sizes = list(range(lo, hi + 1, 2))
    random.shuffle(sizes)
    return sizes


def _candidate_centers(size: int, arm: int):
    lo = arm + TIP_GAP_MIN
    hi = size - arm - TIP_GAP_MIN - 1
    if lo > hi:
        return []
    centers = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    random.shuffle(centers)
    return centers


def _is_good_sample(out: np.ndarray, stats: dict) -> bool:
    fill = float(np.mean(out == _COLOR))
    if not (FILL_LO <= fill <= FILL_HI):
        return False
    if stats["moves"] < MIN_MOVES:
        return False
    if stats["turn_right"] < MIN_RIGHT_TURNS:
        return False
    if stats["generated_cells"] <= 4 * stats["arm_len"]:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    for size in _candidate_sizes(arm):
        for cr, cc in _candidate_centers(size, arm):
            inp = _build_input(arm, size, cr, cc)
            try:
                out, stats = _solve_with_stats(inp)
            except Exception:
                continue
            if _is_good_sample(out, stats) and np.array_equal(main(inp), out):
                return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
