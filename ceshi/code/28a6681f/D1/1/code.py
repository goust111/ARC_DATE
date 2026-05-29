# Variant of 28a6681f
# Dimension: D1 — 流动路径规则 (path)
# Change: gravity + flow——水像素优先垂直下落，被阻挡时通过唯一开放侧的对角线滑动，支撑后沿水平方向滑向最近可落点... → 通道跟随式流动

from __future__ import annotations

from collections import deque
import random

import numpy as np

# Design analysis for the D1 variant
# a) Concrete behavior and trace
#    - Original 28a6681f: blue water was a set of independent 1x1 pixels in open space.
#      Each pixel repeatedly preferred vertical falling, then a unique diagonal slip,
#      then a supported horizontal slide toward a drop.
#    - This variant replaces that open-space physics with a pipe-like corridor.
#      The moving water is a single rigid 2x2 blue plug sitting inside a 0-colored
#      channel. All non-zero, non-water pixels are static walls. The channel bends,
#      but its traversable width is controlled by the enclosing walls.
#    - Small trace:
#      Original: if a blue pixel is above a basin, it simply falls downward until it
#      finds support; turning left or right only happens as a consequence of gravity.
#      Variant: if a 2x2 water plug starts at the end of a corridor that first goes
#      left, then down, then left again, the plug follows those turns even when there
#      is no "deepest" basin below. If a 1-cell neck appears later, the 2x2 plug
#      stops at the last 2x2 placement before the neck instead of squeezing through.
#    - Flow impact:
#      Input construction changes the most: the generator must now build enclosed
#      corridors rather than open cups. Solver logic changes from iterative local
#      gravity to rigid-body path tracing on legal 2x2 placements. Output generation
#      is still pure relocation with wall pixels unchanged and water count conserved.
#
# b) Complexity calibration
#    - The width-constrained path rule is already the new concept, so the plug size is
#      fixed at 2x2 instead of randomizing many sizes. This keeps the width threshold
#      visible and avoids hidden parameters that would be hard to infer from 2-4 ARC
#      examples.
#    - The placement graph is restricted to a single non-branching corridor component
#      from the source. That prevents ambiguous branch-choice logic from leaking in as
#      another dimension change.
#    - A visible 1-cell chokepoint is present in many samples, but not all. This gives
#      repeated evidence for the width rule while still allowing some samples where the
#      plug reaches the corridor end.
#
# c) Implicit-assumption audit against the old solver
#    - Old input asserts "water must be supported from below" and "cannot have bg on
#      both horizontal sides" depended on gravity semantics. Those assumptions are now
#      invalid and removed.
#    - Old settle() assumed motion decisions could be made from immediate local cells:
#      down, diagonal, then horizontal. That no longer holds because the new rule needs
#      a global corridor view and a fixed 2x2 footprint check.
#    - Old processing iterated over many independent water pixels. That would be wrong
#      here because independent 1x1 motion would let water leak through 1-cell necks.
#      The new solver therefore treats the full water component as one rigid square.
#    - Old output check only required support on left/below/right. That is irrelevant
#      in a corridor. The new output check instead preserves "solid 2x2 plug placed on
#      a legal corridor position."
#
# d) Placement legality guarantees in the generator
#    - The canvas starts as walls everywhere, so the only 0-cells are deliberate
#      corridor cells. This avoids accidental open-space motion.
#    - A template path is drawn on the 2x2 top-left placement grid and carved into the
#      wall canvas as the union of 2x2 windows. A sample is accepted only if the legal
#      2x2 placements recovered from the rendered grid exactly match the intended
#      reachable prefix.
#    - When a chokepoint is added, it is inserted on a straight segment so a single
#      restored wall pixel produces a clear visible neck without damaging the upstream
#      source placement.
#
# e) Visual observability
#    - The key visible cue is geometric: the same 2x2 blue plug moves along a winding
#      0-channel bounded by colored walls and halts at a 1-cell neck. No rule depends
#      on hidden ordering of color ids.
#    - Random symmetry transforms vary whether the plug moves mostly left, right, up,
#      or down, making it obvious that gravity is no longer the control law.
#
# f) Visual cleanliness
#    - There is only one moving object, one corridor, and 2-3 broad wall color zones.
#      This keeps the grid legible. Decorative variety comes from path shape and wall
#      zoning, not from extra overlapping symbols.
#
# g) Color-rule audit
#    - Colors do not encode branching or ranking. Blue (1) is always the moving water,
#      0 is always the corridor interior, and all other colors are equivalent walls.
#
# h) New solver assumptions and how generate_input() enforces them
#    - Assumption: there is exactly one solid 2x2 water component.
#      Generator: always places one 2x2 blue plug.
#    - Assumption: the source lies at one endpoint of the reachable placement path.
#      Generator: uses an endpoint of the transformed template as the source.
#    - Assumption: the reachable legal-placement component is a simple path.
#      Generator: uses pre-vetted template paths and rejects any rendered sample whose
#      recovered placement graph is not a simple path.

BG = 0
WATER = 1
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDE = 2
H = 10
W = 10


def _water_square(grid: np.ndarray) -> tuple[int, int, int]:
    pts = np.argwhere(grid == int(WATER))
    assert len(pts) > 0, "missing water"
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w == SIDE, "water must be one solid 2x2 plug"
    assert len(pts) == int(SIDE * SIDE), "water plug must not have holes"
    assert bool((grid[r0 : r0 + SIDE, c0 : c0 + SIDE] == int(WATER)).all()), "water plug must be solid"
    return r0, c0, SIDE


def _feasible_placements(grid: np.ndarray, side: int) -> np.ndarray:
    free = (grid == int(BG)) | (grid == int(WATER))
    h2 = int(grid.shape[0] - side + 1)
    w2 = int(grid.shape[1] - side + 1)
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(free[r : r + side, c : c + side].all())
    return feasible


def _neighbors(pos: tuple[int, int], feasible: np.ndarray) -> list[tuple[int, int]]:
    r, c = pos
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < feasible.shape[0] and 0 <= cc < feasible.shape[1] and bool(feasible[rr, cc]):
            out.append((rr, cc))
    return out


def _reachable_component(feasible: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in _neighbors(cur, feasible):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def _follow_corridor(
    feasible: np.ndarray,
    start: tuple[int, int],
    check_input: bool,
) -> list[tuple[int, int]]:
    comp = _reachable_component(feasible, start)
    deg = {pos: len([nxt for nxt in _neighbors(pos, feasible) if nxt in comp]) for pos in comp}
    if check_input:
        assert len(comp) >= 2, "corridor must allow visible movement"
        assert all(d <= 2 for d in deg.values()), "reachable legal placements must form a simple corridor"
        assert deg[start] == 1, "source water must start at a corridor endpoint"

    path = [start]
    seen = {start}
    prev: tuple[int, int] | None = None
    cur = start
    for _ in range(feasible.size):
        forward = [nxt for nxt in _neighbors(cur, feasible) if nxt in comp and nxt != prev and nxt not in seen]
        if not forward:
            break
        assert len(forward) == 1, f"ambiguous corridor branch: {forward}"
        nxt = forward[0]
        path.append(nxt)
        seen.add(nxt)
        prev, cur = cur, nxt
    if check_input:
        assert len(path) == len(comp), "corridor component must be a single non-branching path"
    return path


def _trace_state(
    grid: np.ndarray,
    check_input: bool,
) -> tuple[int, tuple[int, int], list[tuple[int, int]], np.ndarray]:
    sr, sc, side = _water_square(grid)
    feasible = _feasible_placements(grid, side)
    start = (int(sr), int(sc))
    assert bool(feasible[start]), "source plug must occupy a legal corridor placement"
    path = _follow_corridor(feasible, start, check_input=check_input)
    return side, start, path, feasible


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        d1 = (int(path[i][0] - path[i - 1][0]), int(path[i][1] - path[i - 1][1]))
        d2 = (int(path[i + 1][0] - path[i][0]), int(path[i + 1][1] - path[i][1]))
        if d1 != d2:
            turns += 1
    return turns


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Move a compact water plug through a wall-bounded corridor until the corridor ends or narrows.

    Input structure:
    - bg = 0. Water pixels = color 1, but they appear as one solid 2x2 plug.
    - Other colors are fixed wall pixels; together they enclose a single winding channel of bg cells.
    - The source plug starts at one endpoint of the reachable corridor. Downstream sections may bend and may later narrow.

    Rule:
    - Treat the blue water component as a rigid 2x2 body, not as independent falling pixels.
    - A move is legal only if the full 2x2 footprint fits inside corridor cells (bg or current water).
    - Starting from the source placement, follow the unique sequence of legal 4-neighbor placements along the corridor.
    - Relocate the plug to the last reachable placement before the corridor terminates or becomes too narrow.

    Key insight: Motion is governed by corridor topology and width, not gravity; the plug can follow turns, but a 1-cell chokepoint blocks it.
    """
    g = np.array(grid, dtype=int).copy()
    side, start, path, _feasible = _trace_state(g, check_input=check_input)
    end_r, end_c = path[-1]

    out = g.copy()
    sr, sc = start
    out[sr : sr + side, sc : sc + side] = int(BG)
    out[end_r : end_r + side, end_c : end_c + side] = int(WATER)

    if check_output:
        rr, cc, side2 = _water_square(out)
        assert side2 == side
        feasible_out = _feasible_placements(out, side)
        assert bool(feasible_out[rr, cc]), "output plug must stay inside the corridor"

    return out


_BASE_PATHS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((1, 7), (2, 7), (3, 7), (3, 6), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (6, 3)),
    ((1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 5), (3, 6), (4, 6), (5, 6), (5, 5), (5, 4), (6, 4)),
    ((2, 7), (2, 6), (2, 5), (3, 5), (4, 5), (4, 4), (4, 3), (5, 3), (6, 3), (6, 2), (6, 1)),
)


def _transform_point(pos: tuple[int, int], kind: int, size: int) -> tuple[int, int]:
    r, c = pos
    last = int(size - 1)
    if kind == 0:
        return int(r), int(c)
    if kind == 1:
        return int(c), int(last - r)
    if kind == 2:
        return int(last - r), int(last - c)
    if kind == 3:
        return int(last - c), int(r)
    if kind == 4:
        return int(r), int(last - c)
    if kind == 5:
        return int(last - r), int(c)
    if kind == 6:
        return int(c), int(r)
    if kind == 7:
        return int(last - c), int(last - r)
    raise ValueError(kind)


def _transform_path(path: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    size = H - SIDE + 1
    kind = random.randrange(8)
    out = [_transform_point(pos, kind, size) for pos in path]
    if random.random() < 0.5:
        out.reverse()
    return out


def _wall_canvas(colors: list[int]) -> np.ndarray:
    canvas = np.empty((H, W), dtype=int)
    if len(colors) == 2:
        split = 5
        canvas[:, :split] = int(colors[0])
        canvas[:, split:] = int(colors[1])
    else:
        canvas[:, :3] = int(colors[0])
        canvas[:, 3:7] = int(colors[1])
        canvas[:, 7:] = int(colors[2])
    return canvas


def _carve_corridor(grid: np.ndarray, path: list[tuple[int, int]]) -> None:
    for r, c in path:
        grid[r : r + SIDE, c : c + SIDE] = int(BG)


def _choke_candidates(path: list[tuple[int, int]]) -> list[int]:
    steps = [(int(path[i + 1][0] - path[i][0]), int(path[i + 1][1] - path[i][1])) for i in range(len(path) - 1)]
    candidates: list[int] = []
    for i in range(2, len(path) - 3):
        if steps[i] == steps[i - 1] or steps[i] == steps[i + 1]:
            candidates.append(i)
    return candidates


def _insert_chokepoint(grid: np.ndarray, walls: np.ndarray, path: list[tuple[int, int]], edge_idx: int) -> None:
    r0, c0 = path[edge_idx]
    r1, c1 = path[edge_idx + 1]
    side_pick = random.choice([0, SIDE - 1])

    if r1 == r0 and c1 == c0 + 1:
        rr = int(r0 + side_pick)
        cc = int(c0 + SIDE)
    elif r1 == r0 and c1 == c0 - 1:
        rr = int(r0 + side_pick)
        cc = int(c0 - 1)
    elif r1 == r0 + 1 and c1 == c0:
        rr = int(r0 + SIDE)
        cc = int(c0 + side_pick)
    elif r1 == r0 - 1 and c1 == c0:
        rr = int(r0 - 1)
        cc = int(c0 + side_pick)
    else:
        raise AssertionError("path must move in 4-neighbor steps")

    grid[rr, cc] = int(walls[rr, cc])


def _expected_stop(path: list[tuple[int, int]], choke_idx: int | None) -> tuple[int, int]:
    if choke_idx is None:
        return path[-1]
    return path[choke_idx]


def _generate_case(return_meta: bool = False):
    for _ in range(400):
        path = _transform_path(random.choice(_BASE_PATHS))
        want_choke = random.random() < 0.7
        colors = random.sample([2, 3, 4, 5, 6, 7, 8, 9], k=random.choice([2, 3]))

        walls = _wall_canvas(colors)
        grid = walls.copy()
        _carve_corridor(grid, path)

        choke_idx: int | None = None
        if want_choke:
            candidates = _choke_candidates(path)
            if not candidates:
                continue
            choke_idx = random.choice(candidates)
            _insert_chokepoint(grid, walls, path, choke_idx)

        sr, sc = path[0]
        grid[sr : sr + SIDE, sc : sc + SIDE] = int(WATER)

        try:
            _side, _start, reachable, _feasible = _trace_state(grid, check_input=True)
            out = main(grid, check_input=True, check_output=True)
        except AssertionError:
            continue

        expected_end = _expected_stop(path, choke_idx)
        if reachable[-1] != expected_end:
            continue
        if choke_idx is None and len(reachable) != len(path):
            continue
        if choke_idx is not None and len(reachable) != choke_idx + 1:
            continue
        if len(reachable) < 5 or _turn_count(reachable) < 2:
            continue
        if np.array_equal(out, grid):
            continue

        meta = {
            "has_choke": choke_idx is not None,
            "path_len": len(path),
            "reachable_len": len(reachable),
            "blocked_tail": int(len(path) - len(reachable)),
            "turns": _turn_count(reachable),
            "start": path[0],
            "end": reachable[-1],
        }
        if return_meta:
            return grid, meta
        return grid
    raise RuntimeError("failed to generate a valid corridor sample")


def generate_input() -> np.ndarray:
    return _generate_case(return_meta=False)
