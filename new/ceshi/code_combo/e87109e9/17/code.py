# Combo variant of e87109e9 (index 17)
# combo_spec: D8/1+D12/2
# Name: 障碍物生长加高密度

# Design analysis for combo D8/1 + D12/2
# 1. Concrete behavior and trace
#    - Original task: the 2x2 or 3x3 teal source emits four cardinal beams. A beam
#      checks the next translated source-footprint; if that footprint overlaps one
#      obstacle color, the legend decides a 90-degree turn, but obstacles stay fixed.
#    - New task: after every successful hit, the struck obstacle grows by one strip
#      toward the incoming beam. Example with a north-going beam:
#        * the next footprint touches the obstacle's north-east corner;
#        * the legend says "turn left", so the beam will head west;
#        * before that westward continuation matters for later beams, the obstacle
#          grows one row southward (toward the source / incoming side);
#        * that new row may erase previously painted teal trail cells and can alter
#          the future path of an east/south/west beam.
#    - D8/1 contributes 6-8 initial obstacle components packed into the central
#      corridors, so repeated growth can visibly thicken several blocks and cause
#      dynamic blockages rather than isolated one-off turns.
#
# 2. Complexity calibration
#    - If growth only changed obstacle area but never changed any beam path, the
#      fusion would be too weak. The generator therefore rejects samples unless the
#      dynamic solver and a static "original-style" solver disagree on teal pixels.
#    - If every obstacle had its own color, the 6-8 component regime would exceed the
#      4-entry legend. So colors are reused, but each hit still resolves a *single*
#      connected component of that color. This keeps the legend small and inferable.
#    - The generator uses compact corner-anchored rectangles. They are dense enough to
#      create a maze-like center, but still visually separable in the input.
#
# 3. Broken original assumptions and how they are repaired
#    - Original assumption: one obstacle color roughly meant one obstacle. Broken:
#      6-8 components now reuse the same 4 legend colors. The solver must identify
#      the connected component actually touched by the next beam footprint.
#    - Original assumption: beam state depended only on (r, c, d). Broken: growth
#      mutates the grid, so revisiting a geometric state can be legal after a hit.
#      The solver therefore uses a step budget instead of a pure cycle assertion.
#    - Original assumption: beams were independent. Broken: obstacle growth makes the
#      order observable. The new rule is fixed and explicit: process beams in the
#      order up, right, down, left. The generator constructs samples for that order.
#    - Original assumption: obstacle cells never overwrite painted trail cells.
#      Broken: growth must be able to block corridors, so it overwrites teal trail
#      cells but never overwrites a different obstacle color.
#
# 4. Placement legality and visual cleanliness
#    - Input-time obstacles are placed only on currently untouched background cells in
#      the generator's sequential simulation, so the final replay from the clean input
#      matches the constructed dynamic trace.
#    - Obstacles are corner-anchored relative to the next collision footprint. This
#      guarantees the collision happens at the intended step while still leaving an
#      escape route in the turned direction after growth.
#    - The generator rejects any sample where the source square is damaged, where the
#      initial obstacle component count is not 6-8, or where the dynamic path does not
#      visibly differ from the static original-style path.

from collections import deque
import random

import numpy as np

from arc_utils import bbox_coords, components_mask

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
LEGEND_BG = 5
TEAL = 8

GRID_H_RANGE = (26, 26)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]

TEAL_SZ2_PROB = 1.0
SOURCE_MARGIN = 8
MIN_STRAIGHT_BEFORE_PLACE = 2
OBS_FRONT_RANGE = (2, 3)
OBS_SIDE_RANGE = (2, 3)
MAX_RETRIES = 300


def _rot(d: int, turn: str) -> int:
    return (d - 1) % 4 if turn == "L" else (d + 1) % 4


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    turns: dict[int, str] = {}
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        interior = legend[1 : legend_h - 1, a + 1 : b]
        colors = set(map(int, np.unique(interior)))
        colors.discard(bg)
        colors.discard(0)
        assert len(colors) == 1, "each legend cell must encode exactly one color"
        color = int(next(iter(colors)))
        xs = np.where(interior == color)[1]
        x = int(xs[0])
        turns[color] = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
    assert len(turns) == 4, "legend must encode four turn colors"
    return turns


def _find_teal_square(grid: np.ndarray, teal: int = TEAL) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for cells in components_mask(grid == teal, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh != ww:
            continue
        if int(np.sum(grid[r0 : r1 + 1, c0 : c1 + 1] == teal)) != hh * ww:
            continue
        candidates.append((int(r0), int(c0), int(hh)))
    assert len(candidates) == 1, "main grid must contain exactly one solid teal square"
    return candidates[0]


def _in_bounds(shape: tuple[int, int], r: int, c: int, sz: int) -> bool:
    return 0 <= r and 0 <= c and r + sz - 1 < shape[0] and c + sz - 1 < shape[1]


def _paint_block(grid: np.ndarray, r: int, c: int, bg: int) -> None:
    view = grid[r : r + _paint_block.sz, c : c + _paint_block.sz]
    view[view == bg] = TEAL


_paint_block.sz = 0


def _component_mask(grid: np.ndarray, color: int, seeds: list[tuple[int, int]]) -> np.ndarray:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r, c in seeds:
        if not seen[r, c]:
            seen[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and grid[nr, nc] == color:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _obstacle_info(grid: np.ndarray, nr: int, nc: int, sz: int, bg: int) -> tuple[int, np.ndarray] | None:
    blk = grid[nr : nr + sz, nc : nc + sz]
    colors = set(map(int, np.unique(blk)))
    colors.discard(bg)
    colors.discard(TEAL)
    if not colors:
        return None
    assert len(colors) == 1, "beam footprint must not hit mixed obstacle colors"
    color = int(next(iter(colors)))
    local = np.argwhere(blk == color)
    seeds = [(int(nr + rr), int(nc + cc)) for rr, cc in local]
    comp = _component_mask(grid, color, seeds)
    for r, c in seeds:
        assert comp[r, c], "same-color cells in the footprint must belong to one component"
    return color, comp


def _grow_obstacle(grid: np.ndarray, color: int, comp: np.ndarray, toward: int, bg: int) -> int:
    coords = np.argwhere(comp)
    assert coords.size, "grown obstacle component must be non-empty"
    written = 0

    if toward == _N:
        rr = int(coords[:, 0].min() - 1)
        cols = sorted(set(map(int, coords[:, 1])))
        targets = [(rr, c) for c in cols]
    elif toward == _S:
        rr = int(coords[:, 0].max() + 1)
        cols = sorted(set(map(int, coords[:, 1])))
        targets = [(rr, c) for c in cols]
    elif toward == _W:
        cc = int(coords[:, 1].min() - 1)
        rows = sorted(set(map(int, coords[:, 0])))
        targets = [(r, cc) for r in rows]
    else:
        cc = int(coords[:, 1].max() + 1)
        rows = sorted(set(map(int, coords[:, 0])))
        targets = [(r, cc) for r in rows]

    H, W = grid.shape
    for r, c in targets:
        if not (0 <= r < H and 0 <= c < W):
            continue
        if grid[r, c] in (bg, TEAL, color):
            if grid[r, c] != color:
                written += 1
            grid[r, c] = color
    return written


def _simulate(base: np.ndarray, turns: dict[int, str], growth: bool) -> tuple[np.ndarray, dict[str, object]]:
    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    sr, sc, sz = _find_teal_square(base, teal=TEAL)

    out = base.copy()
    _paint_block.sz = sz
    H, W = out.shape

    hit_counts: list[int] = []
    total_growth = 0

    for d0 in range(4):
        r, c, d = sr, sc, d0
        beam_hits = 0
        seen: set[tuple[int, int, int]] = set()
        budget = H * W * 12
        while budget > 0:
            budget -= 1
            state = (r, c, d)
            if not growth:
                assert state not in seen, "static beam path must not cycle"
            else:
                if state in seen and beam_hits == 0:
                    raise AssertionError("beam looped before any growth changed the grid")
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not _in_bounds(out.shape, nr, nc, sz):
                break

            info = _obstacle_info(out, nr, nc, sz, bg)
            if info is None:
                r, c = nr, nc
                _paint_block(out, r, c, bg)
                continue

            color, comp = info
            assert color in turns, "obstacle color must be encoded by the legend"
            beam_hits += 1
            if growth:
                total_growth += _grow_obstacle(out, color, comp, (d + 2) % 4, bg)
                seen.clear()
            d = _rot(d, turns[color])

        assert budget > 0, "dynamic beam exceeded safety budget"
        hit_counts.append(beam_hits)

    stats = {
        "hit_counts": hit_counts,
        "total_hits": int(sum(hit_counts)),
        "growth_cells": int(total_growth),
    }
    return out, stats


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _rect_from_corner(nr: int, nc: int, sz: int, d: int, turn: str, front: int, side: int) -> tuple[int, int, int, int]:
    front_right = turn == "L"
    if d == _N:
        if front_right:
            return nr - front + 1, nr, nc + sz - 1, nc + sz - 1 + side - 1
        return nr - front + 1, nr, nc - side + 1, nc
    if d == _E:
        if front_right:
            return nr + sz - 1, nr + sz - 1 + side - 1, nc + sz - 1, nc + sz - 1 + front - 1
        return nr - side + 1, nr, nc + sz - 1, nc + sz - 1 + front - 1
    if d == _S:
        if front_right:
            return nr + sz - 1, nr + sz - 1 + front - 1, nc - side + 1, nc
        return nr + sz - 1, nr + sz - 1 + front - 1, nc + sz - 1, nc + sz - 1 + side - 1
    if front_right:
        return nr - side + 1, nr, nc - front + 1, nc
    return nr + sz - 1, nr + sz - 1 + side - 1, nc - front + 1, nc


def _rect_ok(sim: np.ndarray, rect: tuple[int, int, int, int], bg: int, src: tuple[int, int, int]) -> bool:
    r0, r1, c0, c1 = rect
    H, W = sim.shape
    if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
        return False
    sr, sc, sz = src
    if not (r1 < sr - 1 or r0 > sr + sz or c1 < sc - 1 or c0 > sc + sz):
        return False
    return bool(np.all(sim[r0 : r1 + 1, c0 : c1 + 1] == bg))


def _rect_ok_spaced(sim: np.ndarray, rect: tuple[int, int, int, int], bg: int, src: tuple[int, int, int]) -> bool:
    if not _rect_ok(sim, rect, bg, src):
        return False
    r0, r1, c0, c1 = rect
    H, W = sim.shape
    er0, er1 = max(0, r0 - 1), min(H - 1, r1 + 1)
    ec0, ec1 = max(0, c0 - 1), min(W - 1, c1 + 1)
    return bool(np.all(sim[er0 : er1 + 1, ec0 : ec1 + 1] == bg))


def _place_obstacle_ahead(
    base: np.ndarray,
    sim: np.ndarray,
    turns: dict[int, str],
    colors: list[int],
    bg: int,
    src: tuple[int, int, int],
    state: tuple[int, int, int],
) -> bool:
    r, c, d = state
    sr, sc, sz = src
    nr, nc = r + _DIRS[d][0], c + _DIRS[d][1]
    if not _in_bounds(sim.shape, nr, nc, sz):
        return False

    trial_colors = colors[:]
    random.shuffle(trial_colors)
    for color in trial_colors:
        turn = turns[color]
        sizes = [
            (front, side)
            for front in range(OBS_FRONT_RANGE[0], OBS_FRONT_RANGE[1] + 1)
            for side in range(OBS_SIDE_RANGE[0], OBS_SIDE_RANGE[1] + 1)
        ]
        random.shuffle(sizes)
        for front, side in sizes:
            rect = _rect_from_corner(nr, nc, sz, d, turn, front, side)
            if not _rect_ok(sim, rect, bg, src):
                continue

            tmp = sim.copy()
            r0, r1, c0, c1 = rect
            tmp[r0 : r1 + 1, c0 : c1 + 1] = color
            try:
                info = _obstacle_info(tmp, nr, nc, sz, bg)
            except AssertionError:
                continue
            if info is None or info[0] != color:
                continue
            growth_written = _grow_obstacle(tmp, color, info[1], (d + 2) % 4, bg)
            if growth_written == 0:
                continue
            d2 = _rot(d, turn)
            rr, cc = r + _DIRS[d2][0], c + _DIRS[d2][1]
            if not _in_bounds(tmp.shape, rr, cc, sz):
                continue
            try:
                info2 = _obstacle_info(tmp, rr, cc, sz, bg)
            except AssertionError:
                continue
            if info2 is not None and info2[0] == color:
                continue
            if np.any(tmp[sr : sr + sz, sc : sc + sz] != TEAL):
                continue

            base[r0 : r1 + 1, c0 : c1 + 1] = color
            sim[r0 : r1 + 1, c0 : c1 + 1] = color
            return True
    return False


def _build_beam(
    base: np.ndarray,
    sim: np.ndarray,
    turns: dict[int, str],
    colors: list[int],
    bg: int,
    src: tuple[int, int, int],
    d0: int,
    placements_goal: int,
) -> bool:
    sr, sc, sz = src
    _paint_block.sz = sz
    r, c, d = sr, sc, d0
    straight = 0
    placed = 0
    hits = 0
    budget = sim.shape[0] * sim.shape[1] * 12

    while budget > 0:
        budget -= 1
        dr, dc = _DIRS[d]
        nr, nc = r + dr, c + dc
        if not _in_bounds(sim.shape, nr, nc, sz):
            break

        try:
            info = _obstacle_info(sim, nr, nc, sz, bg)
        except AssertionError:
            return False
        if info is None:
            if placed < placements_goal and straight >= MIN_STRAIGHT_BEFORE_PLACE:
                if _place_obstacle_ahead(base, sim, turns, colors, bg, src, (r, c, d)):
                    placed += 1
                    continue
            r, c = nr, nc
            _paint_block(sim, r, c, bg)
            straight += 1
            continue

        color, comp = info
        hits += 1
        _grow_obstacle(sim, color, comp, (d + 2) % 4, bg)
        if np.any(sim[sr : sr + sz, sc : sc + sz] != TEAL):
            return False
        d = _rot(d, turns[color])
        straight = 0

    return budget > 0 and placed == placements_goal and hits >= placements_goal


def _count_obstacle_components(base: np.ndarray, bg: int) -> int:
    mask = (base != bg) & (base != TEAL)
    count = 0
    seen = np.zeros_like(mask, dtype=bool)
    H, W = mask.shape
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for dr, dc in _DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _place_decoy(
    base: np.ndarray,
    sim: np.ndarray,
    bg: int,
    src: tuple[int, int, int],
    colors: list[int],
) -> bool:
    H, W = base.shape
    for _ in range(80):
        hh = random.randint(2, 3)
        ww = random.randint(2, 3)
        r0 = random.randint(1, H - hh - 2)
        c0 = random.randint(1, W - ww - 2)
        rect = (r0, r0 + hh - 1, c0, c0 + ww - 1)
        if not _rect_ok_spaced(sim, rect, bg, src):
            continue
        color = random.choice(colors)
        base[r0 : r0 + hh, c0 : c0 + ww] = color
        sim[r0 : r0 + hh, c0 : c0 + ww] = color
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    H, W = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    min_r, max_r = SOURCE_MARGIN, H - sz - SOURCE_MARGIN
    min_c, max_c = SOURCE_MARGIN, W - sz - SOURCE_MARGIN
    if min_r > max_r or min_c > max_c:
        return None
    sr = random.randint(min_r, max_r)
    sc = random.randint(min_c, max_c)

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    shuffled = legend_colors[:]
    random.shuffle(shuffled)
    turns = {color: ("L" if i < 2 else "R") for i, color in enumerate(shuffled)}
    bg_choices = [c for c in BG_COLOR_POOL if c not in legend_colors]
    bg = random.choice(bg_choices)

    base = np.full((H, W), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL
    sim = base.copy()

    obstacle_target = 6
    beam_goals = [2, 1, 0, 0]

    for d0 in range(4):
        if beam_goals[d0] == 0:
            continue
        if not _build_beam(base, sim, turns, legend_colors, bg, (sr, sc, sz), d0, beam_goals[d0]):
            return None

    while _count_obstacle_components(base, bg) < obstacle_target:
        if not _place_decoy(base, sim, bg, (sr, sc, sz), legend_colors):
            return None

    if np.any(sim[sr : sr + sz, sc : sc + sz] != TEAL):
        return None

    component_count = _count_obstacle_components(base, bg)
    if component_count != obstacle_target:
        return None

    try:
        dynamic_out, dyn_stats = _simulate(base, turns, growth=True)
        static_out, static_stats = _simulate(base, turns, growth=False)
    except AssertionError:
        return None

    teal_delta = int(np.count_nonzero((dynamic_out == TEAL) != (static_out == TEAL)))
    obstacle_growth = int(np.count_nonzero(((dynamic_out != bg) & (dynamic_out != TEAL)) & (base == bg)))

    if dyn_stats["total_hits"] < 4:
        return None
    if sum(h >= 1 for h in dyn_stats["hit_counts"]) < 3:
        return None
    if dyn_stats["growth_cells"] < 4:
        return None
    if teal_delta < 2:
        return None
    if obstacle_growth < 4:
        return None
    if np.array_equal(dynamic_out, static_out):
        return None

    legend = _make_legend(legend_colors, turns)
    inp = np.vstack([legend, base])
    actual = main(inp)
    if not np.array_equal(actual, dynamic_out):
        return None
    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four teal beams through a dense field of turn-coded obstacles whose shapes
    can expand after each hit.

    Input structure:
    - A 6x24 legend band sits at the top on grey (5), with four colored vertical bars.
      Left bar means counterclockwise turn; right bar means clockwise turn.
    - Below the legend is the main grid: one solid teal source square, a single
      background color, and 6-8 compact obstacle rectangles reusing the legend colors.

    Rule:
    - Remove the legend in the output.
    - Process the four beams sequentially in the fixed order up, right, down, left.
    - Each beam is a moving copy of the teal source square and paints teal only onto
      background cells.
    - If the next footprint hits obstacle color C, read C in the legend, rotate 90
      degrees accordingly, and also grow the struck obstacle by one row or column
      toward the incoming beam.
    - Growth may overwrite earlier teal trail cells, may squeeze against other
      obstacles, and therefore can change the paths of later beams.

    Key insight:
    The legend still tells you *how* each color turns, but the puzzle is no longer a
    static lookup. Because obstacles expand on contact and beams are processed in a
    fixed order, you must track the evolving grid state while following later beams.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend must form a top band"

    turns = _parse_legend(g, legend_h)
    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after removing the legend"

    out, _stats = _simulate(base, turns, growth=True)
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 combo D8/1+D12/2 input")
