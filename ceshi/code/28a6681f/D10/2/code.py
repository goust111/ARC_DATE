# Variant of 28a6681f
# Dimension: D10 — guide/引导结构 (obstacle)
# Change: 可选的上部guide/斜坡，从网格左或右边缘进入，单斜臂或双臂漏斗形，4-连通，引导小量水进入目标盆地... → 内部管道式guide

from __future__ import annotations

import random

import numpy as np


# Design analysis for the D10 internal-pipe variant.
# 1. Concrete realization:
#    - The grid is still 10x10 and the bottom half still contains adjacent gravity basins.
#    - The old upper sloped ramp is replaced by one floating pipe shell in rows 1..5.
#      The shell is a single-color solid component that does not touch the left/right border.
#    - Inside the shell there is a covered cavity: a 2x3 source chamber feeding a short
#      horizontal corridor with exactly one bottom opening above the target basin.
#    - Input water is packed only inside that cavity, so every source pixel is visibly "under
#      a roof". The outlet then releases the water into the same lower overflow physics as the
#      original task.
# 2. Behavior trace:
#    - Original D10 guide samples used an exposed diagonal arm entering from column 0 or 9.
#      Water moved in open space beside that arm.
#    - Here a left sample may start with water at (4,2),(4,3),(3,2),(3,3),(2,2),(2,3), all
#      inside a covered shell. The old local rule is unchanged: bottom-row source water slides
#      right along row 4 to the only drop at (5,4), falls into the left basin, and may then
#      overflow across the lower divider into the neighbor.
#    - The visible difference is not a hidden color code; it is that the guide is now a closed
#      obstacle outline with an internal cavity and a single outlet.
# 3. Complexity calibration:
#    - The basin system stays at two adjacent basins, which is enough for overflow and keeps D10
#      visually readable.
#    - Water count is limited to 3..6 so some samples stop in the first basin while others spill
#      into the second. This keeps both "no overflow" and "overflow" branches alive.
#    - Pipe placement has two observable positions: above the left basin or above the right basin.
# 4. Implicit-assumption audit against the original solver:
#    - The original guide check depended on many diagonal links and on touching a grid edge.
#      That is invalid here because the new pipe is orthogonal and internal, so that logic is
#      removed and replaced with cavity/outlet checks.
#    - The original support constraint for input water is still valid, but the generator must
#      fill chamber cells in a support-safe order because no source pixel may float.
#    - The original settle() routine remains correct because the pipe only constrains background
#      cells; water still moves by vertical fall, unique diagonal slip, and supported horizontal
#      sliding toward a unique drop.
# 5. Placement legality:
#    - The pipe bbox is separated from the bottom walls, never touches the outer border, and has
#      one opening on its bottom row above a basin interior column.
#    - Basin walls stay bottom-anchored and do not overlap the pipe.
#    - The divider wall is lower than the source-side outer wall, so overflow is physically
#      possible when the water count is large enough.
# 6. Visual observability and cleanliness:
#    - The pipe uses its own solid color, distinct from the two basin colors.
#    - The source water is visibly enclosed by a roofed shell, which makes the D10 change clear
#      from only a few examples.
#    - No sorting or behavior depends on color ids; colors only separate visible object roles.


BG = 0
WATER = 1
H = 10
W = 10
DIVIDER_HEIGHT = 3
SOURCE_WATER_COUNTS = [4, 4, 6, 6]
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(points: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(points)
    comps: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        comp = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in N4:
                nxt = (r + dr, c + dc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    comp.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


def _find_pipe_candidates(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(set(map(int, g.flatten())) - {BG, WATER})
    candidates: list[dict[str, object]] = []
    for color in colors:
        pts = {(int(r), int(c)) for r, c in np.argwhere(g == int(color))}
        for comp in _components(pts):
            if any(r == H - 1 for r, _c in comp):
                continue
            rs = [r for r, _c in comp]
            cs = [c for _r, c in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            if r1 > 5 or len(comp) < 12:
                continue
            cavity = {
                (r, c)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
                if (r, c) not in comp and int(g[r, c]) in (BG, WATER)
            }
            water_cells = {p for p in cavity if int(g[p[0], p[1]]) == WATER}
            if not water_cells:
                continue
            openings = {
                (r, c)
                for (r, c) in cavity
                if r == r1 and r + 1 < H and int(g[r + 1, c]) == BG
            }
            covered_water = {
                (r, c)
                for (r, c) in water_cells
                if any(int(g[rr, c]) == int(color) for rr in range(r0, r))
            }
            reachable: set[tuple[int, int]] = set()
            if openings:
                stack = [next(iter(openings))]
                reachable.add(stack[0])
                while stack:
                    r, c = stack.pop()
                    for dr, dc in N4:
                        nxt = (r + dr, c + dc)
                        if nxt in cavity and nxt not in reachable:
                            reachable.add(nxt)
                            stack.append(nxt)
            candidates.append(
                {
                    "color": int(color),
                    "bbox": (r0, r1, c0, c1),
                    "cavity": cavity,
                    "water_cells": water_cells,
                    "covered_water": covered_water,
                    "openings": openings,
                    "reachable": reachable,
                }
            )
    return candidates


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Settle blue water that starts inside a covered internal guide and then drains into lower basins.

    Input structure:
    - 0 is background and 1 is water. The other colors are rigid basin walls plus one upper pipe shell.
    - The pipe is an internal obstacle component with a roofed cavity and a single bottom outlet above one basin.
    - Bottom-anchored walls form adjacent basins so released water may stay in the target basin or spill into its neighbor.

    Rule:
    - Water keeps the original local physics: it falls through background, uses the only open diagonal slip when blocked, and when supported it slides along its row toward the nearest unique drop.
    - The new guide does not move water by teleportation; it only constrains those same motions inside a covered cavity until the outlet releases the water.
    - Iterate pixel settling until no water moves, while keeping every non-water structure cell unchanged.

    Key insight: The only new reasoning burden is to read the enclosed upper cavity as a one-way pipe, then continue the usual basin-filling and overflow simulation after the outlet.
    """
    g = np.array(grid, dtype=int).copy()

    if check_input:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

        upper_water = {(int(r), int(c)) for r, c in np.argwhere(g[:6] == int(WATER))}
        if upper_water:
            owners = [pipe for pipe in _find_pipe_candidates(g) if upper_water <= pipe["cavity"]]
            assert len(owners) == 1, "upper source water must lie inside exactly one internal pipe"
            pipe = owners[0]
            r0, r1, c0, c1 = pipe["bbox"]
            assert c0 > 0 and c1 < W - 1, "internal pipe cannot touch the left or right grid edge"
            assert len(pipe["openings"]) == 1, "internal pipe must have exactly one outlet"
            assert upper_water <= pipe["reachable"], "all upper source water must connect to the pipe outlet"
            assert len(pipe["covered_water"]) >= min(2, len(upper_water)), "pipe water must visibly sit under a roof"
            assert r0 <= 1 and r1 == 5, "pipe should occupy the upper guide band and end above the basins"

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            opts = []
            if (
                rr + 1 < H
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == int(BG)
                and int(g[rr + 1, cc - 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc - 1)))
            if (
                rr + 1 < H
                and cc + 1 < W
                and int(g[rr, cc + 1]) == int(BG)
                and int(g[rr + 1, cc + 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != int(BG):
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
                    left -= 1
                right = int(cc)
                while right + 1 < W and int(g[rr, right + 1]) == int(BG):
                    right += 1

                drops = [x for x in range(left, right + 1) if int(g[rr + 1, x]) == int(BG)]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                best = nearest[0]
                assert best != int(cc)
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert int(g[rr, nx]) == int(BG)
                cc = nx
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc = settle(r, c)
            moved = bool(moved or (rr, cc) != (r, c))
            g[rr, cc] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert cc == 0 or int(g[rr, cc - 1]) != int(BG)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG)
            assert cc == W - 1 or int(g[rr, cc + 1]) != int(BG)

    return g


def _draw_basins(grid: np.ndarray, target_side: str, basin_color: int, divider_color: int) -> None:
    left_height = 4 if target_side == "left" else 5
    right_height = 5 if target_side == "left" else 4

    grid[9, 2:5] = int(basin_color)
    grid[9, 5] = int(divider_color)
    grid[9, 6:9] = int(basin_color)

    for r in range(H - left_height, H):
        grid[r, 2] = int(basin_color)
    for r in range(H - DIVIDER_HEIGHT, H):
        grid[r, 5] = int(divider_color)
    for r in range(H - right_height, H):
        grid[r, 8] = int(basin_color)


def _carve_pipe(grid: np.ndarray, target_side: str, pipe_color: int) -> dict[str, object]:
    if target_side == "left":
        r0, r1, c0, c1 = 1, 5, 1, 6
        cavity = {(2, 2), (2, 3), (3, 2), (3, 3), (4, 2), (4, 3), (4, 4), (5, 4)}
        source_order = [(4, 2), (4, 3), (3, 2), (3, 3), (2, 2), (2, 3)]
        opening = (5, 4)
        far_cols = {6, 7}
    else:
        r0, r1, c0, c1 = 1, 5, 2, 8
        cavity = {
            (2, 3),
            (2, 4),
            (3, 3),
            (3, 4),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),
            (5, 7),
        }
        source_order = [(4, 3), (4, 4), (3, 3), (3, 4), (2, 3), (2, 4)]
        opening = (5, 7)
        far_cols = {3, 4}

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            grid[r, c] = int(pipe_color)
    for r, c in cavity:
        grid[r, c] = int(BG)

    return {
        "cavity": cavity,
        "source_order": source_order,
        "opening": opening,
        "far_cols": far_cols,
    }


def _variant_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    pipes = _find_pipe_candidates(inp)
    assert len(pipes) == 1
    pipe = pipes[0]
    opening = next(iter(pipe["openings"]))
    if opening[1] <= 4:
        far_cols = {6, 7}
    else:
        far_cols = {3, 4}
    overflow = sum(1 for r, c in np.argwhere(out == int(WATER)) if int(c) in far_cols)
    return {
        "covered_water": len(pipe["covered_water"]),
        "pipe_cells": len(pipe["reachable"]),
        "overflow_cells": int(overflow),
        "outlet_col": int(opening[1]),
    }


def _generate_once() -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    basin_color, divider_color, pipe_color = random.sample([2, 3, 4, 5, 6, 7, 8, 9], 3)
    target_side = random.choice(["left", "right"])
    water_count = random.choice(SOURCE_WATER_COUNTS)

    _draw_basins(grid, target_side, basin_color, divider_color)
    pipe = _carve_pipe(grid, target_side, pipe_color)

    for r, c in pipe["source_order"][:water_count]:
        grid[r, c] = int(WATER)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(200):
        inp = _generate_once()
        out = main(inp)
        assert int(np.sum(inp == int(WATER))) == int(np.sum(out == int(WATER)))
        assert not np.any(out[:6] == int(WATER)), "pipe should empty into the basins"
        metrics = _variant_metrics(inp, out)
        if metrics["covered_water"] >= 2 and metrics["pipe_cells"] >= 6:
            return inp
    raise AssertionError("failed to generate a valid internal-pipe sample")
