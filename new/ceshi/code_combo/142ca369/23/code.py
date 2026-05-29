# Combo variant of 142ca369 (index 23)
# combo_spec: D3/1+D17/0
# Name: 间歇跳跃轨迹+固定段长

from __future__ import annotations

"""
Design analysis for the fused variant (D3/1 + D17/0).

a) Concrete behavior and trace:
   - Objects stay the same as the base task: 3-pixel L-shaped arrows plus mono-color
     wall segments made of 1 or 3 pixels.
   - Fusion rule:
       1. A ray still walks diagonally one cell at a time.
       2. Coloring is dashed: visited steps 1,3,5,... are painted; 2,4,6,... stay bg.
       3. Motion is segmented: every completed 5-step block triggers a turn event.
       4. At that scheduled endpoint, a side wall may flip one axis and donate its color.
       5. If no side wall is present, the ray still turns automatically by flipping dr.
   - Small trace from start (5,5), direction (1,1), color A:
       steps 1..5 visit (6,6),(7,7),(8,8),(9,9),(10,10)
       painted cells are (6,6),(8,8),(10,10)
       if there is no side wall at step 5, direction becomes (-1,1)
       if instead a horizontal side wall sits at (11,10), step 5 keeps the same geometry
       but flips dr and the painted endpoint (10,10) changes to the wall color.
   - Difference from the original task:
       original turn timing depended on wall location and trail was solid;
       now turn timing depends on the 5-step counter and trail is dashed.

b) Complexity calibration:
   - The task family uses one global segment length only: SEG_LEN = 5.
   - The dash cadence is also global and fixed: paint every other visited cell.
   - Generator forces at least one auto-turn and one wall-turn in each sample, plus
     multiple hidden traversed cells, so both halves of the fusion are visually exposed.
   - Arrow count is kept moderate to preserve whitespace for repeated zigzags.

c) Implicit-assumption audit against the original solver:
   - "Every visited bg cell is painted" is false now; painting depends on step parity.
   - "Side walls are checked after every advance" is false now; side walls matter only at
     scheduled segment endpoints.
   - "Position+direction+color is enough for loop detection" is false now; future output also
     depends on the dash phase and on steps already spent inside the current 5-step block.
   - "Visited cells equal visible trail cells" is false now; skipped cells are invisible but
     still participate in blocking, turn timing, and legality.
   - "Wall placement determines segment length" is false now; segment length is fixed, and walls
     only choose the axis/color at the scheduled turn event.

d) Placement legality in the generator:
   - Output-first planning reserves every traversed cell, even skipped ones.
   - Arrows, traversed cells, and wall cells are pairwise disjoint and in-bounds.
   - Planned wall cells are forbidden from lying on the diagonal path.
   - Auto-turn endpoints are kept free of side walls so the auto-turn is unambiguous.
   - Final segments are forced to have length 1..4 so each ray exits before another full block.

e) Visual observability:
   - Dashed diagonals are directly visible in the output because hidden traversed cells leave
     regular gaps.
   - Fixed-length turning is visible because at least one turn occurs with no nearby wall.
   - Wall-driven recoloring still appears elsewhere, so the wall objects remain meaningful.

f) Visual cleanliness:
   - No overlaps are allowed among arrows, walls, or traversed cells.
   - Wall groups touch only the intended turn cell.
   - Moderate object counts and enforced exit segments prevent the grid from becoming dense.

g) Color inferability:
   - Colors change only at wall-turn endpoints, never at auto-turns.
   - Because wall colors are copied onto both the wall and later painted trail cells, the human
     can still infer that walls donate color while the 5-step clock decides when turning happens.

h) New solver assumptions guaranteed by the generator:
   - Every intended turn happens exactly at a completed 5-step block.
   - At most one side-wall axis is intentionally active at a turn endpoint.
   - No ray revisits its own cells, so writing onto bg cells remains safe.
   - Diagonal wall hits are handled defensively in the solver, but generated samples do not rely
     on them; the fusion signal comes from scheduled turns plus dashed rendering.
"""

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg


SEG_LEN = 5
SKIP_GAP = 1
TRAIL_PERIOD = SKIP_GAP + 1
AUTO_FLIP_AXIS = "dr"


def _should_paint(step_idx: int) -> bool:
    return int(step_idx) % int(TRAIL_PERIOD) == 1


def _is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[tuple[int, int]]):
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    pts_set = set(pts)
    miss = [p for p in corners if p not in pts_set]
    if len(miss) != 1:
        return None

    mr, mc = miss[0]
    dr = 1 if mr == r0 else -1
    dc = 1 if mc == c0 else -1

    deg = {p: 0 for p in pts}
    for r, c in pts:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in pts_set:
                deg[(r, c)] += 1
    start = max(deg, key=lambda p: deg[p])
    return (int(dr), int(dc)), (int(start[0]), int(start[1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-shaped arrows with fused timing + rendering rules.

    Variant rule:
    - The ray advances one diagonal cell per internal step.
    - Every other visited cell is painted, giving a dashed trail.
    - Every completed 5-step block triggers a turn event, regardless of walls.
    - At that endpoint, a side wall may flip one axis and donate its color.
    - Without a side wall, the ray auto-turns by flipping dr.
    - The last segment can be shorter because the ray exits the grid.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)

    wall_color = {
        (int(r), int(c)): int(g[r, c])
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }
    wall_cells = set(wall_color)
    out = g.copy()

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue

        (dr, dc), cur = info
        cur_col = int(start_col)
        step_idx = 0
        steps_in_segment = 0
        seen: set[tuple[tuple[int, int], int, int, int, int, int]] = set()

        while True:
            phase = int(step_idx) % int(TRAIL_PERIOD)
            state = (cur, int(dr), int(dc), int(cur_col), int(steps_in_segment), phase)
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break
            if (nr, nc) in arrow_cells:
                break

            # Defensive fallback only. Generated samples avoid head-on wall hits.
            if (nr, nc) in wall_cells:
                wcol = int(wall_color[(nr, nc)])
                if _should_paint(step_idx) and int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                continue

            cur = (int(nr), int(nc))
            step_idx += 1
            steps_in_segment += 1
            paint_now = _should_paint(step_idx)

            if paint_now and int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if steps_in_segment < SEG_LEN:
                continue

            steps_in_segment = 0
            turned = False

            side_r = (nr + dr, nc)
            if side_r in wall_cells:
                wcol = int(wall_color[side_r])
                if paint_now and int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)
                turned = True

            side_c = (nr, nc + dc)
            if side_c in wall_cells:
                wcol = int(wall_color[side_c])
                if paint_now and int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)
                turned = True

            if not turned:
                if AUTO_FLIP_AXIS == "dr":
                    dr = -int(dr)
                else:
                    dc = -int(dc)

    return out


def _arrow_points(sr: int, sc: int, dr: int, dc: int) -> list[tuple[int, int]]:
    mr, mc = sr - dr, sc - dc
    return [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]


def _max_steps_to_edge(r: int, c: int, dr: int, dc: int, H: int, W: int) -> int:
    return min((H - 1 - r) if dr == 1 else r, (W - 1 - c) if dc == 1 else c)


def _sample_turn_types(n_turns: int, *, require_auto: bool, require_wall: bool) -> list[str]:
    options = ["auto", "wall_dr", "wall_dc"]
    weights = [4, 2, 2]
    turns = [_rng.choices(options, weights=weights)[0] for _ in range(n_turns)]

    if require_auto and "auto" not in turns:
        turns[_rng.randrange(n_turns)] = "auto"
    if require_wall and all(t == "auto" for t in turns):
        turns[_rng.randrange(n_turns)] = _rng.choice(["wall_dr", "wall_dc"])
    if require_auto and require_wall and n_turns >= 2:
        if "auto" not in turns:
            turns[0] = "auto"
        if all(t == "auto" for t in turns):
            turns[1 % n_turns] = _rng.choice(["wall_dr", "wall_dc"])
        _rng.shuffle(turns)
    return turns


def _wall_options(
    cur_r: int,
    cur_c: int,
    dr: int,
    dc: int,
    axis: str,
    H: int,
    W: int,
) -> list[list[tuple[int, int]]]:
    opts: list[list[tuple[int, int]]] = []
    if axis == "dr":
        wr, wc = cur_r + dr, cur_c
        if 0 <= wr < H and 0 <= wc < W:
            opts.append([(int(wr), int(wc))])
            w3 = [(int(wr), int(wc + d)) for d in (-1, 0, 1)]
            if all(0 <= c < W for _, c in w3):
                opts.append(w3)
    else:
        wr, wc = cur_r, cur_c + dc
        if 0 <= wr < H and 0 <= wc < W:
            opts.append([(int(wr), int(wc))])
            w3 = [(int(wr + d), int(wc)) for d in (-1, 0, 1)]
            if all(0 <= r < H for r, _ in w3):
                opts.append(w3)
    _rng.shuffle(opts)
    if len(opts) == 2 and _rng.random() < 0.6:
        opts.sort(key=len, reverse=True)
    return opts


def _wall_at(
    pos: tuple[int, int],
    planned_walls: dict[tuple[int, int], int],
    input_walls: dict[tuple[int, int], int],
) -> int | None:
    if pos in planned_walls:
        return int(planned_walls[pos])
    if pos in input_walls:
        return int(input_walls[pos])
    return None


def _gen_attempt() -> np.ndarray | None:
    """
    Output-first generator for the fused variant.

    Each ray consists of several full 5-step blocks plus a final 1..4 step exit block.
    Full-block endpoints are either auto-turns or wall-turns; painting remains dashed across
    the entire ray, not reset per segment.
    """
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(2, 4)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)

    occupied: set[tuple[int, int]] = set()
    input_walls: dict[tuple[int, int], int] = {}

    total_hidden_steps = 0
    saw_auto = False
    saw_wall = False

    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        placed = False

        for _ in range(220):
            dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
            n_turns = _rng.randint(2, 3) if ai == 0 else _rng.randint(1, 3)
            turn_types = _sample_turn_types(
                n_turns,
                require_auto=(ai == 0),
                require_wall=(ai == 0),
            )

            r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
            c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
            if r_lo > r_hi or c_lo > c_hi:
                continue

            sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)
            arrow_pts = _arrow_points(sr, sc, dr, dc)
            arrow_set = set(arrow_pts)
            if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
                continue

            cur_r, cur_c = int(sr), int(sc)
            cur_dr, cur_dc = int(dr), int(dc)
            cur_col = int(color)
            step_idx = 0
            hidden_steps = 0
            painted_steps = 0
            local_auto = 0
            local_wall = 0
            trail_steps: list[list[int | bool]] = []
            ray_set: set[tuple[int, int]] = set()
            planned_walls: dict[tuple[int, int], int] = {}
            wall_groups: list[tuple[list[tuple[int, int]], int]] = []
            ok = True

            for turn_type in turn_types:
                if _max_steps_to_edge(cur_r, cur_c, cur_dr, cur_dc, H, W) < SEG_LEN:
                    ok = False
                    break

                for _step in range(SEG_LEN):
                    nr, nc = int(cur_r + cur_dr), int(cur_c + cur_dc)
                    if not (0 <= nr < H and 0 <= nc < W):
                        ok = False
                        break
                    pos = (int(nr), int(nc))
                    if pos in occupied or pos in ray_set or pos in arrow_set or pos in planned_walls:
                        ok = False
                        break
                    step_idx += 1
                    paint_now = _should_paint(step_idx)
                    trail_steps.append([int(nr), int(nc), int(cur_col), bool(paint_now)])
                    if paint_now:
                        painted_steps += 1
                    else:
                        hidden_steps += 1
                    ray_set.add(pos)
                    cur_r, cur_c = int(nr), int(nc)
                if not ok:
                    break

                side_dr = (int(cur_r + cur_dr), int(cur_c))
                side_dc = (int(cur_r), int(cur_c + cur_dc))
                if _wall_at(side_dr, planned_walls, input_walls) is not None or _wall_at(
                    side_dc, planned_walls, input_walls
                ) is not None:
                    ok = False
                    break

                if turn_type == "auto":
                    local_auto += 1
                    if AUTO_FLIP_AXIS == "dr":
                        cur_dr = -int(cur_dr)
                    else:
                        cur_dc = -int(cur_dc)
                    continue

                flip_axis = "dr" if turn_type == "wall_dr" else "dc"
                other_side = side_dc if flip_axis == "dr" else side_dr
                candidates = _wall_options(cur_r, cur_c, cur_dr, cur_dc, flip_axis, H, W)
                chosen = None
                forbidden = occupied | ray_set | arrow_set | set(planned_walls) | set(input_walls)
                for cells in candidates:
                    if any(cell in forbidden for cell in cells):
                        continue
                    if any(cell == other_side for cell in cells):
                        continue
                    chosen = cells
                    break
                if chosen is None:
                    ok = False
                    break

                wcol_choices = [p for p in palette if int(p) != int(cur_col)]
                wcol = int(_rng.choice(wcol_choices or palette))
                for cell in chosen:
                    planned_walls[(int(cell[0]), int(cell[1]))] = int(wcol)
                wall_groups.append((chosen, int(wcol)))
                if trail_steps and bool(trail_steps[-1][3]):
                    trail_steps[-1][2] = int(wcol)
                cur_col = int(wcol)
                local_wall += 1
                if flip_axis == "dr":
                    cur_dr = -int(cur_dr)
                else:
                    cur_dc = -int(cur_dc)

            if not ok:
                continue

            final_steps = _max_steps_to_edge(cur_r, cur_c, cur_dr, cur_dc, H, W)
            if final_steps <= 0 or final_steps >= SEG_LEN:
                continue

            for _step in range(final_steps):
                nr, nc = int(cur_r + cur_dr), int(cur_c + cur_dc)
                pos = (int(nr), int(nc))
                if pos in occupied or pos in ray_set or pos in arrow_set or pos in planned_walls:
                    ok = False
                    break
                step_idx += 1
                paint_now = _should_paint(step_idx)
                trail_steps.append([int(nr), int(nc), int(cur_col), bool(paint_now)])
                if paint_now:
                    painted_steps += 1
                else:
                    hidden_steps += 1
                ray_set.add(pos)
                cur_r, cur_c = int(nr), int(nc)

            if not ok:
                continue
            if len(ray_set) < SEG_LEN + 2 or hidden_steps < 2 or painted_steps < 3:
                continue

            new_cells = set(arrow_pts) | ray_set | set(planned_walls)
            expected = len(arrow_pts) + len(ray_set) + len(planned_walls)
            if len(new_cells) != expected or new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[int(r), int(c)] = int(color)
                out[int(r), int(c)] = int(color)
            for r, c, col, paint_now in trail_steps:
                if bool(paint_now):
                    out[int(r), int(c)] = int(col)
            for wpxs, wcol in wall_groups:
                for r, c in wpxs:
                    grid[int(r), int(c)] = int(wcol)
                    out[int(r), int(c)] = int(wcol)
                    input_walls[(int(r), int(c))] = int(wcol)

            occupied |= new_cells
            total_hidden_steps += hidden_steps
            saw_auto = saw_auto or (local_auto > 0)
            saw_wall = saw_wall or (local_wall > 0)
            placed = True
            break

        if not placed:
            return None

    if total_hidden_steps < 4 or not (saw_auto and saw_wall):
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the fused dashed fixed-segment variant."""
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
