# Variant of 142ca369
# Dimension: D17 — 射线段长度约束 (meta)
# Change: 反射之间的射线段长度为2-16格（由网格空间自然决定）... → 固定段长度

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

# Design analysis for the D17 variant
# a) Concrete behavior:
#    Objects stay the same: 3-pixel L-shaped arrows and mono-color wall segments
#    (1-pixel or 3-pixel axis-aligned). The visible change is temporal: rays no
#    longer turn only when a wall is encountered at a geometrically convenient
#    bounce point. Instead, a ray walks exactly SEG_LEN diagonal steps, then a
#    turn event happens. At that scheduled endpoint:
#      - a horizontal side wall at (r+dr, c) flips dr and gives its color;
#      - a vertical side wall at (r, c+dc) flips dc and gives its color;
#      - if neither wall exists, the ray still turns automatically by flipping
#        dr, keeping its current color.
#    Small trace:
#      original: start -> move until side wall exists -> bounce there.
#      variant: start -> step 5 cells regardless -> inspect side walls there.
#      If no wall, the old task would continue straight; the variant turns anyway.
# b) Complexity calibration:
#    SEG_LEN is global and fixed at 5. That is short enough that multiple turns
#    appear in a 18-20 sized grid, but long enough that the ray is still clearly
#    diagonal. Generator guarantees at least one auto-turn and one wall-driven
#    turn in every sample so the new counting rule is observable.
# c) Old assumptions audited:
#    - Old solver checked side walls after every advance; this depended on walls
#      deciding turn timing. That is no longer valid, so side-wall checks move to
#      scheduled turn events only.
#    - Old solver stored state without a segment counter because direction/color
#      fully determined future motion. Now the remaining steps inside the current
#      5-step block matter, so `steps_in_segment` is part of the seen-state.
#    - Old generator chose arbitrary segment lengths from geometry. That must be
#      replaced by fixed 5-step segments plus a final partial segment that exits.
# d) Placement legality:
#    Generator plans the whole output first. It forbids overlaps among arrows,
#    ray cells, and wall cells; forbids diagonal wall cells on the path; forbids
#    accidental side walls at auto-turn endpoints; and requires the final segment
#    to exit before another full 5-step block can complete.
# e) Visual observability:
#    Every generated grid has at least one 5-step turn with no nearby wall, so a
#    human can see that turns are not wall-timed anymore. Wall-driven turns still
#    appear elsewhere to preserve the original wall/color role.
# f) Visual cleanliness:
#    Rays never overlap other rays, arrows, or walls. Wall endpoints touch only
#    the intended ray turn cell. Arrow count is kept moderate to leave whitespace
#    for repeated 5-step zigzags to read cleanly.
# g) Color inferability:
#    Only wall-triggered turns recolor the endpoint; auto-turns do not. This
#    keeps the color rule structured and easy to separate from the timing rule.
# h) New solver assumptions:
#    The generator guarantees that intentional side-wall interactions happen only
#    at scheduled 5-step endpoints and at most one side-wall axis is present for
#    each planned turn. The solver still handles rare diagonal wall hits
#    defensively, but the generated distribution does not rely on them.

SEG_LEN = 5
AUTO_FLIP_AXIS = "dr"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-shaped arrows.

    Variant rule:
    - A ray advances diagonally one cell at a time.
    - Every completed 5-step block triggers a turn event.
    - At that event, a side wall can flip one axis and donate its color.
    - If no side wall is present, the ray still auto-turns by flipping dr.
    - The final segment may be shorter because the ray exits the grid.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
        if len(pts) != 3:
            return False
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)

    def arrow_dir_and_start(pts: list[tuple[int, int]]):
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

    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if is_arrow_pts(pts):
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
        info = arrow_dir_and_start(pts)
        if info is None:
            continue

        (dr, dc), cur = info
        cur_col = int(start_col)
        steps_in_segment = 0
        seen: set[tuple[tuple[int, int], int, int, int, int]] = set()

        while True:
            state = (cur, int(dr), int(dc), int(cur_col), int(steps_in_segment))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break
            if (nr, nc) in arrow_cells:
                break

            # Defensive fallback: diagonal wall hits still change direction/color,
            # but they do not reset the fixed-length step counter.
            if (nr, nc) in wall_cells:
                wcol = int(wall_color[(nr, nc)])
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                continue

            cur = (int(nr), int(nc))
            steps_in_segment += 1
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if steps_in_segment < SEG_LEN:
                continue

            steps_in_segment = 0
            turned = False

            side_r = (nr + dr, nc)
            if int(dr) == 1 and side_r in wall_cells:
                wcol = int(wall_color[side_r])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)
                turned = True
            elif int(dr) == -1 and side_r in wall_cells:
                wcol = int(wall_color[side_r])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)
                turned = True

            side_c = (nr, nc + dc)
            if int(dc) == 1 and side_c in wall_cells:
                wcol = int(wall_color[side_c])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)
                turned = True
            elif int(dc) == -1 and side_c in wall_cells:
                wcol = int(wall_color[side_c])
                if int(g[nr, nc]) == int(bg):
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
        idx = _rng.randrange(n_turns)
        turns[idx] = _rng.choice(["wall_dr", "wall_dc"])
    if require_auto and require_wall and n_turns >= 2:
        if turns.count("auto") == n_turns:
            turns[0] = "auto"
            turns[1] = _rng.choice(["wall_dr", "wall_dc"])
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
            opts.append([(wr, wc)])
            w3 = [(wr, wc + d) for d in (-1, 0, 1)]
            if all(0 <= c < W for _, c in w3):
                opts.append(w3)
    else:
        wr, wc = cur_r, cur_c + dc
        if 0 <= wr < H and 0 <= wc < W:
            opts.append([(wr, wc)])
            w3 = [(wr + d, wc) for d in (-1, 0, 1)]
            if all(0 <= r < H for r, _ in w3):
                opts.append(w3)
    _rng.shuffle(opts)
    if len(opts) == 2 and _rng.random() < 0.6:
        opts.sort(key=len, reverse=True)
    return opts


def _gen_attempt() -> np.ndarray | None:
    """
    Output-first generator for the D17 variant.

    Every fully completed segment has length SEG_LEN. At scheduled endpoints the
    generator either places a side wall to control the axis/color change or
    leaves the endpoint wall-free so the solver must auto-turn.
    """
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(2, 5)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)

    occupied: set[tuple[int, int]] = set()
    input_walls: dict[tuple[int, int], int] = {}

    saw_auto = False
    saw_wall = False

    def wall_at(pos: tuple[int, int], planned_walls: dict[tuple[int, int], int]) -> int | None:
        if pos in planned_walls:
            return int(planned_walls[pos])
        if pos in input_walls:
            return int(input_walls[pos])
        return None

    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        placed = False

        for _ in range(160):
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

            cur_r, cur_c = sr, sc
            cur_dr, cur_dc = dr, dc
            cur_col = color
            ray_cells: list[tuple[int, int, int]] = []
            ray_set: set[tuple[int, int]] = set()
            planned_walls: dict[tuple[int, int], int] = {}
            wall_groups: list[tuple[list[tuple[int, int]], int]] = []
            ok = True
            local_auto = 0
            local_wall = 0

            for turn_type in turn_types:
                if _max_steps_to_edge(cur_r, cur_c, cur_dr, cur_dc, H, W) < SEG_LEN:
                    ok = False
                    break

                for _step in range(SEG_LEN):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        ok = False
                        break
                    pos = (int(nr), int(nc))
                    if pos in occupied or pos in ray_set or pos in arrow_set or pos in planned_walls:
                        ok = False
                        break
                    ray_cells.append((int(nr), int(nc), int(cur_col)))
                    ray_set.add(pos)
                    cur_r, cur_c = int(nr), int(nc)
                if not ok:
                    break

                side_dr = (cur_r + cur_dr, cur_c)
                side_dc = (cur_r, cur_c + cur_dc)
                if turn_type == "auto":
                    if wall_at(side_dr, planned_walls) is not None or wall_at(side_dc, planned_walls) is not None:
                        ok = False
                        break
                    local_auto += 1
                    cur_dr = -cur_dr
                    continue

                flip_axis = "dr" if turn_type == "wall_dr" else "dc"
                other_side = side_dc if flip_axis == "dr" else side_dr
                if wall_at(side_dr, planned_walls) is not None or wall_at(side_dc, planned_walls) is not None:
                    ok = False
                    break

                candidates = _wall_options(cur_r, cur_c, cur_dr, cur_dc, flip_axis, H, W)
                chosen = None
                forbidden = occupied | ray_set | arrow_set | set(planned_walls)
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
                    planned_walls[cell] = int(wcol)
                wall_groups.append((chosen, int(wcol)))
                lr, lc, _ = ray_cells[-1]
                ray_cells[-1] = (int(lr), int(lc), int(wcol))
                cur_col = int(wcol)
                local_wall += 1
                if flip_axis == "dr":
                    cur_dr = -cur_dr
                else:
                    cur_dc = -cur_dc

            if not ok:
                continue

            final_steps = _max_steps_to_edge(cur_r, cur_c, cur_dr, cur_dc, H, W)
            if final_steps <= 0 or final_steps >= SEG_LEN:
                continue

            for _step in range(final_steps):
                nr, nc = cur_r + cur_dr, cur_c + cur_dc
                pos = (int(nr), int(nc))
                if pos in occupied or pos in ray_set or pos in arrow_set or pos in planned_walls:
                    ok = False
                    break
                ray_cells.append((int(nr), int(nc), int(cur_col)))
                ray_set.add(pos)
                cur_r, cur_c = int(nr), int(nc)
            if not ok or len(ray_cells) < SEG_LEN + 2:
                continue

            new_cells = set(arrow_pts) | ray_set | set(planned_walls)
            expected = len(arrow_pts) + len(ray_set) + len(planned_walls)
            if len(new_cells) != expected or new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = out[r, c] = color
            for r, c, col in ray_cells:
                out[r, c] = int(col)
            for wpxs, wcol in wall_groups:
                for r, c in wpxs:
                    grid[r, c] = out[r, c] = int(wcol)
                    input_walls[(r, c)] = int(wcol)

            occupied |= new_cells
            saw_auto = saw_auto or (local_auto > 0)
            saw_wall = saw_wall or (local_wall > 0)
            placed = True
            break

        if not placed:
            return None

    if not (saw_auto and saw_wall):
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D17 fixed-segment variant."""
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
