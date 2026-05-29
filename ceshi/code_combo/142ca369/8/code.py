# Combo variant of 142ca369 (index 8)
# combo_spec: D2/1+D19/0
# Name: 颜色编码转向+网格边缘图例

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg


"""
Design analysis for the combo variant
====================================

a) Concrete behavior trace
   - Keep the original object vocabulary in the playfield: black background, sparse
     3-pixel L-arrows, and 1-pixel / 3-pixel monochrome wall segments.
   - Add a dedicated 2-row legend strip on the top edge with grey background. Each
     legend entry is a 2-pixel mini-symbol for one wall color:
       top middle + lower-left  => CCW
       top middle + lower-right => CW
   - Example: a ray moves with direction (1,1). In the original task, a wall below
     the next ray cell flips dr and yields (-1,1). In this combo, the same geometry
     only identifies which wall was touched; the wall color is then looked up in the
     legend. If the legend says CW, (1,1) becomes (1,-1); if the legend says CCW,
     it becomes (-1,1). The path therefore depends on both local contact geometry
     and the edge legend.

b) Complexity calibration
   - Each sample uses exactly two rule-wall colors. This is enough to force legend
     reading, but still small enough to infer from 2-4 ARC examples.
   - Both rule colors are reused in actual bounce events, and each sample has at
     least four total turns. That gives repeated evidence for both legend entries.
   - Arrow colors are chosen from the remaining palette, so arrows and walls stay
     visually separable.

c) Implicit-assumption audit against the original solver
   - Original assumption: every non-bg non-arrow pixel belongs to the playfield.
     Broken by the new legend strip, so `main()` must parse and remove the legend
     before arrow/wall detection.
   - Original assumption: wall geometry determines which axis flips. Broken by D2/1:
     side-touch and front-contact now only identify the contacted wall color; the
     actual 90-degree turn comes from the legend mapping.
   - Original sequential side-touch checks tolerated double contacts. In the combo,
     two simultaneous side walls would make the legend lookup ambiguous, so `main()`
     treats that as invalid and the generator forbids it.
   - Original code allowed arbitrary wall colors. Here every wall color must appear
     in the legend, so the generator restricts playfield walls to exactly the two
     legend colors.

d) Placement legality
   - Grey (5) is reserved for the legend strip and never appears in the playfield.
   - Generator remains output-first: plan diagonal segments and turn colors first,
     then place arrows and walls, and finally verify with `main(input) == output`.
   - Rays, arrows, and walls do not overlap. Same-color wall cells from different
     segments are kept 4-disconnected so they cannot accidentally form L-arrows.
   - Each planned turn has exactly one contacted wall side; all other early contacts
     are rejected during planning.

e) Visual observability
   - The top strip is an obvious non-playfield region, so humans can notice it as
     instruction metadata rather than obstacle geometry.
   - The generator deliberately places each wall on the side that would *not* produce
     the same next direction under the original mirror rule. That makes the combo
     visibly different from the source task on the same input.

f) Visual cleanliness
   - The legend is isolated to the top 2 rows; the lower area remains a sparse field
     of arrows, walls, and diagonal traces.
   - Only two wall colors carry turning meaning per sample, which keeps the codebook
     clean and avoids color clutter.

g) Inferability of color-driven changes
   - The only semantic color mapping is wall-color -> CW/CCW, and it is exposed by
     the legend with a fixed mini-symbol language.
   - Because those same colors reappear on bounce-causing walls, the solver can infer
     the mapping structurally rather than from raw numeric color ids.

h) Quantitative anti-degeneracy target
   - A generated sample is accepted only if every realized turn event is a side-touch
     event whose resulting direction differs from the original mirror-reflection
     result for that same contacted wall side. This directly measures whether the
     fusion logic is behaviorally active rather than decorative.

i) New solver assumptions and generator guarantees
   - Assumption: the legend is the contiguous top prefix containing grey cells.
     Guarantee: generator uses grey only in the 2-row legend strip.
   - Assumption: each rule color has exactly one legend entry. Guarantee: generator
     builds exactly two separated entries.
   - Assumption: each contacted wall color is defined in the legend. Guarantee:
     generator draws all playfield walls from the legend colors only.
   - Assumption: no ambiguous double side-touch occurs at one ray cell. Guarantee:
     generator never places both row-side and col-side contacts for the same turn.
"""


LEGEND_BG = 5
LEGEND_H = 2
PLAY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _rot_cw(dr: int, dc: int) -> tuple[int, int]:
    return int(dc), int(-dr)


def _rot_ccw(dr: int, dc: int) -> tuple[int, int]:
    return int(-dc), int(dr)


def _apply_turn(dr: int, dc: int, turn: str) -> tuple[int, int]:
    return _rot_cw(dr, dc) if turn == "cw" else _rot_ccw(dr, dc)


def _parse_legend(grid: np.ndarray) -> tuple[int, dict[int, str]]:
    h, w = grid.shape
    legend_h = 0
    while legend_h < h and np.any(grid[legend_h] == LEGEND_BG):
        legend_h += 1
    if legend_h < LEGEND_H:
        return 0, {}

    legend = grid[:legend_h, :]
    sep_cols = [c for c in range(w) if np.all(legend[:, c] == LEGEND_BG)]
    turn_by_color: dict[int, str] = {}
    for left, right in zip(sep_cols, sep_cols[1:]):
        interior = legend[:, left + 1 : right]
        if interior.shape[0] != LEGEND_H or interior.shape[1] < 2:
            continue
        colors = set(map(int, np.unique(interior)))
        colors.discard(0)
        colors.discard(LEGEND_BG)
        if len(colors) != 1:
            continue
        wall_color = int(next(iter(colors)))
        pts = [(int(r), int(c)) for r, c in np.argwhere(interior == wall_color)]
        if len(pts) != 2:
            continue
        lower = [c for r, c in pts if r == LEGEND_H - 1]
        if len(lower) != 1:
            continue
        turn_by_color[wall_color] = "ccw" if lower[0] <= (interior.shape[1] - 1) // 2 else "cw"
    return legend_h, turn_by_color


def _is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
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


def _rotate_by_legend_color(dr: int, dc: int, wall_color: int, turn_by_color: dict[int, str]) -> tuple[int, int] | None:
    turn = turn_by_color.get(int(wall_color))
    if turn is None:
        return None
    return _apply_turn(int(dr), int(dc), turn)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-arrows using a top-edge legend for wall-color turns.

    Rule summary:
    - The top legend strip maps wall colors to 90-degree CW/CCW turns.
    - Rays still inherit the wall color at each contact.
    - Contact geometry only decides *which* wall was touched, not the turn axis.
    """
    g = np.array(grid, dtype=int)
    legend_h, turn_by_color = _parse_legend(g)
    if legend_h >= g.shape[0]:
        return g.copy()

    core = g[legend_h:, :].copy()
    h, w = core.shape
    bg = int(get_bg(core))

    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(core, exclude={bg}):
        for coords in components_mask(core == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)

    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c))
        for r, c in np.argwhere(core != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }

    out = core.copy()

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
        cur_col = int(start_col)
        seen: set[tuple[tuple[int, int], int, int, int]] = set()

        while True:
            state = (cur, int(dr), int(dc), int(cur_col))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                break

            if (nr, nc) in wall_cells:
                wcol = int(core[nr, nc])
                turned = _rotate_by_legend_color(dr, dc, wcol, turn_by_color)
                if turned is None:
                    break
                if int(core[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = turned
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            touches: list[tuple[int, int]] = []
            if (nr + int(dr), nc) in wall_cells:
                touches.append((int(nr + dr), int(nc)))
            if (nr, nc + int(dc)) in wall_cells:
                touches.append((int(nr), int(nc + dc)))
            if len(touches) > 1:
                break
            if len(touches) == 1:
                wr, wc = touches[0]
                wcol = int(core[wr, wc])
                turned = _rotate_by_legend_color(dr, dc, wcol, turn_by_color)
                if turned is None:
                    break
                if int(core[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr, dc = turned

    result = g.copy()
    result[legend_h:, :] = out
    return result


def _build_legend(width: int, turn_by_color: dict[int, str], order: list[int]) -> np.ndarray:
    legend = np.full((LEGEND_H, width), LEGEND_BG, dtype=int)
    for i, wall_color in enumerate(order):
        start = 1 + 4 * i
        legend[0, start + 1] = int(wall_color)
        legend[1, start + (0 if turn_by_color[int(wall_color)] == "ccw" else 2)] = int(wall_color)
    return legend


def _same_color_clear(
    cells: list[tuple[int, int]],
    global_same_color: set[tuple[int, int]],
    local_same_color: set[tuple[int, int]],
) -> bool:
    occupied = global_same_color | local_same_color
    for r, c in cells:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in occupied:
                return False
    return True


def _max_diagonal_steps(r: int, c: int, dr: int, dc: int, h: int, w: int) -> int:
    return min((h - 1 - r) if dr == 1 else r, (w - 1 - c) if dc == 1 else c)


def _mismatch_side(dr: int, dc: int, wall_color: int, turn_by_color: dict[int, str]) -> str:
    turned = _rotate_by_legend_color(dr, dc, wall_color, turn_by_color)
    row_reflect = (-int(dr), int(dc))
    col_reflect = (int(dr), -int(dc))
    if turned == row_reflect and turned != col_reflect:
        return "col"
    if turned == col_reflect and turned != row_reflect:
        return "row"
    return "row"


def _pick_wall_cells(
    h: int,
    w: int,
    end_r: int,
    end_c: int,
    dr: int,
    dc: int,
    wall_color: int,
    turn_by_color: dict[int, str],
    blocked: set[tuple[int, int]],
    global_same_color: set[tuple[int, int]],
    local_same_color: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], str] | None:
    preferred_side = _mismatch_side(dr, dc, wall_color, turn_by_color)
    options: list[tuple[list[tuple[int, int]], str]] = []

    if preferred_side == "row":
        wr, wc = end_r + dr, end_c
        if 0 <= wr < h and 0 <= wc < w:
            options.append(([(wr, wc)], "row"))
            line = [(wr, wc + d) for d in (-1, 0, 1)]
            if all(0 <= cc < w for _rr, cc in line):
                options.append((line, "row"))
    else:
        wr, wc = end_r, end_c + dc
        if 0 <= wr < h and 0 <= wc < w:
            options.append(([(wr, wc)], "col"))
            line = [(wr + d, wc) for d in (-1, 0, 1)]
            if all(0 <= rr < h for rr, _cc in line):
                options.append((line, "col"))

    if not options:
        return None

    _rng.shuffle(options)
    options.sort(key=lambda item: len(item[0]), reverse=(_rng.random() < 0.6))
    for cells, side in options:
        if any(cell in blocked for cell in cells):
            continue
        if not _same_color_clear(cells, global_same_color, local_same_color):
            continue
        return cells, side
    return None


def _advance_cells(
    cur_r: int,
    cur_c: int,
    dr: int,
    dc: int,
    steps: int,
    blocked: set[tuple[int, int]],
    wall_blockers: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    rr, cc = cur_r, cur_c
    cells: list[tuple[int, int]] = []
    for _ in range(steps):
        rr += dr
        cc += dc
        if not (0 <= rr < h and 0 <= cc < w):
            return None
        if (rr, cc) in blocked:
            return None
        if (rr + dr, cc) in wall_blockers:
            return None
        if (rr, cc + dc) in wall_blockers:
            return None
        cells.append((int(rr), int(cc)))
    return cells


def _plan_arrow(
    h: int,
    w: int,
    arrow_color: int,
    n_turns: int,
    turn_by_color: dict[int, str],
    occupied: set[tuple[int, int]],
    global_wall_cells: set[tuple[int, int]],
    wall_color_cells_global: dict[int, set[tuple[int, int]]],
    forced_turn_colors: list[int],
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int, int]],
    list[tuple[list[tuple[int, int]], int]],
    int,
] | None:
    legend_colors = list(turn_by_color)
    for _ in range(160):
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        r_lo, r_hi = (1 if dr == -1 else 0), (h - 2 if dr == 1 else h - 1)
        c_lo, c_hi = (1 if dc == -1 else 0), (w - 2 if dc == 1 else w - 1)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)
        mr, mc = sr - dr, sc - dc
        arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
        if any(not (0 <= r < h and 0 <= c < w) or (r, c) in occupied for r, c in arrow_pts):
            continue

        cur_r, cur_c = sr, sc
        cur_dr, cur_dc = dr, dc
        cur_col = int(arrow_color)
        ray_cells: list[tuple[int, int, int]] = []
        walls: list[tuple[list[tuple[int, int]], int]] = []
        ray_set: set[tuple[int, int]] = set()
        local_wall_cells: set[tuple[int, int]] = set()
        local_wall_color_cells = {color: set() for color in legend_colors}
        forced_used = 0
        ok = True

        for _turn_idx in range(n_turns):
            blocked = occupied | set(arrow_pts) | ray_set | local_wall_cells
            wall_blockers = global_wall_cells | local_wall_cells
            max_steps = _max_diagonal_steps(cur_r, cur_c, cur_dr, cur_dc, h, w)
            if max_steps < 2:
                ok = False
                break

            wall_color = (
                int(forced_turn_colors[forced_used])
                if forced_used < len(forced_turn_colors)
                else int(_rng.choice(legend_colors))
            )
            candidate_lengths = list(range(2, max_steps + 1))
            _rng.shuffle(candidate_lengths)
            candidate_lengths.sort(key=lambda n: abs(n - max_steps // 2))

            committed = False
            for seg_len in candidate_lengths:
                seg_cells = _advance_cells(
                    cur_r,
                    cur_c,
                    cur_dr,
                    cur_dc,
                    seg_len,
                    blocked,
                    wall_blockers,
                    h,
                    w,
                )
                if seg_cells is None:
                    continue
                end_r, end_c = seg_cells[-1]
                wall_choice = _pick_wall_cells(
                    h,
                    w,
                    end_r,
                    end_c,
                    cur_dr,
                    cur_dc,
                    wall_color,
                    turn_by_color,
                    blocked | set(seg_cells),
                    wall_color_cells_global[wall_color],
                    local_wall_color_cells[wall_color],
                )
                if wall_choice is None:
                    continue

                next_dir = _rotate_by_legend_color(cur_dr, cur_dc, wall_color, turn_by_color)
                if next_dir is None:
                    continue
                next_dr, next_dc = next_dir
                wall_cells, _side = wall_choice
                preview = _advance_cells(
                    end_r,
                    end_c,
                    next_dr,
                    next_dc,
                    2,
                    blocked | set(seg_cells) | set(wall_cells),
                    wall_blockers | set(wall_cells),
                    h,
                    w,
                )
                if preview is None:
                    continue

                for rr, cc in seg_cells:
                    ray_cells.append((int(rr), int(cc), int(cur_col)))
                    ray_set.add((int(rr), int(cc)))
                walls.append((wall_cells, int(wall_color)))
                local_wall_cells.update(wall_cells)
                local_wall_color_cells[wall_color].update(wall_cells)
                lr, lc, _old = ray_cells[-1]
                ray_cells[-1] = (int(lr), int(lc), int(wall_color))
                cur_r, cur_c = end_r, end_c
                cur_dr, cur_dc = int(next_dr), int(next_dc)
                cur_col = int(wall_color)
                forced_used += 1
                committed = True
                break

            if not committed:
                ok = False
                break

        if not ok:
            continue

        blocked = occupied | set(arrow_pts) | ray_set | local_wall_cells
        wall_blockers = global_wall_cells | local_wall_cells
        tail_steps = _max_diagonal_steps(cur_r, cur_c, cur_dr, cur_dc, h, w)
        if tail_steps < 2:
            continue
        tail_cells = _advance_cells(
            cur_r,
            cur_c,
            cur_dr,
            cur_dc,
            tail_steps,
            blocked,
            wall_blockers,
            h,
            w,
        )
        if tail_cells is None:
            continue
        for rr, cc in tail_cells:
            ray_cells.append((int(rr), int(cc), int(cur_col)))
            ray_set.add((int(rr), int(cc)))

        if len(ray_cells) < 5:
            continue

        new_cells = set(arrow_pts) | ray_set | local_wall_cells
        expected = len(arrow_pts) + len(ray_set) + len(local_wall_cells)
        if len(new_cells) != expected:
            continue
        if new_cells & occupied:
            continue

        return arrow_pts, ray_cells, walls, forced_used

    return None


def _trace_turn_events(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.array(grid, dtype=int)
    legend_h, turn_by_color = _parse_legend(g)
    if legend_h >= g.shape[0]:
        return []

    core = g[legend_h:, :]
    h, w = core.shape
    bg = int(get_bg(core))

    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(core, exclude={bg}):
        for coords in components_mask(core == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)

    wall_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(core != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }

    events: list[dict[str, object]] = []
    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
        cur_col = int(start_col)
        seen: set[tuple[tuple[int, int], int, int, int]] = set()

        while True:
            state = (cur, int(dr), int(dc), int(cur_col))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                break

            if (nr, nc) in wall_cells:
                wcol = int(core[nr, nc])
                turned = _rotate_by_legend_color(dr, dc, wcol, turn_by_color)
                if turned is None:
                    break
                events.append(
                    {
                        "kind": "front",
                        "wall_color": int(wcol),
                        "incoming": (int(dr), int(dc)),
                    }
                )
                cur_col = int(wcol)
                dr, dc = turned
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            touches: list[tuple[str, tuple[int, int]]] = []
            if (nr + int(dr), nc) in wall_cells:
                touches.append(("row", (int(nr + dr), int(nc))))
            if (nr, nc + int(dc)) in wall_cells:
                touches.append(("col", (int(nr), int(nc + dc))))
            if len(touches) != 1:
                if len(touches) > 1:
                    events.append({"kind": "ambiguous"})
                continue

            side, (wr, wc) = touches[0]
            wcol = int(core[wr, wc])
            turned = _rotate_by_legend_color(dr, dc, wcol, turn_by_color)
            if turned is None:
                break
            old_result = (-int(dr), int(dc)) if side == "row" else (int(dr), -int(dc))
            events.append(
                {
                    "kind": "side",
                    "wall_color": int(wcol),
                    "incoming": (int(dr), int(dc)),
                    "side": side,
                    "matches_old_rule": turned == old_result,
                }
            )
            cur_col = int(wcol)
            dr, dc = turned

    return events


def _behavior_metrics(grid: np.ndarray) -> dict[str, object]:
    legend_h, turn_by_color = _parse_legend(np.array(grid, dtype=int))
    events = _trace_turn_events(grid)
    side_events = [e for e in events if e.get("kind") == "side"]
    return {
        "legend_h": int(legend_h),
        "legend_colors": sorted(map(int, turn_by_color)),
        "total_turns": int(len([e for e in events if e.get("kind") in {"side", "front"}])),
        "side_turns": int(len(side_events)),
        "front_turns": int(len([e for e in events if e.get("kind") == "front"])),
        "ambiguous_turns": int(len([e for e in events if e.get("kind") == "ambiguous"])),
        "mismatch_turns": int(len([e for e in side_events if not bool(e["matches_old_rule"])])),
        "used_rule_colors": sorted({int(e["wall_color"]) for e in side_events}),
    }


def _gen_attempt() -> np.ndarray | None:
    total_h, w = _rng.randint(18, 20), _rng.randint(18, 20)
    h = total_h - LEGEND_H
    if h < 14:
        return None

    n_arrows = _rng.randint(2, 4)
    palette = PLAY_COLORS[:]
    _rng.shuffle(palette)
    wall_colors = palette[:2]
    arrow_colors = palette[2 : 2 + n_arrows]
    if len(arrow_colors) != n_arrows:
        return None

    turn_labels = ["cw", "ccw"]
    _rng.shuffle(turn_labels)
    turn_by_color = {int(wall_colors[i]): turn_labels[i] for i in range(2)}
    legend_order = wall_colors[:]
    _rng.shuffle(legend_order)

    target_total_turns = _rng.randint(4, min(8, n_arrows * 3))
    turn_counts = [1] * n_arrows
    for _ in range(target_total_turns - n_arrows):
        candidates = [i for i, count in enumerate(turn_counts) if count < 3]
        if not candidates:
            break
        turn_counts[_rng.choice(candidates)] += 1

    grid = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    global_wall_cells: set[tuple[int, int]] = set()
    wall_color_cells_global = {int(color): set() for color in wall_colors}
    forced_turn_colors = [int(wall_colors[0]), int(wall_colors[1]), int(wall_colors[0]), int(wall_colors[1])]

    for arrow_color, n_turns in zip(arrow_colors, turn_counts):
        planned = _plan_arrow(
            h,
            w,
            int(arrow_color),
            int(n_turns),
            turn_by_color,
            occupied,
            global_wall_cells,
            wall_color_cells_global,
            forced_turn_colors,
        )
        if planned is None:
            return None
        arrow_pts, ray_cells, walls, forced_used = planned

        for r, c in arrow_pts:
            grid[r, c] = int(arrow_color)
            out[r, c] = int(arrow_color)
        for r, c, ray_col in ray_cells:
            out[r, c] = int(ray_col)
        for wall_cells, wall_color in walls:
            for r, c in wall_cells:
                grid[r, c] = int(wall_color)
                out[r, c] = int(wall_color)
                global_wall_cells.add((int(r), int(c)))
                wall_color_cells_global[int(wall_color)].add((int(r), int(c)))

        occupied |= set(arrow_pts)
        occupied |= {(int(r), int(c)) for r, c, _col in ray_cells}
        occupied |= {(int(r), int(c)) for wall_cells, _wcol in walls for r, c in wall_cells}
        forced_turn_colors = forced_turn_colors[forced_used:]

    legend = _build_legend(w, turn_by_color, legend_order)
    full_grid = np.vstack([legend, grid])
    full_out = np.vstack([legend, out])

    metrics = _behavior_metrics(full_grid)
    if metrics["legend_h"] != LEGEND_H:
        return None
    if metrics["total_turns"] < 4:
        return None
    if metrics["front_turns"] != 0 or metrics["ambiguous_turns"] != 0:
        return None
    if metrics["side_turns"] != metrics["mismatch_turns"]:
        return None
    if set(metrics["used_rule_colors"]) != set(map(int, wall_colors)):
        return None
    if not np.array_equal(main(full_grid), full_out):
        return None
    return full_grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
