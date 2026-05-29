# Variant of 142ca369
# Dimension: D19 — 符号/图例编码 (symbol_instructions)
# Change: 无显式图例——规则完全从输入输出对中推断... → 网格边缘放置图例

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg


"""
Design analysis for the D19 variant
==================================

a) Concrete visual change and behavior trace
   - Original input: only arrows + colored walls on black background.
   - Variant input: add a 2-row legend strip on the top edge, with grey (5) background.
     Each legend entry is a 2-pixel mini-symbol tied to a wall color:
       row 0: [grey, wall_color, grey]
       row 1: [wall_color, grey, grey]  -> CCW turn
       row 1: [grey, grey, wall_color]  -> CW turn
     Entries are separated by solid-grey columns, so the solver can parse them.
   - Main playfield stays the same object-wise: L-shaped arrows, sparse colored walls,
     diagonal ray painting, and wall-color inheritance.
   - Core rule change: when a ray touches a wall of color C, it must first read the
     legend entry for C, then rotate 90 degrees clockwise or counterclockwise.
     The wall color is still inherited by the ray and still recolors the bounce cell.
   - Example trace at a bounce:
       original: ray direction (1,1), side-touch wall below -> flip dr -> (-1,1)
       variant:  same geometry, wall color = red, legend says "CW"
                 so (1,1) -> (1,-1)
       difference: geometry is no longer recoverable from local wall placement alone;
       the legend mediates the turn.

b) Complexity calibration
   - The legend uses exactly two wall colors per sample: one CW and one CCW.
     This keeps the codebook small enough for humans, while still forcing them to
     read the legend rather than infer a single fixed reflection rule.
   - Both legend colors are reused by actual walls in every generated sample.
   - The rest of the task keeps the original visual vocabulary, so the novelty is
     concentrated in D19 instead of spreading across multiple dimensions.

c) Implicit-assumption audit against the original solver
   - Original `main()` treated every non-arrow non-bg pixel as a wall inside a single
     playfield. That breaks once legend pixels are added, so this variant must parse
     and remove the legend region before arrow/wall detection.
   - Original side-touch handling assumed wall position fully determined the axis flip.
     That assumption is invalid now: wall color decides the turn. The new solver still
     detects contact geometrically, but consults the legend to choose CW vs CCW.
   - Original code could harmlessly inspect `g[nr, nc]` because all non-bg pixels in the
     playfield were part of the physical puzzle. In the variant, top-edge non-bg pixels
     are instructions, not obstacles, so they must be excluded from tracing.
   - Original sequential checks for dr-wall then dc-wall were okay because the generator
     never created ambiguous double-touch cells. The new solver makes that assumption
     explicit and stops on ambiguous double-touch instead of silently privileging order.

d) Placement legality
   - Grey (5) is reserved for the legend strip and never appears in the playfield.
     This makes legend parsing stable and prevents walls from leaking into the parser.
   - Arrows, ray cells, and walls remain non-overlapping in the playfield.
   - Legend entries are separated by full-grey columns, so no accidental arrow-like
     components are formed in the instruction strip.
   - Generator still uses output-first planning and validates `main(input)` exactly.

e) Visual observability
   - The top-edge legend is a salient, dedicated region, so humans can notice that it
     is not part of the obstacle field.
   - Every sample uses both turn types at least once, so the legend is behaviorally live.
   - Because the output retains the legend unchanged, the instructional area remains
     visible in both input and output and can be compared across examples.

f) Visual cleanliness
   - The legend is isolated into the top 2 rows; the rest of the grid is the playfield.
   - Playfield colors avoid grey to preserve a clean separation.
   - Wall shapes remain 1-pixel or 3-pixel axis-aligned segments, matching the original.

g) Color-change inferability
   - Only two wall colors are used for turning semantics per sample.
   - Those same colors appear in the legend and on the actual walls, giving a direct,
     low-entropy mapping from symbol to behavior.
   - Ray color inheritance remains unchanged, so color still has a consistent second role.

h) Quantitative anti-degeneracy target
   - A generated sample is considered behaviorally non-degenerate if, after removing the
     legend strip, the variant solver's playfield output differs from the original
     142ca369 solver's output on the same playfield.
   - The post-implementation check measures how often this happens and also confirms
     that both legend colors are used in bounce events.

i) New solver assumptions and generator guarantees
   - Assumption: legend rows are the contiguous top prefix containing grey (5).
     Guarantee: generator reserves grey exclusively for the legend strip.
   - Assumption: each legend color has exactly one 2-pixel entry.
     Guarantee: generator creates exactly two separated entries.
   - Assumption: each bounce cell touches at most one wall side.
     Guarantee: generator never places both dr-side and dc-side walls for one bounce.
   - Assumption: every playfield wall color is defined in the legend.
     Guarantee: generator draws all walls from the two legend colors only.
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
    if legend_h == 0:
        return 0, {}

    legend = grid[:legend_h, :]
    sep_cols = [c for c in range(w) if np.all(legend[:, c] == LEGEND_BG)]
    turn_by_color: dict[int, str] = {}
    for a, b in zip(sep_cols, sep_cols[1:]):
        interior = legend[:, a + 1 : b]
        if interior.shape[0] != LEGEND_H or interior.shape[1] < 2:
            continue
        colors = set(map(int, np.unique(interior)))
        colors.discard(LEGEND_BG)
        colors.discard(0)
        if len(colors) != 1:
            continue
        col = int(next(iter(colors)))
        pts = [(int(r), int(c)) for r, c in np.argwhere(interior == col)]
        if len(pts) != 2:
            continue
        lower = [c for r, c in pts if r == LEGEND_H - 1]
        if len(lower) != 1:
            continue
        turn_by_color[col] = "ccw" if lower[0] <= (interior.shape[1] - 1) // 2 else "cw"
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-arrows; wall colors now use a top-edge legend.

    Input structure:
    - Top 2 rows: grey legend strip with one mini-symbol per wall color.
    - Below: original 142ca369 playfield with arrows and walls.

    Rule:
    - Each arrow emits a diagonal ray away from its missing corner.
    - On hitting or side-touching a wall of color C, the ray adopts color C and
      rotates 90 degrees according to the legend entry for C:
        lower-left pixel -> counterclockwise
        lower-right pixel -> clockwise
    - Bounce cell recoloring and arrow blocking remain as in the original task.
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
                turn = turn_by_color.get(wcol)
                if turn is None:
                    break
                if int(core[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = _apply_turn(dr, dc, turn)
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            touches: list[tuple[int, int]] = []
            if (nr + dr, nc) in wall_cells:
                touches.append((int(nr + dr), int(nc)))
            if (nr, nc + dc) in wall_cells:
                touches.append((int(nr), int(nc + dc)))
            if len(touches) > 1:
                break
            if len(touches) == 1:
                wr, wc = touches[0]
                wcol = int(core[wr, wc])
                turn = turn_by_color.get(wcol)
                if turn is None:
                    break
                if int(core[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr, dc = _apply_turn(dr, dc, turn)

    result = g.copy()
    result[legend_h:, :] = out
    return result


def _build_legend(width: int, turn_by_color: dict[int, str], order: list[int]) -> np.ndarray:
    legend = np.full((LEGEND_H, width), LEGEND_BG, dtype=int)
    for i, col in enumerate(order):
        start = 1 + 4 * i
        legend[0, start + 1] = int(col)
        legend[1, start + (0 if turn_by_color[int(col)] == "ccw" else 2)] = int(col)
    return legend


def _wall_options(cur_r: int, cur_c: int, cur_dr: int, cur_dc: int, h: int, w: int) -> list[list[tuple[int, int]]]:
    options: list[list[tuple[int, int]]] = []

    wr, wc = cur_r + cur_dr, cur_c
    if 0 <= wr < h and 0 <= wc < w:
        options.append([(wr, wc)])

    wr, wc = cur_r, cur_c + cur_dc
    if 0 <= wr < h and 0 <= wc < w:
        options.append([(wr, wc)])

    _rng.shuffle(options)
    return options


def _remaining_steps(r: int, c: int, dr: int, dc: int, h: int, w: int) -> int:
    return min((h - 1 - r) if dr == 1 else r, (w - 1 - c) if dc == 1 else c)


def _gen_attempt() -> np.ndarray | None:
    total_h, w = _rng.randint(18, 20), _rng.randint(18, 20)
    h = total_h - LEGEND_H
    if h < 14:
        return None

    palette = PLAY_COLORS[:]
    _rng.shuffle(palette)
    wall_colors = palette[:2]
    arrow_color = int(palette[2])

    turn_by_color = {int(wall_colors[0]): "cw", int(wall_colors[1]): "ccw"}
    legend_order = wall_colors[:]
    _rng.shuffle(legend_order)

    grid = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()

    def original_turn(dr: int, dc: int, side: str) -> tuple[int, int]:
        return (-int(dr), int(dc)) if side == "dr" else (int(dr), -int(dc))

    def wall_options_for_side(
        side: str, cur_r: int, cur_c: int, cur_dr: int, cur_dc: int
    ) -> list[list[tuple[int, int]]]:
        options: list[list[tuple[int, int]]] = []
        if side == "dr":
            wr, wc = cur_r + cur_dr, cur_c
            if 0 <= wr < h and 0 <= wc < w:
                options.append([(wr, wc)])
        else:
            wr, wc = cur_r, cur_c + cur_dc
            if 0 <= wr < h and 0 <= wc < w:
                options.append([(wr, wc)])
        _rng.shuffle(options)
        return options

    def trace_segment(
        cur_r: int,
        cur_c: int,
        cur_dr: int,
        cur_dc: int,
        steps: int,
        arrow_set: set[tuple[int, int]],
        blocked_cells: set[tuple[int, int]],
        local_walls: set[tuple[int, int]],
    ) -> list[tuple[int, int]] | None:
        seg: list[tuple[int, int]] = []
        rr, cc = cur_r, cur_c
        for _ in range(steps):
            nr, nc = rr + cur_dr, cc + cur_dc
            if not (0 <= nr < h and 0 <= nc < w):
                return None
            if (nr, nc) in occupied or (nr, nc) in arrow_set or (nr, nc) in blocked_cells or (nr, nc) in local_walls:
                return None
            if (nr + cur_dr, nc) in local_walls:
                return None
            if (nr, nc + cur_dc) in local_walls:
                return None
            seg.append((int(nr), int(nc)))
            rr, cc = int(nr), int(nc)
        return seg

    def trace_tail(
        cur_r: int,
        cur_c: int,
        cur_dr: int,
        cur_dc: int,
        arrow_set: set[tuple[int, int]],
        blocked_cells: set[tuple[int, int]],
        local_walls: set[tuple[int, int]],
        cur_col: int,
    ) -> list[tuple[int, int, int]] | None:
        seg: list[tuple[int, int, int]] = []
        rr, cc = cur_r, cur_c
        while True:
            nr, nc = rr + cur_dr, cc + cur_dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if (nr, nc) in occupied or (nr, nc) in arrow_set or (nr, nc) in blocked_cells or (nr, nc) in local_walls:
                return None
            if (nr + cur_dr, nc) in local_walls:
                return None
            if (nr, nc + cur_dc) in local_walls:
                return None
            seg.append((int(nr), int(nc), int(cur_col)))
            rr, cc = int(nr), int(nc)
        return seg if len(seg) >= 2 else None

    turn_seq = [("cw", int(wall_colors[0])), ("ccw", int(wall_colors[1]))]
    if _rng.random() < 0.5:
        turn_seq.reverse()

    placed = False
    for _ in range(300):
        dr0, dc0 = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        r_lo, r_hi = (1 if dr0 == -1 else 0), (h - 2 if dr0 == 1 else h - 1)
        c_lo, c_hi = (1 if dc0 == -1 else 0), (w - 2 if dc0 == 1 else w - 1)
        if r_lo > r_hi or c_lo > c_hi:
            continue
        sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)
        mr, mc = sr - dr0, sc - dc0
        arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
        arrow_set = set((int(r), int(c)) for r, c in arrow_pts)
        if any(not (0 <= r < h and 0 <= c < w) or (r, c) in occupied for r, c in arrow_set):
            continue

        max1 = _remaining_steps(sr, sc, dr0, dc0, h, w)
        if max1 < 5:
            continue
        side1_candidates = [
            side for side in ("dr", "dc") if _apply_turn(dr0, dc0, turn_seq[0][0]) != original_turn(dr0, dc0, side)
        ]
        if not side1_candidates:
            continue
        side1 = _rng.choice(side1_candidates)

        seg1_lengths = list(range(2, max1))
        _rng.shuffle(seg1_lengths)
        for seg1_len in seg1_lengths:
            first = trace_segment(sr, sc, dr0, dc0, seg1_len, arrow_set, set(), set())
            if first is None:
                continue
            br1, bc1 = first[-1]
            wall1_opts = wall_options_for_side(side1, br1, bc1, dr0, dc0)
            _rng.shuffle(wall1_opts)
            for wall1 in wall1_opts:
                wall1_set = set((int(r), int(c)) for r, c in wall1)
                if any((r, c) in arrow_set or (r, c) in set(first) for r, c in wall1_set):
                    continue
                dr1, dc1 = _apply_turn(dr0, dc0, turn_seq[0][0])
                max2 = _remaining_steps(br1, bc1, dr1, dc1, h, w)
                if max2 < 4:
                    continue
                side2_candidates = [
                    side
                    for side in ("dr", "dc")
                    if _apply_turn(dr1, dc1, turn_seq[1][0]) != original_turn(dr1, dc1, side)
                ]
                if not side2_candidates:
                    continue
                side2 = _rng.choice(side2_candidates)
                seg2_lengths = list(range(2, max2))
                _rng.shuffle(seg2_lengths)
                blocked1 = set(first)
                for seg2_len in seg2_lengths:
                    second = trace_segment(br1, bc1, dr1, dc1, seg2_len, arrow_set, blocked1, wall1_set)
                    if second is None:
                        continue
                    br2, bc2 = second[-1]
                    wall2_opts = wall_options_for_side(side2, br2, bc2, dr1, dc1)
                    _rng.shuffle(wall2_opts)
                    for wall2 in wall2_opts:
                        wall2_set = set((int(r), int(c)) for r, c in wall2)
                        if any((r, c) in arrow_set or (r, c) in blocked1 or (r, c) in set(second) or (r, c) in wall1_set for r, c in wall2_set):
                            continue
                        dr2, dc2 = _apply_turn(dr1, dc1, turn_seq[1][0])
                        tail = trace_tail(
                            br2,
                            bc2,
                            dr2,
                            dc2,
                            arrow_set,
                            blocked1 | set(second),
                            wall1_set | wall2_set,
                            turn_seq[1][1],
                        )
                        if tail is None:
                            continue

                        ray_cells = [(int(r), int(c), int(arrow_color)) for r, c in first]
                        lr, lc, _old = ray_cells[-1]
                        ray_cells[-1] = (int(lr), int(lc), int(turn_seq[0][1]))
                        second_cells = [(int(r), int(c), int(turn_seq[0][1])) for r, c in second]
                        lr, lc, _old = second_cells[-1]
                        second_cells[-1] = (int(lr), int(lc), int(turn_seq[1][1]))
                        ray_cells.extend(second_cells)
                        ray_cells.extend(tail)

                        for r, c in arrow_set:
                            grid[r, c] = out[r, c] = int(arrow_color)
                        for r, c, col in ray_cells:
                            out[r, c] = int(col)
                        for r, c in wall1_set | wall2_set:
                            wall_col = int(turn_seq[0][1]) if (r, c) in wall1_set else int(turn_seq[1][1])
                            grid[r, c] = out[r, c] = wall_col
                        occupied.update(arrow_set)
                        occupied.update((int(r), int(c)) for r, c, _col in ray_cells)
                        occupied.update(wall1_set | wall2_set)
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                break
        if placed:
            break

    if not placed:
        return None

    legend = _build_legend(w, turn_by_color, legend_order)
    full_grid = np.vstack([legend, grid])
    full_out = np.vstack([legend, out])

    if not np.array_equal(main(full_grid), full_out):
        return None
    return full_grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
