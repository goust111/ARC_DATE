# Variant of 142ca369
# Dimension: D17 — 射线段长度约束 (meta)
# Change: 反射之间的射线段长度为2-16格（由网格空间自然决定）... → 段长度递增（斐波那契式）

from __future__ import annotations

import random as _rng
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import colors_present, components_mask, get_bg


# Design analysis for the D17 variant
#
# a) Concrete behavior:
# - The visible objects stay the same: 3-pixel L-arrows, single-pixel/3-pixel walls,
#   diagonal rays with wall-color inheritance.
# - The changed behavior is only "when a side wall is allowed to reflect the ray":
#   the 1st reflected segment must have length 2, the 2nd length 3, the 3rd length 5.
# - Therefore output rays show short/medium/long diagonal runs before successive bounces.
# - Small trace:
#   Original: if a side wall sits next to the 1st ray cell, the ray bounces immediately.
#   Variant: the same early wall is ignored; the first bounce can only happen after 2 cells.
#   This is how the generator creates non-degenerate "decoy walls".
#
# b) Complexity calibration:
# - The sequence is fixed to Fibonacci-style 2,3,5. This is small enough to infer from
#   2-4 ARC examples, but different enough from the original free-form lengths.
# - A sample typically contains one 3-bounce arrow and one 2-bounce arrow, so the pattern
#   repeats inside a single grid and is not a one-off accident.
#
# c) Hidden assumptions in the original solver that must change:
# - Original side-bounce code assumed "any adjacent wall in the travel direction reflects".
#   That is invalid here because early adjacent walls must be ignored.
# - Original cycle detection ignored segment index / segment step count; here both are part
#   of the state because hitting the same cell before or after a scheduled bounce differs.
# - Original generator freely chose any segment length from available space; here exact
#   lengths 2,3,5 must be planned before placing the bounce wall.
#
# d) Placement legality:
# - Rays, arrows, true walls, and decoy walls never overlap.
# - The generator places arrows first, then exact-length ray cells, then true bounce walls,
#   then one early decoy wall per arrow, and finally validates with the variant solver.
# - Decoys are only placed on the first segment after step 1, which keeps them visually
#   obvious while avoiding accidental later interactions.
#
# e) Visual observability:
# - Every generated grid includes at least one 3-bounce arrow, so the lengths 2,3,5 appear.
# - Decoys make the rule observable: users see a wall beside the first segment after 1 step,
#   but the output ray does not bounce there.
#
# f) Visual cleanliness:
# - Grid size stays near the original (19-20), arrow count is reduced to 2-3, and decoys
#   are single pixels only. This keeps the scene readable despite the extra clue walls.
#
# g) Color inference:
# - Color behavior is unchanged except at true bounces: rays inherit the true wall color.
#   Decoy walls do not recolor the ray, giving a structural clue without adding a new
#   color-mapping rule.
#
# h) Variant-specific generator guard:
# - The generator rejects any sample whose output also matches the legacy solver, so every
#   accepted input genuinely depends on the new segment-length rule.
#
# i) New solver assumptions audited against generation:
# - `main()` assumes side reflections happen only at segment lengths 2,3,5.
#   `generate_input()` always places true bounce walls exactly at those milestones.
# - `main()` assumes at most one intended side bounce per milestone in the generated data.
#   The generator enforces a single bounce wall near each milestone.
# - `main()` treats early diagonal wall hits as blockers; the generator never places such
#   walls on the planned path.


def _fib_segment_lengths(n: int) -> list[int]:
    seq = [2, 3]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]


def _is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[tuple[int, int]]):
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


def _detect_arrows_and_walls(g: np.ndarray):
    bg = int(get_bg(g))
    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)

    wall_map = {
        (int(r), int(c)): int(g[r, c])
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }
    return bg, arrows, arrow_cells, wall_map


def _choose_color(palette: list[int], avoid: set[int]) -> int:
    choices = [c for c in palette if c not in avoid]
    return int(_rng.choice(choices if choices else palette))


def _side_hits(
    nr: int,
    nc: int,
    dr: int,
    dc: int,
    wall_map: dict[tuple[int, int], int],
) -> list[tuple[str, int]]:
    hits: list[tuple[str, int]] = []
    if dr == 1 and (nr + 1, nc) in wall_map:
        hits.append(("dr", int(wall_map[(nr + 1, nc)])))
    elif dr == -1 and (nr - 1, nc) in wall_map:
        hits.append(("dr", int(wall_map[(nr - 1, nc)])))

    if dc == 1 and (nr, nc + 1) in wall_map:
        hits.append(("dc", int(wall_map[(nr, nc + 1)])))
    elif dc == -1 and (nr, nc - 1) in wall_map:
        hits.append(("dc", int(wall_map[(nr, nc - 1)])))
    return hits


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Original task solver, kept only to reject degenerate variant samples."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, arrows, arrow_cells, wall_map = _detect_arrows_and_walls(g)
    wall_cells = set(wall_map)
    out = g.copy()

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
        cur_col = int(start_col)
        seen = set()

        while True:
            state = (cur, int(dr), int(dc), int(cur_col))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in wall_cells:
                wcol = int(g[nr, nc])
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                wcol = int(g[nr + 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)
            elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                wcol = int(g[nr - 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)

            if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                wcol = int(g[nr, nc + 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)
            elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                wcol = int(g[nr, nc - 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-arrows, but allow side reflections only after
    Fibonacci-length segments: 2 cells, then 3 cells, then 5 cells.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, arrows, arrow_cells, wall_map = _detect_arrows_and_walls(g)
    wall_cells = set(wall_map)

    out = g.copy()

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue

        (dr, dc), cur = info
        cur_col = int(start_col)
        segment_lengths = _fib_segment_lengths(3)
        segment_idx = 0
        steps_in_segment = 0
        seen = set()

        while True:
            state = (cur, int(dr), int(dc), int(cur_col), int(segment_idx), int(steps_in_segment))
            if state in seen:
                break
            seen.add(state)

            target_len = segment_lengths[segment_idx] if segment_idx < len(segment_lengths) else None
            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in wall_cells:
                if target_len is None or steps_in_segment != target_len:
                    break
                wcol = int(wall_map[(nr, nc)])
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                segment_idx += 1
                steps_in_segment = 0
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            steps_in_segment += 1
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if target_len is None or steps_in_segment != target_len:
                continue

            hits = _side_hits(nr, nc, int(dr), int(dc), wall_map)
            if not hits:
                continue

            for axis, wcol in hits:
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                if axis == "dr":
                    dr = -int(dr)
                else:
                    dc = -int(dc)

            segment_idx += 1
            steps_in_segment = 0

    return out


def _wall_options(
    cur_r: int,
    cur_c: int,
    cur_dr: int,
    cur_dc: int,
    H: int,
    W: int,
) -> list[tuple[list[tuple[int, int]], str]]:
    options: list[tuple[list[tuple[int, int]], str]] = []

    wr, wc = cur_r + cur_dr, cur_c
    if 0 <= wr < H:
        options.append(([(wr, wc)], "dr"))
        if 0 <= wc - 1 and wc + 1 < W:
            options.append(([(wr, wc - 1), (wr, wc), (wr, wc + 1)], "dr"))

    wr2, wc2 = cur_r, cur_c + cur_dc
    if 0 <= wc2 < W:
        options.append(([(wr2, wc2)], "dc"))
        if 0 <= wr2 - 1 and wr2 + 1 < H:
            options.append(([(wr2 - 1, wc2), (wr2, wc2), (wr2 + 1, wc2)], "dc"))

    _rng.shuffle(options)
    return options


def _plan_arrow(
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    color: int,
    palette: list[int],
    n_bounces: int,
):
    seg_targets = _fib_segment_lengths(n_bounces)

    for _ in range(180):
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
        c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)
        mr, mc = sr - dr, sc - dc
        arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
        arrow_set = set(arrow_pts)
        if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
            continue

        cur_r, cur_c = sr, sc
        cur_dr, cur_dc = dr, dc
        cur_col = int(color)
        blocked = set(occupied) | arrow_set
        ray_cells: list[tuple[int, int, int]] = []
        true_walls: list[tuple[list[tuple[int, int]], int]] = []
        segment_records: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
        ok = True

        for target_len in seg_targets:
            seg_dir = (cur_dr, cur_dc)
            seg_cells: list[tuple[int, int]] = []

            for _step in range(target_len):
                nr, nc = cur_r + cur_dr, cur_c + cur_dc
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                if (nr, nc) in blocked:
                    ok = False
                    break
                ray_cells.append((nr, nc, cur_col))
                seg_cells.append((nr, nc))
                blocked.add((nr, nc))
                cur_r, cur_c = nr, nc
            if not ok:
                break

            options = _wall_options(cur_r, cur_c, cur_dr, cur_dc, H, W)
            chosen_wall = None
            for wpxs, flip in options:
                if any((r, c) in blocked for r, c in wpxs):
                    continue
                chosen_wall = (wpxs, flip)
                break
            if chosen_wall is None:
                ok = False
                break

            wpxs, flip = chosen_wall
            wcol = _choose_color(palette, {cur_col})
            true_walls.append((wpxs, wcol))
            for cell in wpxs:
                blocked.add(cell)

            lr, lc, _ = ray_cells[-1]
            ray_cells[-1] = (lr, lc, wcol)
            cur_col = int(wcol)
            segment_records.append((seg_dir, seg_cells))

            if flip == "dr":
                cur_dr = -cur_dr
            else:
                cur_dc = -cur_dc

        if not ok:
            continue

        tail_len = 0
        while True:
            nr, nc = cur_r + cur_dr, cur_c + cur_dc
            if not (0 <= nr < H and 0 <= nc < W):
                break
            if (nr, nc) in blocked:
                ok = False
                break
            ray_cells.append((nr, nc, cur_col))
            blocked.add((nr, nc))
            cur_r, cur_c = nr, nc
            tail_len += 1
        if not ok or tail_len < 2:
            continue

        # One early side-wall decoy on the first segment forces divergence from the
        # legacy rule while remaining visually simple and easy to infer.
        decoys: list[tuple[list[tuple[int, int]], int]] = []
        if not segment_records or not segment_records[0][1]:
            continue
        first_dir, first_cells = segment_records[0]
        fr, fc = first_cells[0]
        fdr, fdc = first_dir
        decoy_candidates = [(fr + fdr, fc), (fr, fc + fdc)]
        _rng.shuffle(decoy_candidates)
        placed_decoy = False
        for rr, cc in decoy_candidates:
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if (rr, cc) in blocked:
                continue
            dcol = _choose_color(palette, {color, ray_cells[0][2]})
            decoys.append(([(rr, cc)], dcol))
            blocked.add((rr, cc))
            placed_decoy = True
            break
        if not placed_decoy:
            continue

        new_cells = set(arrow_pts) | {(r, c) for r, c, _ in ray_cells}
        expected_count = len(arrow_pts) + len(ray_cells)
        for wpxs, _ in true_walls + decoys:
            new_cells.update(wpxs)
            expected_count += len(wpxs)
        if len(new_cells) != expected_count or new_cells & occupied:
            continue

        return arrow_pts, ray_cells, true_walls, decoys, new_cells

    return None


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(19, 20), _rng.randint(19, 20)
    n_arrows = _rng.randint(2, 3)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]

    bounce_counts = [3, 2]
    if n_arrows == 3:
        bounce_counts.append(_rng.choice([1, 2]))

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    total_decoys = 0

    for color, n_bounces in zip(arrow_colors, bounce_counts):
        planned = _plan_arrow(H, W, occupied, int(color), palette, int(n_bounces))
        if planned is None:
            return None

        arrow_pts, ray_cells, true_walls, decoys, new_cells = planned

        for r, c in arrow_pts:
            grid[r, c] = int(color)
            out[r, c] = int(color)
        for r, c, col in ray_cells:
            out[r, c] = int(col)
        for wpxs, wcol in true_walls + decoys:
            for r, c in wpxs:
                grid[r, c] = int(wcol)
                out[r, c] = int(wcol)

        occupied |= new_cells
        total_decoys += len(decoys)

    if total_decoys == 0:
        return None
    if not np.array_equal(main(grid), out):
        return None
    if np.array_equal(_legacy_main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a valid D17 variant input for task 142ca369."""
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
