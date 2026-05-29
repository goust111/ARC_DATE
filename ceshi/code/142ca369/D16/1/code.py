# Variant of 142ca369
# Dimension: D16 — 颜色使用策略 (meta)
# Change: 2-7种非背景色，箭头和墙壁各自单色，射线颜色随反射继承墙壁色... → 颜色配对决定反射行为

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

# Design analysis for the D16 variant
# ----------------------------------
# source_task note:
# - The prompt points to 16de56c4 for "color_pairing", but that solver file is not
#   present in this workspace. Per instruction, that reference implementation is skipped.
#
# a) Concrete behavior and trace
# - Visual objects stay the same as the base task:
#   - 3-cell monochrome L-shaped arrows in a 2x2 box.
#   - Single-pixel or 3-pixel axis-aligned monochrome walls.
#   - Rays move diagonally from each arrow's right-angle corner.
# - The changed point is only "what a wall does when the ray meets it".
# - Old behavior:
#   - Any wall contact always reflects.
#   - The ray always adopts the wall color on that reflection.
# - New behavior:
#   - Compare current ray color with the touched wall color.
#   - Same-color pair => no reflection.
#     - Side-touch: ignore the wall and keep moving.
#     - Corner hit: pass into the wall cell and keep moving.
#   - Different-color pair => normal base-task reflection plus color inheritance.
# - Small side-touch example:
#   - Ray color = 2, direction = down-right, current advanced cell = (r,c).
#   - Wall sits at (r+1,c), i.e. horizontal wall that would flip dr.
#   - Old task: dr flips, (r,c) is recolored to wall color, path bends.
#   - Variant:
#     - if wall color = 2: dr does not flip, (r,c) stays 2, path continues straight.
#     - if wall color = 5: dr flips to -dr, (r,c) becomes 5, later path continues in 5.
#
# b) Complexity calibration
# - The new rule is binary and local: same-color vs different-color.
# - To keep that visible, the generator uses a small palette (2-5 active colors)
#   and deliberately includes both same-pair and different-pair interactions.
# - This avoids a degenerate regime where every wall happens to be different-colored,
#   which would collapse back to the original task.
#
# c) Hidden assumptions in the original solver that must change
# - Original corner-hit block assumed every wall cell is an impenetrable blocker.
#   That is invalid now because same-color corner hits must pass through.
# - Original side-touch blocks unconditionally flipped dr/dc when a wall was present.
#   That is invalid now because same-color touches must not flip.
# - Original recoloring of the bounce cell on every touch was harmless because every
#   touch changed state. In the variant, unconditional recoloring would be wrong for
#   same-color side-touch: there is no reflection and no state change.
# - The state cache already includes current color, so after introducing color-based
#   branching it remains sufficient to prevent loops.
#
# d) Placement legality
# - Generator still places arrows, ray cells, and walls with no cell overlap.
# - Planned interactions remain side-touch only in generated data, because that is the
#   cleanest way to make the same/different branch visually readable.
# - Same-color pass events place walls exactly where a reflection wall would have been,
#   but keep the ray direction unchanged. Since the wall is only side-adjacent, this
#   does not create overlap between the ray path and the wall cells.
#
# e) Visual observability
# - Each generated grid is forced to include both:
#   - at least one same-color pass-through event
#   - at least one different-color reflection event
# - This makes the rule inferable from few examples: some walls of the same color are
#   visibly "ignored", while different-colored walls still bend and recolor the ray.
#
# f) Visual cleanliness
# - The generator keeps the original sparse layout and prevents overlaps.
# - It also limits the number of arrows and interactions so the grid stays readable.
#
# g) Color-rule inferability
# - The color rule is intentionally low-arity: only equality vs inequality matters.
# - Colors repeat across arrows and walls, so the learner sees both outcomes for the
#   same wall geometry and can attribute the difference to the color pairing itself.


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


def _detect_arrows_and_walls(
    g: np.ndarray, bg: int
) -> tuple[list[tuple[int, list[tuple[int, int]]]], set[tuple[int, int]], dict[tuple[int, int], int]]:
    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)

    wall_colors = {
        (int(r), int(c)): int(g[r, c])
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }
    return arrows, arrow_cells, wall_colors


def _trace(grid: np.ndarray, collect_stats: bool = False):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    arrows, arrow_cells, wall_colors = _detect_arrows_and_walls(g, bg)
    wall_cells = set(wall_colors)

    out = g.copy()
    stats = {
        "same_color_passes": 0,
        "diff_color_reflections": 0,
        "same_corner_passes": 0,
        "same_side_passes": 0,
        "diff_corner_reflections": 0,
        "diff_side_reflections": 0,
    }

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
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in wall_cells:
                wcol = int(wall_colors[(nr, nc)])
                if int(cur_col) == int(wcol):
                    cur = (int(nr), int(nc))
                    stats["same_color_passes"] += 1
                    stats["same_corner_passes"] += 1
                    continue
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                stats["diff_color_reflections"] += 1
                stats["diff_corner_reflections"] += 1
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                wcol = int(wall_colors[(nr + 1, nc)])
                if int(cur_col) == int(wcol):
                    stats["same_color_passes"] += 1
                    stats["same_side_passes"] += 1
                else:
                    if int(g[nr, nc]) == int(bg):
                        out[nr, nc] = int(wcol)
                    cur_col = int(wcol)
                    dr = -int(dr)
                    stats["diff_color_reflections"] += 1
                    stats["diff_side_reflections"] += 1
            elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                wcol = int(wall_colors[(nr - 1, nc)])
                if int(cur_col) == int(wcol):
                    stats["same_color_passes"] += 1
                    stats["same_side_passes"] += 1
                else:
                    if int(g[nr, nc]) == int(bg):
                        out[nr, nc] = int(wcol)
                    cur_col = int(wcol)
                    dr = -int(dr)
                    stats["diff_color_reflections"] += 1
                    stats["diff_side_reflections"] += 1

            if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                wcol = int(wall_colors[(nr, nc + 1)])
                if int(cur_col) == int(wcol):
                    stats["same_color_passes"] += 1
                    stats["same_side_passes"] += 1
                else:
                    if int(g[nr, nc]) == int(bg):
                        out[nr, nc] = int(wcol)
                    cur_col = int(wcol)
                    dc = -int(dc)
                    stats["diff_color_reflections"] += 1
                    stats["diff_side_reflections"] += 1
            elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                wcol = int(wall_colors[(nr, nc - 1)])
                if int(cur_col) == int(wcol):
                    stats["same_color_passes"] += 1
                    stats["same_side_passes"] += 1
                else:
                    if int(g[nr, nc]) == int(bg):
                        out[nr, nc] = int(wcol)
                    cur_col = int(wcol)
                    dc = -int(dc)
                    stats["diff_color_reflections"] += 1
                    stats["diff_side_reflections"] += 1

    if collect_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - same ray/wall color => no reflection
    - different ray/wall color => original reflection + color inheritance
    """
    return _trace(grid, collect_stats=False)


def _wall_options(
    cur_r: int,
    cur_c: int,
    cur_dr: int,
    cur_dc: int,
    axis: str,
    H: int,
    W: int,
) -> list[list[tuple[int, int]]]:
    options: list[list[tuple[int, int]]] = []
    if axis == "dr":
        wr, wc = cur_r + cur_dr, cur_c
        if 0 <= wr < H:
            w3 = [(wr, wc + d) for d in (-1, 0, 1)]
            if all(0 <= c < W for _, c in w3):
                options.append(w3)
            options.append([(wr, wc)])
    else:
        wr, wc = cur_r, cur_c + cur_dc
        if 0 <= wc < W:
            w3 = [(wr + d, wc) for d in (-1, 0, 1)]
            if all(0 <= r < H for r, _ in w3):
                options.append(w3)
            options.append([(wr, wc)])
    _rng.shuffle(options)
    prefer_long = _rng.random() < 0.6
    options.sort(key=len, reverse=prefer_long)
    return options


def _pick_other_color(palette: list[int], cur_col: int) -> int | None:
    choices = [col for col in palette if int(col) != int(cur_col)]
    if not choices:
        return None
    return int(_rng.choice(choices))


def _gen_attempt() -> np.ndarray | None:
    """
    Output-first generator for the D16 variant.

    Planned events are side-touch interactions only:
    - pass_dr / pass_dc: wall has same color as ray, so no flip
    - reflect_dr / reflect_dc: wall has different color, so flip the chosen axis
    """
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_colors = _rng.randint(2, 5)
    palette = _rng.sample(list(range(1, 10)), k=n_colors)
    n_arrows = _rng.randint(2, 5)
    arrow_colors = [_rng.choice(palette) for _ in range(n_arrows)]

    n_events_per_arrow = [_rng.choices([1, 2, 3], weights=[2, 3, 2])[0] for _ in range(n_arrows)]
    total_events = sum(n_events_per_arrow)
    if total_events < 2:
        return None
    pair_kinds = ["same", "diff"] + [_rng.choice(["same", "diff"]) for _ in range(total_events - 2)]
    _rng.shuffle(pair_kinds)

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    used_colors: set[int] = set()

    pair_idx = 0
    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        used_colors.add(color)
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        event_specs = []
        for _ in range(n_events_per_arrow[ai]):
            axis = _rng.choice(["dr", "dc"])
            kind = pair_kinds[pair_idx]
            pair_idx += 1
            event_specs.append((axis, kind))

        placed = False
        for _ in range(100):
            r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
            c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)

            mr, mc = sr - dr, sc - dc
            arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
            if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
                continue

            cur_r, cur_c = sr, sc
            cur_dr, cur_dc = dr, dc
            cur_col = color
            ray_cells: list[tuple[int, int, int]] = []
            walls: list[tuple[list[tuple[int, int]], int]] = []
            ray_set: set[tuple[int, int]] = set()
            arrow_set = set(arrow_pts)
            ok = True

            for event_i, (axis, kind) in enumerate(event_specs):
                max_steps = min(
                    (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                    (W - 1 - cur_c) if cur_dc == 1 else cur_c,
                )
                remaining = len(event_specs) - event_i - 1
                min_tail = 2 * remaining
                if max_steps < 2 or max_steps - 1 < min_tail:
                    ok = False
                    break
                seg_len_hi = max(2, max_steps - 1 - min_tail)
                seg_len = _rng.randint(2, seg_len_hi)

                for _ in range(seg_len):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        ok = False
                        break
                    if (nr, nc) in occupied or (nr, nc) in ray_set or (nr, nc) in arrow_set:
                        ok = False
                        break
                    ray_cells.append((nr, nc, cur_col))
                    ray_set.add((nr, nc))
                    cur_r, cur_c = nr, nc
                if not ok:
                    break

                chosen_wall = None
                for wpxs in _wall_options(cur_r, cur_c, cur_dr, cur_dc, axis, H, W):
                    if any((r, c) in occupied or (r, c) in ray_set or (r, c) in arrow_set for r, c in wpxs):
                        continue
                    chosen_wall = wpxs
                    break
                if chosen_wall is None:
                    ok = False
                    break

                if kind == "same":
                    wcol = int(cur_col)
                else:
                    other = _pick_other_color(palette, cur_col)
                    if other is None:
                        ok = False
                        break
                    wcol = int(other)
                    lr, lc, _ = ray_cells[-1]
                    ray_cells[-1] = (lr, lc, wcol)
                    cur_col = int(wcol)
                    if axis == "dr":
                        cur_dr = -cur_dr
                    else:
                        cur_dc = -cur_dc
                walls.append((chosen_wall, wcol))
                used_colors.add(wcol)

            if not ok:
                continue

            max_steps = min(
                (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                (W - 1 - cur_c) if cur_dc == 1 else cur_c,
            )
            if max_steps < 2:
                continue
            for _ in range(max_steps):
                nr, nc = cur_r + cur_dr, cur_c + cur_dc
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                if (nr, nc) in occupied or (nr, nc) in ray_set or (nr, nc) in arrow_set:
                    ok = False
                    break
                ray_cells.append((nr, nc, cur_col))
                ray_set.add((nr, nc))
                cur_r, cur_c = nr, nc
            if not ok or len(ray_cells) < 4:
                continue

            new_cells = set(arrow_pts) | ray_set
            for wpxs, _ in walls:
                new_cells.update(wpxs)
            expected_count = len(arrow_pts) + len(ray_cells) + sum(len(w) for w, _ in walls)
            if len(new_cells) != expected_count or new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = out[r, c] = color
            for r, c, col in ray_cells:
                out[r, c] = col
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[r, c] = out[r, c] = wcol

            occupied |= new_cells
            placed = True
            break

        if not placed:
            return None

    if len(used_colors) < 2:
        return None

    solved_out, stats = _trace(grid, collect_stats=True)
    if not np.array_equal(solved_out, out):
        return None
    if stats["same_color_passes"] < 1 or stats["diff_color_reflections"] < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D16 variant of 142ca369."""
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")


def _variant_stats(grid: np.ndarray) -> dict[str, int]:
    """Helper for external validation scripts."""
    _out, stats = _trace(grid, collect_stats=True)
    return stats
