# Variant of 142ca369
# Dimension: D6 — 射线终止条件 (dst)
# Change: 射线离开网格边界时终止（out of grid）... → 到达目标像素时终止

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

# Design analysis for the D6 variant
# ----------------------------------
# variant_spec:
#   preserved_core:
#     - Grid size stays 18-20 x 18-20 with bg = 0.
#     - Arrows stay as 3-cell monochrome L-shapes in a 2x2 box.
#     - Rays still move diagonally, inherit wall color on bounce, and use the
#       same corner-hit / side-touch reflection logic as the original task.
#   d6_change:
#     - Each arrow now has one dedicated target pixel.
#     - The target is a single isolated non-arrow pixel. When the ray's next
#       diagonal step lands on it, the ray enters that cell and immediately
#       terminates instead of continuing until it leaves the grid.
#     - The target pixel keeps its own marker color in the output, so the final
#       image visibly shows a path ending at a designated dot rather than at the
#       border.
#   necessary_adaptation:
#     - Original walls could be 1-pixel or 3-pixel segments. That becomes
#       visually ambiguous once singleton target pixels are introduced, because a
#       solver could not distinguish "wall singleton" from "target singleton" by
#       local shape alone.
#     - To keep D6 observable and inferable, the generator now uses only 3-pixel
#       axis-aligned wall segments. Therefore every non-arrow singleton is a
#       target, which the solver can detect robustly.
#   behavior_trace:
#     - Original: last segment keeps painting diagonal bg cells until the next
#       step would leave the grid.
#     - Variant: last segment paints as usual, but when the next diagonal cell
#       is the target marker, the ray steps into that target cell and stops. No
#       cells beyond the target are painted even if the original rule would have
#       continued further.
#   implicit-assumption-audit:
#     - Original `wall_cells = all non-bg non-arrow cells` is no longer valid:
#       target pixels would be mistaken for walls and cause reflections. The new
#       solver splits non-arrow components into singleton targets vs multi-cell
#       walls.
#     - Original generator assumed the final segment should run to the boundary.
#       The new generator must instead leave spare room before the boundary and
#       place a target on the ray path.
#     - Because target cells remain non-bg in both input and output, the solver
#       must stop before wall/side-bounce handling on that step.
#   legality:
#     - Targets never overlap arrows, walls, or painted ray cells.
#     - Targets are kept off the border so "stop at target" is visually distinct
#       from "stop because there is nowhere else to go".
#     - All same-color targets are isolated 4-connected singletons.
#   observability:
#     - A repeated dedicated target color appears as several isolated dots.
#     - Rays terminate in the interior of the board at those dots, making the
#       changed stop condition visible from input/output pairs.


def _is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
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

    # The ray goes away from the missing corner.
    dr = 1 if mr == r0 else -1
    dc = 1 if mc == c0 else -1

    deg = {p: 0 for p in pts}
    for r, c in pts:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in pts_set:
                deg[(r, c)] += 1
    start = max(deg, key=lambda p: deg[p])  # right-angle corner
    return (int(dr), int(dc)), (int(start[0]), int(start[1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-shaped arrows, reflecting off walls, but stop at
    designated singleton target pixels instead of running out of the grid.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    other_components: list[list[tuple[int, int]]] = []

    # Detect arrows first; all remaining singleton components are targets and
    # all remaining multi-cell components are walls.
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)
            else:
                other_components.append(pts)

    target_cells: set[tuple[int, int]] = {pts[0] for pts in other_components if len(pts) == 1}
    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c))
        for pts in other_components
        if len(pts) != 1
        for r, c in pts
    }

    out = g.copy()

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

            # D6 change: entering a target cell terminates the ray immediately.
            if (nr, nc) in target_cells:
                cur = (int(nr), int(nc))
                break

            # Diagonal hit into a wall: corner reflection.
            if (nr, nc) in wall_cells:
                wcol = int(g[nr, nc])
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                continue

            # Arrow pixels act like blockers (sources only).
            if (nr, nc) in arrow_cells:
                break

            # Advance.
            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            # Side-touch bounces only when a wall lies in the direction of travel.
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


def _orth_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) == 1


def _gen_attempt() -> np.ndarray | None:
    """Output-first generator for the D6 target-stop variant."""
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(2, 6)

    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]
    target_color = palette[n_arrows]
    wall_palette = [c for c in palette if c != target_color]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    target_cells: set[tuple[int, int]] = set()

    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = _rng.choices([0, 1, 2, 3, 4], weights=[1, 3, 3, 2, 1])[0]

        placed = False
        for _ in range(120):
            r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
            c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)

            mr, mc = sr - dr, sc - dc
            arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
            if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
                continue

            cur_r, cur_c = int(sr), int(sc)
            cur_dr, cur_dc = int(dr), int(dc)
            cur_col = int(color)
            ray_cells: list[tuple[int, int, int]] = []
            walls: list[tuple[list[tuple[int, int]], int]] = []
            ray_set: set[tuple[int, int]] = set()
            arrow_set = set(arrow_pts)
            ok = True

            for _seg in range(n_bounces):
                max_steps = min(
                    (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                    (W - 1 - cur_c) if cur_dc == 1 else cur_c,
                )
                if max_steps < 4:
                    ok = False
                    break
                seg_len = _rng.randint(3, max(3, max_steps - 1))

                for _ in range(seg_len):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        ok = False
                        break
                    if (nr, nc) in occupied or (nr, nc) in ray_set or (nr, nc) in arrow_set:
                        ok = False
                        break
                    ray_cells.append((int(nr), int(nc), int(cur_col)))
                    ray_set.add((int(nr), int(nc)))
                    cur_r, cur_c = int(nr), int(nc)
                if not ok:
                    break

                options: list[tuple[list[tuple[int, int]], str]] = []

                wr, wc = cur_r + cur_dr, cur_c
                if 0 <= wr < H:
                    w3 = [(int(wr), int(wc + d)) for d in (-1, 0, 1)]
                    if all(0 <= c < W for _, c in w3):
                        options.append((w3, "dr"))

                wr2, wc2 = cur_r, cur_c + cur_dc
                if 0 <= wc2 < W:
                    w3v = [(int(wr2 + d), int(wc2)) for d in (-1, 0, 1)]
                    if all(0 <= r < H for r, _ in w3v):
                        options.append((w3v, "dc"))

                if not options:
                    ok = False
                    break

                _rng.shuffle(options)
                chosen_wall = None
                for wpxs, flip in options:
                    if any((r, c) in occupied or (r, c) in ray_set or (r, c) in arrow_set for r, c in wpxs):
                        continue
                    chosen_wall = (wpxs, flip)
                    break
                if chosen_wall is None:
                    ok = False
                    break

                wpxs, flip = chosen_wall
                wcol = int(_rng.choice(wall_palette))
                walls.append((wpxs, wcol))

                lr, lc, _ = ray_cells[-1]
                ray_cells[-1] = (int(lr), int(lc), int(wcol))
                cur_col = int(wcol)
                if flip == "dr":
                    cur_dr = -int(cur_dr)
                else:
                    cur_dc = -int(cur_dc)

            if not ok:
                continue

            max_steps = min(
                (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                (W - 1 - cur_c) if cur_dc == 1 else cur_c,
            )
            if max_steps < 3:
                continue

            target_steps = _rng.randint(2, max_steps - 1)
            target = None
            for step_idx in range(1, target_steps + 1):
                nr, nc = cur_r + cur_dr, cur_c + cur_dc
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                if (nr, nc) in occupied or (nr, nc) in ray_set or (nr, nc) in arrow_set:
                    ok = False
                    break
                cur_r, cur_c = int(nr), int(nc)
                if step_idx == target_steps:
                    target = (int(nr), int(nc))
                else:
                    ray_cells.append((int(nr), int(nc), int(cur_col)))
                    ray_set.add((int(nr), int(nc)))
            if not ok or target is None:
                continue

            if target[0] in (0, H - 1) or target[1] in (0, W - 1):
                continue
            if any(_orth_adjacent(target, other) for other in target_cells):
                continue

            if len(ray_cells) < 3:
                continue

            new_cells = set(arrow_pts) | ray_set | {target}
            for wpxs, _ in walls:
                new_cells.update(wpxs)
            expected_count = len(arrow_pts) + len(ray_set) + 1 + sum(len(w) for w, _ in walls)
            if len(new_cells) != expected_count or (new_cells & occupied):
                continue

            for r, c in arrow_pts:
                grid[r, c] = out[r, c] = int(color)
            for r, c, col in ray_cells:
                out[r, c] = int(col)
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[r, c] = out[r, c] = int(wcol)
            grid[target] = out[target] = int(target_color)

            occupied |= new_cells
            target_cells.add(target)
            placed = True
            break

        if not placed:
            return None

    if len(target_cells) != n_arrows:
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D6 target-stop variant."""
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
