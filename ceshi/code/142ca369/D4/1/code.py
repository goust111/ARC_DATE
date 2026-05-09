# Variant of 142ca369
# Dimension: D4 — 反射时颜色继承规则 (trajectory)
# Change: color_change_on_bounce——射线在每次反射时采用墙壁的颜色，反射点也被重新涂为墙壁颜色... → 保持初始颜色不变

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg


# Design analysis for this D4 variant
# a) Concrete behavior:
#    Objects stay the same as the original task: 3-cell L-shaped arrows plus 1-cell or 3-cell walls.
#    The only semantic change is the ray color after a bounce. Originally, if a red ray touched a grey wall,
#    the bounce cell was recolored grey and all later ray cells also became grey. In this variant, that same
#    ray still reflects at the same location and with the same direction flip, but every ray cell before and
#    after the bounce remains red. The wall color is now only a reflector marker, not a paint source.
#    Example: arrow color 2, vertical wall color 5 adjacent to bounce cell.
#    Original: ... 2, 2, [bounce becomes 5], then later cells 5, 5, 5 ...
#    Variant : ... 2, 2, [bounce stays 2],  then later cells 2, 2, 2 ...
#    Therefore the changed stage is "bounce handling" in both solver and output-first generator.
# b) Complexity calibration:
#    If we allowed all arrows to have zero bounces, the sample could look identical to the original task and the
#    D4 change would be invisible. The generator therefore ensures at least one bounced ray per sample. To make
#    the invariant-color rule visually obvious, wall colors are sampled from colors not used by arrows, so the
#    human sees repeated cases where a differently colored wall reflects the ray without recoloring it.
# c) Hidden assumptions in the original main():
#    - `cur_col` changed on each bounce. That no longer holds; ray color must stay equal to the arrow color.
#    - Recoloring the current/bounce cell with `wcol` was safe before because bounce cells were meant to change.
#      In the variant this would overwrite a correct ray-color pixel, so those assignments must be removed.
#    - The loop-state memo originally included `cur_col`; with constant color that state dimension is irrelevant.
#      Keeping only position and direction avoids implying color can vary.
# d) Placement legality:
#    Generator still enforces in-bounds arrow boxes, diagonal ray paths, non-overlap with prior objects, and
#    valid 1-cell-off wall placement. Additionally it tracks planned wall cells during a single arrow trace so
#    later segments cannot accidentally step onto an earlier wall and create an unintended corner bounce.
# e) Visual observability:
#    Every generated sample contains at least one bounce, and bounced rays keep crossing beside walls whose color
#    differs from the ray color. Across 2-4 examples this should make the new rule legible: walls redirect rays
#    but do not donate color.
# f) Visual cleanliness:
#    Rays, arrows, and walls remain disjoint. Wall colors are reused from a small wall-only pool, which keeps
#    the grid readable instead of introducing many unrelated color changes along ray trajectories.
# g) Color-change inferability:
#    Because walls never share colors with arrows in generated data, a solver cannot explain the output via
#    accidental same-color matches. The repeated structure is: "different-colored wall adjacent to bounce, but
#    ray color does not change". That directly exposes the D4 rule.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from triangular arrows, reflecting off colored wall pixels.

    Variant rule:
    - Rays still reflect exactly as in the original task.
    - Rays always keep the arrow's initial color; wall colors never recolor the path.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
        if len(pts) != 3:
            return False
        rs = [r for r, _c in pts]
        cs = [c for _r, c in pts]
        return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)

    def arrow_dir_and_start(pts: list[tuple[int, int]]):
        rs = [r for r, _c in pts]
        cs = [c for _r, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        miss = [p for p in corners if p not in set(pts)]
        if len(miss) != 1:
            return None
        mr, mc = miss[0]
        dr = 1 if mr == r0 else -1
        dc = 1 if mc == c0 else -1

        pts_set = set(pts)
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

    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()

    for start_col, pts in arrows:
        info = arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
        ray_col = int(start_col)
        seen: set[tuple[tuple[int, int], int, int]] = set()

        while True:
            state = (cur, int(dr), int(dc))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in wall_cells:
                dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = ray_col

            if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                dr = -int(dr)
            elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                dr = -int(dr)

            if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                dc = -int(dc)
            elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                dc = -int(dc)

    return out


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input for the D4 variant."""
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(1, 7)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]
    wall_palette = palette[n_arrows:]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    total_bounces = 0

    for ai in range(n_arrows):
        color = arrow_colors[ai]
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        if ai == 0:
            n_bounces = _rng.choices([1, 2, 3, 4], weights=[3, 3, 2, 1])[0]
        else:
            n_bounces = _rng.choices([0, 1, 2, 3, 4], weights=[2, 3, 3, 2, 1])[0]

        placed = False
        for _ in range(60):
            r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
            c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)

            mr, mc = sr - dr, sc - dc
            arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
            if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
                continue

            cur_r, cur_c, cur_dr, cur_dc = sr, sc, dr, dc
            ray_cells: list[tuple[int, int, int]] = []
            walls: list[tuple[list[tuple[int, int]], int]] = []
            ray_set: set[tuple[int, int]] = set()
            wall_set: set[tuple[int, int]] = set()
            arrow_set = set(arrow_pts)
            ok = True

            for seg_idx in range(n_bounces + 1):
                max_steps = min(
                    (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                    (W - 1 - cur_c) if cur_dc == 1 else cur_c,
                )
                if seg_idx < n_bounces:
                    if max_steps < 4:
                        ok = False
                        break
                    seg_len = _rng.randint(3, max_steps - 1)
                else:
                    seg_len = max_steps

                for _ in range(seg_len):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                    if (
                        (nr, nc) in occupied
                        or (nr, nc) in ray_set
                        or (nr, nc) in arrow_set
                        or (nr, nc) in wall_set
                    ):
                        ok = False
                        break
                    ray_cells.append((nr, nc, color))
                    ray_set.add((nr, nc))
                    cur_r, cur_c = nr, nc
                if not ok:
                    break

                if seg_idx >= n_bounces:
                    continue

                options: list[tuple[list[tuple[int, int]], str]] = []
                wr, wc = cur_r + cur_dr, cur_c
                if 0 <= wr < H:
                    w3 = [(wr, wc + d) for d in (-1, 0, 1)]
                    if all(0 <= c < W for _, c in w3):
                        options.append((w3, "dr"))
                    options.append(([(wr, wc)], "dr"))

                wr2, wc2 = cur_r, cur_c + cur_dc
                if 0 <= wc2 < W:
                    w3v = [(wr2 + d, wc2) for d in (-1, 0, 1)]
                    if all(0 <= r < H for r, _ in w3v):
                        options.append((w3v, "dc"))
                    options.append(([(wr2, wc2)], "dc"))

                if not options:
                    ok = False
                    break

                _rng.shuffle(options)
                prefer_3 = _rng.random() < 0.6
                options.sort(key=lambda o: len(o[0]), reverse=prefer_3)
                chosen_wall = None
                for wpxs, flip in options:
                    if any(
                        (r, c) in occupied
                        or (r, c) in ray_set
                        or (r, c) in arrow_set
                        or (r, c) in wall_set
                        for r, c in wpxs
                    ):
                        continue
                    chosen_wall = (wpxs, flip)
                    break
                if chosen_wall is None:
                    ok = False
                    break

                wpxs, flip = chosen_wall
                wcol = _rng.choice(wall_palette)
                walls.append((wpxs, wcol))
                wall_set.update(wpxs)
                if flip == "dr":
                    cur_dr = -cur_dr
                else:
                    cur_dc = -cur_dc

            if not ok or len(ray_cells) < 3:
                continue

            new_cells = set(arrow_pts) | ray_set | wall_set
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
            total_bounces += len(walls)
            occupied |= new_cells
            placed = True
            break

        if not placed:
            return None

    if total_bounces == 0:
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for this D4 variant."""
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
