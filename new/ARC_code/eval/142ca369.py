from __future__ import annotations
import numpy as np
from arc_utils import colors_present, components_mask, get_bg

def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from triangular arrows, reflecting off colored wall pixels.

    Input structure:
    - bg grid with monochrome 3-pixel L-shaped "arrows" (2x2 bbox missing one corner) and scattered colored wall pixels.

    Rule:
    - Each arrow fires a ray diagonally away from its missing corner, painting bg cells in its color.
    - A head-on wall hit reverses both axes; a side-touch wall hit flips the touched axis only.
    - On any bounce the ray adopts the wall's color and recolors the contact cell.
    - Ray stops when it exits the grid or reaches another arrow.

    Key insight: Two distinct reflection types (corner vs side-touch) plus color inheritance at each bounce drive the path geometry.
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
        # direction away from missing corner
        dr = 1 if mr == r0 else -1
        dc = 1 if mc == c0 else -1

        pts_set = set(pts)
        deg = {p: 0 for p in pts}
        for r, c in pts:
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (rr, cc) in pts_set:
                    deg[(r, c)] += 1
        start = max(deg, key=lambda p: deg[p])  # right-angle corner
        return (int(dr), int(dc)), (int(start[0]), int(start[1]))

    # Detect arrows per-color (cannot rely on global non-bg connectivity).
    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)

    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c)) for r, c in np.argwhere(g != int(bg)) if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()

    for start_col, pts in arrows:
        info = arrow_dir_and_start(pts)
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


import random as _rng


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input for task 142ca369.

    Output-first: for each arrow, plan the full zigzag ray path (with 0-4 side-touch
    bounces), place walls at bounce points, then derive the input (arrows + walls).
    """
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(1, 7)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()

    for ai in range(n_arrows):
        color = arrow_colors[ai]
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = _rng.choices([0, 1, 2, 3, 4], weights=[2, 3, 3, 2, 1])[0]

        placed = False
        for _ in range(60):
            # Start position with room for the 2x2 arrow bbox
            r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
            c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)

            # Arrow = 3 corners of 2x2 bbox, missing corner opposite to fire direction
            mr, mc = sr - dr, sc - dc
            arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
            if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
                continue

            # Trace ray path with planned bounces
            cur_r, cur_c, cur_dr, cur_dc, cur_col = sr, sc, dr, dc, color
            ray_cells: list[tuple[int, int, int]] = []
            walls: list[tuple[list[tuple[int, int]], int]] = []
            ray_set: set[tuple[int, int]] = set()
            arrow_set = set(arrow_pts)
            ok = True
            bounces_left = n_bounces

            for seg_idx in range(n_bounces + 1):
                max_steps = min(
                    (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                    (W - 1 - cur_c) if cur_dc == 1 else cur_c,
                )
                if seg_idx < n_bounces and bounces_left > 0:
                    if max_steps < 4:
                        bounces_left = 0
                        seg_len = max_steps
                    else:
                        seg_len = _rng.randint(3, max(3, max_steps - 1))
                else:
                    seg_len = max_steps

                for _ in range(seg_len):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                    if (nr, nc) in occupied or (nr, nc) in ray_set or (nr, nc) in arrow_set:
                        ok = False; break
                    ray_cells.append((nr, nc, cur_col))
                    ray_set.add((nr, nc))
                    cur_r, cur_c = nr, nc
                if not ok:
                    break

                if seg_idx >= n_bounces or bounces_left <= 0:
                    continue
                bounces_left -= 1

                # Collect feasible wall placements for this bounce
                options: list[tuple[list[tuple[int, int]], str]] = []
                # Horizontal wall (flips dr): center pixel at (cur_r+cur_dr, cur_c)
                wr, wc = cur_r + cur_dr, cur_c
                if 0 <= wr < H:
                    w3 = [(wr, wc + d) for d in (-1, 0, 1)]
                    if all(0 <= c < W for _, c in w3):
                        options.append((w3, 'dr'))
                    options.append(([(wr, wc)], 'dr'))
                # Vertical wall (flips dc): center pixel at (cur_r, cur_c+cur_dc)
                wr2, wc2 = cur_r, cur_c + cur_dc
                if 0 <= wc2 < W:
                    w3v = [(wr2 + d, wc2) for d in (-1, 0, 1)]
                    if all(0 <= r < H for r, _ in w3v):
                        options.append((w3v, 'dc'))
                    options.append(([(wr2, wc2)], 'dc'))
                if not options:
                    ok = False; break

                # Mix of 1-pixel and 3-pixel walls; shuffle with size preference
                _rng.shuffle(options)
                prefer_3 = _rng.random() < 0.6
                options.sort(key=lambda o: len(o[0]), reverse=prefer_3)
                chosen_wall = None
                for wpxs, flip in options:
                    if any((r, c) in occupied or (r, c) in ray_set or (r, c) in arrow_set
                           for r, c in wpxs):
                        continue
                    chosen_wall = (wpxs, flip)
                    break
                if chosen_wall is None:
                    ok = False; break

                wpxs, flip = chosen_wall
                wcol = _rng.choice(palette)
                walls.append((wpxs, wcol))

                # Bounce cell adopts wall color in output
                if ray_cells:
                    lr, lc, _ = ray_cells[-1]
                    ray_cells[-1] = (lr, lc, wcol)
                cur_col = wcol
                if flip == 'dr':
                    cur_dr = -cur_dr
                else:
                    cur_dc = -cur_dc

            if not ok or len(ray_cells) < 3:
                continue

            # Check no internal or external collisions
            new_cells = set(arrow_pts) | ray_set
            for wpxs, _ in walls:
                new_cells.update(wpxs)
            expected_count = len(arrow_pts) + len(ray_cells) + sum(len(w) for w, _ in walls)
            if len(new_cells) != expected_count or new_cells & occupied:
                continue

            # Commit
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

    # Oracle check: solver must reproduce planned output
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for task 142ca369."""
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")


