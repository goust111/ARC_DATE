# Variant of 142ca369
# Dimension: D3 — 轨迹涂色模式 (trajectory)
# Change: 连续轨迹（continuous_trail）——射线经过的每个背景格子都被涂色，形成连续的彩色线条... → 周期性图案轨迹

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg


# Design analysis for the D3 variant
# ----------------------------------
# a) Concrete behavior:
#    The original task already determines a unique diagonal ray geometry from each
#    3-cell L arrow and from the side-touch wall placements. This variant keeps the
#    geometry unchanged and changes only how traversed background cells are colored.
#    For one ray we keep two colors in play:
#      A = the arrow's own color (fixed for the whole ray)
#      B = the currently inherited wall color (initially equal to A; after a bounce
#          it becomes that wall's color)
#    Between launch/bounces, the ray paints traversed cells with the 4-step cycle
#    A, A, B, B, then repeats. The cycle phase resets after every bounce. The bounce
#    contact cell is still overwritten by the wall color B, matching the old task's
#    "inherit wall color on contact" rule. Example:
#      original: arrow color 2, wall color 5 -> outgoing cells after the bounce are
#      2, 2, 5, 5, 2, 2, 5, 5...
#      old solver instead painted 5, 5, 5, 5...
#    So the changed step is exactly the "paint traversed bg cell" action; reflection,
#    blocking, and wall color adoption stay the same.
#
# b) Complexity calibration:
#    A 2-color, length-4 period is enough to be visible but is still simple enough
#    for humans to infer from 2-4 examples. Shorter patterns such as AB can be too
#    easy to confuse with noise; more than two colors would add unnecessary freedom.
#    The generator therefore forces at least one bounce and a reasonably long
#    post-bounce segment, otherwise the new coloring would collapse back to the
#    original monochrome trail or be too short to notice.
#
# c) Implicit-assumption audit of the original solver:
#    - Arrow detection per monochrome connected component still holds, so generator
#      keeps arrows monochrome and avoids wall colors that equal any arrow color.
#    - `out[nr, nc] = cur_col` used to be safe because every traversed cell had one
#      uniform segment color. That is no longer true; now the color depends on the
#      segment-local phase, so the solver must compute the cycle explicitly.
#    - Bounce handling that overwrites the contact cell with wall color remains
#      correct and is intentionally preserved; otherwise the wall-inheritance clue
#      would disappear.
#    - The old loop-detection state ignored any paint phase. In this variant, the
#      future output can differ if the same geometric state is revisited with a
#      different phase, so the phase modulo 4 is part of the seen-state.
#
# d) Placement legality:
#    The generator still plans paths output-first. Rays, arrows, and walls never
#    overlap; walls are placed 1 cell off the diagonal exactly as in the original
#    side-touch construction. Every candidate is oracle-checked with `main(grid)`.
#
# e) Visual observability:
#    A pure first segment would still look like the old task because initially A=B.
#    To make the variant visible, generator samples 1-4 bounces but requires at
#    least one bounce and at least one post-bounce segment of length >= 4.
#
# f) Visual cleanliness:
#    Wall colors are chosen from colors unused by arrows, which avoids same-color
#    arrow/wall mergers and keeps the object classes visually separable.
#
# g) Color-rule inferability:
#    The color rule is structurally tied to visible objects: the repeated "AABB"
#    pattern uses the arrow color and the most recently touched wall color. The
#    generator keeps the palette small and reuses this same rule in every example,
#    so humans can align stripes with nearby arrow/wall colors rather than guessing
#    arbitrary color remappings.
#
# h) New solver assumptions checked against generator:
#    - Each true arrow is still an isolated monochrome 3-cell L in a 2x2 bbox.
#    - Non-arrow non-bg cells are walls only.
#    - Side-touch bounces dominate generated data, but corner reflection remains
#      implemented for robustness.
#    - Since wall colors avoid arrow colors, per-color component scanning cannot
#      accidentally absorb wall pixels into an arrow component.


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


def _periodic_color(start_col: int, cur_col: int, phase: int) -> int:
    return int(start_col if (phase % 4) < 2 else cur_col)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from monochrome L-arrows.

    Geometry is unchanged from 142ca369:
    - diagonal motion away from the missing corner
    - side-touch wall reflections flip one axis
    - corner hits flip both axes
    - wall color is inherited at each bounce

    Variant change:
    - traversed background cells are not filled with a solid segment color
    - instead each launch/bounce starts a fresh AABB cycle where:
      A = arrow color, B = current inherited color
    - the bounce contact cell is still recolored to the wall color B
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

    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
        cur_col = int(start_col)
        phase = 0
        seen = set()

        while True:
            state = (cur, int(dr), int(dc), int(cur_col), int(phase % 4))
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
                phase = 0
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = _periodic_color(int(start_col), int(cur_col), int(phase))
            phase = (int(phase) + 1) % 4

            bounced = False
            if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                wcol = int(g[nr + 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)
                bounced = True
            elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                wcol = int(g[nr - 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)
                bounced = True

            if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                wcol = int(g[nr, nc + 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)
                bounced = True
            elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                wcol = int(g[nr, nc - 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)
                bounced = True

            if bounced:
                phase = 0

    return out


def _gen_attempt() -> np.ndarray | None:
    """
    Output-first generator for the periodic-trajectory variant.

    Geometry is planned exactly as in the original family, but we now compute the
    target output with the segment-local AABB trail rule and reject candidates whose
    visible post-bounce trail is too short to expose the variant.
    """
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(1, 5)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]
    wall_palette = [c for c in palette if c not in set(arrow_colors)]
    if len(wall_palette) < 2:
        return None

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()

    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        n_bounces = _rng.choices([1, 2, 3, 4], weights=[2, 4, 3, 1])[0]

        placed = False
        for _ in range(120):
            dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
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
            phase = 0

            ray_cells: list[tuple[int, int, int]] = []
            walls: list[tuple[list[tuple[int, int]], int]] = []
            ray_set: set[tuple[int, int]] = set()
            arrow_set = set(arrow_pts)
            segment_lengths: list[int] = []
            wall_colors: list[int] = []
            ok = True

            for seg_idx in range(n_bounces + 1):
                max_steps = min(
                    (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                    (W - 1 - cur_c) if cur_dc == 1 else cur_c,
                )
                is_final = seg_idx == n_bounces
                if is_final:
                    min_len = 4
                    if max_steps < min_len:
                        ok = False
                        break
                    seg_len = _rng.randint(min_len, max_steps)
                else:
                    min_len = 2 if seg_idx == 0 else 4
                    hi = min(5 if seg_idx == 0 else 8, max_steps - 1)
                    if hi < min_len:
                        ok = False
                        break
                    seg_len = _rng.randint(min_len, hi)

                moved = 0
                for _step in range(seg_len):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        ok = False
                        break
                    if (nr, nc) in occupied or (nr, nc) in ray_set or (nr, nc) in arrow_set:
                        ok = False
                        break
                    ray_cells.append((int(nr), int(nc), _periodic_color(int(color), int(cur_col), int(phase))))
                    ray_set.add((int(nr), int(nc)))
                    cur_r, cur_c = int(nr), int(nc)
                    phase = (int(phase) + 1) % 4
                    moved += 1
                if not ok:
                    break
                segment_lengths.append(int(moved))

                if is_final:
                    continue

                options: list[tuple[list[tuple[int, int]], str]] = []
                wr, wc = cur_r + cur_dr, cur_c
                if 0 <= wr < H:
                    w3 = [(wr, wc + d) for d in (-1, 0, 1)]
                    if all(0 <= c < W for _rr, c in w3):
                        options.append((w3, "dr"))
                    options.append(([(wr, wc)], "dr"))
                wr2, wc2 = cur_r, cur_c + cur_dc
                if 0 <= wc2 < W:
                    w3v = [(wr2 + d, wc2) for d in (-1, 0, 1)]
                    if all(0 <= r < H for r, _cc in w3v):
                        options.append((w3v, "dc"))
                    options.append(([(wr2, wc2)], "dc"))
                if not options:
                    ok = False
                    break

                _rng.shuffle(options)
                options.sort(key=lambda item: len(item[0]), reverse=_rng.random() < 0.6)
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
                wall_choices = [c for c in wall_palette if c != int(color)]
                if not wall_choices:
                    ok = False
                    break
                wcol = int(_rng.choice(wall_choices))
                walls.append((wpxs, int(wcol)))
                wall_colors.append(int(wcol))

                if ray_cells:
                    lr, lc, _old = ray_cells[-1]
                    ray_cells[-1] = (int(lr), int(lc), int(wcol))
                cur_col = int(wcol)
                phase = 0
                if flip == "dr":
                    cur_dr = -int(cur_dr)
                else:
                    cur_dc = -int(cur_dc)

            if not ok or len(ray_cells) < 7:
                continue
            if len(wall_colors) != n_bounces:
                continue
            if any(w == int(color) for w in wall_colors):
                continue
            if max(segment_lengths[1:], default=0) < 4:
                continue
            if sum(segment_lengths[1:]) < 6:
                continue

            new_cells = set(arrow_pts) | ray_set
            for wpxs, _wcol in walls:
                new_cells.update(wpxs)
            expected_count = len(arrow_pts) + len(ray_cells) + sum(len(w) for w, _wcol in walls)
            if len(new_cells) != expected_count or new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = out[r, c] = int(color)
            for r, c, col in ray_cells:
                out[r, c] = int(col)
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[r, c] = out[r, c] = int(wcol)
            occupied |= new_cells
            placed = True
            break

        if not placed:
            return None

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
