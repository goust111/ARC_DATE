# Variant of 142ca369
# Dimension: D9 — 输出网格形状 (grid_shape)
# Change: 输出与输入尺寸相同（same size）——在原网格上叠加射线轨迹... → 裁剪到射线覆盖区域

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg


# Design analysis for the D9 variant
# ----------------------------------
# a) Concrete behavior and trace
# - Input objects are unchanged: 3-pixel L arrows and detached wall pixels/segments.
# - Original output kept the whole input canvas and overpainted every diagonal ray.
# - This variant keeps the same ray physics, colors, bounces, and stopping rules,
#   but changes the final framing step:
#   1. First compute the same full-canvas overlay as the source task.
#   2. Track exactly which cells were visited by rays after leaving their arrows.
#   3. Return only the minimal axis-aligned rectangle covering all visited ray cells.
# - Small trace:
#   suppose an arrow occupies rows 0..1, cols 0..1 and fires down-right. The ray
#   visits (2,2),(3,3),..., reaches the lower/right side, and walls recolor some
#   of those visited cells. The old output returned the whole 19x19 board. The new
#   output returns the slice whose top-left is the first visited ray cell and whose
#   bottom-right is the furthest visited ray cell across all arrows. The arrow or
#   a wall may disappear entirely if they lie outside that ray bbox.
#
# b) Complexity calibration
# - The only rule change is output framing, but that framing is determined by the
#   dynamic ray coverage, not by a static non-background bbox. This keeps the task
#   non-trivial: one must understand where rays actually travel before cropping.
# - The generator forces crop visibility by making all rays avoid one chosen corner
#   region of the board. Therefore output height and width are both strictly smaller
#   than the input, and the crop is obvious in every sample.
#
# c) Hidden-assumption audit of the original solver
# - `out = g.copy()` and final `return out` assumed same-size output forever. That
#   assumption is invalid under D9 and must be replaced by `return out[bbox(ray)]`.
# - Arrow detection by monochrome 4-connected 3-cell components remains valid
#   because the input object vocabulary is unchanged.
# - Wall detection as "non-bg cells not belonging to arrows" remains valid because
#   the crop happens only after full simulation; it does not alter input parsing.
# - The bounce logic sometimes recolors the current cell using a wall color. Under
#   D9, those recolored cells must still count as part of the ray-covered region,
#   so the solver now tracks a `ray_mask` during advance steps instead of trying to
#   infer the crop from final non-background cells.
# - Using final non-background bbox would be wrong in this variant: detached arrows
#   and walls may sit outside the ray span, and they must not enlarge the crop.
#
# d) Placement legality
# - The generator still builds valid non-overlapping arrows, rays, and walls using
#   an output-first plan and the original reflection rules.
# - Additionally, every ray cell is constrained to stay away from one chosen corner
#   by at least a small margin. This guarantees a visible crop without changing the
#   underlying motion law.
# - Oracle validation remains strict: the generated input is accepted only when the
#   new `main()` exactly matches the planned cropped output.
#
# e) Visual observability
# - Humans can see that outputs are smaller and centered on the ray activity rather
#   than on the whole board. Because source arrows are often clipped away on the
#   avoided-corner side, the crop cue is strong and repeated across examples.
# - Reflection behavior still appears because some rays recolor at bounce points and
#   continue in a new direction before the final crop is taken.
#
# f) Visual cleanliness
# - The crop boundary is a plain rectangle derived from ray coverage only; there is
#   no extra annotation layer. Objects remain separated in the input, and the output
#   contains only the relevant activity region.
#
# g) Color inferability
# - No new color mapping is introduced. Ray color inheritance from walls stays
#   identical to the source task, so the only new inference burden is geometric:
#   determine the full ray span, then crop to that span.
#
# New solver assumptions and generator guarantees
# - Assumption: at least one ray cell exists. Guaranteed by requiring every arrow to
#   have a planned ray of length >= 3.
# - Assumption: the correct crop is the bbox of visited ray cells, not the bbox of
#   all non-background output cells. Guaranteed visible because the generator keeps
#   ray coverage away from one corner while allowing arrows/walls to occupy margins.
# - Assumption: crop height and width are both smaller than input dimensions.
#   Guaranteed by constraining all ray cells away from one horizontal side and one
#   vertical side of the canvas in every generated sample.


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


def _find_arrows_and_walls(g: np.ndarray, bg: int) -> tuple[list[tuple[int, list[tuple[int, int]]]], set[tuple[int, int]], set[tuple[int, int]]]:
    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)

    wall_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }
    return arrows, arrow_cells, wall_cells


def _trace_full_output_and_ray_mask(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    arrows, arrow_cells, wall_cells = _find_arrows_and_walls(g, bg)

    out = g.copy()
    ray_mask = np.zeros((H, W), dtype=bool)

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
                    ray_mask[int(cur[0]), int(cur[1])] = True
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            ray_mask[nr, nc] = True
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

    return out, ray_mask


def _crop_to_ray_bbox(full_out: np.ndarray, ray_mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(ray_mask)
    if coords.size == 0:
        return np.array(full_out, dtype=int, copy=True)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.array(full_out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int, copy=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-shaped arrows, then tight-crop to the ray coverage.

    The ray motion is identical to the source task:
    - start at the right-angle corner of each 3-pixel 2x2 L,
    - move diagonally away from the missing corner,
    - adopt wall colors on bounce,
    - stop on exit or when another arrow is hit.

    D9 change:
    - Do not return the full overlaid canvas.
    - Return only the minimal rectangle covering all ray-visited cells.
    """
    full_out, ray_mask = _trace_full_output_and_ray_mask(grid)
    return _crop_to_ray_bbox(full_out, ray_mask)


def _anchor_config(anchor: str) -> tuple[int, int]:
    mapping = {
        "tl": (1, 1),
        "tr": (1, -1),
        "bl": (-1, 1),
        "br": (-1, -1),
    }
    return mapping[anchor]


def _ray_cell_allowed(r: int, c: int, H: int, W: int, anchor: str, margin: int) -> bool:
    if "t" in anchor and r < margin:
        return False
    if "b" in anchor and r > H - 1 - margin:
        return False
    if "l" in anchor and c < margin:
        return False
    if "r" in anchor and c > W - 1 - margin:
        return False
    return True


def _near_avoided_corner(sr: int, sc: int, H: int, W: int, anchor: str) -> bool:
    pad = 4
    row_ok = sr <= pad if "t" in anchor else sr >= H - 1 - pad
    col_ok = sc <= pad if "l" in anchor else sc >= W - 1 - pad
    return bool(row_ok and col_ok)


def _ray_bbox_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not cells:
        return None
    rs = [r for r, _c in cells]
    cs = [c for _r, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _gen_attempt() -> np.ndarray | None:
    """
    Output-first generator for the D9 crop variant.

    Relative to the source task, the motion plan is unchanged, but we additionally
    constrain all ray cells to avoid one chosen corner so the final crop is always
    visibly smaller in both dimensions.
    """
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(1, 6)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]

    anchor = _rng.choice(["tl", "tr", "bl", "br"])
    dr0, dc0 = _anchor_config(anchor)
    margin = 2

    grid = np.zeros((H, W), dtype=int)
    full_out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    all_ray_cells: set[tuple[int, int]] = set()

    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        dr, dc = int(dr0), int(dc0)
        n_bounces = _rng.choices([0, 1, 2, 3, 4], weights=[2, 3, 3, 2, 1])[0]

        placed = False
        for _ in range(120):
            r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
            c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)
            if ai == 0 and not _near_avoided_corner(sr, sc, H, W, anchor):
                continue

            mr, mc = sr - dr, sc - dc
            arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
            if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
                continue

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
                    if not _ray_cell_allowed(nr, nc, H, W, anchor, margin):
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

                if seg_idx >= n_bounces or bounces_left <= 0:
                    continue
                bounces_left -= 1

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
                    if any((r, c) in occupied or (r, c) in ray_set or (r, c) in arrow_set for r, c in wpxs):
                        continue
                    chosen_wall = (wpxs, flip)
                    break
                if chosen_wall is None:
                    ok = False
                    break

                wpxs, flip = chosen_wall
                wcol = int(_rng.choice(palette))
                walls.append((wpxs, wcol))
                if ray_cells:
                    lr, lc, _ = ray_cells[-1]
                    ray_cells[-1] = (lr, lc, wcol)
                cur_col = wcol
                if flip == "dr":
                    cur_dr = -cur_dr
                else:
                    cur_dc = -cur_dc

            if not ok or len(ray_cells) < 3:
                continue

            bbox = _ray_bbox_from_cells(ray_set)
            if bbox is None:
                continue

            new_cells = set(arrow_pts) | ray_set
            for wpxs, _ in walls:
                new_cells.update(wpxs)
            expected_count = len(arrow_pts) + len(ray_cells) + sum(len(w) for w, _ in walls)
            if len(new_cells) != expected_count or new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = full_out[r, c] = color
            for r, c, col in ray_cells:
                full_out[r, c] = col
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[r, c] = full_out[r, c] = wcol

            occupied |= new_cells
            all_ray_cells |= ray_set
            placed = True
            break

        if not placed:
            return None

    bbox = _ray_bbox_from_cells(all_ray_cells)
    if bbox is None:
        return None
    r0, c0, r1, c1 = bbox

    cropped = np.array(full_out[r0 : r1 + 1, c0 : c1 + 1], dtype=int, copy=True)
    if cropped.shape[0] >= H or cropped.shape[1] >= W:
        return None

    solved = main(grid)
    if not np.array_equal(solved, cropped):
        return None

    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D9 crop variant of task 142ca369."""
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
