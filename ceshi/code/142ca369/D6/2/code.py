# Variant of 142ca369
# Dimension: D6 — 射线终止条件 (dst)
# Change: 射线离开网格边界时终止（out of grid）... → 遇到另一条射线轨迹终止
#
# Design analysis for this variant
# a) Concrete behavior:
#    - The input still contains monochrome 3-pixel L-shaped arrows plus mono-color wall
#      pixels/segments. Rays still move diagonally, bounce from side-touch walls, and inherit
#      wall colors exactly as in the original task.
#    - The new visible behavior is sequential firing. Arrows fire in row-major order of their
#      right-angle vertex (the ray start cell). After one ray is drawn, its painted diagonal
#      cells become "occupied track" for later rays. A later ray does not paint into that cell:
#      it stops immediately before entering the earlier track.
#    - Small example: if arrow A paints cells (5,5),(6,6),(7,7), and later arrow B is heading
#      toward (7,7), then original logic would keep extending B beyond (7,7) until boundary.
#      The variant instead paints B only up to (6,8) / the cell before collision and leaves
#      the earlier painted cell unchanged.
# b) Complexity calibration:
#    - The only rule change is the new stop condition plus the need for a deterministic firing
#      order. Reflection and color inheritance stay unchanged.
#    - To keep the change visually obvious, the generator always creates at least one genuine
#      track-collision and usually several; otherwise the task would collapse back to the
#      original out-of-grid termination.
# c) Hidden assumptions reviewed in the original main():
#    - Iterating arrows in detection order was harmless before because rays never interacted.
#      It is no longer harmless; we now sort arrows by their start cell so the interaction
#      order is deterministic and inferable.
#    - Writing `out[nr,nc]` on every step was harmless before because no later ray reused the
#      same cell. In the variant, later rays must stop before earlier tracks, so the solver now
#      checks the next diagonal cell against `prior_ray_cells` before advancing.
#    - Arrow cells still block rays exactly as before. Wall color inheritance still applies
#      only on bounces, not on ray-ray collision.
# d) Placement legality:
#    - Generator keeps arrows, walls, and already painted ray cells disjoint in the input.
#    - The only permitted "contact" with an earlier ray is the would-be next diagonal cell of
#      a later ray, which is not painted by the later ray.
#    - Candidate later arrows are simulated against existing walls and arrows so accidental
#      extra bounces are rejected.
# e) Visual observability:
#    - The first ray is long enough to serve as an obvious target. Later rays terminate early
#      at pre-existing painted diagonals, so the outputs show truncated rays whose endpoints
#      line up with older tracks rather than the border.
#    - The order cue is spatial: arrows fire by row-major order of their start vertex, and the
#      generator constructs them in that same order.
# f) Visual cleanliness:
#    - Only one ray may occupy a painted cell. Later rays stop one cell before collision, so
#      the output does not contain overdrawn crossings.
#    - Wall density is kept modest; only the foundation ray is guaranteed to use walls, which
#      preserves the original bounce vocabulary without creating a cluttered mesh.
# g) Color reasoning:
#    - Color change behavior is unchanged from the original task: rays begin in the arrow color
#      and adopt the contacted wall color on bounce. Ray-ray collision does not recolor.

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg


Coord = tuple[int, int]


def _is_arrow_pts(pts: list[Coord]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[Coord]) -> tuple[tuple[int, int], Coord] | None:
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


def _arrow_points_from_start(start: Coord, direction: tuple[int, int]) -> list[Coord]:
    sr, sc = start
    dr, dc = direction
    mr, mc = sr - dr, sc - dc
    return [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]


def _detect_arrows(g: np.ndarray, bg: int) -> tuple[list[dict], set[Coord]]:
    arrows: list[dict] = []
    arrow_cells: set[Coord] = set()
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if not _is_arrow_pts(pts):
                continue
            info = _arrow_dir_and_start(pts)
            if info is None:
                continue
            direction, start = info
            arrows.append(
                {
                    "color": int(col),
                    "pts": pts,
                    "dir": direction,
                    "start": start,
                }
            )
            arrow_cells |= set(pts)
    arrows.sort(key=lambda a: (a["start"][0], a["start"][1], a["color"]))
    return arrows, arrow_cells


def _trace_arrow(
    g: np.ndarray,
    bg: int,
    arrow: dict,
    wall_cells: set[Coord],
    arrow_cells: set[Coord],
    prior_ray_cells: set[Coord],
) -> tuple[list[tuple[int, int, int]], set[Coord], str]:
    H, W = g.shape
    dr, dc = arrow["dir"]
    cur = arrow["start"]
    cur_col = int(arrow["color"])
    seen = set()

    ray_cells: list[list[int]] = []
    ray_index: dict[Coord, int] = {}

    def recolor_current(new_col: int) -> None:
        idx = ray_index.get(cur)
        if idx is not None:
            ray_cells[idx][2] = int(new_col)

    while True:
        state = (cur, int(dr), int(dc), int(cur_col))
        if state in seen:
            return [(r, c, col) for r, c, col in ray_cells], set(ray_index), "loop"
        seen.add(state)

        nr = int(cur[0] + dr)
        nc = int(cur[1] + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            return [(r, c, col) for r, c, col in ray_cells], set(ray_index), "out"

        if (nr, nc) in prior_ray_cells:
            return [(r, c, col) for r, c, col in ray_cells], set(ray_index), "prior"

        if (nr, nc) in wall_cells:
            wcol = int(g[nr, nc])
            if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                recolor_current(wcol)
            cur_col = int(wcol)
            dr, dc = -int(dr), -int(dc)
            continue

        if (nr, nc) in arrow_cells:
            return [(r, c, col) for r, c, col in ray_cells], set(ray_index), "arrow"

        cur = (int(nr), int(nc))
        if cur not in ray_index:
            ray_index[cur] = len(ray_cells)
            ray_cells.append([nr, nc, int(cur_col)])

        if int(dr) == 1 and (nr + 1, nc) in wall_cells:
            wcol = int(g[nr + 1, nc])
            if int(g[nr, nc]) == int(bg):
                recolor_current(wcol)
            cur_col = int(wcol)
            dr = -int(dr)
        elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
            wcol = int(g[nr - 1, nc])
            if int(g[nr, nc]) == int(bg):
                recolor_current(wcol)
            cur_col = int(wcol)
            dr = -int(dr)

        if int(dc) == 1 and (nr, nc + 1) in wall_cells:
            wcol = int(g[nr, nc + 1])
            if int(g[nr, nc]) == int(bg):
                recolor_current(wcol)
            cur_col = int(wcol)
            dc = -int(dc)
        elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
            wcol = int(g[nr, nc - 1])
            if int(g[nr, nc]) == int(bg):
                recolor_current(wcol)
            cur_col = int(wcol)
            dc = -int(dc)


def _simulate(grid: np.ndarray, *, stop_on_prior: bool) -> tuple[np.ndarray, list[dict]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    arrows, arrow_cells = _detect_arrows(g, bg)
    wall_cells: set[Coord] = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()
    prior_ray_cells: set[Coord] = set()
    trace_meta: list[dict] = []

    for arrow in arrows:
        active_prior = prior_ray_cells if stop_on_prior else set()
        ray_cells, ray_set, reason = _trace_arrow(g, bg, arrow, wall_cells, arrow_cells, active_prior)
        for r, c, col in ray_cells:
            if int(out[r, c]) == int(bg):
                out[r, c] = int(col)
        prior_ray_cells |= ray_set
        trace_meta.append(
            {
                "start": arrow["start"],
                "reason": reason,
                "ray_len": len(ray_cells),
                "ray_set": ray_set,
            }
        )

    return out, trace_meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-shaped arrows with wall reflections and ordered track collision.

    Variant rule:
    - Arrows fire in row-major order of their right-angle vertex.
    - Earlier rays are painted first.
    - A later ray stops immediately before entering any already-painted ray cell.

    Everything else matches the original solver:
    - diagonal step per tick
    - side-touch or corner reflections from wall pixels
    - wall color inheritance at each bounce
    - arrow cells still block rays
    """
    out, _ = _simulate(np.array(grid, dtype=int), stop_on_prior=True)
    return out


def _row_major_gt(a: Coord, b: Coord) -> bool:
    return (a[0], a[1]) > (b[0], b[1])


def _plan_foundation_arrow(
    H: int,
    W: int,
    color: int,
    occupied: set[Coord],
) -> tuple[Coord, tuple[int, int], list[Coord], list[tuple[int, int, int]], list[tuple[list[Coord], int]]] | None:
    for _ in range(140):
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = _rng.choices([0, 1, 2, 3], weights=[1, 3, 3, 2])[0]

        r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
        c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        # Bias the first start earlier in row-major order so several later arrows can follow it.
        r_hi = min(r_hi, H - 4)
        c_hi = min(c_hi, W - 4)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)
        start = (int(sr), int(sc))
        arrow_pts = _arrow_points_from_start(start, (dr, dc))
        if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
            continue

        cur_r, cur_c, cur_dr, cur_dc, cur_col = sr, sc, dr, dc, color
        ray_cells: list[tuple[int, int, int]] = []
        walls: list[tuple[list[Coord], int]] = []
        ray_set: set[Coord] = set()
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

            options: list[tuple[list[Coord], str]] = []
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
            wcol = _rng.randint(1, 9)
            walls.append((wpxs, wcol))
            if ray_cells:
                lr, lc, _ = ray_cells[-1]
                ray_cells[-1] = (lr, lc, wcol)
            cur_col = wcol
            if flip == "dr":
                cur_dr = -cur_dr
            else:
                cur_dc = -cur_dc

        if not ok or len(ray_cells) < 7:
            continue

        new_cells = set(arrow_pts) | ray_set
        for wpxs, _ in walls:
            new_cells.update(wpxs)
        expected_count = len(arrow_pts) + len(ray_cells) + sum(len(w) for w, _ in walls)
        if len(new_cells) != expected_count or new_cells & occupied:
            continue

        return start, (dr, dc), arrow_pts, ray_cells, walls

    return None


def _try_collision_arrow(
    g: np.ndarray,
    color: int,
    occupied: set[Coord],
    arrow_cells: set[Coord],
    wall_cells: set[Coord],
    prior_ray_cells: set[Coord],
    last_start: Coord,
) -> tuple[Coord, list[Coord], list[tuple[int, int, int]], str] | None:
    H, W = g.shape
    targets = list(prior_ray_cells)
    _rng.shuffle(targets)
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for target in targets:
        _rng.shuffle(directions)
        for dr, dc in directions:
            max_dist = min(H, W)
            dists = list(range(3, max_dist))
            _rng.shuffle(dists)
            for dist in dists:
                sr = int(target[0] - dr * dist)
                sc = int(target[1] - dc * dist)
                start = (sr, sc)
                if not (0 <= sr < H and 0 <= sc < W):
                    continue
                if not _row_major_gt(start, last_start):
                    continue

                arrow_pts = _arrow_points_from_start(start, (dr, dc))
                arrow_set = set(arrow_pts)
                if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
                    continue

                ray_prefix = [(sr + dr * k, sc + dc * k) for k in range(1, dist)]
                if len(ray_prefix) < 2:
                    continue
                if any(cell in occupied or cell in arrow_set for cell in ray_prefix):
                    continue

                test_arrow = {"color": int(color), "pts": arrow_pts, "dir": (dr, dc), "start": start}
                test_arrow_cells = set(arrow_cells) | arrow_set
                ray_cells, _, reason = _trace_arrow(g, 0, test_arrow, wall_cells, test_arrow_cells, prior_ray_cells)
                if reason != "prior" or len(ray_cells) < 2:
                    continue

                return start, arrow_pts, ray_cells, reason

    return None


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    target_arrows = _rng.choices([2, 3, 4, 5], weights=[1, 3, 3, 2])[0]

    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:target_arrows]

    grid = np.zeros((H, W), dtype=int)
    occupied: set[Coord] = set()
    arrow_cells: set[Coord] = set()
    wall_cells: set[Coord] = set()
    prior_ray_cells: set[Coord] = set()

    foundation = _plan_foundation_arrow(H, W, arrow_colors[0], occupied)
    if foundation is None:
        return None
    last_start, _, arrow_pts, ray_cells, walls = foundation

    for r, c in arrow_pts:
        grid[r, c] = arrow_colors[0]
    arrow_cells |= set(arrow_pts)
    occupied |= set(arrow_pts)

    for r, c, _ in ray_cells:
        occupied.add((r, c))
        prior_ray_cells.add((r, c))

    for wpxs, wcol in walls:
        for r, c in wpxs:
            grid[r, c] = wcol
            occupied.add((r, c))
            wall_cells.add((r, c))

    if len(prior_ray_cells) < 7:
        return None

    placed = 1
    collision_count = 0
    for color in arrow_colors[1:]:
        candidate = _try_collision_arrow(
            grid,
            color,
            occupied,
            arrow_cells,
            wall_cells,
            prior_ray_cells,
            last_start,
        )
        if candidate is None:
            return None
        start, pts, ray2, reason = candidate
        for r, c in pts:
            grid[r, c] = color
        arrow_cells |= set(pts)
        occupied |= set(pts)
        for r, c, _ in ray2:
            occupied.add((r, c))
            prior_ray_cells.add((r, c))
        last_start = start
        placed += 1
        if reason == "prior":
            collision_count += 1

    if placed != target_arrows or collision_count < 1:
        return None

    out, meta = _simulate(grid, stop_on_prior=True)
    reasons = [m["reason"] for m in meta]
    if len(meta) != target_arrows:
        return None
    if reasons.count("prior") < target_arrows - 1:
        return None
    if sum(m["ray_len"] for m in meta) < 12:
        return None

    # Reject degenerate inputs that behave the same as the original out-of-grid rule.
    old_out, _ = _simulate(grid, stop_on_prior=False)
    if np.array_equal(out, old_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
