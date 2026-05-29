# Combo variant of 142ca369 (index 17)
# combo_spec: D9/0+D11/0+D12/1
# Name: 单箭头+高反射+裁剪输出

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - Source task: one or more 3-cell L arrows each emit a diagonal ray; detached wall
#   pixels/segments cause side-touch reflections that flip one axis and transfer the
#   wall color onto the bounce cell and subsequent ray.
# - D11/0 collapses the scene to exactly one arrow. This removes multi-source
#   interference, so the key inference burden shifts to following a single long path.
# - D12/1 then raises the reflection count to 5-10. In a concrete trace, an arrow
#   near the upper-left fires down-right, reaches a bounce cell, touches a wall
#   directly below, flips vertical direction, later touches a wall to the right,
#   flips horizontal direction, and continues through several more color changes.
# - D9/0 changes only the framing step: after tracing the full-canvas overlay, the
#   output keeps only the minimal rectangle covering visited ray cells. The arrow and
#   some walls can be partially or fully cropped away if they sit outside that ray
#   bbox. The crop therefore depends on dynamic trajectory, not on static objects.
#
# b) Complexity calibration
# - Using a single arrow could be too easy if the path were short, so the generator
#   forces 5-10 reflections and uses a larger 27-33 canvas to create a visibly long
#   zigzag. The scene still has repeated local evidence: every bounce is marked by an
#   adjacent wall and a color change, so humans can infer the same rule across many
#   repeated instances inside one sample.
# - The generator keeps wall shapes to single pixels. This narrows one degree of
#   freedom on purpose so the added difficulty comes from path length and cropping,
#   not from noisy wall geometry.
#
# c) Hidden-assumption audit of the original solver
# - `return out` assumed output size always equals input size; D9 invalidates that,
#   so the solver must track `ray_mask` and crop to the bbox of visited ray cells.
# - The original solver allowed many arrows. That logic still works, but the combo
#   generator now guarantees exactly one arrow. Solver-side arrow detection by
#   monochrome 4-connected 3-cell components remains valid because the input object
#   vocabulary is unchanged.
# - Wall detection as "all non-bg cells not in arrows" still holds, but the generator
#   must now prevent accidental wall components from mimicking arrow-shaped 3-cell
#   components. It does so by using only single-pixel walls and forbidding same-color
#   orthogonal wall adjacency, so every wall color component stays size 1.
# - The bounce code recolors the current ray cell at reflection time. Under cropping,
#   that cell must still count as visited even if it now shares the wall color, so the
#   solver records ray coverage explicitly instead of inferring it from non-bg cells.
# - The original generator tolerated 0-4 bounces and small boards. Those assumptions
#   fail under D12/1, so the new generator uses a larger board and a backtracking path
#   planner that explicitly reserves room for many reflections.
#
# d) Placement legality
# - The generator is output-first in geometry: it first plans one non-self-overlapping
#   ray path, then places one wall per intended bounce, then derives the input grid.
# - Every ray cell must stay inside bounds, avoid the arrow cells, avoid all previous
#   walls, and avoid reusing a ray cell. Side-touch with an old wall is rejected
#   unless that wall is the newly created reflector for the current segment end.
# - All ray cells are constrained to rows >= margin and cols >= margin. This keeps the
#   visited region away from the top and left borders so the crop is always smaller
#   than the input in both dimensions, while the last segment is still free to exit at
#   the bottom or right edge.
#
# e) Visual observability
# - In examples, the input shows one small arrow near the upper-left plus many detached
#   wall pixels spread through a larger board. The output becomes a tighter rectangle
#   centered on a long multicolor zigzag. That makes both changes legible by eye:
#   there is clearly only one source, and the output framing follows the ray rather
#   than the whole board.
# - Repeated recolor-at-bounce events make the high-reflection rule inferable from
#   2-4 examples without any hidden ordering or numeric-color trick.
#
# f) Visual cleanliness
# - Single-pixel walls reduce clutter. Because there is only one ray, there is no
#   overlap between multiple trajectories, and the crop removes unused empty margins.
# - The generator rejects self-intersections and unintended wall contacts, so the ray
#   remains a readable zigzag instead of degenerating into a tangled cluster.
#
# g) Color inferability
# - No arbitrary color mapping is introduced. The only color rule is inherited from
#   the source task: the ray adopts the wall color at each bounce.
# - To preserve arrow detectability, wall colors are assigned greedily so orthogonally
#   adjacent wall pixels never share a color. This is a generator-side legality rule,
#   not a new puzzle rule, and it is invisible to solvers because each wall still acts
#   purely as a local reflector.
#
# New solver assumptions and generator guarantees
# - Assumption: the correct crop is the bbox of visited ray cells only. Guaranteed by
#   tracking `ray_mask` in the solver and by generating at least one ray cell.
# - Assumption: there is exactly one arrow in combo samples. Guaranteed by generating
#   a single monochrome L-shape and preventing any wall configuration from creating an
#   accidental second arrow component.
# - Assumption: samples exhibit high reflection rather than degenerate short paths.
#   Guaranteed by planning exactly 5-10 walls, one per bounce, before the exit segment.
# - Assumption: crop size is strictly smaller than input size. Guaranteed because every
#   ray cell stays at least `margin` away from the top and left borders.


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


def _find_arrows_and_walls(
    g: np.ndarray, bg: int
) -> tuple[list[tuple[int, list[tuple[int, int]]]], set[tuple[int, int]], set[tuple[int, int]]]:
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
    Trace the arrow ray with original reflection physics, then crop to ray coverage.

    Combo effect:
    - there is only one arrow in generated samples,
    - that single ray performs many reflections,
    - the returned output is the minimal bbox covering visited ray cells.
    """
    full_out, ray_mask = _trace_full_output_and_ray_mask(grid)
    return _crop_to_ray_bbox(full_out, ray_mask)


_MIN_SIDE = 27
_MAX_SIDE = 33
_MARGIN = 2
_MIN_BOUNCES = 5
_MAX_BOUNCES = 10
_MIN_SEG_LEN = 2
_MAX_SEG_LEN = 7
_MIN_FINAL_LEN = 4


def _allowed_ray_cell(r: int, c: int, H: int, W: int, margin: int) -> bool:
    return 0 <= r < H and 0 <= c < W and r >= margin and c >= margin


def _segment_prefix(
    cur_r: int,
    cur_c: int,
    dr: int,
    dc: int,
    H: int,
    W: int,
    margin: int,
    blocked: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    rr, cc = int(cur_r), int(cur_c)
    while True:
        nr, nc = rr + dr, cc + dc
        if not _allowed_ray_cell(nr, nc, H, W, margin):
            break
        if (nr, nc) in blocked or (nr, nc) in wall_cells:
            break
        if (nr + dr, nc) in wall_cells or (nr, nc + dc) in wall_cells:
            break
        out.append((int(nr), int(nc)))
        rr, cc = int(nr), int(nc)
    return out


def _can_exit_cleanly(
    cur_r: int,
    cur_c: int,
    dr: int,
    dc: int,
    H: int,
    W: int,
    margin: int,
    blocked: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    path: list[tuple[int, int]] = []
    rr, cc = int(cur_r), int(cur_c)
    while True:
        nr, nc = rr + dr, cc + dc
        if not (0 <= nr < H and 0 <= nc < W):
            break
        if not _allowed_ray_cell(nr, nc, H, W, margin):
            return None
        if (nr, nc) in blocked or (nr, nc) in wall_cells:
            return None
        if (nr + dr, nc) in wall_cells or (nr, nc + dc) in wall_cells:
            return None
        path.append((int(nr), int(nc)))
        rr, cc = int(nr), int(nc)
    if len(path) < _MIN_FINAL_LEN:
        return None
    return path


def _plan_geometry(
    H: int,
    W: int,
    start: tuple[int, int],
    arrow_pts: list[tuple[int, int]],
    n_bounces: int,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    arrow_set = set(arrow_pts)

    def rec(
        cur_r: int,
        cur_c: int,
        dr: int,
        dc: int,
        remaining: int,
        ray_cells: set[tuple[int, int]],
        wall_cells: set[tuple[int, int]],
        segments: list[list[tuple[int, int]]],
        walls_in_order: list[tuple[int, int]],
        last_flip: str | None,
    ) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
        blocked = arrow_set | ray_cells

        if remaining == 0:
            final_segment = _can_exit_cleanly(cur_r, cur_c, dr, dc, H, W, _MARGIN, blocked, wall_cells)
            if final_segment is None:
                return None
            return segments + [final_segment], walls_in_order

        prefix = _segment_prefix(cur_r, cur_c, dr, dc, H, W, _MARGIN, blocked, wall_cells)
        if len(prefix) < _MIN_SEG_LEN:
            return None

        max_len = min(len(prefix), _MAX_SEG_LEN)
        lens = list(range(_MIN_SEG_LEN, max_len + 1))
        _rng.shuffle(lens)
        lens.sort(key=lambda n: abs(n - min(_MAX_SEG_LEN, 4)))

        for seg_len in lens:
            segment = prefix[:seg_len]
            end_r, end_c = segment[-1]

            flip_opts = ["dr", "dc"]
            _rng.shuffle(flip_opts)
            if last_flip == "dr":
                flip_opts.sort(key=lambda f: 0 if f == "dc" else 1)
            elif last_flip == "dc":
                flip_opts.sort(key=lambda f: 0 if f == "dr" else 1)

            for flip in flip_opts:
                wr = end_r + dr if flip == "dr" else end_r
                wc = end_c if flip == "dr" else end_c + dc
                wall = (int(wr), int(wc))
                other_touch = (int(end_r), int(end_c + dc)) if flip == "dr" else (int(end_r + dr), int(end_c))

                if not (0 <= wr < H and 0 <= wc < W):
                    continue
                if wall in arrow_set or wall in ray_cells or wall in wall_cells:
                    continue
                if other_touch in wall_cells:
                    continue

                new_dr = -dr if flip == "dr" else dr
                new_dc = dc if flip == "dr" else -dc

                future = rec(
                    end_r,
                    end_c,
                    new_dr,
                    new_dc,
                    remaining - 1,
                    ray_cells | set(segment),
                    wall_cells | {wall},
                    segments + [segment],
                    walls_in_order + [wall],
                    flip,
                )
                if future is not None:
                    return future

        return None

    return rec(
        int(start[0]),
        int(start[1]),
        1,
        1,
        int(n_bounces),
        set(),
        set(),
        [],
        [],
        None,
    )


def _assign_wall_colors(walls: list[tuple[int, int]], arrow_color: int) -> list[int] | None:
    palette = [c for c in range(1, 10) if c != int(arrow_color)]
    neighbors: list[set[int]] = [set() for _ in walls]
    index = {cell: i for i, cell in enumerate(walls)}
    for i, (r, c) in enumerate(walls):
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            j = index.get(nb)
            if j is not None:
                neighbors[i].add(j)

    order = list(range(len(walls)))
    order.sort(key=lambda i: len(neighbors[i]), reverse=True)
    colors = [-1] * len(walls)
    for i in order:
        options = palette[:]
        _rng.shuffle(options)
        used = {colors[j] for j in neighbors[i] if colors[j] != -1}
        choice = next((col for col in options if col not in used), None)
        if choice is None:
            return None
        colors[i] = int(choice)
    return [int(c) for c in colors]


def _ray_bbox(ray_path: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _c in ray_path]
    cs = [c for _r, c in ray_path]
    return min(rs), min(cs), max(rs), max(cs)


def _gen_attempt() -> np.ndarray | None:
    H = _rng.randint(_MIN_SIDE, _MAX_SIDE)
    W = _rng.randint(_MIN_SIDE, _MAX_SIDE)
    n_bounces = _rng.randint(_MIN_BOUNCES, _MAX_BOUNCES)

    arrow_color = _rng.randint(1, 9)
    sr = _rng.randint(_MARGIN + 2, _MARGIN + 5)
    sc = _rng.randint(_MARGIN + 2, _MARGIN + 5)

    mr, mc = sr - 1, sc - 1
    arrow_pts = [(mr, sc), (sr, mc), (sr, sc)]
    if any(not (0 <= r < H and 0 <= c < W) for r, c in arrow_pts):
        return None

    plan = _plan_geometry(H, W, (sr, sc), arrow_pts, n_bounces)
    if plan is None:
        return None
    segments, walls = plan
    if len(segments) != n_bounces + 1 or len(walls) != n_bounces:
        return None

    wall_colors = _assign_wall_colors(walls, arrow_color)
    if wall_colors is None:
        return None

    grid = np.zeros((H, W), dtype=int)
    full_out = np.zeros((H, W), dtype=int)
    ray_path: list[tuple[int, int]] = [cell for seg in segments for cell in seg]
    ray_set = set(ray_path)
    if len(ray_set) != len(ray_path):
        return None

    for r, c in arrow_pts:
        grid[r, c] = arrow_color
        full_out[r, c] = arrow_color

    cur_color = int(arrow_color)
    for seg_idx, segment in enumerate(segments):
        for r, c in segment:
            full_out[r, c] = cur_color
        if seg_idx < len(walls):
            wall_r, wall_c = walls[seg_idx]
            wall_color = int(wall_colors[seg_idx])
            grid[wall_r, wall_c] = wall_color
            full_out[wall_r, wall_c] = wall_color
            bounce_r, bounce_c = segment[-1]
            full_out[bounce_r, bounce_c] = wall_color
            cur_color = wall_color

    r0, c0, r1, c1 = _ray_bbox(ray_path)
    cropped = np.array(full_out[r0 : r1 + 1, c0 : c1 + 1], dtype=int, copy=True)

    if cropped.shape[0] >= H or cropped.shape[1] >= W:
        return None
    if r0 < _MARGIN or c0 < _MARGIN:
        return None

    solved = main(grid)
    if not np.array_equal(solved, cropped):
        return None

    return grid


def generate_input() -> np.ndarray:
    """Generate the single-arrow, high-reflection, cropped-output combo variant."""
    for _ in range(1200):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 1200 attempts")
