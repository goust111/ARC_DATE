# Variant of 142ca369
# Dimension: D7 — 墙壁/障碍物形状 (obstacle)
# Change: 单像素或3像素轴对齐线段墙壁，放置在射线对角线路径旁偏移1格的位置... → L形或T形墙壁

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

# Design analysis for the D7 variant
# ----------------------------------
# a) Concrete behavior
#    - Arrows stay unchanged: a monochrome 3-cell L inside a 2x2 box.
#    - Walls become 4-cell polyominoes: either an L tetromino or a T tetromino.
#    - The ray still moves diagonally cell by cell. After it advances to a ray cell
#      (r,c), it now inspects the *shape* touching that cell:
#         row-side touch at (r+dr, c)    -> flip dr
#         col-side touch at (r, c+dc)    -> flip dc
#      If the same T wall occupies both touch positions, the T orientation decides
#      which axis flips:
#         horizontal-bar T -> flip dr
#         vertical-bar T   -> flip dc
#      That "two-edge touch but still single-axis reflection" is the new behavior
#      that cannot be produced by the original single-pixel / straight-segment walls.
#    - Small example with dr=dc=+1:
#         old: a vertical 3-line at {(r-1,c+1),(r,c+1),(r+1,c+1)} can only flip dc.
#         new: a T wall at {(r+1,c),(r,c+1),(r+1,c+1),(r+2,c+1)} touches both sides,
#              so the ray cell (r,c) flips both dr and dc and leaves in (-1,-1).
#
# b) Complexity calibration
#    - Walls use only two families (L/T), keeping the shape vocabulary small enough
#      for ARC, but large enough that the visible silhouette matters.
#    - We force at least one T corner-touch bounce and at least one L wall in every
#      generated grid, so the variant is observable without exploding case count.
#
# c) Implicit assumptions that break from the original solver
#    - Old code assumed "wall_cells" is enough. That fails here because two wall
#      pixels may belong to one shape whose exposed sides jointly determine a single
#      bounce axis. The new solver tracks wall components and inspects the component's
#      continuation to resolve T-corner touches.
#    - Old generator relied on wall shapes being tiny/linear and locally harmless.
#      Here an extra arm of an L/T can accidentally touch an earlier ray cell and
#      create a hidden bounce. The new generator rejects any wall placement that
#      falls on a previously reserved side-touch slot.
#    - Arrow detection still assumes only 3-cell 2x2 L components are arrows. To
#      keep that valid, walls are restricted to size 4 polyominoes only.
#
# d) Placement legality
#    - No overlaps between arrows, rays, and walls.
#    - No wall cell may occupy a previously reserved side-touch position of another
#      already-traced ray cell.
#    - Same-color 4-neighbor mergers are rejected so arrow/wall components remain
#      separable under the per-color component scan used by main().
#
# e) Visual observability
#    - Every sample contains actual polyomino walls, not degenerate point/line walls.
#    - At least one bounce is caused by touching two sides of the same T wall, and
#      the T's orientation determines whether the ray turns "vertically" or
#      "horizontally" at that corner.
#
# f) Visual cleanliness
#    - The generator spaces objects conservatively: new walls cannot intrude into old
#      ray-touch slots, and same-color mergers are blocked. This keeps silhouettes and
#      ray paths visually readable.
#
# g) Color inference
#    - Color inheritance is unchanged from the source task: rays adopt the color of
#      the wall they touch. The new difficulty comes from geometry, not from adding a
#      new color mapping rule.


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _transform_shape(
    cells: set[tuple[int, int]], rot: int, reflect: bool
) -> set[tuple[int, int]]:
    out = set(cells)
    if reflect:
        out = {(r, -c) for r, c in out}
    for _ in range(rot % 4):
        out = {(c, -r) for r, c in out}
    return set(_normalize_shape(out))


def _all_orientations(
    base: set[tuple[int, int]], *, allow_reflect: bool
) -> list[tuple[tuple[int, int], ...]]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    for rot in range(4):
        variants.add(_normalize_shape(_transform_shape(base, rot, False)))
        if allow_reflect:
            variants.add(_normalize_shape(_transform_shape(base, rot, True)))
    return sorted(variants)


_BASE_L4 = {(0, 0), (1, 0), (2, 0), (2, 1)}
_BASE_T4 = {(0, 0), (0, 1), (0, 2), (1, 1)}
_WALL_VARIANTS = {
    "L": _all_orientations(_BASE_L4, allow_reflect=True),
    "T": _all_orientations(_BASE_T4, allow_reflect=False),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _max_diag_steps(H: int, W: int, r: int, c: int, dr: int, dc: int) -> int:
    return min((H - 1 - r) if dr == 1 else r, (W - 1 - c) if dc == 1 else c)


def _flip_dir(dr: int, dc: int, flip_kind: str) -> tuple[int, int]:
    if flip_kind in {"dr", "both"}:
        dr = -dr
    if flip_kind in {"dc", "both"}:
        dc = -dc
    return int(dr), int(dc)


def _candidate_effect_kind(
    wall_cells: set[tuple[int, int]], r: int, c: int, dr: int, dc: int, flip_kind: str
) -> str:
    if flip_kind != "both":
        return flip_kind
    if (r + dr, c + 2 * dc) in wall_cells:
        return "dr"
    if (r + 2 * dr, c + dc) in wall_cells:
        return "dc"
    return "both"


def _touch_positions_from_states(
    states: list[tuple[int, int, int, int]]
) -> set[tuple[int, int]]:
    touch = set()
    for r, c, dr, dc in states:
        touch.add((int(r + dr), int(c)))
        touch.add((int(r), int(c + dc)))
    return touch


def _relative_wall_candidates(dr: int, dc: int, flip_kind: str) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
    mandatory = set()
    forbidden = set()
    if flip_kind in {"dr", "both"}:
        mandatory.add((dr, 0))
    else:
        forbidden.add((dr, 0))
    if flip_kind in {"dc", "both"}:
        mandatory.add((0, dc))
    else:
        forbidden.add((0, dc))

    opts: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    seen: set[tuple[str, tuple[tuple[int, int], ...]]] = set()
    first_req = next(iter(mandatory))
    for shape_name, variants in _WALL_VARIANTS.items():
        for variant in variants:
            pts = list(variant)
            for vr, vc in pts:
                shift_r = first_req[0] - vr
                shift_c = first_req[1] - vc
                shifted = tuple(sorted((r + shift_r, c + shift_c) for r, c in pts))
                shifted_set = set(shifted)
                if mandatory <= shifted_set and forbidden.isdisjoint(shifted_set) and (0, 0) not in shifted_set:
                    key = (shape_name, shifted)
                    if key not in seen:
                        seen.add(key)
                        opts.append(key)
    return opts


def _wall_component_info(g: np.ndarray, arrow_cells: set[tuple[int, int]], bg: int):
    wall_id_by_cell: dict[tuple[int, int], int] = {}
    wall_color: dict[int, int] = {}
    wall_cells_by_id: dict[int, set[tuple[int, int]]] = {}
    wid = 0
    for col in colors_present(g, exclude={bg}):
        mask = (g == int(col))
        for coords in components_mask(mask, connectivity=4):
            pts = {(int(r), int(c)) for r, c in coords}
            if pts & arrow_cells:
                continue
            if not pts:
                continue
            wall_color[wid] = int(col)
            wall_cells_by_id[wid] = pts
            for cell in pts:
                wall_id_by_cell[cell] = wid
            wid += 1
    return wall_id_by_cell, wall_color, wall_cells_by_id


def _touch_effect_axis(
    wall_cells: set[tuple[int, int]], nr: int, nc: int, dr: int, dc: int
) -> set[str]:
    row_touch = (nr + dr, nc) in wall_cells
    col_touch = (nr, nc + dc) in wall_cells
    if row_touch and not col_touch:
        return {"dr"}
    if col_touch and not row_touch:
        return {"dc"}
    if row_touch and col_touch:
        if (nr + dr, nc + 2 * dc) in wall_cells:
            return {"dr"}
        if (nr + 2 * dr, nc + dc) in wall_cells:
            return {"dc"}
        return {"dr", "dc"}
    return set()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from 3-cell L-shaped arrows.

    Variant rule:
    - Walls are connected L/T polyominoes instead of isolated point/line walls.
    - After a ray advances to (r,c), it examines wall contact on the row-side
      (r+dr,c) and col-side (r,c+dc). Each touched side flips one axis.
    - If a T wall occupies both touched side positions, the T's orientation chooses
      whether the bounce flips dr or dc.
    - On any bounce the ray adopts that wall's color and recolors the bounce cell.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
        if len(pts) != 3:
            return False
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)

    def arrow_dir_and_start(pts: list[tuple[int, int]]):
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
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
            for rr, cc in _neighbors4(r, c):
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

    wall_id_by_cell, wall_color, wall_cells_by_id = _wall_component_info(g, arrow_cells, bg)
    wall_cells = set(wall_id_by_cell)

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

            if (nr, nc) in wall_cells:
                wid = wall_id_by_cell[(nr, nc)]
                wcol = int(wall_color[wid])
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

            touched: dict[int, set[str]] = {}
            row_side = (nr + int(dr), nc)
            col_side = (nr, nc + int(dc))
            if row_side in wall_id_by_cell:
                wid = wall_id_by_cell[row_side]
                touched.setdefault(wid, set()).update(
                    _touch_effect_axis(wall_cells_by_id[wid], nr, nc, int(dr), int(dc))
                )
            if col_side in wall_id_by_cell:
                wid = wall_id_by_cell[col_side]
                touched.setdefault(wid, set()).update(
                    _touch_effect_axis(wall_cells_by_id[wid], nr, nc, int(dr), int(dc))
                )

            if touched:
                best_wid = min(touched, key=lambda wid: (-len(touched[wid]), wid))
                axes = set().union(*touched.values())
                wcol = int(wall_color[best_wid])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                if "dr" in axes:
                    dr = -int(dr)
                if "dc" in axes:
                    dc = -int(dc)

    return out


def _gen_attempt() -> np.ndarray | None:
    """Output-first generator for the D7 wall-shape variant."""
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(2, 3)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied_all: set[tuple[int, int]] = set()
    occupied_walls: set[tuple[int, int]] = set()
    committed_ray_states: list[tuple[int, int, int, int]] = []

    for ai in range(n_arrows):
        color = arrow_colors[ai]
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        if ai == 0:
            bounce_plan = [("T", "both")]
        elif ai == 1:
            bounce_plan = [("L", _rng.choice(["dr", "dc"]))]
        else:
            bounce_plan = []
            for _ in range(_rng.choices([1, 2], weights=[3, 1])[0]):
                flip_kind = _rng.choices(["dr", "dc", "both"], weights=[4, 4, 1])[0]
                shape_name = "T" if flip_kind == "both" else _rng.choice(["L", "T"])
                bounce_plan.append((shape_name, flip_kind))
        n_bounces = len(bounce_plan)

        placed = False
        for _ in range(140):
            r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
            c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)

            mr, mc = sr - dr, sc - dc
            arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
            arrow_set = set(arrow_pts)
            if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied_all for r, c in arrow_pts):
                continue

            cur_r, cur_c, cur_dr, cur_dc, cur_col = sr, sc, dr, dc, color
            ray_cells: list[tuple[int, int, int]] = []
            ray_states: list[tuple[int, int, int, int]] = []
            ray_set: set[tuple[int, int]] = set()
            walls: list[tuple[set[tuple[int, int]], int, str, str]] = []
            local_wall_set: set[tuple[int, int]] = set()
            local_input_colors: dict[tuple[int, int], int] = {cell: color for cell in arrow_set}
            ok = True

            for seg_idx in range(n_bounces + 1):
                max_steps = _max_diag_steps(H, W, cur_r, cur_c, cur_dr, cur_dc)
                if seg_idx < n_bounces:
                    min_seg = 2
                    max_seg = max_steps - 1
                    if max_seg < min_seg:
                        ok = False
                        break
                    seg_len = _rng.randint(min_seg, max_seg)
                else:
                    seg_len = max_steps

                for _step in range(seg_len):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        ok = False
                        break
                    if (nr, nc) in occupied_all or (nr, nc) in ray_set or (nr, nc) in arrow_set or (nr, nc) in local_wall_set:
                        ok = False
                        break
                    if (nr + cur_dr, nc) in occupied_walls or (nr + cur_dr, nc) in local_wall_set:
                        ok = False
                        break
                    if (nr, nc + cur_dc) in occupied_walls or (nr, nc + cur_dc) in local_wall_set:
                        ok = False
                        break
                    ray_cells.append((nr, nc, cur_col))
                    ray_states.append((nr, nc, cur_dr, cur_dc))
                    ray_set.add((nr, nc))
                    cur_r, cur_c = nr, nc
                if not ok:
                    break

                if seg_idx >= n_bounces:
                    continue

                remaining_bounces = n_bounces - seg_idx - 1
                target_shape, target_flip = bounce_plan[seg_idx]
                bounce_options: list[tuple[str, str, set[tuple[int, int]], tuple[int, int]]] = []
                reserved_touch = _touch_positions_from_states(committed_ray_states + ray_states[:-1])
                for flip_kind in [target_flip]:
                    for shape_name, rel_cells in _relative_wall_candidates(cur_dr, cur_dc, flip_kind):
                        if shape_name != target_shape:
                            continue
                        abs_cells = {(cur_r + rr, cur_c + cc) for rr, cc in rel_cells}
                        if any(not (0 <= r < H and 0 <= c < W) for r, c in abs_cells):
                            continue
                        if abs_cells & occupied_all or abs_cells & ray_set or abs_cells & arrow_set or abs_cells & local_wall_set:
                            continue
                        if abs_cells & reserved_touch:
                            continue
                        effect_kind = _candidate_effect_kind(abs_cells, cur_r, cur_c, cur_dr, cur_dc, flip_kind)
                        next_dr, next_dc = _flip_dir(cur_dr, cur_dc, effect_kind)
                        future_steps = _max_diag_steps(H, W, cur_r, cur_c, next_dr, next_dc)
                        if future_steps < (2 if remaining_bounces > 0 else 1):
                            continue
                        bounce_options.append((shape_name, flip_kind, abs_cells, (next_dr, next_dc)))

                if not bounce_options:
                    ok = False
                    break

                shape_name, flip_kind, wall_cells, (next_dr, next_dc) = _rng.choice(bounce_options)

                forbidden_same_color: set[int] = set()
                for r, c in wall_cells:
                    for rr, cc in _neighbors4(r, c):
                        if (rr, cc) in local_input_colors:
                            forbidden_same_color.add(int(local_input_colors[(rr, cc)]))
                        elif 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) != 0:
                            forbidden_same_color.add(int(grid[rr, cc]))
                wall_palette = [col for col in palette if col not in forbidden_same_color]
                if not wall_palette:
                    ok = False
                    break
                wcol = int(_rng.choice(wall_palette))

                walls.append((wall_cells, wcol, shape_name, flip_kind))
                for cell in wall_cells:
                    local_wall_set.add(cell)
                    local_input_colors[cell] = wcol

                if ray_cells:
                    lr, lc, _ = ray_cells[-1]
                    ray_cells[-1] = (lr, lc, wcol)
                cur_col = wcol
                cur_dr, cur_dc = next_dr, next_dc

            if not ok or len(ray_cells) < 4:
                continue

            new_cells = set(arrow_set) | set(ray_set) | set(local_wall_set)
            if new_cells & occupied_all:
                continue

            for r, c in arrow_pts:
                grid[r, c] = out[r, c] = color
            for r, c, col in ray_cells:
                out[r, c] = col
            for wall_cells, wcol, shape_name, flip_kind in walls:
                for r, c in wall_cells:
                    grid[r, c] = out[r, c] = wcol

            occupied_all |= new_cells
            occupied_walls |= local_wall_set
            committed_ray_states.extend(ray_states)
            placed = True
            break

        if not placed:
            return None

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D7 variant of 142ca369."""
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
