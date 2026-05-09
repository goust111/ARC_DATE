# Variant of 142ca369
# Dimension: D10 — 反射后变换效果 (transform)
# Change: 反射点重新涂色为墙壁颜色（recolor），射线继承墙壁颜色... → 反射时墙壁扩展

from __future__ import annotations

"""
Design analysis for the D10 variant
==================================

source_task:
  none

variant_spec:
  core_change:
    original: at each bounce the ray adopts the contacted wall color and only the
      single reflection cell is recolored.
    variant: keep color inheritance, but replace reflection-cell recolor with
      wall growth. The contacted monochrome wall component expands by 1 cell in
      the direction the ray came from.
  concrete_pixel_behavior:
    side_bounce:
      horizontal_hit: if the wall lies at (r+dr, c), the whole connected
        component of that wall color containing the hit pixel is copied one row
        toward -dr.
      vertical_hit: if the wall lies at (r, c+dc), the whole connected component
        is copied one column toward -dc.
      effect: for a 3-pixel line wall, one bounce turns a thin line into a
        thicker 2x3 or 3x2 block. The center reflection cell may become wall,
        but extra adjacent wall pixels also appear, making the D10 change visible.
    corner_bounce:
      rare in generated data, but supported for robustness. The contacted wall
      component grows by (-dr, -dc), then the ray reverses both axes.

  flow_trace_example:
    original:
      - ray advances diagonally to bounce cell P
      - adjacent 3-pixel wall below P flips dr
      - P is recolored to wall color
      - wall shape itself stays a 3-pixel line
    variant:
      - ray advances diagonally to bounce cell P
      - the same wall flips dr
      - the entire 3-pixel wall is extruded upward by 1 row
      - output now shows a thicker 2x3 wall region, and future rays treat the
        new cells as real obstacles

  complexity_calibration:
    - Keep arrows unchanged so the matching problem remains the same.
    - Bias generation toward 3-pixel walls and 1-4 bounces so growth is visible.
    - Preserve a small number of wall types (1-pixel and 3-pixel axis-aligned
      segments) to avoid adding unrelated freedoms.

  implicit_assumption_audit:
    - original `seen` state assumed walls were static. That is false now, so the
      dynamic simulator includes a wall-layout signature in the loop state.
    - original side-bounce code could mutate the output cell unconditionally
      because only recolor changed. That is false now: the bounce cell should
      keep ray paint unless wall growth physically occupies it.
    - original wall lookup used the input grid `g` because walls never changed.
      That is false now, so dynamic wall colors are stored in a mutable map.
    - original generator reserved only initial walls. That is false now: it must
      also reserve the cells that will be occupied after growth, otherwise later
      rays or arrows could be placed into future obstacles.

  placement_legality:
    - arrows, initial walls, planned ray cells, and future grown wall cells are
      kept disjoint in generation, except that a growth strip may include the
      current bounce cell because that is the intended contact point.
    - growth is clipped only by bounds / arrows / differently-colored walls in
      the solver; generation avoids such clipping so examples stay clean.

  visual_observability:
    - at least one 3-pixel wall bounce is required by the generator.
    - generated inputs are accepted only if the variant output differs from the
      original solver output, preventing silent regression to the old behavior.

  visual_cleanliness:
    - future grown cells are reserved, so the final boards do not accumulate
      ambiguous overlaps between unrelated arrows, rays, and expanded walls.

  color_reasoning:
    - color inheritance is unchanged from the base task, so the only new visual
      rule is geometric growth, not a new color mapping burden.

  new_solver_assumptions_and_generator_guarantees:
    - solver assumes every true arrow is still an isolated 3-cell L component;
      generator keeps arrow colors unique and avoids 4-neighbor touching.
    - solver assumes growth should not overwrite arrows; generator reserves that.
    - solver allows repeated hits on already-grown components; generator mostly
      uses single-hit clean layouts, but the logic remains correct for cumulative
      growth when it occurs.
"""

import random as _rng
from collections import deque

import numpy as np

from arc_utils import colors_present, components_mask, get_bg


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in pts_set:
                deg[(r, c)] += 1
    start = max(deg, key=lambda p: deg[p])
    return (int(dr), int(dc)), (int(start[0]), int(start[1]))


def _detect_arrows(g: np.ndarray, bg: int):
    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)
    return arrows, arrow_cells


def _wall_component(
    wall_colors: dict[tuple[int, int], int], start: tuple[int, int]
) -> list[tuple[int, int]]:
    color = int(wall_colors[start])
    q = deque([start])
    seen = {start}
    comp: list[tuple[int, int]] = []
    while q:
        cell = q.popleft()
        comp.append(cell)
        r, c = cell
        for nb in _neighbors4(r, c):
            if nb in seen:
                continue
            if int(wall_colors.get(nb, -1)) != color:
                continue
            seen.add(nb)
            q.append(nb)
    return comp


def _grow_wall(
    out: np.ndarray,
    wall_colors: dict[tuple[int, int], int],
    arrow_cells: set[tuple[int, int]],
    hit: tuple[int, int],
    grow_vec: tuple[int, int],
    *,
    collect: dict | None = None,
) -> int:
    H, W = out.shape
    color = int(wall_colors[hit])
    comp = _wall_component(wall_colors, hit)
    if collect is not None:
        collect["growth_events"] += 1
        collect["hit_component_sizes"].append(len(comp))

    for r, c in comp:
        tr = int(r + grow_vec[0])
        tc = int(c + grow_vec[1])
        if not (0 <= tr < H and 0 <= tc < W):
            continue
        if (tr, tc) in arrow_cells:
            continue
        prev = wall_colors.get((tr, tc))
        if prev is not None and int(prev) != color:
            continue
        if prev is None:
            wall_colors[(tr, tc)] = color
            if collect is not None:
                collect["grown_cells"].add((tr, tc))
        out[tr, tc] = color
    return color


def _simulate(grid: np.ndarray, *, grow_walls: bool, collect_meta: bool = False):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    arrows, arrow_cells = _detect_arrows(g, bg)
    wall_colors: dict[tuple[int, int], int] = {
        (int(r), int(c)): int(g[r, c])
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }
    initial_wall_cells = set(wall_colors)
    out = g.copy()
    meta = None
    if collect_meta:
        meta = {
            "growth_events": 0,
            "grown_cells": set(),
            "hit_component_sizes": [],
            "initial_wall_cells": initial_wall_cells,
        }

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
        cur_col = int(start_col)
        seen = set()
        for _ in range(H * W * 8):
            wall_signature = tuple(sorted(wall_colors.items())) if grow_walls else ()
            state = (cur, int(dr), int(dc), int(cur_col), wall_signature)
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in wall_colors:
                if grow_walls:
                    wcol = _grow_wall(
                        out,
                        wall_colors,
                        arrow_cells,
                        (nr, nc),
                        (-int(dr), -int(dc)),
                        collect=meta,
                    )
                else:
                    wcol = int(wall_colors[(nr, nc)])
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

            orig_dr, orig_dc = int(dr), int(dc)
            hit_dr = None
            hit_dc = None
            if orig_dr == 1 and (nr + 1, nc) in wall_colors:
                hit_dr = (nr + 1, nc)
            elif orig_dr == -1 and (nr - 1, nc) in wall_colors:
                hit_dr = (nr - 1, nc)
            if orig_dc == 1 and (nr, nc + 1) in wall_colors:
                hit_dc = (nr, nc + 1)
            elif orig_dc == -1 and (nr, nc - 1) in wall_colors:
                hit_dc = (nr, nc - 1)

            if hit_dr is not None:
                if grow_walls:
                    wcol = _grow_wall(
                        out,
                        wall_colors,
                        arrow_cells,
                        hit_dr,
                        (-orig_dr, 0),
                        collect=meta,
                    )
                else:
                    wcol = int(wall_colors[hit_dr])
                    if int(g[nr, nc]) == int(bg):
                        out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -orig_dr

            if hit_dc is not None:
                if grow_walls:
                    wcol = _grow_wall(
                        out,
                        wall_colors,
                        arrow_cells,
                        hit_dc,
                        (0, -orig_dc),
                        collect=meta,
                    )
                else:
                    wcol = int(wall_colors[hit_dc])
                    if int(g[nr, nc]) == int(bg):
                        out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -orig_dc

        else:
            break

    if meta is not None:
        meta["added_wall_cells"] = len(meta["grown_cells"] - initial_wall_cells)
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from 3-cell L arrows.

    Variant rule:
    - Rays still inherit wall color at every bounce.
    - Instead of recoloring only the bounce point, the hit wall grows by 1 cell
      toward the incoming ray direction.
    - Grown wall cells become real obstacles for later motion.
    """
    return _simulate(grid, grow_walls=True, collect_meta=False)[0]


def _baseline_main(grid: np.ndarray) -> np.ndarray:
    """Original-task behavior used only to reject degenerate generated inputs."""
    return _simulate(grid, grow_walls=False, collect_meta=False)[0]


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    variant_out, meta = _simulate(grid, grow_walls=True, collect_meta=True)
    base_out = _baseline_main(grid)
    return {
        "diff_vs_original": int(np.count_nonzero(variant_out != base_out)),
        "growth_events": int(meta["growth_events"]),
        "added_wall_cells": int(meta["added_wall_cells"]),
        "max_hit_component": int(max(meta["hit_component_sizes"], default=0)),
    }


def _gen_attempt() -> np.ndarray | None:
    """
    Generate a clean input with planned side bounces and reserved growth zones.

    The planner is still output-aware, but the final oracle is the new solver.
    It guarantees at least one visible 3-pixel wall expansion and rejects any
    sample whose output matches the old task.
    """
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(2, 5)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = sorted(palette[:n_arrows])
    wall_palette = [c for c in palette if c not in arrow_colors] or palette[:]

    grid = np.zeros((H, W), dtype=int)
    blocked: set[tuple[int, int]] = set()
    used_long_wall = False
    total_bounces = 0

    for color in arrow_colors:
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = _rng.choices([1, 2, 3, 4], weights=[3, 4, 2, 1])[0]
        placed = False

        for _ in range(80):
            r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
            c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)
            mr, mc = sr - dr, sc - dc
            arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
            arrow_set = set(arrow_pts)
            if any(
                not (0 <= r < H and 0 <= c < W) or (r, c) in blocked
                for r, c in arrow_pts
            ):
                continue

            cur_r, cur_c = sr, sc
            cur_dr, cur_dc = dr, dc
            ray_set: set[tuple[int, int]] = set()
            local_future: set[tuple[int, int]] = set()
            walls_to_place: list[tuple[list[tuple[int, int]], int]] = []
            ok = True
            local_long_wall = False

            for seg_idx in range(n_bounces + 1):
                max_steps = min(
                    (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                    (W - 1 - cur_c) if cur_dc == 1 else cur_c,
                )
                if seg_idx < n_bounces:
                    if max_steps < 3:
                        ok = False
                        break
                    hi = max(2, max_steps - 1)
                    seg_len = _rng.randint(2, hi)
                else:
                    if max_steps < 1:
                        ok = False
                        break
                    seg_len = max_steps

                for _step in range(seg_len):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        ok = False
                        break
                    if (
                        (nr, nc) in blocked
                        or (nr, nc) in arrow_set
                        or (nr, nc) in ray_set
                        or (nr, nc) in local_future
                    ):
                        ok = False
                        break
                    ray_set.add((nr, nc))
                    cur_r, cur_c = nr, nc
                if not ok:
                    break

                if seg_idx >= n_bounces:
                    continue

                bounce_cell = (cur_r, cur_c)
                options: list[tuple[list[tuple[int, int]], str, set[tuple[int, int]]]] = []

                wr, wc = cur_r + cur_dr, cur_c
                if 0 <= wr < H:
                    cand = [[(wr, wc)]]
                    if 0 <= wc - 1 and wc + 1 < W:
                        cand.append([(wr, wc - 1), (wr, wc), (wr, wc + 1)])
                    for wpxs in cand:
                        grown = {(r - cur_dr, c) for r, c in wpxs}
                        options.append((wpxs, "dr", grown))

                wr2, wc2 = cur_r, cur_c + cur_dc
                if 0 <= wc2 < W:
                    cand = [[(wr2, wc2)]]
                    if 0 <= wr2 - 1 and wr2 + 1 < H:
                        cand.append([(wr2 - 1, wc2), (wr2, wc2), (wr2 + 1, wc2)])
                    for wpxs in cand:
                        grown = {(r, c - cur_dc) for r, c in wpxs}
                        options.append((wpxs, "dc", grown))

                if not options:
                    ok = False
                    break

                _rng.shuffle(options)
                require_long_here = (not used_long_wall) and (not local_long_wall)
                options.sort(key=lambda item: len(item[0]), reverse=require_long_here or (_rng.random() < 0.7))

                chosen = None
                for wpxs, flip, grown in options:
                    if any((r, c) in blocked or (r, c) in ray_set or (r, c) in arrow_set or (r, c) in local_future for r, c in wpxs):
                        continue
                    valid = True
                    for cell in grown:
                        gr, gc = cell
                        if not (0 <= gr < H and 0 <= gc < W):
                            valid = False
                            break
                        if cell in arrow_set:
                            valid = False
                            break
                        if cell == bounce_cell:
                            continue
                        if cell in blocked or cell in ray_set or cell in local_future:
                            valid = False
                            break
                    if not valid:
                        continue
                    chosen = (wpxs, flip, grown)
                    break

                if chosen is None:
                    ok = False
                    break

                wpxs, flip, grown = chosen
                wcol = _rng.choice(wall_palette)
                walls_to_place.append((wpxs, wcol))
                local_future.update(wpxs)
                local_future.update(grown)
                local_long_wall |= len(wpxs) == 3
                total_bounces += 1

                if flip == "dr":
                    cur_dr = -cur_dr
                else:
                    cur_dc = -cur_dc

            if not ok or len(ray_set) < 4 or not walls_to_place:
                total_bounces -= len(walls_to_place)
                continue

            for r, c in arrow_pts:
                grid[r, c] = color
            for wpxs, wcol in walls_to_place:
                for r, c in wpxs:
                    grid[r, c] = wcol

            blocked.update(arrow_set)
            blocked.update(ray_set)
            blocked.update(local_future)
            used_long_wall |= local_long_wall
            placed = True
            break

        if not placed:
            return None

    if not used_long_wall or total_bounces < n_arrows:
        return None

    variant_out = main(grid)
    base_out = _baseline_main(grid)
    if np.array_equal(variant_out, base_out):
        return None

    metrics = _variant_metrics(grid)
    if metrics["diff_vs_original"] < 2:
        return None
    if metrics["added_wall_cells"] < 2:
        return None
    if metrics["max_hit_component"] < 3:
        return None
    if metrics["growth_events"] > 16:
        return None
    if metrics["added_wall_cells"] > 36:
        return None
    if metrics["max_hit_component"] > 20:
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D10 wall-growth variant."""
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
