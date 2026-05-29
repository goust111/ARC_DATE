# Variant of 142ca369
# Dimension: D20 — 射线与射线的交互 (multi_step)
# Change: 射线之间无交互——各自独立涂色，后处理的射线可以覆盖先前射线的轨迹... → 先到先得规则

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

# Design analysis
# spec:
#   concrete_behavior:
#     objects:
#       arrows: unchanged 3-pixel L in a 2x2 box; missing corner still determines
#         the diagonal launch direction; the right-angle vertex is still the source.
#       walls: unchanged single pixels or 3-pixel axis-aligned segments; they still
#         cause the same side-touch / corner reflections and color inheritance.
#       rays: still paint diagonally on bg cells, but after each arrow finishes, its
#         painted cells become dynamic blockers for all later arrows.
#     ordered_processing:
#       rule: arrows are processed by the source pixel (right-angle vertex) in
#         lexicographic (row, col) order, i.e. top-left to bottom-right.
#       blocker_scope: only cells painted by already completed rays are blockers.
#         Input arrows and input walls keep their original meaning and are handled
#         exactly as before.
#   behavior_trace:
#     setup:
#       arrow_a: start=(3,3), dir=(1,1), paints the \ diagonal through (8,8)
#       arrow_b: start=(13,3), dir=(-1,1), would cross at (8,8)
#     original_solver:
#       arrow_a paints first; arrow_b later steps onto (8,8) and keeps going,
#       because the old solver only writes when out[nr,nc] == bg and never treats
#       existing ray paint as a collision.
#     variant_solver:
#       after arrow_a finishes, (8,8) is in blocked_cells. arrow_b can paint up to
#       (9,7); when its next step would enter (8,8), it terminates immediately.
#     visible_difference:
#       the same input now yields a truncated later ray instead of a full X crossing.
#   complexity_calibration:
#     keep: reflections, color inheritance, arrow detection, and stop-on-arrow rules.
#     change_only: dynamic interaction between rays.
#     generator_adaptation: every accepted sample must contain at least one later ray
#       that would continue under the old solver but is truncated by blocked_cells.
#   implicit_assumption_review:
#     old_main_dependency_1:
#       code: if out[nr, nc] == bg: out[nr, nc] = cur_col
#       old assumption: a non-bg out cell is harmless bookkeeping and does not alter
#         motion; later rays can pass through it.
#       variant impact: no longer valid; later rays must stop before entering such a
#         previously painted ray cell.
#     old_main_dependency_2:
#       code: arrows are iterated in detection order.
#       old assumption: order is irrelevant because rays do not interact.
#       variant impact: order now changes the answer, so we sort arrows by source.
#     old_main_dependency_3:
#       code: seen state omits any global occupancy.
#       old assumption: only self-loops matter.
#       variant impact: still valid because blockers come only from earlier rays; the
#         current ray does not collide with its own trail unless it creates a loop.
#   placement_legality:
#     mandatory_interaction:
#       two arrows are placed so their straight diagonals cross, with the earlier
#       source above the later source; the crossing is well inside the grid so the
#       later ray has a visible missing tail after collision.
#     optional_extra_arrows:
#       planned with the original bounce generator style, but only in later source
#       positions and with no overlap against earlier output cells, so the solver's
#       dynamic blockers never invalidate the planned path.
#     wall_colors:
#       chosen from colors not used by arrows, preventing same-color 4-connected
#       merges that could break arrow component detection.
#   visual_observability:
#     the generator rejects any sample where new_output == old_output, so examples
#     consistently display at least one truncated crossing.
#   visual_cleanliness:
#     only one deliberate ray-ray collision is required; all other planned paths are
#     non-overlapping, keeping the grid readable.
#   color_inference:
#     D20 itself is geometric, not a new color mapping. Existing wall-color adoption
#     stays in place, and only a small color palette subset is used per sample.
#   new_main_assumptions_audit:
#     assumption_1: each arrow source is uniquely locatable from its 3-pixel L shape.
#       generator guarantee: arrows never overlap and keep a clear 2x2 footprint.
#     assumption_2: ordering by source pixel is deterministic.
#       generator guarantee: starts are distinct; extras are only accepted if their
#       source sorts after the mandatory interacting pair.
#     assumption_3: blocked cells come only from earlier rays, not input walls.
#       generator guarantee: later planned rays avoid existing output cells unless the
#       intended mandatory collision is the one that demonstrates the variant.


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


def _extract_scene(g: np.ndarray) -> tuple[int, list[dict[str, object]], set[tuple[int, int]], set[tuple[int, int]]]:
    bg = int(get_bg(g))
    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[dict[str, object]] = []

    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if not _is_arrow_pts(pts):
                continue
            info = _arrow_dir_and_start(pts)
            if info is None:
                continue
            (dr, dc), start = info
            arrows.append(
                {
                    "color": int(col),
                    "pts": pts,
                    "dir": (int(dr), int(dc)),
                    "start": (int(start[0]), int(start[1])),
                }
            )
            arrow_cells |= set(pts)

    arrows.sort(key=lambda a: (a["start"], a["pts"], a["color"]))
    wall_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }
    return bg, arrows, arrow_cells, wall_cells


def _simulate(
    grid: np.ndarray,
    *,
    stop_on_painted: bool,
    collect_meta: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, arrows, arrow_cells, wall_cells = _extract_scene(g)
    out = g.copy()
    blocked_cells: set[tuple[int, int]] = set()
    ray_records: list[dict[str, object]] = []

    for arrow in arrows:
        start_col = int(arrow["color"])
        dr, dc = arrow["dir"]
        cur = tuple(arrow["start"])
        cur_col = int(start_col)
        seen = set()
        local_painted: set[tuple[int, int]] = set()
        stop_reason = "exit"

        while True:
            state = (cur, int(dr), int(dc), int(cur_col))
            if state in seen:
                stop_reason = "loop"
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                stop_reason = "exit"
                break

            if (nr, nc) in wall_cells:
                wcol = int(g[nr, nc])
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                stop_reason = "corner_bounce"
                continue

            if (nr, nc) in arrow_cells:
                stop_reason = "arrow"
                break

            if stop_on_painted and (nr, nc) in blocked_cells:
                stop_reason = "painted_collision"
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)
            local_painted.add(cur)
            stop_reason = "exit"

            if int(dr) == 1 and nr + 1 < H and (nr + 1, nc) in wall_cells:
                wcol = int(g[nr + 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)
                stop_reason = "side_bounce"
            elif int(dr) == -1 and nr - 1 >= 0 and (nr - 1, nc) in wall_cells:
                wcol = int(g[nr - 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)
                stop_reason = "side_bounce"

            if int(dc) == 1 and nc + 1 < W and (nr, nc + 1) in wall_cells:
                wcol = int(g[nr, nc + 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)
                stop_reason = "side_bounce"
            elif int(dc) == -1 and nc - 1 >= 0 and (nr, nc - 1) in wall_cells:
                wcol = int(g[nr, nc - 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)
                stop_reason = "side_bounce"

        blocked_cells |= local_painted
        ray_records.append(
            {
                "start": tuple(arrow["start"]),
                "dir": tuple(arrow["dir"]),
                "color": int(start_col),
                "painted_count": len(local_painted),
                "stop_reason": stop_reason,
            }
        )

    meta = {
        "ray_records": ray_records,
        "blocked_count": len(blocked_cells),
        "arrow_count": len(arrows),
    }
    if collect_meta:
        return out, meta
    return out, {}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-shaped arrows with ordered ray-ray collisions.

    The original wall reflection and color inheritance behavior is preserved.
    New behavior: arrows are processed from top-left to bottom-right by source
    position, and cells painted by earlier rays become solid blockers for later rays.
    """
    return _simulate(grid, stop_on_painted=True)[0]


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, stop_on_painted=False)[0]


def _build_arrow_pts(start: tuple[int, int], dr: int, dc: int) -> list[tuple[int, int]]:
    sr, sc = start
    mr, mc = sr - dr, sc - dc
    return [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]


def _mandatory_pair(
    H: int,
    W: int,
    color_a: int,
    color_b: int,
) -> tuple[list[dict[str, object]], list[tuple[int, int]], tuple[int, int]] | None:
    if H < 18 or W < 18:
        return None

    for _ in range(80):
        case = _rng.choice([0, 1])
        cr = _rng.randint(5, H - 6)
        cc = _rng.randint(5, W - 6)
        ta = _rng.randint(3, 5)
        tb = _rng.randint(3, 5)

        if case == 0:
            dir_a = (1, 1)
            dir_b = (-1, 1)
            start_a = (cr - ta, cc - ta)
            start_b = (cr + tb, cc - tb)
            tail_a = min(H - 1 - cr, W - 1 - cc)
            tail_b = min(cr, W - 1 - cc)
            path_a = [(r, c) for r, c in zip(range(start_a[0] + 1, H), range(start_a[1] + 1, W))]
            path_b = []
            r, c = start_b
            while True:
                r -= 1
                c += 1
                if not (0 <= r < H and 0 <= c < W):
                    break
                if (r, c) == (cr, cc):
                    break
                path_b.append((r, c))
        else:
            dir_a = (1, -1)
            dir_b = (-1, -1)
            start_a = (cr - ta, cc + ta)
            start_b = (cr + tb, cc + tb)
            tail_a = min(H - 1 - cr, cc)
            tail_b = min(cr, cc)
            path_a = [(r, c) for r, c in zip(range(start_a[0] + 1, H), range(start_a[1] - 1, -1, -1))]
            path_b = []
            r, c = start_b
            while True:
                r -= 1
                c -= 1
                if not (0 <= r < H and 0 <= c < W):
                    break
                if (r, c) == (cr, cc):
                    break
                path_b.append((r, c))

        if tail_a < 4 or tail_b < 4:
            continue

        arrow_a = _build_arrow_pts(start_a, *dir_a)
        arrow_b = _build_arrow_pts(start_b, *dir_b)
        if any(not (0 <= r < H and 0 <= c < W) for r, c in arrow_a + arrow_b):
            continue

        cells_a = set(arrow_a)
        cells_b = set(arrow_b)
        if cells_a & cells_b:
            continue
        if set(path_a) & (cells_a | cells_b):
            continue
        if set(path_b) & (cells_a | cells_b):
            continue
        if (cr, cc) in cells_a or (cr, cc) in cells_b:
            continue
        if tuple(start_a) >= tuple(start_b):
            continue
        if len(path_b) < 3:
            continue

        arrows = [
            {"start": start_a, "dir": dir_a, "color": color_a, "pts": arrow_a, "ray_cells": path_a},
            {"start": start_b, "dir": dir_b, "color": color_b, "pts": arrow_b, "ray_cells": path_b},
        ]
        return arrows, path_a + path_b, start_b
    return None


def _plan_extra_arrow(
    H: int,
    W: int,
    color: int,
    wall_palette: list[int],
    occupied_input: set[tuple[int, int]],
    occupied_output: set[tuple[int, int]],
    min_start: tuple[int, int],
) -> tuple[dict[str, object], set[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(120):
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = _rng.choices([0, 1, 2], weights=[2, 3, 2])[0]

        r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 2)
        c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 2)
        if r_lo > r_hi or c_lo > c_hi:
            continue
        sr = _rng.randint(r_lo, r_hi)
        sc = _rng.randint(c_lo, c_hi)
        start = (sr, sc)
        if start <= min_start:
            continue

        arrow_pts = _build_arrow_pts(start, dr, dc)
        if any(not (0 <= r < H and 0 <= c < W) for r, c in arrow_pts):
            continue
        arrow_set = set(arrow_pts)
        if any((r, c) in occupied_input or (r, c) in occupied_output for r, c in arrow_pts):
            continue

        cur_r, cur_c = sr, sc
        cur_dr, cur_dc = dr, dc
        cur_col = color
        ray_cells: list[tuple[int, int, int]] = []
        ray_set: set[tuple[int, int]] = set()
        walls: list[tuple[list[tuple[int, int]], int]] = []
        local_input = set(arrow_set)
        local_output = set(arrow_set)
        ok = True
        bounces_left = n_bounces

        for seg_idx in range(n_bounces + 1):
            max_steps = min((H - 1 - cur_r) if cur_dr == 1 else cur_r, (W - 1 - cur_c) if cur_dc == 1 else cur_c)
            if seg_idx < n_bounces and bounces_left > 0:
                if max_steps < 4:
                    bounces_left = 0
                    seg_len = max_steps
                else:
                    seg_len = _rng.randint(3, max(3, max_steps - 1))
            else:
                seg_len = max_steps

            for _step in range(seg_len):
                nr, nc = cur_r + cur_dr, cur_c + cur_dc
                if not (0 <= nr < H and 0 <= nc < W):
                    break
                if (nr, nc) in occupied_output or (nr, nc) in ray_set or (nr, nc) in arrow_set:
                    ok = False
                    break
                ray_cells.append((nr, nc, cur_col))
                ray_set.add((nr, nc))
                local_output.add((nr, nc))
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
            chosen = None
            for wpxs, flip in options:
                if any(
                    (r, c) in occupied_input
                    or (r, c) in occupied_output
                    or (r, c) in ray_set
                    or (r, c) in arrow_set
                    or (r, c) in local_input
                    for r, c in wpxs
                ):
                    continue
                chosen = (wpxs, flip)
                break
            if chosen is None or not wall_palette:
                ok = False
                break

            wpxs, flip = chosen
            wcol = _rng.choice(wall_palette)
            walls.append((wpxs, wcol))
            for cell in wpxs:
                local_input.add(cell)
                local_output.add(cell)

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

        return (
            {"start": start, "dir": (dr, dc), "color": color, "pts": arrow_pts, "ray_cells": ray_cells, "walls": walls},
            local_input,
            local_output,
        )
    return None


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    new_out, meta = _simulate(grid, stop_on_painted=True, collect_meta=True)
    old_out = _legacy_main(grid)
    collision_count = sum(1 for rec in meta["ray_records"] if rec["stop_reason"] == "painted_collision")
    changed_cells = int(np.count_nonzero(new_out != old_out))
    return {
        "collision_count": int(collision_count),
        "changed_cells_vs_legacy": changed_cells,
        "arrow_count": int(meta["arrow_count"]),
        "blocked_count": int(meta["blocked_count"]),
    }


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:3]
    wall_palette = palette[3:] if len(palette) > 3 else [9]

    pair = _mandatory_pair(H, W, arrow_colors[0], arrow_colors[1])
    if pair is None:
        return None
    arrows, reserved_paths, min_start = pair

    grid = np.zeros((H, W), dtype=int)
    occupied_input: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set(reserved_paths)

    for arrow in arrows:
        for r, c in arrow["pts"]:
            if (r, c) in occupied_input:
                return None
            grid[r, c] = int(arrow["color"])
            occupied_input.add((r, c))
            occupied_output.add((r, c))

    if _rng.random() < 0.75:
        extra = _plan_extra_arrow(
            H,
            W,
            arrow_colors[2],
            wall_palette,
            occupied_input,
            occupied_output,
            min_start,
        )
        if extra is not None:
            arrow, local_input, local_output = extra
            for r, c in arrow["pts"]:
                grid[r, c] = int(arrow["color"])
            for wpxs, wcol in arrow.get("walls", []):
                for r, c in wpxs:
                    grid[r, c] = int(wcol)
            occupied_input |= local_input
            occupied_output |= local_output

    metrics = _variant_metrics(grid)
    if metrics["collision_count"] < 1:
        return None
    if metrics["changed_cells_vs_legacy"] < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate valid inputs that visibly exercise ordered ray-ray blocking."""
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
