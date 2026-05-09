# Combo variant of 142ca369 (index 15)
# combo_spec: D3/0+D6/1+D17/0
# Name: 周期图案轨迹+固定步数终止+固定段长

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, in_bounds


# Design analysis for the fused variant
# =====================================
# a) Concrete behavior on the pixel grid
# - Base objects stay recognizable: each launcher is still a monochrome 3-cell L
#   in a 2x2 box, and walls are still mono-color 1-pixel or 3-pixel side segments.
# - To fuse D6 and D17 without breaking arrow detection, each arrow now carries two
#   detached same-color diagonal markers:
#     1) reverse-diagonal tail behind the missing corner, length = EXTRA_STEPS.
#     2) mirrored diagonal tail behind the other side of the arrow, length = SEG_LEN.
#   Both start two cells away from the right-angle vertex, so they are never 4-
#   connected to the arrow itself.
# - Ray execution is now controlled by both markers:
#     * total step budget = SEG_LEN + EXTRA_STEPS;
#     * advance one diagonal cell at a time;
#     * every SEG_LEN advances, a scheduled turn event happens;
#     * if a side wall is present at that endpoint, it chooses the flipped axis and
#       donates its color;
#     * if no side wall is present, the ray still auto-turns by flipping dr;
#     * once TOTAL_STEPS advances have been consumed, the ray stops even if more
#       empty cells remain ahead.
# - Coloring fuses D3 into the same trace: after launch and after every scheduled
#   turn, the segment paints with a fresh AABB cycle where
#       A = arrow color, B = current inherited color.
#   Wall-turn endpoints are overwritten to the wall color, exactly like the source
#   task's bounce contact recoloring.
# - Example trace:
#     arrow color 2, SEG_LEN=5, EXTRA_STEPS=4, first turn hits wall color 7.
#     Segment 1: 2 2 2 2   (A=B at launch, so pattern is visually monochrome)
#     Turn 1: endpoint becomes 7
#     Segment 2: 2 2 7 7   (visible AABB, then stop because total budget is exhausted)
#   Original task would instead keep going until the border and would only turn at
#   wall-timed positions, so both path timing and stopping behavior diverge.
#
# b) Complexity calibration
# - The generator restricts SEG_LEN to {4, 5}. Each grid contains at least one
#   wall-turn arrow and one auto-turn arrow, rather than forcing every arrow to do
#   both. That keeps the scene readable and avoids impossible long-tail geometries.
# - The wall-turn arrow uses SEG_LEN=5 and EXTRA_STEPS=4 so the post-wall segment
#   is long enough to expose the AABB pattern. The auto-turn arrow uses smaller
#   EXTRA_STEPS so the fixed total-step rule is visible independently of walls.
# - Total budgets therefore stay in a small band (5-9) while still producing both
#   scheduled turning behavior and an interior stop.
#
# c) Implicit assumptions broken by the fusion
# - Original `wall_cells = all non-bg non-arrow` no longer works, because marker
#   tails would be mistaken for walls. The solver must parse markers first.
# - Original "turn timing is determined by adjacent walls" is false; side walls are
#   inspected only at scheduled segment endpoints.
# - Original `while True` with border termination is false; finite total-step budget
#   must be tracked explicitly.
# - Original uniform-segment coloring is false; output now depends on segment-local
#   phase, so `phase` joins the trace state.
# - Original state also omitted `steps_used` and `steps_in_segment`; both matter now
#   because future motion depends on remaining global budget and remaining steps in
#   the current fixed-length block.
#
# d) Placement legality strategy
# - Generator is output-first. It reserves arrow cells plus both marker tails before
#   planning the ray path.
# - Rays, markers, arrows, and walls never overlap. Auto-turn endpoints also reserve
#   their side-adjacent cells so later objects cannot accidentally create a wall.
# - Final ray endpoints reserve the next diagonal continuation cell, guaranteeing the
#   fixed-step stop is not secretly equivalent to border termination.
#
# e) Visual observability
# - A solver can infer SEG_LEN by counting the mirrored tail and EXTRA_STEPS by
#   counting the reverse-diagonal tail, then add them to get the total step budget.
#   The two counts repeat across examples with the same semantics.
# - The wall-turn arrow always has a post-wall segment of length 4, so the AABB
#   striping is actually visible. A separate auto-turn arrow shows that turn timing
#   is not wall-triggered anymore.
#
# f) Visual cleanliness
# - Arrow colors are unique per grid and wall colors exclude arrow colors, so arrow +
#   marker ownership is unambiguous and walls do not merge into markers.
# - Marker tails live on two diagonals behind the arrow, away from the ray path, so
#   the scene reads as "arrow + two count cues + walls + output trace" rather than a
#   tangled mass of same-color pixels.
#
# g) Color-rule inferability
# - No arbitrary recolor mapping is introduced. The periodic trail only alternates
#   between the arrow color and the most recently inherited wall color, both visible
#   in the input. Auto-turns keep the current inherited color, which makes the color
#   rule stable across turns instead of case-by-case.
#
# h) New solver assumptions and generator guarantees
# - Assumption: for each arrow, the reverse-diagonal same-color run starting two cells
#   behind the start is the extra-step marker, and the mirrored diagonal run is the
#   segment-length marker. Generator reserves exactly those cells and keeps them clear
#   of all other same-color objects.
# - Assumption: wall colors do not equal any arrow color. Generator enforces this so
#   marker counting cannot absorb unrelated wall cells.
# - Assumption: at scheduled wall-turns, at most one side-wall axis is intentionally
#   present; at scheduled auto-turns, neither side axis is occupied. Generator enforces
#   that, while the solver remains defensive on malformed inputs.
#
# i) Non-degeneracy target
# - Samples are rejected unless they contain:
#     * at least one wall-driven scheduled turn,
#     * at least one wall-free scheduled auto-turn,
#     * a visible post-wall AABB segment,
#     * a final in-bounds continuation cell left unused by the total-step stop.
#   Those constraints ensure the fused behavior differs from the original task on the
#   generated distribution instead of merely being latent in the code.


AUTO_FLIP_AXIS = "dr"


def _is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_start_missing(pts: list[tuple[int, int]]):
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
    return (int(dr), int(dc)), (int(start[0]), int(start[1])), (int(mr), int(mc))


def _arrow_points(start_r: int, start_c: int, dr: int, dc: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    miss = (int(start_r - dr), int(start_c - dc))
    pts = [
        (int(r), int(c))
        for r in (miss[0], start_r)
        for c in (miss[1], start_c)
        if (int(r), int(c)) != miss
    ]
    return pts, miss


def _tail_points(start_r: int, start_c: int, dr: int, dc: int, steps: int) -> list[tuple[int, int]]:
    return [(int(start_r - k * dr), int(start_c - k * dc)) for k in range(2, steps + 2)]


def _seg_marker_points(start_r: int, start_c: int, dr: int, dc: int, seg_len: int) -> list[tuple[int, int]]:
    return [(int(start_r - k * dr), int(start_c + k * dc)) for k in range(2, seg_len + 2)]


def _count_marker_run(
    grid: np.ndarray,
    start: tuple[int, int],
    delta: tuple[int, int],
    color: int,
) -> int:
    H, W = grid.shape
    sr, sc = start
    dr, dc = delta
    count = 0
    k = 2
    while True:
        rr = int(sr + k * dr)
        cc = int(sc + k * dc)
        if not in_bounds((H, W), rr, cc):
            break
        if int(grid[rr, cc]) != int(color):
            break
        count += 1
        k += 1
    return int(count)


def _pattern_color(start_col: int, cur_col: int, phase: int) -> int:
    return int(start_col if (phase % 4) < 2 else cur_col)


def _wall_options(
    cur_r: int,
    cur_c: int,
    dr: int,
    dc: int,
    axis: str,
    H: int,
    W: int,
) -> list[list[tuple[int, int]]]:
    opts: list[list[tuple[int, int]]] = []
    if axis == "dr":
        wr, wc = int(cur_r + dr), int(cur_c)
        if 0 <= wr < H and 0 <= wc < W:
            opts.append([(wr, wc)])
            w3 = [(int(wr), int(wc + d)) for d in (-1, 0, 1)]
            if all(0 <= cc < W for _, cc in w3):
                opts.append(w3)
    else:
        wr, wc = int(cur_r), int(cur_c + dc)
        if 0 <= wr < H and 0 <= wc < W:
            opts.append([(wr, wc)])
            w3 = [(int(wr + d), int(wc)) for d in (-1, 0, 1)]
            if all(0 <= rr < H for rr, _ in w3):
                opts.append(w3)
    _rng.shuffle(opts)
    if len(opts) == 2 and _rng.random() < 0.6:
        opts.sort(key=len, reverse=True)
    return opts


def _extract_scene(grid: np.ndarray) -> tuple[list[dict[str, object]], set[tuple[int, int]], set[tuple[int, int]], dict[tuple[int, int], int]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    arrows: list[dict[str, object]] = []
    arrow_cells: set[tuple[int, int]] = set()
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if not _is_arrow_pts(pts):
                continue
            info = _arrow_dir_start_missing(pts)
            if info is None:
                continue
            direction, start, missing = info
            arrows.append(
                {
                    "color": int(col),
                    "pts": pts,
                    "dir": direction,
                    "start": start,
                    "missing": missing,
                }
            )
            arrow_cells |= set(pts)

    marker_cells: set[tuple[int, int]] = set()
    for arrow in arrows:
        color = int(arrow["color"])
        dr, dc = arrow["dir"]  # type: ignore[assignment]
        start = arrow["start"]  # type: ignore[assignment]
        extra_steps = _count_marker_run(g, start, (-int(dr), -int(dc)), color)
        seg_len = _count_marker_run(g, start, (-int(dr), int(dc)), color)
        total_steps = int(extra_steps + seg_len)
        arrow["extra_steps"] = int(extra_steps)
        arrow["segment_len"] = int(seg_len)
        arrow["total_steps"] = int(total_steps)
        marker_cells.update(_tail_points(start[0], start[1], int(dr), int(dc), int(extra_steps)))
        marker_cells.update(_seg_marker_points(start[0], start[1], int(dr), int(dc), int(seg_len)))

    wall_color = {
        (int(r), int(c)): int(g[r, c])
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells and (int(r), int(c)) not in marker_cells
    }
    return arrows, arrow_cells, marker_cells, wall_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace rays from L-arrows with two arrow-local numeric encodings.

    Input structure:
    - each arrow is the original 3-cell L in a 2x2 box
    - reverse diagonal same-color tail length = extra steps after the first full segment
    - mirrored diagonal same-color tail length = fixed segment length
    - non-arrow, non-marker non-bg cells are walls

    Rule:
    - total ray advances = segment_len + extra_steps
    - advance diagonally and color cells with a segment-local AABB cycle
      (A = arrow color, B = current inherited color)
    - after every completed segment_len advances, force a turn:
      side walls can flip an axis and donate color; otherwise flip dr
    - stop once total_steps advances have been used, even if the next
      diagonal cell is still empty and in-bounds
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrows, arrow_cells, marker_cells, wall_color = _extract_scene(g)
    wall_cells = set(wall_color)
    out = g.copy()

    for arrow in arrows:
        start_col = int(arrow["color"])
        dr, dc = arrow["dir"]  # type: ignore[assignment]
        cur = arrow["start"]  # type: ignore[assignment]
        total_steps = int(arrow.get("total_steps", 0))
        seg_len = int(arrow.get("segment_len", 0))
        if total_steps <= 0 or seg_len <= 0:
            continue

        cur_col = int(start_col)
        steps_used = 0
        steps_in_segment = 0
        phase = 0
        seen: set[tuple[tuple[int, int], int, int, int, int, int, int]] = set()

        while steps_used < total_steps:
            state = (
                cur,
                int(dr),
                int(dc),
                int(cur_col),
                int(steps_used),
                int(steps_in_segment),
                int(phase % 4),
            )
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break
            if (nr, nc) in arrow_cells or (nr, nc) in marker_cells:
                break

            # Defensive fallback for malformed inputs: corner hits still bounce and
            # recolor but do not consume a step because no movement occurred.
            if (nr, nc) in wall_cells:
                wcol = int(wall_color[(nr, nc)])
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                phase = 0
                continue

            cur = (int(nr), int(nc))
            steps_used += 1
            steps_in_segment += 1
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = _pattern_color(int(start_col), int(cur_col), int(phase))
            phase = (int(phase) + 1) % 4

            if steps_used >= total_steps or steps_in_segment < seg_len:
                continue

            steps_in_segment = 0
            phase = 0
            turned = False

            side_r = (int(nr + dr), int(nc))
            if side_r in wall_cells:
                wcol = int(wall_color[side_r])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)
                turned = True

            side_c = (int(nr), int(nc + dc))
            if side_c in wall_cells:
                wcol = int(wall_color[side_c])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)
                turned = True

            if not turned:
                if AUTO_FLIP_AXIS == "dr":
                    dr = -int(dr)
                else:
                    dc = -int(dc)

    return out


def _trace_plan(
    H: int,
    W: int,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    start_col: int,
    seg_len: int,
    turn_types: list[str],
    final_len: int,
    wall_palette: list[int],
    blocked: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> dict[str, object] | None:
    if seg_len <= 0:
        return None

    if not (1 <= final_len < seg_len):
        return None
    total_steps = int(len(turn_types) * seg_len + final_len)

    cur_r, cur_c = int(start_r), int(start_c)
    cur_dr, cur_dc = int(dr), int(dc)
    cur_col = int(start_col)
    phase = 0

    ray_cells: list[tuple[int, int, int]] = []
    ray_set: set[tuple[int, int]] = set()
    walls: list[tuple[list[tuple[int, int]], int]] = []
    wall_cells: set[tuple[int, int]] = set()
    local_blocked = set(reserved)
    clearance: set[tuple[int, int]] = set()

    saw_wall = False
    saw_auto = False
    post_wall_visible = False

    def advance(steps: int) -> bool:
        nonlocal cur_r, cur_c, phase
        for _ in range(steps):
            nr = int(cur_r + cur_dr)
            nc = int(cur_c + cur_dc)
            pos = (int(nr), int(nc))
            if not in_bounds((H, W), nr, nc):
                return False
            if pos in blocked or pos in local_blocked or pos in ray_set:
                return False
            ray_cells.append((int(nr), int(nc), _pattern_color(int(start_col), int(cur_col), int(phase))))
            ray_set.add(pos)
            cur_r, cur_c = int(nr), int(nc)
            phase = (int(phase) + 1) % 4
        return True

    for turn_idx, turn_type in enumerate(turn_types):
        if not advance(seg_len):
            return None

        side_dr = (int(cur_r + cur_dr), int(cur_c))
        side_dc = (int(cur_r), int(cur_c + cur_dc))

        if turn_type == "auto":
            for pos in (side_dr, side_dc):
                if in_bounds((H, W), pos[0], pos[1]) and pos in blocked:
                    return None
                if in_bounds((H, W), pos[0], pos[1]) and pos in local_blocked:
                    return None
            clearance.update(pos for pos in (side_dr, side_dc) if in_bounds((H, W), pos[0], pos[1]))
            phase = 0
            saw_auto = True
            if AUTO_FLIP_AXIS == "dr":
                cur_dr = -int(cur_dr)
            else:
                cur_dc = -int(cur_dc)
            continue

        flip_axis = "dr" if turn_type == "wall_dr" else "dc"
        other_side = side_dc if flip_axis == "dr" else side_dr
        if in_bounds((H, W), other_side[0], other_side[1]) and other_side in blocked:
            return None
        if in_bounds((H, W), other_side[0], other_side[1]) and other_side in local_blocked:
            return None

        forbidden = blocked | local_blocked | ray_set
        chosen = None
        for cells in _wall_options(cur_r, cur_c, cur_dr, cur_dc, flip_axis, H, W):
            if any(cell in forbidden for cell in cells):
                continue
            if any(cell == other_side for cell in cells):
                continue
            chosen = cells
            break
        if chosen is None:
            return None

        wall_choices = [c for c in wall_palette if int(c) != int(cur_col) and int(c) != int(start_col)]
        if not wall_choices:
            wall_choices = [c for c in wall_palette if int(c) != int(cur_col)] or list(wall_palette)
        wcol = int(_rng.choice(wall_choices))
        walls.append((chosen, int(wcol)))
        wall_cells.update((int(r), int(c)) for r, c in chosen)
        local_blocked.update((int(r), int(c)) for r, c in chosen)
        if in_bounds((H, W), other_side[0], other_side[1]):
            clearance.add(other_side)

        lr, lc, _ = ray_cells[-1]
        ray_cells[-1] = (int(lr), int(lc), int(wcol))
        cur_col = int(wcol)
        phase = 0
        saw_wall = True
        remaining_after_turn = int((len(turn_types) - turn_idx - 1) * seg_len + final_len)
        if remaining_after_turn >= 4:
            post_wall_visible = True

        if flip_axis == "dr":
            cur_dr = -int(cur_dr)
        else:
            cur_dc = -int(cur_dc)

    if not advance(final_len):
        return None

    next_diag = (int(cur_r + cur_dr), int(cur_c + cur_dc))
    if not in_bounds((H, W), next_diag[0], next_diag[1]):
        return None
    if next_diag in blocked or next_diag in local_blocked or next_diag in ray_set:
        return None
    clearance.add(next_diag)

    if len(ray_cells) < seg_len + 1:
        return None

    return {
        "ray_cells": ray_cells,
        "ray_set": ray_set,
        "walls": walls,
        "wall_cells": wall_cells,
        "clearance": clearance,
        "metrics": {
            "total_steps": int(total_steps),
            "segment_len": int(seg_len),
            "final_len": int(final_len),
            "scheduled_turns": len(turn_types),
            "wall_turns": int(sum(t != "auto" for t in turn_types)),
            "auto_turns": int(sum(t == "auto" for t in turn_types)),
            "saw_wall": bool(saw_wall),
            "saw_auto": bool(saw_auto),
            "post_wall_visible": bool(post_wall_visible),
        },
    }


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(19, 20), _rng.randint(19, 20)
    n_arrows = 2
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]
    wall_palette = [c for c in palette if c not in arrow_colors]
    if len(wall_palette) < 2:
        return None

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()

    saw_auto = False
    saw_wall = False
    saw_interior_stop = False
    saw_periodic = False

    arrow_specs = [
        {"role": "wall", "seg_len_choices": [5], "turn_sampler": lambda: [_rng.choice(["wall_dr", "wall_dc"])], "final_choices": [4]},
        {"role": "auto", "seg_len_choices": [4, 5], "turn_sampler": lambda: ["auto"], "final_choices": None},
    ]

    for color, spec in zip(arrow_colors, arrow_specs):
        placed = False
        for _ in range(220):
            dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
            seg_len = int(_rng.choice(spec["seg_len_choices"]))  # type: ignore[index]
            if spec["final_choices"] is None:
                final_len = int(_rng.choice([r for r in (1, 2, 3) if r < seg_len]))
            else:
                final_len = int(_rng.choice(spec["final_choices"]))  # type: ignore[index]
            extra_steps = int(final_len)
            total_steps = int(seg_len + extra_steps)
            turn_types = spec["turn_sampler"]()  # type: ignore[operator]

            start_r = _rng.randint(0, H - 1)
            start_c = _rng.randint(0, W - 1)
            arrow_pts, _missing = _arrow_points(start_r, start_c, dr, dc)
            tail_pts = _tail_points(start_r, start_c, dr, dc, extra_steps)
            seg_pts = _seg_marker_points(start_r, start_c, dr, dc, seg_len)
            reserved = set(arrow_pts) | set(tail_pts) | set(seg_pts)

            if any(not in_bounds((H, W), r, c) or (r, c) in blocked for r, c in reserved):
                continue

            plan = _trace_plan(
                H=H,
                W=W,
                start_r=start_r,
                start_c=start_c,
                dr=dr,
                dc=dc,
                start_col=int(color),
                seg_len=int(seg_len),
                turn_types=turn_types,
                final_len=int(final_len),
                wall_palette=wall_palette,
                blocked=blocked,
                reserved=reserved,
            )
            if plan is None:
                continue

            ray_cells = plan["ray_cells"]  # type: ignore[assignment]
            ray_set = plan["ray_set"]  # type: ignore[assignment]
            walls = plan["walls"]  # type: ignore[assignment]
            wall_cells = plan["wall_cells"]  # type: ignore[assignment]
            clearance = plan["clearance"]  # type: ignore[assignment]
            metrics = plan["metrics"]  # type: ignore[assignment]
            if spec["role"] == "wall" and not bool(metrics["post_wall_visible"]):
                continue
            if spec["role"] == "wall" and int(metrics["wall_turns"]) <= 0:
                continue
            if spec["role"] == "auto" and int(metrics["auto_turns"]) <= 0:
                continue
            if spec["role"] == "auto" and int(metrics["wall_turns"]) != 0:
                continue

            new_cells = set(reserved) | set(ray_set) | set(wall_cells)
            expected = len(reserved) + len(ray_set) + sum(len(wpxs) for wpxs, _ in walls)
            if len(new_cells) != expected or new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = int(color)
                out[r, c] = int(color)
            for r, c in tail_pts:
                grid[r, c] = int(color)
                out[r, c] = int(color)
            for r, c in seg_pts:
                grid[r, c] = int(color)
                out[r, c] = int(color)
            for r, c, col in ray_cells:
                out[r, c] = int(col)
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[r, c] = int(wcol)
                    out[r, c] = int(wcol)

            occupied |= new_cells
            blocked |= new_cells | set(clearance)
            saw_wall = saw_wall or (int(metrics["wall_turns"]) > 0)
            saw_auto = saw_auto or (int(metrics["auto_turns"]) > 0)
            saw_interior_stop = saw_interior_stop or (int(metrics["final_len"]) > 0)
            saw_periodic = saw_periodic or (int(metrics["segment_len"]) >= 4)
            placed = True
            break

        if not placed:
            return None

    if not (saw_wall and saw_auto and saw_interior_stop and saw_periodic):
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D3/0 + D6/1 + D17/0 combo variant."""
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
