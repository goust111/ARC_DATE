# Variant of 142ca369
# Dimension: D6 — 射线终止条件 (dst)
# Change: 射线离开网格边界时终止（out of grid）... → 固定步数终止
#
# Design analysis for this D6 variant
# ===================================
# a) Concrete behavior
# - Keep the original objects: 3-pixel L-shaped arrows and colored wall pixels/segments.
# - Add one new visible cue per arrow: a same-color diagonal "tail" behind the arrow, on the
#   reverse firing diagonal. Tail length = total number of ray advances.
# - The ray still starts from the right-angle arrow vertex, still moves diagonally, still
#   side-bounces off walls, and still inherits wall colors. The only semantic change is the
#   stopping rule: after exactly N advances, where N is the tail length, the ray stops even if
#   more empty space remains.
# - Small trace example:
#   original: arrow fires, advances until leaving the grid;
#   variant: arrow with 5 tail pixels fires, advances exactly 5 cells total, may bounce during
#   those 5 steps, then terminates in the interior with the next diagonal cell still available.
#
# b) Complexity calibration
# - Using a nearby counted structure is more observable than encoding distance directly by color.
# - The tail is diagonal, not 4-connected to the arrow, so arrow detection stays stable.
# - Step counts are limited to a small visible band (4-8) to keep counting human-tractable while
#   still producing clearly shorter-than-boundary traces.
#
# c) Old assumptions that break under D6
# - The original `while True` relied on "eventually exits grid" as the primary valid stop case.
#   That is no longer true; valid outputs now often stop mid-grid.
# - The original `seen` set would wrongly truncate legitimate revisits if a finite-step path
#   loops back to a prior state after making progress. Here the state includes `steps_used`.
# - The original wall set was "all non-bg non-arrow cells". That would misclassify the new tail
#   markers as walls, so markers must be excluded explicitly.
#
# d) Placement legality
# - Generator reserves space behind each arrow for the full tail.
# - Rays, walls, arrows, tails, and terminal-next-cell clearance are all checked for overlap.
# - Final ray endpoints are required to have one more empty in-bounds diagonal continuation cell,
#   so the fixed-step stop is not accidentally equivalent to boundary termination.
#
# e) Visual observability
# - Across examples, longer tails produce longer outputs; shorter tails produce shorter outputs.
# - Because rays often stop in the interior while the next diagonal cell is empty, the new stop
#   rule is directly visible without inspecting code.
#
# f) Visual cleanliness
# - Tails sit behind arrows and never overlap walls or rays.
# - Arrow colors are unique per grid, so each tail-arrow pairing is visually unambiguous.
#
# g) Color inference
# - No new abstract color mapping is introduced. Tail color matches its arrow color.
# - Wall recoloring behavior from the original task is preserved unchanged.
#
# h) Generator non-degeneracy target
# - Generated examples must make the old and new stop rules diverge: terminal cells stay away
#   from the border and have at least one unused forward diagonal step available.
#
# i) New solver assumptions and how the generator guarantees them
# - Assumption: each arrow has a consecutive same-color reverse-diagonal tail. Generator always
#   places one and keeps arrow colors unique.
# - Assumption: wall colors do not reuse arrow colors. Generator enforces this so tail counting
#   cannot absorb unrelated same-color wall pixels.
# - Assumption: markers are not on the traced ray path. Generator reserves those cells up front.

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, in_bounds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-shaped arrows, but stop after a tail-encoded step budget.

    Input structure:
    - bg grid with monochrome 3-pixel L-shaped arrows and colored wall pixels/segments.
    - Each arrow also has a same-color reverse-diagonal tail behind its missing corner.

    Rule:
    - Tail length = total number of ray advances for that arrow.
    - Ray direction, side-touch reflections, corner reflections, and wall-color inheritance
      remain the same as in the original task.
    - The ray stops after exactly that many advances, usually before reaching the boundary.
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

    def arrow_dir_start_missing(pts: list[tuple[int, int]]):
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

    arrows: list[dict[str, object]] = []
    arrow_cells: set[tuple[int, int]] = set()
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if not is_arrow_pts(pts):
                continue
            info = arrow_dir_start_missing(pts)
            if info is None:
                continue
            (dr, dc), start, missing = info
            arrows.append(
                {
                    "color": int(col),
                    "pts": pts,
                    "dir": (int(dr), int(dc)),
                    "start": start,
                    "missing": missing,
                }
            )
            arrow_cells |= set(pts)

    marker_cells: set[tuple[int, int]] = set()
    for arrow in arrows:
        color = int(arrow["color"])
        dr, dc = arrow["dir"]  # type: ignore[assignment]
        sr, sc = arrow["start"]  # type: ignore[assignment]
        steps = 0
        k = 2
        while True:
            rr = int(sr - k * dr)
            cc = int(sc - k * dc)
            if not in_bounds((H, W), rr, cc):
                break
            if int(g[rr, cc]) != color:
                break
            marker_cells.add((rr, cc))
            steps += 1
            k += 1
        arrow["steps"] = int(steps)

    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells and (int(r), int(c)) not in marker_cells
    }

    out = g.copy()

    for arrow in arrows:
        start_col = int(arrow["color"])
        dr, dc = arrow["dir"]  # type: ignore[assignment]
        cur = arrow["start"]  # type: ignore[assignment]
        step_limit = int(arrow.get("steps", 0))
        if step_limit <= 0:
            continue

        cur_col = int(start_col)
        steps_used = 0
        seen: set[tuple[tuple[int, int], int, int, int, int]] = set()

        while steps_used < step_limit:
            state = (cur, int(dr), int(dc), int(cur_col), int(steps_used))
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
                continue

            if (nr, nc) in arrow_cells or (nr, nc) in marker_cells:
                break

            cur = (int(nr), int(nc))
            steps_used += 1
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

    return out


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


def _sample_segment_lengths(total_steps: int, n_bounces: int) -> list[int] | None:
    parts = n_bounces + 1
    base = [2] * parts
    remaining = total_steps - 2 * parts
    if remaining < 0:
        return None
    for _ in range(remaining):
        base[_rng.randrange(parts)] += 1
    _rng.shuffle(base)
    return base


def _trace_plan(
    H: int,
    W: int,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    start_col: int,
    segment_lengths: list[int],
    wall_palette: list[int],
    blocked: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> dict[str, object] | None:
    cur_r, cur_c = int(start_r), int(start_c)
    cur_dr, cur_dc = int(dr), int(dc)
    cur_col = int(start_col)

    ray_cells: list[tuple[int, int, int]] = []
    ray_set: set[tuple[int, int]] = set()
    walls: list[tuple[list[tuple[int, int]], int]] = []
    wall_cells: set[tuple[int, int]] = set()
    local_blocked = set(reserved)

    for seg_idx, seg_len in enumerate(segment_lengths):
        for _ in range(seg_len):
            nr = int(cur_r + cur_dr)
            nc = int(cur_c + cur_dc)
            if not in_bounds((H, W), nr, nc):
                return None
            if (nr, nc) in blocked or (nr, nc) in local_blocked or (nr, nc) in ray_set:
                return None
            ray_cells.append((nr, nc, cur_col))
            ray_set.add((nr, nc))
            cur_r, cur_c = nr, nc

        if seg_idx == len(segment_lengths) - 1:
            break

        options: list[tuple[list[tuple[int, int]], str]] = []
        wr, wc = int(cur_r + cur_dr), int(cur_c)
        if 0 <= wr < H:
            options.append(([(wr, wc)], "dr"))
            w3 = [(int(wr), int(wc + d)) for d in (-1, 0, 1)]
            if all(0 <= cc < W for _, cc in w3):
                options.append((w3, "dr"))

        wr2, wc2 = int(cur_r), int(cur_c + cur_dc)
        if 0 <= wc2 < W:
            options.append(([(wr2, wc2)], "dc"))
            w3v = [(int(wr2 + d), int(wc2)) for d in (-1, 0, 1)]
            if all(0 <= rr < H for rr, _ in w3v):
                options.append((w3v, "dc"))

        if not options:
            return None

        _rng.shuffle(options)
        options.sort(key=lambda item: len(item[0]), reverse=_rng.random() < 0.6)
        chosen = None
        for wpxs, flip in options:
            if any((r, c) in blocked or (r, c) in local_blocked or (r, c) in ray_set for r, c in wpxs):
                continue
            chosen = (wpxs, flip)
            break
        if chosen is None:
            return None

        wpxs, flip = chosen
        w_choices = [c for c in wall_palette if int(c) != int(cur_col)] or wall_palette
        wcol = int(_rng.choice(w_choices))
        walls.append((wpxs, wcol))
        wall_cells.update((int(r), int(c)) for r, c in wpxs)
        local_blocked.update((int(r), int(c)) for r, c in wpxs)

        lr, lc, _ = ray_cells[-1]
        ray_cells[-1] = (int(lr), int(lc), int(wcol))
        cur_col = int(wcol)
        if flip == "dr":
            cur_dr = -int(cur_dr)
        else:
            cur_dc = -int(cur_dc)

    next_diag = (int(cur_r + cur_dr), int(cur_c + cur_dc))
    if not in_bounds((H, W), next_diag[0], next_diag[1]):
        return None
    if next_diag in blocked or next_diag in local_blocked or next_diag in ray_set:
        return None

    side_checks = [(int(cur_r + cur_dr), int(cur_c)), (int(cur_r), int(cur_c + cur_dc))]
    for pos in side_checks:
        if pos in blocked or pos in wall_cells:
            return None

    clearance = {next_diag, *side_checks}

    return {
        "ray_cells": ray_cells,
        "ray_set": ray_set,
        "walls": walls,
        "wall_cells": wall_cells,
        "clearance": clearance,
        "end": (int(cur_r), int(cur_c)),
        "next": next_diag,
        "final_dir": (int(cur_dr), int(cur_dc)),
    }


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(2, 4)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]
    wall_palette = [c for c in palette if c not in arrow_colors]
    if not wall_palette:
        return None

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()

    for color in arrow_colors:
        placed = False
        for _ in range(120):
            dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
            steps = _rng.choices([4, 5, 6, 7, 8], weights=[1, 2, 3, 2, 1])[0]
            max_bounces = min(3, steps // 2 - 1)
            if max_bounces < 0:
                continue
            bounce_choices = list(range(max_bounces + 1))
            bounce_weights = [4, 3, 2, 1][: len(bounce_choices)]
            n_bounces = _rng.choices(bounce_choices, weights=bounce_weights)[0]
            segment_lengths = _sample_segment_lengths(int(steps), int(n_bounces))
            if segment_lengths is None:
                continue

            r_lo = steps + 1 if dr == 1 else 0
            r_hi = H - steps - 2 if dr == -1 else H - 1
            c_lo = steps + 1 if dc == 1 else 0
            c_hi = W - steps - 2 if dc == -1 else W - 1
            if r_lo > r_hi or c_lo > c_hi:
                continue

            start_r = _rng.randint(r_lo, r_hi)
            start_c = _rng.randint(c_lo, c_hi)
            arrow_pts, _missing = _arrow_points(start_r, start_c, dr, dc)
            tail_pts = _tail_points(start_r, start_c, dr, dc, int(steps))

            reserved = set(arrow_pts) | set(tail_pts)
            if any(
                not in_bounds((H, W), r, c) or (r, c) in blocked
                for r, c in reserved
            ):
                continue

            plan = _trace_plan(
                H=H,
                W=W,
                start_r=start_r,
                start_c=start_c,
                dr=dr,
                dc=dc,
                start_col=int(color),
                segment_lengths=segment_lengths,
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

            new_cells = set(reserved) | set(ray_set) | set(wall_cells)
            expected_count = len(reserved) + len(ray_cells) + sum(len(wpxs) for wpxs, _ in walls)
            if len(new_cells) != expected_count or new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = int(color)
                out[r, c] = int(color)
            for r, c in tail_pts:
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
            placed = True
            break

        if not placed:
            return None

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D6 fixed-step variant of 142ca369."""
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
