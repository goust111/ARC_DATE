# Variant of 142ca369
# Dimension: D8 — 多步处理与射线顺序 (multi_step)
# Change: multi_bounce——每条射线独立地通过所有可达墙壁逐个反射，多条射线之间无顺序依赖... → 链式反应传播
#
# Design analysis
# ---------------
# variant_spec:
#   keep_from_original:
#     - grid size stays 18-20 by 18-20 with bg 0
#     - the only arrow shape is still a 3-pixel L inside a 2x2 box
#     - rays still move on diagonals
#     - walls are still monochrome size-1 or size-3 axis-aligned segments
#     - wall color is still inherited by the propagation after a hit
#   core_change:
#     - original: each arrow owns one ray that keeps bouncing through multiple walls
#     - variant: only one initial arrow is generated; when a ray touches a wall, that
#       ray stops there and the hit wall becomes the new source
#     - for a size-3 wall, all three wall pixels emit parallel child rays in the
#       reflected direction, so the propagation visibly fans out
#   concrete_pixel_behavior:
#     - the arrow still fires from its right-angle vertex away from the missing corner
#     - a corner hit into a wall cell spawns a child source from that wall with
#       direction (-dr, -dc)
#     - a side-touch hit from ray cell (r,c) into a wall adjacent on the row side
#       flips dr and activates that wall as the new source
#     - a side-touch hit into a wall adjacent on the column side flips dc and
#       activates that wall as the new source
#     - on a hit, the last painted bounce cell takes the wall's color, matching the
#       original "adopt wall color at the interaction point" cue
#   small_example:
#     - original flow: arrow ray walks NE, touches a horizontal wall, flips vertical
#       direction, continues itself, later touches another wall, and so on
#     - variant flow: arrow ray walks NE, touches the same horizontal wall, stops,
#       recolors the bounce cell, and the wall's pixels now emit new rays NW
#     - difference: later painted cells no longer belong to one continuously bounced
#       ray; they belong to a sequence of newly activated sources
#   complexity_calibration:
#     - generator always uses exactly one initial arrow to make the "chain reaction"
#       interpretation unambiguous
#     - at least one activated wall is size 3, so the new source is visibly a wall
#       segment rather than a hidden scalar state
#     - chain length is 2-4 wall activations: enough for multi-step reasoning, small
#       enough to keep the board legible
#   implicit_assumption_audit_for_main:
#     - original loop assumed every arrow can be solved independently; that no longer
#       holds, so main now uses a source queue
#     - original code could mutate (dr,dc,cur_col) in-place for one ray; that no
#       longer matches the variant because a hit terminates the current ray and
#       creates fresh child sources instead of continuing the same one
#     - original wall_cells was only a blocker set; now each wall component must keep
#       its full pixel list because the whole wall acts as the spawned source
#     - original overlap safety relied on independent arrows and never had to guard
#       against processing the same wall repeatedly; now main keeps an activated set
#       keyed by wall id and direction
#   placement_legality:
#     - generator plans sources sequentially along one continuing chain, but every
#       activated size-3 wall emits parallel side rays that simply exit the board
#     - after a ray is committed, generator records "future wall forbidden" cells
#       beside that ray, so no later wall can retroactively create an earlier hit
#     - rays, arrows, and walls never share cells; walls are also kept away from
#       previously reserved side-touch cells
#   visual_observability:
#     - the input contains one arrow but multiple isolated walls; the output shows
#       several separately launched ray families, often a 3-ray fan after a size-3
#       wall activates
#     - this is visually distinct from the original evaluation behavior where each
#       arrow contributes one thin zigzag trajectory
#   new_main_assumptions_checked_against_generator:
#     - every non-arrow component is a legal wall segment of size 1 or 3
#     - repeated activation of the same wall-direction pair is not needed; generator
#       avoids cycles and main deduplicates them anyway
#     - final boards avoid ray-ray overlaps, so out only paints bg cells and never
#       needs conflict resolution

from __future__ import annotations

from collections import deque
import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg


def _is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _is_wall_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) == 1:
        return True
    if len(pts) != 3:
        return False
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    if len(set(rs)) == 1 and max(cs) - min(cs) == 2 and len(set(cs)) == 3:
        return True
    if len(set(cs)) == 1 and max(rs) - min(rs) == 2 and len(set(rs)) == 3:
        return True
    return False


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Chain-reaction version of 142ca369.

    Only the initial arrow is a native source. Every time a ray touches a wall,
    that ray terminates and the wall component becomes a new source that emits
    child rays in the reflected direction. Size-3 walls therefore launch three
    parallel child rays, making the multi-step propagation visible.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[dict] = []
    walls: list[dict] = []

    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append({"color": int(col), "pts": pts})
                arrow_cells |= set(pts)
            elif _is_wall_pts(pts):
                walls.append({"color": int(col), "pts": sorted(pts)})

    cell_to_wall: dict[tuple[int, int], int] = {}
    for wid, wall in enumerate(walls):
        for p in wall["pts"]:
            cell_to_wall[p] = wid

    out = g.copy()
    queue: deque[tuple[tuple[str, int], list[tuple[int, int]], int, int, int]] = deque()
    activated: set[tuple[str, int, int, int]] = set()

    def enqueue_source(kind: str, idx: int, pixels: list[tuple[int, int]], dr: int, dc: int, color: int) -> None:
        state = (kind, int(idx), int(dr), int(dc))
        if state in activated:
            return
        activated.add(state)
        queue.append(((kind, int(idx)), list(pixels), int(dr), int(dc), int(color)))

    for ai, arrow in enumerate(arrows):
        info = _arrow_dir_and_start(arrow["pts"])
        if info is None:
            continue
        (dr, dc), start = info
        enqueue_source("arrow", ai, [start], dr, dc, int(arrow["color"]))

    def paint_if_bg(r: int, c: int, color: int) -> None:
        if int(out[r, c]) == bg:
            out[r, c] = int(color)

    def trace_pixel_ray(start: tuple[int, int], dr: int, dc: int, color: int) -> None:
        r, c = int(start[0]), int(start[1])
        seen: set[tuple[int, int, int, int]] = set()
        while True:
            state = (r, c, int(dr), int(dc))
            if state in seen:
                break
            seen.add(state)

            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in arrow_cells:
                break

            if (nr, nc) in cell_to_wall:
                wid = int(cell_to_wall[(nr, nc)])
                wcol = int(walls[wid]["color"])
                if int(g[r, c]) == bg:
                    out[r, c] = wcol
                enqueue_source("wall", wid, walls[wid]["pts"], -dr, -dc, wcol)
                break

            r, c = int(nr), int(nc)
            paint_if_bg(r, c, color)

            hits: list[tuple[int, tuple[int, int]]] = []
            row_side = (r + dr, c)
            col_side = (r, c + dc)
            if row_side in cell_to_wall:
                hits.append((int(cell_to_wall[row_side]), (-dr, dc)))
            if col_side in cell_to_wall:
                wid2 = int(cell_to_wall[col_side])
                new_state = (wid2, (dr, -dc))
                if all((wid != wid2 or nd != (dr, -dc)) for wid, nd in hits):
                    hits.append(new_state)

            if hits:
                first_wcol = int(walls[hits[0][0]]["color"])
                if int(g[r, c]) == bg:
                    out[r, c] = first_wcol
                for wid, (ndr, ndc) in hits:
                    enqueue_source("wall", wid, walls[wid]["pts"], ndr, ndc, int(walls[wid]["color"]))
                break

    while queue:
        (_kind, _idx), pixels, dr, dc, color = queue.popleft()
        for start in pixels:
            trace_pixel_ray(start, dr, dc, color)

    return out


def _in_bounds(H: int, W: int, r: int, c: int) -> bool:
    return 0 <= int(r) < H and 0 <= int(c) < W


def _wall_shape(center: tuple[int, int], axis: str, size: int) -> list[tuple[int, int]]:
    r, c = center
    if size == 1:
        return [(int(r), int(c))]
    if axis == "dr":
        return [(int(r), int(c - 1)), (int(r), int(c)), (int(r), int(c + 1))]
    return [(int(r - 1), int(c)), (int(r), int(c)), (int(r + 1), int(c))]


def _max_steps_to_edge(H: int, W: int, start: tuple[int, int], dr: int, dc: int) -> int:
    r, c = start
    row_steps = (H - 1 - r) if dr == 1 else r
    col_steps = (W - 1 - c) if dc == 1 else c
    return int(min(row_steps, col_steps))


def _trace_exit_candidate(
    H: int,
    W: int,
    start: tuple[int, int],
    dr: int,
    dc: int,
    color: int,
    occupied_input: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    ray_colors: dict[tuple[int, int], int],
) -> list[tuple[int, int]] | None:
    r, c = start
    path: list[tuple[int, int]] = []
    local: dict[tuple[int, int], int] = {}
    while True:
        nr, nc = r + dr, c + dc
        if not _in_bounds(H, W, nr, nc):
            return path
        if (nr, nc) in occupied_input:
            return None
        prev = local.get((nr, nc), ray_colors.get((nr, nc)))
        if prev is not None and int(prev) != int(color):
            return None
        if (nr + dr, nc) in wall_cells or (nr, nc + dc) in wall_cells:
            return None
        path.append((int(nr), int(nc)))
        local[(int(nr), int(nc))] = int(color)
        r, c = int(nr), int(nc)


def _trace_hit_candidate(
    H: int,
    W: int,
    start: tuple[int, int],
    dr: int,
    dc: int,
    path_color: int,
    hit_color: int,
    seg_len: int,
    axis: str,
    occupied_input: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    ray_colors: dict[tuple[int, int], int],
    new_wall_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    r, c = start
    path: list[tuple[int, int]] = []
    local: dict[tuple[int, int], int] = {}
    for step in range(1, int(seg_len) + 1):
        nr, nc = r + dr, c + dc
        if not _in_bounds(H, W, nr, nc):
            return None
        if (nr, nc) in occupied_input:
            return None
        expected = int(hit_color if step == int(seg_len) else path_color)
        prev = local.get((nr, nc), ray_colors.get((nr, nc)))
        if prev is not None and int(prev) != expected:
            return None

        row_side = (nr + dr, nc)
        col_side = (nr, nc + dc)
        intended = row_side if axis == "dr" else col_side
        intended_hits = [p for p in (row_side, col_side) if p in new_wall_cells]
        if step < seg_len:
            if (row_side in wall_cells) or (col_side in wall_cells) or intended_hits:
                return None
        else:
            if intended not in new_wall_cells:
                return None
            unexpected_existing = (row_side in wall_cells) or (col_side in wall_cells)
            unexpected_new = any(p in new_wall_cells and p != intended for p in (row_side, col_side))
            if unexpected_existing or unexpected_new:
                return None

        path.append((int(nr), int(nc)))
        local[(int(nr), int(nc))] = expected
        r, c = int(nr), int(nc)
    return path


def _reserve_future_wall_cells(
    H: int,
    W: int,
    path: list[tuple[int, int]],
    dr: int,
    dc: int,
    allow: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]]:
    allow = allow or set()
    reserved: set[tuple[int, int]] = set()
    for r, c in path:
        for p in ((r + dr, c), (r, c + dc)):
            if p in allow:
                continue
            if _in_bounds(H, W, p[0], p[1]):
                reserved.add((int(p[0]), int(p[1])))
    return reserved


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_color, wall1_color, wall2_color = map(int, palette[:3])

    center_r, center_c = H // 2, W // 2
    start_offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)]
    _rng.shuffle(start_offsets)
    starts = []
    for ro, co in start_offsets:
        sr, sc = center_r + ro, center_c + co
        if 3 <= sr <= H - 4 and 3 <= sc <= W - 4:
            starts.append((int(sr), int(sc)))
    if not starts:
        return None

    dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    _rng.shuffle(dirs)
    axis1_order = ["dr", "dc"]
    axis2_order = ["dr", "dc"]
    len1_order = [3, 4, 5, 6]
    len2_order = [3, 4, 5, 6]
    _rng.shuffle(axis1_order)
    _rng.shuffle(axis2_order)
    _rng.shuffle(len1_order)
    _rng.shuffle(len2_order)

    for sr, sc in starts:
        for dr, dc in dirs:
            mr, mc = sr - dr, sc - dc
            arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
            if any(not _in_bounds(H, W, r, c) for r, c in arrow_pts):
                continue

            for axis1 in axis1_order:
                for len1 in len1_order:
                    bounce1 = (sr + dr * len1, sc + dc * len1)
                    center1 = (bounce1[0] + dr, bounce1[1]) if axis1 == "dr" else (bounce1[0], bounce1[1] + dc)
                    wall1 = _wall_shape(center1, axis1, 3)
                    if any(not _in_bounds(H, W, r, c) for r, c in wall1):
                        continue

                    child_dir = (-dr, dc) if axis1 == "dr" else (dr, -dc)
                    source2 = sorted(wall1)[1]

                    for axis2 in axis2_order:
                        for len2 in len2_order:
                            bounce2 = (source2[0] + child_dir[0] * len2, source2[1] + child_dir[1] * len2)
                            center2 = (
                                (bounce2[0] + child_dir[0], bounce2[1])
                                if axis2 == "dr"
                                else (bounce2[0], bounce2[1] + child_dir[1])
                            )
                            wall2 = _wall_shape(center2, axis2, 1)
                            if any(not _in_bounds(H, W, r, c) for r, c in wall2):
                                continue

                            cells = set(arrow_pts) | set(wall1) | set(wall2)
                            if len(cells) != len(arrow_pts) + len(wall1) + len(wall2):
                                continue

                            g = np.zeros((H, W), dtype=int)
                            for r, c in arrow_pts:
                                g[r, c] = arrow_color
                            for r, c in wall1:
                                g[r, c] = wall1_color
                            for r, c in wall2:
                                g[r, c] = wall2_color

                            out = main(g)
                            arrow_extra = int(((out == arrow_color) & (g != arrow_color)).sum())
                            wall1_extra = int(((out == wall1_color) & (g != wall1_color)).sum())
                            wall2_extra = int(((out == wall2_color) & (g != wall2_color)).sum())
                            total_new = int((out != g).sum())

                            if arrow_extra < 2:
                                continue
                            if wall1_extra < 4:
                                continue
                            if wall2_extra < 2:
                                continue
                            if total_new < 12:
                                continue
                            return g

    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
