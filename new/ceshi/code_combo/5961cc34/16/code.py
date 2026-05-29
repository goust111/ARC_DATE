# Combo variant of 5961cc34 (index 16)
# combo_spec: D1/1+D6/3
# Name: 反射射线+目标线停止

from __future__ import annotations

import random
import numpy as np
from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste

# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# concretization:
# - Keep the original ingredients: one bottom start marker, 1-3 blue diamond or
#   octagon blobs, and green port bars on exactly one side of each blob.
# - Add one extra special color that forms a full interior row or full interior
#   column. This is the stop line and it is erased in the output except where a
#   red ray actually touches it.
# - Rays stay axis-aligned, but they no longer die at the canvas boundary.
#   Instead, when a step would leave the grid, the direction flips and the ray
#   continues along the reflected path.
#
# behavior trace:
# - Start leg: same as the original task, the thin ray rises from the yellow cap
#   and activates the first blob directly above it.
# - After activation, the new reflected rule matters on emitted thick rays. In a
#   typical sample the blob emits, reaches the side boundary, reverses, passes
#   back through already-activated red material, and then either hits a new blob
#   on the opposite side or stops on the full stop line.
# - Difference from the original: "no blob ahead" no longer means "run to the
#   border". The solver must mentally unfold a bounce and then decide whether a
#   newly approached blob or the special line is encountered first.
#
# complexity calibration:
# - The reflection choice is kept low-branching: the generator only uses
#   axis-aligned rays and at most one meaningful bounce per emitted leg. This is
#   visually obvious but does not explode into many path classes.
# - The stop marker is always a single full row or full column, so the new
#   stopping condition is easy to spot and is supported by repeated evidence
#   across examples.
# - Ports are forced to turn 90 degrees relative to the incoming hit direction,
#   which prevents messy self-overlap and makes the reflected leg stand out.
#
# old-assumption audit:
# - The original solver assumed the largest non-start color was blue. That is no
#   longer reliable because a full stop line can dominate pixel counts, so the
#   line is detected first and blue/green are inferred from adjacency afterward.
# - The original ray tracer stopped on any blue cell forever. Reflection makes
#   that too strong: a bounced ray would instantly die when it revisits its own
#   already-activated source blob. The fused rule therefore treats activated
#   blobs as red, transparent material for later rays, while still letting
#   untouched blue blobs block and activate normally.
# - Sequentially tracing width-2 rays would be wrong if lane A activates a blob
#   and lane B is then allowed to pass through it. The solver therefore traces
#   each emitted multi-lane ray against a snapshot of the activated set and only
#   applies newly hit activations after the whole emission is resolved.
#
# placement legality:
# - Chain blobs are placed output-first. The first blob is hit directly from the
#   start ray; later blobs are placed on the opposite side of the emitting blob
#   so they are reached only after a boundary reflection.
# - The stop line is chosen only after all active blobs and earlier rays are
#   fixed; its full row or column must be completely empty in the input so it
#   cannot overwrite objects or accidentally stop an earlier leg.
# - Distractors are placed last and must avoid every active ray cell and the
#   reserved stop line.
#
# visual observability / cleanliness / inferability:
# - Human-solvable cues remain structural rather than numeric: a unique full
#   line, repeated blue blobs with green emission bars, and red outputs that
#   visibly bounce off a wall before turning back.
# - Overlap is limited to "ray passing through an already-red active blob",
#   which reads cleanly because both are the same output color; unrelated ray
#   crossings are avoided by construction.
# - Only one new color role is added, so the color mapping is still easy to
#   infer from 2-4 examples.
# -----------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red activation network whose emitted rays bounce off borders and can
    terminate on a special interior line.

    Input structure:
    - bg = most common color. Blue diamond or octagon blobs carry 1-2 adjacent
      green pixels on one edge; that green side is the blob's emission port.
    - Start marker: a non-bg cap above a short red vertical segment touching the
      bottom edge.
    - One extra color forms exactly one full interior row or one full interior
      column; this is the stop line.

    Rule:
    - Fire the thin start ray upward. The first untouched blue blob it reaches
      activates and becomes red.
    - Each activated blob emits a thick ray from its port. Rays reflect when
      they hit the grid boundary, keep traveling, stop on the stop line, and
      activate any untouched blue blob they reach first.
    - Once a blob has activated, later rays may pass through its now-red body;
      continue the breadth-first cascade until no new blobs are hit.

    Key insight: the full row or column is a stopping device, while reflection
    means emitted rays can return across the canvas. Solving the task requires
    tracing the bounce path and distinguishing untouched blue blockers from
    already-activated red blobs that no longer halt later rays.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the start marker (red vertical segment with a cap above) ---
    start_candidates: list[tuple[int, int, int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=8):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            if c0 != c1 or (r1 - r0 + 1) < 2:
                continue
            if r1 != H - 1:
                continue
            if any(int(g[rr, c0]) != int(col) for rr in range(r0, r1 + 1)):
                continue
            if r0 - 1 < 0:
                continue
            cap = int(g[r0 - 1, c0])
            if cap == bg or cap == int(col):
                continue
            start_candidates.append(
                (int(col), int(cap), int(c0), [(int(rr), int(c0)) for rr in range(r0, r1 + 1)])
            )

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]

    # --- STEP 2) Detect the unique full stop line ---
    stop_candidates: list[tuple[int, np.ndarray]] = []
    for col in colors_present(g, exclude={bg, red_col, yellow_col}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        rows = sorted(set(int(r) for r in pts[:, 0]))
        cols = sorted(set(int(c) for c in pts[:, 1]))
        if len(rows) == 1 and len(cols) == W and len(pts) == W:
            mask = np.zeros((H, W), dtype=bool)
            mask[int(rows[0]), :] = True
            stop_candidates.append((int(col), mask))
        elif len(cols) == 1 and len(rows) == H and len(pts) == H:
            mask = np.zeros((H, W), dtype=bool)
            mask[:, int(cols[0])] = True
            stop_candidates.append((int(col), mask))

    assert len(stop_candidates) == 1
    line_col, line_mask = stop_candidates[0]

    # --- STEP 3) Infer blue and green after excluding start colors and stop line ---
    remaining = colors_present(g, exclude={bg, red_col, yellow_col, line_col})
    assert len(remaining) == 2

    def every_pixel_touches(src_col: int, dst_col: int) -> bool:
        pts = np.argwhere(g == int(src_col))
        assert pts.size > 0
        for r, c in pts:
            r = int(r)
            c = int(c)
            ok = False
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(dst_col):
                    ok = True
                    break
            if not ok:
                return False
        return True

    green_candidates = [
        int(col)
        for col in remaining
        if every_pixel_touches(int(col), int(remaining[0] if col == remaining[1] else remaining[1]))
    ]
    assert len(green_candidates) == 1
    green_col = int(green_candidates[0])
    blue_col = int(remaining[0] if green_col == int(remaining[1]) else remaining[1])

    out = np.full_like(g, int(bg))

    # --- STEP 4) Extract blue blobs and build a pixel -> blob map ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = len(blobs)
        for r, c in pts:
            blob_id[int(r), int(c)] = int(bi)
        blobs.append({"pts": pts, "ports": []})

    # --- STEP 5) Read each blob's green port side ---
    for blob in blobs:
        pts = blob["pts"]
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in pts:
            for ddr, ddc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + ddr)
                cc = int(c + ddc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(green_col):
                    seen_dirs.add((int(ddr), int(ddc)))
        for ddr, ddc in seen_dirs:
            port_pixels = [
                (int(r), int(c))
                for (r, c) in pts
                if 0 <= int(r + ddr) < H
                and 0 <= int(c + ddc) < W
                and int(g[int(r + ddr), int(c + ddc)]) == int(green_col)
            ]
            assert port_pixels
            blob["ports"].append({"dir": (int(ddr), int(ddc)), "pixels": port_pixels})

    activated: set[int] = set()
    queue: list[int] = []

    def activate_blob(bi: int) -> None:
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(red_col)

    def trace_emission(
        starts: list[tuple[int, int]],
        d: tuple[int, int],
        *,
        include_origin: bool,
    ) -> None:
        active_snapshot = set(activated)
        newly_hit: set[int] = set()
        dr0, dc0 = map(int, d)

        for sr, sc in starts:
            r, c = int(sr), int(sc)
            dr, dc = int(dr0), int(dc0)
            if include_origin:
                out[int(r), int(c)] = int(red_col)
            seen: set[tuple[int, int, int, int]] = set()

            while True:
                state = (int(r), int(c), int(dr), int(dc))
                if state in seen:
                    break
                seen.add(state)

                nr = int(r + dr)
                nc = int(c + dc)
                if not (0 <= nr < H and 0 <= nc < W):
                    dr = -int(dr)
                    dc = -int(dc)
                    continue

                r, c = int(nr), int(nc)
                out[int(r), int(c)] = int(red_col)

                if bool(line_mask[int(r), int(c)]):
                    break

                bi = int(blob_id[int(r), int(c)])
                if bi >= 0 and bi not in active_snapshot:
                    newly_hit.add(int(bi))
                    break

        for bi in sorted(newly_hit):
            activate_blob(int(bi))

    # --- STEP 6) Paint the start marker and fire the initial upward ray ---
    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[int(top_r) - 1, int(start_col)]) == int(yellow_col)
    trace_emission([(int(top_r) - 1, int(start_col))], (-1, 0), include_origin=True)

    # --- STEP 7) BFS cascade: each newly activated blob emits from its green port ---
    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            trace_emission(port["pixels"], port["dir"], include_origin=False)

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    return out


# --------------- generator (reflected cascade + interior stop line) ----------

# --- Colors ---
BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
LINE = 6

# --- Directions ---
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
PERP = {
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
    RIGHT: (UP, DOWN),
}

# --- Diamond/octagon shapes (h, w) ---
SHAPES = ((3, 4), (4, 4), (5, 4), (5, 6), (6, 6), (8, 8))
_SHAPE_WIDTHS = {
    (3, 4): (2, 4, 2),
    (4, 4): (2, 4, 4, 2),
    (5, 4): (2, 4, 4, 4, 2),
    (5, 6): (2, 4, 6, 4, 2),
    (6, 6): (2, 4, 6, 6, 4, 2),
    (8, 8): (2, 4, 6, 8, 8, 6, 4, 2),
}

# --- Canvas bounds ---
H_MIN, H_MAX = 15, 27
W_MIN, W_MAX = 17, 27
START_COL_PAD = 2

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3

# --- Chain / distractor ---
CHAIN_CHOICES = [1, 2]
# Chain length 2 is harder to place than length 1, so bias generation toward it
# to keep row-stop and col-stop samples both common in the final dataset.
CHAIN_WEIGHTS = [1, 4]
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3
MIN_STOP_GAP = 3
STOP_EDGE_PAD = 1

# --- Retry ---
GEN_RETRIES = 500


def _shape_mask(h: int, w: int) -> np.ndarray:
    widths = _SHAPE_WIDTHS[(h, w)]
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(widths):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return rc[1] if d[0] else rc[0]


def _along_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return rc[0] if d[0] else rc[1]


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    return sorted(coords, key=lambda p: _perp_value(p, d))


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _blob_spec(size: tuple[int, int], port_dir: tuple[int, int]) -> dict | None:
    h, w = size
    mask = _shape_mask(h, w)
    green = _edge_offsets(mask, port_dir)
    if len(green) not in (1, 2):
        return None

    green_set = set(green)
    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    for r, c in green:
        patch[r, c] = GREEN

    port_pixels: list[tuple[int, int]] = []
    for gr, gc in green:
        br = int(gr - port_dir[0])
        bc = int(gc - port_dir[1])
        if not (0 <= br < h and 0 <= bc < w):
            return None
        if not mask[br, bc] or (br, bc) in green_set:
            return None
        port_pixels.append((int(br), int(bc)))

    blue = [(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) not in green_set]
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "size": (h, w),
        "mask": mask,
        "patch": patch,
        "blue": blue,
        "full": full,
        "port_dir": port_dir,
        "port_pixels": _sort_on_ray(port_pixels, port_dir),
    }


def _all_specs() -> list[dict]:
    out = [s for size in SHAPES for d in DIRS if (s := _blob_spec(size, d)) is not None]
    random.shuffle(out)
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= int(r) < H and 0 <= int(c) < W for r, c in coords)


def _path_to_entries(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    entries: list[tuple[int, int]],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        if include_origin:
            cells.append((int(r), int(c)))
        guard = H + W + 10
        while (int(r), int(c)) != (int(er), int(ec)):
            r = int(r + dr)
            c = int(c + dc)
            guard -= 1
            if guard < 0 or not (0 <= int(r) < H and 0 <= int(c) < W):
                return None
            cells.append((int(r), int(c)))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


def _try_place_along_axis(
    spec: dict,
    window: list[tuple[int, int]],
    d: tuple[int, int],
    lane_vals: list[int],
    starts: list[tuple[int, int]],
    include_origin: bool,
    min_gap: int,
    occ_pad: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    H: int,
    W: int,
) -> list[dict]:
    h, w = spec["mask"].shape
    win_vals = [_perp_value(p, d) for p in window]
    edge_line = _along_value(window[0], d)
    start_line = _along_value(starts[0], d)
    is_vertical = bool(d[0])

    fixed = int(lane_vals[0] - win_vals[0])
    extent = H - h if is_vertical else W - w
    low = int(edge_line)
    high = int(extent + edge_line)
    if d in (UP, LEFT):
        high = min(high, int(start_line - min_gap))
    else:
        low = max(low, int(start_line + min_gap))

    entry_lines = list(range(low, high + 1))
    random.shuffle(entry_lines)
    candidates: list[dict] = []

    for entry_line in entry_lines:
        origin_along = int(entry_line - edge_line)
        r0 = int(origin_along if is_vertical else fixed)
        c0 = int(fixed if is_vertical else origin_along)
        if is_vertical:
            entry_cells = [(int(entry_line), int(c0 + c)) for _, c in window]
        else:
            entry_cells = [(int(r0 + r), int(entry_line)) for r, _ in window]
        full = _offset(spec["full"], r0, c0)
        if not _in_bounds(full, H, W):
            continue
        if any(bool(occ_pad[r, c]) or bool(ray_mask[r, c]) for r, c in full):
            continue
        path = _path_to_entries(starts, d, entry_cells, include_origin, H, W)
        if path is None:
            continue
        entry_set = set(entry_cells)
        if any(bool(blue_mask[r, c]) for r, c in path if (int(r), int(c)) not in entry_set):
            continue
        candidates.append(
            {
                "spec": spec,
                "r0": r0,
                "c0": c0,
                "entry": entry_cells,
                "path": path,
                "blue": _offset(spec["blue"], r0, c0),
                "full": full,
                "ports": _sort_on_ray(_offset(spec["port_pixels"], r0, c0), spec["port_dir"]),
            }
        )

    return candidates


def _aligned_blob_candidates(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    travel_dir: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    *,
    min_gap: int,
    allowed_ports: tuple[tuple[int, int], ...],
) -> list[dict]:
    starts = _sort_on_ray(starts, travel_dir)
    thickness = len(starts)
    lane_vals = [_perp_value(p, travel_dir) for p in starts]
    lane_delta = [int(v - lane_vals[0]) for v in lane_vals]
    entry_side = OPP[travel_dir]
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []

    for spec in _all_specs():
        if spec["port_dir"] not in allowed_ports:
            continue
        if spec["port_dir"] == entry_side:
            continue
        edge = _edge_offsets(spec["mask"], entry_side)
        if len(edge) < thickness:
            continue
        for k in range(len(edge) - thickness + 1):
            window = edge[k : k + thickness]
            win_delta = [_perp_value(p, travel_dir) - _perp_value(window[0], travel_dir) for p in window]
            if [int(v) for v in win_delta] != lane_delta:
                continue
            candidates.extend(
                _try_place_along_axis(
                    spec,
                    window,
                    travel_dir,
                    lane_vals,
                    starts,
                    include_origin,
                    min_gap,
                    occ_pad,
                    blue_mask,
                    ray_mask,
                    H,
                    W,
                )
            )

    return candidates


def _trace_planned_ray(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    *,
    include_origin: bool,
    H: int,
    W: int,
    passable_mask: np.ndarray,
    target_mask: np.ndarray | None = None,
    line_mask: np.ndarray | None = None,
) -> dict | None:
    dr0, dc0 = map(int, d)
    cells: list[tuple[int, int]] = []
    stop_cells: list[tuple[int, int]] = []
    reasons: list[str] = []
    bounces: list[int] = []

    target_mask = np.zeros((H, W), dtype=bool) if target_mask is None else target_mask
    line_mask = np.zeros((H, W), dtype=bool) if line_mask is None else line_mask

    for sr, sc in starts:
        r, c = int(sr), int(sc)
        dr, dc = int(dr0), int(dc0)
        lane_bounces = 0
        if include_origin:
            cells.append((int(r), int(c)))
        seen: set[tuple[int, int, int, int]] = set()

        while True:
            state = (int(r), int(c), int(dr), int(dc))
            if state in seen:
                return None
            seen.add(state)

            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                dr = -int(dr)
                dc = -int(dc)
                lane_bounces += 1
                continue

            r, c = int(nr), int(nc)
            cells.append((int(r), int(c)))

            if bool(line_mask[int(r), int(c)]):
                stop_cells.append((int(r), int(c)))
                reasons.append("line")
                bounces.append(int(lane_bounces))
                break

            if bool(target_mask[int(r), int(c)]):
                stop_cells.append((int(r), int(c)))
                reasons.append("target")
                bounces.append(int(lane_bounces))
                break

            if bool(passable_mask[int(r), int(c)]):
                continue

    return {
        "cells": cells,
        "stop_cells": stop_cells,
        "reasons": reasons,
        "bounces": bounces,
    }


def _candidate_reflected_blob(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    emit_dir: tuple[int, int],
    occupied: np.ndarray,
    active_blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> dict | None:
    travel_dir = OPP[emit_dir]
    allowed_ports = PERP[travel_dir]
    placements = _aligned_blob_candidates(
        H,
        W,
        starts,
        travel_dir,
        False,
        occupied,
        np.zeros_like(active_blue_mask, dtype=bool),
        np.zeros_like(ray_mask, dtype=bool),
        min_gap=MIN_GAP_CHAIN,
        allowed_ports=allowed_ports,
    )
    random.shuffle(placements)

    for cand in placements:
        target_mask = np.zeros((H, W), dtype=bool)
        paint_coords(target_mask, cand["blue"], True)
        trace = _trace_planned_ray(
            starts,
            emit_dir,
            include_origin=False,
            H=H,
            W=W,
            passable_mask=active_blue_mask,
            target_mask=target_mask,
        )
        if trace is None:
            continue
        if trace["reasons"] != ["target"] * len(starts):
            continue
        if any(int(b) != 1 for b in trace["bounces"]):
            continue
        entry_set = set((int(r), int(c)) for r, c in cand["entry"])
        if any((int(r), int(c)) not in entry_set for r, c in trace["stop_cells"]):
            continue
        cell_set = {(int(r), int(c)) for r, c in trace["cells"]}
        if any(bool(ray_mask[r, c]) and not bool(active_blue_mask[r, c]) for r, c in cell_set):
            continue
        cand = dict(cand)
        cand["path"] = [(int(r), int(c)) for r, c in trace["cells"]]
        cand["bounces"] = trace["bounces"]
        return cand

    return None


def _candidate_reflected_stop_line(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    emit_dir: tuple[int, int],
    occupied: np.ndarray,
    active_blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> dict | None:
    travel_dir = OPP[emit_dir]
    starts = _sort_on_ray(starts, travel_dir)
    candidates: list[dict] = []

    if travel_dir in (UP, DOWN):
        start_line = int(starts[0][0])
        if travel_dir == UP:
            line_indices = list(range(STOP_EDGE_PAD, start_line - MIN_STOP_GAP + 1))
        else:
            line_indices = list(range(start_line + MIN_STOP_GAP, H - STOP_EDGE_PAD))
        random.shuffle(line_indices)
        for line_idx in line_indices:
            if occupied[int(line_idx), :].any() or ray_mask[int(line_idx), :].any():
                continue
            line_mask = np.zeros((H, W), dtype=bool)
            line_mask[int(line_idx), :] = True
            trace = _trace_planned_ray(
                starts,
                emit_dir,
                include_origin=False,
                H=H,
                W=W,
                passable_mask=active_blue_mask,
                line_mask=line_mask,
            )
            if trace is None:
                continue
            if trace["reasons"] != ["line"] * len(starts):
                continue
            if any(int(b) != 1 for b in trace["bounces"]):
                continue
            cell_set = {(int(r), int(c)) for r, c in trace["cells"]}
            if any(bool(ray_mask[r, c]) and not bool(active_blue_mask[r, c]) for r, c in cell_set):
                continue
            candidates.append(
                {
                    "axis": "row",
                    "index": int(line_idx),
                    "cells": [(int(line_idx), int(c)) for c in range(W)],
                    "path": [(int(r), int(c)) for r, c in trace["cells"]],
                    "bounces": trace["bounces"],
                }
            )
    else:
        start_line = int(starts[0][1])
        if travel_dir == LEFT:
            line_indices = list(range(STOP_EDGE_PAD, start_line - MIN_STOP_GAP + 1))
        else:
            line_indices = list(range(start_line + MIN_STOP_GAP, W - STOP_EDGE_PAD))
        random.shuffle(line_indices)
        for line_idx in line_indices:
            if occupied[:, int(line_idx)].any() or ray_mask[:, int(line_idx)].any():
                continue
            line_mask = np.zeros((H, W), dtype=bool)
            line_mask[:, int(line_idx)] = True
            trace = _trace_planned_ray(
                starts,
                emit_dir,
                include_origin=False,
                H=H,
                W=W,
                passable_mask=active_blue_mask,
                line_mask=line_mask,
            )
            if trace is None:
                continue
            if trace["reasons"] != ["line"] * len(starts):
                continue
            if any(int(b) != 1 for b in trace["bounces"]):
                continue
            cell_set = {(int(r), int(c)) for r, c in trace["cells"]}
            if any(bool(ray_mask[r, c]) and not bool(active_blue_mask[r, c]) for r, c in cell_set):
                continue
            candidates.append(
                {
                    "axis": "col",
                    "index": int(line_idx),
                    "cells": [(int(r), int(line_idx)) for r in range(H)],
                    "path": [(int(r), int(c)) for r, c in trace["cells"]],
                    "bounces": trace["bounces"],
                }
            )

    return random.choice(candidates) if candidates else None


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[int(r), int(c)] = True
    for r, c in blob["blue"]:
        blue_mask[int(r), int(c)] = True


def _place_stop_line(inp: np.ndarray, occupied: np.ndarray, stop: dict) -> None:
    paint_coords(inp, stop["cells"], LINE)
    paint_coords(occupied, stop["cells"], True)


def _place_distractor(
    inp: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> bool:
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []
    for spec in _all_specs():
        h, w = spec["mask"].shape
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                full = _offset(spec["full"], r0, c0)
                if any(bool(occ_pad[r, c]) or bool(ray_mask[r, c]) for r, c in full):
                    continue
                candidates.append(
                    {
                        "spec": spec,
                        "r0": int(r0),
                        "c0": int(c0),
                        "full": full,
                        "blue": _offset(spec["blue"], r0, c0),
                    }
                )
    if not candidates:
        return False
    _place_blob(inp, occupied, blue_mask, random.choice(candidates))
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    active_blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    # Place start segment (yellow cap + red column at bottom).
    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = int(H - start_h - 1)
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    inp[int(cap_r), int(start_col)] = YELLOW
    out[int(cap_r), int(start_col)] = RED
    ray_mask[int(cap_r), int(start_col)] = True
    occupied[int(cap_r), int(start_col)] = True
    for r in range(int(cap_r) + 1, H):
        inp[int(r), int(start_col)] = RED
        out[int(r), int(start_col)] = RED
        occupied[int(r), int(start_col)] = True
        ray_mask[int(r), int(start_col)] = True

    chain_len = random.choices(CHAIN_CHOICES, weights=CHAIN_WEIGHTS)[0]
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(0, max_dist)

    # First blob: direct upward hit from the start.
    first_candidates = _aligned_blob_candidates(
        H,
        W,
        [(int(cap_r), int(start_col))],
        UP,
        True,
        occupied,
        blue_mask,
        ray_mask,
        min_gap=MIN_GAP_FIRST,
        allowed_ports=PERP[UP],
    )
    if not first_candidates:
        return None
    first_blob = random.choice(first_candidates)
    _mark_ray(out, ray_mask, first_blob["path"])
    _place_blob(inp, occupied, blue_mask, first_blob)
    paint_coords(out, first_blob["blue"], RED)
    paint_coords(active_blue_mask, first_blob["blue"], True)
    starts = first_blob["ports"]
    emit_dir = first_blob["spec"]["port_dir"]

    # Reflected chain legs.
    for _ in range(chain_len - 1):
        blob = _candidate_reflected_blob(H, W, starts, emit_dir, occupied, active_blue_mask, ray_mask)
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["blue"], RED)
        paint_coords(active_blue_mask, blob["blue"], True)
        starts = blob["ports"]
        emit_dir = blob["spec"]["port_dir"]

    # Terminal reflected leg to the stop line.
    stop = _candidate_reflected_stop_line(H, W, starts, emit_dir, occupied, active_blue_mask, ray_mask)
    if stop is None:
        return None
    _place_stop_line(inp, occupied, stop)
    _mark_ray(out, ray_mask, stop["path"])

    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
