# Variant of 5961cc34
# Dimension: D6 — 色块停止条件与终点 (dst)
# Change: 射线遇到色块时停止并激活该色块；若未遇到色块则延伸到网格边界... → 射线到达目标线停止

from __future__ import annotations

import random
import numpy as np
from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste

# -----------------------------------------------------------------------------
# Design analysis for the D6 variant
#
# concretization:
# - Input keeps the original start marker and blue+green port blobs.
# - A new special stop line is added: exactly one 1-cell-thick full row or full
#   column in a distinct color (LINE). It is always strictly inside the grid.
# - Rays still activate the first blue blob they hit. The D6 change is only for
#   rays that would otherwise continue to the boundary: they now stop on the
#   stop line instead. The hit cell(s) on that line become red in the output,
#   while the untouched part of the line disappears to bg.
#
# behavior trace:
# - Original final leg: start/blob port emits a ray, no blob ahead, so the ray
#   runs until it exits the canvas.
# - Variant final leg: same ray direction and thickness, but a full interior row
#   or column lies ahead. The ray paints up to that row/column and stops there.
# - Difference: blob-hit activation is unchanged; only the terminal destination
#   for "no more blobs ahead" changes from edge to line.
#
# complexity calibration:
# - The line is structurally obvious: one full row/column of one color.
# - It is not allowed on the outer border, so samples visibly differ from the
#   original boundary-reaching behavior.
# - Orientation is not fixed globally: vertical final rays use horizontal stop
#   lines; horizontal final rays use vertical stop lines. This keeps 2 cases,
#   enough variation without making the task ambiguous.
#
# old-assumption audit:
# - Original main() assumed blue was the largest remaining color. That breaks
#   once a full-width/full-height line is present, so blue/green are now inferred
#   from adjacency structure after extracting the stop line first.
# - Original ray tracing only stopped at blobs or out-of-bounds. That would run
#   past the new marker, so every ray extension now checks the stop line before
#   falling through to boundary.
# - Original generator sent the terminal ray to the border. That path builder is
#   replaced with explicit interior-line placement and a path that ends on it.
#
# placement legality:
# - The stop line may not overlap any blob, the start marker, or any earlier ray;
#   otherwise it would either overwrite input objects or prematurely stop a
#   non-terminal ray. The generator searches only fully clear rows/columns.
# - Distractors are placed after the line is reserved, so they cannot overwrite
#   it and cannot sit on any active ray.
#
# visual observability / cleanliness:
# - The stop line is a single strong cue rather than many small markers.
# - Only one such line exists, and it is always actually used by the red network.
# - Because the untouched line vanishes in the output, examples clearly show that
#   it functions as a destination/barrier rather than as an object to preserve.
# -----------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red ray network whose terminal leg stops on a special interior line.

    Input structure:
    - bg = most common color. Large blue blobs have adjacent green pixels that
      mark each blob's emission side.
    - Start marker: a non-bg cap above a short red vertical segment at the
      bottom edge.
    - One extra color forms a full horizontal or vertical line inside the grid;
      this is the destination line.

    Rule:
    - Launch the thin start ray upward. If a ray hits a blue blob first, that
      blob activates, turns red, and later emits thick rays through its green
      port side.
    - Any ray that reaches the special line before another blob stops there.
    - Continue the blob-activation cascade breadth-first until no new blobs are
      reached.

    Key insight: the new full row/column is not a drawable object to keep; it is
    a structurally unique stopping marker that replaces the old "run to edge"
    terminal behavior while leaving blob-triggered chaining intact.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the start "port" (red vertical segment with a non-red cap above) ---
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
            if r1 != int(H - 1):
                continue
            if any(int(g[rr, c0]) != int(col) for rr in range(int(r0), int(r1) + 1)):
                continue
            if int(r0) - 1 < 0:
                continue
            cap = int(g[int(r0) - 1, int(c0)])
            if cap == int(bg) or cap == int(col):
                continue

            start_candidates.append(
                (
                    int(col),
                    int(cap),
                    int(c0),
                    [(int(rr), int(c0)) for rr in range(int(r0), int(r1) + 1)],
                )
            )

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]

    # --- STEP 2) Detect the unique stop line and its color ---
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

    # --- STEP 3) Identify blob/body colors after excluding the stop line ---
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

    # --- STEP 4) Extract blue blobs as connected components, and build a pixel->blob map ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []

    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[int(r), int(c)] = bi
        blobs.append({"pts": pts, "ports": []})

    # --- STEP 5) For each blob, find its ports from adjacent green markers ---
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

    def activate_blob_at(r: int, c: int) -> None:
        """Recolor a blob to red when first hit by a ray; enqueue it to emit its ports."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(red_col)

    # --- STEP 6) Draw the start ray upward (thin) and stop at blob or stop line ---
    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    top_r = min(int(r) for r, _c in start_seg)
    assert int(top_r) > 0 and int(g[int(top_r) - 1, int(start_col)]) == int(yellow_col)
    out[int(top_r) - 1, int(start_col)] = int(red_col)
    for r in range(int(top_r) - 1, -1, -1):
        out[int(r), int(start_col)] = int(red_col)
        if bool(line_mask[int(r), int(start_col)]):
            break
        if int(g[int(r), int(start_col)]) == int(blue_col):
            activate_blob_at(int(r), int(start_col))
            break

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        """Extend a ray from each port pixel until it hits a blob or the stop line."""
        ddr, ddc = map(int, d)
        for pr, pc in port_pixels:
            r, c = int(pr), int(pc)
            while True:
                r += int(ddr)
                c += int(ddc)
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[int(r), int(c)] = int(red_col)
                if bool(line_mask[int(r), int(c)]):
                    break
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c))
                    break

    # --- STEP 7) Propagate (BFS): activated blob -> emit rays -> activate more blobs ---
    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"

    return out


# --------------- generator (a2_cascade + interior stop line) ---------------

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
H_MIN, H_MAX = 13, 27
W_MIN, W_MAX = 15, 27
START_COL_PAD = 2

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3

# --- Chain / distractor ---
CHAIN_LEN_WEIGHTS = [1, 3, 3, 3]
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3
MIN_STOP_GAP = 3
STOP_EDGE_PAD = 1

# --- Retry ---
GEN_RETRIES = 400


def _shape_mask(h: int, w: int) -> np.ndarray:
    """Build a boolean diamond/octagon mask for the given (h, w) shape."""
    widths = _SHAPE_WIDTHS[(h, w)]
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(widths):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    """Coordinate perpendicular to direction d (column for vertical, row for horizontal)."""
    return rc[1] if d[0] else rc[0]


def _along_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    """Coordinate along direction d (row for vertical, column for horizontal)."""
    return rc[0] if d[0] else rc[1]


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    """Sort coordinates by their perpendicular value relative to direction d."""
    return sorted(coords, key=lambda p: _perp_value(p, d))


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    """Return sorted mask-local coordinates on the edge facing direction d."""
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _blob_spec(size: tuple[int, int], port_dir: tuple[int, int]) -> dict | None:
    """Build a blob specification dict for a given shape and port direction, or None if invalid."""
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
        br, bc = gr - port_dir[0], gc - port_dir[1]
        if not (0 <= br < h and 0 <= bc < w):
            return None
        if not mask[br, bc] or (br, bc) in green_set:
            return None
        port_pixels.append((br, bc))

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


def _all_specs(port_dir: tuple[int, int] | None = None) -> list[dict]:
    """Generate all valid blob specs, optionally filtered by port direction, in shuffled order."""
    dirs = [port_dir] if port_dir is not None else list(DIRS)
    out = [s for size in SHAPES for d in dirs if (s := _blob_spec(size, d)) is not None]
    random.shuffle(out)
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    """Dilate a boolean mask by 1 pixel in all 8 directions."""
    H, W = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    """Translate local offsets to global coordinates."""
    return [(r0 + r, c0 + c) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in coords)


def _path_to_entries(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    entries: list[tuple[int, int]],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    """Trace ray lanes from starts to entries; return cell list or None on failure."""
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        if include_origin:
            cells.append((r, c))
        guard = H + W + 10
        while (r, c) != (er, ec):
            r, c = r + dr, c + dc
            guard -= 1
            if guard < 0 or not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((r, c))
    return cells


def _path_to_stop_line(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    line_idx: int,
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    """Trace ray lanes from starts until they reach the chosen stop line."""
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        if include_origin:
            cells.append((r, c))
        guard = H + W + 10
        while True:
            r, c = r + dr, c + dc
            guard -= 1
            if guard < 0 or not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((r, c))
            if (dr and r == int(line_idx)) or (dc and c == int(line_idx)):
                break
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    """Paint cells red in the output grid and mark them in the ray mask."""
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
    """Find all valid placements for a blob spec along the ray's travel axis."""
    h, w = spec["mask"].shape
    win_vals = [_perp_value(p, d) for p in window]
    edge_line = _along_value(window[0], d)
    start_line = _along_value(starts[0], d)
    is_vertical = bool(d[0])

    fixed = lane_vals[0] - win_vals[0]
    extent = H - h if is_vertical else W - w
    low = edge_line
    high = extent + edge_line
    if d in (UP, LEFT):
        high = min(high, start_line - min_gap)
    else:
        low = max(low, start_line + min_gap)

    entry_lines = list(range(low, high + 1))
    random.shuffle(entry_lines)
    candidates: list[dict] = []

    for entry_line in entry_lines:
        origin_along = entry_line - edge_line
        r0 = origin_along if is_vertical else fixed
        c0 = fixed if is_vertical else origin_along
        if is_vertical:
            entry_cells = [(entry_line, c0 + c) for _, c in window]
        else:
            entry_cells = [(r0 + r, entry_line) for r, _ in window]
        full = _offset(spec["full"], r0, c0)
        if not _in_bounds(full, H, W):
            continue
        if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
            continue
        path = _path_to_entries(starts, d, entry_cells, include_origin, H, W)
        if path is None:
            continue
        entry_set = set(entry_cells)
        if any(blue_mask[r, c] for r, c in path if (r, c) not in entry_set):
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


def _candidate_next_blob(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> dict | None:
    """Find a random valid placement for the next blob in the cascade chain."""
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    candidates: list[dict] = []

    for spec in _all_specs():
        if spec["port_dir"] == entry_side:
            continue
        edge = _edge_offsets(spec["mask"], entry_side)
        if len(edge) < thickness:
            continue
        for k in range(len(edge) - thickness + 1):
            window = edge[k : k + thickness]
            win_delta = [_perp_value(p, d) - _perp_value(window[0], d) for p in window]
            if win_delta != lane_delta:
                continue
            candidates.extend(
                _try_place_along_axis(
                    spec,
                    window,
                    d,
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

    return random.choice(candidates) if candidates else None


def _candidate_stop_line(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> dict | None:
    """Find a clear interior line that the terminal ray can reach without hitting a blob."""
    starts = _sort_on_ray(starts, d)
    candidates: list[dict] = []

    if d in (UP, DOWN):
        start_line = int(starts[0][0])
        if d == UP:
            line_indices = list(range(STOP_EDGE_PAD, start_line - MIN_STOP_GAP + 1))
        else:
            line_indices = list(range(start_line + MIN_STOP_GAP, H - STOP_EDGE_PAD))
        random.shuffle(line_indices)
        for line_idx in line_indices:
            if occupied[int(line_idx), :].any() or ray_mask[int(line_idx), :].any():
                continue
            path = _path_to_stop_line(starts, d, int(line_idx), include_origin, H, W)
            if path is None:
                continue
            if any(blue_mask[r, c] for r, c in path):
                continue
            candidates.append(
                {
                    "axis": "row",
                    "index": int(line_idx),
                    "cells": [(int(line_idx), int(c)) for c in range(W)],
                    "path": path,
                }
            )
    else:
        start_line = int(starts[0][1])
        if d == LEFT:
            line_indices = list(range(STOP_EDGE_PAD, start_line - MIN_STOP_GAP + 1))
        else:
            line_indices = list(range(start_line + MIN_STOP_GAP, W - STOP_EDGE_PAD))
        random.shuffle(line_indices)
        for line_idx in line_indices:
            if occupied[:, int(line_idx)].any() or ray_mask[:, int(line_idx)].any():
                continue
            path = _path_to_stop_line(starts, d, int(line_idx), include_origin, H, W)
            if path is None:
                continue
            if any(blue_mask[r, c] for r, c in path):
                continue
            candidates.append(
                {
                    "axis": "col",
                    "index": int(line_idx),
                    "cells": [(int(r), int(line_idx)) for r in range(H)],
                    "path": path,
                }
            )

    return random.choice(candidates) if candidates else None


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    """Paste a blob's patch onto the input grid and update tracking masks."""
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[r, c] = True
    for r, c in blob["blue"]:
        blue_mask[r, c] = True


def _place_stop_line(inp: np.ndarray, occupied: np.ndarray, stop: dict) -> None:
    """Paint the destination line into the input and reserve its cells."""
    paint_coords(inp, stop["cells"], LINE)
    paint_coords(occupied, stop["cells"], True)


def _place_distractor(
    inp: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> bool:
    """Try to place a single distractor blob in free space. Return success."""
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []
    for spec in _all_specs():
        h, w = spec["mask"].shape
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                full = _offset(spec["full"], r0, c0)
                if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
                    continue
                candidates.append(
                    {
                        "spec": spec,
                        "r0": r0,
                        "c0": c0,
                        "full": full,
                        "blue": _offset(spec["blue"], r0, c0),
                    }
                )
    if not candidates:
        return False
    _place_blob(inp, occupied, blue_mask, random.choice(candidates))
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on placement failure."""
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    # Place start segment (yellow cap + red column at bottom)
    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = H - start_h - 1
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    inp[cap_r, start_col] = YELLOW
    out[cap_r, start_col] = RED
    ray_mask[cap_r, start_col] = True
    occupied[cap_r, start_col] = True
    for r in range(cap_r + 1, H):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        occupied[r, start_col] = True

    # Decide chain length and distractor count
    chain_len = random.choices(range(len(CHAIN_LEN_WEIGHTS)), weights=CHAIN_LEN_WEIGHTS)[0]
    min_dist = 1 if chain_len == 0 else 0
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(min_dist, max_dist)

    # Build cascade chain up to the final emitting blob (or the start cap).
    starts = [(cap_r, start_col)]
    direction = UP
    include_origin = True
    for _ in range(chain_len):
        blob = _candidate_next_blob(H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask)
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["blue"], RED)
        starts = blob["ports"]
        direction = blob["spec"]["port_dir"]
        include_origin = False

    # Reserve an interior stop line for the terminal leg, then draw up to it.
    stop = _candidate_stop_line(H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask)
    if stop is None:
        return None
    _place_stop_line(inp, occupied, stop)
    _mark_ray(out, ray_mask, stop["path"])

    # Place distractors after the line is fixed.
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
    raise RuntimeError("failed to generate a valid 5961cc34 D6 input")
