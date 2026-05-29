# Variant of 5961cc34
# Dimension: D12 — 起始射线方向 (path)
# Change: 起始射线固定从底部向上发射... → 任意边缘起始

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste

# Design analysis for the D12 variant
# a) Concrete behavior:
#    The original start marker is a bottom-anchored red vertical segment with a yellow cap
#    above it, so the first ray always goes upward. In this variant the same two-part marker
#    can appear on any edge:
#    - bottom edge: red segment below yellow cap, ray goes up
#    - top edge: red segment above yellow cap, ray goes down
#    - left edge: red segment left of yellow cap, ray goes right
#    - right edge: red segment right of yellow cap, ray goes left
#    The blob cascade itself is unchanged. Example trace: if the marker is on the left edge as
#    `224` along one row, the old solver would miss it because it only scans for a bottom
#    vertical segment and only draws a vertical ray. The new solver must instead read the cap
#    position and launch rightward from that cap.
# b) Complexity calibration:
#    Only four start directions are allowed, matching the four grid edges. The start marker
#    keeps the same colors and short length, so the change is visible without introducing extra
#    object types or hidden state. Uniform sampling across the four inward directions ensures
#    the variation appears repeatedly in generated examples.
# c) Implicit assumptions audited from the original main():
#    - start detection assumed a vertical component (`c0 == c1`)
#    - it assumed that component touched the bottom (`r1 == H - 1`)
#    - it assumed the cap sat one row above the segment
#    - it stored only `start_col`, enough only for upward scanning
#    - it hard-coded the start ray loop as `for r in range(top_r - 1, -1, -1)`
#    Those assumptions break once the marker may live on any edge. The new code replaces them
#    with a boundary-segment detector that returns segment cells, cap cell, and inward
#    direction. All later logic consumes that generic start description.
# d) Placement legality:
#    The generator still works output-first. It now chooses a random inward start direction,
#    places the red segment flush with the matching edge, puts the yellow cap one cell inward,
#    and then reuses the existing chain placer with `starts=[cap]` and `direction=start_dir`.
#    The same occupancy, dilation, and ray-intersection checks keep blobs separated and stop
#    distractors from landing on active rays.
# e) Visual observability:
#    Humans can see the rule because the start marker visibly rotates with the edge while the
#    output always extends red away from that cap into the grid. Across a few examples, the
#    only changing control variable is which edge launches the first ray.
# f) Visual cleanliness:
#    The marker still sits a couple of cells away from corners, preserving a clear silhouette.
#    Rays and blobs use the original sparse layout rules, so the grid does not become crowded.
# g) Color inference:
#    No color remapping is introduced. Blue bodies, green ports, yellow cap, and red activated
#    paths retain their original semantic roles.
# h) New solver assumptions and generator guarantees:
#    - among non-background colors, blue remains the dominant body color by count
#    - green remains the unique non-start color 4-adjacent to blue
#    - the two remaining colors belong to the start marker, with exactly one straight boundary
#      segment of length 2-3 and a cap immediately inward from its inner end
#    `generate_input()` enforces all three properties.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a red activation network that begins from whichever edge carries the start marker.

    Input structure:
    - The background is the dominant color. Blue diamond/octagon blobs may have green pixels on one edge to mark emission ports.
    - A short red segment touches exactly one border, with a yellow cap immediately inward from its inner tip.

    Rule:
    - Determine which border holds the start marker, then fire a thin ray from the yellow cap into the grid.
    - The first blue blob hit turns red and emits thick rays from its green-marked edge pixels; any newly hit blob does the same.
    - Continue the breadth-first cascade until every reachable blob has fired, leaving unreached blobs erased to background.

    Key insight:
    - The start marker's orientation, not a fixed row/column convention, determines the initial ray direction; after that, green adjacency still encodes each blob's outgoing ray direction and width.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Identify semantic colors before looking for the start marker ---
    counts = Counter(map(int, g.flatten()))
    non_bg = [int(col) for col in colors_present(g, exclude={bg})]
    assert non_bg

    max_count = max(int(counts[col]) for col in non_bg)
    blue_candidates = [int(col) for col in non_bg if int(counts[col]) == max_count]
    assert len(blue_candidates) == 1
    blue_col = int(blue_candidates[0])

    green_candidates: list[int] = []
    for col in colors_present(g, exclude={bg, blue_col}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        ok = False
        for r, c in pts:
            rr = int(r)
            cc = int(c)
            for dr, dc in DIRS:
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < H and 0 <= nc < W and int(g[nr, nc]) == int(blue_col):
                    ok = True
                    break
            if ok:
                break
        if ok:
            green_candidates.append(int(col))

    assert len(green_candidates) == 1
    green_col = int(green_candidates[0])

    start_colors = colors_present(g, exclude={bg, blue_col, green_col})
    assert len(start_colors) == 2

    # --- STEP 2) Detect the boundary start segment and infer its inward direction ---
    start_candidates: list[dict] = []

    for seg_col in start_colors:
        cap_col = int(start_colors[0] if seg_col == start_colors[1] else start_colors[1])
        for coords in components_mask(g == int(seg_col), connectivity=8):
            if coords.size == 0:
                continue
            pts = [(int(r), int(c)) for r, c in coords]
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

            direction: tuple[int, int] | None = None
            cap: tuple[int, int] | None = None
            seg_cells: list[tuple[int, int]] | None = None

            if c0 == c1 and len(pts) == (r1 - r0 + 1) and (r1 - r0 + 1) >= 2:
                if r1 == H - 1 and r0 > 0:
                    direction = UP
                    cap = (int(r0 - 1), int(c0))
                    seg_cells = [(int(r), int(c0)) for r in range(int(r0), int(r1) + 1)]
                elif r0 == 0 and r1 < H - 1:
                    direction = DOWN
                    cap = (int(r1 + 1), int(c0))
                    seg_cells = [(int(r), int(c0)) for r in range(int(r0), int(r1) + 1)]
            elif r0 == r1 and len(pts) == (c1 - c0 + 1) and (c1 - c0 + 1) >= 2:
                if c1 == W - 1 and c0 > 0:
                    direction = LEFT
                    cap = (int(r0), int(c0 - 1))
                    seg_cells = [(int(r0), int(c)) for c in range(int(c0), int(c1) + 1)]
                elif c0 == 0 and c1 < W - 1:
                    direction = RIGHT
                    cap = (int(r0), int(c1 + 1))
                    seg_cells = [(int(r0), int(c)) for c in range(int(c0), int(c1) + 1)]

            if direction is None or cap is None or seg_cells is None:
                continue
            cr, cc = cap
            if int(g[cr, cc]) != int(cap_col):
                continue
            start_candidates.append(
                {
                    "red": int(seg_col),
                    "yellow": int(cap_col),
                    "cap": (int(cr), int(cc)),
                    "direction": (int(direction[0]), int(direction[1])),
                    "segment": seg_cells,
                }
            )

    assert len(start_candidates) == 1
    red_col = int(start_candidates[0]["red"])
    yellow_col = int(start_candidates[0]["yellow"])
    start_cap = tuple(start_candidates[0]["cap"])
    start_dir = tuple(start_candidates[0]["direction"])
    start_seg = list(start_candidates[0]["segment"])
    assert int(g[start_cap]) == int(yellow_col)

    out = np.full_like(g, int(bg))

    # --- STEP 3) Extract blue blobs as connected components, and build a pixel->blob map ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []

    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[r, c] = bi
        blobs.append({"pts": pts, "ports": []})

    # --- STEP 4) For each blob, find its ports from adjacent green markers ---
    for blob in blobs:
        pts = blob["pts"]
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in pts:
            for ddr, ddc in DIRS:
                rr = int(r + ddr)
                cc = int(c + ddc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(green_col):
                    seen_dirs.add((int(ddr), int(ddc)))
        for ddr, ddc in seen_dirs:
            port_pixels = [
                (int(r), int(c))
                for (r, c) in pts
                if 0 <= int(r + ddr) < H and 0 <= int(c + ddc) < W and int(g[int(r + ddr), int(c + ddc)]) == int(green_col)
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

    # --- STEP 5) Draw the initial thin ray from the detected edge marker ---
    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    cr, cc = map(int, start_cap)
    dr, dc = map(int, start_dir)
    while 0 <= cr < H and 0 <= cc < W:
        out[int(cr), int(cc)] = int(red_col)
        if int(g[int(cr), int(cc)]) == int(blue_col):
            activate_blob_at(int(cr), int(cc))
            break
        cr += int(dr)
        cc += int(dc)

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        """Extend a ray from each port pixel until boundary or blob hit."""
        ddr, ddc = map(int, d)
        for pr, pc in port_pixels:
            r, c = int(pr), int(pc)
            while True:
                r += int(ddr)
                c += int(ddc)
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[int(r), int(c)] = int(red_col)
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c))
                    break

    # --- STEP 6) Propagate (BFS): activated blob -> emit rays -> activate more blobs ---
    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"

    return out


# --------------- generator (a2_cascade, D12 variant) ---------------

# --- Colors ---
BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

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
START_EDGE_PAD = 2  # keep the marker away from corners so its orientation is clear

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3

# --- Chain / distractor ---
CHAIN_LEN_WEIGHTS = [1, 3, 3, 3]
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 300


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
    return [(int(r0 + r), int(c0 + c)) for r, c in offsets]


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
            cells.append((int(r), int(c)))
        guard = H + W + 10
        rr, cc = int(r), int(c)
        while (rr, cc) != (int(er), int(ec)):
            rr += int(dr)
            cc += int(dc)
            guard -= 1
            if guard < 0 or not (0 <= rr < H and 0 <= cc < W):
                return None
            cells.append((int(rr), int(cc)))
    return cells


def _path_to_boundary(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """Trace ray lanes from starts to grid boundary."""
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        rr, cc = int(r), int(c)
        if include_origin:
            cells.append((int(rr), int(cc)))
        while True:
            rr += int(dr)
            cc += int(dc)
            if not (0 <= rr < H and 0 <= cc < W):
                break
            cells.append((int(rr), int(cc)))
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

    fixed = int(lane_vals[0] - win_vals[0])
    extent = H - h if is_vertical else W - w
    low = int(edge_line)
    high = int(extent + edge_line)
    if d in (UP, LEFT):
        high = min(high, int(start_line - min_gap))
    else:
        low = max(low, int(start_line + min_gap))

    entry_lines = list(range(int(low), int(high) + 1))
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
        if any(bool(occ_pad[r, c] or ray_mask[r, c]) for r, c in full):
            continue
        path = _path_to_entries(starts, d, entry_cells, include_origin, H, W)
        if path is None:
            continue
        entry_set = set(entry_cells)
        if any(bool(blue_mask[r, c]) for r, c in path if (r, c) not in entry_set):
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
    lane_delta = [int(v - lane_vals[0]) for v in lane_vals]
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
            win_delta = [int(_perp_value(p, d) - _perp_value(window[0], d)) for p in window]
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


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    """Paste a blob's patch onto the input grid and update tracking masks."""
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[r, c] = True
    for r, c in blob["blue"]:
        blue_mask[r, c] = True


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
                if any(bool(occ_pad[r, c] or ray_mask[r, c]) for r, c in full):
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


def _start_layout(H: int, W: int, start_h: int, start_dir: tuple[int, int]) -> dict:
    """Build the edge marker geometry for a chosen inward direction."""
    if start_dir == UP:
        lane = random.randint(START_EDGE_PAD, W - START_EDGE_PAD - 1)
        cap = (int(H - start_h - 1), int(lane))
        segment = [(int(r), int(lane)) for r in range(int(cap[0] + 1), H)]
    elif start_dir == DOWN:
        lane = random.randint(START_EDGE_PAD, W - START_EDGE_PAD - 1)
        cap = (int(start_h), int(lane))
        segment = [(int(r), int(lane)) for r in range(0, int(cap[0]))]
    elif start_dir == LEFT:
        lane = random.randint(START_EDGE_PAD, H - START_EDGE_PAD - 1)
        cap = (int(lane), int(W - start_h - 1))
        segment = [(int(lane), int(c)) for c in range(int(cap[1] + 1), W)]
    else:
        lane = random.randint(START_EDGE_PAD, H - START_EDGE_PAD - 1)
        cap = (int(lane), int(start_h))
        segment = [(int(lane), int(c)) for c in range(0, int(cap[1]))]
    return {"cap": cap, "segment": segment, "direction": start_dir}


def _place_start(
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    ray_mask: np.ndarray,
    start_h: int,
    start_dir: tuple[int, int],
) -> dict:
    """Draw the start marker into the input/output canvases and occupancy masks."""
    H, W = inp.shape
    layout = _start_layout(H, W, start_h, start_dir)
    cap_r, cap_c = map(int, layout["cap"])
    inp[cap_r, cap_c] = YELLOW
    out[cap_r, cap_c] = RED
    ray_mask[cap_r, cap_c] = True
    occupied[cap_r, cap_c] = True
    for r, c in layout["segment"]:
        inp[int(r), int(c)] = RED
        out[int(r), int(c)] = RED
        occupied[int(r), int(c)] = True
    return layout


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on placement failure."""
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    start_h = random.randint(START_H_MIN, START_H_MAX)
    start_dir = random.choice(DIRS)
    start = _place_start(inp, out, occupied, ray_mask, start_h, start_dir)

    chain_len = random.choices(range(len(CHAIN_LEN_WEIGHTS)), weights=CHAIN_LEN_WEIGHTS)[0]
    min_dist = 1 if chain_len == 0 else 0
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(min_dist, max_dist)

    starts = [tuple(start["cap"])]
    direction = tuple(start["direction"])
    include_origin = True
    for _ in range(chain_len):
        blob = _candidate_next_blob(H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask)
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["blue"], RED)
        starts = list(blob["ports"])
        direction = tuple(blob["spec"]["port_dir"])
        include_origin = False

    final_path = _path_to_boundary(starts, direction, include_origin, H, W)
    if any(bool(blue_mask[r, c]) for r, c in final_path):
        return None
    _mark_ray(out, ray_mask, final_path)

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
    raise RuntimeError("failed to generate a valid 5961cc34 D12 input")
