# Variant of 5961cc34
# Dimension: D6 — 色块停止条件与终点 (dst)
# Change: 射线遇到色块时停止并激活该色块；若未遇到色块则延伸到网格边界... → 距离编码停止

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste

# Design analysis (D6 distance-encoding)
# spec:
#   variant_core:
#     original_behavior:
#       - start ray goes upward until first blue hit or top edge
#       - each activated blob emits through its green side until another blob or boundary
#     new_behavior:
#       - every ray has a fixed travel budget measured in cells beyond its origin
#       - start budget = red bottom segment height
#       - blob budget = blob visible span along the emission axis
#         vertical emission -> blob height, horizontal emission -> blob width
#       - hitting blue still activates that blob, but no longer truncates the ray
#       - after activation the ray keeps painting until its budget is exhausted
#   concrete_pixel_effect:
#     - the yellow-capped start marker still sits at the bottom, but the upward red trace
#       now stops after a short visible distance rather than always climbing to the first hit/top
#     - activated blobs still turn red and still seed BFS, yet their outgoing rays may
#       visibly continue past the blob they activate or stop in empty space before the boundary
#   behavior_trace:
#     original:
#       - start height 3, first blob 6 cells above the cap -> original ray would reach it and stop
#     variant:
#       - same start height 3 -> the ray only paints 3 cells above the cap
#       - therefore that far blob is not activated unless another shorter-placement example is used
#       - when a blob of size 5x4 emits upward/rightward, its ray length is 5 or 4 cells respectively
#       - if the next blob is hit at step 2 of a length-5 ray, steps 3..5 are still painted red
#     visible_difference:
#       - rays are no longer explained by "keep going until collision/border"
#       - humans can instead match ray extent to a visible size cue in the source object
#   complexity_calibration:
#     - keep only one distance rule for all blob rays: span along the outgoing axis
#     - keep start distance tied directly to the visible red-segment height
#     - keep the original blob language (same shapes, same single green edge) so the only
#       conceptual novelty is the stopping condition, not a new object vocabulary
#     - generator biases toward pass-through cases where a ray activates a blob before its
#       distance budget is exhausted; this prevents collapse back to the original behavior
#   implicit_assumption_audit_main:
#     - start detection still assumes exactly one thin bottom-touching vertical segment with a non-bg cap
#       generator preserves that
#     - blue is still inferred as the dominant non-start/non-bg color; generator always includes at least one blob
#     - green is still the unique color 4-adjacent to blue; generator keeps one green edge per blob
#     - blue connected components still identify blobs; green replaces edge cells but never disconnects blobs
#     - original "break on first blue hit" is invalid under D6 and must be removed
#     - original "extend until out-of-bounds" is invalid under D6 and must be replaced by a counted loop
#     - blob size for distance cannot use blue-only bbox, because a full green top row would shrink it;
#       solver therefore reconstructs visible extent from blue plus adjacent green edge cells
#   placement_legality:
#     - generator traces each ray for its exact fixed distance before placing future blobs/distractors
#     - candidate next blobs must intersect that fixed ray corridor, stay within bounds, and keep
#       one-cell separation from existing blobs
#     - except for the emitting blob's own green edge cells, fixed-length rays may not cross existing objects
#     - distractors are placed only off the accumulated ray mask
#   observability_and_cleanliness:
#     - start heights are enlarged to 4-7 so the first distance cue is easy to see
#     - final rays frequently stop mid-grid, and many intermediate rays continue after activation,
#       making the D6 change visually obvious in 2-3 examples
#     - colors stay unchanged: blue bodies, green ports, yellow cap, red result, azure background
#   solver_generator_contract:
#     - every emitting blob has exactly one green side
#     - the next chain blob is the only blob intersecting that ray corridor
#     - if a blob emits, there is enough free space to paint its full fixed distance in-bounds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red cascade of fixed-length rays triggered by blue blobs with green ports.

    Input structure:
    - bg is the dominant color. Blue diamond/octagon blobs each have one green edge marking the outgoing direction.
    - At the bottom, a yellow cap sits on a short red vertical segment that encodes the first ray length.

    Rule:
    - Recolor the start marker to red and fire upward for exactly the red-segment height.
    - Any blue blob touched by a ray activates, turns red, and emits from its green side for a distance equal to its visible size along that direction.
    - Rays do not stop when they hit a blob; they keep going until their encoded distance is used up, and BFS continues until no new blobs activate.

    Key insight: Distances are encoded by object size, not by obstacles, so the solver must count cells from the start segment or from an activated blob's span.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the start marker (bottom red segment with a differently colored cap) ---
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
            if cap in (int(bg), int(col)):
                continue
            start_candidates.append(
                (
                    int(col),
                    int(cap),
                    int(c0),
                    [(int(rr), int(c0)) for rr in range(r0, r1 + 1)],
                )
            )

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]

    # --- STEP 2) Identify blob colors ---
    counts = Counter(map(int, g.flatten()))
    other_cols = [(int(cnt), int(col)) for col, cnt in counts.items() if int(col) not in (bg, red_col, yellow_col)]
    assert other_cols
    max_count = max(cnt for cnt, _col in other_cols)
    blue_candidates = [int(col) for cnt, col in other_cols if cnt == max_count]
    assert len(blue_candidates) == 1
    blue_col = int(blue_candidates[0])

    green_candidates: list[int] = []
    for col in colors_present(g, exclude={bg, red_col, yellow_col, blue_col}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        ok = False
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(blue_col):
                    ok = True
                    break
            if ok:
                break
        if ok:
            green_candidates.append(int(col))

    assert len(green_candidates) == 1
    green_col = int(green_candidates[0])

    out = np.full_like(g, int(bg))

    # --- STEP 3) Extract blue blobs and index their pixels ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[int(r), int(c)] = bi
        blobs.append({"pts": pts, "ports": [], "visible_pts": set(pts)})

    # --- STEP 4) Recover green ports and visible blob spans ---
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
                if 0 <= int(r + ddr) < H and 0 <= int(c + ddc) < W and int(g[int(r + ddr), int(c + ddc)]) == int(green_col)
            ]
            assert port_pixels
            green_pixels = [(int(r + ddr), int(c + ddc)) for r, c in port_pixels]
            blob["visible_pts"].update(green_pixels)
            blob["ports"].append(
                {
                    "dir": (int(ddr), int(ddc)),
                    "pixels": port_pixels,
                    "green": green_pixels,
                }
            )
        assert len(blob["ports"]) == 1
        vis = np.array(sorted(blob["visible_pts"]), dtype=int)
        vr0, vc0 = vis.min(axis=0)
        vr1, vc1 = vis.max(axis=0)
        blob["height"] = int(vr1 - vr0 + 1)
        blob["width"] = int(vc1 - vc0 + 1)
        for port in blob["ports"]:
            ddr, ddc = map(int, port["dir"])
            port["distance"] = int(blob["height"] if ddr else blob["width"])

    activated: set[int] = set()
    queue: list[int] = []

    def activate_blob_at(r: int, c: int) -> None:
        """Activate the blob containing (r,c) once and recolor its blue body red."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(red_col)

    def extend_fixed(starts: list[tuple[int, int]], d: tuple[int, int], steps: int, include_origin: bool) -> None:
        """Paint a ray for exactly `steps` cells beyond each start, activating any blue cells it crosses."""
        ddr, ddc = map(int, d)
        for sr, sc in starts:
            r, c = int(sr), int(sc)
            if include_origin:
                out[int(r), int(c)] = int(red_col)
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c))
            for _ in range(int(steps)):
                r += int(ddr)
                c += int(ddc)
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[int(r), int(c)] = int(red_col)
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c))

    # --- STEP 5) Fire the start ray using the segment height as the distance code ---
    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    top_r = min(int(r) for r, _c in start_seg)
    cap_r = int(top_r) - 1
    assert cap_r >= 0 and int(g[int(cap_r), int(start_col)]) == int(yellow_col)
    extend_fixed([(int(cap_r), int(start_col))], (-1, 0), len(start_seg), include_origin=True)

    # --- STEP 6) BFS cascade: each activated blob emits its own fixed-length ray ---
    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            extend_fixed(port["pixels"], port["dir"], int(port["distance"]), include_origin=False)

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    return out


# --------------- generator (distance-encoded cascade) ---------------

BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

SHAPES = ((3, 4), (4, 4), (5, 4), (5, 6), (6, 6), (8, 8))
_SHAPE_WIDTHS = {
    (3, 4): (2, 4, 2),
    (4, 4): (2, 4, 4, 2),
    (5, 4): (2, 4, 4, 4, 2),
    (5, 6): (2, 4, 6, 4, 2),
    (6, 6): (2, 4, 6, 6, 4, 2),
    (8, 8): (2, 4, 6, 8, 8, 6, 4, 2),
}

H_MIN, H_MAX = 13, 27
W_MIN, W_MAX = 15, 27
START_COL_PAD = 2

START_H_MIN, START_H_MAX = 4, 7

CHAIN_LEN_WEIGHTS = [1, 3, 3, 3]
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

MIN_GAP_FIRST = 2
MIN_GAP_CHAIN = 2

GEN_RETRIES = 500


def _shape_mask(h: int, w: int) -> np.ndarray:
    widths = _SHAPE_WIDTHS[(h, w)]
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(widths):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return int(rc[1] if d[0] else rc[0])


def _along_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return int(rc[0] if d[0] else rc[1])


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
        br, bc = gr - port_dir[0], gc - port_dir[1]
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
        "green": green,
        "port_dir": port_dir,
        "port_pixels": _sort_on_ray(port_pixels, port_dir),
    }


def _all_specs(port_dir: tuple[int, int] | None = None) -> list[dict]:
    dirs = [port_dir] if port_dir is not None else list(DIRS)
    out = [s for size in SHAPES for d in dirs if (s := _blob_spec(size, d)) is not None]
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


def _trace_steps(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    steps: int,
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    dr, dc = map(int, d)
    cells: list[tuple[int, int]] = []
    for sr, sc in starts:
        r, c = int(sr), int(sc)
        if include_origin:
            if not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((int(r), int(c)))
        for _ in range(int(steps)):
            r += int(dr)
            c += int(dc)
            if not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((int(r), int(c)))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


def _ray_span_for_size(size: tuple[int, int], d: tuple[int, int]) -> int:
    return int(size[0] if d[0] else size[1])


def _step_index_on_ray(
    cell: tuple[int, int],
    lane_origin: dict[int, int],
    d: tuple[int, int],
) -> int:
    lane = _perp_value(cell, d)
    return int(abs(_along_value(cell, d) - lane_origin[int(lane)]))


def _candidate_next_blob(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    steps_limit: int,
    allowed_overlap: set[tuple[int, int]],
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    prefer_passthrough: bool,
) -> dict | None:
    starts = _sort_on_ray(starts, d)
    full_path = _trace_steps(starts, d, steps_limit, include_origin, H, W)
    if full_path is None:
        return None

    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [int(v - lane_vals[0]) for v in lane_vals]
    lane_origin = {int(_perp_value(p, d)): int(_along_value(p, d)) for p in starts}
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    sign = -1 if d in (UP, LEFT) else 1
    start_line = _along_value(starts[0], d)

    candidates: list[dict] = []
    passthrough: list[dict] = []

    for spec in _all_specs():
        if spec["port_dir"] == entry_side:
            continue
        edge = _edge_offsets(spec["mask"], entry_side)
        if len(edge) < thickness:
            continue
        h, w = spec["mask"].shape
        edge_line = _along_value(edge[0], d)
        for k in range(len(edge) - thickness + 1):
            window = edge[k : k + thickness]
            win_delta = [_perp_value(p, d) - _perp_value(window[0], d) for p in window]
            if win_delta != lane_delta:
                continue
            fixed = lane_vals[0] - _perp_value(window[0], d)
            for hit_step in range(int(min_gap), int(steps_limit) + 1):
                entry_line = start_line + sign * int(hit_step)
                origin_along = int(entry_line - edge_line)
                r0 = int(origin_along if d[0] else fixed)
                c0 = int(fixed if d[0] else origin_along)
                full = _offset(spec["full"], r0, c0)
                if not _in_bounds(full, H, W):
                    continue
                if any(bool(occ_pad[r, c] or ray_mask[r, c]) for r, c in full):
                    continue

                full_set = set(full)
                if any(
                    bool(occupied[r, c] or ray_mask[r, c])
                    for r, c in full_path
                    if (r, c) not in allowed_overlap and (r, c) not in full_set
                ):
                    continue
                if any(bool(blue_mask[r, c]) for r, c in full_path if (r, c) not in full_set):
                    continue

                blue = _offset(spec["blue"], r0, c0)
                blue_set = set(blue)
                touched_blue = [cell for cell in full_path if cell in blue_set]
                if not touched_blue:
                    continue
                first_hit = min(_step_index_on_ray(cell, lane_origin, d) for cell in touched_blue)
                overshoot = int(steps_limit - first_hit)

                cand = {
                    "spec": spec,
                    "r0": int(r0),
                    "c0": int(c0),
                    "path": list(full_path),
                    "blue": blue,
                    "full": full,
                    "ports": _sort_on_ray(_offset(spec["port_pixels"], r0, c0), spec["port_dir"]),
                    "green": _sort_on_ray(_offset(spec["green"], r0, c0), spec["port_dir"]),
                    "hit_step": int(first_hit),
                    "overshoot": int(overshoot),
                }
                candidates.append(cand)
                if int(overshoot) > 0:
                    passthrough.append(cand)

    if prefer_passthrough and passthrough:
        return random.choice(passthrough)
    return random.choice(candidates) if candidates else None


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[int(r), int(c)] = True
    for r, c in blob["blue"]:
        blue_mask[int(r), int(c)] = True


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
                if any(bool(occ_pad[r, c] or ray_mask[r, c]) for r, c in full):
                    continue
                candidates.append({"spec": spec, "r0": r0, "c0": c0, "full": full, "blue": _offset(spec["blue"], r0, c0)})
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
    ray_mask = np.zeros((H, W), dtype=bool)

    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = H - start_h - 1
    if cap_r <= 0:
        return None
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    inp[int(cap_r), int(start_col)] = YELLOW
    occupied[int(cap_r), int(start_col)] = True
    for r in range(int(cap_r) + 1, H):
        inp[int(r), int(start_col)] = RED
        out[int(r), int(start_col)] = RED
        occupied[int(r), int(start_col)] = True

    chain_len = random.choices(range(len(CHAIN_LEN_WEIGHTS)), weights=CHAIN_LEN_WEIGHTS)[0]
    min_dist = 1 if chain_len == 0 else 0
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(min_dist, max_dist)

    starts = [(int(cap_r), int(start_col))]
    direction = UP
    include_origin = True
    allowed_overlap = {(int(cap_r), int(start_col))}
    ray_steps = int(start_h)

    for idx in range(int(chain_len)):
        prefer_passthrough = bool(random.random() < 0.7 or idx == 0)
        blob = _candidate_next_blob(
            H,
            W,
            starts,
            direction,
            include_origin,
            ray_steps,
            allowed_overlap,
            occupied,
            blue_mask,
            ray_mask,
            prefer_passthrough,
        )
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["blue"], RED)

        starts = blob["ports"]
        direction = blob["spec"]["port_dir"]
        include_origin = False
        allowed_overlap = set(blob["green"])
        ray_steps = _ray_span_for_size(blob["spec"]["size"], direction)

    final_path = _trace_steps(starts, direction, ray_steps, include_origin, H, W)
    if final_path is None:
        return None
    if any(
        bool(occupied[r, c] or ray_mask[r, c])
        for r, c in final_path
        if (r, c) not in allowed_overlap
    ):
        return None
    if any(bool(blue_mask[r, c]) for r, c in final_path if (r, c) not in allowed_overlap):
        return None
    _mark_ray(out, ray_mask, final_path)

    for _ in range(int(num_distractors)):
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
