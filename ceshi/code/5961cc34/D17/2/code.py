# Variant of 5961cc34
# Dimension: D17 — 干扰色块处理 (obstacle)
# Change: 干扰色块在输出中被完全擦除为背景色（蓝色像素和绿色端口都消失）... → 干扰色块作为射线阻挡物

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# -----------------------------------------------------------------------------
# Design analysis for the D17 variant
#
# a) Concrete behavior
# - We keep the original object vocabulary: azure background, blue blob body,
#   green port edge, yellow start cap, red rays/activated blobs.
# - The visible new case is: a ray can now end directly in front of a blue/green
#   blob whose green side faces the incoming ray. The blocker blob is erased in
#   the output, but the ray no longer continues through it.
# - Small trace:
#   * Original: start ray goes upward; any blue cell hit becomes an activated
#     blob, and its own green side becomes the outgoing red ray.
#   * Variant: if the ray would first touch a blob on that blob's green side,
#     that green side acts like a shield. The ray stops on the previous cell,
#     the blob never activates, and the whole blob disappears in the output.
#   * Difference: "blue-side hit => activate", "green-side hit => block".
#
# b) Complexity calibration
# - The role distinction uses an already-visible feature: the side carrying the
#   green markers. No hidden ordering, color-id arithmetic, or extra symbols.
# - There is still exactly one start marker and one outgoing side per blob, so
#   the search space remains close to the original task.
# - To make D17 observable, every generated grid contains exactly one blocker on
#   an actually traversed ray.
#
# c) Original solver assumptions that break under D17
# - Original start ray loop paints first and only stops on blue; it would paint
#   through a blocking green edge.
# - Original thick-ray loop also paints through every green cell; that was safe
#   only because distractors were guaranteed off-ray and activated blobs always
#   exposed blue on the incoming side.
# - Original generator forbids next blob port_dir == incoming side because that
#   would put green on the entry edge. D17 intentionally needs that geometry for
#   blockers, so generation must now allow that case for the final blocker only.
#
# d) Placement legality
# - Activated-chain blobs still use the original placement constraints.
# - The blocker is placed by lane-matching against the incoming ray thickness,
#   just like a normal target, except the matched entry cells are the green edge
#   itself and the red path ends one step before them.
# - The blocker footprint never overlaps other blobs or painted ray cells.
#
# e) Visual observability
# - Across examples, some blobs are entered from a non-green side and turn red.
#   One blob is approached from its green side and causes a red line to stop
#   short. This is visible in 2-3 examples without needing numeric clues.
#
# f) Clean visualization
# - Only one on-path blocker is used.
# - We do not add extra off-path distractors, avoiding clutter and keeping the
#   one important exception easy to notice.
#
# g) Color reasoning
# - Color roles stay identical to the original task: blue body, green port, red
#   result, yellow starter. The only new inference is whether the ray meets the
#   blue side or the green side first.
#
# h) Generator guarantees for solver assumptions
# - Every blocker's green side faces the incoming ray.
# - Every activatable blob's incoming side is blue, never green.
# - There is at most one non-activated blob total, so the solver's final sanity
#   check remains meaningful.
# -----------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a red cascade where some blue-green blobs can terminate the beam instead of activating.

    Input structure:
    - Background is the dominant color. Blue blobs still carry one green edge that marks their port side.
    - A yellow cap on top of a short bottom red segment marks the initial upward ray.

    Rule:
    - Rays repaint traversed cells red. A blob hit through a blue side activates, turns red, and emits along its green side.
    - If a ray reaches another blob on that blob's green side, the ray stops immediately before the blob; that blob is not activated.
    - Continue the breadth-first cascade until no newly activated blobs remain.

    Key insight: The same blue-green shape has two roles depending on which side the ray meets first: blue-side contact activates, while green-side contact acts as a blocker.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the start "port" (bottom vertical segment with a distinct cap) ---
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
                (
                    int(col),
                    int(cap),
                    int(c0),
                    [(int(rr), int(c0)) for rr in range(int(r0), int(r1) + 1)],
                )
            )

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]

    # --- STEP 2) Identify the blob body color and the port-marker color ---
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
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == blue_col:
                    ok = True
                    break
            if ok:
                break
        if ok:
            green_candidates.append(int(col))

    assert len(green_candidates) == 1
    green_col = int(green_candidates[0])

    out = np.full_like(g, bg)

    # --- STEP 3) Extract blue components and record their port geometry ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == blue_col, connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = len(blobs)
        for r, c in pts:
            blob_id[r, c] = bi
        blobs.append({"pts": pts, "ports": [], "port_dirs": set()})

    green_owner: dict[tuple[int, int], int] = {}
    for bi, blob in enumerate(blobs):
        pts = blob["pts"]
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in pts:
            for ddr, ddc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + ddr)
                cc = int(c + ddc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == green_col:
                    seen_dirs.add((int(ddr), int(ddc)))
        for ddr, ddc in seen_dirs:
            port_pixels = [
                (int(r), int(c))
                for (r, c) in pts
                if 0 <= int(r + ddr) < H and 0 <= int(c + ddc) < W and int(g[int(r + ddr), int(c + ddc)]) == green_col
            ]
            green_pixels = [(int(r + ddr), int(c + ddc)) for r, c in port_pixels]
            assert port_pixels
            blob["ports"].append({"dir": (int(ddr), int(ddc)), "pixels": port_pixels, "green": green_pixels})
            blob["port_dirs"].add((int(ddr), int(ddc)))
            for gp in green_pixels:
                green_owner[gp] = bi

    activated: set[int] = set()
    queue: list[int] = []

    def activate_blob_at(r: int, c: int) -> int:
        """Recolor a blob to red when first hit through a blue side; enqueue it once."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return bi
        activated.add(bi)
        queue.append(bi)
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = red_col
        return bi

    def trace_ray(
        starts: list[tuple[int, int]],
        d: tuple[int, int],
        *,
        include_origin: bool,
        source_blob: int | None,
    ) -> None:
        """
        Trace one thin or thick ray.

        Green cells belonging to the source blob's emitting side are traversable;
        any later green cell is treated as a blocking edge and is not painted.
        """
        ddr, ddc = map(int, d)
        allowed_green: set[tuple[int, int]] = set()
        if source_blob is not None:
            for port in blobs[source_blob]["ports"]:
                if tuple(port["dir"]) == (ddr, ddc):
                    allowed_green.update((int(r), int(c)) for r, c in port["green"])

        for sr, sc in starts:
            r, c = int(sr), int(sc)
            if include_origin:
                out[r, c] = red_col
            while True:
                nr = int(r + ddr)
                nc = int(c + ddc)
                if not (0 <= nr < H and 0 <= nc < W):
                    break
                if (nr, nc) in allowed_green:
                    out[nr, nc] = red_col
                    r, c = nr, nc
                    continue
                cell = int(g[nr, nc])
                if cell == green_col and (nr, nc) in green_owner:
                    break
                out[nr, nc] = red_col
                if cell == blue_col:
                    activate_blob_at(nr, nc)
                    break
                r, c = nr, nc

    # --- STEP 4) Paint the start segment and launch the initial upward ray ---
    for r, c in start_seg:
        out[int(r), int(c)] = red_col
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, start_col]) == yellow_col
    out[top_r - 1, start_col] = red_col
    trace_ray([(top_r - 1, start_col)], (-1, 0), include_origin=False, source_blob=None)

    # --- STEP 5) Propagate by BFS: each activated blob emits through its own green side ---
    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            trace_ray(port["pixels"], tuple(port["dir"]), include_origin=False, source_blob=bi)

    assert len(blobs) - len(activated) <= 1, "expected at most one non-activated blocker"
    return out


# --------------- generator (a2_cascade_obstacle) ---------------

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
START_COL_PAD = 2

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3

# --- Activated chain ---
CHAIN_LEN_WEIGHTS = [1, 3, 3, 3]
MAX_ACTIVATED = 3

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 400


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
        "green": _sort_on_ray(green, port_dir),
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
    return [(r0 + r, c0 + c) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in coords)


def _path_to_targets(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    targets: list[tuple[int, int]],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for (r, c), (tr, tc) in zip(starts, targets):
        if include_origin:
            cells.append((r, c))
        guard = H + W + 10
        while (r, c) != (tr, tc):
            r, c = r + dr, c + dc
            guard -= 1
            if guard < 0 or not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((r, c))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


def _try_place_activator_along_axis(
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
        path = _path_to_targets(starts, d, entry_cells, include_origin, H, W)
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


def _try_place_blocker_along_axis(
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
        stop_cells = [(int(r - d[0]), int(c - d[1])) for r, c in entry_cells]
        full = _offset(spec["full"], r0, c0)
        if not _in_bounds(full, H, W) or not _in_bounds(stop_cells, H, W):
            continue
        if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
            continue
        if any(occ_pad[r, c] for r, c in stop_cells):
            continue
        path = _path_to_targets(starts, d, stop_cells, include_origin, H, W)
        if path is None:
            continue
        if any(blue_mask[r, c] for r, c in path):
            continue
        candidates.append(
            {
                "spec": spec,
                "r0": r0,
                "c0": c0,
                "entry": entry_cells,
                "stop": stop_cells,
                "path": path,
                "blue": _offset(spec["blue"], r0, c0),
                "full": full,
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
                _try_place_activator_along_axis(
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


def _candidate_blocker(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> dict | None:
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    candidates: list[dict] = []

    for spec in _all_specs(entry_side):
        edge = spec["green"]
        if len(edge) < thickness:
            continue
        for k in range(len(edge) - thickness + 1):
            window = edge[k : k + thickness]
            win_delta = [_perp_value(p, d) - _perp_value(window[0], d) for p in window]
            if win_delta != lane_delta:
                continue
            candidates.extend(
                _try_place_blocker_along_axis(
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
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[r, c] = True
    for r, c in blob["blue"]:
        blue_mask[r, c] = True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    # Place start marker.
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

    chain_len = min(random.choices(range(len(CHAIN_LEN_WEIGHTS)), weights=CHAIN_LEN_WEIGHTS)[0], MAX_ACTIVATED)

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

    blocker = _candidate_blocker(H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask)
    if blocker is None:
        return None
    _mark_ray(out, ray_mask, blocker["path"])
    _place_blob(inp, occupied, blue_mask, blocker)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 D17 obstacle input")
