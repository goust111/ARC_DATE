# Variant of 5961cc34
# Dimension: D2 — 级联激活机制 (path)
# Change: BFS 级联：命中色块后立即激活并发射新射线，所有被间接命中的色块也被激活，直到无新命中... → 距离衰减激活

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# Design analysis for the D2 variant
# ----------------------------------
# a) Concrete behavior / trace
#    - The object vocabulary stays the same: blue diamond/octagon blobs, one green
#      port edge per blob, and a bottom start marker made of a yellow cap over a
#      red vertical segment.
#    - The only semantic change is that the height of the red start segment is now
#      the global ray budget. The start ray and every later thick port-ray can move
#      at most that many cells from its launch cells.
#    - Example trace:
#      * Original: start height is incidental. A start ray reaches blob A, A emits
#        right, the rightward ray reaches blob B no matter how far B is, then B
#        emits upward, etc.
#      * Variant: if the start height is 5, then the start ray can travel only 5
#        cells above the yellow cap. Likewise, A's rightward ray can move only 5
#        cells from A's port pixels. Blob B activates only if its entry edge lies
#        within 5 steps; otherwise the ray stops in background and B disappears in
#        the output.
#      * Difference: the cascade is still port-driven, but every hop is range
#        limited, so on-line blobs can remain unactivated purely because they are
#        too far away.
# b) Complexity calibration
#    - The budget is a visible count, not a hidden mapping: budget == start-segment
#      height. To make this readable, generator uses start heights 4..7 instead of
#      the original 2..3.
#    - Each sample contains at least one "overflow" blob placed on a valid ray line
#      beyond the budget. This makes the new behavior visually distinct from the
#      original unlimited version.
#    - We keep only 0..2 reachable activations, so the short budget remains
#      compatible with the original blob sizes and the grid does not become crowded.
# c) Implicit-assumption audit for main()
#    - Original STEP 5 assumed the start ray should continue until first hit or top.
#      That is no longer valid; it must stop after `len(start_seg)` steps.
#    - Original `extend_thick` assumed every emitted ray extends to boundary or hit.
#      That is also invalid; emitted rays now consume the same fixed budget.
#    - Original final assert allowed at most one unactivated blob because only
#      off-ray distractors stayed blue. In the variant, an intentionally aligned
#      beyond-range blob is normal, so that assert must be removed.
#    - All color-detection logic and blob/port extraction logic still hold because
#      the visible object grammar is unchanged.
# d) Placement legality
#    - Reachable blobs are placed only when their entry edge is within the budget
#      and the whole approach path is clear.
#    - The overflow blob is placed on the final active ray line with entry distance
#      strictly greater than the budget, so the solver never touches it but the
#      unlimited original rule would have.
#    - Object masks still cannot overlap or leave bounds, and actual output rays are
#      recorded in `ray_mask` so later placements avoid them.
# e) Visual observability
#    - Humans can count the start height and compare it with the visible red ray
#      lengths. Several samples will show rays terminating early with a blue blob
#      still farther along the same line in the input.
#    - No hidden use of color ids or non-visual ordering is introduced.
# f) Visual cleanliness
#    - Extra off-ray distractors are removed in this variant. The aligned overflow
#      blob is sufficient to exhibit the new rule and keeps the scene legible.
# g) Color-change inference
#    - Color behavior is unchanged from the base task: activated structure becomes
#      red, inactive blobs disappear to background. The only new inference is when
#      activation happens, and that is supported by the visible distance budget.
# h) New solver assumptions guaranteed by the generator
#    - Exactly one bottom start marker exists.
#    - Start-segment height is in 4..7, so a nontrivial budget is always visible.
#    - Every activated blob has one valid port side, and each reachable hop lies
#      within the budget while the final overflow target lies beyond it.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a red signal whose travel budget is encoded by the start-column height.

    Input structure:
    - Azure background with blue diamond/octagon blobs; green edge pixels mark each blob's outgoing port side.
    - A yellow cap sits above a bottom red segment, and that red segment's height is the ray budget.

    Rule:
    - Recolor the start marker red and launch the upward ray for at most that many steps.
    - Any blue blob reached within budget turns red and emits thick rays through its green port, but every emitted ray gets the same fixed budget and disappears once it spends it.

    Key insight: The cascade still follows the port directions, but the start-segment height globally limits every hop, so farther blobs on the same line stay inactive.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

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
                (int(col), cap, c0, [(int(rr), c0) for rr in range(r0, r1 + 1)])
            )

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]
    ray_limit = int(len(start_seg))

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

    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == blue_col, connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[r, c] = bi
        blobs.append({"pts": pts, "ports": []})

    for blob in blobs:
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
            assert port_pixels
            blob["ports"].append({"dir": (int(ddr), int(ddc)), "pixels": port_pixels})

    activated: set[int] = set()
    queue: list[int] = []

    def activate_blob_at(r: int, c: int) -> None:
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(bi)
        queue.append(bi)
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = red_col

    def extend_limited(origins: list[tuple[int, int]], d: tuple[int, int], include_origin: bool) -> None:
        ddr, ddc = map(int, d)
        for pr, pc in origins:
            r, c = int(pr), int(pc)
            if include_origin:
                out[r, c] = red_col
            for _ in range(ray_limit):
                r += ddr
                c += ddc
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[r, c] = red_col
                if int(g[r, c]) == blue_col:
                    activate_blob_at(r, c)
                    break

    for r, c in start_seg:
        out[int(r), int(c)] = red_col
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, start_col]) == yellow_col
    extend_limited([(top_r - 1, start_col)], (-1, 0), include_origin=True)

    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            extend_limited(port["pixels"], port["dir"], include_origin=False)

    return out


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

H_MIN, H_MAX = 18, 27
W_MIN, W_MAX = 15, 27
START_COL_PAD = 2

START_H_MIN, START_H_MAX = 4, 7

MAX_REACHABLE_CHAIN = 2
MAX_BLOB_COUNT = 4

MIN_GAP_FIRST = 2
MIN_GAP_CHAIN = 2
OVERFLOW_EXTRA_MAX = 6

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
            cells.append((r, c))
        guard = H + W + 10
        while (r, c) != (er, ec):
            r, c = r + dr, c + dc
            guard -= 1
            if guard < 0 or not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((r, c))
    return cells


def _path_with_limit(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    limit: int,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        r = int(r)
        c = int(c)
        if include_origin:
            cells.append((r, c))
        for _ in range(int(limit)):
            r, c = r + dr, c + dc
            if not (0 <= r < H and 0 <= c < W):
                break
            cells.append((r, c))
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
    step_lo: int,
    step_hi: int,
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
        steps = abs(int(entry_line - start_line))
        if steps < int(step_lo) or steps > int(step_hi):
            continue

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
                "steps": steps,
                "blue": _offset(spec["blue"], r0, c0),
                "full": full,
                "ports": _sort_on_ray(_offset(spec["port_pixels"], r0, c0), spec["port_dir"]),
            }
        )

    return candidates


def _candidate_blob(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    step_lo: int,
    step_hi: int,
    allow_terminal: bool,
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
        if not allow_terminal and spec["port_dir"] == entry_side:
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
                    step_lo,
                    step_hi,
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


def _reachable_chain_len(ray_limit: int) -> int:
    if ray_limit <= 5:
        return random.choices([0, 1], weights=[1, 4])[0]
    return random.choices([0, 1, 2], weights=[1, 3, 3])[0]


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
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    inp[cap_r, start_col] = YELLOW
    out[cap_r, start_col] = RED
    ray_mask[cap_r, start_col] = True
    occupied[cap_r, start_col] = True
    for r in range(cap_r + 1, H):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        occupied[r, start_col] = True

    ray_limit = start_h
    reachable_len = _reachable_chain_len(ray_limit)
    if reachable_len > MAX_REACHABLE_CHAIN:
        return None

    starts = [(cap_r, start_col)]
    direction = UP
    include_origin = True

    for _ in range(reachable_len):
        blob = _candidate_blob(
            H,
            W,
            starts,
            direction,
            include_origin,
            occupied,
            blue_mask,
            ray_mask,
            step_lo=1,
            step_hi=ray_limit,
            allow_terminal=False,
        )
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["blue"], RED)
        starts = blob["ports"]
        direction = blob["spec"]["port_dir"]
        include_origin = False

    overflow_hi = min(ray_limit + OVERFLOW_EXTRA_MAX, H + W)
    overflow_blob = _candidate_blob(
        H,
        W,
        starts,
        direction,
        include_origin,
        occupied,
        blue_mask,
        ray_mask,
        step_lo=ray_limit + 1,
        step_hi=overflow_hi,
        allow_terminal=True,
    )
    if overflow_blob is None:
        return None
    _place_blob(inp, occupied, blue_mask, overflow_blob)

    final_path = _path_with_limit(starts, direction, include_origin, ray_limit, H, W)
    if any(blue_mask[r, c] for r, c in final_path):
        return None
    _mark_ray(out, ray_mask, final_path)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 D2 distance-limited input")
