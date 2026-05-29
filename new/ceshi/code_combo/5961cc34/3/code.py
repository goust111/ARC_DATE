# Combo variant of 5961cc34 (index 3)
# combo_spec: D2/2+D3/0
# Name: 距离衰减+宽度累加

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior / trace
#    - The visible objects stay close to the base task: blue diamond/octagon blobs,
#      one green side-port per blob, and a bottom launcher made from a yellow cap
#      over a red post.
#    - Two path parameters now evolve together:
#      * Distance budget: the red post height is a global hop budget. Every ray,
#        including the start ray and every later emission, can travel at most that
#        many cells from its launch cells.
#      * Width growth: the start ray is width 1. Once blob A is hit, A emits width
#        2; if that hits blob B, B emits width 3; no emitted width comes from the
#        green-bar length anymore.
#    - Example trace:
#      * Base task: start hits A; A emits width = len(green_A) forever until the
#        next blob or boundary; later hops are also unlimited.
#      * Combo task: if the red post height is 5, the start ray gets 5 upward
#        steps. If it reaches A, A emits width 2 for at most 5 steps. If that
#        reaches B, B emits width 3 for at most 5 steps. A farther aligned blob C
#        can sit on B's outgoing line but survive because the 5-step budget expires
#        before reaching C.
#      * Difference: the solver must track both "how far can this hop go?" and
#        "what width does the next hop have?" at the same time.
# b) Complexity calibration
#    - To keep the distance rule visually countable, the generator uses start-post
#      heights 4..7 instead of 2..3.
#    - Every sample contains at least one reachable activation and one aligned
#      overflow blob beyond budget. This prevents degeneration into either the
#      original task or a pure width-only task.
#    - Reachable chain length is capped at 2 blobs. That keeps visible widths in
#      the range 1->2 or 1->2->3, which the stock blob shapes can legally support
#      without forcing width 4 placement.
# c) Implicit-assumption audit for main()
#    - Base STEP 5 assumed the start ray continues until top/hit. That breaks:
#      the combo solver must stop after `len(start_seg)` steps.
#    - Base `extend_thick(port_pixels, d)` assumed width == green length and range
#      is unlimited. Both assumptions break. The combo solver therefore extracts
#      green cells only for direction/centering, derives launch lanes from the BFS
#      depth, and applies the same fixed distance budget to every hop.
#    - Base final assert allowed only off-ray distractors to survive. That breaks
#      because a deliberately aligned beyond-budget blob is now normal.
#    - Queue payload must change: a blob activation no longer needs just `blob_id`,
#      it must carry the emitted width for that blob's next hop.
# d) Placement legality
#    - Generator is still output-first/path-first: it constructs the reachable
#      cascade in output space, then places the corresponding blue+green blobs into
#      the input.
#    - Each reachable next blob is placed only if its incoming edge window matches
#      the current ray width and lies within the hop budget.
#    - Each emitting blob is additionally checked so that its widened launch strip
#      for the next hop is fully in bounds and does not collide with old objects.
#    - The final overflow blob is placed on the active ray line with entry distance
#      strictly greater than the budget, guaranteeing that combo/D2 logic leaves it
#      inactive while unlimited-width-only logic would continue farther.
# e) Visual observability
#    - The distance cue is directly visible as the red post height; the output rays
#      end after comparable lengths, and a farther aligned blob remains blue in the
#      input but absent in the output.
#    - The width cue is also direct: the first activated blob is preferentially
#      given a 1-pixel green port while its emitted ray is width 2, so "green size
#      = beam size" is visibly false across the training set.
#    - Samples with two activations often show a width-3 final ray, reinforcing the
#      additive growth pattern instead of a single one-off exception.
# f) Visual cleanliness
#    - No extra off-path distractors are added. The aligned overflow blob already
#      demonstrates the distance limit and keeps the scene readable.
#    - Minimum gaps are kept small enough for feasibility but large enough that the
#      blue blobs and widened launch strips remain visually separable.
# g) Change inferability
#    - Color behavior remains simple and structured: activated material and all ray
#      traces become red; untouched blobs disappear into background. The only new
#      inference is deciding when activation happens and how width grows, and both
#      are backed by visible geometric cues rather than hidden ids or ordering.
# h) Quantitative anti-degeneration target
#    - A good sample should differ from both single variants on the same input:
#      * from D2-only because the first emitted ray is width 2 despite a 1-cell
#        green port;
#      * from D3-only because the final aligned overflow blob lies beyond budget.
#    - The generator is deliberately biased so these two facts hold in almost every
#      accepted sample.
# i) New solver assumptions guaranteed by the generator
#    - Exactly one bottom launcher exists.
#    - Every blob has one contiguous green segment on one extreme side.
#    - Widened launch strips use the fixed left/up bias implemented by
#      `_expanded_starts`, and the generator validates those strips in bounds.
#    - Reachable blobs sit within the visible distance budget; the final overflow
#      blob sits beyond it on the current active ray line.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a budget-limited red cascade whose beam grows one pixel wider at every activation.

    Input structure:
    - Azure background with blue diamond/octagon blobs; each blob has one contiguous green side-port that shows the outgoing direction.
    - A yellow cap above a bottom red post marks the launcher, and the post height encodes the maximum travel distance for every hop.

    Rule:
    - Recolor the launcher red and fire a 1-cell upward ray, but stop each ray after the encoded number of steps.
    - When a blue blob is reached, recolor it red and emit again through its green-marked side; each cascade level increases the emitted width by 1.
    - Any blob beyond the distance limit stays inactive, and untouched blobs disappear into the background.

    Key insight: The green edge only fixes direction and centering, while the start-post height limits every hop and the cascade depth determines the next beam width.
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
                rr = r + dr
                cc = c + dc
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

    def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
        if d[0]:
            return sorted(coords, key=lambda p: (p[1], p[0]))
        return sorted(coords, key=lambda p: (p[0], p[1]))

    def _expanded_starts(coords: list[tuple[int, int]], d: tuple[int, int], width: int) -> list[tuple[int, int]]:
        coords = _sort_on_ray(coords, d)
        assert coords
        if d[0]:
            row = coords[0][0]
            vals = [c for _r, c in coords]
            assert all(r == row for r, _c in coords)
            span = vals[-1] - vals[0] + 1
            extra = int(width) - int(span)
            assert extra >= 0
            lo = vals[0] - (extra + 1) // 2
            hi = vals[-1] + extra // 2
            return [(int(row), int(c)) for c in range(int(lo), int(hi) + 1)]
        col = coords[0][1]
        vals = [r for r, _c in coords]
        assert all(c == col for _r, c in coords)
        span = vals[-1] - vals[0] + 1
        extra = int(width) - int(span)
        assert extra >= 0
        lo = vals[0] - (extra + 1) // 2
        hi = vals[-1] + extra // 2
        return [(int(r), int(col)) for r in range(int(lo), int(hi) + 1)]

    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[r, c] = bi
        blobs.append({"pts": pts, "ports": []})

    for blob in blobs:
        pts = blob["pts"]
        greens_set: set[tuple[int, int]] = set()
        for r, c in pts:
            for ddr, ddc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + ddr
                cc = c + ddc
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == green_col:
                    greens_set.add((int(rr), int(cc)))
        greens = sorted(greens_set)
        assert greens

        full_pts = pts + greens
        min_r = min(r for r, _c in full_pts)
        max_r = max(r for r, _c in full_pts)
        min_c = min(c for _r, c in full_pts)
        max_c = max(c for _r, c in full_pts)

        port_dirs: list[tuple[int, int]] = []
        if all(r == min_r for r, _c in greens):
            port_dirs.append(UP)
        if all(r == max_r for r, _c in greens):
            port_dirs.append(DOWN)
        if all(c == min_c for _r, c in greens):
            port_dirs.append(LEFT)
        if all(c == max_c for _r, c in greens):
            port_dirs.append(RIGHT)
        if len(port_dirs) != 1:
            raise AssertionError("ambiguous green port placement")
        port_dir = port_dirs[0]

        greens = _sort_on_ray(greens, port_dir)
        blob["ports"].append({"dir": port_dir, "greens": greens})

    activated: set[int] = set()
    queue: list[tuple[int, int]] = []

    def activate_blob_at(r: int, c: int, next_emit_width: int) -> None:
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append((int(bi), int(next_emit_width)))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(red_col)

    def extend_limited(
        starts: list[tuple[int, int]],
        d: tuple[int, int],
        include_origin: bool,
        next_emit_width: int,
    ) -> None:
        ddr, ddc = map(int, d)
        for sr, sc in starts:
            r = int(sr)
            c = int(sc)
            if not (0 <= r < H and 0 <= c < W):
                continue
            if include_origin:
                out[r, c] = int(red_col)
            for _ in range(ray_limit):
                r += ddr
                c += ddc
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[r, c] = int(red_col)
                if int(g[r, c]) == int(blue_col):
                    activate_blob_at(int(r), int(c), next_emit_width)
                    break

    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, start_col]) == int(yellow_col)
    extend_limited([(top_r - 1, start_col)], (-1, 0), include_origin=True, next_emit_width=2)

    while queue:
        bi, emit_width = queue.pop(0)
        for port in blobs[int(bi)]["ports"]:
            starts = _expanded_starts(port["greens"], port["dir"], int(emit_width))
            extend_limited(starts, port["dir"], include_origin=True, next_emit_width=int(emit_width) + 1)

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


def _expanded_starts(coords: list[tuple[int, int]], d: tuple[int, int], width: int) -> list[tuple[int, int]]:
    coords = _sort_on_ray(coords, d)
    assert coords
    if d[0]:
        row = coords[0][0]
        vals = [c for _r, c in coords]
        assert all(r == row for r, _c in coords)
        span = vals[-1] - vals[0] + 1
        extra = int(width) - int(span)
        assert extra >= 0
        lo = vals[0] - (extra + 1) // 2
        hi = vals[-1] + extra // 2
        return [(int(row), int(c)) for c in range(int(lo), int(hi) + 1)]
    col = coords[0][1]
    vals = [r for r, _c in coords]
    assert all(c == col for _r, c in coords)
    span = vals[-1] - vals[0] + 1
    extra = int(width) - int(span)
    assert extra >= 0
    lo = vals[0] - (extra + 1) // 2
    hi = vals[-1] + extra // 2
    return [(int(r), int(col)) for r in range(int(lo), int(hi) + 1)]


def _blob_spec(size: tuple[int, int], port_dir: tuple[int, int], green: list[tuple[int, int]]) -> dict | None:
    h, w = size
    mask = _shape_mask(h, w)
    edge = _edge_offsets(mask, port_dir)
    green = _sort_on_ray(green, port_dir)
    if not green or len(green) not in (1, 2):
        return None
    edge_set = set(edge)
    if any(pt not in edge_set for pt in green):
        return None
    edge_sets = {d: set(_edge_offsets(mask, d)) for d in DIRS}
    if any(sum(int(pt in edge_sets[d]) for d in DIRS) != 1 for pt in green):
        return None

    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    for r, c in green:
        patch[r, c] = GREEN

    green_set = set(green)
    blue = [(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) not in green_set]
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "size": (h, w),
        "mask": mask,
        "patch": patch,
        "blue": blue,
        "full": full,
        "port_dir": port_dir,
        "green": green,
    }


def _all_specs(port_dir: tuple[int, int] | None = None) -> list[dict]:
    dirs = [port_dir] if port_dir is not None else list(DIRS)
    out: list[dict] = []
    for size in SHAPES:
        mask = _shape_mask(*size)
        for d in dirs:
            edge = _edge_offsets(mask, d)
            for green_len in (1, 2):
                if green_len > len(edge):
                    continue
                for k in range(len(edge) - green_len + 1):
                    spec = _blob_spec(size, d, edge[k : k + green_len])
                    if spec is not None:
                        out.append(spec)
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


def _emit_starts_global(spec: dict, r0: int, c0: int, width: int) -> list[tuple[int, int]]:
    local = _expanded_starts(spec["green"], spec["port_dir"], width)
    return [(r0 + r, c0 + c) for r, c in local]


def _try_place_along_axis(
    spec: dict,
    window: list[tuple[int, int]],
    d: tuple[int, int],
    lane_vals: list[int],
    starts: list[tuple[int, int]],
    first_leg: bool,
    step_lo: int,
    step_hi: int,
    occ_pad: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    H: int,
    W: int,
    require_emit_starts: bool,
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
    min_gap = MIN_GAP_FIRST if first_leg else MIN_GAP_CHAIN
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

        path = _path_to_entries(starts, d, entry_cells, include_origin=True, H=H, W=W)
        if path is None:
            continue
        entry_set = set(entry_cells)
        if any(blue_mask[r, c] for r, c in path if (r, c) not in entry_set):
            continue

        candidate = {
            "spec": spec,
            "r0": r0,
            "c0": c0,
            "entry": entry_cells,
            "path": path,
            "steps": steps,
            "blue": _offset(spec["blue"], r0, c0),
            "full": full,
        }
        if require_emit_starts:
            emit_starts = _emit_starts_global(spec, r0, c0, len(starts) + 1)
            if not _in_bounds(emit_starts, H, W):
                continue
            own_full = set(full)
            if any((r, c) not in own_full and (occupied[r, c] or ray_mask[r, c]) for r, c in emit_starts):
                continue
            candidate["emit_starts"] = emit_starts

        candidates.append(candidate)

    return candidates


def _candidate_blob(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    first_leg: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    step_lo: int,
    step_hi: int,
    allow_terminal: bool,
    allowed_port_dirs: tuple[tuple[int, int], ...] | None = None,
    forbidden_blue_cells: set[tuple[int, int]] | None = None,
) -> dict | None:
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []

    for spec in _all_specs():
        if allowed_port_dirs is not None and spec["port_dir"] not in allowed_port_dirs:
            continue
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
                    first_leg,
                    step_lo,
                    step_hi,
                    occ_pad,
                    occupied,
                    blue_mask,
                    ray_mask,
                    H,
                    W,
                    require_emit_starts=not allow_terminal,
                )
            )

    if forbidden_blue_cells is not None:
        candidates = [
            cand for cand in candidates if not any((r, c) in forbidden_blue_cells for r, c in cand["blue"])
        ]

    if not candidates:
        return None
    if thickness == 1 and not allow_terminal:
        single_green = [cand for cand in candidates if len(cand["spec"]["green"]) == 1]
        if single_green:
            candidates = single_green
    return random.choice(candidates)


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[r, c] = True
    for r, c in blob["blue"]:
        blue_mask[r, c] = True


def _reachable_chain_len(ray_limit: int) -> int:
    if ray_limit <= 5:
        return random.choices([1, 2], weights=[2, 3])[0]
    return random.choices([1, 2], weights=[1, 5])[0]


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    start_h = random.randint(START_H_MIN, START_H_MAX)
    ray_limit = start_h
    reachable_len = _reachable_chain_len(ray_limit)
    if reachable_len > MAX_REACHABLE_CHAIN:
        return None

    cap_r = H - start_h - 1
    start_lo = START_COL_PAD
    start_hi = W - START_COL_PAD - 1
    if reachable_len == 2:
        mid_pad = max(START_COL_PAD, W // 4)
        start_lo = max(start_lo, mid_pad)
        start_hi = min(start_hi, W - mid_pad - 1)
    if start_lo > start_hi:
        return None
    start_col = random.randint(start_lo, start_hi)
    inp[cap_r, start_col] = YELLOW
    out[cap_r, start_col] = RED
    ray_mask[cap_r, start_col] = True
    occupied[cap_r, start_col] = True
    for r in range(cap_r + 1, H):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        occupied[r, start_col] = True

    starts = [(cap_r, start_col)]
    direction = UP
    first_leg = True
    leg_plan: list[dict] = []
    if reachable_len == 2:
        leg_plan = [
            {"step_lo": 1, "step_hi": max(1, ray_limit - 2), "allowed_port_dirs": (UP,)},
            {"step_lo": 1, "step_hi": ray_limit, "allowed_port_dirs": (LEFT, RIGHT)},
        ]
    else:
        leg_plan = [{"step_lo": 1, "step_hi": ray_limit, "allowed_port_dirs": None}]

    for leg_cfg in leg_plan:
        blob = _candidate_blob(
            H,
            W,
            starts,
            direction,
            first_leg,
            occupied,
            blue_mask,
            ray_mask,
            step_lo=int(leg_cfg["step_lo"]),
            step_hi=int(leg_cfg["step_hi"]),
            allow_terminal=False,
            allowed_port_dirs=leg_cfg["allowed_port_dirs"],
        )
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["blue"], RED)
        starts = blob["emit_starts"]
        direction = blob["spec"]["port_dir"]
        first_leg = False

    overflow_hi = min(ray_limit + OVERFLOW_EXTRA_MAX, H + W)
    final_path = _path_with_limit(starts, direction, include_origin=True, limit=ray_limit, H=H, W=W)
    overflow_blob = _candidate_blob(
        H,
        W,
        starts,
        direction,
        first_leg,
        occupied,
        blue_mask,
        ray_mask,
        step_lo=ray_limit + 1,
        step_hi=overflow_hi,
        allow_terminal=True,
        forbidden_blue_cells=set(final_path),
    )
    if overflow_blob is None:
        return None
    _place_blob(inp, occupied, blue_mask, overflow_blob)

    if any(blue_mask[r, c] for r, c in final_path):
        return None
    _mark_ray(out, ray_mask, final_path)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
