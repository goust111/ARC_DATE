# Combo variant of 5961cc34 (index 11)
# combo_spec: D3/0+D8/0
# Name: 宽度累加+间歇涂色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste

# Design analysis for the combo variant
# - Concrete behavior:
#   The input still shows one yellow-capped bottom launcher plus blue diamond/octagon
#   blobs with a single contiguous green side-port. The output change is entirely in
#   the red trajectories: the launch leg is width 1, the first activated blob emits
#   width 2, the next emits width 3, and every ray lane is painted intermittently as
#   paint-gap-paint-gap starting from the launcher/port itself.
# - Behavior trace:
#   Original: start(1, solid) -> blob A -> emit width = len(green_A), solid.
#   Combo:    start(1, dashed) -> blob A -> emit width 2, dashed.
#             If blob A has one green pixel, the first outgoing ray is already wider
#             than the port. If blob B is then hit, blob B emits width 3 even if its
#             green bar is only length 1 or 2. The dash cadence is independent of both
#             green length and BFS depth.
# - Complexity calibration:
#   Width 3 is enough to make "growth" visually obvious, while width 4 is hard to fit
#   cleanly into the stock blob silhouettes. The generator therefore caps the cascade
#   chain at two activated blobs, so samples usually show 1->2 or 1->2->3 widths and
#   long enough corridors to expose the alternating gaps.
# - Old implicit assumptions that break:
#   The original `extend_thick(port_pixels, d)` assumed thickness equals the number of
#   port-backed blue cells and that every traversed cell is painted. Both assumptions
#   fail here: width comes from cascade depth, and hit detection must keep walking over
#   skipped cells because a blob can be reached through a blank gap.
# - Placement legality:
#   The generator still builds the output path first, but it now tracks the full travel
#   corridor separately from the painted subset. Distractors and later blobs are banned
#   from all traversed cells, including skipped dash gaps, so no hidden overlap can sit
#   inside an apparently empty interval.
# - Visual observability:
#   Samples are biased toward at least one activated blob and often two. The first port
#   emission prefers a 1-pixel green bar, making a visible width-2-vs-port-1 mismatch.
#   Every emitted leg also contains at least one skipped background cell, so the dashed
#   mode cannot collapse back into the solid original.
# - New solver assumptions and generator guarantees:
#   The solver assumes each blob has one extreme-side green segment and that widening is
#   centered on that segment with the same fixed left/up bias used by the D3 reference.
#   The generator only constructs blobs that satisfy those assumptions and rejects any
#   sample where the width-growth or dash metrics are not actually present.


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

# --- Chain / distractor ---
TARGET_CHAIN_LENGTHS = (1, 2)
TARGET_CHAIN_WEIGHTS = (1, 2)
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 700


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return rc[1] if d[0] else rc[0]


def _along_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return rc[0] if d[0] else rc[1]


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    return sorted(coords, key=lambda p: (_perp_value(p, d), _along_value(p, d)))


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


def _shape_mask(h: int, w: int) -> np.ndarray:
    widths = _SHAPE_WIDTHS[(h, w)]
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(widths):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + r, c0 + c) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in coords)


def _infer_extreme_side(
    full_coords: list[tuple[int, int]],
    side_coords: list[tuple[int, int]],
) -> tuple[int, int] | None:
    rows = [int(r) for r, _c in full_coords]
    cols = [int(c) for _r, c in full_coords]
    side_rows = [int(r) for r, _c in side_coords]
    side_cols = [int(c) for _r, c in side_coords]
    candidates: list[tuple[int, int]] = []
    if all(r == min(rows) for r in side_rows):
        candidates.append(UP)
    if all(r == max(rows) for r in side_rows):
        candidates.append(DOWN)
    if all(c == min(cols) for c in side_cols):
        candidates.append(LEFT)
    if all(c == max(cols) for c in side_cols):
        candidates.append(RIGHT)
    if len(candidates) != 1:
        return None
    return candidates[0]


def _dilate8(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _lane_traces_to_entries(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    entries: list[tuple[int, int]],
    include_origin: bool,
    H: int,
    W: int,
) -> list[list[tuple[int, int]]] | None:
    dr, dc = d
    traces: list[list[tuple[int, int]]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        lane: list[tuple[int, int]] = []
        if include_origin:
            lane.append((int(r), int(c)))
        guard = H + W + 10
        while (r, c) != (er, ec):
            r, c = r + dr, c + dc
            guard -= 1
            if guard < 0 or not (0 <= r < H and 0 <= c < W):
                return None
            lane.append((int(r), int(c)))
        traces.append(lane)
    return traces


def _lane_traces_to_boundary(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    H: int,
    W: int,
) -> list[list[tuple[int, int]]]:
    dr, dc = d
    traces: list[list[tuple[int, int]]] = []
    for r, c in starts:
        lane: list[tuple[int, int]] = []
        if include_origin:
            lane.append((int(r), int(c)))
        while True:
            r, c = r + dr, c + dc
            if not (0 <= r < H and 0 <= c < W):
                break
            lane.append((int(r), int(c)))
        traces.append(lane)
    return traces


def _flatten_traces(traces: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    return [cell for lane in traces for cell in lane]


def _dashed_cells(traces: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    return [cell for lane in traces for step, cell in enumerate(lane) if step % 2 == 0]


def _mark_dashed_ray(out: np.ndarray, ray_mask: np.ndarray, traces: list[list[tuple[int, int]]]) -> None:
    paint_coords(ray_mask, _flatten_traces(traces), True)
    paint_coords(out, _dashed_cells(traces), RED)


def _emit_starts_global(spec: dict, r0: int, c0: int, width: int) -> list[tuple[int, int]]:
    local = _expanded_starts(spec["green"], spec["port_dir"], width)
    return [(r0 + r, c0 + c) for r, c in local]


def _solve(grid: np.ndarray, collect_stats: bool = False):
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
                (
                    int(col),
                    int(cap),
                    int(c0),
                    [(int(rr), int(c0)) for rr in range(int(r0), int(r1) + 1)],
                )
            )

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]

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
            for dr, dc in DIRS:
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
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []

    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = len(blobs)
        for r, c in pts:
            blob_id[r, c] = int(bi)
        blobs.append({"pts": pts, "ports": []})

    for blob in blobs:
        pts = blob["pts"]
        greens: set[tuple[int, int]] = set()
        for r, c in pts:
            for ddr, ddc in DIRS:
                rr = r + ddr
                cc = c + ddc
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == green_col:
                    greens.add((int(rr), int(cc)))
        assert greens
        green_list = sorted(greens)
        port_dir = _infer_extreme_side(pts + green_list, green_list)
        assert port_dir is not None
        blob["ports"].append({"dir": port_dir, "greens": _sort_on_ray(green_list, port_dir)})

    activated: set[int] = set()
    queue: list[tuple[int, int]] = []
    stats = {"legs": []} if collect_stats else None

    def note_leg(kind: str, width: int, green_len: int | None, painted: int, skipped: int) -> None:
        if stats is None:
            return
        stats["legs"].append(
            {
                "kind": str(kind),
                "width": int(width),
                "green_len": None if green_len is None else int(green_len),
                "painted": int(painted),
                "skipped": int(skipped),
            }
        )

    def activate_blob_at(r: int, c: int, emit_width: int) -> None:
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append((int(bi), int(emit_width)))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(red_col)

    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)

    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, start_col]) == int(yellow_col)

    painted = 0
    skipped = 0
    for step, r in enumerate(range(int(top_r) - 1, -1, -1)):
        rr = int(r)
        if step % 2 == 0:
            out[rr, int(start_col)] = int(red_col)
            painted += 1
        else:
            skipped += 1
        if int(g[rr, int(start_col)]) == int(blue_col):
            activate_blob_at(rr, int(start_col), emit_width=2)
            break
    note_leg("start", 1, None, painted, skipped)

    def extend_growing(greens: list[tuple[int, int]], d: tuple[int, int], width: int) -> None:
        starts = _expanded_starts(greens, d, width)
        ddr, ddc = map(int, d)
        leg_painted = 0
        leg_skipped = 0
        for sr, sc in starts:
            if not (0 <= sr < H and 0 <= sc < W):
                continue
            r, c = int(sr), int(sc)
            out[r, c] = int(red_col)
            leg_painted += 1
            step = 0
            while True:
                r += ddr
                c += ddc
                step += 1
                if not (0 <= r < H and 0 <= c < W):
                    break
                if step % 2 == 0:
                    out[int(r), int(c)] = int(red_col)
                    leg_painted += 1
                else:
                    leg_skipped += 1
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c), emit_width=width + 1)
                    break
        note_leg("port", width, len(greens), leg_painted, leg_skipped)

    while queue:
        bi, emit_width = queue.pop(0)
        for port in blobs[int(bi)]["ports"]:
            extend_growing(port["greens"], port["dir"], int(emit_width))

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    if collect_stats:
        stats["activated"] = int(len(activated))
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red ray cascade whose beams both widen with depth and blink as dashes.

    Input structure:
    - The dominant background surrounds blue diamond/octagon blobs, each carrying one contiguous green side-port.
    - A short red post touching the bottom has a single yellow cap above it, fixing the launch column.

    Rule:
    - Fire a 1-pixel ray upward from the cap, repainting the traversed trajectory in a paint-gap-paint-gap rhythm.
    - When a blue blob is reached, turn that blob red and emit a new dashed ray from its green side; each cascade level increases beam width by 1.
    - Continue breadth-first until every reachable blob has fired, then erase untouched blobs and all unused green markers to background.

    Key insight: Green still encodes direction, but not thickness; beam width follows cascade depth, while the visible red trajectory is an alternating dashed sampling of the full path.
    """
    return _solve(grid, collect_stats=False)


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
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if _infer_extreme_side(full, green) != port_dir:
        return None

    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    for r, c in green:
        patch[r, c] = GREEN

    green_set = set(green)
    blue = [(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) not in green_set]
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


def _try_place_along_axis(
    spec: dict,
    window: list[tuple[int, int]],
    d: tuple[int, int],
    lane_vals: list[int],
    starts: list[tuple[int, int]],
    first_leg: bool,
    occ_pad: np.ndarray,
    occupied: np.ndarray,
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
    min_gap = MIN_GAP_FIRST if first_leg else MIN_GAP_CHAIN
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

        traces = _lane_traces_to_entries(starts, d, entry_cells, include_origin=True, H=H, W=W)
        if traces is None:
            continue
        path_cells = _flatten_traces(traces)
        entry_set = set(entry_cells)
        if any(blue_mask[r, c] for r, c in path_cells if (r, c) not in entry_set):
            continue

        emit_starts = _emit_starts_global(spec, r0, c0, len(starts) + 1)
        if not _in_bounds(emit_starts, H, W):
            continue
        own_full = set(full)
        if any((r, c) not in own_full and (occupied[r, c] or ray_mask[r, c]) for r, c in emit_starts):
            continue

        candidates.append(
            {
                "spec": spec,
                "r0": r0,
                "c0": c0,
                "entry": entry_cells,
                "traces": traces,
                "blue": _offset(spec["blue"], r0, c0),
                "full": full,
                "emit_starts": emit_starts,
            }
        )

    return candidates


def _candidate_next_blob(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    first_leg: bool,
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
                    first_leg,
                    occ_pad,
                    occupied,
                    blue_mask,
                    ray_mask,
                    H,
                    W,
                )
            )

    if not candidates:
        return None
    if thickness == 1:
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


def characterize_variant(grid: np.ndarray) -> dict[str, int]:
    _, stats = _solve(grid, collect_stats=True)
    widths = [int(leg["width"]) for leg in stats["legs"]]
    mismatches = [
        1
        for leg in stats["legs"]
        if leg["kind"] == "port" and leg["green_len"] is not None and int(leg["width"]) != int(leg["green_len"])
    ]
    total_skipped = sum(int(leg["skipped"]) for leg in stats["legs"])
    return {
        "num_legs": int(len(stats["legs"])),
        "num_port_legs": int(sum(1 for leg in stats["legs"] if leg["kind"] == "port")),
        "max_width": int(max(widths) if widths else 0),
        "num_width_mismatches": int(len(mismatches)),
        "total_skipped_cells": int(total_skipped),
    }


def _gen_attempt(chain_len: int | None = None) -> np.ndarray | None:
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
    occupied[cap_r, start_col] = True
    for r in range(cap_r + 1, H):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        occupied[r, start_col] = True

    if chain_len is None:
        chain_len = random.choices(TARGET_CHAIN_LENGTHS, weights=TARGET_CHAIN_WEIGHTS)[0]
    min_dist = 1 if chain_len == 0 else 0
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(min_dist, max_dist)

    starts = [(cap_r, start_col)]
    direction = UP
    first_leg = True
    for _ in range(chain_len):
        blob = _candidate_next_blob(H, W, starts, direction, first_leg, occupied, blue_mask, ray_mask)
        if blob is None:
            return None
        _mark_dashed_ray(out, ray_mask, blob["traces"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["blue"], RED)
        starts = blob["emit_starts"]
        direction = blob["spec"]["port_dir"]
        first_leg = False

    final_traces = _lane_traces_to_boundary(starts, direction, include_origin=True, H=H, W=W)
    final_cells = _flatten_traces(final_traces)
    if any(blue_mask[r, c] for r, c in final_cells):
        return None
    _mark_dashed_ray(out, ray_mask, final_traces)

    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    solved, stats = _solve(inp, collect_stats=True)
    if not np.array_equal(solved, out):
        return None
    num_port_legs = int(sum(1 for leg in stats["legs"] if leg["kind"] == "port"))
    num_width_mismatches = int(
        sum(
            1
            for leg in stats["legs"]
            if leg["kind"] == "port" and leg["green_len"] is not None and int(leg["width"]) != int(leg["green_len"])
        )
    )
    total_skipped_cells = int(sum(int(leg["skipped"]) for leg in stats["legs"]))
    if num_port_legs < 1:
        return None
    if num_width_mismatches < 1:
        return None
    if total_skipped_cells < 3:
        return None
    return inp


def generate_input() -> np.ndarray:
    target_chain_len = random.choices(TARGET_CHAIN_LENGTHS, weights=TARGET_CHAIN_WEIGHTS)[0]
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(chain_len=target_chain_len)
        if grid is not None:
            return grid
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
