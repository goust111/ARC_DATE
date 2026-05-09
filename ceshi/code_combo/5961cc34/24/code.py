# Combo variant of 5961cc34 (index 24)
# combo_spec: D3/0+D7/2+D10/2
# Name: 宽度累加+深度颜色编码+长链

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste

# Design analysis for the combo variant
# a) Concrete behavior and trace
# - Input still has the same semantic parts: a bottom launcher made of a yellow cap
#   above a short red stem, plus several large blue blobs with one green edge-port
#   each. The changed behavior is entirely in how the cascade is rendered.
# - Example trace:
#   source task : start ray width 1 hits A -> A turns red -> A emits width set by
#   green length (1 or 2) -> B turns red -> ...
#   combo task  : start ray width 1 hits A -> A becomes red -> A emits width 2;
#   B becomes orange -> B emits width 3; C becomes yellow -> C emits width 4;
#   later blobs stay yellow while widths keep growing to 5, 6, 7.
# - So the same chain now carries two coupled signals: geometric depth is visible
#   both in the widening strip and in the warming blob color.
#
# b) Complexity calibration
# - Chain length is forced to 4-6 so width growth beyond the original 1-2 range is
#   always visible, and the output never collapses back to the source behavior.
# - The color rule follows the spec exactly: first hit red, second orange, third
#   and later yellow. On longer chains the width cue keeps increasing after the
#   color cue saturates, so both dimensions still contribute to the pattern.
# - Total blob count is raised to 5-7 by adding 0-3 distractors. This keeps the
#   reasoning multi-step without making the grid unreadable.
#
# c) Implicit-assumption audit
# - The source solver assumed green length equals emitted width. That breaks under
#   D3/0, so width must be derived from activation depth instead.
# - The source solver assumed all activated blobs become red. That breaks under D7,
#   so activation needs a depth-aware color map.
# - The source solver wrote ray color before checking for a blue hit. That would
#   overwrite later-depth blob entry cells, so hit detection must happen first.
# - The source solver assumed at most one distractor. D10/2 breaks that, so the
#   postcondition becomes "up to three unactivated blobs may remain".
# - The original small diamond/octagon catalog does not reliably accept incoming
#   widths 4-6 on every side. The generator therefore switches to larger symmetric
#   rectangles so every side can legally receive long, wide strips.
#
# d) Placement legality
# - Generation stays output-first/path-first. Each next blob is placed only if the
#   current strip reaches a matching window on its incoming side, the blob stays
#   in-bounds, and its widened outgoing mouth also stays in-bounds.
# - Occupied blobs are dilated before testing new placements, preserving at least
#   one background cell of separation. Distractors are placed only off all ray
#   paths, so they can never be accidentally activated.
#
# e) Visual observability
# - Humans can read the chain order directly from the launcher and ray geometry.
#   The first hit blob is the thinnest and warmest-start color, later blobs are
#   reached further along the unique cascade and sit on wider strips.
# - Because green ports stay length 1-2 while many outputs contain widths 4-7, the
#   rule "port length = beam thickness" is repeatedly contradicted by the samples.
#
# f) Visual cleanliness
# - Rays remain pure red and blob interiors carry the depth colors. This separates
#   "path" from "activated object" visually instead of stacking several meanings on
#   the same pixels.
# - Using larger rectangular blobs makes the long chain easier to parse than a set
#   of tiny shapes with high overlap pressure.
#
# g) Change inferability
# - The warm-color mapping is monotone and low-cardinality. A solver can infer that
#   color progresses from earlier to later activations while width increases one
#   step at a time. The generator guarantees repeated long chains so both cues are
#   present in most samples.


# --- Colors ---
BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
DEPTH_PALETTE = (RED, ORANGE, YELLOW)

# --- Directions ---
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

# --- Large symmetric blob sizes (rectangles) ---
SHAPES = ((6, 6), (6, 7), (7, 7), (7, 8), (8, 8), (8, 10), (9, 9))

# --- Canvas / chain ---
CHAIN_LENGTHS = (4, 5, 6)
CHAIN_LEN_WEIGHTS = (3, 4, 3)
TOTAL_BLOBS_MIN = 5
TOTAL_BLOBS_MAX = 7
MAX_DISTRACTORS = TOTAL_BLOBS_MAX - min(CHAIN_LENGTHS)
START_COL_PAD = 3
START_H_MIN, START_H_MAX = 2, 3
MIN_GAP_FIRST = 5
MIN_GAP_CHAIN = 4
GEN_RETRIES = 800


def _depth_color(depth: int) -> int:
    """Map cascade depth to a small warm palette, clamping once yellow is reached."""
    idx = min(max(int(depth) - 1, 0), len(DEPTH_PALETTE) - 1)
    return int(DEPTH_PALETTE[idx])


def _shape_mask(h: int, w: int) -> np.ndarray:
    """Large symmetric rectangles keep every edge wide enough for long incoming strips."""
    return np.ones((h, w), dtype=bool)


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


def _expanded_starts(coords: list[tuple[int, int]], d: tuple[int, int], width: int) -> list[tuple[int, int]]:
    """Expand a 1-2 pixel green port into the depth-controlled outgoing strip."""
    coords = _sort_on_ray(coords, d)
    assert coords
    if d[0]:
        row = int(coords[0][0])
        vals = [int(c) for _r, c in coords]
        assert all(int(r) == row for r, _c in coords)
        span = vals[-1] - vals[0] + 1
        extra = int(width) - int(span)
        assert extra >= 0
        lo = vals[0] - (extra + 1) // 2
        hi = vals[-1] + extra // 2
        return [(row, int(c)) for c in range(int(lo), int(hi) + 1)]
    col = int(coords[0][1])
    vals = [int(r) for r, _c in coords]
    assert all(int(c) == col for _r, c in coords)
    span = vals[-1] - vals[0] + 1
    extra = int(width) - int(span)
    assert extra >= 0
    lo = vals[0] - (extra + 1) // 2
    hi = vals[-1] + extra // 2
    return [(int(r), col) for r in range(int(lo), int(hi) + 1)]


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

    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    for r, c in green:
        patch[int(r), int(c)] = GREEN

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


def _centered_green_windows(edge: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Use one centered 1-cell port and one centered 2-cell port to keep the spec set compact."""
    edge = list(edge)
    assert edge
    n = len(edge)
    out = [[edge[n // 2]]]
    if n >= 2:
        k = (n - 2) // 2
        out.append(edge[k : k + 2])
    return out


def _build_all_specs() -> list[dict]:
    out: list[dict] = []
    for size in SHAPES:
        mask = _shape_mask(*size)
        for d in DIRS:
            edge = _edge_offsets(mask, d)
            for green in _centered_green_windows(edge):
                spec = _blob_spec(size, d, green)
                if spec is not None:
                    out.append(spec)
    return out


ALL_SPECS = _build_all_specs()
SPECS_BY_DIR = {d: [spec for spec in ALL_SPECS if spec["port_dir"] == d] for d in DIRS}


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= int(r) < H and 0 <= int(c) < W for r, c in coords)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _path_to_entries(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    entries: list[tuple[int, int]],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    dr, dc = map(int, d)
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        r = int(r)
        c = int(c)
        er = int(er)
        ec = int(ec)
        if include_origin:
            cells.append((r, c))
        guard = H + W + 10
        while (r, c) != (er, ec):
            r += dr
            c += dc
            guard -= 1
            if guard < 0 or not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((r, c))
    return cells


def _path_to_boundary(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    dr, dc = map(int, d)
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        r = int(r)
        c = int(c)
        if include_origin:
            cells.append((r, c))
        while True:
            r += dr
            c += dc
            if not (0 <= r < H and 0 <= c < W):
                break
            cells.append((r, c))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


def _mark_ray_preserving_blue(
    out: np.ndarray,
    ray_mask: np.ndarray,
    blue_mask: np.ndarray,
    cells: list[tuple[int, int]],
) -> None:
    safe = [(int(r), int(c)) for r, c in cells if not bool(blue_mask[int(r), int(c)])]
    paint_coords(out, safe, RED)
    paint_coords(ray_mask, cells, True)


def _centered_entry_window(mask: np.ndarray, side: tuple[int, int], thickness: int) -> list[tuple[int, int]] | None:
    edge = _edge_offsets(mask, side)
    if len(edge) < thickness:
        return None
    k = (len(edge) - thickness) // 2
    return edge[k : k + thickness]


def _emit_starts_global(spec: dict, r0: int, c0: int, width: int) -> list[tuple[int, int]]:
    local = _expanded_starts(spec["green"], spec["port_dir"], width)
    return [(int(r0 + r), int(c0 + c)) for r, c in local]


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

    fixed = int(lane_vals[0] - win_vals[0])
    extent = H - h if is_vertical else W - w
    low = int(edge_line)
    high = int(extent + edge_line)
    min_gap = MIN_GAP_FIRST if first_leg else MIN_GAP_CHAIN
    if d in (UP, LEFT):
        high = min(high, int(start_line - min_gap))
    else:
        low = max(low, int(start_line + min_gap))
    if low > high:
        return []

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
        if any(bool(occ_pad[r, c] or ray_mask[r, c]) for r, c in full):
            continue
        path = _path_to_entries(starts, d, entry_cells, include_origin=True, H=H, W=W)
        if path is None:
            continue
        start_set = set(starts)
        entry_set = set(entry_cells)
        if any(bool(blue_mask[r, c]) for r, c in path if (r, c) not in entry_set and (r, c) not in start_set):
            continue

        emit_starts = _emit_starts_global(spec, r0, c0, len(starts) + 1)
        if not _in_bounds(emit_starts, H, W):
            continue
        own_full = set(full)
        if any((r, c) not in own_full and bool(occupied[r, c] or ray_mask[r, c]) for r, c in emit_starts):
            continue

        candidates.append(
            {
                "spec": spec,
                "r0": r0,
                "c0": c0,
                "path": path,
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
    lane_delta = [int(v - lane_vals[0]) for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    specs = ALL_SPECS[:]
    random.shuffle(specs)
    candidates: list[dict] = []

    for spec in specs:
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
        if candidates:
            break

    return random.choice(candidates) if candidates else None


def _direction_plan(chain_len: int) -> list[tuple[int, int]]:
    base = [RIGHT, UP, LEFT, UP, RIGHT, UP]
    return base[: int(chain_len)]


def _planned_next_blob(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    incoming_dir: tuple[int, int],
    outgoing_dir: tuple[int, int],
    first_leg: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> dict | None:
    starts = _sort_on_ray(starts, incoming_dir)
    thickness = len(starts)
    lane_vals = [_perp_value(p, incoming_dir) for p in starts]
    lane_delta = [int(v - lane_vals[0]) for v in lane_vals]
    entry_side = OPP[incoming_dir]
    occ_pad = _dilate8(occupied)
    specs = SPECS_BY_DIR[outgoing_dir][:]
    random.shuffle(specs)
    gap_lo = MIN_GAP_FIRST if first_leg else MIN_GAP_CHAIN
    gap_choices = list(range(gap_lo, gap_lo + 4))

    for spec in specs:
        window = _centered_entry_window(spec["mask"], entry_side, thickness)
        if window is None:
            continue
        win_delta = [_perp_value(p, incoming_dir) - _perp_value(window[0], incoming_dir) for p in window]
        if win_delta != lane_delta:
            continue
        random.shuffle(gap_choices)
        for gap in gap_choices:
            edge_line = _along_value(window[0], incoming_dir)
            start_line = _along_value(starts[0], incoming_dir)
            if incoming_dir in (UP, LEFT):
                entry_line = int(start_line - gap - 1)
            else:
                entry_line = int(start_line + gap + 1)
            origin_along = int(entry_line - edge_line)
            fixed = int(lane_vals[0] - _perp_value(window[0], incoming_dir))
            if incoming_dir[0]:
                r0 = origin_along
                c0 = fixed
                entry_cells = [(int(entry_line), int(c0 + c)) for _, c in window]
            else:
                r0 = fixed
                c0 = origin_along
                entry_cells = [(int(r0 + r), int(entry_line)) for r, _ in window]
            full = _offset(spec["full"], r0, c0)
            if not _in_bounds(full, H, W):
                continue
            if any(bool(occ_pad[r, c] or ray_mask[r, c]) for r, c in full):
                continue
            path = _path_to_entries(starts, incoming_dir, entry_cells, include_origin=True, H=H, W=W)
            if path is None:
                continue
            start_set = set(starts)
            entry_set = set(entry_cells)
            if any(bool(blue_mask[r, c]) for r, c in path if (r, c) not in entry_set and (r, c) not in start_set):
                continue
            emit_starts = _emit_starts_global(spec, r0, c0, thickness + 1)
            if not _in_bounds(emit_starts, H, W):
                continue
            own_full = set(full)
            if any((r, c) not in own_full and bool(occupied[r, c] or ray_mask[r, c]) for r, c in emit_starts):
                continue
            return {
                "spec": spec,
                "r0": r0,
                "c0": c0,
                "path": path,
                "blue": _offset(spec["blue"], r0, c0),
                "full": full,
                "emit_starts": emit_starts,
            }
    return None


def _planned_final_blob(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    incoming_dir: tuple[int, int],
    first_leg: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> tuple[dict, tuple[int, int]] | None:
    preferred = [UP, RIGHT, LEFT, DOWN]
    entry_side = OPP[incoming_dir]
    dirs = [d for d in preferred if d != entry_side]
    for outgoing_dir in dirs:
        blob = _planned_next_blob(H, W, starts, incoming_dir, outgoing_dir, first_leg, occupied, blue_mask, ray_mask)
        if blob is None:
            continue
        final_path = _path_to_boundary(blob["emit_starts"], outgoing_dir, include_origin=True, H=H, W=W)
        start_set = set(blob["emit_starts"])
        if any(bool(blue_mask[r, c]) for r, c in final_path if (r, c) not in start_set):
            continue
        return blob, outgoing_dir
    return None


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
    specs = ALL_SPECS[:]
    random.shuffle(specs)
    for _ in range(300):
        spec = random.choice(specs)
        h, w = spec["mask"].shape
        r0 = random.randint(0, H - h)
        c0 = random.randint(0, W - w)
        full = _offset(spec["full"], r0, c0)
        if any(bool(occ_pad[r, c] or ray_mask[r, c]) for r, c in full):
            continue
        blob = {
            "spec": spec,
            "r0": r0,
            "c0": c0,
            "full": full,
            "blue": _offset(spec["blue"], r0, c0),
        }
        _place_blob(inp, occupied, blue_mask, blob)
        return True
    return False


def _solve_trace(grid: np.ndarray) -> dict:
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
            if c0 != c1 or (r1 - r0 + 1) < 2 or r1 != H - 1:
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
                    [(int(rr), int(c0)) for rr in range(r0, r1 + 1)],
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

    out = np.full_like(g, int(bg))
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[int(r), int(c)] = bi
        blobs.append({"pts": pts, "ports": []})

    for blob in blobs:
        pts = blob["pts"]
        pt_set = set(pts)
        greens: set[tuple[int, int]] = set()
        for r, c in pts:
            for ddr, ddc in DIRS:
                rr = int(r + ddr)
                cc = int(c + ddc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(green_col):
                    greens.add((int(rr), int(cc)))
        if not greens:
            continue
        rows = {int(r) for r, _c in greens}
        cols = {int(c) for _r, c in greens}
        blue_rows = [int(r) for r, _c in pts]
        blue_cols = [int(c) for _r, c in pts]
        r0 = min(blue_rows)
        r1 = max(blue_rows)
        c0 = min(blue_cols)
        c1 = max(blue_cols)
        if len(rows) == 1 and next(iter(rows)) == r0:
            port_dir = UP
        elif len(rows) == 1 and next(iter(rows)) == r1:
            port_dir = DOWN
        elif len(cols) == 1 and next(iter(cols)) == c0:
            port_dir = LEFT
        elif len(cols) == 1 and next(iter(cols)) == c1:
            port_dir = RIGHT
        else:
            raise AssertionError("green port must occupy one bbox side")
        greens_list = _sort_on_ray(sorted(greens), port_dir)
        assert len(greens_list) in (1, 2)
        for gr, gc in greens_list:
            br = int(gr - port_dir[0])
            bc = int(gc - port_dir[1])
            assert (br, bc) in pt_set
        blob["ports"].append({"dir": port_dir, "greens": greens_list})

    activated: set[int] = set()
    activation_depth: dict[int, int] = {}
    queue: deque[tuple[int, int]] = deque()

    def activate_blob_at(r: int, c: int, depth: int) -> None:
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        activation_depth[int(bi)] = int(depth)
        queue.append((int(bi), int(depth)))
        fill = _depth_color(int(depth))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(fill)

    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, start_col]) == int(yellow_col)
    out[int(top_r) - 1, int(start_col)] = int(red_col)
    for r in range(int(top_r) - 1, -1, -1):
        if int(g[int(r), int(start_col)]) == int(blue_col):
            activate_blob_at(int(r), int(start_col), depth=1)
            break
        out[int(r), int(start_col)] = int(red_col)

    def extend_growing(greens: list[tuple[int, int]], d: tuple[int, int], emitter_depth: int) -> None:
        starts = _expanded_starts(greens, d, emitter_depth + 1)
        ddr, ddc = map(int, d)
        for sr, sc in starts:
            if 0 <= sr < H and 0 <= sc < W and int(g[int(sr), int(sc)]) != int(blue_col):
                out[int(sr), int(sc)] = int(red_col)
            r = int(sr)
            c = int(sc)
            while True:
                r += ddr
                c += ddc
                if not (0 <= r < H and 0 <= c < W):
                    break
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c), emitter_depth + 1)
                    break
                out[int(r), int(c)] = int(red_col)

    while queue:
        bi, depth = queue.popleft()
        for port in blobs[int(bi)]["ports"]:
            extend_growing(port["greens"], port["dir"], int(depth))

    assert len(blobs) - len(activated) <= MAX_DISTRACTORS
    return {
        "out": out,
        "num_blobs": len(blobs),
        "num_activated": len(activated),
        "max_depth": max(activation_depth.values(), default=0),
        "activation_depth": activation_depth,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a long warm-colored cascade whose rays widen by one pixel at every activation step.

    Input structure:
    - The background is dominant. Several large blue blobs each have one short green side-port.
    - A yellow cap above a short red bottom stem marks the launcher column.

    Rule:
    - Fire a 1-pixel start ray upward until it hits the first blue blob.
    - Every activated blob recolors by cascade depth and emits again from its green side; the emitted width is depth+1, so the chain grows from 1-wide to 2-wide to 3-wide and so on.
    - Rays stay red, while blob bodies warm from red to yellow with depth; untouched blobs and unused green cells disappear into the background.

    Key insight: Green marks only the outgoing side and local centerline. The two meaningful changing signals are activation depth: it controls both the beam width and the blob's warm color along the long chain.
    """
    return _solve_trace(np.array(grid, dtype=int))["out"]


def _gen_attempt() -> np.ndarray | None:
    chain_len = random.choices(CHAIN_LENGTHS, weights=CHAIN_LEN_WEIGHTS)[0]
    base = 24 + 3 * chain_len
    H = random.randint(base, base + 5)
    W = random.randint(base, base + 5)

    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = H - start_h - 1
    mid = W // 2
    lo = max(START_COL_PAD, mid - 2)
    hi = min(W - START_COL_PAD - 1, mid + 2)
    start_col = random.randint(lo, hi)
    inp[int(cap_r), int(start_col)] = YELLOW
    out[int(cap_r), int(start_col)] = RED
    ray_mask[int(cap_r), int(start_col)] = True
    occupied[int(cap_r), int(start_col)] = True
    for r in range(int(cap_r) + 1, H):
        inp[int(r), int(start_col)] = RED
        out[int(r), int(start_col)] = RED
        occupied[int(r), int(start_col)] = True

    total_options = list(range(max(TOTAL_BLOBS_MIN, chain_len), TOTAL_BLOBS_MAX + 1))
    total_weights = list(range(len(total_options), 0, -1))
    total_blobs = random.choices(total_options, weights=total_weights)[0]
    num_distractors = int(total_blobs - chain_len)

    starts = [(int(cap_r), int(start_col))]
    direction = UP
    first_leg = True
    plan = _direction_plan(chain_len)
    for depth, next_dir in enumerate(plan, start=1):
        if depth == chain_len:
            final_choice = _planned_final_blob(H, W, starts, direction, first_leg, occupied, blue_mask, ray_mask)
            if final_choice is None:
                return None
            blob, next_dir = final_choice
        else:
            blob = _planned_next_blob(H, W, starts, direction, next_dir, first_leg, occupied, blue_mask, ray_mask)
        if blob is None:
            return None
        _mark_ray_preserving_blue(out, ray_mask, blue_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["blue"], _depth_color(depth))
        starts = blob["emit_starts"]
        direction = next_dir
        first_leg = False

    final_path = _path_to_boundary(starts, direction, include_origin=True, H=H, W=W)
    start_set = set(starts)
    if any(bool(blue_mask[r, c]) for r, c in final_path if (r, c) not in start_set):
        return None
    _mark_ray_preserving_blue(out, ray_mask, blue_mask, final_path)

    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    trace = _solve_trace(inp)
    if trace["max_depth"] != chain_len:
        return None
    if trace["num_blobs"] != total_blobs:
        return None
    if not np.array_equal(trace["out"], out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
