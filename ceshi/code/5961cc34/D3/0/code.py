# Variant of 5961cc34
# Dimension: D3 — 射线宽度规则 (path)
# Change: 起始射线宽度为1像素；后续射线宽度等于端口绿色像素数（1或2像素），即端口条的长度决定射线粗细... → 宽度累加

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste

# Design analysis for the D3 variant
# - Concrete behavior:
#   The input still shows blue blobs, a yellow-capped red launcher at the bottom,
#   and one green side-port per blob. The visual change is only in the output ray
#   widths: the launch from the bottom is width 1, the first activated blob emits
#   width 2, the next emits width 3, and so on. Green no longer sets thickness;
#   it only fixes the outgoing side and the centerline of the emission.
# - Behavior trace:
#   Original: start ray (1) hits blob A -> A emits width = len(green_A) (1 or 2).
#   Variant:  start ray (1) hits blob A -> A emits width 2 no matter whether
#   green_A has length 1 or 2; if that hits blob B, blob B emits width 3.
#   The difference appears exactly at the "emit from port" step.
# - Complexity calibration:
#   Width 3 must appear often enough to be observable, but the stock blob shapes
#   do not reliably accept width 4 incoming hits on their exposed edge lines.
#   To keep the task readable and legal, the chain length is capped at 2 blobs,
#   so the visible widths are typically 1->2 or 1->2->3.
# - Old implicit assumptions that break:
#   `extend_thick(port_pixels, d)` assumed one launch lane per green-backed blue
#   cell, so thickness equaled green length. That is no longer valid. The solver
#   now derives lanes from green cells plus BFS depth, and the queue carries the
#   outgoing width for each newly activated blob.
# - Placement legality:
#   The generator remains output-first/path-first. It places each next blob so
#   the current strip of lane cells reaches a matching window on the next blob's
#   incoming side. Outgoing widened mouths are checked to stay in-bounds and not
#   overlap prior objects/rays except the emitting blob's own green cells.
# - Visual observability:
#   Samples frequently contain either width-3 final rays or width-2 rays emitted
#   from a 1-pixel green port, so the new rule cannot collapse back to "green
#   length = width". Port lengths are still only 1 or 2, giving repeated support
#   that direction is encoded by green while thickness follows cascade depth.
# - New solver assumptions and generator guarantees:
#   The solver assumes each blob has a single contiguous green segment on one
#   extreme side, and it widens around that segment with a fixed left/up bias
#   when an odd number of extra lanes is needed. The generator constructs only
#   such ports and uses the same widening convention.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red cascade where each new emission grows one pixel wider than the previous one.

    Input structure:
    - The background is the dominant color; large blue blobs each have one contiguous green side-port.
    - A short bottom red post with a different cap above it marks the initial launcher column.

    Rule:
    - Fire a 1-pixel ray upward from the launcher and activate the first blue blob it reaches.
    - Every activated blob turns red and emits from its green-marked side; after each cascade step, the emitted ray width increases by 1.
    - Rays continue until they hit another blob or leave the grid; untouched blobs disappear into the background.

    Key insight: The green cells only indicate emission side and beam center; the actual beam thickness is determined by activation depth in the cascade.
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

    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []

    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = len(blobs)
        for r, c in pts:
            blob_id[r, c] = int(bi)
        blobs.append({"pts": pts, "ports": []})

    def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
        if d[0]:
            return sorted(coords, key=lambda p: (p[1], p[0]))
        return sorted(coords, key=lambda p: (p[0], p[1]))

    def _expanded_starts(greens: list[tuple[int, int]], d: tuple[int, int], width: int) -> list[tuple[int, int]]:
        greens = _sort_on_ray(greens, d)
        assert greens
        if d[0]:
            row = greens[0][0]
            assert all(r == row for r, _c in greens)
            vals = [c for _r, c in greens]
            span = vals[-1] - vals[0] + 1
            extra = int(width) - int(span)
            assert extra >= 0
            lo = vals[0] - (extra + 1) // 2
            hi = vals[-1] + extra // 2
            return [(int(row), int(c)) for c in range(int(lo), int(hi) + 1)]
        col = greens[0][1]
        assert all(c == col for _r, c in greens)
        vals = [r for r, _c in greens]
        span = vals[-1] - vals[0] + 1
        extra = int(width) - int(span)
        assert extra >= 0
        lo = vals[0] - (extra + 1) // 2
        hi = vals[-1] + extra // 2
        return [(int(r), int(col)) for r in range(int(lo), int(hi) + 1)]

    for blob in blobs:
        pts = blob["pts"]
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in pts:
            for ddr, ddc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + ddr
                cc = c + ddc
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == green_col:
                    seen_dirs.add((int(ddr), int(ddc)))
        for ddr, ddc in seen_dirs:
            port_pixels = []
            greens = []
            for r, c in pts:
                rr = r + ddr
                cc = c + ddc
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == green_col:
                    port_pixels.append((int(r), int(c)))
                    greens.append((int(rr), int(cc)))
            assert port_pixels
            greens = _sort_on_ray(sorted(set(greens)), (int(ddr), int(ddc)))
            blob["ports"].append({"dir": (int(ddr), int(ddc)), "greens": greens})

    activated: set[int] = set()
    queue: list[tuple[int, int]] = []

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
    out[top_r - 1, start_col] = int(red_col)
    for r in range(top_r - 1, -1, -1):
        out[int(r), int(start_col)] = int(red_col)
        if int(g[int(r), int(start_col)]) == int(blue_col):
            activate_blob_at(int(r), int(start_col), emit_width=2)
            break

    def extend_growing(greens: list[tuple[int, int]], d: tuple[int, int], width: int) -> None:
        starts = _expanded_starts(greens, d, width)
        ddr, ddc = map(int, d)
        for sr, sc in starts:
            if 0 <= sr < H and 0 <= sc < W:
                out[int(sr), int(sc)] = int(red_col)
            r, c = int(sr), int(sc)
            while True:
                r += ddr
                c += ddc
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[int(r), int(c)] = int(red_col)
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c), emit_width=width + 1)
                    break

    while queue:
        bi, emit_width = queue.pop(0)
        for port in blobs[int(bi)]["ports"]:
            extend_growing(port["greens"], port["dir"], int(emit_width))

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"

    return out


# --------------- generator (a2_cascade_d3_width_growth) ---------------

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
CHAIN_LEN_WEIGHTS = [0, 3, 7]  # chain_len in [0, 1, 2]
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

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
    if not green:
        return None
    if len(green) not in (1, 2):
        return None
    edge_set = set(edge)
    if any(pt not in edge_set for pt in green):
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


def _path_to_boundary(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        if include_origin:
            cells.append((r, c))
        while True:
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
        path = _path_to_entries(starts, d, entry_cells, include_origin=True, H=H, W=W)
        if path is None:
            continue
        entry_set = set(entry_cells)
        if any(blue_mask[r, c] for r, c in path if (r, c) not in entry_set):
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

    chain_len = random.choices(range(len(CHAIN_LEN_WEIGHTS)), weights=CHAIN_LEN_WEIGHTS)[0]
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
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["blue"], RED)
        starts = blob["emit_starts"]
        direction = blob["spec"]["port_dir"]
        first_leg = False

    final_path = _path_to_boundary(starts, direction, include_origin=True, H=H, W=W)
    if any(blue_mask[r, c] for r, c in final_path):
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
    raise RuntimeError("failed to generate a valid 5961cc34 D3 input")
