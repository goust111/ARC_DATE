# Variant of 5961cc34
# Dimension: D3 — 射线宽度规则 (path)
# Change: 起始射线宽度为1像素；后续射线宽度等于端口绿色像素数（1或2像素），即端口条的长度决定射线粗细... → 宽度由色块大小决定

from __future__ import annotations

# Design analysis for the D3 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - Keep the same objects: azure background, blue diamond/octagon blobs, one green
#      port mark on exactly one side, and a yellow-over-red start marker at the bottom.
#    - The only rule change is how a blob emits after activation:
#      old rule: width = number of green port cells (1 or 2).
#      new rule: width = blob span perpendicular to motion.
#      For LEFT/RIGHT beams this is the blob height in rows; for UP/DOWN beams this is
#      the blob width in columns. The beam still travels straight in the green-marked
#      direction.
#    - To make that geometric rule concrete on the pixel grid, each beam lane starts
#      from the furthest blue cell in its own row/column on the emitting side. A 3x4
#      blob firing right therefore launches one lane from each of its 3 rows.
#    - Example trace:
#      * Original: a 5x6 blob with a 1-pixel green tip firing right would emit a
#        1-lane ray because only one green cell exists.
#      * Variant: the same blob emits a 5-lane ray because its silhouette occupies
#        5 rows; the green tip still only says "fire right".
#      * Difference: solver must infer width from blob shape, and generator must place
#        later blobs against those wide multi-lane rays rather than against a 1-2 lane
#        port bar.
# b) Complexity calibration
#    - Port lengths remain 1 or 2 so direction cues stay familiar, while actual widths
#      become 3-8 depending on blob size/orientation. This keeps the variant visibly
#      distinct without adding extra object types or colors.
#    - Width is still drawn from a small repeated set of values inherited from the
#      shape catalogue, so examples support pattern induction rather than arbitrary math.
# c) Implicit-assumption audit against the original solver
#    - Original STEP 4/extend_thick assumed "one ray lane per port-adjacent blue cell".
#      That is no longer valid: those blue cells are only enough to recover direction.
#    - Original generator used `_edge_offsets(..., entry_side)` because entry lanes lived
#      on a single extreme line. Under the variant, the source/entry lanes are stepped
#      boundary cells spanning all rows/columns, so placement logic must align general
#      frontier windows instead of one flat edge bar.
#    - Original activation painted only blue cells red and relied on the emitted ray to
#      recolor green cells. That remains safe here because every activated blob still
#      emits immediately, and at least one lane passes through each green port cell.
# d) Placement legality
#    - Generator still works output-first: it lays down rays and chain blobs first, then
#      derives the input and validates it with `main`.
#    - A chain blob is only accepted if:
#      * its full footprint is in bounds,
#      * it stays 8-neighbor separated from other blobs,
#      * the traced incoming lanes reach its chosen entry frontier cells without touching
#        any earlier blue body,
#      * its outgoing direction differs from the incoming side so the cascade continues.
# e) Visual observability
#    - The rule is visible because every activated blob with any outgoing beam produces
#      a thickness of at least 3, while the green port still has length only 1 or 2.
#      Across 2-4 examples a human can see that beam thickness tracks blob size, not the
#      green mark length.
# f) Visual cleanliness
#    - Only one beam family is active at a time in the constructed chain, rays stay
#      axis-aligned, distractors are kept off all ray paths, and 8-neighbor spacing is
#      preserved between blobs. This avoids ambiguous overlaps.
# g) Color-change inferability
#    - No new color mapping is introduced. Green still marks direction, blue still marks
#      inactive bodies, and red still means activated body or traversed ray.

from collections import Counter
import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


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
CHAIN_LEN_WEIGHTS = [1, 3, 3, 3]
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 500


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return rc[1] if d[0] else rc[0]


def _along_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return rc[0] if d[0] else rc[1]


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    return sorted(coords, key=lambda p: _perp_value(p, d))


def _lane_frontier(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    """One source/hit cell per perpendicular lane, using the furthest cell in direction d."""
    buckets: dict[int, list[int]] = {}
    for r, c in coords:
        key = int(c) if d[0] else int(r)
        val = int(r) if d[0] else int(c)
        buckets.setdefault(key, []).append(val)

    out: list[tuple[int, int]] = []
    for key in sorted(buckets):
        vals = buckets[key]
        chosen = min(vals) if d in (UP, LEFT) else max(vals)
        if d[0]:
            out.append((int(chosen), int(key)))
        else:
            out.append((int(key), int(chosen)))
    return _sort_on_ray(out, d)


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
            cells.append((int(r), int(c)))
        guard = H + W + 10
        while (r, c) != (er, ec):
            r, c = int(r + dr), int(c + dc)
            guard -= 1
            if guard < 0 or not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((int(r), int(c)))
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
            cells.append((int(r), int(c)))
        while True:
            r, c = int(r + dr), int(c + dc)
            if not (0 <= r < H and 0 <= c < W):
                break
            cells.append((int(r), int(c)))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


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

    blue = [(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) not in green_set]
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    emit_starts = _lane_frontier(blue, port_dir)
    if not emit_starts:
        return None

    entry_fronts = {d: _lane_frontier(blue, d) for d in DIRS}
    if any(not front for front in entry_fronts.values()):
        return None

    return {
        "size": (h, w),
        "mask": mask,
        "patch": patch,
        "blue": blue,
        "full": full,
        "port_dir": port_dir,
        "emit_starts": emit_starts,
        "entry_fronts": entry_fronts,
        "green": green,
    }


def _all_specs(port_dir: tuple[int, int] | None = None) -> list[dict]:
    dirs = [port_dir] if port_dir is not None else list(DIRS)
    out = [s for size in SHAPES for d in dirs if (s := _blob_spec(size, d)) is not None]
    random.shuffle(out)
    return out


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
    """Find all placements whose stepped entry frontier matches the incoming lanes."""
    h, w = spec["mask"].shape
    is_vertical = bool(d[0])

    if is_vertical:
        c0 = lane_vals[0] - _perp_value(window[0], d)
        if any(c0 + c != lane for (_, c), lane in zip(window, lane_vals)):
            return []
        varying = list(range(H - h + 1))
    else:
        r0 = lane_vals[0] - _perp_value(window[0], d)
        if any(r0 + r != lane for (r, _), lane in zip(window, lane_vals)):
            return []
        varying = list(range(W - w + 1))

    random.shuffle(varying)
    candidates: list[dict] = []

    for origin_along in varying:
        r0 = origin_along if is_vertical else r0
        c0 = c0 if is_vertical else origin_along

        full = _offset(spec["full"], r0, c0)
        if not _in_bounds(full, H, W):
            continue
        if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
            continue

        entry_cells = _offset(window, r0, c0)
        ok_gap = True
        for start, entry in zip(starts, entry_cells):
            delta = _along_value(entry, d) - _along_value(start, d)
            if d in (UP, LEFT):
                if delta > -min_gap:
                    ok_gap = False
                    break
            else:
                if delta < min_gap:
                    ok_gap = False
                    break
        if not ok_gap:
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
                "r0": int(r0),
                "c0": int(c0),
                "entry": entry_cells,
                "path": path,
                "blue": _offset(spec["blue"], r0, c0),
                "full": full,
                "emit_starts": _sort_on_ray(_offset(spec["emit_starts"], r0, c0), spec["port_dir"]),
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
        edge = spec["entry_fronts"][entry_side]
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red beam cascade whose outgoing widths are determined by blob size.

    Input structure:
    - The background is the majority color; blue diamond/octagon blobs carry a short green mark on one side.
    - A yellow cap sitting above a short red bottom segment marks the unique starting launcher.

    Rule:
    - Fire the starting beam straight upward and activate the first blue blob it reaches.
    - Each activated blob turns red and emits in its green-marked direction.
    - The emitted width equals the blob's full span perpendicular to travel, so the cascade continues with shape-sized beams until no new blob is hit.

    Key insight:
    - Green still tells you which way a blob fires, but the blob silhouette itself determines how many parallel ray lanes leave that side.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the start marker (red vertical segment with a non-red cap above) ---
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

    # --- STEP 2) Identify blob/body colors ---
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

    # --- STEP 3) Extract full blobs (blue+green), keeping a blue-pixel activation map ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []

    full_mask = (g == int(blue_col)) | (g == int(green_col))
    for coords in components_mask(full_mask, connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        blue_pts = [(r, c) for r, c in pts if int(g[r, c]) == int(blue_col)]
        green_pts = [(r, c) for r, c in pts if int(g[r, c]) == int(green_col)]
        assert blue_pts and green_pts

        seen_dirs: set[tuple[int, int]] = set()
        for r, c in blue_pts:
            for ddr, ddc in DIRS:
                rr = int(r + ddr)
                cc = int(c + ddc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(green_col):
                    seen_dirs.add((int(ddr), int(ddc)))
        assert len(seen_dirs) == 1
        port_dir = next(iter(seen_dirs))
        emit_starts = _lane_frontier(blue_pts, port_dir)
        assert emit_starts

        bi = int(len(blobs))
        for r, c in blue_pts:
            blob_id[r, c] = bi

        blobs.append(
            {
                "blue": blue_pts,
                "green": green_pts,
                "emit_starts": emit_starts,
                "port_dir": port_dir,
            }
        )

    activated: set[int] = set()
    queue: list[int] = []

    def activate_blob_at(r: int, c: int) -> None:
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))
        for pr, pc in blobs[bi]["blue"]:
            out[int(pr), int(pc)] = int(red_col)

    # --- STEP 4) Draw the thin start beam upward and stop at the first blob hit ---
    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    top_r = min(int(r) for r, _c in start_seg)
    assert int(top_r) > 0 and int(g[int(top_r) - 1, int(start_col)]) == int(yellow_col)
    out[int(top_r) - 1, int(start_col)] = int(red_col)
    for r in range(int(top_r) - 1, -1, -1):
        out[int(r), int(start_col)] = int(red_col)
        if int(g[int(r), int(start_col)]) == int(blue_col):
            activate_blob_at(int(r), int(start_col))
            break

    def extend_shape_beam(source_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        """Emit one lane per row/column spanned by the blob on the firing side."""
        ddr, ddc = map(int, d)
        for sr, sc in source_pixels:
            r, c = int(sr), int(sc)
            while True:
                r += int(ddr)
                c += int(ddc)
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[int(r), int(c)] = int(red_col)
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c))
                    break

    # --- STEP 5) Propagate the BFS cascade using size-derived beam widths ---
    while queue:
        bi = int(queue.pop(0))
        extend_shape_beam(blobs[bi]["emit_starts"], blobs[bi]["port_dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"

    return out


def _gen_attempt() -> np.ndarray | None:
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

    chain_len = random.choices(range(len(CHAIN_LEN_WEIGHTS)), weights=CHAIN_LEN_WEIGHTS)[0]
    min_dist = 1 if chain_len == 0 else 0
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(min_dist, max_dist)

    starts = [(int(cap_r), int(start_col))]
    direction = UP
    include_origin = True
    for _ in range(chain_len):
        blob = _candidate_next_blob(H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask)
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["blue"], RED)
        starts = blob["emit_starts"]
        direction = blob["spec"]["port_dir"]
        include_origin = False

    final_path = _path_to_boundary(starts, direction, include_origin, H, W)
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
    raise RuntimeError("failed to generate a valid 5961cc34 input")
