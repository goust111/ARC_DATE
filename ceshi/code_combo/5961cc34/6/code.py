# Combo variant of 5961cc34 (index 6)
# combo_spec: D7/2+D8/2
# Name: 深度颜色编码+反射时变色轨迹

from __future__ import annotations

# Design analysis for the combo D7/2 + D8/2
# ------------------------------------------
# a) Concrete behavior and trace
# - Input geometry stays exactly like the source task: a bottom start marker
#   (yellow cap above a short red stem), blue diamond/octagon blobs, and one
#   green port side per blob.
# - Output changes in one fused way rather than two independent ones:
#   the warm color of a ray segment and the blob it next activates are unified
#   into one stage code.
#   * start ray before any hit: red
#   * first hit blob: red
#   * segment emitted after passing that blob: orange
#   * second hit blob: orange
#   * next emitted segment: yellow
#   * third hit blob: yellow
#   * any terminal segment after the third blob stays yellow instead of adding
#     a fourth hue, so the whole puzzle still teaches one 3-color depth code.
# - Compared with the source task, the cascade geometry is unchanged; only the
#   recoloring logic now depends on traversal stage.
#
# b) Complexity calibration
# - The visible palette is limited to three warm colors: red, orange, yellow.
#   This matches the original generator's maximum of three activated blobs and
#   avoids a hidden fourth stage color that would be hard to justify from only
#   2-4 demonstrations.
# - Samples are biased toward chain lengths 2-3 so both blob-depth changes and
#   ray-stage changes appear often enough to be learnable.
# - Chain length 0 remains possible but rare; otherwise the variant would risk
#   collapsing into a fixed template rather than a structural rule.
#
# c) Implicit-assumption audit of the original solver
# - Original assumption: every painted active structure is red. Broken here.
#   Fix: centralize warm-color selection in `_stage_color(stage)`.
# - Original assumption: a ray cell can be painted before checking whether the
#   input cell is blue. Broken here because that would overwrite the entry cell
#   of a depth-2/depth-3 blob with the previous segment color. Fix: test for a
#   blue hit first, then recolor the whole blob in `activate_blob_at(...)`.
# - Original queue stored only blob ids. Broken here because each blob must emit
#   in the next traversal stage. Fix: queue `(blob_id, emit_stage)`.
# - New fusion choice: D8 alone could introduce a fourth color after the third
#   blob, but D7 fixes the visible depth vocabulary at three blob colors.
#   The fusion resolves this by saturating later trajectory stages at yellow.
#
# d) Placement legality
# - Geometry generation remains output-first: lay down the staged cascade in the
#   output, invert it into an input with blue+green blobs, then validate by
#   running `main(inp)`.
# - All source legality checks still matter: blobs are in-bounds, 8-neighborhood
#   separated, rays do not cross distractors, and each incoming lane has a clear
#   path to its target edge cells.
# - No new spatial constraint is needed beyond the source task because the combo
#   only changes colors, not object footprints or ray topology.
#
# e) Visual observability
# - A solver can infer the rule from structure alone: warm color increases only
#   after a blob is traversed, and the next activated blob shares that new color.
# - The mapping does not depend on numeric color ids or hidden ordering. It is
#   tied to visible cascade depth and repeated across multiple samples.
#
# f) Visual cleanliness
# - Blob silhouettes remain intact because whole blob bodies are recolored with a
#   single stage color.
# - Rays still form straight thin/thick lanes; the only added variation is warm
#   hue changes at obvious bounce points.
# - Distractors are erased to background in the output, so they do not compete
#   with the staged cascade for attention.
#
# g) Change inferability
# - There are only three observable warm states, and each is anchored by both a
#   trajectory segment and (when hit) a blob of the same color.
# - The generator makes orange and yellow common enough that the user can infer
#   "later in the BFS cascade => later warm color" from a small training set.
#
# h) New main() assumptions and generator guarantees
# - `main()` assumes at most three activated blobs if stage-coded blob colors are
#   to remain non-ambiguous. The generator keeps the original chain limit 0-3.
# - `main()` assumes every blob still has exactly one port side whose blue-adjacent
#   lane is clear. `_candidate_next_blob(...)` and distractor placement preserve
#   the source task's legality guarantees.
# - `main()` assumes the start ray is the unique bottom-touching thin segment
#   with a non-background cap above it. The generator constructs exactly one such
#   marker and keeps all blobs away from the stem column.

from collections import Counter, deque
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

WARM_STAGES = (RED, ORANGE, YELLOW)

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
START_H_MIN, START_H_MAX = 2, 3

# Keep short chains possible, but bias strongly toward 3 hits so the full
# red -> orange -> yellow blob-depth code is usually visible in training sets.
CHAIN_LEN_WEIGHTS = [1, 1, 2, 10]
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

GEN_RETRIES = 300


def _stage_color(stage: int) -> int:
    """Map traversal stage to the fused warm palette, saturating at yellow."""
    idx = min(max(int(stage), 0), len(WARM_STAGES) - 1)
    return int(WARM_STAGES[idx])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a staged ray network whose blobs and ray segments share one depth code.

    Input structure:
    - The dominant color is the background. Blue diamond/octagon blobs carry one green edge port that specifies the outgoing ray side and thickness.
    - A unique start marker sits on the bottom edge: a yellow cap directly above a short red vertical stem.

    Rule:
    - Send the thin start ray upward in the first warm stage (red). When it first reaches a blue blob, recolor that whole blob with the same stage color.
    - Every activated blob emits its thick port ray in the next warm stage, so later trajectory segments and the later blobs they hit advance together through red -> orange -> yellow.
    - Continue breadth-first until all rays either leave the grid or stop at another blob; untouched blobs and unused green ports disappear into the background.

    Key insight: The cascade geometry is unchanged from the source task; the only new rule is that traversal depth is encoded uniformly, so each bounce increments the shared warm color used by subsequent rays and by the next newly activated blob.
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
            rr = int(r)
            cc = int(c)
            for dr, dc in DIRS:
                nr = rr + int(dr)
                nc = cc + int(dc)
                if 0 <= nr < H and 0 <= nc < W and int(g[nr, nc]) == int(blue_col):
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
        bi = len(blobs)
        for r, c in pts:
            blob_id[r, c] = int(bi)
        blobs.append({"pts": pts, "ports": []})

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
    queue: deque[tuple[int, int]] = deque()

    def activate_blob_at(r: int, c: int, incoming_stage: int) -> None:
        """Color a newly reached blob and queue its next-stage emission."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append((int(bi), int(incoming_stage) + 1))
        fill = _stage_color(incoming_stage)
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(fill)

    start_stage = 0
    start_color = _stage_color(start_stage)
    for r, c in start_seg:
        out[int(r), int(c)] = int(start_color)
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, int(start_col)]) == int(yellow_col)
    out[int(top_r) - 1, int(start_col)] = int(start_color)
    for r in range(int(top_r) - 1, -1, -1):
        if int(g[int(r), int(start_col)]) == int(blue_col):
            activate_blob_at(int(r), int(start_col), start_stage)
            break
        out[int(r), int(start_col)] = int(start_color)

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int], stage: int) -> None:
        """Extend one stage-colored ray lane until boundary or the next blob."""
        ddr, ddc = map(int, d)
        ray_color = _stage_color(stage)
        for pr, pc in port_pixels:
            r, c = int(pr), int(pc)
            while True:
                r += ddr
                c += ddc
                if not (0 <= r < H and 0 <= c < W):
                    break
                if int(g[r, c]) == int(blue_col):
                    activate_blob_at(r, c, stage)
                    break
                out[int(r), int(c)] = int(ray_color)

    while queue:
        bi, emit_stage = queue.popleft()
        for port in blobs[int(bi)]["ports"]:
            extend_thick(port["pixels"], port["dir"], int(emit_stage))

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    return out


def _shape_mask(h: int, w: int) -> np.ndarray:
    """Build the diamond/octagon footprint for a blob size."""
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
    """Return mask-local cells on the edge facing direction d."""
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _blob_spec(size: tuple[int, int], port_dir: tuple[int, int]) -> dict | None:
    """Build a blob template for one size and one outgoing direction."""
    h, w = size
    mask = _shape_mask(h, w)
    green = _edge_offsets(mask, port_dir)
    if len(green) not in (1, 2):
        return None

    green_set = set(green)
    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    for r, c in green:
        patch[int(r), int(c)] = GREEN

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


def _path_to_entries(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    entries: list[tuple[int, int]],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    """Trace all ray lanes from starts to the target entry cells."""
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
            cells.append((int(r), int(c)))
    return cells


def _path_to_boundary(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """Trace all ray lanes from starts until they leave the grid."""
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
            cells.append((int(r), int(c)))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]], stage: int) -> None:
    """Paint one whole trajectory segment using the fused stage color."""
    paint_coords(out, cells, _stage_color(stage))
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
    """Enumerate legal placements for a blob aligned to the incoming ray lanes."""
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
        r0 = int(origin_along if is_vertical else fixed)
        c0 = int(fixed if is_vertical else origin_along)
        if is_vertical:
            entry_cells = [(int(entry_line), int(c0 + c)) for _, c in window]
        else:
            entry_cells = [(int(r0 + r), int(entry_line)) for r, _ in window]
        full = _offset(spec["full"], r0, c0)
        if not _in_bounds(full, H, W):
            continue
        if any(bool(occ_pad[r, c]) or bool(ray_mask[r, c]) for r, c in full):
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
    """Choose one valid next blob that the current trajectory can hit."""
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
    """Place one unused blob away from all ray cells."""
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []
    for spec in _all_specs():
        h, w = spec["mask"].shape
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                full = _offset(spec["full"], r0, c0)
                if any(bool(occ_pad[r, c]) or bool(ray_mask[r, c]) for r, c in full):
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


def _gen_attempt() -> np.ndarray | None:
    """Construct one valid input by first painting the fused staged output."""
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
    inp[int(cap_r), int(start_col)] = YELLOW
    out[int(cap_r), int(start_col)] = _stage_color(0)
    ray_mask[int(cap_r), int(start_col)] = True
    occupied[int(cap_r), int(start_col)] = True
    for r in range(cap_r + 1, H):
        inp[int(r), int(start_col)] = RED
        out[int(r), int(start_col)] = _stage_color(0)
        occupied[int(r), int(start_col)] = True

    chain_len = random.choices(range(len(CHAIN_LEN_WEIGHTS)), weights=CHAIN_LEN_WEIGHTS)[0]
    min_dist = 1 if chain_len == 0 else 0
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(min_dist, max_dist)

    starts = [(int(cap_r), int(start_col))]
    direction = UP
    include_origin = True
    stage = 0

    for _ in range(chain_len):
        blob = _candidate_next_blob(H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask)
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"], stage)
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["blue"], _stage_color(stage))
        starts = blob["ports"]
        direction = blob["spec"]["port_dir"]
        include_origin = False
        stage += 1

    final_path = _path_to_boundary(starts, direction, include_origin, H, W)
    if any(bool(blue_mask[r, c]) for r, c in final_path):
        return None
    _mark_ray(out, ray_mask, final_path, stage)

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
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
