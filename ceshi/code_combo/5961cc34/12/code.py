# Combo variant of 5961cc34 (index 12)
# combo_spec: D7/3+D13/0
# Name: 未命中色块保留+裁剪到活跃区域

from __future__ import annotations

# Design analysis for combo D7/3 + D13/0
# a) Concrete behavior trace:
#    Original task builds a full-size red activation network, then erases every
#    untouched blue/green blob back to background. This combo changes two
#    separate stages of that pipeline. First, untouched distractor blobs are no
#    longer erased: they stay blue with their green port markers. Second, the
#    final full-size picture is cropped to the minimal bounding box of red
#    pixels only. A useful mental example is: the bottom start marker activates
#    one blob, that blob emits a horizontal ray, and an untouched distractor
#    sits inside the rectangle spanned by the vertical and horizontal red legs.
#    In the original task the distractor disappears and the output keeps the
#    whole canvas; here the distractor survives because it lies inside the red
#    crop box, while all outside margins are cut away.
# b) Complexity calibration:
#    If samples were allowed to omit either cropping or preserved distractors,
#    the fusion would often collapse back to the original visual behavior. The
#    generator therefore forces two visible signals at once: output height is
#    reduced by keeping the top red row away from row 0, and exactly one
#    distractor blob is placed fully inside the final red crop box. Chain length
#    still varies from 1 to 3 to preserve the original cascade flavor.
# c) Hidden assumptions audit:
#    The original solver assumes the output starts as pure background and only
#    activated objects are painted. That is invalid now, because untouched
#    blobs must survive. The new renderer therefore starts from a copy of the
#    input and paints activation red on top of it. The ray logic still assumes
#    that intended hits occur on blue body pixels rather than green port pixels;
#    the generator preserves that by aligning incoming rays to blue entry cells.
#    Cropping also introduces a new assumption: there is always at least one red
#    pixel. The bottom start segment guarantees that.
# d) Placement legality:
#    Distractors are sampled only when their whole footprint lies inside the
#    future red bounding box, is in-bounds, does not overlap any object, and
#    stays at least one cell away from rays and other blobs.
# e) Visual observability:
#    Every generated sample simultaneously shows blue/green pixels in the
#    output and a strictly smaller output height than the input, so a human can
#    infer both "preserve untouched distractor" and "crop to active red area"
#    from just a few examples.
# f) Visual cleanliness:
#    The kept distractor is separated from red rays by background, so the crop
#    region remains legible rather than becoming a pileup of overlapping marks.
# g) Inferability of the change:
#    No hidden numeric ordering is used. The preserved colors remain the same
#    blue/green object colors seen in the input, while red still marks the
#    activation network, so the new behavior is recoverable from structure.

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


def _red_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_to_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _render_full_output(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
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
                (int(col), int(cap), int(c0), [(int(rr), int(c0)) for rr in range(r0, r1 + 1)])
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

    out = g.copy()

    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == blue_col, connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[r, c] = bi
        blobs.append({"pts": pts, "ports": []})

    for blob in blobs:
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in blob["pts"]:
            for ddr, ddc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + ddr)
                cc = int(c + ddc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == green_col:
                    seen_dirs.add((int(ddr), int(ddc)))
        for ddr, ddc in seen_dirs:
            port_pixels = [
                (int(r), int(c))
                for (r, c) in blob["pts"]
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

    for r, c in start_seg:
        out[int(r), int(c)] = red_col
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, start_col]) == yellow_col
    out[top_r - 1, start_col] = red_col
    for r in range(top_r - 1, -1, -1):
        out[r, start_col] = red_col
        if int(g[r, start_col]) == blue_col:
            activate_blob_at(r, start_col)
            break

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        ddr, ddc = map(int, d)
        for pr, pc in port_pixels:
            r, c = int(pr), int(pc)
            while True:
                r += ddr
                c += ddc
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[r, c] = red_col
                if int(g[r, c]) == blue_col:
                    activate_blob_at(r, c)
                    break

    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    red_box = _red_bbox(out == red_col)
    return out, red_box


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build the activation network, keep untouched distractors, then crop to the red-active box.

    Input structure:
    - Background is the most common color. Blue diamond/octagon blobs may have green edge markers that show port direction.
    - The launcher is still a short bottom red column capped by one yellow pixel directly above it.

    Rule:
    - Fire the thin start ray upward, activate every blue blob reached by the BFS-style cascade, and paint all activated bodies and ray paths red.
    - Blobs never reached by the rays are not erased anymore; they stay blue with their green ports.
    - After the full scene is rendered, crop the output to the smallest rectangle containing every red pixel.

    Key insight:
    - The activation mechanics are unchanged from the base task; the new twist is to distinguish red-active structure from untouched blue/green structure, then crop using only the red support while letting any untouched distractor inside that box remain visible.
    """
    full, red_box = _render_full_output(grid)
    return _crop_to_bbox(full, red_box)


# --------------- generator (a2_cascade combo) ---------------

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
START_H_MIN, START_H_MAX = 2, 3
CHAIN_LENGTHS = (1, 2, 3)
CHAIN_LEN_WEIGHTS = (3, 3, 2)
MAX_BLOB_COUNT = 4
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3
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
    allowed_port_dirs: tuple[tuple[int, int], ...] | None = None,
) -> dict | None:
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    allowed = None if allowed_port_dirs is None else set(allowed_port_dirs)
    candidates: list[dict] = []

    for spec in _all_specs():
        if spec["port_dir"] == entry_side:
            continue
        if allowed is not None and spec["port_dir"] not in allowed:
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
        occupied[r, c] = True
    for r, c in blob["blue"]:
        blue_mask[r, c] = True


def _place_distractor_in_bbox(
    inp: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    box: tuple[int, int, int, int],
) -> bool:
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    ray_pad = _dilate8(ray_mask)
    br0, bc0, br1, bc1 = box
    candidates: list[dict] = []

    for spec in _all_specs():
        h, w = spec["mask"].shape
        for r0 in range(br0, br1 - h + 2):
            for c0 in range(bc0, bc1 - w + 2):
                full = _offset(spec["full"], r0, c0)
                if not _in_bounds(full, H, W):
                    continue
                if any(r < br0 or r > br1 or c < bc0 or c > bc1 for r, c in full):
                    continue
                if any(occ_pad[r, c] or ray_pad[r, c] for r, c in full):
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


def _expected_from_active(inp: np.ndarray, active_red: np.ndarray) -> np.ndarray:
    full = np.array(inp, dtype=int)
    full[active_red == RED] = RED
    return _crop_to_bbox(full, _red_bbox(active_red == RED))


def _gen_attempt(chain_len: int | None = None) -> np.ndarray | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    active_red = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = H - start_h - 1
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    inp[cap_r, start_col] = YELLOW
    active_red[cap_r, start_col] = RED
    ray_mask[cap_r, start_col] = True
    occupied[cap_r, start_col] = True
    for r in range(cap_r + 1, H):
        inp[r, start_col] = RED
        active_red[r, start_col] = RED
        occupied[r, start_col] = True

    if chain_len is None:
        chain_len = random.choices(CHAIN_LENGTHS, weights=CHAIN_LEN_WEIGHTS)[0]
    if chain_len + 1 > MAX_BLOB_COUNT:
        return None

    starts = [(cap_r, start_col)]
    direction = UP
    include_origin = True
    for idx in range(chain_len):
        allowed = (LEFT, RIGHT) if idx == chain_len - 1 else None
        blob = _candidate_next_blob(
            H,
            W,
            starts,
            direction,
            include_origin,
            occupied,
            blue_mask,
            ray_mask,
            allowed_port_dirs=allowed,
        )
        if blob is None:
            return None
        _mark_ray(active_red, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(active_red, blob["blue"], RED)
        starts = blob["ports"]
        direction = blob["spec"]["port_dir"]
        include_origin = False

    final_path = _path_to_boundary(starts, direction, include_origin, H, W)
    if any(blue_mask[r, c] for r, c in final_path):
        return None
    _mark_ray(active_red, ray_mask, final_path)

    red_box = _red_bbox(active_red == RED)
    if red_box[0] <= 0:
        return None

    if not _place_distractor_in_bbox(inp, occupied, blue_mask, ray_mask, red_box):
        return None

    expected = _expected_from_active(inp, active_red)
    if expected.shape == inp.shape:
        return None
    if not np.any(np.isin(expected, (BLUE, GREEN))):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    preferred_chain_len = random.choice(CHAIN_LENGTHS)
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(preferred_chain_len)
        if grid is not None:
            return grid
    for chain_len in CHAIN_LENGTHS:
        if chain_len == preferred_chain_len:
            continue
        for _ in range(GEN_RETRIES // 2):
            grid = _gen_attempt(chain_len)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
