# Variant of 5961cc34
# Dimension: D4 — 起始标记结构 (src)
# Change: 网格底部有一根高度2-3的红色垂直段，顶部有一个黄色像素作为帽盖，从黄色像素向上发射起始射线... → 边缘控制像素

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# Design analysis for the D4 variant
# ----------------------------------
# a) Concrete behavior
#    - Old input: a bottom-anchored red stem capped by one yellow pixel. The first
#      ray always started at that cap and always went upward.
#    - New input: exactly one red control pixel sits on one border cell, away from
#      corners. Its border side defines the ray direction:
#         top -> downward, bottom -> upward, left -> rightward, right -> leftward.
#      The rest of the puzzle is unchanged: blue blobs with one green port side,
#      activated blobs become red and emit thick rays from their ports.
#    - Small trace:
#         original: yellow at (H-4,c), red stem below it -> scan rows r-1,r-2,...
#         variant : red ctrl at (r,0) on left edge       -> scan cols 1,2,3,...
#      The first-hit logic is identical; only start detection and first ray
#      direction differ.
# b) Complexity calibration
#    - The control marker is still just one visible object, so the puzzle does not
#      become harder by hidden encoding.
#    - The start edge varies across samples, which makes the D4 change visually
#      obvious and prevents degeneration into "always bottom" behavior.
#    - The control pixel uses the same red color as the final activated network,
#      preserving the original color semantics instead of introducing a new color.
# c) Original assumptions that no longer hold
#    - Old solver assumed there was one thin vertical non-bg component touching the
#      bottom and one non-red cap above it. This is invalid now.
#    - Old solver also assumed the initial direction was always UP and that the
#      output must repaint a whole stem plus the cap. This is invalid now.
#    - Blob/port extraction, blue/green identification, and BFS propagation do not
#      depend on the bottom stem, so they remain valid.
# d) Placement legality
#    - Generator keeps the output-first pipeline. It now places a single red border
#      pixel as the ray origin and constructs the cascade from that point.
#    - To keep solver parsing unambiguous, blobs are forbidden from touching the
#      grid border. Therefore the only non-background border cell in the input is
#      the control pixel.
#    - Existing no-overlap, in-bounds, and ray-clearance checks are preserved.
# e) Visual observability
#    - Across 2-4 examples a human can directly see that the trigger is no longer
#      a bottom stem: the start is a lone border pixel and its edge determines the
#      initial ray orientation.
#    - The downstream red cascade still mirrors the original task, so the changed
#      factor is isolated and legible.
# f) Visual cleanliness
#    - Keeping blobs off the border avoids clutter between start controls and blob
#      silhouettes.
#    - Only one border marker exists, so the start cue is always prominent.
# g) Color-change inferability
#    - No hidden color remapping is introduced. Blue bodies plus adjacent green
#      ports still turn into a red activated network, exactly as before.
# h) Quantifiable variant signal
#    - The natural metric is the control edge distribution. If many generated
#      samples use top/left/right starts, the variant is active and not collapsing
#      back to the original bottom-only structure.
# i) New solver assumptions to be guaranteed by the generator
#    - Exactly one non-background border cell exists, and it is the red control.
#    - Every blob lies strictly inside the border so border scanning is unique.
#    - There is at most one distractor blob, matching the retained solver assert.


# --- Colors ---
BG = 8
BLUE = 1
RED = 2
GREEN = 3

# --- Directions ---
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

EDGE_TO_DIR = {
    "top": DOWN,
    "bottom": UP,
    "left": RIGHT,
    "right": LEFT,
}

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
EDGE_PAD = 2

# --- Chain / distractor ---
CHAIN_LEN_WEIGHTS = [1, 3, 3, 3]
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 300


def _control_direction(r: int, c: int, H: int, W: int) -> tuple[int, int]:
    if int(r) == 0:
        return DOWN
    if int(r) == int(H - 1):
        return UP
    if int(c) == 0:
        return RIGHT
    if int(c) == int(W - 1):
        return LEFT
    raise AssertionError("control pixel must lie on the border")


def _touches_border(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return any(int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1)) for r, c in coords)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red ray network from a border control pixel through ported blue blobs.

    Input structure:
    - bg is the dominant color. Blue blobs sit inside the grid and green pixels on one blob edge mark emission ports.
    - Start marker is a single red control pixel on the border; its edge determines the inward launch direction.

    Rule:
    - Shoot a thin red ray straight inward from the control pixel until it leaves the grid or first hits a blue blob.
    - Each newly hit blob turns red and emits thick rays from its green-marked port edge; continue breadth-first until no new blob is reached.
    - Keep only the red control pixel, all traversed rays, and every activated blob in the output.

    Key insight: The only changed cue is the start marker geometry: a lone border pixel encodes the initial ray direction, while green port sides still encode every subsequent emission.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the unique border control pixel and its launch direction. ---
    border_pts: list[tuple[int, int]] = []
    for c in range(W):
        if int(g[0, c]) != int(bg):
            border_pts.append((0, int(c)))
        if int(H - 1) != 0 and int(g[H - 1, c]) != int(bg):
            border_pts.append((int(H - 1), int(c)))
    for r in range(1, H - 1):
        if int(g[r, 0]) != int(bg):
            border_pts.append((int(r), 0))
        if int(W - 1) != 0 and int(g[r, W - 1]) != int(bg):
            border_pts.append((int(r), int(W - 1)))

    assert len(border_pts) == 1
    start_r, start_c = border_pts[0]
    red_col = int(g[start_r, start_c])
    start_dir = _control_direction(int(start_r), int(start_c), H, W)

    # --- STEP 2) Identify blob colors. ---
    counts = Counter(map(int, g.flatten()))
    other_cols = [(int(cnt), int(col)) for col, cnt in counts.items() if int(col) not in (bg, red_col)]
    assert other_cols
    max_count = max(cnt for cnt, _col in other_cols)
    blue_candidates = [int(col) for cnt, col in other_cols if cnt == max_count]
    assert len(blue_candidates) == 1
    blue_col = int(blue_candidates[0])

    green_candidates: list[int] = []
    for col in colors_present(g, exclude={bg, red_col, blue_col}):
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

    # --- STEP 3) Extract blue blobs and build a pixel->blob lookup. ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[int(r), int(c)] = bi
        blobs.append({"pts": pts, "ports": []})

    # --- STEP 4) Infer each blob's ports from adjacent green markers. ---
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
    queue: list[int] = []

    def activate_blob_at(r: int, c: int) -> None:
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(red_col)

    # --- STEP 5) Draw the initial inward ray from the border control pixel. ---
    dr, dc = start_dir
    r, c = int(start_r), int(start_c)
    while 0 <= r < H and 0 <= c < W:
        out[int(r), int(c)] = int(red_col)
        if int(g[int(r), int(c)]) == int(blue_col):
            activate_blob_at(int(r), int(c))
            break
        r += int(dr)
        c += int(dc)

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        ddr, ddc = map(int, d)
        for pr, pc in port_pixels:
            r, c = int(pr), int(pc)
            while True:
                r += int(ddr)
                c += int(ddc)
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[int(r), int(c)] = int(red_col)
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c))
                    break

    # --- STEP 6) Propagate breadth-first through all newly activated blobs. ---
    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    return out


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
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        if include_origin:
            cells.append((int(r), int(c)))
        guard = H + W + 10
        while (int(r), int(c)) != (int(er), int(ec)):
            r = int(r + dr)
            c = int(c + dc)
            guard -= 1
            if guard < 0 or not (0 <= int(r) < H and 0 <= int(c) < W):
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
            r = int(r + dr)
            c = int(c + dc)
            if not (0 <= int(r) < H and 0 <= int(c) < W):
                break
            cells.append((int(r), int(c)))
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
    low = int(edge_line)
    high = int(extent + edge_line)
    if d in (UP, LEFT):
        high = min(high, int(start_line - min_gap))
    else:
        low = max(low, int(start_line + min_gap))

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
        if _touches_border(full, H, W):
            continue
        if any(bool(occ_pad[r, c]) or bool(ray_mask[r, c]) for r, c in full):
            continue
        path = _path_to_entries(starts, d, entry_cells, include_origin, H, W)
        if path is None:
            continue
        entry_set = set(entry_cells)
        if any(bool(blue_mask[r, c]) for r, c in path if (int(r), int(c)) not in entry_set):
            continue
        candidates.append({
            "spec": spec,
            "r0": r0,
            "c0": c0,
            "entry": entry_cells,
            "path": path,
            "blue": _offset(spec["blue"], r0, c0),
            "full": full,
            "ports": _sort_on_ray(_offset(spec["port_pixels"], r0, c0), spec["port_dir"]),
        })

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
    lane_delta = [int(v - lane_vals[0]) for v in lane_vals]
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
            win_delta = [int(_perp_value(p, d) - _perp_value(window[0], d)) for p in window]
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
                if _touches_border(full, H, W):
                    continue
                if any(bool(occ_pad[r, c]) or bool(ray_mask[r, c]) for r, c in full):
                    continue
                candidates.append({
                    "spec": spec,
                    "r0": int(r0),
                    "c0": int(c0),
                    "full": full,
                    "blue": _offset(spec["blue"], r0, c0),
                })
    if not candidates:
        return False
    _place_blob(inp, occupied, blue_mask, random.choice(candidates))
    return True


def _sample_control(H: int, W: int) -> tuple[tuple[int, int], tuple[int, int], str]:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge == "top":
        coord = (0, random.randint(EDGE_PAD, W - EDGE_PAD - 1))
    elif edge == "bottom":
        coord = (int(H - 1), random.randint(EDGE_PAD, W - EDGE_PAD - 1))
    elif edge == "left":
        coord = (random.randint(EDGE_PAD, H - EDGE_PAD - 1), 0)
    else:
        coord = (random.randint(EDGE_PAD, H - EDGE_PAD - 1), int(W - 1))
    return (int(coord[0]), int(coord[1])), EDGE_TO_DIR[edge], edge


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    # Place a single red border control pixel.
    (start_r, start_c), direction, _edge = _sample_control(H, W)
    inp[start_r, start_c] = RED
    out[start_r, start_c] = RED
    occupied[start_r, start_c] = True
    ray_mask[start_r, start_c] = True

    chain_len = random.choices(range(len(CHAIN_LEN_WEIGHTS)), weights=CHAIN_LEN_WEIGHTS)[0]
    min_dist = 1 if chain_len == 0 else 0
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(min_dist, max_dist)

    starts = [(int(start_r), int(start_c))]
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

    final_path = _path_to_boundary(starts, direction, include_origin, H, W)
    if any(bool(blue_mask[r, c]) for r, c in final_path):
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
    raise RuntimeError("failed to generate a valid 5961cc34 D4 variant input")
