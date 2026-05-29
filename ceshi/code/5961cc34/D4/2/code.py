from __future__ import annotations

# Variant of 5961cc34
# Dimension: D4 — 起始标记结构 (src)
# Change: 网格底部有一根高度2-3的红色垂直段，顶部有一个黄色像素作为帽盖，从黄色像素向上发射起始射线... → 形状指示方向的起始标记

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# Design analysis for this D4 variant
# - Concrete form: the old bottom "red stem + yellow cap" launcher is replaced by a
#   3x3 two-color arrow sprite. Four red body pixels form a small wedge/T shape,
#   and one yellow tip pixel protrudes on the firing side. The tail cell touches
#   the border opposite the firing direction, so the ray clearly enters from an edge.
# - Behavior trace: in the original task, Step 1 hard-coded "find a bottom vertical
#   segment, then shoot upward from the cap". Here that step changes to "find the
#   unique border-anchored arrow, infer its direction from the protruding yellow tip,
#   then shoot from the tip". Once a blue blob is hit, the existing green-port BFS
#   cascade is unchanged.
# - Complexity calibration: there are only four cardinal directions and one fixed
#   sprite family, so the variation is visible without exploding the case count. The
#   start colors remain the original red/yellow pair, which keeps color roles stable.
# - Implicit-assumption audit: the original solver assumed the start body was a
#   1-cell-wide bottom segment and that the cap sat directly above it. That is false
#   now, so start detection must match geometry instead of relative height. Later
#   color inference still assumes start colors can be excluded from blob colors; the
#   generator guarantees the arrow is the only red/yellow object.
# - Placement legality / cleanliness: the generator anchors the arrow tail to one
#   boundary, keeps the arrow isolated from blobs via the same occupied-mask logic,
#   and reuses the original path-first cascade placement so rays and blobs still do
#   not overlap illegally.
# - Observability: start directions are sampled across all four cardinals, including
#   horizontal launches, so the new rule is visible in a few examples and cannot be
#   mistaken for the original always-up start.
# - New solver assumptions that the generator enforces: exactly one arrow exists; its
#   red body has area 4, it has exactly one 4-neighbor yellow tip, the union fits a
#   3x3 template, and the tail cell lies on the boundary opposite the launch direction.


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

# --- Start arrow ---
START_FREE_PAD = 1
START_TEMPLATES = {
    UP: {
        "body": {(1, 0), (1, 1), (1, 2), (2, 1)},
        "tip": (0, 1),
        "tail": (2, 1),
    },
    DOWN: {
        "body": {(0, 1), (1, 0), (1, 1), (1, 2)},
        "tip": (2, 1),
        "tail": (0, 1),
    },
    LEFT: {
        "body": {(0, 1), (1, 1), (1, 2), (2, 1)},
        "tip": (1, 0),
        "tail": (1, 2),
    },
    RIGHT: {
        "body": {(0, 1), (1, 0), (1, 1), (2, 1)},
        "tip": (1, 2),
        "tail": (1, 0),
    },
}
for _spec in START_TEMPLATES.values():
    _spec["full"] = set(_spec["body"]) | {_spec["tip"]}


def _normalize(coords: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return {(r - r0, c - c0) for r, c in coords}, (r0, c0)


def _tail_on_required_boundary(
    tail: tuple[int, int], d: tuple[int, int], H: int, W: int,
) -> bool:
    r, c = tail
    if d == UP:
        return r == H - 1
    if d == DOWN:
        return r == 0
    if d == LEFT:
        return c == W - 1
    return c == 0


def _find_start_marker(g: np.ndarray, bg: int) -> dict:
    H, W = g.shape
    candidates: list[dict] = []

    for body_col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(body_col), connectivity=8):
            body = {(int(r), int(c)) for r, c in coords}
            if len(body) != 4:
                continue

            tip_neighbors: set[tuple[int, int]] = set()
            for r, c in body:
                for dr, dc in DIRS:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W:
                        v = int(g[rr, cc])
                        if v != int(bg) and v != int(body_col):
                            tip_neighbors.add((rr, cc))
            if len(tip_neighbors) != 1:
                continue

            tip = next(iter(tip_neighbors))
            tip_col = int(g[tip])
            if tip_col == int(body_col):
                continue

            full = set(body) | {tip}
            if len(full) != 5:
                continue
            norm_full, origin = _normalize(full)
            norm_body = {(r - origin[0], c - origin[1]) for r, c in body}

            matches: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
            for d, spec in START_TEMPLATES.items():
                if norm_full != spec["full"] or norm_body != spec["body"]:
                    continue
                expected_tip = (origin[0] + spec["tip"][0], origin[1] + spec["tip"][1])
                tail = (origin[0] + spec["tail"][0], origin[1] + spec["tail"][1])
                if tip != expected_tip:
                    continue
                if not _tail_on_required_boundary(tail, d, H, W):
                    continue
                matches.append((d, expected_tip, tail))

            if len(matches) != 1:
                continue
            d, tip_rc, tail_rc = matches[0]
            candidates.append(
                {
                    "body_col": int(body_col),
                    "tip_col": int(tip_col),
                    "dir": d,
                    "tip": tip_rc,
                    "tail": tail_rc,
                    "full": sorted(full),
                }
            )

    assert len(candidates) == 1
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red ray network, but read the first ray direction from a small arrow sprite.

    Input structure:
    - bg is the dominant color. Blue diamond/octagon blobs have adjacent green edge pixels
      that mark their emission ports.
    - The start marker is a tiny red arrow with one yellow protruding tip on a grid edge;
      the tip side reveals the initial firing direction.

    Rule:
    - Recolor the whole start sprite red, then launch a thin ray from the yellow tip in the
      direction implied by the arrow shape.
    - When a ray first hits a blue blob, recolor that blob red; each activated blob then emits
      thick rays through its green-marked side until another blob or the boundary stops them.
    - Continue breadth-first until no new blobs are reached; untouched blobs and ports disappear.

    Key insight:
    - The only asymmetric two-color sprite is the launcher. Its protruding yellow tip gives the
      first direction, and after that the original green-port cascade logic applies unchanged.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the start arrow sprite and infer the initial ray direction ---
    start = _find_start_marker(g, bg)
    red_col = int(start["body_col"])
    yellow_col = int(start["tip_col"])
    start_dir = start["dir"]
    start_tip = tuple(map(int, start["tip"]))
    start_full = [(int(r), int(c)) for r, c in start["full"]]

    # --- STEP 2) Identify blob colors ---
    counts = Counter(map(int, g.flatten()))
    other_cols = [
        (int(cnt), int(col))
        for col, cnt in counts.items()
        if int(col) not in (bg, red_col, yellow_col)
    ]
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

    # --- STEP 3) Extract blue blobs as connected components, and build a pixel->blob map ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []

    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[int(r), int(c)] = bi
        blobs.append({"pts": pts, "ports": []})

    # --- STEP 4) For each blob, find its ports from adjacent green markers ---
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
                if 0 <= int(r + ddr) < H
                and 0 <= int(c + ddc) < W
                and int(g[int(r + ddr), int(c + ddc)]) == int(green_col)
            ]
            assert port_pixels
            blob["ports"].append({"dir": (int(ddr), int(ddc)), "pixels": port_pixels})

    activated: set[int] = set()
    queue: list[int] = []

    def activate_blob_at(r: int, c: int) -> None:
        """Recolor a blob to red when first hit by a ray; enqueue it to emit its ports."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(red_col)

    # --- STEP 5) Draw the start ray from the arrow tip in the inferred direction ---
    for r, c in start_full:
        out[int(r), int(c)] = int(red_col)

    dr, dc = map(int, start_dir)
    r, c = map(int, start_tip)
    while 0 <= r < H and 0 <= c < W:
        out[int(r), int(c)] = int(red_col)
        if int(g[int(r), int(c)]) == int(blue_col):
            activate_blob_at(int(r), int(c))
            break
        r += int(dr)
        c += int(dc)

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        """Extend a ray from each port pixel until boundary or blob hit."""
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

    # --- STEP 6) Propagate (BFS): activated blob -> emit rays -> activate more blobs ---
    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"

    return out


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

# --- Chain / distractor ---
CHAIN_LEN_WEIGHTS = [1, 3, 3, 3]
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 300


def _shape_mask(h: int, w: int) -> np.ndarray:
    """Build a boolean diamond/octagon mask for the given (h, w) shape."""
    widths = _SHAPE_WIDTHS[(h, w)]
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(widths):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    """Coordinate perpendicular to direction d (column for vertical, row for horizontal)."""
    return rc[1] if d[0] else rc[0]


def _along_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    """Coordinate along direction d (row for vertical, column for horizontal)."""
    return rc[0] if d[0] else rc[1]


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    """Sort coordinates by their perpendicular value relative to direction d."""
    return sorted(coords, key=lambda p: _perp_value(p, d))


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    """Return sorted mask-local coordinates on the edge facing direction d."""
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _blob_spec(size: tuple[int, int], port_dir: tuple[int, int]) -> dict | None:
    """Build a blob specification dict for a given shape and port direction, or None if invalid."""
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
        br = gr - port_dir[0]
        bc = gc - port_dir[1]
        if not (0 <= br < h and 0 <= bc < w):
            return None
        if not mask[br, bc] or (br, bc) in green_set:
            return None
        port_pixels.append((br, bc))

    blue = [
        (int(r), int(c))
        for r, c in np.argwhere(mask)
        if (int(r), int(c)) not in green_set
    ]
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
    """Generate all valid blob specs, optionally filtered by port direction, in shuffled order."""
    dirs = [port_dir] if port_dir is not None else list(DIRS)
    out = [s for size in SHAPES for d in dirs if (s := _blob_spec(size, d)) is not None]
    random.shuffle(out)
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    """Dilate a boolean mask by 1 pixel in all 8 directions."""
    H, W = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    """Translate local offsets to global coordinates."""
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
    """Trace ray lanes from starts to entries; return cell list or None on failure."""
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
    """Trace ray lanes from starts to grid boundary."""
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
    """Paint cells red in the output grid and mark them in the ray mask."""
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
    """Find all valid placements for a blob spec along the ray's travel axis."""
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
) -> dict | None:
    """Find a random valid placement for the next blob in the cascade chain."""
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


def _place_blob(
    inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict,
) -> None:
    """Paste a blob's patch onto the input grid and update tracking masks."""
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
    """Try to place a single distractor blob in free space. Return success."""
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


def _make_start_patch(direction: tuple[int, int]) -> np.ndarray:
    patch = np.full((3, 3), BG, dtype=int)
    spec = START_TEMPLATES[direction]
    for r, c in spec["body"]:
        patch[r, c] = RED
    tr, tc = spec["tip"]
    patch[tr, tc] = YELLOW
    return patch


def _start_placement(
    H: int, W: int, direction: tuple[int, int],
) -> tuple[np.ndarray, int, int, tuple[int, int], list[tuple[int, int]]]:
    """Create the start arrow patch, its top-left origin, tip, and occupied cells."""
    if direction in (UP, DOWN):
        lane = random.randint(START_FREE_PAD, W - START_FREE_PAD - 1)
        r0 = H - 3 if direction == UP else 0
        c0 = lane - 1
    else:
        lane = random.randint(START_FREE_PAD, H - START_FREE_PAD - 1)
        r0 = lane - 1
        c0 = W - 3 if direction == LEFT else 0

    patch = _make_start_patch(direction)
    spec = START_TEMPLATES[direction]
    tip = (r0 + spec["tip"][0], c0 + spec["tip"][1])
    full = _offset(sorted(spec["full"]), r0, c0)
    return patch, r0, c0, tip, full


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on placement failure."""
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    # Place the directional start arrow.
    direction = random.choice(DIRS)
    start_patch, start_r0, start_c0, start_tip, start_full = _start_placement(H, W, direction)
    paste(inp, start_patch, start_r0, start_c0, transparent=BG)
    paint_coords(out, start_full, RED)
    for r, c in start_full:
        occupied[r, c] = True
    ray_mask[start_tip] = True

    # Decide chain length and distractor count.
    chain_len = random.choices(range(len(CHAIN_LEN_WEIGHTS)), weights=CHAIN_LEN_WEIGHTS)[0]
    min_dist = 1 if chain_len == 0 else 0
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(min_dist, max_dist)

    # Build cascade chain.
    starts = [start_tip]
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

    # Final ray to grid boundary.
    final_path = _path_to_boundary(starts, direction, include_origin, H, W)
    if any(blue_mask[r, c] for r, c in final_path):
        return None
    _mark_ray(out, ray_mask, final_path)

    # Place distractors.
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
    raise RuntimeError("failed to generate a valid 5961cc34 D4 input")
