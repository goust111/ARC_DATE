# Variant of 5961cc34
# Dimension: D12 — 起始射线方向 (path)
# Change: 起始射线固定从底部向上发射... → 从中心向外

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# Design analysis for the D12 variant
# -----------------------------------
# 1) Concrete visual change:
#    The old task has one start source: a yellow cap above a short red bottom stem.
#    This variant replaces it with a central cross marker: one yellow center pixel
#    with four orthogonally adjacent red pixels. In the output that yellow center
#    turns red, and four thin rays continue outward from the center.
# 2) Behavior trace:
#    Old flow: bottom stem -> one upward ray -> first blue blob -> BFS via green ports.
#    New flow: center cross -> four simultaneous thin rays (up/down/left/right) ->
#    each direction either hits its first blue blob or exits the grid -> every hit
#    blob joins the same BFS queue and emits thick rays through its green ports.
#    The blob and port mechanics stay unchanged; only the initial source geometry
#    and fan-out change.
# 3) Complexity calibration:
#    To make the variant visibly different, the generator usually activates 2-4
#    initial directions, not just one. Each branch is still constrained to short
#    chains, so the task remains legible and does not become a dense maze.
# 4) Implicit-assumption audit from the original solver:
#    - Start detection can no longer rely on a bottom-touching red component.
#    - The initial ray can no longer assume a single column or a single hit.
#    - The old unconditional "paint one upward line then stop" would miss three
#      start directions here.
#    - The BFS and port decoding are still valid because blue blobs and green port
#      semantics are unchanged.
# 5) Placement legality:
#    The generator reserves the central cross, keeps blobs separated by a 1-pixel
#    8-neighborhood buffer, prevents rays from crossing existing blobs/markers,
#    and disallows ray overlaps away from the shared center.
# 6) Visual observability:
#    The center cross is an obvious new source marker, and the output shows a red
#    cross-shaped launch pattern before the blob cascade. This is distinguishable
#    from the original bottom-to-top behavior in 1-2 examples.
# 7) Solver/generator contract:
#    `main()` assumes exactly one yellow-centered red cross. `generate_input()`
#    guarantees that uniqueness, keeps green adjacent only to blue blobs, and
#    limits non-activated blobs to at most one distractor.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red ray network by activating blue blobs from a central cross source.

    Input structure:
    - bg = most common color. Blue diamond/octagon blobs use adjacent green pixels as directional ports.
    - Start marker: one yellow center pixel with four red arm pixels forming a plus near the grid center.

    Rule:
    - Turn the center marker red and launch four thin rays outward from it.
    - When a ray first reaches a blue blob, recolor that blob red and emit thick rays through its green-marked side.
    - Continue the cascade in BFS order until no newly hit blobs remain.

    Key insight: The green port encoding is unchanged; the only new twist is that the initial activation comes from a four-way central source instead of a single upward launcher.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the central start cross (yellow center with four red arms) ---
    start_candidates: list[tuple[int, int, tuple[int, int]]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            center_col = int(g[r, c])
            if center_col == int(bg):
                continue
            arm_cols = {
                int(g[r - 1, c]),
                int(g[r + 1, c]),
                int(g[r, c - 1]),
                int(g[r, c + 1]),
            }
            if len(arm_cols) != 1:
                continue
            arm_col = int(next(iter(arm_cols)))
            if arm_col in (int(bg), center_col):
                continue
            start_candidates.append((arm_col, center_col, (int(r), int(c))))

    assert len(start_candidates) == 1
    red_col, yellow_col, start_center = start_candidates[0]

    # --- STEP 2) Identify blob colors ---
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
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < H and 0 <= nc < W and int(g[nr, nc]) == int(blue_col):
                    ok = True
                    break
            if ok:
                break
        if ok:
            green_candidates.append(int(col))

    assert len(green_candidates) == 1
    green_col = int(green_candidates[0])

    out = np.full_like(g, int(bg))

    # --- STEP 3) Extract blue blobs and build a pixel->blob map ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []

    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[int(r), int(c)] = bi
        blobs.append({"pts": pts, "ports": []})

    # --- STEP 4) Decode each blob's port directions from adjacent green markers ---
    for blob in blobs:
        pts = blob["pts"]
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in pts:
            for ddr, ddc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
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
        """Recolor a blob to red when first hit by a ray; enqueue it to emit its ports."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(red_col)

    # --- STEP 5) Paint the central cross and emit four initial thin rays ---
    sr, sc = start_center
    out[int(sr), int(sc)] = int(red_col)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r = int(sr)
        c = int(sc)
        while True:
            r += int(dr)
            c += int(dc)
            if not (0 <= r < H and 0 <= c < W):
                break
            out[int(r), int(c)] = int(red_col)
            if int(g[int(r), int(c)]) == int(blue_col):
                activate_blob_at(int(r), int(c))
                break

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        """Extend a ray from each port pixel until boundary or blob hit."""
        ddr, ddc = map(int, d)
        for pr, pc in port_pixels:
            r = int(pr)
            c = int(pc)
            while True:
                r += int(ddr)
                c += int(ddc)
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[int(r), int(c)] = int(red_col)
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c))
                    break

    # --- STEP 6) Propagate (BFS): every activated blob emits from its ports ---
    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"

    return out


# --------------- generator (a2_cascade_d12_center_cross) ---------------

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

# --- Chain / distractor ---
ACTIVE_DIR_CHOICES = (2, 3, 4)
ACTIVE_DIR_WEIGHTS = (5, 3, 1)
EXTRA_BLOB_PROB = 0.45
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 3
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 400


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
            r0 = max(0, dr)
            r1 = H + min(0, dr)
            c0 = max(0, dc)
            c1 = W + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    """Translate local offsets to global coordinates."""
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
    """Trace ray lanes from starts to entries; return cell list or None on failure."""
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        if include_origin:
            cells.append((int(r), int(c)))
        guard = H + W + 10
        rr = int(r)
        cc = int(c)
        while (rr, cc) != (int(er), int(ec)):
            rr += int(dr)
            cc += int(dc)
            guard -= 1
            if guard < 0 or not (0 <= rr < H and 0 <= cc < W):
                return None
            cells.append((int(rr), int(cc)))
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
        rr = int(r)
        cc = int(c)
        if include_origin:
            cells.append((int(rr), int(cc)))
        while True:
            rr += int(dr)
            cc += int(dc)
            if not (0 <= rr < H and 0 <= cc < W):
                break
            cells.append((int(rr), int(cc)))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    """Paint cells red in the output grid and mark them in the ray mask."""
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


def _first_step_cells(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    H: int,
    W: int,
) -> set[tuple[int, int]]:
    """Cells immediately beyond each start lane, used to ignore the emitting blob's green port cells."""
    out: set[tuple[int, int]] = set()
    dr, dc = d
    for r, c in starts:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < H and 0 <= cc < W:
            out.add((int(rr), int(cc)))
    return out


def _path_blocked(
    path: list[tuple[int, int]],
    occupied: np.ndarray,
    ray_mask: np.ndarray,
    allowed: set[tuple[int, int]],
    ignore: set[tuple[int, int]] | None = None,
) -> bool:
    """Return True if a path would overlap existing objects/rays outside allowed cells."""
    ignore = ignore or set()
    for r, c in path:
        if (r, c) in allowed or (r, c) in ignore:
            continue
        if occupied[r, c] or ray_mask[r, c]:
            return True
    return False


def _try_place_along_axis(
    spec: dict,
    window: list[tuple[int, int]],
    d: tuple[int, int],
    lane_vals: list[int],
    starts: list[tuple[int, int]],
    include_origin: bool,
    min_gap: int,
    occupied: np.ndarray,
    occ_pad: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    H: int,
    W: int,
) -> list[dict]:
    """Find all valid placements for a blob spec along the current ray axis."""
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
        if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
            continue
        path = _path_to_entries(starts, d, entry_cells, include_origin, H, W)
        if path is None:
            continue
        entry_set = set(entry_cells)
        if any(blue_mask[r, c] for r, c in path if (r, c) not in entry_set):
            continue
        allowed = set(starts) if include_origin else set()
        extra_ignore = set(entry_set)
        if not include_origin:
            extra_ignore |= _first_step_cells(starts, d, H, W)
        if _path_blocked(path, occupied, ray_mask, allowed, ignore=extra_ignore):
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
    """Find a random valid placement for the next blob in a branch."""
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
                    occupied,
                    occ_pad,
                    blue_mask,
                    ray_mask,
                    H,
                    W,
                )
            )

    return random.choice(candidates) if candidates else None


def _place_blob(
    inp: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    blob: dict,
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
    """Try to place a single distractor blob in free space."""
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


def _branch_lengths() -> dict[tuple[int, int], int]:
    """Choose how many blobs each of the four initial directions should activate."""
    active_count = random.choices(ACTIVE_DIR_CHOICES, weights=ACTIVE_DIR_WEIGHTS)[0]
    active_dirs = set(random.sample(list(DIRS), k=int(active_count)))
    lengths = {d: (1 if d in active_dirs else 0) for d in DIRS}
    if active_dirs and random.random() < EXTRA_BLOB_PROB:
        lengths[random.choice(list(active_dirs))] += 1
    return lengths


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on placement failure."""
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    # Place center cross start marker.
    sr = H // 2
    sc = W // 2
    cross = [(sr, sc), (sr - 1, sc), (sr + 1, sc), (sr, sc - 1), (sr, sc + 1)]
    inp[sr, sc] = YELLOW
    for r, c in cross[1:]:
        inp[r, c] = RED
    paint_coords(out, cross, RED)
    paint_coords(occupied, cross, True)
    paint_coords(ray_mask, cross, True)

    lengths = _branch_lengths()
    order = random.sample(list(DIRS), k=len(DIRS))
    total_chain_blobs = sum(lengths.values())

    for initial_dir in order:
        starts = [(int(sr + initial_dir[0]), int(sc + initial_dir[1]))]
        direction = initial_dir
        include_origin = True

        for _ in range(int(lengths[initial_dir])):
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
        allowed = set(starts) if include_origin else set()
        if any(blue_mask[r, c] for r, c in final_path if (r, c) not in allowed):
            return None
        ignore = _first_step_cells(starts, direction, H, W) if not include_origin else set()
        if _path_blocked(final_path, occupied, ray_mask, allowed, ignore=ignore):
            return None
        _mark_ray(out, ray_mask, final_path)

    num_distractors = random.randint(0, 1 if total_chain_blobs <= 4 else 0)
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
    raise RuntimeError("failed to generate a valid 5961cc34 D12 input")
