# Combo variant of 5961cc34 (index 13)
# combo_spec: D4/1+D12/2
# Name: 多起始点+从中心向外发射

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - Original task: one bottom stem launches one upward thin ray, that ray hits at most one
#      first blob, and then the usual green-port BFS takes over.
#    - Fused task: there are 2-3 start markers, and each marker is now a central cross:
#      one yellow center with four red arms. Every center launches four thin rays at once.
#    - Example trace:
#         source A at (ra, ca): rays go up/down/left/right.
#         source B at (rb, cb): rays also go up/down/left/right.
#         Before any blob is hit, A's horizontal ray and B's vertical ray may cross at
#         (ra, cb); that overlap is legal because both layers are simply red output rays.
#         If A-up reaches a blob first, that blob activates and emits a thick green-port ray.
#         If B-right reaches another blob, that blob joins the same BFS frontier. Activation
#         remains idempotent: a blob hit twice still turns red once.
#    - Difference from the original is therefore twofold at the same time:
#         1) there are multiple independent sources;
#         2) each source is itself a four-way launcher rather than a one-way bottom stem.
# b) Complexity calibration:
#    - Every sample has 2 or 3 start crosses, so the multi-source change is never silent.
#    - Every start cross always emits in all four directions, so the "center-to-outward"
#      change is also never silent.
#    - To avoid turning the grid into unreadable noise, each source only gets 1-2 active
#      first-hit directions, with an occasional length-2 chain on one branch.
# c) Implicit assumptions from the original solver that break:
#    - Start detection cannot assume a single bottom-touching red component.
#    - There is no single start column, no single initial direction, and no single first hit.
#    - A generator that forbids ray-ray overlap is no longer valid, because multiple center
#      crosses naturally create horizontal/vertical red crossings.
#    - The blue/green blob decoding logic still works unchanged.
# d) Placement legality:
#    - Start centers are chosen in a central window, with distinct rows and distinct columns,
#      and enough separation that their 5-pixel crosses stay disjoint.
#    - Blob bodies never overlap starts, other blobs, or already committed ray cells.
#    - Rays may overlap other rays, but they may not pass through occupied non-ray objects
#      except for their own explicit origin cell and the emitting blob's first green step.
# e) Visual observability:
#    - A human can immediately see 2-3 yellow-centered red crosses near the middle, instead
#      of one yellow-capped bottom stem.
#    - The output shows outward red plus-shaped launches from every source, often with visible
#      crossings between different sources' thin rays.
# f) Clean layout:
#    - Distinct source rows/columns avoid marker collisions while still allowing red crossings
#      away from the markers.
#    - Active branch count is capped so the result stays legible even with multiple sources.
# g) Color-change inferability:
#    - The color semantics stay simple: yellow centers become red, blue blobs hit by the ray
#      network become red, and green only encodes blob emission direction.
#    - No hidden ordering by color id is used; all logic is geometric.
# h) Non-degeneracy target:
#    - Every generated input has at least 2 start crosses.
#    - Because rows and columns are distinct, the initial ray system has off-center crossings,
#      a behavior impossible in the original task and in the single-cross D12 variant.
# i) Solver/generator contract:
#    - `_find_starts()` assumes each source is exactly a yellow center with four red orthogonal
#      neighbors. `generate_input()` enforces 2-3 such sources and keeps them unique.
#    - `main()` allows ray-ray overlap but assumes occupied non-ray objects are never placed on
#      those paths unexpectedly. The generator enforces that by reserving blob/start occupancy.


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
THREE_START_H_MIN = 19
THREE_START_W_MIN = 21

# --- Start crosses ---
START_COUNT_WEIGHTS = {2: 5, 3: 2}
MIN_CENTER_CHEB = 4
MIN_CENTER_MANHATTAN = 6

# --- Chain / distractor ---
ACTIVE_DIR_CHOICES = (1, 2)
ACTIVE_DIR_WEIGHTS = (3, 4)
EXTRA_BLOB_PROB = 0.35
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 3
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 700


def _find_starts(grid: np.ndarray, bg: int) -> list[dict]:
    """Find every yellow-centered red cross source."""
    H, W = grid.shape
    starts: list[dict] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            center_col = int(grid[r, c])
            if center_col == int(bg):
                continue
            arm_vals = [
                int(grid[r - 1, c]),
                int(grid[r + 1, c]),
                int(grid[r, c - 1]),
                int(grid[r, c + 1]),
            ]
            if len(set(arm_vals)) != 1:
                continue
            arm_col = int(arm_vals[0])
            if arm_col in (int(bg), center_col):
                continue
            starts.append(
                {
                    "arm_color": arm_col,
                    "center_color": int(center_col),
                    "center": (int(r), int(c)),
                    "arms": {
                        UP: (int(r - 1), int(c)),
                        DOWN: (int(r + 1), int(c)),
                        LEFT: (int(r), int(c - 1)),
                        RIGHT: (int(r), int(c + 1)),
                    },
                }
            )
    starts.sort(key=lambda s: s["center"])
    return starts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red ray network by activating blue blobs from several central cross launchers.

    Input structure:
    - bg is the dominant color. Blue diamond/octagon blobs still use adjacent green pixels as directional ports.
    - Start markers are 2-3 yellow-centered red crosses placed near the middle of the grid.

    Rule:
    - Turn every start cross fully red and launch four thin rays outward from each center.
    - Any ray that first reaches a blue blob activates that blob; activated blobs then emit thick rays through their green-marked side.
    - Continue the shared BFS cascade until no newly hit blob remains; overlapping rays simply stay red.

    Key insight: The blob/port mechanism is unchanged, but the initial frontier is now the union of several four-way central launches rather than one bottom-up beam.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect all central cross starts ---
    starts = _find_starts(g, bg)
    assert 2 <= len(starts) <= 3
    red_cols = {int(s["arm_color"]) for s in starts}
    yellow_cols = {int(s["center_color"]) for s in starts}
    assert len(red_cols) == 1 and len(yellow_cols) == 1
    red_col = int(next(iter(red_cols)))
    yellow_col = int(next(iter(yellow_cols)))

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
            for dr, dc in DIRS:
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
    queue: deque[int] = deque()

    def activate_blob_at(r: int, c: int) -> None:
        """Recolor a blob to red when first hit by any ray; enqueue it once."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(red_col)

    # --- STEP 5) Paint every start cross and emit four initial thin rays ---
    for start in starts:
        sr, sc = start["center"]
        out[int(sr), int(sc)] = int(red_col)
        for arm in start["arms"].values():
            out[int(arm[0]), int(arm[1])] = int(red_col)
        for dr, dc in DIRS:
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
        bi = int(queue.popleft())
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    return out


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
        rr = int(r)
        cc = int(c)
        guard = H + W + 10
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
    allowed: set[tuple[int, int]],
    ignore: set[tuple[int, int]] | None = None,
) -> bool:
    """Return True if a path would overlap occupied non-ray objects outside allowed cells."""
    ignore = ignore or set()
    for r, c in path:
        if (r, c) in allowed or (r, c) in ignore:
            continue
        if occupied[r, c]:
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
        if _path_blocked(path, occupied, allowed, ignore=extra_ignore):
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
    """Try to place one distractor blob in free space away from rays."""
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


def _choose_num_starts() -> int:
    """Sample the intended source-count branch for this puzzle instance."""
    weights = [START_COUNT_WEIGHTS[n] for n in (2, 3)]
    return int(random.choices((2, 3), weights=weights)[0])


def _center_window(H: int, W: int, num_starts: int) -> list[tuple[int, int]]:
    row_span = 4 if num_starts >= 3 else 3
    col_span = 5 if num_starts >= 3 else 4
    r0 = max(2, H // 2 - row_span)
    r1 = min(H - 3, H // 2 + row_span)
    c0 = max(2, W // 2 - col_span)
    c1 = min(W - 3, W // 2 + col_span)
    return [(int(r), int(c)) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _centers_compatible(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ra, ca = a
    rb, cb = b
    if ra == rb or ca == cb:
        return False
    if max(abs(ra - rb), abs(ca - cb)) < MIN_CENTER_CHEB:
        return False
    if abs(ra - rb) + abs(ca - cb) < MIN_CENTER_MANHATTAN:
        return False
    return True


def _choose_centers(H: int, W: int, num_starts: int) -> list[tuple[int, int]] | None:
    """Pick 2-3 central start centers with disjoint crosses and distinct rows/cols."""
    candidates = _center_window(H, W, num_starts)
    if len(candidates) < num_starts:
        return None
    for _ in range(300):
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for rc in candidates:
            if all(_centers_compatible(rc, other) for other in chosen):
                chosen.append(rc)
                if len(chosen) == num_starts:
                    return sorted(chosen)
    return None


def _branch_lengths(num_starts: int) -> list[dict[tuple[int, int], int]]:
    """Choose how many blobs each source/direction activates."""
    out: list[dict[tuple[int, int], int]] = []
    active_slots: list[tuple[int, tuple[int, int]]] = []
    for si in range(num_starts):
        active_count = int(random.choices(ACTIVE_DIR_CHOICES, weights=ACTIVE_DIR_WEIGHTS)[0])
        active_dirs = set(random.sample(list(DIRS), k=active_count))
        branch = {d: (1 if d in active_dirs else 0) for d in DIRS}
        out.append(branch)
        for d in active_dirs:
            active_slots.append((si, d))
    assert active_slots
    if random.random() < EXTRA_BLOB_PROB:
        si, d = random.choice(active_slots)
        out[si][d] += 1
    return out


def _choose_canvas(num_starts: int) -> tuple[int, int]:
    """Bias 3-start samples toward roomier grids so that branch coverage is observable."""
    if num_starts >= 3:
        h_low = max(H_MIN, THREE_START_H_MIN)
        w_low = max(W_MIN, THREE_START_W_MIN)
    else:
        h_low = H_MIN
        w_low = W_MIN
    return random.randint(h_low, H_MAX), random.randint(w_low, W_MAX)


def _build_start(
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    ray_mask: np.ndarray,
    center: tuple[int, int],
) -> dict:
    """Place one central cross source."""
    sr, sc = center
    arms = {
        UP: (int(sr - 1), int(sc)),
        DOWN: (int(sr + 1), int(sc)),
        LEFT: (int(sr), int(sc - 1)),
        RIGHT: (int(sr), int(sc + 1)),
    }
    inp[int(sr), int(sc)] = YELLOW
    for r, c in arms.values():
        inp[int(r), int(c)] = RED
    cross = [(int(sr), int(sc)), *[(int(r), int(c)) for r, c in arms.values()]]
    paint_coords(out, cross, RED)
    paint_coords(occupied, cross, True)
    paint_coords(ray_mask, cross, True)
    return {"center": (int(sr), int(sc)), "arms": arms}


def _initial_crossing_count(starts: list[dict]) -> int:
    """Count off-center crossings induced by distinct source rows/columns."""
    centers = [s["center"] for s in starts]
    rows = [r for r, _ in centers]
    cols = [c for _, c in centers]
    crossings = 0
    center_set = set(centers)
    for r in rows:
        for c in cols:
            if (r, c) not in center_set:
                crossings += 1
    return int(crossings)


def _gen_attempt(num_starts: int) -> np.ndarray | None:
    """Single attempt to generate a valid fused multi-cross input grid."""
    H, W = _choose_canvas(num_starts)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    centers = _choose_centers(H, W, num_starts)
    if centers is None:
        return None

    starts = [_build_start(inp, out, occupied, ray_mask, center) for center in centers]
    if _initial_crossing_count(starts) <= 0:
        return None

    lengths = _branch_lengths(num_starts)
    active_tasks = [(si, d, lengths[si][d]) for si in range(num_starts) for d in DIRS if lengths[si][d] > 0]
    idle_tasks = [(si, d) for si in range(num_starts) for d in DIRS if lengths[si][d] == 0]
    random.shuffle(active_tasks)
    random.shuffle(idle_tasks)

    for si, initial_dir, branch_len in active_tasks:
        starts_lane = [starts[si]["arms"][initial_dir]]
        direction = initial_dir
        include_origin = True

        for _ in range(int(branch_len)):
            blob = _candidate_next_blob(H, W, starts_lane, direction, include_origin, occupied, blue_mask, ray_mask)
            if blob is None:
                return None
            _mark_ray(out, ray_mask, blob["path"])
            _place_blob(inp, occupied, blue_mask, blob)
            paint_coords(out, blob["blue"], RED)
            starts_lane = blob["ports"]
            direction = blob["spec"]["port_dir"]
            include_origin = False

        final_path = _path_to_boundary(starts_lane, direction, include_origin, H, W)
        allowed = set(starts_lane) if include_origin else set()
        ignore = _first_step_cells(starts_lane, direction, H, W) if not include_origin else set()
        if any(blue_mask[r, c] for r, c in final_path if (r, c) not in allowed):
            return None
        if _path_blocked(final_path, occupied, allowed, ignore=ignore):
            return None
        _mark_ray(out, ray_mask, final_path)

    for si, initial_dir in idle_tasks:
        starts_lane = [starts[si]["arms"][initial_dir]]
        final_path = _path_to_boundary(starts_lane, initial_dir, True, H, W)
        allowed = set(starts_lane)
        if any(blue_mask[r, c] for r, c in final_path if (r, c) not in allowed):
            return None
        if _path_blocked(final_path, occupied, allowed):
            return None
        _mark_ray(out, ray_mask, final_path)

    total_chain_blobs = sum(lengths[si][d] for si in range(num_starts) for d in DIRS)
    max_distractors = 1 if total_chain_blobs <= 5 else 0
    num_distractors = random.randint(0, min(MAX_DISTRACTORS, max_distractors))
    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    found = _find_starts(inp, BG)
    if len(found) != num_starts:
        return None
    if _initial_crossing_count(found) <= 0:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    target_starts = _choose_num_starts()

    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(target_starts)
        if grid is not None:
            return grid

    fallback = 2 if target_starts == 3 else 3
    for _ in range(GEN_RETRIES // 3):
        grid = _gen_attempt(fallback)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
