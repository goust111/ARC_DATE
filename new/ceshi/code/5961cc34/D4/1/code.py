# Variant of 5961cc34
# Dimension: D4 — 起始标记结构 (src)
# Change: 网格底部有一根高度2-3的红色垂直段，顶部有一个黄色像素作为帽盖，从黄色像素向上发射起始射线... → 多起始点

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste

# Design analysis for the D4 variant
# ----------------------------------
# a) Concrete behavior:
#    - The input still contains blue diamond/octagon blobs with green ports, and each
#      activated blob still emits a thick ray in the port direction.
#    - The only semantic change is the start-marker multiplicity: instead of exactly one
#      yellow-over-red stem at the bottom, there are always 2 or 3 such stems in distinct
#      columns. Every cap launches its own thin upward ray. All first hits are inserted into
#      the same activation frontier, so multiple source trees propagate in parallel.
#    - Small trace example:
#        original: one bottom cap at column c -> one upward ray -> first blob activates.
#        variant : caps at columns c1 and c2 -> two upward rays -> maybe two different
#        blobs activate, or one source misses while the other hits; afterwards the BFS queue
#        contains every newly reached blob. Order no longer matters because activation is
#        idempotent and all painted cells become red.
# b) Complexity calibration:
#    - The visual difference is explicit because every sample has 2-3 starts.
#    - Blob shapes, colors, and port semantics stay unchanged, preventing a second hidden
#      rule from being introduced.
#    - Start count is limited to 2 or 3 so the new degree of freedom is visible but still
#      learnable from a few examples.
# c) Original implicit assumptions that break:
#    - `assert len(start_candidates) == 1` is invalid.
#    - A single `start_col`, `start_seg`, and single initial activation are no longer enough.
#    - Any generator logic that reserves space for just one source column must be generalized.
#    - Output painting must tolerate overlap: a later source may repaint already-red cells.
# d) Placement legality guarantees in this file:
#    - Start columns are separated by at least `MIN_START_SEP`, so markers remain visually
#      distinct and do not accidentally fuse into one thicker red object.
#    - New blobs cannot overlap existing blobs or start markers, and distractors stay off all
#      ray cells.
#    - Ray paths may overlap earlier rays, but they may not pass through occupied objects
#      except their own explicit origin cell.
# e) Visual observability:
#    - The new rule is directly visible from the bottom boundary: there are multiple identical
#      yellow-capped red stems.
#    - No hidden color-id ordering is used; all decisions are geometric.
# f) Clean layout:
#    - Sources are spaced apart, blobs are still separated by a one-pixel safety halo, and
#      only red output layers are allowed to overlap.
# g) Color-change inferability:
#    - The color behavior is unchanged from the original task: activated blue blobs and all
#      traversed rays become red. The only new inference is "do this from every start marker".
# h) Non-degeneracy target:
#    - Every generated grid has at least 2 starts, so the old one-source solver would reject
#      the input. This prevents silent regression to the original task.
# i) New solver assumptions audited against the generator:
#    - `_find_starts()` assumes each source is a 1-cell-wide vertical bottom segment of height
#      2 or 3 with a single non-background cap directly above. `generate_input()` enforces it.
#    - Blue remains the dominant non-marker color; generator keeps enough blue mass for that.
#    - Port color is still the unique non-marker color 4-adjacent to blue; generator preserves
#      the original blob construction.


def _find_starts(grid: np.ndarray, bg: int) -> list[dict]:
    """Find every yellow-capped bottom stem. The variant expects 2-3 of them."""
    H, W = grid.shape
    starts: list[dict] = []
    for c in range(W):
        stem_col = int(grid[H - 1, c])
        if stem_col == int(bg):
            continue
        r0 = H - 1
        while r0 >= 0 and int(grid[r0, c]) == int(stem_col):
            r0 -= 1
        top_r = int(r0 + 1)
        height = int(H - top_r)
        if height not in (2, 3):
            continue
        cap_r = int(top_r - 1)
        if cap_r < 0:
            continue
        cap_col = int(grid[cap_r, c])
        if cap_col in (int(bg), int(stem_col)):
            continue
        starts.append(
            {
                "stem_color": int(stem_col),
                "cap_color": int(cap_col),
                "col": int(c),
                "cap_r": int(cap_r),
                "seg": [(int(rr), int(c)) for rr in range(int(top_r), H)],
            }
        )
    starts.sort(key=lambda s: s["col"])
    return starts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red ray network by activating blue blobs from multiple bottom launchers.

    Input structure:
    - bg = most common color. Large blue blobs carry green side markers that encode emission direction and thickness.
    - Start markers: 2-3 yellow-capped short red stems sit on the bottom edge, each in its own column.

    Rule:
    - Fire a thin upward ray from every yellow cap; each ray stops on its first blue hit or exits the grid.
    - Every blob reached by any source turns red, then emits thick rays through its green-marked side.
    - Continue the cascade until no newly reached blobs remain; overlapping activations simply stay red.

    Key insight: The local blob logic is unchanged from the original task; the only twist is that the same activation
    process must be seeded from every bottom marker rather than from exactly one source.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect all bottom start markers (short red stem + yellow cap) ---
    starts = _find_starts(g, bg)
    assert 2 <= len(starts) <= 3
    red_cols = {int(s["stem_color"]) for s in starts}
    yellow_cols = {int(s["cap_color"]) for s in starts}
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
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
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

    # --- STEP 5) Draw every start ray upward and seed the activation frontier ---
    for start in starts:
        for r, c in start["seg"]:
            out[int(r), int(c)] = int(red_col)
        cap_r = int(start["cap_r"])
        start_col = int(start["col"])
        assert int(g[cap_r, start_col]) == int(yellow_col)
        out[cap_r, start_col] = int(red_col)
        for r in range(int(cap_r), -1, -1):
            out[int(r), int(start_col)] = int(red_col)
            if int(g[int(r), int(start_col)]) == int(blue_col):
                activate_blob_at(int(r), int(start_col))
                break

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

    # --- STEP 6) Propagate (BFS): every activated blob emits all of its outgoing rays ---
    while queue:
        bi = int(queue.popleft())
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    return out


# --------------- generator (a2_cascade_multi_src) ---------------

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
MIN_START_SEP = 3

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3
START_COUNT_CHOICES = (2, 3)

# --- Chain / distractor ---
CHAIN_LEN_CHOICES = (0, 1)
CHAIN_LEN_WEIGHTS = (1, 5)
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
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
    blob_mask: np.ndarray,
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
    allowed_origin = set(starts) if include_origin else set()

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
        if any(blob_mask[r, c] and (r, c) not in allowed_origin and (r, c) not in entry_set for r, c in path):
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
    blob_mask: np.ndarray,
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
                    blob_mask,
                    occ_pad,
                    blue_mask,
                    ray_mask,
                    H,
                    W,
                )
            )

    return random.choice(candidates) if candidates else None


def _place_blob(
    inp: np.ndarray, occupied: np.ndarray, blob_mask: np.ndarray, blue_mask: np.ndarray, blob: dict
) -> None:
    """Paste a blob's patch onto the input grid and update tracking masks."""
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[r, c] = True
        blob_mask[r, c] = True
    for r, c in blob["blue"]:
        blue_mask[r, c] = True


def _place_distractor(
    inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, ray_mask: np.ndarray
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
    _place_blob(inp, occupied, occupied, blue_mask, random.choice(candidates))
    return True


def _choose_start_cols(W: int, num_starts: int) -> list[int] | None:
    """Pick distinct source columns with a guaranteed gap for visual separation."""
    cols = list(range(START_COL_PAD, W - START_COL_PAD))
    candidates: list[list[int]] = []

    def backtrack(idx: int, chosen: list[int]) -> None:
        if len(chosen) == num_starts:
            candidates.append(chosen.copy())
            return
        if idx >= len(cols):
            return
        for j in range(idx, len(cols)):
            c = cols[j]
            if chosen and c - chosen[-1] < MIN_START_SEP:
                continue
            chosen.append(c)
            backtrack(j + 1, chosen)
            chosen.pop()

    backtrack(0, [])
    return random.choice(candidates) if candidates else None


def _valid_path_against_blobs(
    path: list[tuple[int, int]], blob_mask: np.ndarray, allowed: set[tuple[int, int]]
) -> bool:
    return not any(blob_mask[r, c] and (r, c) not in allowed for r, c in path)


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid multi-source input grid."""
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blob_mask = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    if W >= 21 and H >= 17:
        num_starts = random.choice(START_COUNT_CHOICES)
    else:
        num_starts = 2
    start_cols = _choose_start_cols(W, num_starts)
    if start_cols is None:
        return None

    starts_info: list[dict] = []
    for start_col in start_cols:
        start_h = random.randint(START_H_MIN, START_H_MAX)
        cap_r = H - start_h - 1
        inp[cap_r, start_col] = YELLOW
        out[cap_r, start_col] = RED
        ray_mask[cap_r, start_col] = True
        occupied[cap_r, start_col] = True
        for r in range(cap_r + 1, H):
            inp[r, start_col] = RED
            out[r, start_col] = RED
            occupied[r, start_col] = True
        starts_info.append({"cap_r": cap_r, "col": start_col})

    chain_lengths = random.choices(CHAIN_LEN_CHOICES, weights=CHAIN_LEN_WEIGHTS, k=num_starts)
    if sum(chain_lengths) == 0:
        chain_lengths[random.randrange(num_starts)] = 1
    num_distractors = random.randint(0, MAX_DISTRACTORS)

    for start, chain_len in zip(starts_info, chain_lengths):
        starts = [(int(start["cap_r"]), int(start["col"]))]
        direction = UP
        include_origin = True
        final_path: list[tuple[int, int]]
        if chain_len == 1:
            blob = None
            for _ in range(40):
                cand = _candidate_next_blob(H, W, starts, direction, include_origin, occupied, blob_mask, blue_mask, ray_mask)
                if cand is None:
                    break
                cand_final = _path_to_boundary(cand["ports"], cand["spec"]["port_dir"], False, H, W)
                if any(blob_mask[r, c] for r, c in cand_final):
                    continue
                blob = cand
                final_path = cand_final
                break
            if blob is None:
                return None
            _mark_ray(out, ray_mask, blob["path"])
            _place_blob(inp, occupied, blob_mask, blue_mask, blob)
            paint_coords(out, blob["blue"], RED)
        else:
            final_path = _path_to_boundary(starts, direction, include_origin, H, W)
            allowed = set(starts)
            if any(blue_mask[r, c] for r, c in final_path if (r, c) not in allowed):
                return None
        _mark_ray(out, ray_mask, final_path)

    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    if len(_find_starts(inp, BG)) != num_starts:
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
