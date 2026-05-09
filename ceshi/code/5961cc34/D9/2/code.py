# Variant of 5961cc34
# Dimension: D9 — 色块形状 (obstacle)
# Change: 对称钻石/八边形形状（左右+上下对称），尺寸从3x4到8x8不等，角部有缺口... → 大小不一但形状统一
#
# Design analysis
# - Source-task baseline:
#   The requested reference task is `b10624e5`. Inside the current workspace tree there is no
#   same-name solver copy, but a readable solver snapshot exists at
#   `/home/lixinyu/snap/new_code/solvers/arc2/b10624e5.py`. Its relevant idea is simple scaling:
#   infer one size ratio from a visible anchor, then resize the same motif everywhere else. This
#   variant applies that idea to the obstacle geometry only: all blobs share one template and differ
#   only by an integer scale.
# - a) Concrete variant:
#   Each blob is the same 4x4 diamond/octagon template
#       .##.
#       ####
#       ####
#       .##.
#   scaled by 1x/2x/3x with block scaling. So combined blue+green silhouettes are only 4x4, 8x8,
#   or 12x12. The chosen port side is the entire extreme edge of that scaled mask, so its green bar
#   length is 2*scale and the emitted ray has the same thickness. When a blob activates, the output
#   paints only the outline of that scaled silhouette, not a solid fill. Example trace:
#   original task: a 5x6 blob could emit a 2-wide beam just because that specific shape had a
#   2-cell edge; another 6x6 blob might also emit width 2 for a different reason.
#   variant: a 4x4 blob emits width 2, an 8x8 blob emits width 4, a 12x12 blob emits width 6.
#   The solver therefore has to treat blob size as a structured signal, not as an arbitrary shape.
# - b) Complexity calibration:
#   Only three scales are used, so the rule is visible in 2-4 examples; but it is not trivial,
#   because incoming rays of width 4 or 6 can only hit blobs whose opposite edge is long enough.
#   That keeps scale behavior active in the cascade without introducing many subcases.
# - c) Old implicit assumptions that break:
#   Original STEP 3 extracted only blue connected components. That was harmless when shapes varied
#   freely and green only annotated width, but it loses the true scaled silhouette here because the
#   scale is defined on blue+green together. Original STEP 4 also accepted any adjacent green set as
#   a port; here we must verify that the green cells match exactly one full side of the scaled
#   template, otherwise scale inference would be ambiguous.
# - d) Placement legality:
#   Generator still uses path-first construction, but every candidate blob now comes from the same
#   template. Placement remains valid because we keep the old non-overlap / 8-neighborhood padding
#   checks, ray-clearance checks, and output-first oracle validation with `main(inp) == out`.
# - e) Visual observability:
#   Human-visible support comes from repeated silhouette similarity plus a size-sensitive activation
#   effect: larger scaled blobs collapse from solid blue input masses into red outlines. A solver can
#   see that large blobs always keep the same family shape but no longer fill solidly in the output.
# - f) Visual cleanliness:
#   There is still exactly one port side per blob, all other cells are plain blue, and distractors
#   stay off every ray. This avoids mixed overlays and keeps each object readable.
# - g) Color inference:
#   No new colors or hidden color ordering are introduced. Colors keep their old roles; only shape
#   scaling changes.
# - Post-change solver assumptions and generator guarantees:
#   `main()` now assumes every blob's blue+green bounding box is 4s x 4s and equals the scaled base
#   template for some integer `s in {1,2,3}`; `generate_input()` enforces exactly that. It also
#   assumes the green cells occupy exactly one full side of that template and that blob separation is
#   at least one background cell so blue+green components do not merge. The generator also guarantees
#   that at least one activated blob has scale > 1, so the outline-vs-fill difference is visible.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


BASE_MASK = np.array(
    [
        [0, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 0],
    ],
    dtype=bool,
)


def _scaled_mask(scale: int) -> np.ndarray:
    """Return the common blob template scaled by an integer factor."""
    return np.kron(BASE_MASK, np.ones((scale, scale), dtype=bool))


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
    axis = 0 if d[0] else 1
    extreme = pts[:, axis].min() if d in ((-1, 0), (0, -1)) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _outline_offsets(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return mask-local cells on the 4-neighbor outline of the shape."""
    H, W = mask.shape
    out: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if any(
            not (0 <= rr + dr < H and 0 <= cc + dc < W) or not bool(mask[rr + dr, cc + dc])
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
        ):
            out.append((rr, cc))
    return out


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    """Translate local offsets to global coordinates."""
    return [(r0 + r, c0 + c) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in coords)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red ray network by activating scaled copies of one blob template via green side ports.

    Input structure:
    - bg = most common color. Every obstacle is a blue+green copy of the same 4x4 diamond template,
      scaled by 1x/2x/3x; the green cells occupy exactly one full side of that scaled shape.
    - Start marker: yellow pixel atop a short red segment at grid bottom.

    Rule:
    - Fire a thin ray upward from the start; the first blue blob hit activates.
    - Each activated blob emits a thick ray through its green side; the ray width equals that blob's
      scale-dependent port width, and propagation continues breadth-first until no new blob is hit.
    - Activated blobs are rendered as red outlines of their scaled silhouette; every traversed ray
      cell also becomes red, while untouched blobs disappear to bg.

    Key insight: The blob family is no longer a menu of unrelated silhouettes; all obstacles are one
    scaled template, and the same scale also controls the green bar length and outgoing ray width.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the start "port" (red vertical segment with a non-red cap above) ---
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
        if any(
            0 <= int(r + dr) < H and 0 <= int(c + dc) < W and int(g[int(r + dr), int(c + dc)]) == blue_col
            for r, c in pts
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
        ):
            green_candidates.append(int(col))

    assert len(green_candidates) == 1
    green_col = int(green_candidates[0])

    out = np.full_like(g, bg)

    # --- STEP 3) Extract scaled blobs as blue+green components and validate the common template ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    full_mask = (g == blue_col) | (g == green_col)

    for coords in components_mask(full_mask, connectivity=8):
        pts_full = [(int(r), int(c)) for r, c in coords]
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == w and h % BASE_MASK.shape[0] == 0
        scale = h // BASE_MASK.shape[0]
        assert scale in SCALES

        local_full = np.zeros((h, w), dtype=bool)
        local_green: set[tuple[int, int]] = set()
        for r, c in pts_full:
            local_full[r - r0, c - c0] = True
            if int(g[r, c]) == green_col:
                local_green.add((r - r0, c - c0))

        mask = _scaled_mask(scale)
        assert mask.shape == local_full.shape
        assert np.array_equal(local_full, mask)

        port_dir: tuple[int, int] | None = None
        port_pixels: list[tuple[int, int]] | None = None
        blue_pts: list[tuple[int, int]] = []

        for ddr, ddc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            edge = _edge_offsets(mask, (ddr, ddc))
            if set(edge) != local_green:
                continue
            trial_pixels: list[tuple[int, int]] = []
            ok = True
            for gr, gc in edge:
                br, bc = gr - ddr, gc - ddc
                if not (0 <= br < h and 0 <= bc < w):
                    ok = False
                    break
                if not mask[br, bc] or (br, bc) in local_green:
                    ok = False
                    break
                trial_pixels.append((r0 + br, c0 + bc))
            if ok:
                port_dir = (int(ddr), int(ddc))
                port_pixels = _sort_on_ray(trial_pixels, port_dir)
                break

        assert port_dir is not None and port_pixels is not None
        for r, c in pts_full:
            if int(g[r, c]) == blue_col:
                blob_id[r, c] = len(blobs)
                blue_pts.append((r, c))

        assert blue_pts
        blobs.append(
            {
                "full": pts_full,
                "outline": _offset(_outline_offsets(mask), r0, c0),
                "blue": blue_pts,
                "scale": int(scale),
                "ports": [{"dir": port_dir, "pixels": port_pixels}],
            }
        )

    activated: set[int] = set()
    queue: list[int] = []

    def activate_blob_at(r: int, c: int) -> None:
        """Recolor a blob to red when first hit by a ray; enqueue it to emit its port."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(bi)
        queue.append(bi)
        for pr, pc in blobs[bi]["outline"]:
            out[pr, pc] = red_col

    # --- STEP 4) Draw the start ray upward (thin) and stop at the first blob hit ---
    for r, c in start_seg:
        out[r, c] = red_col
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, start_col]) == yellow_col
    out[top_r - 1, start_col] = red_col
    for r in range(top_r - 1, -1, -1):
        out[r, start_col] = red_col
        if int(g[r, start_col]) == blue_col:
            activate_blob_at(r, start_col)
            break

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        """Extend a ray from each port-lane pixel until boundary or blob hit."""
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

    # --- STEP 5) Propagate (BFS): activated blob -> emit ray -> activate more blobs ---
    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    return out


# --------------- generator (a2_cascade, scaled-template variant) ---------------

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

# --- Unified shape scales ---
SCALES = (1, 2, 3)
SCALE_WEIGHTS = {1: 4, 2: 5, 3: 1}

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
GEN_RETRIES = 300


def _blob_spec(scale: int, port_dir: tuple[int, int]) -> dict | None:
    """Build a blob specification dict for a given scale and port direction."""
    mask = _scaled_mask(scale)
    h, w = mask.shape
    green = _edge_offsets(mask, port_dir)
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
        "scale": int(scale),
        "mask": mask,
        "patch": patch,
        "blue": blue,
        "full": full,
        "outline": _outline_offsets(mask),
        "port_dir": port_dir,
        "port_pixels": _sort_on_ray(port_pixels, port_dir),
    }


def _all_specs(port_dir: tuple[int, int] | None = None) -> list[dict]:
    """Generate all valid scaled blob specs, optionally filtered by port direction."""
    dirs = [port_dir] if port_dir is not None else list(DIRS)
    out: list[dict] = []
    for scale in SCALES:
        for _ in range(SCALE_WEIGHTS[scale]):
            for d in dirs:
                spec = _blob_spec(scale, d)
                if spec is not None:
                    out.append(spec)
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
    for (r, c), (er, ec) in zip(starts, entries, strict=False):
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
                "outline": _offset(spec["outline"], r0, c0),
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


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
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


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on placement failure."""
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
    include_origin = True
    activated_scales: list[int] = []
    for _ in range(chain_len):
        blob = _candidate_next_blob(H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask)
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["outline"], RED)
        activated_scales.append(int(blob["spec"]["scale"]))
        starts = blob["ports"]
        direction = blob["spec"]["port_dir"]
        include_origin = False

    final_path = _path_to_boundary(starts, direction, include_origin, H, W)
    if any(blue_mask[r, c] for r, c in final_path):
        return None
    _mark_ray(out, ray_mask, final_path)

    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    if not any(scale > 1 for scale in activated_scales):
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
