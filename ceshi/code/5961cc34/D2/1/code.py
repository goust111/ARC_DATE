# Variant of 5961cc34
# Dimension: D2 — 级联激活机制 (path)
# Change: BFS 级联：命中色块后立即激活并发射新射线，所有被间接命中的色块也被激活，直到无新命中... → 单步无级联
#
# Design analysis for this variant
# - source_task: None, so the baseline is the original 5961cc34 solver/generator only.
# - Concrete behavior:
#   The input still contains the same objects: one bottom start marker
#   (yellow cap over a short red stem) and 1-4 blue diamond/octagon blobs with
#   exactly one green port side each. The start ray still travels upward and the
#   first blob it touches still becomes active. The key change is what happens
#   next: only that first active blob is allowed to emit its thick port ray.
#   Any blob hit by that emitted ray is recolored red, but it does not emit its
#   own ray. So the visible output can contain:
#   1) the red start stem and cap,
#   2) the thin start ray,
#   3) the first activated blob,
#   4) one emitted thick ray from that blob,
#   5) optionally one more directly hit blob,
#   but never a third wave of rays.
# - Small traced example:
#   Original logic: start -> blob A turns red and enters queue -> A emits and
#   hits blob B -> B turns red and enters queue -> B emits and hits blob C ->
#   C turns red -> repeat until closure.
#   Variant logic: start -> blob A turns red -> A emits and can hit blob B ->
#   B turns red -> stop. Blob B never emits, so any latent blob C stays erased.
#   This keeps the spatial reasoning about ports and clear paths, but removes
#   recursive propagation.
# - Complexity calibration:
#   If the generator only produced a single blob, the change would be visually
#   weak and often indistinguishable from the original task. To avoid that, the
#   generator still constructs a latent chain of 0-3 blobs using the original
#   path-first placement logic. The output then truncates that chain after one
#   emission step. This preserves recognizable structure while making the new
#   rule observable on many samples.
# - Implicit-assumption audit against the original main():
#   * Start-marker detection is unchanged: the task still relies on exactly one
#     thin vertical segment touching the bottom with a non-background cap above.
#   * Blue/green color inference is unchanged: blue is still the largest
#     remaining object color and green is still the unique non-start color
#     adjacent to blue.
#   * Blob extraction and port detection are unchanged: blobs are still connected
#     blue components whose adjacent green side determines one outgoing
#     direction and width.
#   * The old queue/BFS assumption no longer holds. In the original code,
#     `activate_blob_at()` always enqueued a newly hit blob, and the final
#     `while queue:` loop assumed every activated blob must emit. That is the
#     exact place where the old D2 property lived, so this variant removes the
#     queue and emits from only the first start-hit blob.
#   * The original assert `len(blobs) - len(activated) <= 1` is invalid now,
#     because deeper latent chain blobs are intentionally left inactive. The
#     variant solver therefore drops that assert.
# - Placement legality:
#   The generator keeps the original object shapes, spacing rules, and
#   path-first candidate search. It still reserves space for the entire latent
#   chain so blobs never overlap, never leave the grid, and never block the
#   intended direct rays. Distractors are still placed off every reserved ray.
# - Visual observability:
#   The observable cue is structural, not symbolic: examples with latent chain
#   length >= 2 visibly stop after the second activated blob instead of
#   continuing to a third ray segment. Humans can infer this from a few
#   examples because the omitted downstream ray is the only changed behavior.
# - Cleanliness:
#   The output remains sparse: only one thin column, at most one thick emitted
#   ray, and at most two activated blobs are shown. Hidden deeper blobs remain
#   absent, which is visually cleaner than full BFS closure.
# - New solver assumptions and generator guarantees:
#   * The only blob allowed to emit is the first one hit by the upward start
#     ray; the generator guarantees that any deeper blobs, if present, are
#     reachable only through this latent chain and not by stray alternate paths.
#   * When a second blob is shown as directly hit, it is still a valid blob with
#     a port in the input, but that port is irrelevant in the output. The
#     generator guarantees this does not create ambiguity because no visible ray
#     ever originates from that blob in the target output.

from __future__ import annotations

from collections import Counter
import numpy as np
from arc_utils import colors_present, components_mask, get_bg, paste, paint_coords
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the red network produced by a single non-recursive activation wave.

    Input structure:
    - The background is the dominant color; blue diamond/octagon blobs may have a green edge bar that marks one outgoing port.
    - A yellow cap sitting above a short red stem at the bottom marks the unique upward starting ray.

    Rule:
    - Repaint the start marker red and extend its thin ray upward until it reaches the first blue blob or exits the grid.
    - If a blob is reached, recolor that blob red and emit its thick port ray once; blobs struck by that emitted ray also become red, but they do not emit again.

    Key insight: Green still tells you where the first activated blob would shoot, but the recursive chain reaction from the original task is cut off after that single emission step.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the start marker (bottom red stem with a non-red cap) ---
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
                    c0,
                    [(rr, c0) for rr in range(r0, r1 + 1)],
                )
            )

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]

    # --- STEP 2) Infer blob-body and port colors ---
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

    # --- STEP 3) Extract blue blobs and a pixel->blob index ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []

    for coords in components_mask(g == blue_col, connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = len(blobs)
        for r, c in pts:
            blob_id[r, c] = bi
        blobs.append({"pts": pts, "ports": []})

    # --- STEP 4) Recover each blob's port direction and emission width ---
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
            port_pixels = [
                (int(r), int(c))
                for (r, c) in pts
                if 0 <= int(r + ddr) < H
                and 0 <= int(c + ddc) < W
                and int(g[int(r + ddr), int(c + ddc)]) == green_col
            ]
            assert port_pixels
            blob["ports"].append({"dir": (int(ddr), int(ddc)), "pixels": port_pixels})

    activated: set[int] = set()

    def color_blob_at(r: int, c: int) -> int:
        """Recolor the hit blob to red and return its blob id."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi not in activated:
            activated.add(bi)
            for pr, pc in blobs[bi]["pts"]:
                out[int(pr), int(pc)] = red_col
        return bi

    # --- STEP 5) Draw the start ray upward and remember the first blob it hits ---
    for r, c in start_seg:
        out[int(r), int(c)] = red_col
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, int(start_col)]) == yellow_col
    out[top_r - 1, int(start_col)] = red_col

    first_blob: int | None = None
    for r in range(top_r - 1, -1, -1):
        out[int(r), int(start_col)] = red_col
        if int(g[int(r), int(start_col)]) == blue_col:
            first_blob = color_blob_at(int(r), int(start_col))
            break

    def extend_thick_once(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        """Emit one thick ray; directly hit blobs turn red but do not emit further."""
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
                    color_blob_at(r, c)
                    break

    # --- STEP 6) Only the first start-hit blob is allowed to emit ---
    if first_blob is not None:
        for port in blobs[first_blob]["ports"]:
            extend_thick_once(port["pixels"], port["dir"])

    return out


# --------------- generator (a2_single_wave) ---------------

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

# --- Latent chain / distractor ---
CHAIN_LEN_WEIGHTS = [1, 3, 3, 3]  # latent chain length in [0, 1, 2, 3]
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
        br, bc = gr - port_dir[0], gc - port_dir[1]
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


def _mark_ray(ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    """Mark reserved ray cells in the mask."""
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
    """Find a random valid placement for the next blob in the latent chain."""
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


def _build_variant_output(
    H: int,
    W: int,
    chain: list[dict],
    start_seg: list[tuple[int, int]],
    cap_r: int,
    start_col: int,
) -> np.ndarray:
    """Render the non-recursive output from a latent chain description."""
    out = np.full((H, W), BG, dtype=int)
    paint_coords(out, start_seg, RED)

    if not chain:
        paint_coords(out, _path_to_boundary([(cap_r, start_col)], UP, True, H, W), RED)
        return out

    paint_coords(out, chain[0]["path"], RED)
    paint_coords(out, chain[0]["blue"], RED)

    if len(chain) >= 2:
        paint_coords(out, chain[1]["path"], RED)
        paint_coords(out, chain[1]["blue"], RED)
    else:
        paint_coords(
            out,
            _path_to_boundary(chain[0]["ports"], chain[0]["spec"]["port_dir"], False, H, W),
            RED,
        )

    return out


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on placement failure."""
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    # Place start segment (yellow cap + red column at bottom).
    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = H - start_h - 1
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    start_seg = [(r, start_col) for r in range(cap_r + 1, H)]
    inp[cap_r, start_col] = YELLOW
    occupied[cap_r, start_col] = True
    for r, c in start_seg:
        inp[r, c] = RED
        occupied[r, c] = True

    # Decide latent chain length and distractor count.
    chain_len = random.choices(range(len(CHAIN_LEN_WEIGHTS)), weights=CHAIN_LEN_WEIGHTS)[0]
    min_dist = 1 if chain_len == 0 else 0
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(min_dist, max_dist)

    # Build the full latent chain exactly as in the original path-first generator.
    chain: list[dict] = []
    starts = [(cap_r, start_col)]
    direction = UP
    include_origin = True
    for _ in range(chain_len):
        blob = _candidate_next_blob(H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask)
        if blob is None:
            return None
        chain.append(blob)
        _mark_ray(ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        starts = blob["ports"]
        direction = blob["spec"]["port_dir"]
        include_origin = False

    final_path = _path_to_boundary(starts, direction, include_origin, H, W)
    if any(blue_mask[r, c] for r, c in final_path):
        return None
    _mark_ray(ray_mask, final_path)

    # Place distractors after reserving the latent chain space.
    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    out = _build_variant_output(H, W, chain, start_seg, cap_r, start_col)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 D2 variant input")
