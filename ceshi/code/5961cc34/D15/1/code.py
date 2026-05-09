# Variant of 5961cc34
# Dimension: D15 — 颜色角色分配 (meta)
# Change: 固定颜色角色：背景=青色(8)，色块=蓝色(1)，端口=绿色(3)，起始帽=黄色(4)，起始段/激活/射线=红色(2)... → 多类型色块不同颜色

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# -----------------------------------------------------------------------------
# Design analysis for the D15 variant
#
# concrete_variant:
#   - The cascade geometry is unchanged: same start marker, same blob silhouettes,
#     same green port bars, same BFS ray propagation, same "erase untouched blobs"
#     output style.
#   - Only the blob-body color role changes. Instead of one universal blue body,
#     each blob body is one of two visible types:
#       * blue (1)   -> pink (6) after activation
#       * orange (7) -> maroon (9) after activation
#   - Rays and the start marker stay red so the original causal structure remains
#     easy to read; only activated body pixels use the transferred colors.
#
# behavior_trace:
#   - Original local step:
#       input body color is always blue; when a ray hits any body pixel, the whole
#       body becomes red.
#   - Variant local step:
#       the incoming ray still hits the body in the same way, but the recolor now
#       depends on that blob's own input body color:
#         blue body   -> pink output body
#         orange body -> maroon output body
#       the ray itself stays red outside the blob.
#   - Difference:
#       "what gets activated" is still geometric, while "what color the activated
#       blob becomes" is now a fixed type mapping that must be inferred from the
#       examples.
#
# complexity_calibration:
#   - Two body types are enough to make the change obvious without turning the task
#     into a large lookup table. More types would require too many exemplars for a
#     human to be confident.
#   - Most generated chains contain at least two activated blobs, and chains of
#     length >= 2 are forced to contain both body types, so each sample usually
#     displays both color-transfer pairs directly.
#
# implicit_assumptions_review:
#   - Original solver assumed "dominant remaining color == the blob color". This
#     breaks immediately once multiple body colors exist, because no single color
#     covers every blob.
#   - Original solver also assumed "the unique color adjacent to blue == port
#     color". With multiple body colors, the safe approach is to parse each blob
#     as its own connected component, then separate majority body pixels from the
#     minority green port pixels inside that component.
#   - Original hit-detection checked `g[r, c] == blue_col`. In the variant, rays
#     must instead check a body-pixel-to-blob id map so every body color can be hit.
#
# placement_legality:
#   - Geometry and spacing stay identical to the seed task. Blob placement still
#     guarantees in-bounds shapes, one-cell object separation, and clear straight
#     ray lanes between intended contacts.
#   - The generator still uses output-first validation and rejects any sample whose
#     reconstructed output disagrees with `main()`.
#
# visual_observability:
#   - The only visual novelty is the body-color transfer, so the rest of the scene
#     is kept stable: green still means ports, red still means rays/start.
#   - Untouched distractors disappear in the output exactly as before, which makes
#     the activated recolors stand out cleanly instead of competing with leftover
#     objects.
#
# color_inferability:
#   - The mapping is fixed globally across all samples:
#       blue (1)   -> pink (6)
#       orange (7) -> maroon (9)
#   - These four colors are all visibly distinct from bg/green/yellow/red, and the
#     generator deliberately repeats both body types often enough for a human to
#     infer the two-pair mapping from a small train set.
# -----------------------------------------------------------------------------


BODY_TO_ACTIVE = {
    1: 6,  # blue -> pink
    7: 9,  # orange -> maroon
}
BODY_COLORS = tuple(BODY_TO_ACTIVE.keys())


# --- Fixed colors retained from the seed task ---
BG = 8
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


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    """Coordinate perpendicular to direction d (column for vertical, row for horizontal)."""
    return rc[1] if d[0] else rc[0]


def _along_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    """Coordinate along direction d (row for vertical, column for horizontal)."""
    return rc[0] if d[0] else rc[1]


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    """Sort coordinates by their perpendicular value relative to direction d."""
    return sorted(coords, key=lambda p: _perp_value(p, d))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build the same red ray cascade, but recolor each activated blob by its body type.

    Input structure:
    - Background is the most common color; the start marker is a yellow cap above a short red column.
    - Each non-start object is a diamond/octagon blob made from one body color plus a green edge port.

    Rule:
    - Fire the thin red start ray upward; when any ray first reaches a blob body, that blob activates.
    - Activation repaints the blob body with its paired output color, then emits a thick red ray through its green port.
    - Continue breadth-first until no new blob is hit; untouched blobs and all green ports disappear to background.

    Key insight: The ray network is still read from geometry, but the recolor is no longer universal red for bodies; each body color has its own fixed activated color, so blobs must be parsed component-by-component.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the start marker (bottom vertical segment with a differently colored cap) ---
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
            if cap in (int(bg), int(col)):
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
    top_r = min(int(r) for r, _c in start_seg)
    cap_r = int(top_r) - 1
    assert cap_r >= 0 and int(g[cap_r, start_col]) == int(yellow_col)

    # --- STEP 2) Parse each remaining connected object as one blob (body color + green port color) ---
    ignore = np.zeros((H, W), dtype=bool)
    for r, c in start_seg:
        ignore[int(r), int(c)] = True
    ignore[int(cap_r), int(start_col)] = True
    object_mask = (g != int(bg)) & (~ignore)

    out = np.full_like(g, int(bg))
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    port_colors: set[int] = set()

    for coords in components_mask(object_mask, connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        cols = Counter(int(g[r, c]) for r, c in pts)
        assert len(cols) == 2, f"expected exactly 2 colors in a blob component, got {cols}"

        body_col = int(cols.most_common(1)[0][0])
        port_col = next(int(col) for col in cols if int(col) != int(body_col))
        assert body_col in BODY_TO_ACTIVE, f"unexpected body color {body_col}"
        assert cols[body_col] > cols[port_col], f"port must be the minority color: {cols}"

        body_pts = [(r, c) for (r, c) in pts if int(g[r, c]) == int(body_col)]
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in body_pts:
            for ddr, ddc in DIRS:
                rr = int(r + ddr)
                cc = int(c + ddc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(port_col):
                    seen_dirs.add((int(ddr), int(ddc)))
        assert len(seen_dirs) == 1, f"each blob must have exactly one port side, got {seen_dirs}"
        port_dir = seen_dirs.pop()
        port_pixels = [
            (int(r), int(c))
            for (r, c) in body_pts
            if 0 <= int(r + port_dir[0]) < H
            and 0 <= int(c + port_dir[1]) < W
            and int(g[int(r + port_dir[0]), int(c + port_dir[1])]) == int(port_col)
        ]
        assert len(port_pixels) in (1, 2)

        bi = int(len(blobs))
        for r, c in body_pts:
            blob_id[int(r), int(c)] = bi
        blobs.append(
            {
                "body_col": int(body_col),
                "active_col": int(BODY_TO_ACTIVE[int(body_col)]),
                "pts": body_pts,
                "ports": [{"dir": port_dir, "pixels": _sort_on_ray(port_pixels, port_dir)}],
            }
        )
        port_colors.add(int(port_col))

    assert len(port_colors) == 1, f"port color should stay fixed across blobs, got {port_colors}"

    activated: set[int] = set()
    queue: deque[int] = deque()

    def activate_blob_at(r: int, c: int) -> None:
        """Recolor a blob when first hit by a ray; enqueue it once for BFS propagation."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))
        active_col = int(blobs[bi]["active_col"])
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = active_col

    # --- STEP 3) Draw the start ray upward and stop at the first body it hits ---
    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    out[int(cap_r), int(start_col)] = int(red_col)
    for r in range(int(cap_r), -1, -1):
        out[int(r), int(start_col)] = int(red_col)
        if int(blob_id[int(r), int(start_col)]) >= 0:
            activate_blob_at(int(r), int(start_col))
            break

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        """Extend a ray from each port lane until boundary or the first body hit."""
        ddr, ddc = map(int, d)
        for pr, pc in port_pixels:
            r, c = int(pr), int(pc)
            while True:
                r += int(ddr)
                c += int(ddc)
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[int(r), int(c)] = int(red_col)
                if int(blob_id[int(r), int(c)]) >= 0:
                    activate_blob_at(int(r), int(c))
                    break

    # --- STEP 4) Propagate breadth-first through activated blobs ---
    while queue:
        bi = int(queue.popleft())
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    return out


# --------------- generator (a2_cascade, D15 body-color transfer variant) ---------------

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
CHAIN_LENGTHS = (2, 3)
CHAIN_LEN_WEIGHTS = [5, 1]
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


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    """Return sorted mask-local coordinates on the edge facing direction d."""
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _blob_spec(size: tuple[int, int], port_dir: tuple[int, int]) -> dict | None:
    """Build a color-agnostic blob specification for a given shape and port direction."""
    h, w = size
    mask = _shape_mask(h, w)
    green = _edge_offsets(mask, port_dir)
    if len(green) not in (1, 2):
        return None

    green_set = set(green)
    port_pixels: list[tuple[int, int]] = []
    for gr, gc in green:
        br, bc = gr - port_dir[0], gc - port_dir[1]
        if not (0 <= br < h and 0 <= bc < w):
            return None
        if not mask[br, bc] or (br, bc) in green_set:
            return None
        port_pixels.append((int(br), int(bc)))

    body = [(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) not in green_set]
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "size": (h, w),
        "mask": mask,
        "green": green,
        "body": body,
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
            cells.append((int(r), int(c)))
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
            cells.append((int(r), int(c)))
        while True:
            r, c = r + dr, c + dc
            if not (0 <= r < H and 0 <= c < W):
                break
            cells.append((int(r), int(c)))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    """Paint cells red in the output grid and mark them in the ray mask."""
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


def _assign_body_colors(chain_len: int) -> list[int]:
    """Choose body colors for the activated chain, usually showing both transfer pairs."""
    if chain_len == 1:
        return [int(random.choice(BODY_COLORS))]
    colors = list(BODY_COLORS)
    random.shuffle(colors)
    out = [int(colors[0]), int(colors[1])]
    while len(out) < chain_len:
        out.append(int(random.choice(BODY_COLORS)))
    random.shuffle(out)
    return out


def _render_blob_patch(spec: dict, body_col: int) -> np.ndarray:
    """Materialize a blob patch with the chosen body color and fixed green ports."""
    h, w = map(int, spec["size"])
    patch = np.full((h, w), BG, dtype=int)
    for r, c in spec["body"]:
        patch[int(r), int(c)] = int(body_col)
    for r, c in spec["green"]:
        patch[int(r), int(c)] = GREEN
    return patch


def _try_place_along_axis(
    spec: dict,
    window: list[tuple[int, int]],
    d: tuple[int, int],
    lane_vals: list[int],
    starts: list[tuple[int, int]],
    include_origin: bool,
    min_gap: int,
    occ_pad: np.ndarray,
    body_mask: np.ndarray,
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
            entry_cells = [(int(entry_line), int(c0 + c)) for _, c in window]
        else:
            entry_cells = [(int(r0 + r), int(entry_line)) for r, _ in window]
        full = _offset(spec["full"], int(r0), int(c0))
        if not _in_bounds(full, H, W):
            continue
        if any(bool(occ_pad[r, c]) or bool(ray_mask[r, c]) for r, c in full):
            continue
        path = _path_to_entries(starts, d, entry_cells, include_origin, H, W)
        if path is None:
            continue
        entry_set = set(entry_cells)
        if any(bool(body_mask[r, c]) for r, c in path if (r, c) not in entry_set):
            continue
        candidates.append(
            {
                "spec": spec,
                "r0": int(r0),
                "c0": int(c0),
                "entry": entry_cells,
                "path": path,
                "body": _offset(spec["body"], int(r0), int(c0)),
                "full": full,
                "ports": _sort_on_ray(_offset(spec["port_pixels"], int(r0), int(c0)), spec["port_dir"]),
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
    body_mask: np.ndarray,
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
                    body_mask,
                    ray_mask,
                    H,
                    W,
                )
            )

    return random.choice(candidates) if candidates else None


def _place_blob(inp: np.ndarray, occupied: np.ndarray, body_mask: np.ndarray, blob: dict) -> None:
    """Paste a blob into the input grid and update occupancy tracking."""
    patch = _render_blob_patch(blob["spec"], int(blob["body_col"]))
    paste(inp, patch, int(blob["r0"]), int(blob["c0"]), transparent=BG)
    for r, c in blob["full"]:
        occupied[int(r), int(c)] = True
    for r, c in blob["body"]:
        body_mask[int(r), int(c)] = True


def _place_distractor(
    inp: np.ndarray,
    occupied: np.ndarray,
    body_mask: np.ndarray,
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
                full = _offset(spec["full"], int(r0), int(c0))
                if any(bool(occ_pad[r, c]) or bool(ray_mask[r, c]) for r, c in full):
                    continue
                candidates.append(
                    {
                        "spec": spec,
                        "r0": int(r0),
                        "c0": int(c0),
                        "full": full,
                        "body": _offset(spec["body"], int(r0), int(c0)),
                        "body_col": int(random.choice(BODY_COLORS)),
                    }
                )
    if not candidates:
        return False
    _place_blob(inp, occupied, body_mask, random.choice(candidates))
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on placement failure."""
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    body_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    # Place start segment (yellow cap + red column at bottom).
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

    chain_len = random.choices(CHAIN_LENGTHS, weights=CHAIN_LEN_WEIGHTS)[0]
    num_distractors = random.randint(0, min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len))
    chain_colors = _assign_body_colors(chain_len)

    starts = [(int(cap_r), int(start_col))]
    direction = UP
    include_origin = True
    for i in range(chain_len):
        blob = _candidate_next_blob(H, W, starts, direction, include_origin, occupied, body_mask, ray_mask)
        if blob is None:
            return None
        blob["body_col"] = int(chain_colors[i])
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, body_mask, blob)
        paint_coords(out, blob["body"], int(BODY_TO_ACTIVE[int(blob["body_col"])]))
        starts = blob["ports"]
        direction = blob["spec"]["port_dir"]
        include_origin = False

    final_path = _path_to_boundary(starts, direction, include_origin, H, W)
    if any(bool(body_mask[r, c]) for r, c in final_path):
        return None
    _mark_ray(out, ray_mask, final_path)

    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, body_mask, ray_mask):
            return None

    activated_body_cols = {int(col) for col in chain_colors}
    if chain_len >= 2 and activated_body_cols != set(BODY_COLORS):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 D15 input")
