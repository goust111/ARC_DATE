# Combo variant of 5961cc34 (index 25)
# combo_spec: D7/0+D15/1
# Name: 保留色块轮廓+多色色块

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# -----------------------------------------------------------------------------
# Design analysis for combo D7/0 + D15/1
#
# a) Concrete behavior and trace
# - The seed task still contains one bottom start marker, several diamond/octagon
#   blobs, and one green edge port per blob. Rays still propagate exactly as before.
# - D15/1 changes the blob-body role: bodies are no longer all blue. This combo
#   uses two visible body families:
#     * blue body (1)   -> pink interior (6) after activation
#     * orange body (7) -> maroon interior (9) after activation
# - D7/0 then changes how an activated blob is rendered: the whole body is not
#   flattened to one fill. Instead, only the body's interior pixels switch to the
#   mapped active color, while the body's boundary pixels keep their original body
#   color. Green port pixels are not preserved; they still disappear unless a ray
#   travels through them.
# - Example trace:
#     input: start ray rises into a blue blob A, then A emits right into an orange
#            blob B.
#     seed output: both A and B become solid red.
#     D15-only output: A becomes solid pink, B solid maroon.
#     combo output: A becomes blue-rimmed with pink interior; B becomes
#                   orange-rimmed with maroon interior; rays remain red.
# - The two dimensions interact on the same object. The solver must both identify
#   each blob's body color and split that same body into boundary vs interior.
#
# b) Complexity calibration
# - Only two body types are used. That is enough to make the color-role transfer
#   obvious while keeping the mapping small enough for human inference.
# - The generator forces activated chains of length 2 or 3 and assigns both body
#   types somewhere on the chain, so each sample usually exhibits both mappings.
# - To make the outline-retention change visible, samples are rejected unless the
#   activated blobs contribute a non-trivial number of interior pixels; otherwise
#   the output could look too similar to a plain recolor.
#
# c) Implicit-assumption audit
# - Original assumption: "the dominant non-start color is the one blob color".
#   Broken by D15/1. The combo solver instead parses each non-start component as a
#   two-color object and identifies the majority color as that blob's body.
# - Original assumption: "ray hit iff grid cell == blue". Broken by D15/1. The
#   combo solver builds a body-pixel -> blob-id map, so any supported body color
#   can be hit.
# - Original assumption: "activation paints every body pixel the same color".
#   Broken by D7/0. Activation now needs the blob's local body boundary and body
#   interior separately.
# - Original assumption: writing the ray color onto the hit cell is harmless.
#   Under the combo, that would corrupt preserved outlines/interiors. The new ray
#   loops check for a body hit before painting that cell red.
# - New combo-specific assumption: each blob component contains exactly one body
#   color plus one shared port color, and the port color remains the minority. The
#   generator enforces this by construction.
#
# d) Placement legality
# - Geometry remains path-first from the seed task: the chain is laid out along
#   clear lanes, then distractors are placed off all ray paths.
# - Full blob silhouettes are kept in-bounds and 8-separated by an occupied-mask
#   dilation. Rays are forbidden from crossing any pre-existing body cells except
#   the intended terminal entry cells.
# - Because preserved outlines depend on the body shape, the generator keeps using
#   the canonical symmetric diamond/octagon family so the boundary is well defined.
#
# e) Visual observability
# - A human can still read causality from the unchanged red ray network.
# - The added rule is visible from repeated paired evidence:
#     blue blob   -> blue rim + pink fill
#     orange blob -> orange rim + maroon fill
# - This inference uses only visible structure and repeated color pairs, not color
#   ids or hidden ordering.
#
# f) Visual cleanliness
# - Only one extra layer is added inside activated blobs. Rays remain red and ports
#   disappear or become red if traversed, so the scene does not accumulate three
#   competing overlays in the same cells.
# - Distractors still vanish in the output, which keeps attention on the activated
#   chain and makes the rim-vs-fill distinction legible.
#
# g) Change inferability / anti-degeneration checks
# - The generator rejects samples where the activated chain would use only one body
#   type or where preserved interiors are too sparse. This prevents the combo from
#   collapsing into "just D15 color transfer" or a nearly solid-outline picture.
# - A useful quantitative witness is:
#     * `outline_delta_vs_D15`: combo-vs-D15 output pixel difference count
#     * `fill_palette_size`: number of distinct activated interior colors visible
#   Healthy samples have `outline_delta_vs_D15 > 0` and usually
#   `fill_palette_size == 2`.
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
# Keep both chain-length branches visibly present in the training distribution.
# Overweighting length-2 chains makes the length-3 cascade too rare for review
# and weakens evidence that BFS propagation can continue beyond one relay.
CHAIN_LEN_WEIGHTS = [1, 1]
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1


# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3


# --- Observability guard ---
MIN_TOTAL_ACTIVATED_INTERIOR = 6


# --- Retry ---
GEN_RETRIES = 300


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    """Coordinate perpendicular to direction d (column for vertical, row for horizontal)."""
    return rc[1] if d[0] else rc[0]


def _along_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    """Coordinate along direction d (row for vertical, column for horizontal)."""
    return rc[0] if d[0] else rc[1]


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    """Sort coordinates by their perpendicular value relative to direction d."""
    return sorted(coords, key=lambda p: _perp_value(p, d))


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


def _outline_offsets(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return mask-local cells on the 4-neighbor outline of the shape."""
    H, W = mask.shape
    out: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if any(
            not (0 <= rr + dr < H and 0 <= cc + dc < W) or not bool(mask[rr + dr, cc + dc])
            for dr, dc in DIRS
        ):
            out.append((rr, cc))
    return out


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    """Translate local offsets to global coordinates."""
    return [(r0 + r, c0 + c) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in coords)


def _blob_spec(size: tuple[int, int], port_dir: tuple[int, int]) -> dict | None:
    """Build a blob specification for a given shape and port direction."""
    h, w = size
    full_mask = _shape_mask(h, w)
    green = _edge_offsets(full_mask, port_dir)
    if len(green) not in (1, 2):
        return None

    green_set = set(green)
    body_mask = full_mask.copy()
    for r, c in green:
        body_mask[int(r), int(c)] = False

    port_pixels: list[tuple[int, int]] = []
    for gr, gc in green:
        br, bc = gr - port_dir[0], gc - port_dir[1]
        if not (0 <= br < h and 0 <= bc < w):
            return None
        if not bool(body_mask[br, bc]):
            return None
        port_pixels.append((int(br), int(bc)))

    body = [(int(r), int(c)) for r, c in np.argwhere(body_mask)]
    body_outline = _outline_offsets(body_mask)
    body_outline_set = set(body_outline)
    body_interior = [(r, c) for (r, c) in body if (r, c) not in body_outline_set]
    full = [(int(r), int(c)) for r, c in np.argwhere(full_mask)]
    if not body_interior:
        return None
    return {
        "size": (h, w),
        "full_mask": full_mask,
        "body_mask": body_mask,
        "green": green,
        "body": body,
        "body_outline": body_outline,
        "body_interior": body_interior,
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
    """Choose body colors for the activated chain, forcing both mappings to appear."""
    assert chain_len >= 2
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
    h, w = spec["full_mask"].shape
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
                "body_outline": _offset(spec["body_outline"], int(r0), int(c0)),
                "body_interior": _offset(spec["body_interior"], int(r0), int(c0)),
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
        edge = _edge_offsets(spec["body_mask"], entry_side)
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
        h, w = spec["full_mask"].shape
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch the cascade, but activated multi-color blobs keep their body-colored rims.

    Input structure:
    - Background is the dominant color; the start marker is a yellow cap above a short red column.
    - Each non-start object is one connected blob made of one body color plus a green edge port.
    - Body colors come from a small fixed family, and the green port still encodes outgoing direction and width.

    Rule:
    - Fire the thin start ray upward; when any ray first reaches a blob body, that blob activates.
    - Activation keeps the blob body's boundary in its original body color, recolors only the body's interior to that body's paired active color, and then emits a thick red ray through the green port.
    - Continue breadth-first until no new blob is hit; untouched blobs vanish, and only traversed ray cells remain red.

    Key insight: Geometry still determines which blobs activate, but output color is now a two-part rendering rule on each activated blob: parse the blob component-by-component to recover its body color, then split that body into rim vs interior before painting.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the start marker (bottom vertical segment with a differently colored cap). ---
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

    # --- STEP 2) Parse each remaining connected object as one blob (body color + green port color). ---
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

        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        body_mask_local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        body_pts = [(r, c) for (r, c) in pts if int(g[r, c]) == int(body_col)]
        for r, c in body_pts:
            body_mask_local[int(r - r0), int(c - c0)] = True

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

        body_outline = _offset(_outline_offsets(body_mask_local), int(r0), int(c0))
        body_outline_set = set(body_outline)
        body_interior = [(r, c) for (r, c) in body_pts if (r, c) not in body_outline_set]

        bi = int(len(blobs))
        for r, c in body_pts:
            blob_id[int(r), int(c)] = bi
        blobs.append(
            {
                "body_col": int(body_col),
                "active_col": int(BODY_TO_ACTIVE[int(body_col)]),
                "body_pts": body_pts,
                "body_outline": body_outline,
                "body_interior": body_interior,
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
        paint_coords(out, blobs[bi]["body_outline"], int(blobs[bi]["body_col"]))
        paint_coords(out, blobs[bi]["body_interior"], int(blobs[bi]["active_col"]))

    # --- STEP 3) Draw the start ray upward and stop at the first body it hits. ---
    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    out[int(cap_r), int(start_col)] = int(red_col)
    for r in range(int(cap_r) - 1, -1, -1):
        if int(blob_id[int(r), int(start_col)]) >= 0:
            activate_blob_at(int(r), int(start_col))
            break
        out[int(r), int(start_col)] = int(red_col)

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
                if int(blob_id[int(r), int(c)]) >= 0:
                    activate_blob_at(int(r), int(c))
                    break
                out[int(r), int(c)] = int(red_col)

    # --- STEP 4) Propagate breadth-first through activated blobs. ---
    while queue:
        bi = int(queue.popleft())
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    return out


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
    activated_blobs: list[dict] = []

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
        paint_coords(out, blob["body_outline"], int(blob["body_col"]))
        paint_coords(out, blob["body_interior"], int(BODY_TO_ACTIVE[int(blob["body_col"])]))
        activated_blobs.append(blob)
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
    total_interior = sum(len(blob["body_interior"]) for blob in activated_blobs)
    if activated_body_cols != set(BODY_COLORS):
        return None
    if total_interior < MIN_TOTAL_ACTIVATED_INTERIOR:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo D7/0+D15/1 input")
