# Variant of 5961cc34
# Dimension: D5 — 色块端口结构 (src)
# Change: 每个色块恰好一条边有1-2个相邻绿色像素作为端口，端口方向决定射线发射方向... → 端口颜色编码射线颜色

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords


# Design analysis
# a) Concrete behavior and trace:
# - The original geometry stays intact: a red start stem at the bottom shoots a
#   thin upward ray into blue diamond/octagon blobs, and each activated blob
#   emits one thick outgoing ray from its unique port side.
# - The D5 change is localized to the port bar. Instead of every blob using the
#   same green port color and every activated structure becoming red, each blob
#   has a single-color port bar chosen from a small visible palette
#   {green=3, yellow=4, orange=7}. That color determines the outgoing ray color.
# - Output rule used here:
#   * the start stem and its initial upward ray stay red;
#   * when a ray hits a blob, that blob body becomes the incoming ray color;
#   * the blob then emits its next thick ray in the color of its own port bar.
# - Example trace:
#   * Original: red start ray hits blob A, A turns red, A emits red, hits blob B,
#     B turns red, B emits red.
#   * Variant: red start ray hits blob A, A turns red, A has orange port so it
#     emits orange; orange ray hits blob B, B turns orange, B has green port so
#     it emits green. The color change happens only at activated blobs, and the
#     new outgoing color always matches the visible port marker in the input.
#
# b) Complexity calibration:
# - Only three port colors are allowed. This is enough to make the new mechanic
#   visually obvious, but small enough that humans can infer the pairing from a
#   few examples.
# - The generator biases toward chain lengths 2-3 so a sample usually contains
#   at least one color transition. For chains of length >= 2 it also forces at
#   least two distinct port colors among activated blobs.
# - No extra structural freedom is introduced: still exactly one port side per
#   blob, still width 1-2, still one start source, still at most one distractor.
#
# c) Old implicit assumptions that break under this variant:
# - The original solver assumed there was exactly one non-blue color adjacent to
#   blobs and called it `green_col`. That fails once ports may be green, yellow,
#   or orange.
# - It also hard-coded red painting for every activated blob and every port ray.
#   That only worked because color was previously invariant.
# - In this variant the solver must preserve three separate notions:
#   * start color: the red launch ray,
#   * incoming color: the color that paints a hit blob,
#   * outgoing color: the port color used by the activated blob.
# - Another subtle point: yellow may now appear both as the start cap color and
#   as a blob port color. Solver code therefore cannot globally exclude the cap
#   color from port detection; it must only use adjacency-to-blue to identify
#   port markers.
#
# d) Placement legality guarantees used by the generator:
# - Blob footprints never overlap and are padded by 8-neighborhood clearance.
# - Rays never pass through unrelated blue cells before reaching the intended
#   next blob or boundary.
# - The start column stays outside every blob bbox column span, so the initial
#   vertical ray is unambiguous.
# - Every blob still has exactly one edge-side port bar of length 1 or 2, and
#   the bar color is uniform within that blob.
#
# e) Visual observability:
# - The input visibly marks each emitting side with one of three colors, and the
#   output visibly reproduces that color on the next ray segment. Because the
#   color switch happens at the blob itself, the pairing is local and legible.
# - The generator avoids relying on hidden orderings or color ids. All decisions
#   are based on visible geometry, ray direction, and the explicitly painted port
#   colors.
#
# f) Visual cleanliness:
# - The variant does not add new symbols on top of existing rays; it only swaps
#   the port-bar color. Overlap is still constrained to the original blob-hit
#   points, so the image remains readable.
#
# g) Color-change inferability:
# - The output does not use arbitrary recolorings. The only color relation is
#   "outgoing ray color = that blob's port color", which is locally supported by
#   the port touching the emitting side.
# - Activated blob bodies take the incoming color, which makes the moment of
#   color transfer explicit and avoids ambiguous mid-air color changes.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a multicolor ray network by activating blue blobs via colored ports.

    Input structure:
    - bg = most common color. Large blue blobs have exactly one edge-side port,
      shown as 1-2 adjacent non-blue pixels of one uniform color.
    - Start marker: a yellow cap above a short red segment at the grid bottom.

    Rule:
    - Extend a thin red ray upward from the start. When it hits a blue blob, the
      blob is painted with that incoming ray color.
    - Each activated blob emits a thick ray through its port side. The outgoing
      ray color matches the visible port color in the input.
    - Continue the cascade until no new blobs are reached; erase untouched blobs
      and all port markers to background.

    Key insight: Port orientation still determines direction, but port color now
    determines the next ray color; the network is solved by tracking both shape
    connectivity and color handoff at each activated blob.
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
    red_col, cap_col, start_col, start_seg = start_candidates[0]

    # --- STEP 2) Identify blue as the dominant non-start body color ---
    counts = Counter(map(int, g.flatten()))
    other_cols = [
        (int(cnt), int(col))
        for col, cnt in counts.items()
        if int(col) not in (bg, red_col, cap_col)
    ]
    assert other_cols
    max_count = max(cnt for cnt, _col in other_cols)
    blue_candidates = [int(col) for cnt, col in other_cols if cnt == max_count]
    assert len(blue_candidates) == 1
    blue_col = int(blue_candidates[0])

    # Port colors are any non-bg, non-red, non-blue colors that touch blue.
    port_color_candidates: set[int] = set()
    for col in colors_present(g, exclude={bg, red_col, blue_col}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        for r, c in pts:
            r = int(r)
            c = int(c)
            if (
                (r > 0 and int(g[r - 1, c]) == blue_col)
                or (r + 1 < H and int(g[r + 1, c]) == blue_col)
                or (c > 0 and int(g[r, c - 1]) == blue_col)
                or (c + 1 < W and int(g[r, c + 1]) == blue_col)
            ):
                port_color_candidates.add(int(col))
                break

    assert port_color_candidates
    port_colors = set(map(int, port_color_candidates))

    out = np.full_like(g, bg)

    # --- STEP 3) Extract blue blobs and build a pixel->blob map ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []

    for coords in components_mask(g == blue_col, connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = len(blobs)
        for r, c in pts:
            blob_id[r, c] = bi
        blobs.append({"pts": pts})

    # --- STEP 4) For each blob, find its unique colored port ---
    def sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
        return sorted(coords, key=lambda p: p[1] if d[0] else p[0])

    for blob in blobs:
        grouped: dict[tuple[tuple[int, int], int], set[tuple[int, int]]] = {}
        for r, c in blob["pts"]:
            for ddr, ddc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + ddr
                cc = c + ddc
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                col = int(g[rr, cc])
                if col not in port_colors:
                    continue
                key = ((int(ddr), int(ddc)), col)
                grouped.setdefault(key, set()).add((int(r), int(c)))

        assert len(grouped) == 1, f"Each blob must have one colored port, got {grouped.keys()}"
        (port_dir, port_color), port_pixels = next(iter(grouped.items()))
        blob["port"] = {
            "dir": tuple(map(int, port_dir)),
            "color": int(port_color),
            "pixels": sort_on_ray(list(port_pixels), tuple(map(int, port_dir))),
        }

    activated: set[int] = set()
    queue: deque[int] = deque()

    def activate_blob_at(r: int, c: int, fill_col: int) -> None:
        """Paint a blob with the incoming ray color and enqueue it once."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(bi)
        queue.append(bi)
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(fill_col)

    # --- STEP 5) Draw the start ray upward (thin, always red) ---
    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, int(start_col)]) == int(cap_col)
    out[top_r - 1, int(start_col)] = int(red_col)
    for r in range(top_r - 1, -1, -1):
        out[int(r), int(start_col)] = int(red_col)
        if int(g[int(r), int(start_col)]) == int(blue_col):
            activate_blob_at(int(r), int(start_col), int(red_col))
            break

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int], ray_col: int) -> None:
        """Extend a ray from each port-adjacent blue cell until boundary or blob hit."""
        ddr, ddc = map(int, d)
        for pr, pc in port_pixels:
            r, c = int(pr), int(pc)
            while True:
                r += ddr
                c += ddc
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[int(r), int(c)] = int(ray_col)
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c), int(ray_col))
                    break

    # --- STEP 6) Propagate (BFS): activated blob -> emit colored ray -> activate more blobs ---
    while queue:
        bi = int(queue.popleft())
        port = blobs[bi]["port"]
        extend_thick(port["pixels"], port["dir"], port["color"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"

    return out


# --------------- generator (a2_cascade, D5 port-color variant) ---------------

# --- Colors ---
BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
PORT_COLORS = (GREEN, YELLOW, ORANGE)

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
CHAIN_LEN_WEIGHTS = [1, 2, 4, 5]  # emphasize color transitions via longer chains
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


def _blob_geom(size: tuple[int, int], port_dir: tuple[int, int]) -> dict | None:
    """Build the geometric specification for one blob orientation."""
    h, w = size
    mask = _shape_mask(h, w)
    markers = _edge_offsets(mask, port_dir)
    if len(markers) not in (1, 2):
        return None

    marker_set = set(markers)
    emitters: list[tuple[int, int]] = []
    for mr, mc in markers:
        br, bc = mr - port_dir[0], mc - port_dir[1]
        if not (0 <= br < h and 0 <= bc < w):
            return None
        if not mask[br, bc] or (br, bc) in marker_set:
            return None
        emitters.append((int(br), int(bc)))

    body = [
        (int(r), int(c))
        for r, c in np.argwhere(mask)
        if (int(r), int(c)) not in marker_set
    ]
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "size": (h, w),
        "mask": mask,
        "body": body,
        "full": full,
        "port_dir": port_dir,
        "markers": _sort_on_ray(list(markers), port_dir),
        "emitters": _sort_on_ray(emitters, port_dir),
    }


def _all_geoms(port_dir: tuple[int, int] | None = None) -> list[dict]:
    """Generate all valid blob geometries, optionally filtered by port direction."""
    dirs = [port_dir] if port_dir is not None else list(DIRS)
    out = [g for size in SHAPES for d in dirs if (g := _blob_geom(size, d)) is not None]
    random.shuffle(out)
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    """Dilate a boolean mask by 1 pixel in all 8 directions."""
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), h + min(0, dr)
            c0, c1 = max(0, dc), w + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    """Translate local offsets to global coordinates."""
    return [(r0 + r, c0 + c) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _path_to_entries(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    entries: list[tuple[int, int]],
    include_origin: bool,
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    """Trace ray lanes from starts to entries; return cell list or None on failure."""
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        if include_origin:
            cells.append((r, c))
        guard = h + w + 10
        while (r, c) != (er, ec):
            r, c = r + dr, c + dc
            guard -= 1
            if guard < 0 or not (0 <= r < h and 0 <= c < w):
                return None
            cells.append((r, c))
    return cells


def _path_to_boundary(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    """Trace ray lanes from starts to grid boundary."""
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        if include_origin:
            cells.append((r, c))
        while True:
            r, c = r + dr, c + dc
            if not (0 <= r < h and 0 <= c < w):
                break
            cells.append((r, c))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    """Paint cells in the output grid and mark them in the ray mask."""
    paint_coords(out, cells, int(color))
    paint_coords(ray_mask, cells, True)


def _try_place_along_axis(
    geom: dict,
    window: list[tuple[int, int]],
    d: tuple[int, int],
    lane_vals: list[int],
    starts: list[tuple[int, int]],
    include_origin: bool,
    min_gap: int,
    occ_pad: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    h: int,
    w: int,
) -> list[dict]:
    """Find all valid placements for a blob geometry along the ray's travel axis."""
    gh, gw = geom["mask"].shape
    win_vals = [_perp_value(p, d) for p in window]
    edge_line = _along_value(window[0], d)
    start_line = _along_value(starts[0], d)
    is_vertical = bool(d[0])

    fixed = lane_vals[0] - win_vals[0]
    extent = h - gh if is_vertical else w - gw
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
        full = _offset(geom["full"], r0, c0)
        if not _in_bounds(full, h, w):
            continue
        if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
            continue
        path = _path_to_entries(starts, d, entry_cells, include_origin, h, w)
        if path is None:
            continue
        entry_set = set(entry_cells)
        if any(blue_mask[r, c] for r, c in path if (r, c) not in entry_set):
            continue
        candidates.append(
            {
                "geom": geom,
                "r0": r0,
                "c0": c0,
                "entry": entry_cells,
                "path": path,
                "body": _offset(geom["body"], r0, c0),
                "full": full,
                "markers": _offset(geom["markers"], r0, c0),
                "ports": _sort_on_ray(_offset(geom["emitters"], r0, c0), geom["port_dir"]),
            }
        )

    return candidates


def _candidate_next_blob(
    h: int,
    w: int,
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

    for geom in _all_geoms():
        if geom["port_dir"] == entry_side:
            continue
        edge = _edge_offsets(geom["mask"], entry_side)
        if len(edge) < thickness:
            continue
        for k in range(len(edge) - thickness + 1):
            window = edge[k : k + thickness]
            win_delta = [_perp_value(p, d) - _perp_value(window[0], d) for p in window]
            if win_delta != lane_delta:
                continue
            candidates.extend(
                _try_place_along_axis(
                    geom,
                    window,
                    d,
                    lane_vals,
                    starts,
                    include_origin,
                    min_gap,
                    occ_pad,
                    blue_mask,
                    ray_mask,
                    h,
                    w,
                )
            )

    return random.choice(candidates) if candidates else None


def _paint_blob_input(inp: np.ndarray, blob: dict) -> None:
    """Draw one blob into the input grid using blue body + colored port markers."""
    paint_coords(inp, blob["body"], BLUE)
    paint_coords(inp, blob["markers"], blob["port_color"])


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    """Paste a blob into the input grid and update occupancy tracking."""
    _paint_blob_input(inp, blob)
    for r, c in blob["full"]:
        occupied[r, c] = True
    for r, c in blob["body"]:
        blue_mask[r, c] = True


def _place_distractor(
    inp: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> bool:
    """Try to place one distractor blob in free space."""
    h, w = inp.shape
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []
    for geom in _all_geoms():
        gh, gw = geom["mask"].shape
        for r0 in range(h - gh + 1):
            for c0 in range(w - gw + 1):
                full = _offset(geom["full"], r0, c0)
                if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
                    continue
                candidates.append(
                    {
                        "geom": geom,
                        "r0": r0,
                        "c0": c0,
                        "full": full,
                        "body": _offset(geom["body"], r0, c0),
                        "markers": _offset(geom["markers"], r0, c0),
                        "port_color": random.choice(PORT_COLORS),
                    }
                )
    if not candidates:
        return False
    _place_blob(inp, occupied, blue_mask, random.choice(candidates))
    return True


def _choose_port_colors(chain_len: int) -> list[int]:
    """Choose the port colors for the activated chain with repeated visual support."""
    if chain_len == 0:
        return []
    if chain_len == 1:
        return [random.choice(PORT_COLORS)]

    palette_size = 2 if chain_len == 2 else random.choice((2, 3))
    palette = random.sample(list(PORT_COLORS), palette_size)
    colors = palette.copy()
    while len(colors) < chain_len:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return colors[:chain_len]


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on placement failure."""
    h = random.randint(H_MIN, H_MAX)
    w = random.randint(W_MIN, W_MAX)
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    blue_mask = np.zeros((h, w), dtype=bool)
    ray_mask = np.zeros((h, w), dtype=bool)

    # Place start segment (yellow cap + red column at bottom)
    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = h - start_h - 1
    start_col = random.randint(START_COL_PAD, w - START_COL_PAD - 1)
    inp[cap_r, start_col] = YELLOW
    out[cap_r, start_col] = RED
    ray_mask[cap_r, start_col] = True
    occupied[cap_r, start_col] = True
    for r in range(cap_r + 1, h):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        occupied[r, start_col] = True

    # Decide chain length and distractor count
    chain_len = random.choices(range(len(CHAIN_LEN_WEIGHTS)), weights=CHAIN_LEN_WEIGHTS)[0]
    min_dist = 1 if chain_len == 0 else 0
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(min_dist, max_dist)
    port_colors = _choose_port_colors(chain_len)

    # Build cascade chain
    starts = [(cap_r, start_col)]
    direction = UP
    include_origin = True
    current_color = RED
    for step in range(chain_len):
        blob = _candidate_next_blob(h, w, starts, direction, include_origin, occupied, blue_mask, ray_mask)
        if blob is None:
            return None
        blob["port_color"] = int(port_colors[step])
        _mark_ray(out, ray_mask, blob["path"], current_color)
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["body"], current_color)
        starts = blob["ports"]
        direction = blob["geom"]["port_dir"]
        current_color = int(blob["port_color"])
        include_origin = False

    # Final ray to grid boundary
    final_path = _path_to_boundary(starts, direction, include_origin, h, w)
    if any(blue_mask[r, c] for r, c in final_path):
        return None
    _mark_ray(out, ray_mask, final_path, current_color)

    # Place distractors
    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    # Oracle check
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 D5 input")


# Hidden-assumption review for the new solver:
# - `main()` assumes exactly one start stem; the generator creates one bottom
#   vertical red segment with a single non-red cap and no other similar segment.
# - Blue must remain the dominant non-start body color; the generator keeps blobs
#   large and port bars short so blue pixel count wins cleanly.
# - Each blue component must have exactly one adjacent (direction, color) port
#   group; the generator constructs blobs from a single chosen edge bar only.
# - Rays are assumed not to cross unrelated blobs in empty space; the generator's
#   placement checks enforce a clear corridor from every emitter set to its next
#   target or the boundary.
# - Because a blob is filled by incoming color but emits by its own port color,
#   queue order only needs blob ids, not colors; the generator guarantees each
#   blob is activated at most once along a unique intended cascade.
