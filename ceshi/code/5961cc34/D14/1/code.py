# Variant of 5961cc34
# Dimension: D14 — 多步处理顺序 (multi_step)
# Change: BFS 顺序处理级联：先激活直接命中的色块，再处理它们发射的射线命中的色块，逐层扩展... → 逆向级联

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# -----------------------------------------------------------------------------
# Design analysis for the D14 reverse-cascade variant
#
# a) Concrete behavior
# - The original task marks the start with a yellow cap above the bottom red
#   stem, then propagates forward until the last blob emits a final ray.
# - This variant removes that yellow start cap. The only bottom structure is a
#   short red stem touching the bottom edge.
# - The terminal blob is marked instead: it keeps its normal green port, and a
#   yellow bar of the same thickness is placed one cell farther outward beyond
#   that port. This makes the target visually attached to a specific blob edge.
# - Output no longer includes a final ray leaving the last blob. The target blob
#   is the endpoint. The solver must first identify that terminal marker, then
#   walk backwards through the unique predecessor chain until it reaches the
#   bottom stem.
# - Small example:
#   original: stem -> ray -> blob A -> ray -> blob B -> ray to boundary
#   variant : stem -> ray -> blob A -> ray -> blob B[yellow outside its port]
#   Forward BFS would try to emit from A then B. Reverse logic instead starts by
#   declaring B the target, finds that A is the only blob whose port-ray would
#   hit B, then finds that the stem is the only source whose upward ray hits A.
#
# b) Complexity calibration
# - The blob language stays the same: blue diamond/octagon bodies with a single
#   green port side and width 1 or 2.
# - The variant signal is not a hidden ordering convention; it is visible as the
#   yellow marker moving from the bottom stem to the terminal blob.
# - To keep the reverse logic nontrivial, generated instances still use 1-3
#   activated blobs plus up to one distractor.
#
# c) Assumption audit against the original main()
# - Original start detection required "red column touching bottom + non-red cap
#   above". That is no longer valid; the new solver only requires the solid red
#   bottom column.
# - Original color discovery assumed yellow is the cap and green is the only
#   blue-adjacent auxiliary color. The variant keeps that second assumption by
#   moving yellow outside the blob, beyond the green edge, so green remains the
#   unique blue-adjacent port color.
# - Original propagation mutated state with a BFS queue. That ordering-dependent
#   machinery is replaced with a static predecessor graph: start/blob successor
#   rays are computed first, then the target-marked blob is traced backwards.
# - Original output relied on every activated blob eventually emitting a ray, so
#   its green edge became red automatically. Here the target does not emit, so
#   activation must explicitly recolor both blue body cells and green port cells.
#
# d) Placement legality
# - The generator still constructs a single clear ray chain in output-first
#   style, then inverts it to an input.
# - The target marker cells are required to be in-bounds, disjoint from existing
#   rays and blobs, and to point into empty space so the target is truly terminal.
# - Distractors are placed off all rays and off the target marker.
#
# e) Visual observability
# - Across examples, the yellow cue visibly moves from "bottom cap" to "terminal
#   blob tag". The output correspondingly stops at the marked blob instead of
#   extending beyond it.
# - Because the marker sits just beyond the target's green edge, a human can see
#   which blob is special without needing any hidden ordering rule.
#
# f) Visual cleanliness
# - Only one extra layer is added: the yellow bar directly outside one blob edge.
#   No overlapping symbols are used.
# - The target's outward side is kept clear so the marker is isolated and legible.
#
# g) Color inference
# - The palette is unchanged. Yellow has one structural role only: terminal
#   marker. Green still means port. Red still means realized path/output.
#
# New solver assumptions introduced here, and enforced by generate_input():
# - Exactly one solid red bottom stem exists.
# - Exactly one yellow target bar exists, aligned one step beyond one blob port.
# - Every activated blob has exactly one predecessor in the reverse chain, and
#   the target has no forward successor.
# -----------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the red network by tracing backward from a marked terminal blob.

    Input structure:
    - bg = most common color. Blue blobs still have one green edge-port, but the bottom now shows only a short red stem with no yellow cap.
    - One terminal blob is marked by a yellow bar placed just outside its green port, indicating the far endpoint of the cascade.

    Rule:
    - Identify the target blob from the yellow marker, then find which blob or start stem would hit it by following the directed port rays in reverse.
    - Repeat until the predecessor is the bottom stem, and paint exactly that recovered chain: the stem, the connecting rays, and each activated blob in red.

    Key insight: The geometry of the green ports still defines the same directed network, but the visible clue now marks the endpoint, so the solve order is recovered by reverse parent tracing instead of forward BFS expansion.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the bottom start stem (solid vertical red segment touching the bottom) ---
    start_candidates: list[tuple[int, int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=8):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            if c0 != c1 or (r1 - r0 + 1) < 2 or r1 != H - 1:
                continue
            if any(int(g[rr, c0]) != int(col) for rr in range(r0, r1 + 1)):
                continue
            start_candidates.append((int(col), int(c0), [(int(rr), int(c0)) for rr in range(r0, r1 + 1)]))

    assert len(start_candidates) == 1
    red_col, start_col, start_seg = start_candidates[0]

    # --- STEP 2) Identify blue, green, yellow colors under the new marker layout ---
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

    yellow_candidates = colors_present(g, exclude={bg, red_col, blue_col, green_col})
    assert len(yellow_candidates) == 1
    yellow_col = int(yellow_candidates[0])

    out = np.full_like(g, int(bg))

    # --- STEP 3) Extract blobs and a pixel->blob lookup ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = len(blobs)
        for r, c in pts:
            blob_id[r, c] = int(bi)
        blobs.append({"pts": pts, "ports": []})

    # --- STEP 4) Recover each blob's unique green port ---
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
            green_pixels = [(int(r + ddr), int(c + ddc)) for r, c in port_pixels]
            blob["ports"].append(
                {
                    "dir": (int(ddr), int(ddc)),
                    "pixels": port_pixels,
                    "green": green_pixels,
                }
            )
        assert len(blob["ports"]) == 1

    def trace_hit(starts: list[tuple[int, int]], d: tuple[int, int]) -> tuple[list[tuple[int, int]], int | None]:
        """Trace all ray lanes until boundary or first blob hit; return painted cells and the hit blob id."""
        ddr, ddc = map(int, d)
        starts = sorted(starts)
        cells: list[tuple[int, int]] = []
        hit_bi: int | None = None
        for sr, sc in starts:
            r, c = int(sr), int(sc)
            lane: list[tuple[int, int]] = []
            while True:
                r += ddr
                c += ddc
                if not (0 <= r < H and 0 <= c < W):
                    break
                lane.append((int(r), int(c)))
                if int(g[r, c]) == int(blue_col):
                    bi = int(blob_id[r, c])
                    assert bi >= 0
                    if hit_bi is None:
                        hit_bi = bi
                    else:
                        assert hit_bi == bi
                    break
            cells.extend(lane)
        return cells, hit_bi

    # --- STEP 5) Find the terminal target blob from the external yellow bar ---
    target_candidates: list[int] = []
    for bi, blob in enumerate(blobs):
        port = blob["ports"][0]
        ddr, ddc = map(int, port["dir"])
        marker_cells = [(int(gr + ddr), int(gc + ddc)) for gr, gc in port["green"]]
        if marker_cells and all(0 <= r < H and 0 <= c < W and int(g[r, c]) == int(yellow_col) for r, c in marker_cells):
            blob["marker"] = marker_cells
            target_candidates.append(int(bi))

    assert len(target_candidates) == 1
    target_bi = int(target_candidates[0])

    # --- STEP 6) Build the directed predecessor graph (start/blob successor rays) ---
    top_r = min(int(r) for r, _c in start_seg)
    start_path, start_hit = trace_hit([(int(top_r), int(start_col))], (-1, 0))
    assert start_hit is not None

    successor: dict[int, int | None] = {}
    ray_paths: dict[tuple[str, int], list[tuple[int, int]]] = {}
    incoming: dict[int, list[tuple[str, int]]] = {bi: [] for bi in range(len(blobs))}

    ray_paths[("start", -1)] = start_path
    incoming[int(start_hit)].append(("start", -1))

    for bi, blob in enumerate(blobs):
        port = blob["ports"][0]
        path, hit = trace_hit(port["pixels"], port["dir"])
        successor[int(bi)] = hit
        ray_paths[("blob", int(bi))] = path
        if hit is not None:
            incoming[int(hit)].append(("blob", int(bi)))

    assert successor[target_bi] is None

    # --- STEP 7) Reverse-trace from target back to the stem, then paint only that chain ---
    chain_rev: list[int] = []
    edge_rev: list[tuple[str, int]] = []
    cur = int(target_bi)
    seen: set[int] = set()
    while True:
        assert cur not in seen
        seen.add(cur)
        chain_rev.append(int(cur))
        preds = incoming[int(cur)]
        assert len(preds) == 1
        pred_key = preds[0]
        edge_rev.append(pred_key)
        if pred_key[0] == "start":
            break
        cur = int(pred_key[1])

    chain = list(reversed(chain_rev))
    edge_keys = list(reversed(edge_rev))

    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)

    for bi in chain:
        blob = blobs[int(bi)]
        for pr, pc in blob["pts"]:
            out[int(pr), int(pc)] = int(red_col)
        for gr, gc in blob["ports"][0]["green"]:
            out[int(gr), int(gc)] = int(red_col)

    for key in edge_keys:
        for r, c in ray_paths[key]:
            out[int(r), int(c)] = int(red_col)

    assert len(blobs) - len(chain) <= 1, "at most 1 distractor blob"
    return out


# --------------- generator (reverse target-marked cascade) ---------------

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

# --- Chain / distractor ---
CHAIN_LENGTHS = (1, 2, 3)
CHAIN_LEN_WEIGHTS = (2, 3, 3)
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

    blue = [(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) not in green_set]
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "size": (h, w),
        "mask": mask,
        "patch": patch,
        "blue": blue,
        "full": full,
        "green": [(int(r), int(c)) for r, c in green],
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
    """Trace ray lanes from starts to entries; return cell list or None on failure."""
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        if include_origin:
            cells.append((int(r), int(c)))
        guard = H + W + 10
        while (r, c) != (er, ec):
            r, c = int(r + dr), int(c + dc)
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
            r, c = int(r + dr), int(c + dc)
            if not (0 <= r < H and 0 <= c < W):
                break
            cells.append((int(r), int(c)))
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
        candidates.append(
            {
                "spec": spec,
                "r0": int(r0),
                "c0": int(c0),
                "entry": entry_cells,
                "path": path,
                "blue": _offset(spec["blue"], r0, c0),
                "full": full,
                "greens": _offset(spec["green"], r0, c0),
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
    """Find a random valid placement for the next non-terminal blob in the chain."""
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


def _candidate_target_blob(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> dict | None:
    """Find a terminal blob whose outward marker fits and whose port points into empty space."""
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
            for cand in _try_place_along_axis(
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
            ):
                dr, dc = spec["port_dir"]
                marker = [(int(gr + dr), int(gc + dc)) for gr, gc in cand["greens"]]
                if not _in_bounds(marker, H, W):
                    continue
                if any(occupied[r, c] or ray_mask[r, c] for r, c in marker):
                    continue
                tail = _path_to_boundary(marker, spec["port_dir"], True, H, W)
                if any(occupied[r, c] or ray_mask[r, c] or blue_mask[r, c] for r, c in tail):
                    continue
                cand["marker"] = marker
                candidates.append(cand)

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
                        "r0": int(r0),
                        "c0": int(c0),
                        "full": full,
                        "blue": _offset(spec["blue"], r0, c0),
                    }
                )
    if not candidates:
        return False
    _place_blob(inp, occupied, blue_mask, random.choice(candidates))
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid reverse-cascade input. Returns None on placement failure."""
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    # Place the unmarked bottom start stem.
    start_h = random.randint(START_H_MIN, START_H_MAX)
    seg_r0 = H - start_h
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    for r in range(seg_r0, H):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        occupied[r, start_col] = True

    chain_len = random.choices(CHAIN_LENGTHS, weights=CHAIN_LEN_WEIGHTS)[0]
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(0, max_dist)

    starts = [(int(seg_r0), int(start_col))]
    direction = UP
    include_origin = False

    # Place non-terminal blobs.
    for _ in range(chain_len - 1):
        blob = _candidate_next_blob(H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask)
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["blue"], RED)
        paint_coords(out, blob["greens"], RED)
        starts = blob["ports"]
        direction = blob["spec"]["port_dir"]
        include_origin = False

    # Place the marked terminal blob.
    target = _candidate_target_blob(H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask)
    if target is None:
        return None
    _mark_ray(out, ray_mask, target["path"])
    _place_blob(inp, occupied, blue_mask, target)
    paint_coords(inp, target["marker"], YELLOW)
    for r, c in target["marker"]:
        occupied[r, c] = True
    paint_coords(out, target["blue"], RED)
    paint_coords(out, target["greens"], RED)

    # Place distractors.
    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    try:
        oracle = main(inp)
    except Exception:
        return None
    if not np.array_equal(oracle, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 D14 input")
