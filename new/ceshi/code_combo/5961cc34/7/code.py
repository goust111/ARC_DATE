# Combo variant of 5961cc34 (index 7)
# combo_spec: D3/2+D9/2
# Name: 宽度由色块大小决定+统一缩放形状

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Keep the original object roles: azure background, one yellow-over-red start marker,
#      blue blobs, and green pixels that indicate the firing side.
#    - D9/2 changes the blob family: every blob is the same 4x4 diamond/octagon template
#          .##.
#          ####
#          ####
#          .##.
#      scaled by 1x/2x/3x with block replication, so full blue+green silhouettes are only
#      4x4, 8x8, or 12x12.
#    - D3/2 changes the beam-width rule: after activation, width is not the green-bar
#      length. Instead, width equals the blob span perpendicular to travel, implemented as
#      one outgoing lane per occupied row/column on the firing frontier.
#    - Example trace:
#      * Original task: a right-facing 8x8 blob could emit only 2 lanes because its green
#        bar was length 2.
#      * This combo: a scale-2 blob has a full 8-row silhouette, so the same right-facing
#        blob emits 8 lanes, while its green edge is only 4 pixels long.
#      * Difference: the solver must read direction from the green side but width from the
#        scaled silhouette; the generator must align later blobs against stepped 4/8/12-lane
#        frontiers rather than against a short 1/2-lane port bar.
# b) Complexity calibration
#    - Only three scales are allowed, which keeps the rule inferable from a few examples.
#      Width therefore comes from the small repeated set {4, 8, 12}, while green-edge
#      lengths come from {2, 4, 6}. The mismatch is large enough to be visually obvious.
#    - Most samples still contain at least one activated blob and at most one distractor,
#      preserving the original cascade readability.
# c) Implicit-assumption audit
#    - Original `main()` assumed green adjacency was enough to recover both direction and
#      width; that breaks because width no longer equals the green count.
#    - Original blob extraction on blue-only components hid the true scaled silhouette. The
#      combo solver must parse blue+green components and validate them against the common
#      scaled template, otherwise scale inference is ambiguous.
#    - The generator can no longer use flat port bars as incoming windows. It must match
#      rays against the stepped frontier of the scaled silhouette, because thick beams can
#      enter a larger blob through any contiguous subset of its rows/columns.
# d) Placement legality
#    - Generation stays output-first. Each candidate blob is accepted only if its full
#      blue+green footprint is in bounds, 8-neighbor separated from existing blobs, its
#      incoming lanes reach the chosen frontier cells without crossing earlier blue bodies,
#      and its outgoing direction differs from the entry side so the chain can continue.
# e) Visual observability
#    - Human-solvable evidence comes from two repeated cues: all blobs are clearly scaled
#      copies of one motif, and activated large blobs emit visibly wider rays than small
#      blobs even though the green side is only half as long.
# f) Visual cleanliness
#    - The construction still uses axis-aligned rays, a single active cascade, one port side
#      per blob, and off-ray distractors. That keeps overlays limited to the intended ray on
#      the activated blob itself.
# g) Change inferability
#    - No extra colors are introduced. Green still signals direction, blue still marks
#      inactive blob material, and red still marks activated matter or traversed rays. The
#      new inference burden is purely geometric: common-template scaling plus size-derived
#      beam width.
# - Post-change solver assumptions and generator guarantees:
#   `main()` now assumes every blue+green component is exactly one scaled copy of the base
#   template with the green cells occupying one full side. `generate_input()` enforces that,
#   guarantees at least two blobs with different scales somewhere in the scene, and
#   guarantees that at least one activated blob has scale > 1 so the fused behavior is not
#   visually degenerate.

from collections import Counter
import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


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

# --- Unified shape template ---
BASE_MASK = np.array(
    [
        [0, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 0],
    ],
    dtype=bool,
)
SCALES = (1, 2, 3)
SCALE_WEIGHTS = {1: 6, 2: 4, 3: 1}

# --- Canvas bounds ---
H_MIN, H_MAX = 13, 27
W_MIN, W_MAX = 15, 27
START_COL_PAD = 2

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3

# --- Chain / distractor ---
CHAIN_LENGTHS = (1, 2, 3)
CHAIN_LEN_WEIGHTS = [3, 3, 3]
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1
MIN_TOTAL_BLOBS = 2

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 800


def _scaled_mask(scale: int) -> np.ndarray:
    """Return the common blob template scaled by an integer factor."""
    return np.kron(BASE_MASK, np.ones((int(scale), int(scale)), dtype=bool))


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    """Coordinate perpendicular to direction d (column for vertical, row for horizontal)."""
    return int(rc[1]) if d[0] else int(rc[0])


def _along_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    """Coordinate along direction d (row for vertical, column for horizontal)."""
    return int(rc[0]) if d[0] else int(rc[1])


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    """Sort coordinates by their perpendicular value relative to direction d."""
    return sorted(coords, key=lambda p: _perp_value(p, d))


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    """Return mask-local coordinates on the extreme side facing direction d."""
    pts = np.argwhere(mask)
    axis = 0 if d[0] else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _lane_frontier(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    """One occupied frontier cell per row/column, using the furthest cell in direction d."""
    buckets: dict[int, list[int]] = {}
    for r, c in coords:
        key = int(c) if d[0] else int(r)
        val = int(r) if d[0] else int(c)
        buckets.setdefault(key, []).append(val)

    out: list[tuple[int, int]] = []
    for key in sorted(buckets):
        vals = buckets[key]
        chosen = min(vals) if d in (UP, LEFT) else max(vals)
        if d[0]:
            out.append((int(chosen), int(key)))
        else:
            out.append((int(key), int(chosen)))
    return _sort_on_ray(out, d)


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    """Translate local offsets to global coordinates."""
    return [(int(r0 + r), int(c0 + c)) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    """Return True iff every coordinate lies inside an HxW grid."""
    return all(0 <= int(r) < H and 0 <= int(c) < W for r, c in coords)


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
    """Trace ray lanes from starts to target entry cells; return all traversed cells."""
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
    """Trace ray lanes from starts until they leave the grid."""
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
    """Paint cells red in the output grid and record them in the ray mask."""
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


def _blob_spec(scale: int, port_dir: tuple[int, int]) -> dict:
    """Build one scaled blob specification for the given port direction."""
    mask = _scaled_mask(scale)
    h, w = mask.shape
    green = _edge_offsets(mask, port_dir)
    green_set = set(green)

    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    for r, c in green:
        patch[int(r), int(c)] = GREEN

    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    blue = [(r, c) for r, c in full if (r, c) not in green_set]
    emit_starts = _lane_frontier(full, port_dir)
    entry_fronts = {d: _lane_frontier(full, d) for d in DIRS}

    return {
        "scale": int(scale),
        "mask": mask,
        "patch": patch,
        "full": full,
        "blue": blue,
        "green": green,
        "port_dir": port_dir,
        "emit_starts": emit_starts,
        "entry_fronts": entry_fronts,
    }


def _all_specs(port_dir: tuple[int, int] | None = None) -> list[dict]:
    """Generate all scaled blob specs, with mild weighting toward smaller scales."""
    dirs = [port_dir] if port_dir is not None else list(DIRS)
    specs: list[dict] = []
    for scale in SCALES:
        for _ in range(SCALE_WEIGHTS[int(scale)]):
            for d in dirs:
                specs.append(_blob_spec(int(scale), d))
    random.shuffle(specs)
    return specs


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
    """Find placements whose stepped entry frontier matches the incoming lanes."""
    h, w = spec["mask"].shape
    is_vertical = bool(d[0])

    if is_vertical:
        c0 = int(lane_vals[0] - _perp_value(window[0], d))
        if any(c0 + int(c) != lane for (_, c), lane in zip(window, lane_vals)):
            return []
        varying = list(range(H - h + 1))
    else:
        r0_fixed = int(lane_vals[0] - _perp_value(window[0], d))
        if any(r0_fixed + int(r) != lane for (r, _), lane in zip(window, lane_vals)):
            return []
        varying = list(range(W - w + 1))

    random.shuffle(varying)
    candidates: list[dict] = []

    for origin_along in varying:
        r0 = int(origin_along) if is_vertical else int(r0_fixed)
        c0 = int(c0) if is_vertical else int(origin_along)

        full = _offset(spec["full"], r0, c0)
        if not _in_bounds(full, H, W):
            continue
        if any(bool(occ_pad[r, c]) or bool(ray_mask[r, c]) for r, c in full):
            continue

        entry_cells = _offset(window, r0, c0)
        ok_gap = True
        for start, entry in zip(starts, entry_cells):
            delta = int(_along_value(entry, d) - _along_value(start, d))
            if d in (UP, LEFT):
                if delta > -min_gap:
                    ok_gap = False
                    break
            else:
                if delta < min_gap:
                    ok_gap = False
                    break
        if not ok_gap:
            continue

        path = _path_to_entries(starts, d, entry_cells, include_origin, H, W)
        if path is None:
            continue
        entry_set = set(entry_cells)
        if any(bool(blue_mask[r, c]) for r, c in path if (r, c) not in entry_set):
            continue

        candidates.append(
            {
                "spec": spec,
                "r0": int(r0),
                "c0": int(c0),
                "entry": entry_cells,
                "path": path,
                "full": full,
                "blue": _offset(spec["blue"], r0, c0),
                "scale": int(spec["scale"]),
                "emit_starts": _sort_on_ray(_offset(spec["emit_starts"], r0, c0), spec["port_dir"]),
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
    lane_delta = [int(v - lane_vals[0]) for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    candidates: list[dict] = []

    for spec in _all_specs():
        if spec["port_dir"] == entry_side:
            continue
        edge = spec["entry_fronts"][entry_side]
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
    """Paste a blob onto the input grid and update occupancy trackers."""
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[int(r), int(c)] = True
    for r, c in blob["blue"]:
        blue_mask[int(r), int(c)] = True


def _place_distractor(
    inp: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> dict | None:
    """Try to place one off-ray distractor blob. Return the blob dict on success."""
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []
    for spec in _all_specs():
        h, w = spec["mask"].shape
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                full = _offset(spec["full"], r0, c0)
                if any(bool(occ_pad[r, c]) or bool(ray_mask[r, c]) for r, c in full):
                    continue
                candidates.append(
                    {
                        "spec": spec,
                        "r0": int(r0),
                        "c0": int(c0),
                        "full": full,
                        "blue": _offset(spec["blue"], r0, c0),
                        "scale": int(spec["scale"]),
                    }
                )
    if not candidates:
        return None
    blob = random.choice(candidates)
    _place_blob(inp, occupied, blue_mask, blob)
    return blob


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red beam cascade whose widths come from scaled copies of one blob template.

    Input structure:
    - The background is the majority color; every blob is a blue+green scaled copy of one
      4x4 diamond/octagon motif, and the green cells occupy exactly one full side.
    - A yellow cap above a short red bottom segment marks the unique starting launcher.

    Rule:
    - Fire the thin starting ray upward and activate the first blue blob it reaches.
    - Each activated blob turns fully red and emits in its green-marked direction.
    - The outgoing width is the blob's full span perpendicular to travel, so larger
      scaled blobs launch more parallel lanes than smaller ones, and the BFS cascade
      continues until no new blob is hit.

    Key insight:
    - Green tells you which way to fire, but not how thick to fire. Width must be read
      from the common scaled silhouette: all blobs share one template, and scale controls
      the 4/8/12-lane outgoing beams.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the start marker (red vertical segment with a non-red cap above) ---
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

    # --- STEP 2) Identify the blob colors ---
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

    # --- STEP 3) Parse blue+green components as scaled copies of the common template ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    full_mask = (g == int(blue_col)) | (g == int(green_col))

    for coords in components_mask(full_mask, connectivity=8):
        pts_full = [(int(r), int(c)) for r, c in coords]
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert h == w and h % BASE_MASK.shape[0] == 0
        scale = int(h // BASE_MASK.shape[0])
        assert scale in SCALES

        local_full = np.zeros((h, w), dtype=bool)
        local_green: set[tuple[int, int]] = set()
        blue_pts: list[tuple[int, int]] = []
        for r, c in pts_full:
            lr = int(r - r0)
            lc = int(c - c0)
            local_full[lr, lc] = True
            if int(g[r, c]) == int(green_col):
                local_green.add((lr, lc))
            else:
                blue_pts.append((int(r), int(c)))

        mask = _scaled_mask(scale)
        assert mask.shape == local_full.shape
        assert np.array_equal(local_full, mask)

        port_dir: tuple[int, int] | None = None
        for d in DIRS:
            if set(_edge_offsets(mask, d)) == local_green:
                port_dir = d
                break
        assert port_dir is not None

        bi = int(len(blobs))
        for r, c in blue_pts:
            blob_id[int(r), int(c)] = bi

        blobs.append(
            {
                "full": pts_full,
                "blue": blue_pts,
                "scale": scale,
                "port_dir": port_dir,
                "emit_starts": _sort_on_ray(_offset(_lane_frontier([(int(r), int(c)) for r, c in np.argwhere(mask)], port_dir), r0, c0), port_dir),
            }
        )

    activated: set[int] = set()
    queue: list[int] = []

    def activate_blob_at(r: int, c: int) -> None:
        """Recolor an unactivated blob to red and queue it for propagation."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))
        for pr, pc in blobs[bi]["full"]:
            out[int(pr), int(pc)] = int(red_col)

    # --- STEP 4) Draw the start ray upward and stop at the first blob hit ---
    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, start_col]) == int(yellow_col)
    out[top_r - 1, start_col] = int(red_col)
    for r in range(top_r - 1, -1, -1):
        out[int(r), int(start_col)] = int(red_col)
        if int(g[int(r), int(start_col)]) == int(blue_col):
            activate_blob_at(int(r), int(start_col))
            break

    def extend_shape_beam(source_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        """Emit one outgoing lane per occupied frontier row/column of the blob silhouette."""
        dr, dc = d
        for sr, sc in source_pixels:
            r, c = int(sr), int(sc)
            while True:
                r += int(dr)
                c += int(dc)
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[int(r), int(c)] = int(red_col)
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c))
                    break

    # --- STEP 5) Propagate the BFS cascade using scale-derived beam widths ---
    while queue:
        bi = int(queue.pop(0))
        extend_shape_beam(blobs[bi]["emit_starts"], blobs[bi]["port_dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    return out


def _gen_attempt() -> np.ndarray | None:
    """Single output-first construction attempt for the fused variant."""
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    # Place the start marker.
    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = int(H - start_h - 1)
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    inp[cap_r, start_col] = YELLOW
    out[cap_r, start_col] = RED
    ray_mask[cap_r, start_col] = True
    occupied[cap_r, start_col] = True
    for r in range(cap_r + 1, H):
        inp[int(r), int(start_col)] = RED
        out[int(r), int(start_col)] = RED
        occupied[int(r), int(start_col)] = True

    chain_len = random.choices(CHAIN_LENGTHS, weights=CHAIN_LEN_WEIGHTS)[0]
    min_dist = max(0, MIN_TOTAL_BLOBS - int(chain_len))
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(min_dist, max_dist)

    starts = [(int(cap_r), int(start_col))]
    direction = UP
    include_origin = True
    activated_scales: list[int] = []
    placed_scales: list[int] = []

    for _ in range(chain_len):
        blob = _candidate_next_blob(H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask)
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["full"], RED)
        activated_scales.append(int(blob["scale"]))
        placed_scales.append(int(blob["scale"]))
        starts = blob["emit_starts"]
        direction = blob["spec"]["port_dir"]
        include_origin = False

    final_path = _path_to_boundary(starts, direction, include_origin, H, W)
    if any(bool(blue_mask[r, c]) for r, c in final_path):
        return None
    _mark_ray(out, ray_mask, final_path)

    for _ in range(num_distractors):
        blob = _place_distractor(inp, occupied, blue_mask, ray_mask)
        if blob is None:
            return None
        placed_scales.append(int(blob["scale"]))

    if not placed_scales or not any(scale > 1 for scale in placed_scales):
        return None
    if len(set(placed_scales)) < 2:
        return None
    if not any(scale > 1 for scale in activated_scales):
        return None

    try:
        oracle = main(inp)
    except Exception:
        return None
    if not np.array_equal(oracle, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate one valid input grid for the fused variant."""
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
