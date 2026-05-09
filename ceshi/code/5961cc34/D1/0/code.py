# Variant of 5961cc34
# Dimension: D1 — 射线传播方向 (path)
# Change: 起始射线固定向上（垂直），后续射线沿色块端口方向（上/下/左/右四个基本方向）直线传播... → 对角线射线传播

from __future__ import annotations

from collections import Counter
import random
import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# Design analysis
# - concrete_behavior:
#   - Start marker stays a red vertical stem touching the bottom, but the yellow cap is now
#     one cell above-left or above-right of the stem top. That visible offset is the only
#     cue for the first ray: cap at upper-left means the ray continues up-left; cap at
#     upper-right means up-right.
#   - Blobs keep the original blue diamond/octagon silhouettes. Green ports still replace
#     boundary cells of the silhouette, but port directions now use all 8 compass directions.
#     Cardinal ports use an edge bar as before; diagonal ports use the first 1-2 cells on the
#     matching diagonal frontier so the port visibly sits on one corner/rim.
#   - Output still erases everything except the red start marker, activated blobs, and traced
#     rays. The only conceptual change is path direction: every ray segment advances by a fixed
#     step (dr, dc) with dr, dc in {-1, 0, 1}, and diagonal steps are now allowed.
# - behavior_trace:
#   - Original first step: from the yellow cap at (r, c), paint (r-1, c), (r-2, c), ...
#     until a blue pixel is hit.
#   - Variant first step: if the cap is at (r, c-1) relative to the stem top, the path is
#     (r, c-1), (r-1, c-2), (r-2, c-3), ... . The rest of the cascade uses the same rule:
#     each activated blob emits from its blue port pixels through the green port cells, but
#     the propagation direction may now be diagonal as well as axial.
#   - The important implementation difference is that "same ray lane" is no longer described
#     by row-only or col-only alignment; it is described by repeated additions of the same
#     direction vector.
# - complexity_calibration:
#   - The start ray is always diagonal, so the variant is visible in every sample.
#   - Blob ports can use all 8 directions, but width stays in the small original range (1-2)
#     so the task does not explode into many unrelated subcases.
#   - We keep the original chain length / distractor budget to avoid clutter.
# - implicit_assumptions_review:
#   - The original solver inferred ports by checking 4-neighbor green adjacency. That is
#     unsafe here: a diagonal-corner green cell can touch blue pixels from multiple offsets.
#     The new solver therefore matches each blob against the known silhouette templates and
#     matches the green offsets against the unique generated port template.
#   - The original generator only handled axis-aligned entry windows. The variant generator
#     instead validates candidate placements by checking whether each lane reaches a candidate
#     entry cell via repeated additions of one shared direction vector.
#   - The generator assumes blobs are separated by at least one background pixel in 8-neighbor
#     space, so adjacent green cells belong to exactly one blue component. This is enforced via
#     dilated occupancy masks before every placement.
# - placement_legality:
#   - A candidate blob is legal only if its full silhouette stays in bounds, does not touch an
#     existing blob/distractor/start marker after 8-neighbor dilation, and its incoming ray
#     path does not cross prior rays or foreign occupied cells before the intended hit cells.
#   - For emitted rays we allow exactly one occupied step beyond the blue port pixels: the
#     emitter's own green port cells. That keeps the geometry consistent with the input.
# - visual_observability:
#   - The diagonal yellow cap is visible and repeated across all examples, so the first ray's
#     diagonal rule can be learned without hidden metadata.
#   - Diagonal blob ports occupy a distinctive corner/rim, making the 8-direction extension
#     visible even when only one blob is activated.
# - visual_cleanliness:
#   - Rays are prevented from crossing previous rays or passing through unrelated objects.
#   - Distractors are placed fully off every ray path, preserving the original tidy look.


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
UP_LEFT = (-1, -1)
UP_RIGHT = (-1, 1)
DOWN_LEFT = (1, -1)
DOWN_RIGHT = (1, 1)
DIRS = (UP, DOWN, LEFT, RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT)
OPP = {d: (-d[0], -d[1]) for d in DIRS}
DIAGONALS = {UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT}

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
START_COL_PAD = 3

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3
START_DIRS = (UP_LEFT, UP_RIGHT)

# --- Chain / distractor ---
CHAIN_LEN_WEIGHTS = [1, 3, 3, 3]
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


def _lane_key(rc: tuple[int, int], d: tuple[int, int]) -> int:
    """Invariant shared by all points on the same ray lane in direction d."""
    r, c = rc
    dr, dc = d
    return int(r * dc - c * dr)


def _along_key(rc: tuple[int, int], d: tuple[int, int]) -> int:
    """Monotone key along the travel direction."""
    r, c = rc
    dr, dc = d
    return int(r * dr + c * dc)


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    """Sort parallel lanes consistently for a given direction."""
    return sorted(coords, key=lambda p: (_lane_key(p, d), _along_key(p, d), p[0], p[1]))


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    """Return mask-local coordinates on the cardinal edge facing direction d."""
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _diag_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    """Return 1-2 mask-local cells on the diagonal frontier facing direction d."""
    pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    frontier_score = max(int(r * d[0] + c * d[1]) for r, c in pts)
    frontier = [(r, c) for r, c in pts if int(r * d[0] + c * d[1]) == frontier_score]
    frontier.sort(key=lambda p: (p[0], p[1]), reverse=(d[0] > 0))
    return frontier[: min(2, len(frontier))]


def _port_green_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    """Return the generated green port cells for a given direction."""
    return _diag_offsets(mask, d) if d in DIAGONALS else _edge_offsets(mask, d)


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    """Translate local offsets to global coordinates."""
    return [(int(r0 + r), int(c0 + c)) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= int(r) < H and 0 <= int(c) < W for r, c in coords)


def _steps_along(
    start: tuple[int, int], end: tuple[int, int], d: tuple[int, int]
) -> int | None:
    """Return k if end == start + k*d for some k>=0, else None."""
    sr, sc = start
    er, ec = end
    dr, dc = d
    rr = int(er - sr)
    cc = int(ec - sc)

    if dr == 0:
        if rr != 0 or dc == 0 or cc % dc != 0:
            return None
        k = cc // dc
        return int(k) if k >= 0 else None
    if dc == 0:
        if cc != 0 or rr % dr != 0:
            return None
        k = rr // dr
        return int(k) if k >= 0 else None
    if rr % dr != 0 or cc % dc != 0:
        return None
    kr = rr // dr
    kc = cc // dc
    if kr != kc or kr < 0:
        return None
    return int(kr)


def _path_to_entries(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    entries: list[tuple[int, int]],
    include_origin: bool,
    H: int,
    W: int,
) -> tuple[list[tuple[int, int]], list[int]] | None:
    """Trace ray lanes from starts to entry cells; return cells and per-lane lengths."""
    cells: list[tuple[int, int]] = []
    lengths: list[int] = []
    dr, dc = d
    for (r, c), (er, ec) in zip(starts, entries):
        k = _steps_along((r, c), (er, ec), d)
        if k is None:
            return None
        lengths.append(int(k))
        if include_origin:
            cells.append((int(r), int(c)))
        for _ in range(int(k)):
            r += dr
            c += dc
            if not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((int(r), int(c)))
    return cells, lengths


def _path_to_boundary(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """Trace ray lanes from starts to the grid boundary."""
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        if include_origin:
            cells.append((int(r), int(c)))
        while True:
            r += dr
            c += dc
            if not (0 <= r < H and 0 <= c < W):
                break
            cells.append((int(r), int(c)))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    """Paint cells red in the output grid and record them in the ray mask."""
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


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


def _blob_spec(size: tuple[int, int], port_dir: tuple[int, int]) -> dict | None:
    """Build a blob specification dict for a given shape and port direction."""
    h, w = size
    mask = _shape_mask(h, w)
    green = _port_green_offsets(mask, port_dir)
    if len(green) not in (1, 2):
        return None

    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    full_set = set(full)
    green_set = set(green)

    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    for r, c in green:
        patch[int(r), int(c)] = GREEN

    port_pixels: list[tuple[int, int]] = []
    for gr, gc in green:
        br, bc = int(gr - port_dir[0]), int(gc - port_dir[1])
        if (br, bc) not in full_set or (br, bc) in green_set:
            return None
        port_pixels.append((int(br), int(bc)))

    blue = [p for p in full if p not in green_set]
    entry_by_move: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for d in DIRS:
        dr, dc = d
        entries = [
            (int(r), int(c))
            for r, c in blue
            if (int(r - dr), int(c - dc)) not in full_set
        ]
        entry_by_move[d] = _sort_on_ray(entries, d)

    return {
        "size": (h, w),
        "mask": mask,
        "patch": patch,
        "full": full,
        "full_set": full_set,
        "blue": blue,
        "green": list(green),
        "green_set": green_set,
        "port_dir": port_dir,
        "port_pixels": _sort_on_ray(port_pixels, port_dir),
        "entry_by_move": entry_by_move,
    }


ALL_SPECS = [s for size in SHAPES for d in DIRS if (s := _blob_spec(size, d)) is not None]
SPEC_LOOKUP: dict[tuple[tuple[int, int], frozenset[tuple[int, int]]], dict] = {}
for _spec in ALL_SPECS:
    key = (_spec["size"], frozenset(_spec["green"]))
    assert key not in SPEC_LOOKUP
    SPEC_LOOKUP[key] = _spec


def _all_specs() -> list[dict]:
    """Return a shuffled copy of all valid blob specs."""
    specs = list(ALL_SPECS)
    random.shuffle(specs)
    return specs


def _find_start(
    g: np.ndarray, bg: int
) -> tuple[int, int, tuple[int, int], list[tuple[int, int]], tuple[int, int]]:
    """Detect the unique red stem + diagonal cap start marker."""
    H, W = g.shape
    start_candidates: list[
        tuple[int, int, tuple[int, int], list[tuple[int, int]], tuple[int, int]]
    ] = []

    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=8):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

            if c0 != c1 or (r1 - r0 + 1) < 2 or r1 != int(H - 1):
                continue
            if any(int(g[rr, c0]) != int(col) for rr in range(r0, r1 + 1)):
                continue
            top_r = int(r0)
            cap_options: list[tuple[int, int, tuple[int, int]]] = []
            for d in START_DIRS:
                rr = int(top_r + d[0])
                cc = int(c0 + d[1])
                if 0 <= rr < H and 0 <= cc < W:
                    cap = int(g[rr, cc])
                    if cap != int(bg) and cap != int(col):
                        cap_options.append((cap, cc, d))
            if len(cap_options) != 1:
                continue
            cap_col, cap_c, start_dir = cap_options[0]
            cap_pos = (int(top_r + start_dir[0]), int(cap_c))
            start_candidates.append(
                (
                    int(col),
                    int(cap_col),
                    (int(start_dir[0]), int(start_dir[1])),
                    [(int(rr), int(c0)) for rr in range(r0, r1 + 1)],
                    cap_pos,
                )
            )

    assert len(start_candidates) == 1
    return start_candidates[0]


def _match_blob_port(
    g: np.ndarray, blue_pts: list[tuple[int, int]], green_col: int
) -> tuple[dict, tuple[int, int], list[tuple[int, int]]]:
    """Recover a blob's spec and its global blue port pixels from blue+green geometry."""
    H, W = g.shape
    blue_set = set(blue_pts)
    green_pts: set[tuple[int, int]] = set()
    for r, c in blue_pts:
        for dr, dc in DIRS:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(green_col):
                green_pts.add((int(rr), int(cc)))

    assert green_pts
    full_pts = blue_set | green_pts
    rs = [r for r, _ in full_pts]
    cs = [c for _, c in full_pts]
    r0, c0 = min(rs), min(cs)
    r1, c1 = max(rs), max(cs)
    size = (int(r1 - r0 + 1), int(c1 - c0 + 1))
    key = (size, frozenset((int(r - r0), int(c - c0)) for r, c in green_pts))
    spec = SPEC_LOOKUP[key]

    local_full = {(int(r - r0), int(c - c0)) for r, c in full_pts}
    assert local_full == spec["full_set"]

    ports = _offset(spec["port_pixels"], r0, c0)
    return spec, (int(r0), int(c0)), ports


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red ray network by activating blue blobs through 8-direction ports.

    Input structure:
    - Background is the dominant color. Blue diamond/octagon blobs carry embedded green
      port cells on one side or corner, and the bottom start marker is a red vertical stem
      topped by a single yellow cap placed diagonally left or right.

    Rule:
    - The yellow cap sets the first ray direction: continue from the cap along that same
      diagonal until the ray hits a blue blob or exits the grid.
    - A hit blob turns red and emits thick rays through its green port in the port's
      encoded 8-direction heading (cardinal or diagonal).
    - Propagate breadth-first until no new blue blob is reached; erase every untouched
      object back to background.

    Key insight: The cap offset and each blob's green template encode fixed step vectors,
    so solving is a matter of following repeated (dr, dc) ray lanes through the cascade.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    red_col, yellow_col, start_dir, start_seg, cap_pos = _find_start(g, bg)

    # --- STEP 2) Identify blob colors ---
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

    # Green = the unique other color that is 8-adjacent to blue.
    green_candidates: list[int] = []
    for col in colors_present(g, exclude={bg, red_col, yellow_col, blue_col}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        ok = False
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in DIRS:
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

    # --- STEP 3) Extract blue blobs and recover their port geometry from templates ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []

    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[int(r), int(c)] = bi
        spec, origin, port_pixels = _match_blob_port(g, pts, green_col)
        blobs.append(
            {
                "pts": pts,
                "origin": origin,
                "ports": [{"dir": spec["port_dir"], "pixels": port_pixels}],
            }
        )

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

    # --- STEP 4) Draw the diagonal start ray and stop at the first blob hit ---
    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    out[int(cap_pos[0]), int(cap_pos[1])] = int(red_col)
    r, c = cap_pos
    while 0 <= r < H and 0 <= c < W:
        out[int(r), int(c)] = int(red_col)
        if int(g[int(r), int(c)]) == int(blue_col):
            activate_blob_at(int(r), int(c))
            break
        r += int(start_dir[0])
        c += int(start_dir[1])

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

    # --- STEP 5) Propagate (BFS): activated blob -> emit rays -> activate more blobs ---
    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"

    return out


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
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    emit_allow = {
        (int(r + d[0]), int(c + d[1]))
        for r, c in starts
        if 0 <= int(r + d[0]) < H and 0 <= int(c + d[1]) < W
    }
    origin_allow = set(starts) if include_origin else set()

    for spec in _all_specs():
        if spec["port_dir"] == entry_side:
            continue
        entry = spec["entry_by_move"][d]
        if len(entry) < thickness:
            continue
        windows = [entry[k : k + thickness] for k in range(len(entry) - thickness + 1)]
        random.shuffle(windows)
        placements = [(r0, c0) for r0 in range(H - spec["size"][0] + 1) for c0 in range(W - spec["size"][1] + 1)]
        random.shuffle(placements)

        for r0, c0 in placements:
            full = _offset(spec["full"], r0, c0)
            if not _in_bounds(full, H, W):
                continue
            if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
                continue

            for window in windows:
                entry_cells = _offset(window, r0, c0)
                traced = _path_to_entries(starts, d, entry_cells, include_origin, H, W)
                if traced is None:
                    continue
                path, lengths = traced
                if any(int(k) < min_gap for k in lengths):
                    continue

                entry_set = set(entry_cells)
                blocked = False
                for cell in path:
                    if cell in entry_set or cell in origin_allow or cell in emit_allow:
                        continue
                    r, c = cell
                    if ray_mask[r, c] or occupied[r, c]:
                        blocked = True
                        break
                    if blue_mask[r, c]:
                        blocked = True
                        break
                if blocked:
                    continue

                return {
                    "spec": spec,
                    "r0": int(r0),
                    "c0": int(c0),
                    "entry": entry_cells,
                    "path": path,
                    "blue": _offset(spec["blue"], r0, c0),
                    "full": full,
                    "ports": _sort_on_ray(_offset(spec["port_pixels"], r0, c0), spec["port_dir"]),
                }

    return None


def _place_blob(
    inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict
) -> None:
    """Paste a blob's patch onto the input grid and update tracking masks."""
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[int(r), int(c)] = True
    for r, c in blob["blue"]:
        blue_mask[int(r), int(c)] = True


def _place_distractor(
    inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, ray_mask: np.ndarray
) -> bool:
    """Try to place a single distractor blob in free space."""
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []
    for spec in _all_specs():
        h, w = spec["size"]
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
    """Single attempt to generate a valid input grid."""
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    # Place start segment (diagonal yellow cap + red vertical stem).
    start_h = random.randint(START_H_MIN, START_H_MAX)
    top_r = H - start_h
    start_dir = random.choice(START_DIRS)
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    cap_r = int(top_r + start_dir[0])
    cap_c = int(start_col + start_dir[1])
    if not (0 <= cap_r < H and 0 <= cap_c < W):
        return None
    inp[cap_r, cap_c] = YELLOW
    out[cap_r, cap_c] = RED
    occupied[cap_r, cap_c] = True
    ray_mask[cap_r, cap_c] = True
    for r in range(top_r, H):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        occupied[r, start_col] = True

    chain_len = random.choices(range(len(CHAIN_LEN_WEIGHTS)), weights=CHAIN_LEN_WEIGHTS)[0]
    min_dist = 1 if chain_len == 0 else 0
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(min_dist, max_dist)

    starts = [(int(cap_r), int(cap_c))]
    direction = start_dir
    include_origin = True
    for _ in range(chain_len):
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
    final_allow = {
        (int(r + direction[0]), int(c + direction[1]))
        for r, c in starts
        if 0 <= int(r + direction[0]) < H and 0 <= int(c + direction[1]) < W
    }
    for r, c in final_path:
        if (r, c) in final_allow:
            continue
        if blue_mask[r, c] or ray_mask[r, c]:
            return None
        if occupied[r, c]:
            return None
    _mark_ray(out, ray_mask, final_path)

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
    raise RuntimeError("failed to generate a valid 5961cc34 D1 input")
