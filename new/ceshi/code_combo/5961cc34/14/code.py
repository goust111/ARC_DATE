# Combo variant of 5961cc34 (index 14)
# combo_spec: D2/2+D19/2
# Name: 距离衰减+色块分解

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior / trace
#    - The visible vocabulary stays close to the base task: a yellow cap over a bottom
#      red segment, plus blue diamond/octagon blobs with one green port edge each.
#    - Fusion change 1: the red segment height is now the global ray budget. The start
#      ray and every emitted port ray can move at most that many cells from their launch
#      cells, then vanish.
#    - Fusion change 2: a hit blob no longer becomes solid red. Instead, its blue body
#      is split orthogonally to the green port direction; the port-facing half turns red
#      and the opposite half stays blue.
#    - Example trace: if the start segment has height 6, the upward start ray can spend
#      only 6 steps. It hits blob A, whose port is on the right, so A becomes red on the
#      right half and blue on the left half. A then emits a rightward thick ray, also
#      capped at 6 steps. Blob B on that lane activates only if its entry edge is within
#      those 6 steps; a farther blob C on the same lane remains completely absent from
#      the output.
# b) Complexity calibration
#    - The budget is a small visible count, so generator uses start heights 5..7 rather
#      than the original 2..3. This makes the distance rule readable without producing
#      very long chains.
#    - Every generated sample is forced to contain at least one reachable blob (to show
#      the split transform) and one farther aligned blob beyond the budget (to show the
#      distance cutoff). That prevents degeneration into either single-variant behavior.
#    - Reachable chain length stays in {1, 2}; this repeats the rule without crowding
#      the canvas or making placement brittle.
# c) Implicit-assumption audit for main()
#    - Original start-ray code assumed it should run until top edge or first hit. That
#      is invalid now; the start ray must stop after `len(start_seg)` steps.
#    - Original thick-ray code assumed rays run to boundary/hit. That is also invalid;
#      each emitted ray shares the same fixed budget.
#    - Original activation code could paint an entire blob red once and never revisit
#      it. That is invalid after decomposition because ray paint inside the blob may need
#      to be corrected back to blue-half cells. Every hit repaints the blob's split state.
#    - Original final assert about at most one unactivated blob is invalid because an
#      intentionally aligned beyond-range blob is now normal.
# d) Placement legality
#    - Blob masks are still chosen from the original symmetric catalog and placed only
#      when fully in bounds, non-overlapping, and padded away from existing blobs.
#    - Reachable blobs are placed only when their entry edge lies within the budget and
#      the full approach corridor is clear.
#    - The overflow blob is placed on the active lane with entry distance strictly larger
#      than the budget, so it is geometrically targetable but unreachable under the new rule.
# e) Visual observability
#    - Solvers can count the bottom red segment and compare it with the truncated red
#      paths in the output.
#    - Activated blobs visibly contain both red and blue, and the red half always sits
#      on the same side as the green port. Across 2-4 examples this exposes both fused
#      controls: range decides whether a blob appears at all, and port side decides how a
#      reached blob is partitioned.
# f) Visual cleanliness
#    - Off-ray distractors are removed. The single aligned overflow blob is enough to
#      demonstrate distance decay and keeps each scene readable.
#    - We preserve the original one-port-per-blob grammar, so no extra symbols or hidden
#      encodings are introduced.
# g) Color-change inference
#    - Only two output colors matter on activated blobs: red on the port-facing half and
#      blue on the far half. Unreached blobs disappear to background. This keeps the
#      color logic structural rather than arbitrary.
# h) New solver assumptions guaranteed by the generator
#    - Exactly one bottom start marker exists.
#    - Every blob has exactly one port side and therefore one split orientation.
#    - Each sample has >=1 reachable activation and >=1 beyond-budget aligned blob.
#    - The final limited ray does not accidentally intersect any unreached blob.
# i) Review of new assumptions after modification
#    - `main()` now assumes the start-segment height is large enough to admit at least
#      one reachable placement and one beyond-budget placement; generator enforces this
#      with the 5..7 budget range and larger minimum canvas.
#    - `main()` assumes repeated hits on the same blob are harmless because repainting
#      split state is idempotent; generator preserves clear corridors so repeated hits
#      only arise from multi-lane thickness, not from ambiguous topology.


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
H_MIN, H_MAX = 20, 27
W_MIN, W_MAX = 15, 27
START_COL_PAD = 2

# --- Start segment / distance budget ---
START_H_MIN, START_H_MAX = 5, 7

# --- Chain ---
MAX_REACHABLE_CHAIN = 2
MIN_GAP_FIRST = 2
MIN_GAP_CHAIN = 2
OVERFLOW_EXTRA_MAX = 6

# --- Retry ---
GEN_RETRIES = 800


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    """Sort coordinates by their perpendicular value relative to direction d."""
    return sorted(coords, key=lambda p: p[1] if d[0] else p[0])


def _split_coords(
    coords: list[tuple[int, int]], port_dir: tuple[int, int]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split body coords into a port-facing red half and an opposite blue half."""
    assert coords
    dr, dc = map(int, port_dir)
    if dr != 0:
        levels = sorted({int(r) for r, _c in coords})
        take = (len(levels) + 1) // 2
        red_levels = set(levels[:take] if dr < 0 else levels[-take:])
        red_half = [(int(r), int(c)) for r, c in coords if int(r) in red_levels]
    else:
        levels = sorted({int(c) for _r, c in coords})
        take = (len(levels) + 1) // 2
        red_levels = set(levels[:take] if dc < 0 else levels[-take:])
        red_half = [(int(r), int(c)) for r, c in coords if int(c) in red_levels]
    red_set = set(red_half)
    blue_half = [p for p in coords if p not in red_set]
    assert red_half and blue_half
    return red_half, blue_half


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a budget-limited red ray cascade that only reddens the port-facing half of each struck blob.

    Input structure:
    - The dominant background surrounds blue diamond/octagon blobs, each carrying one green port bar on a single edge.
    - A yellow cap sits above a bottom red segment; that segment's height encodes the maximum travel distance for every ray hop.

    Rule:
    - Repaint the start marker red and launch the upward start ray, but stop after the encoded distance budget.
    - Any blob touched within budget is activated: split its blue body across the axis perpendicular to the port direction, color the port-facing half red, keep the far half blue, then emit a thick ray through the green port.
    - Every emitted ray uses the same fixed budget, so farther blobs on the same lane may never be reached.

    Key insight: The bottom red height globally limits propagation, while each blob's green side simultaneously determines both the outgoing ray direction and which half of the hit blob survives in blue.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

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
                (int(col), int(cap), int(c0), [(int(rr), int(c0)) for rr in range(r0, r1 + 1)])
            )

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]
    ray_limit = int(len(start_seg))

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

    out = np.full_like(g, bg)

    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == blue_col, connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[r, c] = bi
        blobs.append({"pts": pts, "ports": []})

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
                if 0 <= int(r + ddr) < H
                and 0 <= int(c + ddc) < W
                and int(g[int(r + ddr), int(c + ddc)]) == int(green_col)
            ]
            assert port_pixels
            blob["ports"].append({"dir": (int(ddr), int(ddc)), "pixels": port_pixels})
        assert len(blob["ports"]) == 1
        split_red, split_blue = _split_coords(pts, blob["ports"][0]["dir"])
        blob["split_red"] = split_red
        blob["split_blue"] = split_blue

    activated: set[int] = set()
    queue: list[int] = []

    def paint_blob_state(bi: int) -> None:
        for pr, pc in blobs[bi]["split_blue"]:
            out[int(pr), int(pc)] = int(blue_col)
        for pr, pc in blobs[bi]["split_red"]:
            out[int(pr), int(pc)] = int(red_col)

    def activate_blob_at(r: int, c: int) -> None:
        """Repaint a hit blob into its split state and enqueue it once."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        paint_blob_state(int(bi))
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))

    def extend_limited(origins: list[tuple[int, int]], d: tuple[int, int], include_origin: bool) -> None:
        """Extend each lane for at most `ray_limit` steps, stopping early on blob hit."""
        ddr, ddc = map(int, d)
        for pr, pc in origins:
            r, c = int(pr), int(pc)
            if include_origin:
                out[r, c] = int(red_col)
            for _ in range(int(ray_limit)):
                r += int(ddr)
                c += int(ddc)
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[int(r), int(c)] = int(red_col)
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c))
                    break

    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, int(start_col)]) == int(yellow_col)
    extend_limited([(int(top_r - 1), int(start_col))], UP, include_origin=True)

    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            extend_limited(port["pixels"], port["dir"], include_origin=False)

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


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    """Return sorted mask-local coordinates on the edge facing direction d."""
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _blob_spec(size: tuple[int, int], port_dir: tuple[int, int]) -> dict | None:
    """Build a blob specification dict for a given shape and port direction."""
    h, w = size
    mask = _shape_mask(h, w)
    green = _edge_offsets(mask, port_dir)
    if len(green) not in (1, 2):
        return None

    green_set = set(green)
    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    for r, c in green:
        patch[int(r), int(c)] = GREEN

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
    split_red, split_blue = _split_coords(blue, port_dir)
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "size": (h, w),
        "mask": mask,
        "patch": patch,
        "blue": blue,
        "full": full,
        "port_dir": port_dir,
        "port_pixels": _sort_on_ray(port_pixels, port_dir),
        "split_red": split_red,
        "split_blue": split_blue,
    }


def _all_specs(port_dir: tuple[int, int] | None = None) -> list[dict]:
    """Generate all valid blob specs, optionally filtered by port direction."""
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
    """Trace ray lanes from starts to intended entry cells."""
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        r = int(r)
        c = int(c)
        er = int(er)
        ec = int(ec)
        if include_origin:
            cells.append((r, c))
        guard = H + W + 10
        while (r, c) != (er, ec):
            r += int(dr)
            c += int(dc)
            guard -= 1
            if guard < 0 or not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((int(r), int(c)))
    return cells


def _path_with_limit(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    limit: int,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """Trace a limited-length ray from the given starts."""
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        r = int(r)
        c = int(c)
        if include_origin:
            cells.append((r, c))
        for _ in range(int(limit)):
            r += int(dr)
            c += int(dc)
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
    step_lo: int,
    step_hi: int,
    occ_pad: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    H: int,
    W: int,
) -> list[dict]:
    """Find valid placements for one blob spec along the active ray axis."""
    h, w = spec["mask"].shape
    win_vals = [_perp_value(p, d) for p in window]
    edge_line = _along_value(window[0], d)
    start_line = _along_value(starts[0], d)
    is_vertical = bool(d[0])

    fixed = int(lane_vals[0] - win_vals[0])
    extent = H - h if is_vertical else W - w
    low = int(edge_line)
    high = int(extent + edge_line)
    if d in (UP, LEFT):
        high = min(high, int(start_line - min_gap))
    else:
        low = max(low, int(start_line + min_gap))

    entry_lines = list(range(low, high + 1))
    random.shuffle(entry_lines)
    candidates: list[dict] = []

    for entry_line in entry_lines:
        steps = abs(int(entry_line - start_line))
        if steps < int(step_lo) or steps > int(step_hi):
            continue

        origin_along = int(entry_line - edge_line)
        r0 = origin_along if is_vertical else fixed
        c0 = fixed if is_vertical else origin_along
        if is_vertical:
            entry_cells = [(int(entry_line), int(c0 + c)) for _, c in window]
        else:
            entry_cells = [(int(r0 + r), int(entry_line)) for r, _ in window]
        full = _offset(spec["full"], r0, c0)
        if not _in_bounds(full, H, W):
            continue
        if any(bool(occ_pad[r, c]) or bool(ray_mask[r, c]) for r, c in full):
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
                "steps": int(steps),
                "blue": _offset(spec["blue"], r0, c0),
                "full": full,
                "ports": _sort_on_ray(_offset(spec["port_pixels"], r0, c0), spec["port_dir"]),
                "split_red": _offset(spec["split_red"], r0, c0),
                "split_blue": _offset(spec["split_blue"], r0, c0),
            }
        )

    return candidates


def _candidate_blob(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    step_lo: int,
    step_hi: int,
    allow_terminal: bool,
) -> dict | None:
    """Find a random valid placement for the next on-lane blob."""
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    candidates_by_dir: dict[tuple[int, int], list[dict]] = {}

    for spec in _all_specs():
        if not allow_terminal and spec["port_dir"] == entry_side:
            continue
        edge = _edge_offsets(spec["mask"], entry_side)
        if len(edge) < thickness:
            continue
        for k in range(len(edge) - thickness + 1):
            window = edge[k : k + thickness]
            win_delta = [_perp_value(p, d) - _perp_value(window[0], d) for p in window]
            if win_delta != lane_delta:
                continue
            new_candidates = _try_place_along_axis(
                spec,
                window,
                d,
                lane_vals,
                starts,
                include_origin,
                min_gap,
                step_lo,
                step_hi,
                occ_pad,
                blue_mask,
                ray_mask,
                H,
                W,
            )
            if new_candidates:
                port_dir = tuple(spec["port_dir"])
                candidates_by_dir.setdefault(port_dir, []).extend(new_candidates)

    if not candidates_by_dir:
        return None
    chosen_dir = random.choice(list(candidates_by_dir))
    return random.choice(candidates_by_dir[chosen_dir])


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    """Paste a blob patch into the input and update occupancy masks."""
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[int(r), int(c)] = True
    for r, c in blob["blue"]:
        blue_mask[int(r), int(c)] = True


def _reachable_chain_len(ray_limit: int) -> int:
    """Every sample must show decomposition, so keep at least one reachable activation."""
    if ray_limit <= 5:
        return 1
    return random.choices([1, 2], weights=[2, 3])[0]


def _gen_attempt() -> np.ndarray | None:
    """Single output-first generation attempt for the fused variant."""
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

    ray_limit = int(start_h)
    reachable_len = _reachable_chain_len(ray_limit)
    if reachable_len > MAX_REACHABLE_CHAIN:
        return None

    starts = [(int(cap_r), int(start_col))]
    direction = UP
    include_origin = True

    for _ in range(reachable_len):
        blob = _candidate_blob(
            H,
            W,
            starts,
            direction,
            include_origin,
            occupied,
            blue_mask,
            ray_mask,
            step_lo=1,
            step_hi=ray_limit,
            allow_terminal=False,
        )
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["split_blue"], BLUE)
        paint_coords(out, blob["split_red"], RED)
        starts = blob["ports"]
        direction = blob["spec"]["port_dir"]
        include_origin = False

    overflow_hi = min(int(ray_limit + OVERFLOW_EXTRA_MAX), H + W)
    overflow_blob = _candidate_blob(
        H,
        W,
        starts,
        direction,
        include_origin,
        occupied,
        blue_mask,
        ray_mask,
        step_lo=ray_limit + 1,
        step_hi=overflow_hi,
        allow_terminal=True,
    )
    if overflow_blob is None:
        return None
    _place_blob(inp, occupied, blue_mask, overflow_blob)

    final_path = _path_with_limit(starts, direction, include_origin, ray_limit, H, W)
    if any(bool(blue_mask[r, c]) for r, c in final_path):
        return None
    _mark_ray(out, ray_mask, final_path)

    if not np.any(out == BLUE):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo D2/2+D19/2 input")
