# Variant of 5961cc34
# Dimension: D14 — 多步处理顺序 (multi_step)
# Change: BFS 顺序处理级联：先激活直接命中的色块，再处理它们发射的射线命中的色块，逐层扩展... → 交替方向处理

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste

# Design analysis for the D14 variant
# a) Concrete behavior:
#    The input visuals stay the same: a bottom start marker (yellow cap over a red
#    stem) and blue diamond/octagon blobs with one green output side. The change is
#    in the cascade schedule after the start ray hits the first blob. We process
#    blobs in BFS-like rounds, but odd rounds only emit horizontal rays and even
#    rounds only emit vertical rays. Each round only acts on the blobs activated in
#    the previous round. If that frontier has no ray of the required axis, the round
#    is empty and nothing new is activated, so the cascade stops there. Example:
#    root blob -> horizontal child -> vertical child is allowed, but if that last
#    child also points vertical, its outgoing ray never appears because the next
#    round expects horizontal.
# b) Complexity calibration:
#    Colors, blob shapes, and one-port-per-blob encoding are unchanged. The only
#    new structure is a controlled "blocked continuation": the generator sometimes
#    places an extra blob exactly where the original BFS would have continued, but
#    gives the parent the wrong-parity port so the new schedule suppresses that
#    whole tail. This makes the D14 change visible without adding extra symbols.
# c) Hidden assumptions from the original solver:
#    The old queue logic assumed every activated blob will eventually emit, which is
#    no longer true for every activated blob. The final "at most 1 distractor"
#    assert also depended on the old linear generator and is removed. The generator
#    can no longer choose arbitrary non-opposite port directions; it must align each
#    blob's port axis with the round in which that blob is supposed to emit, or
#    intentionally misalign it when creating a blocked tail.
# d) Placement legality:
#    We keep the original padded non-overlap rules for blob bodies, keep the start
#    column clear, and only place blobs whose incoming ray reaches their entry edge
#    before touching any other blue cell. Split children are placed one lane at a
#    time with updated occupancy masks so they cannot overlap or clip each other.
# e) Visual observability:
#    The key witness is the blocked tail: the input contains a blob lined up exactly
#    with a visible green port, but the output omits that ray and that follower
#    because the parent blob was activated one round too early or too late for that
#    orientation.
# f) Visual cleanliness:
#    We keep the layouts mostly linear, keep orthogonal rays axis-aligned, and keep
#    at least one-cell padding between blob bodies. Distractors remain optional and
#    are placed away from all actual red rays.
# g) Color inferability:
#    No new color mapping is introduced. Blue+green still means inert blob+port in
#    the input; red still means activated blob or traced ray in the output.


def _axis_of(d: tuple[int, int]) -> str:
    return "V" if int(d[0]) else "H"


def _round_axis(step_idx: int) -> str:
    return "H" if int(step_idx) % 2 == 1 else "V"


def _dir_choices(axis: str) -> tuple[tuple[int, int], ...]:
    return (LEFT, RIGHT) if axis == "H" else (UP, DOWN)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the red activation pattern from blue blobs whose green sides define rays.

    Input structure:
    - The background is the dominant color; blue connected blobs carry one green edge
      that marks a single outgoing direction and the ray thickness.
    - A bottom start marker is still a short red vertical stem capped by one yellow
      pixel.

    Rule:
    - Draw the start ray upward and activate the first blue blob it hits.
    - Afterwards process activations in rounds: odd rounds emit only horizontal rays
      and even rounds emit only vertical rays.
    - Each round only considers the blobs activated in the immediately previous
      round; if none of them has the required orientation, that round is empty and
      the cascade stops. All activated blobs and all traced rays are colored red.

    Key insight:
    The green side still encodes direction and thickness, but a blob now only gets
    one parity-specific chance to fire, so wrong-parity tails disappear.
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
            if cap in (bg, int(col)):
                continue
            start_candidates.append(
                (int(col), cap, c0, [(rr, c0) for rr in range(r0, r1 + 1)])
            )

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]

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

    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = len(blobs)
        for r, c in pts:
            blob_id[r, c] = bi
        blobs.append({"pts": pts, "ports": []})

    for blob in blobs:
        pts = blob["pts"]
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in pts:
            for ddr, ddc in DIRS:
                rr = r + ddr
                cc = c + ddc
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == green_col:
                    seen_dirs.add((int(ddr), int(ddc)))
        for ddr, ddc in seen_dirs:
            port_pixels = [
                (int(r), int(c))
                for (r, c) in pts
                if 0 <= r + ddr < H
                and 0 <= c + ddc < W
                and int(g[r + ddr, c + ddc]) == green_col
            ]
            assert port_pixels
            blob["ports"].append({"dir": (int(ddr), int(ddc)), "pixels": port_pixels})

    activated: set[int] = set()

    def activate_blob_at(r: int, c: int, collector: list[int]) -> None:
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(bi)
        collector.append(bi)
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = red_col

    for r, c in start_seg:
        out[int(r), int(c)] = red_col
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, start_col]) == yellow_col
    out[top_r - 1, start_col] = red_col

    frontier: list[int] = []
    for r in range(top_r - 1, -1, -1):
        out[int(r), int(start_col)] = red_col
        if int(g[int(r), int(start_col)]) == blue_col:
            activate_blob_at(int(r), int(start_col), frontier)
            break

    def extend_thick(
        port_pixels: list[tuple[int, int]],
        d: tuple[int, int],
        collector: list[int],
    ) -> None:
        dr, dc = map(int, d)
        for pr, pc in port_pixels:
            r, c = int(pr), int(pc)
            while True:
                r += dr
                c += dc
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[int(r), int(c)] = red_col
                if int(g[int(r), int(c)]) == blue_col:
                    activate_blob_at(int(r), int(c), collector)
                    break

    step_idx = 1
    while frontier:
        wanted_axis = _round_axis(step_idx)
        eligible: list[dict] = []
        for bi in frontier:
            for port in blobs[bi]["ports"]:
                if _axis_of(port["dir"]) == wanted_axis:
                    eligible.append(port)

        next_frontier: list[int] = []
        for port in eligible:
            extend_thick(port["pixels"], port["dir"], next_frontier)
        frontier = next_frontier
        step_idx += 1

    return out


# --------------- generator ---------------

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

# --- Composition ---
LINE_MODE_WEIGHT = 3
BRANCH_MODE_WEIGHT = 4
LINE_BLOB_WEIGHTS = [2, 4, 3]  # total activated blobs in line mode: 1..3
GRANDCHILD_WEIGHT = [2, 3]     # branch grandchild count in {0,1}
MAX_DISTRACTORS = 1
BRANCH_START_PAD = 5
BRANCH_SIZES = ((3, 4), (4, 4), (5, 4))

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 500


def _shape_mask(h: int, w: int) -> np.ndarray:
    widths = _SHAPE_WIDTHS[(h, w)]
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(widths):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return rc[1] if d[0] else rc[0]


def _along_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return rc[0] if d[0] else rc[1]


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    return sorted(coords, key=lambda p: _perp_value(p, d))


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _blob_spec(size: tuple[int, int], port_dir: tuple[int, int]) -> dict | None:
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
        br = gr - port_dir[0]
        bc = gc - port_dir[1]
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


def _all_specs(
    port_dirs: tuple[tuple[int, int], ...] | None = None,
    port_width: int | None = None,
    sizes: tuple[tuple[int, int], ...] | None = None,
) -> list[dict]:
    dirs = list(port_dirs) if port_dirs is not None else list(DIRS)
    shape_sizes = list(sizes) if sizes is not None else list(SHAPES)
    out = []
    for size in shape_sizes:
        for d in dirs:
            spec = _blob_spec(size, d)
            if spec is None:
                continue
            if port_width is not None and len(spec["port_pixels"]) != int(port_width):
                continue
            out.append(spec)
    random.shuffle(out)
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
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
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        if include_origin:
            cells.append((r, c))
        guard = H + W + 10
        while (r, c) != (er, ec):
            r += dr
            c += dc
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
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        if include_origin:
            cells.append((r, c))
        while True:
            r += dr
            c += dc
            if not (0 <= r < H and 0 <= c < W):
                break
            cells.append((r, c))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
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


def _candidate_blob_options(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    port_dirs: tuple[tuple[int, int], ...] | None = None,
    port_width: int | None = None,
    sizes: tuple[tuple[int, int], ...] | None = None,
    use_padding: bool = True,
) -> list[dict]:
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied) if use_padding else occupied
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    candidates: list[dict] = []

    for spec in _all_specs(port_dirs=port_dirs, port_width=port_width, sizes=sizes):
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

    random.shuffle(candidates)
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
    port_dirs: tuple[tuple[int, int], ...] | None = None,
    port_width: int | None = None,
    sizes: tuple[tuple[int, int], ...] | None = None,
    use_padding: bool = True,
) -> dict | None:
    candidates = _candidate_blob_options(
        H,
        W,
        starts,
        d,
        include_origin,
        occupied,
        blue_mask,
        ray_mask,
        port_dirs=port_dirs,
        port_width=port_width,
        sizes=sizes,
        use_padding=use_padding,
    )
    return random.choice(candidates) if candidates else None


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
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


def _paint_blob_red(out: np.ndarray, blob: dict) -> None:
    paint_coords(out, blob["blue"], RED)


def _place_start(
    H: int,
    W: int,
    start_pad: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = H - start_h - 1
    start_col = random.randint(start_pad, W - start_pad - 1)
    inp[cap_r, start_col] = YELLOW
    out[cap_r, start_col] = RED
    ray_mask[cap_r, start_col] = True
    occupied[cap_r, start_col] = True
    for r in range(cap_r + 1, H):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        occupied[r, start_col] = True

    return inp, out, occupied, blue_mask, ray_mask, cap_r, start_col


def _build_line_case(
    H: int,
    W: int,
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    cap_r: int,
    start_col: int,
) -> dict | None:
    total_blobs = random.choices([1, 2, 3], weights=LINE_BLOB_WEIGHTS)[0]
    blocked_tail = random.random() < 0.65

    starts = [(cap_r, start_col)]
    direction = UP
    include_origin = True
    blobs_placed = 0

    for depth in range(total_blobs):
        emit_axis = _round_axis(depth + 1)
        if blocked_tail and depth == total_blobs - 1:
            emit_axis = "V" if emit_axis == "H" else "H"
        port_dirs = _dir_choices(emit_axis)
        blob = _candidate_next_blob(
            H,
            W,
            starts,
            direction,
            include_origin,
            occupied,
            blue_mask,
            ray_mask,
            port_dirs=port_dirs,
        )
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        _paint_blob_red(out, blob)
        starts = blob["ports"]
        direction = blob["spec"]["port_dir"]
        include_origin = False
        blobs_placed += 1

    if blocked_tail:
        tail = _candidate_next_blob(
            H,
            W,
            starts,
            direction,
            False,
            occupied,
            blue_mask,
            ray_mask,
        )
        if tail is None:
            return None
        _place_blob(inp, occupied, blue_mask, tail)
    else:
        final_path = _path_to_boundary(starts, direction, include_origin, H, W)
        if any(blue_mask[r, c] for r, c in final_path):
            return None
        _mark_ray(out, ray_mask, final_path)

    return {
        "mode": "line",
        "activated_blobs": blobs_placed,
        "skip_rounds": 0,
        "branch_blocked": int(blocked_tail),
    }


def _place_branch_children(
    H: int,
    W: int,
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    lane_starts: list[tuple[int, int]],
    direction: tuple[int, int],
) -> tuple[dict, dict] | None:
    assert len(lane_starts) == 2
    continue_idx = random.choice([0, 1])
    blocked_idx = 1 - continue_idx
    cont_options = _candidate_blob_options(
        H,
        W,
        [lane_starts[continue_idx]],
        direction,
        False,
        occupied,
        blue_mask,
        ray_mask,
        port_dirs=_dir_choices("V"),
        sizes=BRANCH_SIZES,
        use_padding=False,
    )
    blocked_options = _candidate_blob_options(
        H,
        W,
        [lane_starts[blocked_idx]],
        direction,
        False,
        occupied,
        blue_mask,
        ray_mask,
        port_dirs=(direction,),
        sizes=BRANCH_SIZES,
        use_padding=False,
    )

    for cont_blob in cont_options:
        cont_full = set(cont_blob["full"])
        cont_entry = set(cont_blob["entry"])
        cont_path = {cell for cell in cont_blob["path"] if cell not in cont_entry}
        for blocked_blob in blocked_options:
            blocked_full = set(blocked_blob["full"])
            blocked_entry = set(blocked_blob["entry"])
            blocked_path = {cell for cell in blocked_blob["path"] if cell not in blocked_entry}
            if cont_full & blocked_full:
                continue
            if cont_path & blocked_full:
                continue
            if blocked_path & cont_full:
                continue

            for blob in (cont_blob, blocked_blob):
                _mark_ray(out, ray_mask, blob["path"])
                _place_blob(inp, occupied, blue_mask, blob)
                _paint_blob_red(out, blob)
            return cont_blob, blocked_blob

    return None


def _build_branch_case(
    H: int,
    W: int,
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    cap_r: int,
    start_col: int,
) -> dict | None:
    root_options = _candidate_blob_options(
        H,
        W,
        [(cap_r, start_col)],
        UP,
        True,
        occupied,
        blue_mask,
        ray_mask,
        port_dirs=(LEFT, RIGHT),
        port_width=2,
        sizes=BRANCH_SIZES,
    )
    for root in root_options:
        inp2 = inp.copy()
        out2 = out.copy()
        occupied2 = occupied.copy()
        blue_mask2 = blue_mask.copy()
        ray_mask2 = ray_mask.copy()

        _mark_ray(out2, ray_mask2, root["path"])
        _place_blob(inp2, occupied2, blue_mask2, root)
        _paint_blob_red(out2, root)

        children = _place_branch_children(
            H,
            W,
            inp2,
            out2,
            occupied2,
            blue_mask2,
            ray_mask2,
            root["ports"],
            root["spec"]["port_dir"],
        )
        if children is None:
            continue
        continuing_child, blocked_child = children

        activated_blobs = 3
        starts = continuing_child["ports"]
        direction = continuing_child["spec"]["port_dir"]
        if random.choices([0, 1], weights=GRANDCHILD_WEIGHT)[0]:
            grandchild = _candidate_next_blob(
                H,
                W,
                starts,
                direction,
                False,
                occupied2,
                blue_mask2,
                ray_mask2,
                port_dirs=_dir_choices("H"),
                sizes=BRANCH_SIZES,
            )
            if grandchild is not None:
                _mark_ray(out2, ray_mask2, grandchild["path"])
                _place_blob(inp2, occupied2, blue_mask2, grandchild)
                _paint_blob_red(out2, grandchild)
                starts = grandchild["ports"]
                direction = grandchild["spec"]["port_dir"]
                activated_blobs += 1

        final_path = _path_to_boundary(starts, direction, False, H, W)
        if any(blue_mask2[r, c] for r, c in final_path):
            continue
        _mark_ray(out2, ray_mask2, final_path)

        assert _axis_of(blocked_child["spec"]["port_dir"]) == "H"
        inp[:, :] = inp2
        out[:, :] = out2
        occupied[:, :] = occupied2
        blue_mask[:, :] = blue_mask2
        ray_mask[:, :] = ray_mask2
        return {
            "mode": "branch",
            "activated_blobs": activated_blobs,
            "skip_rounds": 0,
            "branch_blocked": 1,
        }

    return None


def _gen_attempt(with_meta: bool = False) -> np.ndarray | tuple[np.ndarray, dict] | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp, out, occupied, blue_mask, ray_mask, cap_r, start_col = _place_start(H, W, START_COL_PAD)
    meta = _build_line_case(H, W, inp, out, occupied, blue_mask, ray_mask, cap_r, start_col)
    if meta is None:
        return None

    for _ in range(random.randint(0, MAX_DISTRACTORS)):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    if not np.array_equal(main(inp), out):
        return None
    return (inp, meta) if with_meta else inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _gen_attempt(with_meta=False)
        if built is not None:
            return built
    raise RuntimeError("failed to generate a valid 5961cc34 D14 input")
