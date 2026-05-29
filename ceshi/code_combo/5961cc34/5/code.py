# Combo variant of 5961cc34 (index 5)
# combo_spec: D4/0+D12/0+D16/0
# Name: 边缘控制像素+任意边缘起始+端口正交约束

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Original task:
#      * Input start cue was a bottom red stem with a yellow cap.
#      * The first ray always launched upward from that cap.
#      * Each hit blob could emit from any non-entry side, so the chain might go
#        straight or turn.
#    - Combo task:
#      * The start cue is now exactly one red control pixel on one border cell.
#      * Its border side determines the inward direction:
#          top -> DOWN, bottom -> UP, left -> RIGHT, right -> LEFT.
#      * Once a ray hits a blue blob, that blob may only emit through a port
#        whose direction is orthogonal to the incoming ray direction.
#      * Therefore the active chain must alternate vertical/horizontal at every
#        activation, producing a strict zigzag.
#    - Small trace:
#      * Suppose the control is at (6, 0), so the first ray goes RIGHT.
#      * It enters blob A from the left; blob A's green side must be UP or DOWN.
#      * If blob A emits UP, blob B is then reached by a vertical ray, so blob B's
#        green side must be LEFT or RIGHT, and so on.
#      * The old solver would fail immediately because it looks for a bottom stem;
#        the old generator would also allow A to emit RIGHT/LEFT, which is now illegal.
# b) Complexity calibration
#    - The start variation is limited to four visible border cases, which is easy
#      to infer from 2-4 examples.
#    - The orthogonal-port rule uses only two categories, vertical vs horizontal,
#      rather than many angle classes, so it stays legible.
#    - We keep the original 1-2 pixel port width and 0-3 chain length so the task
#      still has repeated evidence without becoming crowded.
# c) Implicit-assumption audit for main()
#    - Border scan now assumes the only non-background border cell is the red
#      control pixel. The generator enforces this by keeping all blobs off the border.
#    - Color inference now assumes red is the control color and blue is still the
#      dominant non-background object color. The generator keeps blob area much
#      larger than the single control pixel.
#    - Port extraction assumes each blob has exactly one port side. The blob
#      templates retain that original property.
#    - BFS propagation now carries the incoming direction with each activation.
#      Emission assumes the blob's unique port direction is orthogonal to that
#      incoming direction; generator guarantees this on the active chain.
#    - Because distractors are never activated, they do not need an inferred
#      incoming direction, but they still use the same one-port blob grammar.
# d) Placement legality
#    - The generator remains output-first: choose control edge, simulate the
#      zigzag cascade in the output, then place the corresponding blue+green blobs
#      into the input and erase rays from the input.
#    - Every blob footprint is kept strictly inside the border and separated by
#      an 8-neighborhood dilation check.
#    - Entry windows are aligned to the current ray lanes, and port direction is
#      restricted to orthogonal choices before a candidate placement is accepted.
#    - Distractors are placed only in free space that does not intersect any ray.
# e) Visual observability
#    - Across examples, humans can directly see a single border pixel launching
#      inward from different edges.
#    - The active chain repeatedly turns 90 degrees at blobs, so the orthogonal
#      constraint is visible from the input/output relation instead of hidden in
#      numeric color identities.
# f) Visual cleanliness
#    - Keeping blobs off the border prevents confusion between the control pixel
#      and blob silhouettes.
#    - The original sparse spacing and at-most-one-distractor rule are preserved,
#      so the red output network remains easy to parse.
# g) Change inferability
#    - No new colors are introduced; only geometry changes.
#    - The start rule is conveyed by location-on-border, and the turning rule is
#      conveyed by the alternating orientation of consecutive ray segments.
# h) Quantifiable non-degeneracy signal
#    - Useful metrics are:
#      * control-edge distribution: samples should use all four borders rather
#        than collapsing back to bottom-only starts;
#      * orth-turn rate: every activated blob with an outgoing ray must have
#        port_dir orthogonal to its incoming ray direction.
# i) New solver assumptions guaranteed by generate_input()
#    - Exactly one non-background border cell exists.
#    - Every activated blob lies on a ray lane and has a unique orthogonal port.
#    - There is at most one distractor blob, matching the retained assert.


# --- Colors ---
BG = 8
BLUE = 1
RED = 2
GREEN = 3

# --- Directions ---
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
ORTH = {
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
    RIGHT: (UP, DOWN),
}
EDGE_TO_DIR = {
    "top": DOWN,
    "bottom": UP,
    "left": RIGHT,
    "right": LEFT,
}

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
EDGE_PAD = 2

# --- Chain / distractor ---
CHAIN_LEN_WEIGHTS = [1, 3, 3, 3]
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 300
EDGE_BAG_TEMPLATE = ["top", "bottom", "left", "right"]
CHAIN_LEN_BAG_TEMPLATE = [0, 1, 2, 2, 3, 3]
_edge_bag: list[str] = []
_chain_len_bag: list[int] = []


def _control_direction(r: int, c: int, H: int, W: int) -> tuple[int, int]:
    if int(r) == 0:
        return DOWN
    if int(r) == int(H - 1):
        return UP
    if int(c) == 0:
        return RIGHT
    if int(c) == int(W - 1):
        return LEFT
    raise AssertionError("control pixel must lie on the border")


def _touches_border(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return any(int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1)) for r, c in coords)


def _is_orthogonal(d1: tuple[int, int], d2: tuple[int, int]) -> bool:
    return int(d1[0]) * int(d2[0]) + int(d1[1]) * int(d2[1]) == 0


def _next_edge() -> str:
    global _edge_bag
    if not _edge_bag:
        _edge_bag = list(EDGE_BAG_TEMPLATE)
        random.shuffle(_edge_bag)
    return str(_edge_bag.pop())


def _next_chain_len() -> int:
    global _chain_len_bag
    if not _chain_len_bag:
        _chain_len_bag = list(CHAIN_LEN_BAG_TEMPLATE)
        random.shuffle(_chain_len_bag)
    return int(_chain_len_bag.pop())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a red zigzag ray network from one border control pixel through blue blobs.

    Input structure:
    - The background is the dominant color. Blue diamond/octagon blobs sit away from the border, and each blob has one green-marked port side.
    - Exactly one red control pixel lies on the grid border; its border side determines the initial inward ray direction.

    Rule:
    - Launch a thin red ray from the control pixel into the grid until it exits or first hits a blue blob.
    - A hit blob turns red and may emit only through a port whose direction is orthogonal to the incoming ray direction; emitted rays keep their original thickness and can activate more blobs breadth-first.
    - Leave only the red control pixel, all traversed rays, and all activated blobs in the output; unreached blobs disappear into background.

    Key insight: The border location of the lone red control determines how the cascade starts, and every activated blob must turn the path by 90 degrees because its green port side is constrained to be orthogonal to the ray that entered it.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the unique border control pixel and its inward direction. ---
    border_pts: list[tuple[int, int]] = []
    for c in range(W):
        if int(g[0, c]) != int(bg):
            border_pts.append((0, int(c)))
        if int(H - 1) != 0 and int(g[H - 1, c]) != int(bg):
            border_pts.append((int(H - 1), int(c)))
    for r in range(1, H - 1):
        if int(g[r, 0]) != int(bg):
            border_pts.append((int(r), 0))
        if int(W - 1) != 0 and int(g[r, W - 1]) != int(bg):
            border_pts.append((int(r), int(W - 1)))

    assert len(border_pts) == 1
    start_r, start_c = border_pts[0]
    red_col = int(g[start_r, start_c])
    start_dir = _control_direction(int(start_r), int(start_c), H, W)

    # --- STEP 2) Identify the blob body color and unique port-marker color. ---
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

    # --- STEP 3) Extract blue blobs and record their unique port directions. ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[int(r), int(c)] = bi
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

    activated: set[int] = set()
    queue: list[tuple[int, tuple[int, int]]] = []

    def activate_blob_at(r: int, c: int, incoming_dir: tuple[int, int]) -> None:
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append((int(bi), (int(incoming_dir[0]), int(incoming_dir[1]))))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(red_col)

    # --- STEP 4) Draw the initial inward ray from the control pixel. ---
    dr, dc = start_dir
    r, c = int(start_r), int(start_c)
    while 0 <= r < H and 0 <= c < W:
        out[int(r), int(c)] = int(red_col)
        if int(g[int(r), int(c)]) == int(blue_col):
            activate_blob_at(int(r), int(c), start_dir)
            break
        r += int(dr)
        c += int(dc)

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        ddr, ddc = map(int, d)
        for pr, pc in port_pixels:
            r = int(pr)
            c = int(pc)
            while True:
                r += int(ddr)
                c += int(ddc)
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[int(r), int(c)] = int(red_col)
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c), d)
                    break

    # --- STEP 5) BFS propagation with orthogonal turn enforcement at each hit blob. ---
    while queue:
        bi, incoming_dir = queue.pop(0)
        port = blobs[bi]["ports"][0]
        if not _is_orthogonal(tuple(port["dir"]), incoming_dir):
            continue
        extend_thick(port["pixels"], tuple(port["dir"]))

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    return out


def _shape_mask(h: int, w: int) -> np.ndarray:
    widths = _SHAPE_WIDTHS[(h, w)]
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(widths):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return int(rc[1]) if d[0] else int(rc[0])


def _along_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return int(rc[0]) if d[0] else int(rc[1])


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
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "size": (h, w),
        "mask": mask,
        "patch": patch,
        "blue": blue,
        "full": full,
        "port_dir": (int(port_dir[0]), int(port_dir[1])),
        "port_pixels": _sort_on_ray(port_pixels, port_dir),
    }


def _all_specs(port_dir: tuple[int, int] | None = None) -> list[dict]:
    dirs = [port_dir] if port_dir is not None else list(DIRS)
    out = [s for size in SHAPES for d in dirs if (s := _blob_spec(size, d)) is not None]
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
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        rr = int(r)
        cc = int(c)
        if include_origin:
            cells.append((int(rr), int(cc)))
        guard = H + W + 10
        while (int(rr), int(cc)) != (int(er), int(ec)):
            rr += int(dr)
            cc += int(dc)
            guard -= 1
            if guard < 0 or not (0 <= int(rr) < H and 0 <= int(cc) < W):
                return None
            cells.append((int(rr), int(cc)))
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
        rr = int(r)
        cc = int(c)
        if include_origin:
            cells.append((int(rr), int(cc)))
        while True:
            rr += int(dr)
            cc += int(dc)
            if not (0 <= int(rr) < H and 0 <= int(cc) < W):
                break
            cells.append((int(rr), int(cc)))
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

    fixed = int(lane_vals[0] - win_vals[0])
    extent = H - h if is_vertical else W - w
    low = int(edge_line)
    high = int(extent + edge_line)
    if d in (UP, LEFT):
        high = min(high, int(start_line - min_gap))
    else:
        low = max(low, int(start_line + min_gap))

    entry_lines = list(range(int(low), int(high) + 1))
    random.shuffle(entry_lines)
    candidates: list[dict] = []

    for entry_line in entry_lines:
        origin_along = int(entry_line - edge_line)
        r0 = int(origin_along if is_vertical else fixed)
        c0 = int(fixed if is_vertical else origin_along)
        if is_vertical:
            entry_cells = [(int(entry_line), int(c0 + c)) for _, c in window]
        else:
            entry_cells = [(int(r0 + r), int(entry_line)) for r, _ in window]
        full = _offset(spec["full"], r0, c0)
        if not _in_bounds(full, H, W):
            continue
        if _touches_border(full, H, W):
            continue
        if any(bool(occ_pad[r, c]) or bool(ray_mask[r, c]) for r, c in full):
            continue
        path = _path_to_entries(starts, d, entry_cells, include_origin, H, W)
        if path is None:
            continue
        entry_set = set(entry_cells)
        if any(bool(blue_mask[r, c]) for r, c in path if (int(r), int(c)) not in entry_set):
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
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [int(v - lane_vals[0]) for v in lane_vals]
    entry_side = OPP[d]
    allowed_ports = set(ORTH[d])
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    candidates: list[dict] = []

    for spec in _all_specs():
        if tuple(spec["port_dir"]) not in allowed_ports:
            continue
        edge = _edge_offsets(spec["mask"], entry_side)
        if len(edge) < thickness:
            continue
        for k in range(len(edge) - thickness + 1):
            window = edge[k : k + thickness]
            win_delta = [int(_perp_value(p, d) - _perp_value(window[0], d)) for p in window]
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
) -> bool:
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []
    for spec in _all_specs():
        h, w = spec["mask"].shape
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                full = _offset(spec["full"], r0, c0)
                if _touches_border(full, H, W):
                    continue
                if any(bool(occ_pad[r, c]) or bool(ray_mask[r, c]) for r, c in full):
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


def _sample_control(H: int, W: int, edge: str | None = None) -> tuple[tuple[int, int], tuple[int, int], str]:
    edge = str(edge) if edge is not None else random.choice(("top", "bottom", "left", "right"))
    if edge == "top":
        coord = (0, random.randint(EDGE_PAD, W - EDGE_PAD - 1))
    elif edge == "bottom":
        coord = (int(H - 1), random.randint(EDGE_PAD, W - EDGE_PAD - 1))
    elif edge == "left":
        coord = (random.randint(EDGE_PAD, H - EDGE_PAD - 1), 0)
    else:
        coord = (random.randint(EDGE_PAD, H - EDGE_PAD - 1), int(W - 1))
    return (int(coord[0]), int(coord[1])), EDGE_TO_DIR[edge], edge


def _gen_attempt(forced_edge: str | None = None, forced_chain_len: int | None = None) -> np.ndarray | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    (start_r, start_c), direction, _edge = _sample_control(H, W, forced_edge)
    inp[int(start_r), int(start_c)] = RED
    out[int(start_r), int(start_c)] = RED
    occupied[int(start_r), int(start_c)] = True
    ray_mask[int(start_r), int(start_c)] = True

    if forced_chain_len is None:
        chain_len = random.choices(range(len(CHAIN_LEN_WEIGHTS)), weights=CHAIN_LEN_WEIGHTS)[0]
    else:
        chain_len = int(forced_chain_len)
    min_dist = 1 if chain_len == 0 else 0
    max_dist = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - chain_len)
    num_distractors = random.randint(min_dist, max_dist)

    starts = [(int(start_r), int(start_c))]
    include_origin = True
    for _ in range(chain_len):
        blob = _candidate_next_blob(H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask)
        if blob is None:
            return None
        _mark_ray(out, ray_mask, blob["path"])
        _place_blob(inp, occupied, blue_mask, blob)
        paint_coords(out, blob["blue"], RED)
        starts = list(blob["ports"])
        direction = tuple(blob["spec"]["port_dir"])
        include_origin = False

    final_path = _path_to_boundary(starts, direction, include_origin, H, W)
    if any(bool(blue_mask[r, c]) for r, c in final_path):
        return None
    _mark_ray(out, ray_mask, final_path)

    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    forced_edge = _next_edge()
    forced_chain_len = _next_chain_len()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(forced_edge=forced_edge, forced_chain_len=forced_chain_len)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
