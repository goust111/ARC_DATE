# Combo variant of 6ffbe589 (index 26)
# combo_spec: D10/0+D11/0
# Name: 小画布+紧邻放置

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


# Design analysis for combo D10/0 + D11/0
# a) Concrete behavior trace:
#    - Original task: the solver finds the largest foreground cluster, expands its bbox by
#      margin 3, and assumes every cue still lies outside that crop. The cue area mod 4 then
#      tells how much to rotate the same-color layer inside the square board.
#    - Fused task: the canvas shrinks to 15x15, the board shrinks to side 6..9, and the cue
#      sprites are glued directly against one board edge (bbox gap exactly 1). A margin crop
#      now swallows the cue, especially on a 15x15 canvas where there is almost no spare room.
#    - Example: with an 8x8 board and a 3-cell red cue pasted just above the board, the old
#      crop would treat the red cue as part of the same local region. The new solver therefore
#      searches globally for the true 6..9 square board using its concentric geometry, blanks
#      only that square, and then counts same-color adjacent cues outside it.
# b) Complexity calibration:
#    - Keep exactly 3 board colors and 1..3 cues, so the human still sees the original
#      color-matching idea.
#    - Fix the board grammar to "outer irregular border ring + middle irregular inset ring +
#      centered inner sprite". This is strong enough to isolate the board on a tight canvas,
#      but still flexible enough to vary shape and rotation.
#    - Restrict cues to distinct sides and small areas in {1,2,3,5}. That keeps the cue code
#      legible while making the distance-1 placement visually obvious.
# c) Hidden assumptions broken in the original `main()`:
#    - `crop_components_near_largest(... margin=3)` assumed cues live beyond that halo. False.
#    - Clearing the crop bbox to make `outside_grid` assumed the crop bbox was already the board
#      bbox. False when cues extend the overall cluster.
#    - "At most one outside cue per color" still holds, but only after the true board bbox is
#      recovered first. The generator guarantees one cue at most per color and per side.
# d) Placement legality:
#    - The generator chooses the cue patches first, then samples a board anchor that leaves
#      enough room for those exact side-attached patches.
#    - Cues never overlap, never go out of bounds, and always have bbox-gap 1 from the board.
# e) Visual observability:
#    - The fusion is visible without hidden palette ordering: every sample is 15x15, the board
#      is clearly smaller than the original task, and cue blobs sit flush against the board.
#    - The board corners are always the outer color, the middle layer always has bbox inset 1,
#      and the inner layer stays centered. Those repeated supports make the board square easy
#      to infer from just a few examples.
# f) Visual cleanliness:
#    - No extra legends or frame colors are introduced. The only novelty is spatial: board and
#      cues are packed tightly on the small canvas.
#    - Distinct cue sides avoid corner pileups, so the board outline and the cue blobs remain
#      separable by eye.
# g) New solver assumptions that the generator must satisfy:
#    - Exactly one 6..9 square on the canvas has 3 colors with the required outer/middle/inner
#      bbox pattern.
#    - Every non-board foreground component is a monochrome cue adjacent to that square.
#    - The outer and middle layers always have full ring bboxes, and the inner layer always has
#      a centered bbox strictly inside the 2-cell margin. The generator validates all of this.


CANVAS_SIZE = 15
BOARD_N_MIN, BOARD_N_MAX = 6, 9
NUM_COLORS = 3

OUTER_GAP_LO, OUTER_GAP_HI = 2, 5
MIDDLE_GAP_LO, MIDDLE_GAP_HI = 1, 4
GEN_RETRIES = 500
CUE_GRID = 3
SIDES = ("top", "bottom", "left", "right")
VALID_CUE_AREAS = {1, 2, 3, 5}


def _is_rot_asym(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _is_connected(mask: np.ndarray) -> bool:
    if not np.any(mask):
        return False
    comps = find_components(mask.astype(int), bg=0, connectivity=8, monochromatic=False, return_format="dict")
    return len(comps) == 1


def _perimeter_coords(n: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    last = n - 1
    for c in range(n):
        coords.append((0, c))
    for r in range(1, last):
        coords.append((r, last))
    for c in range(last, -1, -1):
        coords.append((last, c))
    for r in range(last - 1, 0, -1):
        coords.append((r, 0))
    return coords


def _ring_corners(n: int) -> set[tuple[int, int]]:
    last = n - 1
    return {(0, 0), (0, last), (last, 0), (last, last)}


def _make_ring_mask(side: int, gap_lo: int, gap_hi: int) -> np.ndarray | None:
    coords = _perimeter_coords(side)
    area = len(coords)
    mandatory = _ring_corners(side)
    max_gap = min(int(gap_hi), area - len(mandatory))
    min_gap = min(int(gap_lo), max_gap)
    if min_gap < 1 or min_gap > max_gap:
        return None

    for _ in range(120):
        gap = random.randint(min_gap, max_gap)
        start = random.randrange(area)
        removed = {coords[(start + k) % area] for k in range(gap)}
        if removed & mandatory:
            continue
        mask = np.zeros((side, side), dtype=bool)
        for r, c in coords:
            if (r, c) not in removed:
                mask[r, c] = True
        if not _is_connected(mask):
            continue
        if not _is_rot_asym(mask):
            continue
        return mask
    return None


def _make_outer_mask(n: int) -> np.ndarray | None:
    return _make_ring_mask(n, OUTER_GAP_LO, min(OUTER_GAP_HI, n))


def _make_middle_mask(n: int) -> np.ndarray | None:
    local_side = n - 2
    local = _make_ring_mask(local_side, MIDDLE_GAP_LO, min(MIDDLE_GAP_HI, local_side - 1))
    if local is None:
        return None
    mask = np.zeros((n, n), dtype=bool)
    mask[1:-1, 1:-1] = local
    return mask


def _valid_inner_spans(n: int, require_asym: bool) -> list[int]:
    max_span = n - 4
    spans = [s for s in range(1, max_span + 1) if (s % 2) == (n % 2)]
    if require_asym:
        spans = [s for s in spans if s > 1]
    return spans


def _make_inner_mask(n: int, *, require_asym: bool) -> np.ndarray | None:
    spans = _valid_inner_spans(n, require_asym)
    if not spans:
        return None

    for _ in range(120):
        span = random.choice(spans)
        top = (n - span) // 2
        local = np.zeros((span, span), dtype=bool)

        if span == 1:
            local[0, 0] = True
        elif span == 2:
            local[:, :] = True
            local[random.randrange(2), random.randrange(2)] = False
        elif span == 3:
            variants = [
                np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=bool),
                np.array([[0, 1, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
                np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=bool),
            ]
            local = np.rot90(random.choice(variants), random.randrange(4))
        elif span == 4:
            pts = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
            for r, c in pts:
                local[r, c] = True
            if random.randrange(2):
                local = np.fliplr(local)
            local = np.rot90(local, random.randrange(4))
        else:
            pts = [(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 4)]
            for r, c in pts:
                local[r, c] = True
            if random.randrange(2):
                local = np.fliplr(local)
            local = np.rot90(local, random.randrange(4))

        if require_asym and not _is_rot_asym(local):
            continue

        mask = np.zeros((n, n), dtype=bool)
        mask[top : top + span, top : top + span] = local
        return mask
    return None


def _cue_patch(color: int, area: int) -> np.ndarray:
    center = CUE_GRID // 2
    coords = {(center, center)}
    while len(coords) < area:
        frontier = set()
        for r, c in coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < CUE_GRID and 0 <= nc < CUE_GRID and (nr, nc) not in coords:
                        frontier.add((nr, nc))
        coords.add(random.choice(sorted(frontier)))

    patch = np.zeros((CUE_GRID, CUE_GRID), dtype=int)
    for r, c in coords:
        patch[r, c] = int(color)
    pos = np.argwhere(patch != 0)
    r0, c0 = pos.min(axis=0)
    r1, c1 = pos.max(axis=0)
    return patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _cue_area_from_rotation(cw_k: int) -> int:
    rem = (-int(cw_k)) % 4
    if rem == 1:
        return random.choice([1, 5])
    assert rem in {2, 3}
    return rem


def _pick_rotations() -> list[int]:
    cue_count = random.randint(1, NUM_COLORS)
    active = set(random.sample(range(NUM_COLORS), cue_count))
    return [random.randint(1, 3) if idx in active else 0 for idx in range(NUM_COLORS)]


def _bbox_chebyshev(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    dr = max(0, br0 - ar1) if ar1 < br0 else max(0, ar0 - br1)
    dc = max(0, bc0 - ac1) if ac1 < bc0 else max(0, ac0 - bc1)
    return max(dr, dc)


def _cue_adjacent_to_bbox(cue_bbox: tuple[int, int, int, int], board_bbox: tuple[int, int, int, int]) -> bool:
    qr0, qc0, qr1, qc1 = map(int, cue_bbox)
    br0, bc0, br1, bc1 = map(int, board_bbox)
    if _bbox_chebyshev(cue_bbox, board_bbox) != 1:
        return False
    horiz_overlap = max(qc0, bc0) <= min(qc1, bc1)
    vert_overlap = max(qr0, br0) <= min(qr1, br1)
    return bool(
        (qr1 == br0 - 1 and horiz_overlap)
        or (qr0 == br1 + 1 and horiz_overlap)
        or (qc1 == bc0 - 1 and vert_overlap)
        or (qc0 == bc1 + 1 and vert_overlap)
    )


def _centered_bbox(local_bbox: tuple[int, int, int, int], n: int) -> bool:
    r0, c0, r1, c1 = map(int, local_bbox)
    return r0 + r1 == n - 1 and c0 + c1 == n - 1


def _classify_board(board: np.ndarray, bg: int) -> dict[str, int] | None:
    n = int(board.shape[0])
    if board.shape[1] != n or not (BOARD_N_MIN <= n <= BOARD_N_MAX):
        return None

    colors = colors_present(board, exclude={bg})
    if len(colors) != 3:
        return None

    corners = [int(board[0, 0]), int(board[0, -1]), int(board[-1, 0]), int(board[-1, -1])]
    if any(col == bg for col in corners) or len(set(corners)) != 1:
        return None
    outer = int(corners[0])

    outer_coords = np.argwhere(board == outer)
    if outer_coords.size == 0:
        return None
    or0, oc0 = outer_coords.min(axis=0)
    or1, oc1 = outer_coords.max(axis=0)
    if tuple(map(int, (or0, oc0, or1, oc1))) != (0, 0, n - 1, n - 1):
        return None

    remaining = [int(col) for col in colors if int(col) != outer]
    middle = None
    inner = None
    for col in remaining:
        coords = np.argwhere(board == int(col))
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox = tuple(map(int, (r0, c0, r1, c1)))
        if bbox == (1, 1, n - 2, n - 2):
            middle = int(col)
        elif r0 >= 2 and c0 >= 2 and r1 <= n - 3 and c1 <= n - 3 and _centered_bbox(bbox, n):
            inner = int(col)
        else:
            return None

    if middle is None or inner is None:
        return None

    area_outer = int(np.sum(board == outer))
    area_middle = int(np.sum(board == middle))
    area_inner = int(np.sum(board == inner))
    if not (area_outer > area_middle > area_inner > 0):
        return None

    return {"outer": outer, "middle": middle, "inner": inner}


def _collect_adjacent_cues(
    grid: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    board_colors: set[int],
    bg: int,
) -> dict[int, dict] | None:
    r0, c0, r1, c1 = map(int, board_bbox)
    outside = np.array(grid, dtype=int)
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    comps = find_components(outside, bg=int(bg), connectivity=8, monochromatic=True, return_format="dict")

    cues: dict[int, dict] = {}
    for comp in comps:
        col = int(comp["color"])
        if col not in board_colors:
            return None
        if not _cue_adjacent_to_bbox(tuple(map(int, comp["bbox"])), board_bbox):
            return None
        if int(comp["area"]) not in VALID_CUE_AREAS:
            return None
        if col in cues:
            return None
        cues[col] = comp
    if not cues:
        return None
    return cues


def _find_board_bbox(grid: np.ndarray, *, bg: int | None = None) -> tuple[tuple[int, int, int, int], dict[str, int], dict[int, dict]]:
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)
    g = np.array(grid, dtype=int)

    candidates: list[tuple[tuple[int, int, int], tuple[int, int, int, int], dict[str, int], dict[int, dict]]] = []
    for n in range(BOARD_N_MIN, BOARD_N_MAX + 1):
        for r0 in range(CANVAS_SIZE - n + 1):
            for c0 in range(CANVAS_SIZE - n + 1):
                board = g[r0 : r0 + n, c0 : c0 + n]
                info = _classify_board(board, bg)
                if info is None:
                    continue
                bbox = (r0, c0, r0 + n - 1, c0 + n - 1)
                cues = _collect_adjacent_cues(g, bbox, set(info.values()), bg)
                if cues is None:
                    continue
                score = (len(cues), n, int(np.sum(board != bg)))
                candidates.append((score, bbox, info, cues))

    assert candidates, "expected one valid 6..9 square board with adjacent cues"
    best_score = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_score]
    assert len(best) == 1, "expected a unique best square board candidate"
    _score, bbox, info, cues = best[0]
    return bbox, info, cues


def rotate_color_layer_cw(grid: np.ndarray, color: int, cw_k: int, *, bg: int | None = None) -> np.ndarray:
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)
    k = int(cw_k) % 4
    if k == 0:
        return grid

    mask = (grid == int(color)).astype(int)
    rotated = np.rot90(mask, (-k) % 4)
    out = grid.copy()
    out[out == int(color)] = int(bg)
    out[rotated == 1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the small square board, then rotate its color layers from edge-hugging cues.

    Input structure:
    - bg is the most common color on a 15x15 canvas.
    - One 6..9 square board contains 3 concentric color layers: an outer border-heavy layer,
      a one-cell-inset middle ring, and a centered inner sprite.
    - Small monochrome cue blobs of matching colors sit directly against the board edges,
      with bbox gap exactly 1 instead of floating far away.

    Rule:
    - Search the whole canvas for the unique 6..9 square whose 3 colors match the
      outer/middle/inner board grammar.
    - Blank only that square and inspect the remaining same-color cue blobs adjacent to it.
    - For each board color, rotate that full color mask clockwise by `(-cue_area) % 4`
      quarter-turns inside the cropped square.
    - If a board color has no adjacent cue, leave that layer unchanged.

    Key insight:
    Distance no longer separates the cues from the board. The stable signal is the square
    concentric board grammar itself; once that square is found, every remaining adjacent
    same-color blob is a cue whose area still encodes the quarter-turn count.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    board_bbox, _info, cues = _find_board_bbox(g, bg=bg)
    r0, c0, r1, c1 = map(int, board_bbox)
    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    out = crop.copy()

    for col in colors_present(crop, exclude={bg}):
        cue_size = int(cues[int(col)]["area"]) if int(col) in cues else 0
        rot_k = (-cue_size) % 4
        if rot_k:
            out = rotate_color_layer_cw(out, int(col), int(rot_k), bg=bg)
    return out


def _choose_anchor(n: int, side_to_patch: dict[str, np.ndarray]) -> tuple[int, int] | None:
    top_need = int(side_to_patch["top"].shape[0]) if "top" in side_to_patch else 0
    bottom_need = int(side_to_patch["bottom"].shape[0]) if "bottom" in side_to_patch else 0
    left_need = int(side_to_patch["left"].shape[1]) if "left" in side_to_patch else 0
    right_need = int(side_to_patch["right"].shape[1]) if "right" in side_to_patch else 0

    r_lo = top_need
    r_hi = CANVAS_SIZE - n - bottom_need
    c_lo = left_need
    c_hi = CANVAS_SIZE - n - right_need
    if r_lo > r_hi or c_lo > c_hi:
        return None

    choices = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
    return random.choice(choices)


def _place_cue_on_side(
    canvas: np.ndarray,
    patch: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    side: str,
) -> tuple[int, int, int, int] | None:
    h, w = map(int, patch.shape)
    br0, bc0, br1, bc1 = map(int, board_bbox)
    mask = patch != 0
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []

    if side == "top":
        r = br0 - h
        rc_iter = [(r, c) for c in range(bc0, bc1 - w + 2)]
    elif side == "bottom":
        r = br1 + 1
        rc_iter = [(r, c) for c in range(bc0, bc1 - w + 2)]
    elif side == "left":
        c = bc0 - w
        rc_iter = [(r, c) for r in range(br0, br1 - h + 2)]
    else:
        assert side == "right"
        c = bc1 + 1
        rc_iter = [(r, c) for r in range(br0, br1 - h + 2)]

    for r, c in rc_iter:
        if r < 0 or c < 0 or r + h > CANVAS_SIZE or c + w > CANVAS_SIZE:
            continue
        if np.any((canvas[r : r + h, c : c + w] != 0) & mask):
            continue
        bbox = (r, c, r + h - 1, c + w - 1)
        if not _cue_adjacent_to_bbox(bbox, board_bbox):
            continue
        candidates.append((r, c, bbox))

    if not candidates:
        return None

    r, c, bbox = random.choice(candidates)
    paste(canvas, patch, r, c, transparent=0)
    return bbox


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    board_bbox, _info, cues = _find_board_bbox(g, bg=bg)
    r0, c0, r1, c1 = map(int, board_bbox)
    fg = np.argwhere(g != bg)
    fr0, fc0 = fg.min(axis=0)
    fr1, fc1 = fg.max(axis=0)
    cue_gaps = [_bbox_chebyshev(tuple(map(int, comp["bbox"])), board_bbox) for comp in cues.values()]
    return {
        "board_side": int(r1 - r0 + 1),
        "cue_count": len(cues),
        "cluster_h": int(fr1 - fr0 + 1),
        "cluster_w": int(fc1 - fc0 + 1),
        "max_cue_gap": max(cue_gaps),
        "min_cue_gap": min(cue_gaps),
    }


def _gen_attempt() -> np.ndarray | None:
    colors = random.sample(range(1, 10), NUM_COLORS)
    rot_ks = _pick_rotations()
    n = random.randint(BOARD_N_MIN, BOARD_N_MAX)

    outer_mask = _make_outer_mask(n)
    middle_mask = _make_middle_mask(n)
    inner_mask = _make_inner_mask(n, require_asym=rot_ks[2] > 0)
    if outer_mask is None or middle_mask is None or inner_mask is None:
        return None
    if np.any(outer_mask & middle_mask) or np.any(outer_mask & inner_mask) or np.any(middle_mask & inner_mask):
        return None
    if rot_ks[2] > 0 and not _is_rot_asym(inner_mask):
        return None

    out_board = np.zeros((n, n), dtype=int)
    out_board[outer_mask] = int(colors[0])
    out_board[middle_mask] = int(colors[1])
    out_board[inner_mask] = int(colors[2])

    in_board = np.zeros((n, n), dtype=int)
    for col, cw_k in zip(colors, rot_ks):
        inv_mask = np.rot90(out_board == int(col), int(cw_k))
        if np.any((in_board != 0) & inv_mask):
            return None
        in_board[inv_mask] = int(col)

    cue_items = [(int(col), _cue_area_from_rotation(cw_k)) for col, cw_k in zip(colors, rot_ks) if int(cw_k) != 0]
    random.shuffle(cue_items)
    sides = random.sample(SIDES, len(cue_items))
    side_to_patch = {side: _cue_patch(col, area) for (col, area), side in zip(cue_items, sides)}

    anchor = _choose_anchor(n, side_to_patch)
    if anchor is None:
        return None
    r0, c0 = anchor
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    for (col, area), side in zip(cue_items, sides):
        bbox = _place_cue_on_side(canvas, side_to_patch[side], board_bbox, side)
        if bbox is None:
            return None
        cue_bboxes.append(bbox)

    found_bbox, found_info, found_cues = _find_board_bbox(canvas, bg=0)
    if found_bbox != board_bbox:
        return None
    if set(found_info.values()) != set(map(int, colors)):
        return None
    if len(found_cues) != len(cue_items):
        return None
    if any(_bbox_chebyshev(bb, board_bbox) != 1 for bb in cue_bboxes):
        return None
    if not np.array_equal(main(canvas), out_board):
        return None
    metrics = _variant_metrics(canvas)
    if metrics["board_side"] != n or metrics["min_cue_gap"] != 1 or metrics["max_cue_gap"] != 1:
        return None
    if max(metrics["cluster_h"], metrics["cluster_w"]) < n + 1:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused 6ffbe589 input")
