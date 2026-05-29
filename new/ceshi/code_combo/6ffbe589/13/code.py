# Combo variant of 6ffbe589 (index 13)
# combo_spec: D11/0+D16/1
# Name: 精灵紧邻板+边框标记

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


# Design analysis for the fused variant:
# - Original task assumption: the board is the unique largest foreground cluster,
#   and each cue sprite sits several cells away from it. The solver can crop near
#   the largest component and safely treat everything else as "outside".
# - D11/0 breaks that assumption: if a cue is pressed directly against the board,
#   then the largest multicolor component can absorb both the board and the cue.
#   A margin-based crop would now keep the cue inside the working area and lose
#   the inside/outside distinction that the original solver relies on.
# - D16/1 restores a visible separator by wrapping the board with a 1-cell square
#   frame in a fourth color. The cue is not detached anymore; it is 4-neighbor
#   adjacent to the frame. The correct extraction path is therefore:
#   1) find the special monochrome rectangular frame,
#   2) take only its interior as the board,
#   3) inspect frame-touching monochrome components outside the frame as cues.
# - Concrete trace: suppose a red 3-cell cue is glued above the top edge of a
#   gray 12x12 frame, with a 10x10 multicolor board inside. The original solver
#   would crop the whole connected cluster and fail to see an "outside" red cue.
#   The fused solver instead finds the gray rectangular ring, crops the 10x10
#   interior, then measures the red cue area from the component touching the
#   frame's top side.
# - Complexity calibration: only one extra structural color is added, and the cue
#   still uses the original "area mod 4" code. This keeps the mapping compact and
#   human-visible while making the spatial grouping meaningfully different.
# - Hidden assumptions now enforced by the generator:
#   * exactly one largest square frame-like monochrome component exists;
#   * every cue lies fully outside the frame but touches one frame side;
#   * each rotating board color has at most one outside cue;
#   * the frame interior is square, so per-color layer rotation remains valid.
# - Visual cleanliness: the frame makes the board boundary explicit, and cues are
#   limited to at most one per side to avoid cluttered overlaps around corners.


CANVAS_SIZE = 20
BOARD_N_MIN, BOARD_N_MAX = 10, 12
NUM_COLORS = 3
FRAME_WIDTH = 1
SAFE_MARGIN = 3

OUTER_DENSITY_LO = 0.17
OUTER_DENSITY_HI = 0.32
MID_DENSITY_LO = 0.30
MID_DENSITY_HI = 0.50
MID_OUTER_CAP = 0.32

LAYER_BAND_WIDTH = 2
MIDDLE_INSET = 2
ASYM_MARKER = (1, 2)
INNER_SPAN_MARGIN = 8

CUE_GRID = 3
OUTER_AREA_PAD = 4
GEN_RETRIES = 400
SIDES = ("top", "bottom", "left", "right")


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _rot_orbit(r: int, c: int, n: int) -> set[tuple[int, int]]:
    pts = set()
    rr, cc = int(r), int(c)
    for _ in range(4):
        pts.add((rr, cc))
        rr, cc = n - 1 - cc, rr
    return pts


def _is_rot_asym(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _checker_perimeter(n: int, phase: int) -> np.ndarray:
    rr, cc = np.indices((n, n))
    perim = (rr == 0) | (rr == n - 1) | (cc == 0) | (cc == n - 1)
    return perim & (((rr + cc + int(phase)) % 2) == 0)


def _band_cells(n: int, width: int) -> list[tuple[int, int]]:
    rr, cc = np.indices((n, n))
    inside = np.minimum(np.minimum(rr, cc), np.minimum(n - 1 - rr, n - 1 - cc)) < int(width)
    coords = np.argwhere(inside)
    return [(int(r), int(c)) for r, c in coords]


def _fill_to_count(mask: np.ndarray, candidates: list[tuple[int, int]], target: int) -> bool:
    random.shuffle(candidates)
    for r, c in candidates:
        if int(mask.sum()) >= int(target):
            return True
        mask[int(r), int(c)] = True
    return int(mask.sum()) >= int(target)


def _make_asym_layer(
    n: int,
    density_lo: float,
    density_hi: float,
    *,
    extra_lo_terms: list[int] | None = None,
) -> np.ndarray | None:
    mask = _checker_perimeter(n, random.randrange(2))

    orbit = _rot_orbit(*ASYM_MARKER, n)
    for p in orbit - {ASYM_MARKER}:
        mask[p] = False
    mask[ASYM_MARKER] = True

    extra = list(extra_lo_terms or [])
    lo = max(int(np.ceil(density_lo * n * n)), int(mask.sum()), *extra)
    hi = int(np.floor(density_hi * n * n))
    if lo > hi:
        return None
    target = random.randint(lo, hi)

    blocked = orbit - {ASYM_MARKER}
    candidates = [(r, c) for r, c in _band_cells(n, LAYER_BAND_WIDTH) if (r, c) not in blocked and not mask[r, c]]
    if not _fill_to_count(mask, candidates, target):
        return None
    if not _is_rot_asym(mask):
        return None
    return mask


def _make_middle_mask(n: int) -> np.ndarray | None:
    m = n - 2 * MIDDLE_INSET
    hi_cap = int(np.floor(MID_OUTER_CAP * n * n)) - 5
    local = _make_asym_layer(m, MID_DENSITY_LO, min(MID_DENSITY_HI, hi_cap / (m * m)))
    if local is None:
        return None
    mask = np.zeros((n, n), dtype=bool)
    mask[MIDDLE_INSET : MIDDLE_INSET + m, MIDDLE_INSET : MIDDLE_INSET + m] = local
    if not _is_rot_asym(mask):
        return None
    return mask


def _make_outer_mask(n: int, min_area: int) -> np.ndarray | None:
    return _make_asym_layer(n, OUTER_DENSITY_LO, OUTER_DENSITY_HI, extra_lo_terms=[int(min_area)])


def _valid_inner_spans(n: int, require_asym: bool) -> list[int]:
    max_span = n - INNER_SPAN_MARGIN
    spans = [s for s in range(1, max_span + 1) if (s % 2) == (n % 2)]
    if require_asym:
        spans = [s for s in spans if s > 1]
    return spans


def _make_inner_mask(n: int, *, require_asym: bool) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    span = random.choice(_valid_inner_spans(n, require_asym))
    top = (n - span) // 2

    if span == 1:
        local = np.ones((1, 1), dtype=bool)
    elif span == 2:
        local = np.ones((2, 2), dtype=bool)
        local[random.randrange(2), random.randrange(2)] = False
    elif span == 3:
        variants = [
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=bool),
            np.array([[0, 1, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
            np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=bool),
        ]
        local = np.rot90(random.choice(variants), random.randrange(4))
    else:
        local = np.zeros((span, span), dtype=bool)
        pts = (
            [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
            if span == 4
            else [(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 4)]
        )
        for r, c in pts:
            local[r, c] = True
        if random.randrange(2):
            local = np.fliplr(local)
        local = np.rot90(local, random.randrange(4))

    mask[top : top + local.shape[0], top : top + local.shape[1]] = local
    return mask


def _pick_rotations() -> list[int]:
    # Balance the 1/2/3-cue branches explicitly so the fused task does not
    # collapse into mostly "many cues" training examples.
    cue_count = random.randint(1, NUM_COLORS)
    active = set(random.sample(range(NUM_COLORS), cue_count))
    rot_ks = [random.randint(1, 3) if idx in active else 0 for idx in range(NUM_COLORS)]
    return rot_ks


def _cue_patch(color: int, area: int) -> np.ndarray:
    center = CUE_GRID // 2
    coords = {(center, center)}
    while len(coords) < int(area):
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

    arr = np.zeros((CUE_GRID, CUE_GRID), dtype=int)
    for r, c in coords:
        arr[r, c] = int(color)
    pos = np.argwhere(arr != 0)
    r0, c0 = pos.min(axis=0)
    r1, c1 = pos.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _bbox_chebyshev(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    dr = max(0, br0 - ar1) if ar1 < br0 else max(0, ar0 - br1)
    dc = max(0, bc0 - ac1) if ac1 < bc0 else max(0, ac0 - bc1)
    return max(dr, dc)


def _is_full_square_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h != w or h < 4:
        return False
    expected_area = 4 * h - 4
    if int(comp["area"]) != expected_area:
        return False

    local = np.zeros((h, w), dtype=bool)
    coords = comp["coords"]
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True

    frame = np.zeros((h, w), dtype=bool)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    return bool(np.array_equal(local, frame))


def _find_board_frame(grid: np.ndarray, *, bg: int | None = None) -> tuple[int, tuple[int, int, int, int]]:
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frames = [comp for comp in comps if _is_full_square_frame(comp)]
    assert frames, "expected a square frame around the board"

    max_side = max(_bbox_h(comp["bbox"]) for comp in frames)
    largest = [comp for comp in frames if _bbox_h(comp["bbox"]) == max_side]
    assert len(largest) == 1, "expected a unique largest square frame"
    frame = largest[0]
    return int(frame["color"]), tuple(map(int, frame["bbox"]))


def _cue_touches_frame(cue_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]) -> bool:
    qr0, qc0, qr1, qc1 = map(int, cue_bbox)
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)

    horiz_overlap = max(qc0, fc0) <= min(qc1, fc1)
    vert_overlap = max(qr0, fr0) <= min(qr1, fr1)
    return bool(
        (qr1 == fr0 - 1 and horiz_overlap)
        or (qr0 == fr1 + 1 and horiz_overlap)
        or (qc1 == fc0 - 1 and vert_overlap)
        or (qc0 == fc1 + 1 and vert_overlap)
    )


def rotate_color_layer(grid: np.ndarray, color: int, rot_k: int, *, bg: int | None = None) -> np.ndarray:
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    k = int(rot_k) % 4
    if k == 0:
        return grid

    mask = (grid == int(color)).astype(int)
    rmask = np.rot90(mask, k)
    assert rmask.shape == mask.shape, "expected a square crop for layer rotation"

    out = grid.copy()
    out[out == int(color)] = bg
    out[rmask == 1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the framed board interior, then rotate its color layers from frame-touching cues.

    Input structure:
    - bg is the most common color.
    - The actual board sits inside a 1-cell-thick square frame of a special color.
    - Small monochrome cue sprites stay outside the frame, but each cue is pressed
      directly against one side of the frame instead of floating far away.

    Rule:
    - Detect the square border component first and crop only the frame interior.
    - For each board color, look outside the frame for a same-color cue that touches
      the frame.
    - Rotate that interior color layer by the cue's area code: the solver applies
      `np.rot90(layer, (-area) % 4)` inside the cropped square.
    - Colors without a touching outside cue stay unchanged.

    Key insight:
    The cue is no longer separable by distance, so the special frame becomes the
    true delimiter: it tells you where the board ends and which nearby sprite is
    an instruction rather than part of the board itself.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    _frame_color, frame_bbox = _find_board_frame(g, bg=bg)
    fr0, fc0, fr1, fc1 = frame_bbox
    crop = g[fr0 + FRAME_WIDTH : fr1, fc0 + FRAME_WIDTH : fc1].copy()
    assert crop.shape[0] == crop.shape[1] > 0, "expected a non-empty square board interior"

    colors = colors_present(crop, exclude={bg})
    assert colors, "expected the board interior to contain foreground colors"

    out = crop.copy()
    outside = g.copy()
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = bg

    for col in colors:
        comps = find_components(
            outside,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[int(col)],
            return_format="dict",
        )
        touching = [comp for comp in comps if _cue_touches_frame(comp["bbox"], frame_bbox)]
        assert len(touching) <= 1, f"expected at most one frame-touching cue for color {col}"
        cue_size = int(touching[0]["area"]) if touching else 0
        rot_k = (-cue_size) % 4
        if rot_k:
            out = rotate_color_layer(out, int(col), int(rot_k), bg=bg)

    return out


def _area_from_rotation(rot_k: int) -> int:
    rem = (-int(rot_k)) % 4
    if rem == 1:
        return random.choice([1, 5])
    assert rem in {2, 3}
    return rem


def _frame_patch(board: np.ndarray, frame_color: int) -> np.ndarray:
    n = int(board.shape[0])
    patch = np.full((n + 2 * FRAME_WIDTH, n + 2 * FRAME_WIDTH), int(frame_color), dtype=int)
    patch[FRAME_WIDTH:-FRAME_WIDTH, FRAME_WIDTH:-FRAME_WIDTH] = board
    return patch


def _board_anchor(total_side: int) -> tuple[int, int]:
    hi = CANVAS_SIZE - int(total_side) - SAFE_MARGIN
    lo = SAFE_MARGIN
    assert lo <= hi, "board+frame is too large for the requested safety margin"
    return random.randint(lo, hi), random.randint(lo, hi)


def _place_cue_on_side(
    canvas: np.ndarray,
    patch: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    side: str,
    cue_bboxes: list[tuple[int, int, int, int]],
) -> bool:
    h, w = map(int, patch.shape)
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    mask = patch != 0
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []

    if side == "top":
        r = fr0 - h
        cols = range(fc0, fc1 - w + 2)
        rc_iter = [(r, c) for c in cols]
    elif side == "bottom":
        r = fr1 + 1
        cols = range(fc0, fc1 - w + 2)
        rc_iter = [(r, c) for c in cols]
    elif side == "left":
        c = fc0 - w
        rows = range(fr0, fr1 - h + 2)
        rc_iter = [(r, c) for r in rows]
    else:
        assert side == "right"
        c = fc1 + 1
        rows = range(fr0, fr1 - h + 2)
        rc_iter = [(r, c) for r in rows]

    for r, c in rc_iter:
        if r < 0 or c < 0 or r + h > CANVAS_SIZE or c + w > CANVAS_SIZE:
            continue
        if np.any((canvas[r : r + h, c : c + w] != 0) & mask):
            continue
        bb = (r, c, r + h - 1, c + w - 1)
        if any(_bbox_chebyshev(bb, prev) < 2 for prev in cue_bboxes):
            continue
        candidates.append((r, c, bb))

    if not candidates:
        return False

    r, c, bb = random.choice(candidates)
    paste(canvas, patch, r, c, transparent=0)
    cue_bboxes.append(bb)
    return True


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _frame_color, frame_bbox = _find_board_frame(g, bg=bg)
    fr0, fc0, fr1, fc1 = frame_bbox

    outside = g.copy()
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = bg
    comps = find_components(outside, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    touching = [comp for comp in comps if _cue_touches_frame(comp["bbox"], frame_bbox)]
    return {
        "frame_side": _bbox_h(frame_bbox),
        "touching_cues": len(touching),
        "interior_side": _bbox_h(frame_bbox) - 2 * FRAME_WIDTH,
    }


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    board_colors = random.sample(range(1, 10), NUM_COLORS)
    border_color = random.choice([c for c in range(1, 10) if c not in board_colors])
    rot_ks = _pick_rotations()

    valid_ns = [n for n in range(BOARD_N_MIN, BOARD_N_MAX + 1) if _valid_inner_spans(n, rot_ks[2] > 0)]
    n = random.choice(valid_ns)

    mid_mask = _make_middle_mask(n)
    if mid_mask is None:
        return None
    inner_mask = _make_inner_mask(n, require_asym=rot_ks[2] > 0)
    if rot_ks[2] > 0 and not _is_rot_asym(inner_mask):
        return None
    outer_mask = _make_outer_mask(n, int(mid_mask.sum()) + OUTER_AREA_PAD)
    if outer_mask is None:
        return None

    if np.any(outer_mask & mid_mask) or np.any(outer_mask & inner_mask) or np.any(mid_mask & inner_mask):
        return None

    out_board = np.zeros((n, n), dtype=int)
    out_board[outer_mask] = int(board_colors[0])
    out_board[mid_mask] = int(board_colors[1])
    out_board[inner_mask] = int(board_colors[2])

    in_board = np.zeros((n, n), dtype=int)
    for col, k in zip(board_colors, rot_ks):
        inv = np.rot90(out_board == int(col), (-int(k)) % 4)
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = int(col)

    frame_patch = _frame_patch(in_board, border_color)
    total_side = int(frame_patch.shape[0])
    r0, c0 = _board_anchor(total_side)

    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, frame_patch, r0, c0, transparent=0)
    frame_bbox = (r0, c0, r0 + total_side - 1, c0 + total_side - 1)

    cue_items = [(int(col), _area_from_rotation(k)) for col, k in zip(board_colors, rot_ks) if int(k) != 0]
    random.shuffle(cue_items)
    sides = random.sample(SIDES, len(cue_items))
    cue_bboxes: list[tuple[int, int, int, int]] = []
    for (col, area), side in zip(cue_items, sides):
        if not _place_cue_on_side(canvas, _cue_patch(col, area), frame_bbox, side, cue_bboxes):
            return None

    frame_color_found, found_bbox = _find_board_frame(canvas, bg=0)
    if frame_color_found != border_color or found_bbox != frame_bbox:
        return None
    if not np.array_equal(main(canvas), out_board):
        return None

    metrics = _variant_metrics(canvas)
    if metrics["touching_cues"] != len(cue_items):
        return None

    return canvas, out_board


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        item = _gen_attempt()
        if item is not None:
            return item[0]
    raise RuntimeError("failed to generate a valid fused 6ffbe589 input")
