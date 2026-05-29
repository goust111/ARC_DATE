# Combo variant of 5dbc8537 (index 14)
# combo_spec: D5/1+D6/0+D7/0
# Name: 居中模板轨迹涂色全网格输出

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, crop_bbox, find_components, get_bg, paste

# Design analysis for the fused variant
# a) Concrete geometry:
#    The original task crops to the template bbox and treats every non-bg component
#    outside that bbox as a loose rectangle to be packed into internal holes. In this
#    combo, the template becomes a centered floating block. Its holes are rectangular
#    recesses cut from the top/bottom/left/right edges inward. Each source sprite is a
#    same-sized rectangle placed on the matching side of the template and aligned on the
#    orthogonal axis, e.g. a left-side hole at rows 5..7 is paired with a left-side
#    source sprite on those same rows. The output keeps the full grid, clears the source
#    locations, paints a colored straight trail in the gap outside the template, and
#    fills the recess.
# b) Complexity calibration:
#    Using edge recesses rather than arbitrary internal cavities is a deliberate choice:
#    it makes the four-way movement visually obvious and keeps trail pixels from cutting
#    across the base-colored template. Complexity is preserved by using 5-6 pieces, at
#    least one from each side, mixed sizes, and occasional two-color sprite patterns.
# c) Assumption audit against the original solver:
#    - "output is tight-cropped to the template bbox" no longer holds; the new solver
#      must return a full-grid copy.
#    - "every non-bg component outside the bbox is just a source sprite" still holds in
#      the input, but the output adds extra non-bg trail pixels, so source extraction is
#      performed only from the input.
#    - "holes can be found anywhere inside the bbox" is narrowed to rectangular bg
#      recesses that touch exactly one side of the bbox; this is what allows side-aware
#      straight-line motion.
#    - "matching is only by rectangle size" becomes "side + rectangle size + aligned
#      span", because centered placement means sprites can exist on all four sides.
# d) Placement legality:
#    The generator centers the template with >=5-cell margins, keeps holes away from
#    corners, uses depths 2-3 so opposite recesses cannot connect, and aligns source
#    rectangles on the same row-band/col-band as their targets. That guarantees no
#    overlap, no out-of-bounds placements, and non-overlapping trails.
# e) Visual observability / cleanliness:
#    Every sample contains all four directions. Trails live only in the empty margin
#    outside the template, so the filled template remains legible. Source cells become bg
#    in the output, making the "movement" observable instead of looking like duplication.
# f) New solver assumptions and generator support:
#    The solver assumes each hole component is a solid rectangle touching exactly one
#    template side, and each source sprite is a solid rectangle located wholly on one
#    side of the centered template with identical orthogonal span. The generator enforces
#    exactly those constraints.

GRID_MIN, GRID_MAX = 20, 25
TPL_MIN, TPL_MAX = 10, 13
MARGIN_MIN = 5
SIDE_NAMES = ("top", "bottom", "left", "right")
CORNER_BUFFER = 3
HOLE_DEPTH_MIN, HOLE_DEPTH_MAX = 2, 3
HOLE_LEN_MIN, HOLE_LEN_MAX = 1, 4
EXTRA_HOLES_MIN, EXTRA_HOLES_MAX = 1, 2
BI_COLOR_PROB = 0.35
MAX_RETRIES = 400


def _base_color(grid: np.ndarray, bg: int) -> int:
    candidates: list[tuple[int, int]] = []
    for col in colors_present(grid, exclude={bg}):
        max_area = 0
        for coords in components_mask(grid == int(col), connectivity=8):
            max_area = max(int(max_area), int(coords.shape[0]))
        candidates.append((int(max_area), int(col)))
    assert candidates
    candidates.sort(reverse=True)
    if len(candidates) > 1:
        assert candidates[0][0] > candidates[1][0]
    return int(candidates[0][1])


def _hole_side(local_bbox: tuple[int, int, int, int], dh: int, dw: int) -> str:
    r0, c0, r1, c1 = map(int, local_bbox)
    sides: list[str] = []
    if r0 == 0:
        sides.append("top")
    if r1 == dh - 1:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c1 == dw - 1:
        sides.append("right")
    assert len(sides) == 1
    return sides[0]


def _source_side(src_bbox: tuple[int, int, int, int], tpl_bbox: tuple[int, int, int, int]) -> str:
    sr0, sc0, sr1, sc1 = map(int, src_bbox)
    tr0, tc0, tr1, tc1 = map(int, tpl_bbox)
    if sr1 < tr0:
        return "top"
    if sr0 > tr1:
        return "bottom"
    if sc1 < tc0:
        return "left"
    if sc0 > tc1:
        return "right"
    raise AssertionError("source component overlaps template bbox")


def _band_key(side: str, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    if side in {"top", "bottom"}:
        return int(c0), int(c1)
    return int(r0), int(r1)


def _trail_color(patch: np.ndarray) -> int:
    # The trail color should be recoverable from the moved piece itself, so use
    # the visible perimeter color rather than an arbitrary anchor pixel.
    if patch.shape == (1, 1):
        return int(patch[0, 0])
    border = np.concatenate(
        (
            patch[0, :],
            patch[-1, :],
            patch[1:-1, 0],
            patch[1:-1, -1],
        )
    )
    colors, counts = np.unique(border, return_counts=True)
    return int(colors[np.argmax(counts)])


def _paint_trail(out: np.ndarray, side: str, src_bbox: tuple[int, int, int, int], hole_bbox: tuple[int, int, int, int], color: int) -> None:
    sr0, sc0, sr1, sc1 = map(int, src_bbox)
    hr0, hc0, hr1, hc1 = map(int, hole_bbox)
    if side == "top":
        if sr1 + 1 <= hr0 - 1:
            out[sr1 + 1 : hr0, hc0 : hc1 + 1] = int(color)
    elif side == "bottom":
        if hr1 + 1 <= sr0 - 1:
            out[hr1 + 1 : sr0, hc0 : hc1 + 1] = int(color)
    elif side == "left":
        if sc1 + 1 <= hc0 - 1:
            out[hr0 : hr1 + 1, sc1 + 1 : hc0] = int(color)
    else:
        if hc1 + 1 <= sc0 - 1:
            out[hr0 : hr1 + 1, hc1 + 1 : sc0] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move side-aligned outer rectangles into a centered template, keeping the whole canvas and painting trails.

    Input structure:
    - bg = most common color. The largest non-bg connected component is a centered base-colored template.
    - Rectangular bg recesses are carved into the template from all four sides.
    - Outside the template, rectangular sprites sit on the matching side and share the same row-band or col-band as their destination recess.

    Rule:
    - Detect the centered template by its largest base-colored component and keep the output at the original grid size.
    - Match each outside sprite to the unique recess with the same side, size, and aligned span.
    - Remove the sprite from its source location, paint a straight colored trail in the empty gap, and paste the sprite into the recess.

    Key insight:
    - The recesses touch exactly one side of the template bbox, so side + rectangle size + shared alignment band uniquely determines where each sprite slides.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    base_col = _base_color(g, bg)
    base_coords = np.argwhere(g == int(base_col))
    assert base_coords.size > 0
    tr0, tc0, tr1, tc1 = map(int, bbox_coords(base_coords))
    tpl_bbox = (int(tr0), int(tc0), int(tr1), int(tc1))

    tpl = crop_bbox(g, tpl_bbox)
    dh, dw = map(int, tpl.shape)
    hole_components = components_mask(tpl == int(bg), connectivity=4)
    assert hole_components

    holes: dict[tuple[str, int, int, tuple[int, int]], dict] = {}
    for coords in hole_components:
        lr0, lc0, lr1, lc1 = map(int, bbox_coords(coords))
        h = int(lr1 - lr0 + 1)
        w = int(lc1 - lc0 + 1)
        assert int(coords.shape[0]) == int(h * w)
        side = _hole_side((lr0, lc0, lr1, lc1), dh, dw)
        global_bbox = (int(tr0 + lr0), int(tc0 + lc0), int(tr0 + lr1), int(tc0 + lc1))
        key = (side, h, w, _band_key(side, global_bbox))
        assert key not in holes
        holes[key] = {"bbox": global_bbox}

    masked = g.copy()
    masked[tr0 : tr1 + 1, tc0 : tc1 + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert sprite_comps

    sprites: dict[tuple[str, int, int, tuple[int, int]], dict] = {}
    for comp in sprite_comps:
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (sr0, sc0, sr1, sc1))
        h, w = map(int, patch.shape)
        assert np.all(patch != int(bg))
        side = _source_side((sr0, sc0, sr1, sc1), tpl_bbox)
        key = (side, h, w, _band_key(side, (sr0, sc0, sr1, sc1)))
        assert key not in sprites
        sprites[key] = {"bbox": (sr0, sc0, sr1, sc1), "patch": patch, "coords": comp["coords"]}

    assert set(holes) == set(sprites)

    out = g.copy()
    for rec in sprites.values():
        coords = rec["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for key, sp in sprites.items():
        side = key[0]
        hole_bbox = holes[key]["bbox"]
        trail_col = _trail_color(sp["patch"])
        _paint_trail(out, side, sp["bbox"], hole_bbox, trail_col)
        paste(out, sp["patch"], hole_bbox[0], hole_bbox[1], overwrite=True)

    return out


def _sample_spans(length: int, count: int) -> list[tuple[int, int]] | None:
    lo = CORNER_BUFFER
    hi = length - CORNER_BUFFER - 1
    if hi < lo:
        return None
    candidates = list(range(lo, hi + 1))
    for _ in range(80):
        spans: list[tuple[int, int]] = []
        used = np.zeros(length, dtype=bool)
        ok = True
        for _k in range(count):
            seg_len = random.randint(HOLE_LEN_MIN, min(HOLE_LEN_MAX, hi - lo + 1))
            starts: list[int] = []
            for start in candidates:
                end = start + seg_len - 1
                if end > hi:
                    continue
                left = max(0, start - 1)
                right = min(length - 1, end + 1)
                if used[left : right + 1].any():
                    continue
                starts.append(int(start))
            if not starts:
                ok = False
                break
            start = random.choice(starts)
            end = start + seg_len - 1
            used[start : end + 1] = True
            if start - 1 >= 0:
                used[start - 1] = True
            if end + 1 < length:
                used[end + 1] = True
            spans.append((int(start), int(seg_len)))
        if ok:
            spans.sort()
            return spans
    return None


def _make_patch(h: int, w: int, primary: int, secondary: int | None) -> np.ndarray:
    patch = np.full((h, w), int(primary), dtype=int)
    if secondary is None:
        return patch

    inner_cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    if not inner_cells:
        return patch

    mode = random.choice(("dots", "row", "col"))
    if mode == "dots":
        random.shuffle(inner_cells)
        n = max(1, min(len(inner_cells), max(1, ((h - 2) * (w - 2)) // 2)))
        for r, c in inner_cells[:n]:
            patch[r, c] = int(secondary)
    elif mode == "row" and h > 2:
        start = random.randint(1, h - 2)
        patch[start : h - 1, 1:-1] = int(secondary)
    elif w > 2:
        start = random.randint(1, w - 2)
        patch[1:-1, start : w - 1] = int(secondary)
    else:
        r, c = random.choice(inner_cells)
        patch[r, c] = int(secondary)
    return patch


def _build_holes(dh: int, dw: int) -> list[dict] | None:
    side_counts = {side: 1 for side in SIDE_NAMES}
    for _ in range(random.randint(EXTRA_HOLES_MIN, EXTRA_HOLES_MAX)):
        side = random.choice(SIDE_NAMES)
        if side_counts[side] < 2:
            side_counts[side] += 1

    top_spans = _sample_spans(dw, side_counts["top"])
    bottom_spans = _sample_spans(dw, side_counts["bottom"])
    left_spans = _sample_spans(dh, side_counts["left"])
    right_spans = _sample_spans(dh, side_counts["right"])
    if any(spans is None for spans in (top_spans, bottom_spans, left_spans, right_spans)):
        return None

    holes: list[dict] = []
    for side, spans in (
        ("top", top_spans),
        ("bottom", bottom_spans),
        ("left", left_spans),
        ("right", right_spans),
    ):
        assert spans is not None
        for start, seg_len in spans:
            if side in {"top", "bottom"}:
                depth = random.randint(HOLE_DEPTH_MIN, min(HOLE_DEPTH_MAX, dh // 2 - 1))
                r = 0 if side == "top" else dh - depth
                c = start
                h, w = depth, seg_len
            else:
                depth = random.randint(HOLE_DEPTH_MIN, min(HOLE_DEPTH_MAX, dw // 2 - 1))
                r = start
                c = 0 if side == "left" else dw - depth
                h, w = seg_len, depth
            holes.append({"side": side, "r": int(r), "c": int(c), "h": int(h), "w": int(w)})

    mask = np.zeros((dh, dw), dtype=bool)
    for hole in holes:
        r, c, h, w = hole["r"], hole["c"], hole["h"], hole["w"]
        if mask[r : r + h, c : c + w].any():
            return None
        mask[r : r + h, c : c + w] = True
    return holes


def _assign_pieces(holes: list[dict], bg: int, base: int) -> list[dict] | None:
    available = [c for c in range(10) if c not in (bg, base)]
    random.shuffle(available)
    if len(available) < 4:
        return None
    pieces: list[dict] = []
    used_keys: set[tuple[str, int, int, int, int]] = set()
    for hole in holes:
        h, w = hole["h"], hole["w"]
        key = (hole["side"], int(h), int(w), int(hole["r"]), int(hole["c"]))
        assert key not in used_keys
        used_keys.add(key)
        primary = available.pop() if available else random.choice([c for c in range(10) if c not in (bg, base)])
        secondary = None
        if random.random() < BI_COLOR_PROB:
            choices = [c for c in range(10) if c not in (bg, base, primary)]
            if choices:
                secondary = random.choice(choices)
        patch = _make_patch(h, w, primary, secondary)
        pieces.append({**hole, "patch": patch})
    return pieces


def _source_top_left(piece: dict, tr0: int, tc0: int, tr1: int, tc1: int) -> tuple[int, int] | None:
    side = piece["side"]
    r = tr0 + piece["r"]
    c = tc0 + piece["c"]
    h = piece["h"]
    w = piece["w"]

    if side == "top":
        max_gap = tr0 - h
        if max_gap < 2:
            return None
        gap = random.randint(2, max_gap)
        return int(tr0 - gap - h), int(c)
    if side == "bottom":
        max_gap = piece["grid_h"] - (tr1 + 1) - h
        if max_gap < 2:
            return None
        gap = random.randint(2, max_gap)
        return int(tr1 + 1 + gap), int(c)
    if side == "left":
        max_gap = tc0 - w
        if max_gap < 2:
            return None
        gap = random.randint(2, max_gap)
        return int(r), int(tc0 - gap - w)
    max_gap = piece["grid_w"] - (tc1 + 1) - w
    if max_gap < 2:
        return None
    gap = random.randint(2, max_gap)
    return int(r), int(tc1 + 1 + gap)


def _render_input(H: int, W: int, bg: int, base: int, tr0: int, tc0: int, dh: int, dw: int, pieces: list[dict]) -> np.ndarray | None:
    tr1 = tr0 + dh - 1
    tc1 = tc0 + dw - 1
    inp = np.full((H, W), int(bg), dtype=int)
    inp[tr0 : tr1 + 1, tc0 : tc1 + 1] = int(base)

    for piece in pieces:
        r, c, h, w = piece["r"], piece["c"], piece["h"], piece["w"]
        inp[tr0 + r : tr0 + r + h, tc0 + c : tc0 + c + w] = int(bg)

    occupancy = np.zeros((H, W), dtype=bool)
    occupancy[tr0 : tr1 + 1, tc0 : tc1 + 1] = True
    for piece in pieces:
        piece["grid_h"] = int(H)
        piece["grid_w"] = int(W)
        src = _source_top_left(piece, tr0, tc0, tr1, tc1)
        if src is None:
            return None
        sr0, sc0 = src
        h, w = piece["h"], piece["w"]
        if occupancy[sr0 : sr0 + h, sc0 : sc0 + w].any():
            return None
        piece["src_r"] = int(sr0)
        piece["src_c"] = int(sc0)
        paste(inp, piece["patch"], sr0, sc0)
        occupancy[sr0 : sr0 + h, sc0 : sc0 + w] = True
    return inp


def _render_expected(H: int, W: int, bg: int, base: int, tr0: int, tc0: int, dh: int, dw: int, pieces: list[dict]) -> np.ndarray:
    out = np.full((H, W), int(bg), dtype=int)
    out[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)
    for piece in pieces:
        hr0 = tr0 + piece["r"]
        hc0 = tc0 + piece["c"]
        hole_bbox = (int(hr0), int(hc0), int(hr0 + piece["h"] - 1), int(hc0 + piece["w"] - 1))
        src_bbox = (
            int(piece["src_r"]),
            int(piece["src_c"]),
            int(piece["src_r"] + piece["h"] - 1),
            int(piece["src_c"] + piece["w"] - 1),
        )
        _paint_trail(out, piece["side"], src_bbox, hole_bbox, _trail_color(piece["patch"]))
        paste(out, piece["patch"], hr0, hc0)
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(GRID_MIN, GRID_MAX)
    W = random.randint(GRID_MIN, GRID_MAX)
    max_dh = min(TPL_MAX, H - 2 * MARGIN_MIN)
    max_dw = min(TPL_MAX, W - 2 * MARGIN_MIN)
    if max_dh < TPL_MIN or max_dw < TPL_MIN:
        return None
    dh = random.randint(TPL_MIN, max_dh)
    dw = random.randint(TPL_MIN, max_dw)
    tr0 = (H - dh) // 2
    tc0 = (W - dw) // 2

    holes = _build_holes(dh, dw)
    if holes is None:
        return None

    bg, base = random.sample(range(10), 2)
    pieces = _assign_pieces(holes, bg, base)
    if pieces is None:
        return None

    try:
        inp = _render_input(H, W, bg, base, tr0, tc0, dh, dw, pieces)
        if inp is None:
            return None
        expected = _render_expected(H, W, bg, base, tr0, tc0, dh, dw, pieces)
        out = main(inp)
    except Exception:
        return None

    if out.shape != inp.shape or not np.array_equal(out, expected):
        return None

    tpl = inp[tr0 : tr0 + dh, tc0 : tc0 + dw]
    if not np.any(tpl == int(bg)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid combo input for 5dbc8537/14")
