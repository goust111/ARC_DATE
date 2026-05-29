# Combo variant of bf45cf4b (index 27)
# combo_spec: D12/2+D2/3
# Name: 空间位置双重消歧

from __future__ import annotations

"""
Design analysis for the fusion variant.

a) Concrete behavior shift
Original:
- Find two foreground components.
- Use color-count / square-ness to decide which one is the tile.
- Assume the other component is monochrome; only that one color is treated as "filled".

Fusion:
- Find the same two foreground components, but roles come only from spatial order.
- If they are stacked, the upper object is the control mask and the lower object is the tile.
- If they are side-by-side, the left object is the control mask and the right object is the tile.
- Every non-background cell inside the control bbox is treated as filled, so the control may be multicolor.

Mini example:
- Top object (control) =
  2 0 3
  2 3 0
  0 3 3
- Bottom object (tile) =
  4 5
  5 4
Output =
  4 5 0 0 4 5
  5 4 0 0 5 4
  4 5 5 4 0 0
  5 4 4 5 0 0
  0 0 4 5 4 5
  0 0 5 4 5 4
Difference from the original: control colors no longer identify the mask; only non-bg occupancy does.

b) Complexity calibration
- Keep exactly two components, so the task is still compact.
- Restrict the spatial convention to two easy cases only: vertical stack or horizontal row.
- Give both components the same number of foreground colors, and often the same square/non-square status,
  so color statistics and shape preference stop being reliable.
- Keep repeated tile stamping unchanged, preserving the core visual identity of bf45cf4b.

c) Hidden-assumption audit of the old solver
- `len(cols)` as role cue: invalid now, because both objects may have the same color count.
- `square` as tie-break: invalid now, because both may share the same square/non-square status.
- `ctrl_cols[0]` as the only filled color: invalid now, because the control is intentionally multicolor.
- "multicolor object must be tile": invalid now; both may be multicolor.

d) Placement legality
- Generator places exactly two disconnected 8-connected components.
- In vertical layouts, their column ranges overlap and their row ranges are separated by a bg gap.
- In horizontal layouts, their row ranges overlap and their column ranges are separated by a bg gap.
- This makes the intended axis of comparison visually obvious and keeps both objects in bounds.

e) Visual observability
- Across samples the stable cue is always positional order: upper/left = control, lower/right = tile.
- The output still visibly looks like the upper/left footprint expanded into copies of the other patch.
- No invisible ordering such as numeric color id is used.

f) Visual neatness
- Only two objects appear in the input.
- They are separated by a clear gap and aligned enough on the orthogonal axis to avoid diagonal ambiguity.
- Tile patterns stay compact and structured; control masks stay connected and holey.

g) Change inferability
- Colors are not remapped by arbitrary lookup tables.
- The only color effect is that whatever patch is the spatially-later object gets copied repeatedly.
- Generator validation rejects degenerate cases where the old color-based solver would still succeed.

i) New assumptions introduced by the new solver
- Exactly two multicolor foreground components exist.
- Their relation is either clearly vertical or clearly horizontal.
- The generator guarantees those assumptions before returning a sample.
"""

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, paste


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return int(b0 - a1 - 1)
    if b1 < a0:
        return int(a0 - b1 - 1)
    return 0


def _extract_infos(grid: np.ndarray, bg: int) -> list[dict]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 2
    infos = []
    for comp in comps:
        patch = crop_bbox(grid, comp["bbox"])
        infos.append({"bbox": comp["bbox"], "patch": patch})
    return infos


def _assign_roles_by_position(infos: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    a, b = infos
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]

    row_gap = _interval_gap(ar0, ar1, br0, br1)
    col_gap = _interval_gap(ac0, ac1, bc0, bc1)
    rows_overlap = _intervals_overlap(ar0, ar1, br0, br1)
    cols_overlap = _intervals_overlap(ac0, ac1, bc0, bc1)

    if row_gap >= 1 and cols_overlap:
        ctrl, tile = (a, b) if ar0 < br0 else (b, a)
        return ctrl["patch"], tile["patch"]
    if col_gap >= 1 and rows_overlap:
        ctrl, tile = (a, b) if ac0 < bc0 else (b, a)
        return ctrl["patch"], tile["patch"]

    # Fallback for unusual but still readable placements: choose the axis with larger separation.
    a_rc = ((ar0 + ar1) / 2.0, (ac0 + ac1) / 2.0)
    b_rc = ((br0 + br1) / 2.0, (bc0 + bc1) / 2.0)
    if abs(a_rc[0] - b_rc[0]) >= abs(a_rc[1] - b_rc[1]):
        ctrl, tile = (a, b) if a_rc[0] < b_rc[0] else (b, a)
    else:
        ctrl, tile = (a, b) if a_rc[1] < b_rc[1] else (b, a)
    return ctrl["patch"], tile["patch"]


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    mask = (ctrl != int(bg))
    for r in range(ch):
        for c in range(cw):
            if bool(mask[r, c]):
                paste(out, tile, r * th, c * tw)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    infos = _extract_infos(g, bg)
    ctrl, tile = _assign_roles_by_position(infos)
    return _compose_output(tile, ctrl, bg)


def _neighbors8(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _pick_dims(
    rng: np.random.Generator,
    *,
    square: bool,
    min_size: int,
    max_size: int,
) -> tuple[int, int]:
    if square:
        s = int(rng.integers(min_size, max_size + 1))
        return s, s
    while True:
        h = int(rng.integers(min_size, max_size + 1))
        w = int(rng.integers(min_size, max_size + 1))
        if h != w:
            return h, w


def _make_ctrl_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    area = h * w
    min_fill = max(h + w - 1, int(np.ceil(0.45 * area)))
    max_fill = min(area - 1, int(np.floor(0.72 * area)))
    target = int(rng.integers(min_fill, max_fill + 1))

    ar = int(rng.integers(0, h))
    ac = int(rng.integers(0, w))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, ac] = True
    mask[ar, :] = True

    while int(mask.sum()) < target:
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                if any(mask[rr, cc] for rr, cc in _neighbors8(h, w, r, c)):
                    frontier.append((r, c))
        pick = frontier[int(rng.integers(len(frontier)))]
        mask[pick] = True
    return mask


def _paint_mask(mask: np.ndarray, bg: int, palette: list[int], shift: int) -> np.ndarray:
    out = np.full(mask.shape, int(bg), dtype=int)
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    coords.sort(key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]))

    for i, (r, c) in enumerate(coords):
        out[r, c] = int(palette[(r + 2 * c + shift + i) % len(palette)])

    for i, color in enumerate(palette):
        r, c = coords[i]
        out[r, c] = int(color)
    return out


def _make_tile(h: int, w: int, bg: int, palette: list[int], rng: np.random.Generator) -> np.ndarray:
    tile_mask = np.ones((h, w), dtype=bool)
    if h >= 3 and w >= 3 and rng.random() < 0.45:
        rr = int(rng.integers(1, h - 1))
        cc = int(rng.integers(1, w - 1))
        tile_mask[rr, cc] = False
    return _paint_mask(tile_mask, bg, palette, int(rng.integers(0, len(palette))))


def _make_ctrl(h: int, w: int, bg: int, palette: list[int], rng: np.random.Generator) -> np.ndarray:
    mask = _make_ctrl_mask(h, w, rng)
    return _paint_mask(mask, bg, palette, int(rng.integers(0, len(palette))))


def _place_covering_index(span: int, size: int, anchor: int, rng: np.random.Generator) -> int:
    lo = max(0, int(anchor) - int(size) + 1)
    hi = min(int(anchor), int(span) - int(size))
    return int(rng.integers(lo, hi + 1))


def _make_input(ctrl: np.ndarray, tile: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    ch, cw = map(int, ctrl.shape)
    th, tw = map(int, tile.shape)
    horizontal = bool(rng.integers(0, 2))
    gap = int(rng.integers(2, 5))

    if horizontal:
        H = int(rng.integers(max(12, max(ch, th) + 1), 19))
        W = int(rng.integers(max(12, cw + tw + gap), 19))
        shared_row = int(rng.integers(0, H))
        r_ctrl = _place_covering_index(H, ch, shared_row, rng)
        r_tile = _place_covering_index(H, th, shared_row, rng)
        slack = W - (cw + tw + gap)
        c_ctrl = int(rng.integers(0, slack + 1))
        c_tile = int(rng.integers(c_ctrl + cw + gap, W - tw + 1))
    else:
        H = int(rng.integers(max(12, ch + th + gap), 19))
        W = int(rng.integers(max(12, max(cw, tw) + 1), 19))
        shared_col = int(rng.integers(0, W))
        c_ctrl = _place_covering_index(W, cw, shared_col, rng)
        c_tile = _place_covering_index(W, tw, shared_col, rng)
        slack = H - (ch + th + gap)
        r_ctrl = int(rng.integers(0, slack + 1))
        r_tile = int(rng.integers(r_ctrl + ch + gap, H - th + 1))

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, ctrl, r_ctrl, c_ctrl, transparent=bg)
    paste(grid, tile, r_tile, c_tile, transparent=bg)
    return grid


def _old_main_reference(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    infos = []
    for comp in _extract_infos(g, bg):
        patch = comp["patch"]
        cols = sorted(set(int(x) for x in patch.flatten()) - {bg})
        infos.append(
            {
                "patch": patch,
                "cols": cols,
                "square": int(patch.shape[0]) == int(patch.shape[1]),
            }
        )

    infos.sort(
        key=lambda d: (
            len(d["cols"]) >= 2,
            len(d["cols"]),
            int(d["square"]),
        ),
        reverse=True,
    )
    tile = infos[0]["patch"]
    ctrl = infos[1]["patch"]
    ctrl_cols = sorted(set(int(x) for x in ctrl.flatten()) - {bg})
    if not ctrl_cols:
        return np.full((0, 0), int(bg), dtype=int)
    ctrl_col = int(ctrl_cols[0])

    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) == ctrl_col:
                paste(out, tile, r * th, c * tw)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    bg = int(rng.integers(0, 10))
    palette = [int(x) for x in rng.choice([c for c in range(10) if c != bg], size=int(rng.integers(2, 4)), replace=False)]

    square_mode = bool(rng.integers(0, 2))
    ctrl_h, ctrl_w = _pick_dims(rng, square=square_mode, min_size=3, max_size=5)
    tile_h, tile_w = _pick_dims(rng, square=square_mode, min_size=2, max_size=4)

    ctrl = _make_ctrl(ctrl_h, ctrl_w, bg, palette, rng)
    tile = _make_tile(tile_h, tile_w, bg, palette, rng)
    expected = _compose_output(tile, ctrl, bg)
    inp = _make_input(ctrl, tile, bg, rng)

    comps = _extract_infos(inp, bg)
    if len(comps) != 2:
        return None

    got = main(inp)
    if not np.array_equal(got, expected):
        return None

    old = _old_main_reference(inp)
    if np.array_equal(old, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
