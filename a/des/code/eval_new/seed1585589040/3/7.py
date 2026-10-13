from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
FIRST_GAP_OFFSETS = {
    "top": (0, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}
SIDE_ORDER = ["top", "right", "bottom", "left"]
TEMPLATE_COUNT = 4
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(pts)
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _clockwise_border_positions(h: int, w: int) -> list[tuple[str, int, int]]:
    pos: list[tuple[str, int, int]] = []
    for c in range(w):
        pos.append(("top", 0, c))
    for r in range(1, h):
        pos.append(("right", r, w - 1))
    for c in range(w - 2, -1, -1):
        pos.append(("bottom", h - 1, c))
    for r in range(h - 2, 0, -1):
        pos.append(("left", r, 0))
    return pos


def _extract_template_from_patch(patch: np.ndarray, origin: tuple[int, int]) -> dict:
    h, w = patch.shape
    assert h >= 4 and w >= 4
    values = sorted({int(v) for v in np.unique(patch) if int(v) not in (BG, GREY)})
    assert len(values) == 1
    color = values[0]

    border = _clockwise_border_positions(h, w)
    gaps = [(side, r, c) for side, r, c in border if int(patch[r, c]) == BG]
    assert gaps
    for side, r, c in border:
        if int(patch[r, c]) != BG:
            assert int(patch[r, c]) == color

    inner = patch[1:-1, 1:-1].copy()
    assert inner.size > 0
    assert np.any(inner == BG)
    assert np.any(inner == color)

    return {
        "origin": origin,
        "color": color,
        "gap_count": len(gaps),
        "first_side": gaps[0][0],
        "inner": inner,
    }


def _find_separator_col(g: np.ndarray) -> int:
    h, w = g.shape
    candidates = []
    for c in range(w):
        if not np.all(g[:, c] == GREY):
            continue
        has_template_left = c > 0 and np.any((g[:, :c] != BG) & (g[:, :c] != GREY))
        has_grey_right = c + 1 < w and np.any(g[:, c + 1 :] == GREY)
        if has_template_left and has_grey_right:
            candidates.append(c)
    assert len(candidates) == 1
    return candidates[0]


def _find_canvas_bbox(g: np.ndarray, sep_col: int) -> tuple[int, int, int, int]:
    pts = np.argwhere(g[:, sep_col + 1 :] == GREY)
    assert pts.size > 0
    pts[:, 1] += sep_col + 1
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    frame = g[r0 : r1 + 1, c0 : c1 + 1]
    assert frame.shape[0] >= 4 and frame.shape[1] >= 4
    assert np.all(frame[0, :] == GREY)
    assert np.all(frame[-1, :] == GREY)
    assert np.all(frame[:, 0] == GREY)
    assert np.all(frame[:, -1] == GREY)
    return r0, c0, r1, c1


def _extract_templates(g: np.ndarray, sep_col: int) -> list[dict]:
    left = g[:, :sep_col]
    mask = (left != BG) & (left != GREY)
    templates = []
    for pts in _components(mask):
        r0, c0, r1, c1 = _bbox(pts)
        patch = left[r0 : r1 + 1, c0 : c1 + 1]
        templates.append(_extract_template_from_patch(patch, (r0, c0)))
    assert templates
    return templates


def _overlay_with_offset(canvas: np.ndarray, layer: np.ndarray, dr: int, dc: int) -> None:
    h, w = canvas.shape
    src_r0 = max(0, -dr)
    src_c0 = max(0, -dc)
    dst_r0 = max(0, dr)
    dst_c0 = max(0, dc)
    hh = min(h - dst_r0, layer.shape[0] - src_r0)
    ww = min(w - dst_c0, layer.shape[1] - src_c0)
    if hh <= 0 or ww <= 0:
        return

    src = layer[src_r0 : src_r0 + hh, src_c0 : src_c0 + ww]
    dst = canvas[dst_r0 : dst_r0 + hh, dst_c0 : dst_c0 + ww]
    mask = src != BG
    dst[mask] = src[mask]


def _compose_output(templates: list[dict], canvas_frame: np.ndarray, use_offsets: bool = True) -> np.ndarray:
    ordered = sorted(templates, key=lambda t: (int(t["gap_count"]), int(t["origin"][0]), int(t["origin"][1])))
    return _compose_output_in_order(ordered, canvas_frame, use_offsets=use_offsets)


def _compose_output_in_order(ordered: list[dict], canvas_frame: np.ndarray, use_offsets: bool = True) -> np.ndarray:
    out = canvas_frame.copy()
    out[1:-1, 1:-1] = BG
    inner = out[1:-1, 1:-1]

    for template in ordered:
        layer = np.asarray(template["inner"], dtype=int)
        assert layer.shape == inner.shape
        dr, dc = FIRST_GAP_OFFSETS[template["first_side"]] if use_offsets else (0, 0)
        _overlay_with_offset(inner, layer, dr, dc)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay hollow templates onto the right-hand canvas using notch count for layer order and first-notch side for offset.

    Input structure:
    - Black background with one full-height grey separator column splitting the grid into a left template area and a right target area.
    - The left side contains several same-sized monochrome rectangular templates, each with a colored border, interior cut-out holes, and one or more missing border pixels.
    - The right side contains a blank black canvas surrounded by a grey rectangular border whose interior matches a template's interior size.

    Rule:
    - For each template, count the number of missing border pixels and sort templates from fewest to most gaps.
    - Read the first gap encountered when walking the border clockwise from the top-left corner; top means no offset, right means shift right by 1, bottom means shift down by 1, and left means shift left by 1.
    - Remove the border and overlay only the interior pattern onto the right canvas in sorted order, clipping shifted pixels that fall outside the canvas.
    - Non-black template pixels overwrite earlier layers, while black cut-out cells stay transparent and leave lower layers visible.

    Key insight:
    - The border gaps encode two independent signals: total gap count chooses z-order, while the side of the first clockwise gap chooses the template's spatial offset before transparent compositing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_col = _find_separator_col(g)
    canvas_bbox = _find_canvas_bbox(g, sep_col)
    r0, c0, r1, c1 = canvas_bbox
    canvas_frame = g[r0 : r1 + 1, c0 : c1 + 1]

    templates = _extract_templates(g, sep_col)
    inner_shape = canvas_frame.shape[0] - 2, canvas_frame.shape[1] - 2
    assert all(template["inner"].shape == inner_shape for template in templates)

    return _compose_output(templates, canvas_frame, use_offsets=True)


# --------------- generator (codex) ---------------

def _make_inner_mask(h: int, w: int) -> np.ndarray:
    for _ in range(100):
        mask = np.ones((h, w), dtype=bool)
        hole_ops = random.randint(1, 4)
        for _ in range(hole_ops):
            kind = random.choice(["cell", "hline", "vline", "rect"])
            if kind == "cell":
                rr = random.randrange(h)
                cc = random.randrange(w)
                mask[rr, cc] = False
            elif kind == "hline":
                rr = random.randrange(h)
                width = random.randint(1, max(1, w - 1))
                cc0 = random.randint(0, w - width)
                mask[rr, cc0 : cc0 + width] = False
            elif kind == "vline":
                cc = random.randrange(w)
                height = random.randint(1, max(1, h - 1))
                rr0 = random.randint(0, h - height)
                mask[rr0 : rr0 + height, cc] = False
            else:
                hh = random.randint(1, min(3, h))
                ww = random.randint(1, min(3, w))
                rr0 = random.randint(0, h - hh)
                cc0 = random.randint(0, w - ww)
                mask[rr0 : rr0 + hh, cc0 : cc0 + ww] = False

        holes = int((~mask).sum())
        solids = int(mask.sum())
        if 2 <= holes < h * w and solids >= max(6, (h * w) // 2):
            return mask

    mask = np.ones((h, w), dtype=bool)
    mask[1:-1:2, 1:-1:2] = False
    return mask


def _side_positions(h: int, w: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "top": [(0, c) for c in range(1, w - 1)],
        "right": [(r, w - 1) for r in range(1, h - 1)],
        "bottom": [(h - 1, c) for c in range(w - 2, 0, -1)],
        "left": [(r, 0) for r in range(h - 2, 0, -1)],
    }


def _sample_gaps(h: int, w: int, gap_count: int, first_side: str) -> set[tuple[int, int]]:
    side_pos = _side_positions(h, w)
    start_idx = SIDE_ORDER.index(first_side)
    allowed_sides = SIDE_ORDER[start_idx:]
    must = random.choice(side_pos[first_side])
    pool = [pos for side in allowed_sides for pos in side_pos[side] if pos != must]
    assert gap_count - 1 <= len(pool)
    chosen = {must}
    if gap_count > 1:
        chosen.update(random.sample(pool, gap_count - 1))
    return chosen


def _build_template(color: int, inner_h: int, inner_w: int, gap_count: int, first_side: str) -> dict:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    patch = np.full((outer_h, outer_w), color, dtype=int)
    inner_mask = _make_inner_mask(inner_h, inner_w)
    patch[1:-1, 1:-1] = np.where(inner_mask, color, BG)

    for r, c in _sample_gaps(outer_h, outer_w, gap_count, first_side):
        patch[r, c] = BG

    return _extract_template_from_patch(patch, origin=(0, 0)) | {"patch": patch}


def _canvas_frame(outer_h: int, outer_w: int) -> np.ndarray:
    frame = np.full((outer_h, outer_w), BG, dtype=int)
    frame[0, :] = GREY
    frame[-1, :] = GREY
    frame[:, 0] = GREY
    frame[:, -1] = GREY
    return frame


def _shifted_mask(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    tmp = np.where(mask, 1, BG).astype(int)
    canvas = np.zeros_like(tmp, dtype=int)
    _overlay_with_offset(canvas, tmp, dr, dc)
    out[canvas != BG] = True
    return out


def _valid_sample(templates: list[dict], canvas_frame: np.ndarray) -> bool:
    ordered = sorted(templates, key=lambda t: int(t["gap_count"]))
    expected = _compose_output_in_order(ordered, canvas_frame, use_offsets=True)
    reversed_expected = _compose_output_in_order(list(reversed(ordered)), canvas_frame, use_offsets=True)
    no_offset_expected = _compose_output_in_order(ordered, canvas_frame, use_offsets=False)

    inner = expected[1:-1, 1:-1]
    visible_colors = {int(v) for v in inner.ravel() if int(v) != BG}
    if len(visible_colors) < 3:
        return False
    if np.array_equal(expected, reversed_expected):
        return False
    if np.array_equal(expected, no_offset_expected):
        return False

    cover = np.zeros(inner.shape, dtype=int)
    for template in templates:
        dr, dc = FIRST_GAP_OFFSETS[template["first_side"]]
        cover += _shifted_mask(template["inner"] != BG, dr, dc).astype(int)
    if int(cover.max()) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    inner_h = random.randint(4, 6)
    inner_w = random.randint(4, 5)
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    top_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    row_gap = random.randint(1, 3)
    bottom_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)

    height = top_margin + outer_h + row_gap + outer_h + bottom_margin
    width = left_margin + outer_w + 1 + outer_w + 1 + 1 + 1 + outer_w + right_margin
    if height > 30 or width > 30:
        return None

    row0 = top_margin
    row1 = top_margin + outer_h + row_gap
    col0 = left_margin
    col1 = left_margin + outer_w + 1
    sep_col = col1 + outer_w + 1
    canvas_left = sep_col + 2
    canvas_top = random.randint(1, height - outer_h - 1)

    colors = random.sample(COLOR_POOL, TEMPLATE_COUNT)
    sides = random.sample(SIDE_ORDER, TEMPLATE_COUNT)
    templates = [
        _build_template(color, inner_h, inner_w, gap_count=i + 1, first_side=side)
        for i, (color, side) in enumerate(zip(colors, sides, strict=False))
    ]

    canvas_frame = _canvas_frame(outer_h, outer_w)
    if not _valid_sample(templates, canvas_frame):
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[:, sep_col] = GREY

    slots = [(row0, col0), (row0, col1), (row1, col0), (row1, col1)]
    random.shuffle(slots)
    for template, (r0, c0) in zip(templates, slots, strict=False):
        patch = template["patch"]
        grid[r0 : r0 + outer_h, c0 : c0 + outer_w] = patch

    grid[canvas_top : canvas_top + outer_h, canvas_left : canvas_left + outer_w] = canvas_frame

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, _compose_output(templates, canvas_frame, use_offsets=True)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
