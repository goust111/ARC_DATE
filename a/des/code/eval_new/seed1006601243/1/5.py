from __future__ import annotations

import numpy as np


_RNG = np.random.default_rng()

BG = 0
CANVAS = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
STAMP_COUNT_CHOICES = (3, 4)
CANVAS_H_RANGE = (10, 17)
CANVAS_W_RANGE = (10, 17)
MARGIN_RANGE = (4, 7)
BAR_MARGIN_CHOICES = (4, 5)
MAX_RETRIES = 500
MIN_VISIBLE_CELLS = 2


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(pts)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool).copy()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            stack = [(int(r), int(c))]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _embed_mask(mask: np.ndarray, shape: tuple[int, int], top: int, left: int) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    h, w = mask.shape
    out[top : top + h, left : left + w] = mask
    return out


def _canvas_info(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    vals = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(vals))
    counts = {int(color): int(np.sum(g == color)) for color in np.unique(g) if int(color) != bg}
    assert counts, "expected non-background content"
    canvas_color = max(counts, key=counts.get)
    canvas_pts = np.argwhere(g == canvas_color)
    assert canvas_pts.size, "canvas color must be present"
    bbox = _bbox(canvas_pts)
    r0, c0, r1, c1 = bbox
    crop = np.asarray(g[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()
    assert crop.shape[0] >= 4 and crop.shape[1] >= 4
    return canvas_color, bbox, crop


def _find_order_bar(
    grid: np.ndarray,
    bg: int,
    canvas_color: int,
    canvas_bbox: tuple[int, int, int, int],
) -> tuple[list[int], str]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    r0, c0, r1, c1 = canvas_bbox
    candidates: list[tuple[int, list[int], str]] = []

    for c in range(c1 + 2, w):
        rows = np.flatnonzero((g[:, c] != bg) & (g[:, c] != canvas_color))
        if rows.size < 2:
            continue
        rr0, rr1 = int(rows[0]), int(rows[-1])
        if rr1 - rr0 + 1 != int(rows.size):
            continue
        if rr0 > 0 and int(g[rr0 - 1, c]) != bg:
            continue
        if rr1 + 1 < h and int(g[rr1 + 1, c]) != bg:
            continue
        if c > 0 and np.any(g[rr0 : rr1 + 1, c - 1] != bg):
            continue
        if c + 1 < w and np.any(g[rr0 : rr1 + 1, c + 1] != bg):
            continue
        colors = [int(g[r, c]) for r in range(rr0, rr1 + 1)]
        if len(set(colors)) != len(colors):
            continue
        candidates.append((len(colors), colors, "right"))

    for r in range(r1 + 2, h):
        cols = np.flatnonzero((g[r, :] != bg) & (g[r, :] != canvas_color))
        if cols.size < 2:
            continue
        cc0, cc1 = int(cols[0]), int(cols[-1])
        if cc1 - cc0 + 1 != int(cols.size):
            continue
        if cc0 > 0 and int(g[r, cc0 - 1]) != bg:
            continue
        if cc1 + 1 < w and int(g[r, cc1 + 1]) != bg:
            continue
        if r > 0 and np.any(g[r - 1, cc0 : cc1 + 1] != bg):
            continue
        if r + 1 < h and np.any(g[r + 1, cc0 : cc1 + 1] != bg):
            continue
        colors = [int(g[r, c]) for c in range(cc0, cc1 + 1)]
        if len(set(colors)) != len(colors):
            continue
        candidates.append((len(colors), colors, "bottom"))

    assert candidates, "failed to find the order bar"
    candidates.sort(key=lambda item: -item[0])
    best = candidates[0]
    return list(best[1]), best[2]


def _outside_mask(shape: tuple[int, int], canvas_bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    r0, c0, r1, c1 = canvas_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    return mask


def _template_for_color(
    grid: np.ndarray,
    color: int,
    outside: np.ndarray,
) -> np.ndarray:
    comps = _components(outside & (grid == color))
    assert comps, f"missing outside pixels for color {color}"
    comp = max(comps, key=lambda pts: int(pts.shape[0]))
    local = np.zeros(grid.shape, dtype=bool)
    local[tuple(comp.T)] = True
    return _crop_mask(local)


def _recover_full_masks(
    canvas: np.ndarray,
    order: list[int],
    templates: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    h, w = canvas.shape
    recovered: dict[int, np.ndarray] = {}
    later_occupied = np.zeros((h, w), dtype=bool)

    for color in reversed(order):
        template = np.asarray(templates[color], dtype=bool)
        th, tw = template.shape
        visible = canvas == int(color)
        assert int(np.sum(visible)) >= 1, "each stamp color must stay visible on the canvas"
        candidates: list[np.ndarray] = []
        for top in range(h - th + 1):
            for left in range(w - tw + 1):
                shifted = _embed_mask(template, (h, w), top, left)
                if np.array_equal(shifted & ~later_occupied, visible):
                    candidates.append(shifted)
        assert len(candidates) == 1, f"stamp placement must be unique for color {color}"
        recovered[color] = candidates[0]
        later_occupied |= candidates[0]

    return recovered


def _compose_canvas(
    shape: tuple[int, int],
    canvas_color: int,
    order: list[int],
    full_masks: dict[int, np.ndarray],
) -> np.ndarray:
    out = np.full(shape, int(canvas_color), dtype=int)
    for color in order:
        out[full_masks[color]] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the full stamp layout on the gray canvas using outside templates and the order bar.

    Input structure:
    - Black background with one large gray rectangular canvas somewhere inside the grid.
    - Several single-color stamp traces lie on the canvas, and each stamp color also appears once outside the canvas as a complete template on the black background.
    - A one-cell-thick color bar sits either to the right of the canvas as a vertical strip or below it as a horizontal strip.

    Rule:
    - Detect the gray canvas and crop its bounding rectangle.
    - Read the order bar geometrically: top-to-bottom for a right-side bar, left-to-right for a bottom bar; this gives the stamp order from earliest to latest.
    - For each bar color, extract the matching complete template from outside the canvas.
    - Reconstruct stamp placements from latest to earliest: every template cell not already covered by a later stamp must match the currently visible cells of that color exactly.
    - After every full stamp position is recovered, repaint the cropped canvas in chronological order and return only that restored canvas area.

    Key insight:
    The outside same-colored templates reveal each stamp's full mask, but only the order bar tells you which missing pixels are hidden by later stamps rather than impossible, so the puzzle is solved by reversing the layer stack.
    """
    g = np.asarray(grid, dtype=int)
    canvas_color, canvas_bbox, canvas_crop = _canvas_info(g)
    bg = BG if int(np.any(g == BG)) else 0
    outside = _outside_mask(g.shape, canvas_bbox)
    order, _orientation = _find_order_bar(g, bg, canvas_color, canvas_bbox)
    templates = {color: _template_for_color(g, color, outside) for color in order}
    full_masks = _recover_full_masks(canvas_crop, order, templates)
    return _compose_canvas(canvas_crop.shape, canvas_color, order, full_masks)


_BASE_SHAPES = [
    np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 1, 1, 1],
        ],
        dtype=bool,
    ),
]


def _randint(low: int, high: int) -> int:
    return int(_RNG.integers(low, high))


def _choice(seq):
    return seq[int(_RNG.integers(0, len(seq)))]


def _rotated_shape(mask: np.ndarray) -> np.ndarray:
    return _crop_mask(np.rot90(mask, _randint(0, 4)))


def _sample_shapes(count: int) -> list[np.ndarray]:
    idxs = list(_RNG.choice(len(_BASE_SHAPES), size=count, replace=False))
    return [_rotated_shape(_BASE_SHAPES[int(idx)]) for idx in idxs]


def _sample_geometry(canvas_h: int, canvas_w: int, orientation: str) -> tuple[int, int, int, int]:
    top = _randint(*MARGIN_RANGE)
    left = _randint(*MARGIN_RANGE)
    if orientation == "right":
        right = _choice(BAR_MARGIN_CHOICES)
        bottom = _randint(*MARGIN_RANGE)
    else:
        right = _randint(*MARGIN_RANGE)
        bottom = _choice(BAR_MARGIN_CHOICES)
    h = top + canvas_h + bottom
    w = left + canvas_w + right
    assert h <= 30 and w <= 30
    return h, w, top, left


def _place_stamps_on_canvas(
    canvas_shape: tuple[int, int],
    order: list[int],
    templates: dict[int, np.ndarray],
) -> dict[int, np.ndarray] | None:
    h, w = canvas_shape
    full_masks: dict[int, np.ndarray] = {}
    later_occupied = np.zeros((h, w), dtype=bool)
    hidden_flags = [_RNG.random() < 0.7 for _ in range(len(order) - 1)]
    if hidden_flags and not any(hidden_flags):
        hidden_flags[_randint(0, len(hidden_flags))] = True

    latest_color = order[-1]
    latest_template = templates[latest_color]
    lt_h, lt_w = latest_template.shape
    latest_top = _randint(0, h - lt_h + 1)
    latest_left = _randint(0, w - lt_w + 1)
    latest_mask = _embed_mask(latest_template, canvas_shape, latest_top, latest_left)
    full_masks[latest_color] = latest_mask
    later_occupied |= latest_mask

    earlier_colors = list(reversed(order[:-1]))
    for idx, color in enumerate(earlier_colors):
        template = templates[color]
        th, tw = template.shape
        size = int(np.sum(template))
        candidates: list[tuple[int, int, np.ndarray]] = []
        for top in range(h - th + 1):
            for left in range(w - tw + 1):
                shifted = _embed_mask(template, canvas_shape, top, left)
                overlap = int(np.sum(shifted & later_occupied))
                visible = size - overlap
                if visible < MIN_VISIBLE_CELLS:
                    continue
                if overlap > size - MIN_VISIBLE_CELLS:
                    continue
                if hidden_flags[idx] and overlap == 0:
                    continue
                candidates.append((overlap, visible, shifted))
        if not candidates:
            return None
        if hidden_flags[idx]:
            overlapful = [item for item in candidates if item[0] > 0]
            if not overlapful:
                return None
            candidates = overlapful
        overlap_values = np.array([item[0] for item in candidates], dtype=int)
        if hidden_flags[idx]:
            target = int(np.clip(np.round(np.mean(overlap_values)), 1, overlap_values.max()))
            close = [item for item in candidates if abs(item[0] - target) <= 1]
            if close:
                candidates = close
        chosen = candidates[_randint(0, len(candidates))]
        full_masks[color] = chosen[2]
        later_occupied |= chosen[2]
    return full_masks


def _paint_input_canvas(
    canvas_shape: tuple[int, int],
    order: list[int],
    full_masks: dict[int, np.ndarray],
) -> np.ndarray:
    return _compose_canvas(canvas_shape, CANVAS, order, full_masks)


def _reserve_with_padding(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r0 + h + 1)
    cc1 = min(occupied.shape[1], c0 + w + 1)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _place_templates(
    grid: np.ndarray,
    occupied: np.ndarray,
    templates: dict[int, np.ndarray],
    orientation: str,
    bar_anchor: tuple[int, int],
) -> bool:
    h, w = grid.shape
    bar_r, bar_c = bar_anchor
    for color, template in templates.items():
        th, tw = template.shape
        placed = False
        for _ in range(300):
            top = _randint(0, h - th + 1)
            left = _randint(0, w - tw + 1)
            if orientation == "right" and left + tw >= bar_c:
                continue
            if orientation == "bottom" and top + th >= bar_r:
                continue
            if not _reserve_with_padding(occupied, top, left, th, tw):
                continue
            target = grid[top : top + th, left : left + tw]
            if np.any(target[template] != BG):
                continue
            target[template] = int(color)
            occupied[max(0, top - 1) : min(h, top + th + 1), max(0, left - 1) : min(w, left + tw + 1)] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _build_grid(
    canvas_h: int,
    canvas_w: int,
    order: list[int],
    templates: dict[int, np.ndarray],
    full_masks: dict[int, np.ndarray],
    orientation: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    grid_h, grid_w, canvas_top, canvas_left = _sample_geometry(canvas_h, canvas_w, orientation)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    canvas_bbox = (canvas_top, canvas_left, canvas_top + canvas_h - 1, canvas_left + canvas_w - 1)
    r0, c0, r1, c1 = canvas_bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = CANVAS
    canvas_in = _paint_input_canvas((canvas_h, canvas_w), order, full_masks)
    grid[r0 : r1 + 1, c0 : c1 + 1] = canvas_in

    if orientation == "right":
        bar_col = c1 + 2
        start_row = _randint(r0, r1 - len(order) + 2)
        for i, color in enumerate(order):
            grid[start_row + i, bar_col] = int(color)
        bar_anchor = (start_row, bar_col)
    else:
        bar_row = r1 + 2
        start_col = _randint(c0, c1 - len(order) + 2)
        for i, color in enumerate(order):
            grid[bar_row, start_col + i] = int(color)
        bar_anchor = (bar_row, start_col)

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True
    if orientation == "right":
        occupied[max(0, bar_anchor[0] - 1) : min(grid_h, bar_anchor[0] + len(order) + 1), max(0, bar_anchor[1] - 1) : min(grid_w, bar_anchor[1] + 2)] = True
    else:
        occupied[max(0, bar_anchor[0] - 1) : min(grid_h, bar_anchor[0] + 2), max(0, bar_anchor[1] - 1) : min(grid_w, bar_anchor[1] + len(order) + 1)] = True

    if not _place_templates(grid, occupied, templates, orientation, bar_anchor):
        return None

    out = _compose_canvas((canvas_h, canvas_w), CANVAS, order, full_masks)
    return grid, out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    canvas_h = _randint(*CANVAS_H_RANGE)
    canvas_w = _randint(*CANVAS_W_RANGE)
    stamp_count = _choice(STAMP_COUNT_CHOICES)
    orientation = _choice(("right", "bottom"))
    colors = list(_RNG.choice(PALETTE, size=stamp_count, replace=False))
    _RNG.shuffle(colors)
    order = [int(c) for c in colors]
    shape_list = _sample_shapes(stamp_count)
    templates = {color: shape for color, shape in zip(order, shape_list)}

    full_masks = _place_stamps_on_canvas((canvas_h, canvas_w), order, templates)
    if full_masks is None:
        return None

    canvas = _compose_canvas((canvas_h, canvas_w), CANVAS, order, full_masks)
    hidden_exists = False
    later_union = np.zeros((canvas_h, canvas_w), dtype=bool)
    for color in reversed(order):
        visible = int(np.sum(full_masks[color] & ~later_union))
        total = int(np.sum(full_masks[color]))
        if visible < total:
            hidden_exists = True
        if visible < MIN_VISIBLE_CELLS:
            return None
        later_union |= full_masks[color]
    if not hidden_exists:
        return None

    built = _build_grid(canvas_h, canvas_w, order, templates, full_masks, orientation)
    if built is None:
        return None
    inp, out = built
    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    if not np.array_equal(solved, canvas):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        built = _gen_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid stamp-overlap puzzle")
