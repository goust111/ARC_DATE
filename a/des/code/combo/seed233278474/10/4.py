from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREY = 5
TEMPLATE_COLOR_POOL = (1, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    if isinstance(points, np.ndarray):
        r0, c0 = points.min(axis=0)
        r1, c1 = points.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _extract_templates(grid: np.ndarray) -> list[tuple[int, np.ndarray]]:
    legend_mask = (grid != BG) & (grid != RED) & (grid != GREY)
    points = np.argwhere(legend_mask)
    assert len(points) > 0, "expected colored legend templates"

    r0, c0, r1, c1 = _bbox(points)
    band = grid[r0 : r1 + 1, c0 : c1 + 1]
    band_h, band_w = band.shape
    candidates: list[tuple[int, int, list[tuple[int, np.ndarray]]]] = []

    for gap in range(1, band_w):
        span = band_w - 3 * gap
        if span <= 0 or span % 4 != 0:
            continue
        tile_w = span // 4
        if tile_w < 2:
            continue

        pos = 0
        ok = True
        templates: list[tuple[int, np.ndarray]] = []
        for i in range(4):
            sub = band[:, pos : pos + tile_w]
            if sub.shape != (band_h, tile_w):
                ok = False
                break

            colors = [int(v) for v in np.unique(sub) if int(v) != BG]
            if len(colors) != 1:
                ok = False
                break

            color = colors[0]
            mask = sub == color
            if band_h > 2 and tile_w > 2 and np.any(mask[1:-1, 1:-1]):
                ok = False
                break
            if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
                ok = False
                break

            templates.append((color, mask.copy()))
            pos += tile_w
            if i < 3:
                gap_slice = band[:, pos : pos + gap]
                if gap_slice.shape[1] != gap or np.any(gap_slice != BG):
                    ok = False
                    break
                pos += gap

        if ok and pos == band_w:
            candidates.append((tile_w, gap, templates))

    assert candidates, "could not segment the legend into four template slots"
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return candidates[0][2]


def _find_grey_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rectangles: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "grey regions must be closed rectangles"
        local = np.zeros((height, width), dtype=bool)
        for r, c in comp:
            local[r - r0, c - c0] = True
        assert np.array_equal(local, _border_mask(height, width)), "grey component must be a 1-pixel rectangle border"
        rectangles.append((r0, c0, r1, c1))
    rectangles.sort()
    assert rectangles, "expected at least one grey rectangle"
    return rectangles


def _paint_tiled_border(view: np.ndarray, color: int, template_mask: np.ndarray) -> None:
    height, width = view.shape
    tile_h, tile_w = template_mask.shape
    top = template_mask[0, :]
    bottom = template_mask[-1, :]
    left = template_mask[:, 0]
    right = template_mask[:, -1]

    view[0, :] = BG
    view[-1, :] = BG
    view[:, 0] = BG
    view[:, -1] = BG

    for offset in range(width):
        if top[offset % tile_w]:
            view[0, offset] = color
        if bottom[offset % tile_w]:
            view[-1, offset] = color
    for offset in range(height):
        if left[offset % tile_h]:
            view[offset, 0] = color
        if right[offset % tile_h]:
            view[offset, -1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each grey rectangle border by the legend template indexed by the number of red markers inside it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    templates = _extract_templates(g)
    out = g.copy()

    for r0, c0, r1, c1 in _find_grey_rectangles(g):
        view = out[r0 : r1 + 1, c0 : c1 + 1]
        interior = view[1:-1, 1:-1]
        red_count = int(np.sum(g[r0 + 1 : r1, c0 + 1 : c1] == RED))
        assert 1 <= red_count <= 4, "each rectangle must contain between one and four red markers"
        interior[interior == RED] = BG
        color, template_mask = templates[red_count - 1]
        _paint_tiled_border(view, color, template_mask)

    return out


def _style_sequences(size: int) -> list[np.ndarray]:
    center = size // 2
    full = np.ones(size, dtype=bool)

    notch = np.ones(size, dtype=bool)
    notch[center] = False

    corners = np.zeros(size, dtype=bool)
    corners[0] = True
    corners[-1] = True

    mids = np.zeros(size, dtype=bool)
    mids[center] = True

    return [full, notch, corners, mids]


def _template_masks(size: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for seq in _style_sequences(size):
        mask = np.zeros((size, size), dtype=bool)
        mask[0, :] = seq
        mask[-1, :] = seq
        mask[:, 0] = seq
        mask[:, -1] = seq
        masks.append(mask)
    return masks


def _reserve(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, margin: int = 1) -> None:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(blocked.shape[0] - 1, r1 + margin)
    cc1 = min(blocked.shape[1] - 1, c1 + margin)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _can_place(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, margin: int = 1) -> bool:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(blocked.shape[0] - 1, r1 + margin)
    cc1 = min(blocked.shape[1] - 1, c1 + margin)
    return not bool(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _branch_counts(n_rects: int) -> list[int]:
    counts: list[int] = []
    while len(counts) < n_rects:
        batch = [1, 2, 3, 4]
        random.shuffle(batch)
        counts.extend(batch)
    counts = counts[:n_rects]
    random.shuffle(counts)
    return counts


def _render_legend(
    grid: np.ndarray,
    top: int,
    left: int,
    gap: int,
    masks: list[np.ndarray],
    colors: list[int],
) -> None:
    size = masks[0].shape[0]
    for i, (mask, color) in enumerate(zip(masks, colors)):
        c0 = left + i * (size + gap)
        tile = grid[top : top + size, c0 : c0 + size]
        tile[mask] = color


def _place_rectangle(
    grid: np.ndarray,
    blocked: np.ndarray,
    top_min: int,
    count: int,
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    for _ in range(250):
        rect_h = random.randint(5, min(10, h - top_min - 1))
        rect_w = random.randint(5, min(13, w - 2))
        if (rect_h - 2) * (rect_w - 2) < count:
            continue

        if top_min > h - rect_h - 2 or w - rect_w - 2 < 1:
            return None

        r0 = random.randint(top_min, h - rect_h - 2)
        c0 = random.randint(1, w - rect_w - 2)
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1
        if not _can_place(blocked, r0, c0, r1, c1, margin=1):
            continue

        border = _border_mask(rect_h, rect_w)
        tile = grid[r0 : r1 + 1, c0 : c1 + 1]
        tile[border] = GREY
        interior_cells = [(rr, cc) for rr in range(r0 + 1, r1) for cc in range(c0 + 1, c1)]
        for rr, cc in random.sample(interior_cells, count):
            grid[rr, cc] = RED
        _reserve(blocked, r0, c0, r1, c1, margin=1)
        return r0, c0, r1, c1
    return None


def _expected_output(
    inp: np.ndarray,
    rectangles: list[tuple[int, int, int, int, int]],
    template_masks: list[np.ndarray],
    template_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    for r0, c0, r1, c1, count in rectangles:
        view = out[r0 : r1 + 1, c0 : c1 + 1]
        interior = view[1:-1, 1:-1]
        interior[interior == RED] = BG
        _paint_tiled_border(view, template_colors[count - 1], template_masks[count - 1])
    return out


def _generate_case() -> np.ndarray | None:
    template_size = random.choice((5, 7))
    gap = random.randint(1, 3)
    legend_w = 4 * template_size + 3 * gap

    n_rects = random.randint(4, 6)
    height = random.randint(max(22, template_size + 10), 34)
    width = random.randint(max(legend_w + 2, 24), 40)

    legend_top = random.randint(0, 1)
    legend_left = random.randint(1, width - legend_w - 1)
    below_legend = legend_top + template_size + 2

    inp = np.full((height, width), BG, dtype=int)
    blocked = np.zeros_like(inp, dtype=bool)

    template_colors = random.sample(TEMPLATE_COLOR_POOL, 4)
    template_masks = _template_masks(template_size)
    _render_legend(inp, legend_top, legend_left, gap, template_masks, template_colors)
    _reserve(blocked, legend_top, legend_left, legend_top + template_size - 1, legend_left + legend_w - 1, margin=1)

    rectangles: list[tuple[int, int, int, int, int]] = []
    for count in _branch_counts(n_rects):
        placed = _place_rectangle(inp, blocked, below_legend, count)
        if placed is None:
            return None
        rectangles.append((*placed, count))

    expected = _expected_output(inp, rectangles, template_masks, template_colors)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
