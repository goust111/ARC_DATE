from __future__ import annotations

import random

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _cells_of_color(grid: np.ndarray, color: int) -> set[tuple[int, int]]:
    rows, cols = np.where(grid == int(color))
    return {(int(r), int(c)) for r, c in zip(rows.tolist(), cols.tolist(), strict=False)}


def _crop_mask(cells: set[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int]]:
    assert cells
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, c0 = min(rows), min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0)


def _mask_rel_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = np.where(mask)
    return [(int(r), int(c)) for r, c in zip(rows.tolist(), cols.tolist(), strict=False)]


def _placed_cells(rel_cells: list[tuple[int, int]], top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in rel_cells}


def _recover_layers(
    overlay: np.ndarray,
    colors_top_to_bottom: list[int],
    bg: int,
) -> tuple[np.ndarray, list[set[tuple[int, int]]]]:
    assert colors_top_to_bottom
    overlay_h, overlay_w = map(int, overlay.shape)

    top_cells = _cells_of_color(overlay, colors_top_to_bottom[0])
    assert top_cells, "top stamp must be visible"
    shape_mask, _top_origin = _crop_mask(top_cells)
    rel_cells = _mask_rel_cells(shape_mask)
    shape_h, shape_w = map(int, shape_mask.shape)
    assert shape_h <= overlay_h and shape_w <= overlay_w

    layers = [top_cells]
    occupied = set(top_cells)

    for color in colors_top_to_bottom[1:]:
        visible_cells = _cells_of_color(overlay, color)
        assert visible_cells, "every deeper stamp must have at least one visible pixel"

        candidates: list[set[tuple[int, int]]] = []
        for top in range(overlay_h - shape_h + 1):
            for left in range(overlay_w - shape_w + 1):
                cells = _placed_cells(rel_cells, top, left)
                if cells - occupied == visible_cells:
                    candidates.append(cells)

        assert len(candidates) == 1, "stamp placement must be uniquely recoverable"
        layer_cells = candidates[0]
        layers.append(layer_cells)
        occupied |= layer_cells

    overlay_colors = {int(v) for v in np.unique(overlay) if int(v) != bg}
    assert overlay_colors == set(colors_top_to_bottom), "overlay colors must match the legend"
    return shape_mask, layers


def _render_display(
    shape_mask: np.ndarray,
    colors_top_to_bottom: list[int],
    out_h: int,
    out_w: int,
    bg: int,
) -> np.ndarray:
    mask_h, mask_w = map(int, shape_mask.shape)
    needed_w = len(colors_top_to_bottom) * mask_w + (len(colors_top_to_bottom) - 1)
    assert needed_w <= out_w
    assert mask_h <= out_h

    out = np.full((out_h, out_w), int(bg), dtype=int)
    top = (out_h - mask_h) // 2
    for i, color in enumerate(colors_top_to_bottom):
        left = i * (mask_w + 1)
        patch = out[top : top + mask_h, left : left + mask_w]
        patch[shape_mask] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover identical overlapping stamps from a stacked area and unfold them into the blank display area.

    Input structure:
    - The left side contains several same-shape stamps of different colors placed at different offsets.
    - Overlaps show only the topmost color.
    - The rightmost non-background column is a vertical color legend whose non-background cells list
      stamp colors from top layer to bottom layer.
    - All columns to the right of the legend are the blank display area.

    Rule:
    - Read the layer order from the legend.
    - Use the fully visible top stamp as the shared shape template.
    - For each deeper color, find the unique translation of that template whose cells minus the already
      occupied upper layers exactly equal the currently visible pixels of that color.
    - Return only the display area, with the recovered full stamps arranged left-to-right and one
      background column between adjacent stamps.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    nonbg_cols = np.where(np.any(g != bg, axis=0))[0]
    assert nonbg_cols.size > 0, "input must contain non-background pixels"
    legend_col = int(nonbg_cols.max())
    assert legend_col + 1 < g.shape[1], "display area must exist to the right of the legend"

    legend = [int(g[r, legend_col]) for r in range(g.shape[0]) if int(g[r, legend_col]) != bg]
    assert 2 <= len(legend) <= 4
    assert len(set(legend)) == len(legend)

    overlay = g[:, :legend_col]
    display_w = g.shape[1] - legend_col - 1
    shape_mask, _layers = _recover_layers(overlay, legend, bg)
    return _render_display(shape_mask, legend, g.shape[0], display_w, bg)


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    seen = set()
    stack = [next(iter(cells))]
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                stack.append(nxt)
    return len(seen) == len(cells)


def _random_shape_mask() -> np.ndarray | None:
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    min_area = max(h + w - 1, (h * w + 1) // 2)
    max_area = h * w - 1
    if min_area > max_area:
        return None

    target_area = random.randint(min_area, max_area)
    for _ in range(120):
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                        frontier.add((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(list(frontier)))

        if len(cells) != target_area:
            continue

        norm = _normalize(cells)
        rows = [r for r, _c in norm]
        cols = [c for _r, c in norm]
        if max(rows) + 1 != h or max(cols) + 1 != w:
            continue
        if not _is_connected(norm):
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True

        if mask.all():
            continue
        return mask
    return None


def _all_positions(canvas_h: int, canvas_w: int, mask_h: int, mask_w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(canvas_h - mask_h + 1) for c in range(canvas_w - mask_w + 1)]


def _choose_layers(
    shape_mask: np.ndarray,
    stamp_count: int,
    overlay_h: int,
    overlay_w: int,
) -> list[set[tuple[int, int]]] | None:
    mask_h, mask_w = map(int, shape_mask.shape)
    rel_cells = _mask_rel_cells(shape_mask)
    positions = _all_positions(overlay_h, overlay_w, mask_h, mask_w)
    if not positions:
        return None

    top_r, top_c = random.choice(positions)
    layers = [_placed_cells(rel_cells, top_r, top_c)]
    occupied = set(layers[0])

    for _ in range(stamp_count - 1):
        candidates = []
        for r, c in positions:
            cells = _placed_cells(rel_cells, r, c)
            overlap = len(cells & occupied)
            visible = cells - occupied
            if overlap == 0 or not visible:
                continue
            candidates.append(cells)

        if not candidates:
            return None

        random.shuffle(candidates)
        chosen = random.choice(candidates)
        layers.append(chosen)
        occupied |= chosen

    return layers


def _render_overlay(
    overlay_h: int,
    overlay_w: int,
    colors_top_to_bottom: list[int],
    layers_top_to_bottom: list[set[tuple[int, int]]],
    bg: int,
) -> np.ndarray:
    overlay = np.full((overlay_h, overlay_w), int(bg), dtype=int)
    for color, cells in zip(
        reversed(colors_top_to_bottom),
        reversed(layers_top_to_bottom),
        strict=False,
    ):
        for r, c in cells:
            overlay[r, c] = int(color)
    return overlay


def _gen_attempt() -> np.ndarray | None:
    stamp_count = random.randint(2, 4)
    palette = list(range(10))
    bg = random.choice(palette)
    stamp_colors = random.sample([c for c in palette if c != bg], stamp_count)

    shape_mask = _random_shape_mask()
    if shape_mask is None:
        return None

    mask_h, mask_w = map(int, shape_mask.shape)
    overlay_h = random.randint(mask_h + 3, mask_h + 8)
    overlay_w = random.randint(mask_w + 3, mask_w + 8)
    layers = _choose_layers(shape_mask, stamp_count, overlay_h, overlay_w)
    if layers is None:
        return None

    overlay = _render_overlay(overlay_h, overlay_w, stamp_colors, layers, bg)
    display_w = stamp_count * mask_w + (stamp_count - 1) + random.randint(0, 4)
    expected = _render_display(shape_mask, stamp_colors, overlay_h, display_w, bg)

    grid = np.full((overlay_h, overlay_w + 1 + display_w), int(bg), dtype=int)
    grid[:, :overlay_w] = overlay

    legend_top_min = max(0, overlay_h - stamp_count - 2)
    legend_top = random.randint(legend_top_min, overlay_h - stamp_count)
    for i, color in enumerate(stamp_colors):
        grid[legend_top + i, overlay_w] = int(color)

    if np.count_nonzero(grid == bg) <= max(np.count_nonzero(grid == c) for c in stamp_colors):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
