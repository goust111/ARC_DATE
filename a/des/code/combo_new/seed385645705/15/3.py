from __future__ import annotations

import random

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _rectangle_border_values(grid: np.ndarray, inset: int) -> np.ndarray:
    top = inset
    left = inset
    bottom = grid.shape[0] - 1 - inset
    right = grid.shape[1] - 1 - inset
    values: list[int] = []
    values.extend(int(v) for v in grid[top, left : right + 1])
    if bottom > top:
        values.extend(int(v) for v in grid[bottom, left : right + 1])
    if bottom - top > 1:
        values.extend(int(v) for v in grid[top + 1 : bottom, left])
        if right > left:
            values.extend(int(v) for v in grid[top + 1 : bottom, right])
    return np.asarray(values, dtype=int)


def _detect_band_colors(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    colors: list[int] = []
    max_layers = min(h, w) // 2
    for inset in range(max_layers):
        top = inset
        left = inset
        bottom = h - 1 - inset
        right = w - 1 - inset
        if bottom - top < 2 or right - left < 2:
            break
        border = _rectangle_border_values(grid, inset)
        unique = np.unique(border)
        if unique.size == 1 and int(unique[0]) != BG:
            colors.append(int(unique[0]))
        else:
            break
    assert 2 <= len(colors) <= 4, "expected 2-4 concentric color bands"
    assert len(colors) == len(set(colors)), "band colors must be distinct"
    assert h - 2 * len(colors) > 0 and w - 2 * len(colors) > 0
    return colors


def _components_of_color(region: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = region.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(region[r, c]) != color:
                continue
            seen[r, c] = True
            queue = [(r, c)]
            comp: list[tuple[int, int]] = []
            head = 0
            while head < len(queue):
                cr, cc = queue[head]
                head += 1
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(region[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            components.append(comp)
    return components


def _boundary_cells(region: np.ndarray, component: list[tuple[int, int]], color: int) -> list[tuple[int, int]]:
    h, w = region.shape
    boundary: list[tuple[int, int]] = []
    for r, c in component:
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w or int(region[nr, nc]) != color:
                boundary.append((r, c))
                break
    return boundary


def main(grid: np.ndarray) -> np.ndarray:
    """
    Strip one outer layer from interior objects whose colors appear in the nested border bands, following the bands from outside to inside.

    Input structure:
    - The background is black.
    - The grid edge is wrapped by 2-4 one-cell-thick concentric rectangular color bands, each with a distinct color.
    - Inside the banded frame sit several monochrome 4-connected objects; band-colored objects have a peelable outer shell, and optional non-band-colored distractor objects may also appear.

    Rule:
    - Read the band colors from the outermost ring to the innermost ring.
    - Restrict attention to the central region enclosed by the rings.
    - For each band color in that order, find every 4-connected object of that color in the central region and erase all pixels on the object's outer layer: any pixel that touches the background, another color, or the central-region boundary by a 4-neighbor side.
    - Leave objects whose colors do not appear in the border bands unchanged.

    Key insight:
    - The nested border colors act as a visible execution schedule, while the actual spatial edit is a one-step 4-neighbor erosion applied only to interior objects whose colors are listed by those rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    band_colors = _detect_band_colors(g)
    layers = len(band_colors)
    out = g.copy()
    inner = out[layers : g.shape[0] - layers, layers : g.shape[1] - layers]

    for color in band_colors:
        for comp in _components_of_color(inner, int(color)):
            for r, c in _boundary_cells(inner, comp, int(color)):
                inner[r, c] = BG

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    assert rows.size > 0
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _dilate4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        out[r, c] = True
        for dr, dc in DIR4:
            nr = int(r) + dr
            nc = int(c) + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _erode4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        keep = True
        for dr, dc in DIR4:
            nr = int(r) + dr
            nc = int(c) + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w or not bool(mask[nr, nc]):
                keep = False
                break
        if keep:
            out[int(r), int(c)] = True
    return out


def _random_core_mask() -> np.ndarray | None:
    for _ in range(50):
        h = random.randint(1, 4)
        w = random.randint(1, 4)
        area = random.randint(1, min(h * w, 6))
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells and (nr, nc) not in frontier:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randrange(4))
        return _crop_mask(mask)
    return None


def _random_shell_object() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(80):
        core = _random_core_mask()
        if core is None:
            continue
        full = _crop_mask(_dilate4(np.pad(core, 1, constant_values=False)))
        peeled = _erode4(full)
        if peeled.any():
            return full, peeled
    return None


def _paint_ring(grid: np.ndarray, inset: int, color: int) -> None:
    top = inset
    left = inset
    bottom = grid.shape[0] - 1 - inset
    right = grid.shape[1] - 1 - inset
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    window = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    window[mask] = color


def _place_object(
    inp_inner: np.ndarray,
    out_inner: np.ndarray,
    blocked: np.ndarray,
    color: int,
    full_mask: np.ndarray,
    final_mask: np.ndarray,
) -> bool:
    h, w = inp_inner.shape
    oh, ow = full_mask.shape
    if oh + 2 > h or ow + 2 > w:
        return False

    positions = [(r, c) for r in range(1, h - oh) for c in range(1, w - ow)]
    random.shuffle(positions)
    for r, c in positions:
        if blocked[r - 1 : r + oh + 1, c - 1 : c + ow + 1].any():
            continue
        _paint_mask(inp_inner, r, c, full_mask, color)
        _paint_mask(out_inner, r, c, final_mask, color)
        blocked[r - 1 : r + oh + 1, c - 1 : c + ow + 1] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    layers = random.randint(2, 4)
    extra_band_objects = random.randint(0, 2)
    distractor_count = 0 if random.random() < 0.5 else 1
    total_objects = layers + extra_band_objects + distractor_count

    size_options = [
        (ih, iw)
        for ih in range(8, 23)
        for iw in range(8, 23)
        if ih + 2 * layers <= 30 and iw + 2 * layers <= 30 and ih * iw >= total_objects * 24
    ]
    if not size_options:
        return None
    inner_h, inner_w = random.choice(size_options)
    h = inner_h + 2 * layers
    w = inner_w + 2 * layers

    band_colors = random.sample(range(1, 10), layers)
    remaining_colors = [color for color in range(1, 10) if color not in band_colors]
    distractor_colors = random.sample(remaining_colors, distractor_count)

    object_colors = band_colors[:]
    for _ in range(extra_band_objects):
        object_colors.append(random.choice(band_colors))
    object_colors.extend(distractor_colors)
    random.shuffle(object_colors)

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    for inset, color in enumerate(band_colors):
        _paint_ring(inp, inset, int(color))
        _paint_ring(out, inset, int(color))

    inp_inner = inp[layers : h - layers, layers : w - layers]
    out_inner = out[layers : h - layers, layers : w - layers]
    blocked = np.zeros((inner_h, inner_w), dtype=bool)
    band_color_set = set(int(color) for color in band_colors)

    for color in object_colors:
        obj = _random_shell_object()
        if obj is None:
            return None
        full_mask, peeled_mask = obj
        final_mask = peeled_mask if int(color) in band_color_set else full_mask
        if not _place_object(inp_inner, out_inner, blocked, int(color), full_mask, final_mask):
            return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid peeling-bands ARC input")
