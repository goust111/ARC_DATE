import random

import numpy as np


PALETTE = tuple(range(10))
COUNTS = (0, 1, 2, 3)
VERTICAL = "vertical"
HORIZONTAL = "horizontal"
MAX_GENERATION_ATTEMPTS = 200


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _mode(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).ravel()
    colors, counts = np.unique(flat, return_counts=True)
    return int(colors[np.argmax(counts)])


def _border_reachable(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(mask.shape, r, c):
            if mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return seen


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            seen[r, c] = True
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                for nr, nc in _neighbors4(mask.shape, cr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _hole_count(panel: np.ndarray, bg: int) -> int:
    bg_mask = panel == int(bg)
    exterior = _border_reachable(bg_mask)
    enclosed = bg_mask & ~exterior
    return _count_components(enclosed)


def _find_separator_index(grid: np.ndarray, axis: int) -> int:
    length = grid.shape[axis]
    candidates = []
    for idx in range(1, length - 1):
        if idx != length - idx - 1:
            continue
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        if np.all(line == line.flat[0]):
            candidates.append(idx)
    assert len(candidates) == 1, "expected exactly one full separator line on each axis"
    return int(candidates[0])


def _extract_panels(grid: np.ndarray):
    sep_r = _find_separator_index(grid, axis=0)
    sep_c = _find_separator_index(grid, axis=1)
    sep_color = int(grid[sep_r, 0])
    assert np.all(grid[sep_r, :] == sep_color), "separator row must be uniform"
    assert np.all(grid[:, sep_c] == sep_color), "separator column must be uniform"

    mask = np.ones(grid.shape, dtype=bool)
    mask[sep_r, :] = False
    mask[:, sep_c] = False
    bg = _mode(grid[mask])

    top = sep_r
    left = sep_c
    bottom = grid.shape[0] - sep_r - 1
    right = grid.shape[1] - sep_c - 1
    assert top == bottom and left == right, "separator must split four equal panels"

    panels = [
        grid[:sep_r, :sep_c].copy(),
        grid[:sep_r, sep_c + 1 :].copy(),
        grid[sep_r + 1 :, :sep_c].copy(),
        grid[sep_r + 1 :, sep_c + 1 :].copy(),
    ]
    assert len({panel.shape for panel in panels}) == 1, "panel sizes must match"
    return panels, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the four quadrant sprites onto one panel, drawing the most hole-rich sprite first and the fewest-hole sprite last.

    Input structure:
    - One full separator row and one full separator column split the grid into four equal panels.
    - The panels share a background color, and each panel contains exactly one monochrome sprite of a distinct color.
    - A sprite may contain zero or more enclosed background holes.

    Rule:
    - In each panel, count the background 4-connected components that are not connected to the panel border.
    - Remove the cross separator, align the four panels by local coordinates, and use one panel-sized canvas for the output.
    - Paint the sprite pixels in descending hole-count order so that later sprites with fewer holes overwrite earlier ones.

    Key insight:
    - Layer order is computed from the number of enclosed holes, not from quadrant position or color identity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    panels, bg = _extract_panels(g)
    panel_h, panel_w = panels[0].shape
    layers = []
    for panel in panels:
        colors = [int(color) for color in np.unique(panel) if int(color) != bg]
        assert len(colors) == 1, "each panel must contain exactly one sprite color"
        color = colors[0]
        holes = _hole_count(panel, bg)
        layers.append((holes, color, panel))

    hole_counts = [holes for holes, _color, _panel in layers]
    assert len(set(hole_counts)) == len(hole_counts), "hole counts must give a unique layer order"

    out = np.full((panel_h, panel_w), bg, dtype=int)
    for _holes, color, panel in sorted(layers, key=lambda item: item[0], reverse=True):
        out[panel == color] = color
    return out


def _draw_ring(mask: np.ndarray, top: int, left: int) -> None:
    mask[top : top + 3, left : left + 3] = True
    mask[top + 1, left + 1] = False


def _draw_plus(mask: np.ndarray) -> None:
    h, w = mask.shape
    cr = h // 2
    cc = w // 2
    for r, c in (
        (cr, cc),
        (cr - 1, cc),
        (cr + 1, cc),
        (cr, cc - 1),
        (cr, cc + 1),
    ):
        mask[r, c] = True


def _draw_bridge(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        mask[r0, lo : hi + 1] = True
    elif c0 == c1:
        lo, hi = sorted((r0, r1))
        mask[lo : hi + 1, c0] = True
    else:
        raise ValueError("bridges must be axis-aligned")


def _ring_layout(shape: tuple[int, int], orientation: str):
    h, w = shape
    cr = h // 2
    cc = w // 2
    if orientation == VERTICAL:
        return (
            {
                "center": (cr - 1, cc - 1),
                "first": (1, cc - 1),
                "second": (h - 4, cc - 1),
            },
            {
                "first": ((3, cc), (cr - 1, cc)),
                "second": ((cr + 1, cc), (h - 5, cc)),
            },
        )
    return (
        {
            "center": (cr - 1, cc - 1),
            "first": (cr - 1, 1),
            "second": (cr - 1, w - 4),
        },
        {
            "first": ((cr, 3), (cr, cc - 1)),
            "second": ((cr, cc + 1), (cr, w - 5)),
        },
    )


def _build_sprite_mask(
    shape: tuple[int, int],
    hole_count: int,
    orientation: str,
    flip_outer_order: bool,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if hole_count == 0:
        _draw_plus(mask)
        return mask

    anchors, bridges = _ring_layout(shape, orientation)
    outer_keys = ["first", "second"]
    if flip_outer_order:
        outer_keys.reverse()

    _draw_ring(mask, *anchors["center"])
    if hole_count >= 2:
        key = outer_keys[0]
        _draw_ring(mask, *anchors[key])
        _draw_bridge(mask, *bridges[key])
    if hole_count >= 3:
        key = outer_keys[1]
        _draw_ring(mask, *anchors[key])
        _draw_bridge(mask, *bridges[key])
    return mask


def _panel_from_mask(mask: np.ndarray, bg: int, color: int) -> np.ndarray:
    panel = np.full(mask.shape, bg, dtype=int)
    panel[mask] = int(color)
    return panel


def _expected_output(panels: list[np.ndarray], counts: list[int], colors: list[int], bg: int) -> np.ndarray:
    out = np.full(panels[0].shape, bg, dtype=int)
    order = sorted(range(4), key=lambda idx: counts[idx], reverse=True)
    for idx in order:
        out[panels[idx] == colors[idx]] = colors[idx]
    return out


def _sample_panel_shape() -> tuple[str, tuple[int, int], bool]:
    orientation = random.choice((VERTICAL, HORIZONTAL))
    flip_outer_order = random.choice((False, True))
    if orientation == VERTICAL:
        return orientation, (13, random.choice((11, 13))), flip_outer_order
    return orientation, (random.choice((11, 13)), 13), flip_outer_order


def _gen_attempt() -> np.ndarray | None:
    orientation, panel_shape, flip_outer_order = _sample_panel_shape()
    counts = list(COUNTS)
    random.shuffle(counts)

    colors = random.sample(PALETTE, k=6)
    bg = colors[0]
    sep = colors[1]
    sprite_colors = colors[2:]

    panels = []
    for count, color in zip(counts, sprite_colors):
        mask = _build_sprite_mask(panel_shape, count, orientation, flip_outer_order)
        panel = _panel_from_mask(mask, bg, color)
        if _hole_count(panel, bg) != count:
            return None
        panels.append(panel)

    h, w = panel_shape
    grid = np.full((2 * h + 1, 2 * w + 1), bg, dtype=int)
    grid[h, :] = sep
    grid[:, w] = sep
    positions = [
        (slice(0, h), slice(0, w)),
        (slice(0, h), slice(w + 1, 2 * w + 1)),
        (slice(h + 1, 2 * h + 1), slice(0, w)),
        (slice(h + 1, 2 * h + 1), slice(w + 1, 2 * w + 1)),
    ]
    source_order = list(range(4))
    random.shuffle(source_order)
    placed_panels = []
    placed_counts = []
    placed_colors = []
    for pos_idx, src_idx in enumerate(source_order):
        rs, cs = positions[pos_idx]
        grid[rs, cs] = panels[src_idx]
        placed_panels.append(panels[src_idx])
        placed_counts.append(counts[src_idx])
        placed_colors.append(sprite_colors[src_idx])

    expected = _expected_output(placed_panels, placed_counts, placed_colors, bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
