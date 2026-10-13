from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
SEPARATOR_COLOR = 9
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PANEL_SIZES = (7, 9, 11)
SPRITE_PANEL_COUNT_RANGE = (3, 6)
GENERATION_RETRIES = 400


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components


def _rotation_order(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if np.array_equal(mask, np.rot90(mask)):
        return 4
    if np.array_equal(mask, np.rot90(mask, 2)):
        return 2
    return 1


def _panel_spans(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    is_separator = np.zeros(w, dtype=bool)
    for c in range(w):
        column = grid[:, c]
        if int(column[0]) != bg and np.all(column == column[0]):
            is_separator[c] = True

    sep_cols = np.flatnonzero(is_separator)
    assert sep_cols.size > 0, "expected separator columns"
    assert {int(grid[0, c]) for c in sep_cols} == {SEPARATOR_COLOR}, "expected maroon separators"

    spans: list[tuple[int, int]] = []
    start = 0
    while start < w:
        while start < w and is_separator[start]:
            start += 1
        if start >= w:
            break
        end = start
        while end < w and not is_separator[end]:
            end += 1
        spans.append((start, end))
        start = end

    assert len(spans) >= 2, "expected legend plus at least one sprite panel"
    assert all(end > start for start, end in spans)
    assert all(end - start == h for start, end in spans), "panels should be square"
    return spans


def _legend_map(panel: np.ndarray, bg: int) -> dict[int, int]:
    components = _find_components(panel != bg)
    assert len(components) == 3, "legend must contain three colored blocks"

    ordered: list[tuple[float, int]] = []
    for coords in components:
        colors = {int(panel[r, c]) for r, c in coords}
        assert len(colors) == 1, "each legend block must be monochrome"
        ordered.append((float(coords[:, 0].mean()), next(iter(colors))))

    ordered.sort()
    colors = [color for _row, color in ordered]
    assert len(set(colors)) == 3, "legend colors must be distinct"
    return {1: colors[0], 2: colors[1], 4: colors[2]}


def _sprite_mask(panel: np.ndarray, bg: int) -> np.ndarray:
    mask = panel != bg
    components = _find_components(mask)
    assert len(components) == 1, "each sprite panel must contain exactly one sprite"
    colors = {int(panel[r, c]) for r, c in components[0]}
    assert len(colors) == 1, "sprite must be monochrome"
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each centered sprite by its rotational symmetry order using the left legend.

    The leftmost panel contains three legend blocks from top to bottom for
    orders 1, 2, and 4. Every later panel contains a single 4-connected
    monochrome sprite on a black background, separated by full-height maroon
    columns. A sprite invariant under 90-degree rotation gets the order-4
    legend color; otherwise a sprite invariant under 180-degree rotation gets
    the order-2 legend color; all remaining sprites get the order-1 legend
    color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = BG_COLOR
    spans = _panel_spans(g, bg)
    legend = _legend_map(g[:, spans[0][0] : spans[0][1]], bg)

    out = g.copy()
    for start, end in spans[1:]:
        panel = g[:, start:end]
        mask = _sprite_mask(panel, bg)
        out[:, start:end][mask] = legend[_rotation_order(mask)]

    return out


def _is_connected(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    if not bool(mask.any()):
        return False
    return len(_find_components(mask)) == 1


def _touches(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            return True
    return False


def _orbit90(size: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    orbit = set()
    rr, cc = r, c
    for _ in range(4):
        orbit.add((rr, cc))
        rr, cc = cc, size - 1 - rr
    return tuple(sorted(orbit))


def _orbit180(size: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted({(r, c), (size - 1 - r, size - 1 - c)}))


def _orbit_candidates(mask: np.ndarray, orbit_fn) -> list[tuple[tuple[int, int], ...]]:
    size = mask.shape[0]
    seen: set[tuple[tuple[int, int], ...]] = set()
    candidates: list[tuple[tuple[int, int], ...]] = []

    for r in range(size):
        for c in range(size):
            if mask[r, c]:
                continue
            orbit = orbit_fn(size, r, c)
            if orbit in seen:
                continue
            seen.add(orbit)
            new_cells = tuple(cell for cell in orbit if not mask[cell])
            if not new_cells:
                continue
            trial = mask.copy()
            for rr, cc in new_cells:
                trial[rr, cc] = True
            if _is_connected(trial):
                candidates.append(new_cells)
    return candidates


def _generate_order4_mask(max_size: int) -> np.ndarray:
    sizes = [size for size in range(3, max_size + 1, 2)]
    for _ in range(100):
        size = random.choice(sizes)
        center = size // 2
        mask = np.zeros((size, size), dtype=bool)
        mask[center, center] = True
        steps = random.randint(1, min(5, (size * size - 1) // 4))
        for _step in range(steps):
            candidates = _orbit_candidates(mask, _orbit90)
            if not candidates:
                break
            for rr, cc in random.choice(candidates):
                mask[rr, cc] = True
        if _rotation_order(mask) == 4 and _is_connected(mask) and int(mask.sum()) >= 5:
            return mask
    raise RuntimeError("failed to build order-4 sprite")


def _generate_order2_mask(max_size: int) -> np.ndarray:
    sizes = [size for size in range(3, max_size + 1, 2)]
    for _ in range(100):
        size = random.choice(sizes)
        center = size // 2
        mask = np.zeros((size, size), dtype=bool)
        mask[center, center] = True
        steps = random.randint(1, min(7, (size * size - 1) // 2))
        for _step in range(steps):
            candidates = _orbit_candidates(mask, _orbit180)
            if not candidates:
                break
            for rr, cc in random.choice(candidates):
                mask[rr, cc] = True
        if _rotation_order(mask) == 2 and _is_connected(mask) and int(mask.sum()) >= 3:
            return mask
    raise RuntimeError("failed to build order-2 sprite")


def _generate_order1_mask(max_size: int) -> np.ndarray:
    max_arm = max_size // 2
    for _ in range(100):
        arm = random.randint(1, max_arm)
        size = 2 * arm + 1
        center = arm
        mask = np.zeros((size, size), dtype=bool)
        mask[:, center] = True
        mask[center, :] = True

        additions = random.randint(1, min(4, size * size - int(mask.sum())))
        for _step in range(additions):
            choices = [
                (r, c)
                for r in range(size)
                for c in range(size)
                if not mask[r, c] and _touches(mask, r, c)
            ]
            if not choices:
                break
            rr, cc = random.choice(choices)
            mask[rr, cc] = True
        if _rotation_order(mask) == 1 and _is_connected(mask):
            return mask
    raise RuntimeError("failed to build order-1 sprite")


def _generate_sprite_mask(order: int, panel_size: int) -> np.ndarray:
    local_max = panel_size - 2
    if order == 4:
        return _generate_order4_mask(local_max)
    if order == 2:
        return _generate_order2_mask(local_max)
    if order == 1:
        return _generate_order1_mask(local_max)
    raise ValueError(f"unsupported order: {order}")


def _centered_panel(mask: np.ndarray, panel_size: int, color: int) -> np.ndarray:
    panel = np.full((panel_size, panel_size), BG_COLOR, dtype=int)
    h, w = mask.shape
    top = (panel_size - h) // 2
    left = (panel_size - w) // 2
    panel[top : top + h, left : left + w][mask] = color
    return panel


def _legend_panel(panel_size: int, colors: list[int]) -> np.ndarray:
    panel = np.full((panel_size, panel_size), BG_COLOR, dtype=int)
    rows = [1, panel_size // 2, panel_size - 2]
    for row, color in zip(rows, colors):
        width = random.randint(2, panel_size - 3)
        col = random.randint(1, panel_size - width - 1)
        panel[row, col : col + width] = color
    return panel


def _build_pair() -> tuple[np.ndarray, np.ndarray]:
    panel_size = random.choice(PANEL_SIZES)
    sprite_panel_count = random.randint(*SPRITE_PANEL_COUNT_RANGE)

    legend_colors = random.sample(list(range(1, SEPARATOR_COLOR)), 3)
    legend_panel = _legend_panel(panel_size, legend_colors)
    legend_map = {1: legend_colors[0], 2: legend_colors[1], 4: legend_colors[2]}

    sprite_palette = [color for color in range(1, SEPARATOR_COLOR) if color not in legend_colors]
    orders = [1, 2, 4] + [random.choice([1, 2, 4]) for _ in range(sprite_panel_count - 3)]
    random.shuffle(orders)

    input_panels = [legend_panel]
    output_panels = [legend_panel.copy()]
    for order in orders:
        sprite_color = random.choice(sprite_palette)
        mask = _generate_sprite_mask(order, panel_size)
        panel = _centered_panel(mask, panel_size, sprite_color)
        recolored = _centered_panel(mask, panel_size, legend_map[order])
        input_panels.append(panel)
        output_panels.append(recolored)

    panel_count = len(input_panels)
    width = panel_count * panel_size + (panel_count - 1)
    inp = np.full((panel_size, width), BG_COLOR, dtype=int)
    out = np.full((panel_size, width), BG_COLOR, dtype=int)

    cursor = 0
    for idx, (panel_in, panel_out) in enumerate(zip(input_panels, output_panels)):
        inp[:, cursor : cursor + panel_size] = panel_in
        out[:, cursor : cursor + panel_size] = panel_out
        cursor += panel_size
        if idx != panel_count - 1:
            inp[:, cursor] = SEPARATOR_COLOR
            out[:, cursor] = SEPARATOR_COLOR
            cursor += 1

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        inp, expected = _build_pair()
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
