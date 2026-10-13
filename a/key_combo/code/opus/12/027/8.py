import random

import numpy as np


BG = 0
MARKER = 2
NEUTRAL = 5
TEMPLATE_SIZE = 2
SYMBOL_SIZE = 3
CELL_STRIDE = 4
LEGEND_GAP = 2
SEPARATOR_ROWS = 1
SWATCH_COLORS = (1, 3, 4, 6, 7, 8, 9)
SIDE_TO_DIRECTION = {
    "top": "right",
    "right": "down",
    "bottom": "left",
    "left": "up",
}
TEMPLATE_CATALOG = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0]], dtype=bool),
    np.array([[0, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [0, 0]], dtype=bool),
    np.array([[0, 0], [1, 1]], dtype=bool),
)
MAX_GENERATION_ATTEMPTS = 400


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(values.tolist()):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, *, colors: set[int] | None = None) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            if colors is not None and color not in colors:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _marker_side(grid: np.ndarray) -> str:
    sides = []
    if np.all(grid[0, :] == MARKER):
        sides.append("top")
    if np.all(grid[:, -1] == MARKER):
        sides.append("right")
    if np.all(grid[-1, :] == MARKER):
        sides.append("bottom")
    if np.all(grid[:, 0] == MARKER):
        sides.append("left")
    assert len(sides) == 1, sides
    return sides[0]


def _strip_marker(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return grid[1:, :].copy()
    if side == "right":
        return grid[:, :-1].copy()
    if side == "bottom":
        return grid[:-1, :].copy()
    if side == "left":
        return grid[:, 1:].copy()
    raise AssertionError(side)


def _parse_legend(core: np.ndarray) -> tuple[int, dict[int, np.ndarray]]:
    occupied_rows = np.any(core != BG, axis=1)
    legend_r0 = int(np.argmax(occupied_rows))
    legend = core[legend_r0 : legend_r0 + TEMPLATE_SIZE, :]

    swatch_colors = set(int(v) for v in np.unique(legend)) - {BG, NEUTRAL}
    swatches = []
    for comp in _find_components(legend, colors=swatch_colors):
        r0, c0, r1, c1 = comp["bbox"]
        assert comp["area"] == 4
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2)
        assert np.all(legend[r0 : r1 + 1, c0 : c1 + 1] == comp["color"])
        swatches.append((int(c0), int(comp["color"])))
    assert 2 <= len(swatches) <= 3
    swatch_order = [color for _c0, color in sorted(swatches)]

    template_runs = _runs(np.any(legend == NEUTRAL, axis=0))
    assert len(template_runs) == len(swatch_order)
    stencils: dict[int, np.ndarray] = {}
    for color, (c0, c1) in zip(swatch_order, template_runs):
        assert c1 - c0 + 1 == TEMPLATE_SIZE
        mask = legend[:, c0 : c1 + 1] == NEUTRAL
        assert mask.shape == (2, 2)
        assert np.any(mask)
        assert np.all(mask.any(axis=0)) and np.all(mask.any(axis=1))
        stencils[color] = mask
    return legend_r0 + TEMPLATE_SIZE, stencils


def _parse_symbolic(core: np.ndarray, legend_bottom: int) -> np.ndarray:
    lower = core[legend_bottom:, :]
    colors = set(int(v) for v in np.unique(lower)) - {BG}
    components = _find_components(lower, colors=colors)
    objects: list[tuple[int, int, int]] = []
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        if comp["area"] != 8 or (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        color = int(comp["color"])
        patch = lower[r0 : r1 + 1, c0 : c1 + 1]
        expected = np.full((3, 3), color, dtype=int)
        expected[1, 1] = BG
        if np.array_equal(patch, expected):
            objects.append((int(r0), int(c0), color))

    assert 3 <= len(objects) <= 6
    row_positions = [r for r, _c, _color in objects]
    col_positions = [c for _r, c, _color in objects]
    row_base, row_last = min(row_positions), max(row_positions)
    col_base, col_last = min(col_positions), max(col_positions)
    assert (row_last - row_base) % CELL_STRIDE == 0
    assert (col_last - col_base) % CELL_STRIDE == 0
    rows = list(range(row_base, row_last + 1, CELL_STRIDE))
    cols = list(range(col_base, col_last + 1, CELL_STRIDE))
    assert 2 <= len(rows) <= 4
    assert 2 <= len(cols) <= 4

    symbolic = np.full((len(rows), len(cols)), BG, dtype=int)
    for r, c, color in objects:
        rr = (r - row_base) // CELL_STRIDE
        cc = (c - col_base) // CELL_STRIDE
        assert symbolic[rr, cc] == BG
        symbolic[rr, cc] = color
    return symbolic


def _compact(symbolic: np.ndarray, direction: str) -> np.ndarray:
    out = np.full_like(symbolic, BG)
    h, w = symbolic.shape
    if direction == "right":
        for r in range(h):
            values = [int(v) for v in symbolic[r, :] if int(v) != BG]
            out[r, w - len(values) :] = values
    elif direction == "left":
        for r in range(h):
            values = [int(v) for v in symbolic[r, :] if int(v) != BG]
            out[r, : len(values)] = values
    elif direction == "down":
        for c in range(w):
            values = [int(v) for v in symbolic[:, c] if int(v) != BG]
            out[h - len(values) :, c] = values
    elif direction == "up":
        for c in range(w):
            values = [int(v) for v in symbolic[:, c] if int(v) != BG]
            out[: len(values), c] = values
    else:
        raise AssertionError(direction)
    return out


def _expand(symbolic: np.ndarray, stencils: dict[int, np.ndarray]) -> np.ndarray:
    h, w = symbolic.shape
    out = np.full((h * TEMPLATE_SIZE, w * TEMPLATE_SIZE), BG, dtype=int)
    for r in range(h):
        for c in range(w):
            color = int(symbolic[r, c])
            if color == BG:
                continue
            block = out[r * 2 : (r + 1) * 2, c * 2 : (c + 1) * 2]
            block[stencils[color]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compress hollow 3x3 symbols to indexed pixels, apply marker-directed gravity, then expand each pixel through its legend stencil.

    Input structure:
    - The background is black and one outer border edge is a full red marker line.
    - The top legend contains 2x2 solid color swatches and matching 2x2 grey stencil masks, paired left to right.
    - Below a blank separator, the lower region contains several hollow 3x3 monochrome symbols on a 4-step coarse lattice.

    Rule:
    - Read the red edge to obtain the direction by clockwise border travel: top->right, right->down, bottom->left, left->up.
    - Compress every hollow 3x3 symbol into one pixel of the same color at its coarse-grid position, keeping empty coarse cells black.
    - Apply gravity in that direction on the compressed grid, then replace each colored pixel by its paired 2x2 stencil recolored with that pixel's color.

    Key insight:
    - The lower picture is not transformed directly: it must first be reduced to a coarse symbolic grid, rearranged there by gravity, and only then lifted back into visible 2x2 patterns through the legend.
    """
    g = np.asarray(grid, dtype=int)
    side = _marker_side(g)
    direction = SIDE_TO_DIRECTION[side]
    core = _strip_marker(g, side)
    legend_bottom, stencils = _parse_legend(core)
    symbolic = _parse_symbolic(core, legend_bottom)
    compacted = _compact(symbolic, direction)
    return _expand(compacted, stencils)


def _legend_width(color_count: int) -> int:
    swatch_width = color_count * 3 - 1
    stencil_width = color_count * 3 - 1
    return swatch_width + LEGEND_GAP + stencil_width


def _physical_symbol_size(cells: int) -> int:
    return cells * CELL_STRIDE - 1


def _hollow_symbol(color: int) -> np.ndarray:
    patch = np.full((SYMBOL_SIZE, SYMBOL_SIZE), color, dtype=int)
    patch[1, 1] = BG
    return patch


def _build_packed_horizontal(height: int, width: int, color_count: int) -> np.ndarray | None:
    min_count = max(3, color_count, width + 1)
    max_count = min(6, height * width - 1)
    if min_count > max_count:
        return None

    for _ in range(100):
        total = random.randint(min_count, max_count)
        lengths = [0] * height
        full_row = random.randrange(height)
        lengths[full_row] = width
        remaining = total - width

        required_rows = {0, height - 1}
        for row in required_rows:
            if lengths[row] == 0:
                if remaining == 0:
                    break
                lengths[row] = 1
                remaining -= 1
        if remaining < 0 or any(lengths[row] == 0 for row in required_rows):
            continue

        while remaining > 0:
            row = random.randrange(height)
            if lengths[row] == width:
                continue
            lengths[row] += 1
            remaining -= 1

        if not any(0 < length < width for length in lengths):
            continue

        grid = np.full((height, width), BG, dtype=int)
        for r, length in enumerate(lengths):
            if length == 0:
                continue
            grid[r, width - length :] = 9
        return grid
    return None


def _build_packed_vertical(height: int, width: int, color_count: int) -> np.ndarray | None:
    min_count = max(3, color_count, height + 1)
    max_count = min(6, height * width - 1)
    if min_count > max_count:
        return None

    for _ in range(100):
        total = random.randint(min_count, max_count)
        lengths = [0] * width
        full_col = random.randrange(width)
        lengths[full_col] = height
        remaining = total - height

        required_cols = {0, width - 1}
        for col in required_cols:
            if lengths[col] == 0:
                if remaining == 0:
                    break
                lengths[col] = 1
                remaining -= 1
        if remaining < 0 or any(lengths[col] == 0 for col in required_cols):
            continue

        while remaining > 0:
            col = random.randrange(width)
            if lengths[col] == height:
                continue
            lengths[col] += 1
            remaining -= 1

        if not any(0 < length < height for length in lengths):
            continue

        grid = np.full((height, width), BG, dtype=int)
        for c, length in enumerate(lengths):
            if length == 0:
                continue
            grid[height - length :, c] = 9
        return grid
    return None


def _assign_colors(masked: np.ndarray, colors: list[int]) -> np.ndarray:
    coords = [tuple(coord) for coord in np.argwhere(masked != BG)]
    assert len(coords) >= len(colors)
    random.shuffle(coords)
    values = colors[:]
    while len(values) < len(coords):
        values.append(random.choice(colors))
    random.shuffle(values)

    out = np.full_like(masked, BG)
    for (r, c), color in zip(coords, values):
        out[r, c] = int(color)
    return out


def _uncompact_canonical(compacted: np.ndarray, direction: str) -> np.ndarray:
    h, w = compacted.shape
    out = np.full_like(compacted, BG)
    if direction == "right":
        for r in range(h):
            values = [int(v) for v in compacted[r, :] if int(v) != BG]
            out[r, : len(values)] = values
    elif direction == "left":
        for r in range(h):
            values = [int(v) for v in compacted[r, :] if int(v) != BG]
            out[r, w - len(values) :] = values
    elif direction == "down":
        for c in range(w):
            values = [int(v) for v in compacted[:, c] if int(v) != BG]
            out[: len(values), c] = values
    elif direction == "up":
        for c in range(w):
            values = [int(v) for v in compacted[:, c] if int(v) != BG]
            out[h - len(values) :, c] = values
    else:
        raise AssertionError(direction)
    return out


def _build_symbolic(height: int, width: int, direction: str, colors: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    if direction in ("right", "left"):
        packed_mask = _build_packed_horizontal(height, width, len(colors))
        if packed_mask is None:
            return None
        if direction == "left":
            packed_mask = np.fliplr(packed_mask)
    else:
        packed_mask = _build_packed_vertical(height, width, len(colors))
        if packed_mask is None:
            return None
        if direction == "up":
            packed_mask = np.flipud(packed_mask)

    compacted = _assign_colors(packed_mask, colors)
    symbolic = _uncompact_canonical(compacted, direction)
    if np.array_equal(symbolic, compacted):
        return None
    if not np.array_equal(_compact(symbolic, direction), compacted):
        return None
    return symbolic, compacted


def _random_templates(color_count: int) -> list[np.ndarray]:
    chosen = random.sample(TEMPLATE_CATALOG, color_count)
    return [mask.copy() for mask in chosen]


def _render_core(symbolic: np.ndarray, colors: list[int], templates: list[np.ndarray]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    h, w = symbolic.shape
    legend_width = _legend_width(len(colors))
    symbol_width = _physical_symbol_size(w)
    symbol_height = _physical_symbol_size(h)
    core_width = max(legend_width, symbol_width)
    core_height = TEMPLATE_SIZE + SEPARATOR_ROWS + symbol_height
    core = np.full((core_height, core_width), BG, dtype=int)

    for i, color in enumerate(colors):
        c0 = i * 3
        core[0:2, c0 : c0 + 2] = int(color)

    stencil_start = len(colors) * 3 - 1 + LEGEND_GAP
    stencil_map: dict[int, np.ndarray] = {}
    for i, (color, mask) in enumerate(zip(colors, templates)):
        c0 = stencil_start + i * 3
        core[0:2, c0 : c0 + 2] = np.where(mask, NEUTRAL, BG)
        stencil_map[int(color)] = mask

    symbol_top = TEMPLATE_SIZE + SEPARATOR_ROWS
    for r in range(h):
        for c in range(w):
            color = int(symbolic[r, c])
            if color == BG:
                continue
            rr = symbol_top + r * CELL_STRIDE
            cc = c * CELL_STRIDE
            core[rr : rr + 3, cc : cc + 3] = _hollow_symbol(color)
    return core, stencil_map


def _add_marker(core: np.ndarray, side: str) -> np.ndarray:
    h, w = core.shape
    if side == "top":
        grid = np.full((h + 1, w), BG, dtype=int)
        grid[0, :] = MARKER
        grid[1:, :] = core
        return grid
    if side == "right":
        grid = np.full((h, w + 1), BG, dtype=int)
        grid[:, -1] = MARKER
        grid[:, :-1] = core
        return grid
    if side == "bottom":
        grid = np.full((h + 1, w), BG, dtype=int)
        grid[-1, :] = MARKER
        grid[:-1, :] = core
        return grid
    if side == "left":
        grid = np.full((h, w + 1), BG, dtype=int)
        grid[:, 0] = MARKER
        grid[:, 1:] = core
        return grid
    raise AssertionError(side)


def _expected_output(symbolic: np.ndarray, side: str, stencil_map: dict[int, np.ndarray]) -> np.ndarray:
    direction = SIDE_TO_DIRECTION[side]
    compacted = _compact(symbolic, direction)
    return _expand(compacted, stencil_map)


def _generate_case() -> np.ndarray | None:
    side = random.choice(tuple(SIDE_TO_DIRECTION))
    direction = SIDE_TO_DIRECTION[side]
    height = random.randint(2, 4)
    width = random.randint(2, 4)

    max_colors = min(3, min(6, height * width - 1))
    color_count = random.randint(2, max_colors)
    colors = random.sample(list(SWATCH_COLORS), color_count)
    built = _build_symbolic(height, width, direction, colors)
    if built is None:
        return None
    symbolic, compacted = built

    templates = _random_templates(color_count)
    core, stencil_map = _render_core(symbolic, colors, templates)
    grid = _add_marker(core, side)

    if max(grid.shape) > 30:
        return None
    expected = _expand(compacted, stencil_map)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if len(set(int(v) for v in np.unique(compacted)) - {BG}) != color_count:
        return None
    if np.array_equal(symbolic, compacted):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
