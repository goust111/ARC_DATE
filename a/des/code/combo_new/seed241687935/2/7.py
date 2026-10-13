import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = tuple(color for color in range(1, 10) if color != GREY)
LEGEND_SIDES = ("top", "bottom", "left", "right")
GENERATION_ATTEMPTS = 400
RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _contains_or_equal(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _is_rectangular_ring(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    local = np.zeros((height, width), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros_like(local)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(local, expected)


def _find_legend(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    colored = (grid != BLACK) & (grid != GREY)
    candidates: list[tuple[int, int, np.ndarray]] = []
    for comp in _components(colored):
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        touches_border = r0 == 0 or c0 == 0 or r1 == grid.shape[0] - 1 or c1 == grid.shape[1] - 1
        if not touches_border:
            continue
        if height != 1 and width != 1:
            continue
        area = len(comp)
        if area != max(height, width):
            continue
        candidates.append((area, 0 if height == 1 else 1, comp))
    assert len(candidates) == 1, "expected one border legend strip"
    comp = candidates[0][2]
    r0, c0, r1, c1 = _bbox(comp)
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[comp[:, 0], comp[:, 1]] = True
    if r0 == r1:
        ordered = comp[np.argsort(comp[:, 1])]
    else:
        ordered = comp[np.argsort(comp[:, 0])]
    colors = [int(grid[r, c]) for r, c in ordered]
    return legend_mask, colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each nested grey rectangle ring from the outermost layer inward using the border legend strip.

    Input structure:
    - The background is black.
    - One or more structures are made from grey one-cell-thick rectangular rings,
      nested with black gaps between successive rings.
    - A single border legend strip contains one colored pixel per ring layer, laid
      out as a short horizontal or vertical run along the outer edge.

    Rule:
    - Read the legend colors in strip order: left-to-right for a horizontal strip
      or top-to-bottom for a vertical strip.
    - Identify every grey rectangular ring and determine how many larger grey rings
      contain it inside the same nested structure.
    - Recolor each ring with the legend color at that outer-to-inner depth.
    - Erase the legend strip to black and keep all black gaps unchanged.

    Key insight:
    - The legend gives the color sequence, while the ring bounding-box containment
      tree gives the layer index; combining those two cues yields the recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_mask, legend_colors = _find_legend(g)
    ring_components: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for comp in _components(g == GREY):
        assert _is_rectangular_ring(comp), "grey objects must be rectangular rings"
        ring_components.append((_bbox(comp), comp))

    assert ring_components, "expected at least one grey ring structure"
    bboxes = [item[0] for item in ring_components]
    depths = [sum(_strictly_contains(other, box) for other in bboxes) for box in bboxes]
    roots = [idx for idx, depth in enumerate(depths) if depth == 0]
    assert roots, "expected at least one outer ring"
    for root_idx in roots:
        chain_len = sum(_contains_or_equal(bboxes[root_idx], box) for box in bboxes)
        assert chain_len == len(legend_colors), "each structure must match the legend length"
    assert max(depths) + 1 == len(legend_colors), "legend length must match ring depth"

    out = g.copy()
    out[legend_mask] = BLACK
    for (_, comp), depth in zip(ring_components, depths):
        out[comp[:, 0], comp[:, 1]] = legend_colors[depth]
    return out


def _randint(lo: int, hi: int) -> int:
    return int(RNG.integers(lo, hi + 1))


def _sample_outer_size(layers: int, multi: bool) -> tuple[int, int]:
    shrink = 4 * (layers - 1)
    if multi:
        inner_max = 7 if layers == 2 else 4
    else:
        inner_max = 9
    inner_h = _randint(3, inner_max)
    inner_w = _randint(3, inner_max)
    return inner_h + shrink, inner_w + shrink


def _paint_ring(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _draw_structure(grid: np.ndarray, top: int, left: int, height: int, width: int, layers: int) -> None:
    for depth in range(layers):
        offset = 2 * depth
        _paint_ring(
            grid,
            top + offset,
            left + offset,
            top + height - 1 - offset,
            left + width - 1 - offset,
            GREY,
        )


def _split_extra(total: int) -> tuple[int, int]:
    first = _randint(0, total)
    return first, total - first


def _legend_reserve(side: str) -> tuple[int, int, int, int]:
    top = 2 if side == "top" else 1
    bottom = 2 if side == "bottom" else 1
    left = 2 if side == "left" else 1
    right = 2 if side == "right" else 1
    return top, bottom, left, right


def _place_legend(grid: np.ndarray, side: str, colors: list[int]) -> None:
    length = len(colors)
    height, width = grid.shape
    if side in ("top", "bottom"):
        row = 0 if side == "top" else height - 1
        start = _randint(1, width - length - 1)
        for idx, color in enumerate(colors):
            grid[row, start + idx] = color
    else:
        col = 0 if side == "left" else width - 1
        start = _randint(1, height - length - 1)
        for idx, color in enumerate(colors):
            grid[start + idx, col] = color


def _gen_attempt() -> np.ndarray | None:
    layers = _randint(2, 5)
    allow_multiple = layers <= 3
    structure_count = int(RNG.choice([1, 2])) if allow_multiple else 1
    side = str(RNG.choice(LEGEND_SIDES))
    colors = list(RNG.choice(COLOR_POOL, size=layers, replace=False))

    sizes = [_sample_outer_size(layers, structure_count > 1) for _ in range(structure_count)]
    layout = "single"
    gap = 0
    if structure_count == 2:
        layout = str(RNG.choice(["horizontal", "vertical"]))
        gap = _randint(2, 4)

    if layout == "single":
        box_h, box_w = sizes[0]
        offsets = [(0, 0)]
    elif layout == "horizontal":
        box_h = max(size[0] for size in sizes)
        box_w = sizes[0][1] + gap + sizes[1][1]
        offsets = [
            (_randint(0, box_h - sizes[0][0]), 0),
            (_randint(0, box_h - sizes[1][0]), sizes[0][1] + gap),
        ]
    else:
        box_h = sizes[0][0] + gap + sizes[1][0]
        box_w = max(size[1] for size in sizes)
        offsets = [
            (0, _randint(0, box_w - sizes[0][1])),
            (sizes[0][0] + gap, _randint(0, box_w - sizes[1][1])),
        ]

    reserve_top, reserve_bottom, reserve_left, reserve_right = _legend_reserve(side)
    min_height = reserve_top + box_h + reserve_bottom
    min_width = reserve_left + box_w + reserve_right
    if side in ("top", "bottom"):
        min_width = max(min_width, layers + 2)
    else:
        min_height = max(min_height, layers + 2)
    if min_height > 30 or min_width > 30:
        return None

    extra_h = _randint(0, min(6, 30 - min_height))
    extra_w = _randint(0, min(6, 30 - min_width))
    extra_top, extra_bottom = _split_extra(extra_h)
    extra_left, extra_right = _split_extra(extra_w)
    height = min_height + extra_top + extra_bottom
    width = min_width + extra_left + extra_right

    grid = np.zeros((height, width), dtype=int)
    origin_r = reserve_top + extra_top
    origin_c = reserve_left + extra_left
    for (dh, dw), (off_r, off_c) in zip(sizes, offsets):
        _draw_structure(grid, origin_r + off_r, origin_c + off_c, dh, dw, layers)
    _place_legend(grid, side, colors)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if (grid == GREY).sum() == 0:
        return None
    if np.any((grid != BLACK) & (grid != GREY) & (out != BLACK)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
