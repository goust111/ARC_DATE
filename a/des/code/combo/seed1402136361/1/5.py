import numpy as np


GREY = 5
_RNG = np.random.default_rng()
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator configuration.
_GRID_H_RANGE = (15, 25)
_GRID_W_RANGE = (19, 30)
_OBJECT_COUNT_OPTIONS = (3, 4, 5)
_AREA_POOL = tuple(range(3, 10))
_PALETTE = tuple(c for c in range(10) if c != GREY)
_LEGEND_GAP_RANGE = (1, 3)
_SCENE_GAP = 3
_PLACEMENT_RETRIES = 200
_SHAPE_RETRIES = 120
_GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_coords(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _coords_to_mask(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = 1 + max(r for r, _ in coords)
    w = 1 + max(c for _, c in coords)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _legend_color_for_component(
    grid: np.ndarray, coords: list[tuple[int, int]], bg: int, grey: int
) -> int | None:
    h, w = map(int, grid.shape)
    r0, c0, r1, _ = _bbox(coords)
    if c0 < 2:
        return None
    coord_set = set(coords)
    colors: list[int] = []
    row_start = max(0, r0 - 1)
    row_stop = min(h - 2, r1) + 1
    for br in range(row_start, row_stop):
        block = grid[br : br + 2, c0 - 2 : c0]
        color = int(block[0, 0])
        if color in (bg, grey):
            continue
        if np.all(block == color) and any((rr, c0) in coord_set for rr in (br, br + 1)):
            colors.append(color)
    colors = list(dict.fromkeys(colors))
    return colors[0] if len(colors) == 1 else None


def _render_nested_rings(colors: list[int]) -> np.ndarray:
    assert colors, "at least one object is required"
    size = 2 * len(colors) - 1
    out = np.zeros((size, size), dtype=int)
    for layer, color in enumerate(colors):
        out[layer : size - layer, layer : size - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey objects using the left-edge legend, then sort them by area and
    render their colors as nested square rings.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    grey = GREY

    comps = _components(g == grey)
    assert comps, "expected grey components"

    legend: dict[tuple[tuple[int, int], ...], int] = {}
    object_shapes: list[tuple[tuple[int, int], ...]] = []

    for comp in comps:
        shape = _normalize_coords(comp)
        legend_color = _legend_color_for_component(g, comp, bg, grey)
        if legend_color is not None:
            assert shape not in legend, "legend shapes must be unique"
            legend[shape] = int(legend_color)
        else:
            object_shapes.append(shape)

    assert legend, "expected at least one legend entry"
    assert object_shapes, "expected at least one non-legend grey object"

    objects: list[tuple[int, int]] = []
    for shape in object_shapes:
        assert shape in legend, "every grey object must match a legend shape"
        objects.append((len(shape), int(legend[shape])))

    areas = [area for area, _ in objects]
    assert len(set(areas)) == len(areas), "object areas must be distinct"

    colors_in_order = [color for area, color in sorted(objects, key=lambda item: -item[0])]
    return _render_nested_rings(colors_in_order)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    if bool(_RNG.integers(2)):
        out = np.fliplr(out)
    out = np.rot90(out, int(_RNG.integers(4)))
    return _crop_mask(out)


def _random_shape(area: int) -> np.ndarray | None:
    for _ in range(_SHAPE_RETRIES):
        h = int(_RNG.integers(2, min(area, 6) + 1))
        w = int(_RNG.integers(2, min(area, 6) + 1))
        if h * w < area:
            continue
        start = (int(_RNG.integers(h)), int(_RNG.integers(w)))
        cells = {start}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in _NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(frontier[int(_RNG.integers(len(frontier)))])
        if len(cells) != area:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        mask = np.zeros((max(rows) - min(rows) + 1, max(cols) - min(cols) + 1), dtype=bool)
        for r, c in cells:
            mask[r - min(rows), c - min(cols)] = True
        mask = _random_transform(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if not np.any(mask[: min(2, mask.shape[0]), 0]):
            continue
        if area >= 4 and np.all(mask) and bool(_RNG.integers(4)):
            continue
        return mask
    return None


def _can_place(grid: np.ndarray, mask: np.ndarray, top: int, left: int, bg: int) -> bool:
    h, w = map(int, grid.shape)
    mh, mw = map(int, mask.shape)
    if top < 0 or left < 0 or top + mh > h or left + mw > w:
        return False
    rr0, rr1 = max(0, top - 1), min(h, top + mh + 1)
    cc0, cc1 = max(0, left - 1), min(w, left + mw + 1)
    return bool(np.all(grid[rr0:rr1, cc0:cc1] == bg))


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = map(int, mask.shape)
    patch = grid[top : top + h, left : left + w]
    patch[mask] = int(color)


def _expected_from_objects(objects: list[tuple[np.ndarray, int]]) -> np.ndarray:
    colors = [color for mask, color in sorted(objects, key=lambda item: -int(np.sum(item[0])))]
    return _render_nested_rings(colors)


def _gen_attempt() -> np.ndarray | None:
    bg_choices = [c for c in _PALETTE if c != GREY]
    bg = int(bg_choices[int(_RNG.integers(len(bg_choices)))])

    n_objects = int(_OBJECT_COUNT_OPTIONS[int(_RNG.integers(len(_OBJECT_COUNT_OPTIONS)))])
    areas = list(map(int, _RNG.choice(_AREA_POOL, size=n_objects, replace=False)))
    colors = list(map(int, _RNG.choice([c for c in _PALETTE if c != bg], size=n_objects, replace=False)))

    shape_masks: list[np.ndarray] = []
    seen_shapes: set[tuple[tuple[int, int], ...]] = set()
    for area in areas:
        mask = _random_shape(int(area))
        if mask is None:
            return None
        sig = _normalize_coords([(int(r), int(c)) for r, c in zip(*np.where(mask))])
        if sig in seen_shapes:
            return None
        seen_shapes.add(sig)
        shape_masks.append(mask)

    legend_order = list(range(n_objects))
    _RNG.shuffle(legend_order)
    object_order = list(range(n_objects))
    _RNG.shuffle(object_order)

    legend_gap = int(_RNG.integers(_LEGEND_GAP_RANGE[0], _LEGEND_GAP_RANGE[1] + 1))
    legend_entry_heights = [max(2, int(shape_masks[i].shape[0])) for i in legend_order]
    legend_h = sum(legend_entry_heights) + legend_gap * (n_objects - 1)
    legend_w = 2 + max(int(shape_masks[i].shape[1]) for i in legend_order)

    min_h = max(_GRID_H_RANGE[0], legend_h + 2)
    max_h = max(min_h, _GRID_H_RANGE[1])
    min_w = max(_GRID_W_RANGE[0], legend_w + _SCENE_GAP + 6)
    max_w = max(min_w, _GRID_W_RANGE[1])
    h = int(_RNG.integers(min_h, max_h + 1))
    w = int(_RNG.integers(min_w, max_w + 1))

    grid = np.full((h, w), bg, dtype=int)
    scene_c0 = legend_w + _SCENE_GAP

    row = int(_RNG.integers(0, h - legend_h + 1))
    for entry_h, idx in zip(legend_entry_heights, legend_order):
        mask = shape_masks[idx]
        color = colors[idx]
        _paint_mask(grid, np.ones((2, 2), dtype=bool), row, 0, color)
        _paint_mask(grid, mask, row, 2, GREY)
        row += entry_h + legend_gap

    placed_scene: list[tuple[np.ndarray, int]] = []
    for idx in object_order:
        mask = shape_masks[idx]
        color = colors[idx]
        mh, mw = map(int, mask.shape)
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            top = int(_RNG.integers(0, h - mh + 1))
            left = int(_RNG.integers(scene_c0, w - mw + 1))
            if not _can_place(grid, mask, top, left, bg):
                continue
            _paint_mask(grid, mask, top, left, GREY)
            placed_scene.append((mask, color))
            placed = True
            break
        if not placed:
            return None

    expected = _expected_from_objects(placed_scene)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
