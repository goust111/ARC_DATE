import numpy as np
from collections import Counter


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _normalize_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    return max(rows) + 1, max(cols) + 1


def _shape_array(shape: tuple[tuple[int, int], ...], color: int, bg: int = 0) -> np.ndarray:
    h, w = _shape_dims(shape)
    arr = np.full((h, w), bg, dtype=int)
    for r, c in shape:
        arr[r, c] = color
    return arr


def _separator_runs(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]]]:
    cols: list[int] = []
    colors: list[int] = []
    for c in range(grid.shape[1]):
        value = int(grid[0, c])
        if value != bg and np.all(grid[:, c] == value):
            cols.append(c)
            colors.append(value)
    assert cols, "expected full-height separator columns"
    assert len(set(colors)) == 1, "separator columns should share one color"

    runs: list[tuple[int, int]] = []
    start = cols[0]
    prev = cols[0]
    for c in cols[1:]:
        if c == prev + 1:
            prev = c
            continue
        runs.append((start, prev))
        start = prev = c
    runs.append((start, prev))
    return colors[0], runs


def _split_panels(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    _sep_color, runs = _separator_runs(grid, bg)
    bounds: list[tuple[int, int]] = []
    left = 0
    for start, end in runs:
        bounds.append((left, start - 1))
        left = end + 1
    bounds.append((left, grid.shape[1] - 1))

    assert all(lo <= hi for lo, hi in bounds), "invalid panel bounds"
    widths = [hi - lo + 1 for lo, hi in bounds]
    assert 3 <= len(bounds) <= 4, "expected 3-4 panels"
    assert len(set(widths)) == 1, "panels should have equal width"
    return [grid[:, lo : hi + 1] for lo, hi in bounds]


def _panel_outlier(panel: np.ndarray, bg: int) -> np.ndarray:
    parts = _components(panel != bg)
    assert 3 <= len(parts) <= 4, "expected 3-4 objects per panel"

    objects = []
    for coords in parts:
        colors = {int(panel[r, c]) for r, c in coords}
        assert len(colors) == 1, "each object should be monochrome"
        r0, c0, r1, c1 = _bbox(coords)
        crop = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        for r, c in coords:
            crop[r - r0, c - c0] = int(panel[r, c])
        objects.append(
            {
                "shape": _normalize_shape(coords),
                "crop": crop,
            }
        )

    counts = Counter(obj["shape"] for obj in objects)
    assert len(counts) == 2, "expected one majority shape and one outlier shape"
    odd_shapes = [shape for shape, count in counts.items() if count == 1]
    assert len(odd_shapes) == 1, "expected exactly one outlier object"
    assert counts[odd_shapes[0]] == 1
    assert sum(count == len(objects) - 1 for count in counts.values()) == 1

    for obj in objects:
        if obj["shape"] == odd_shapes[0]:
            return obj["crop"]
    raise AssertionError("outlier object not found")


def _stack_crops(crops: list[np.ndarray], bg: int) -> np.ndarray:
    out_h = sum(crop.shape[0] for crop in crops) + len(crops) - 1
    out_w = max(crop.shape[1] for crop in crops)
    out = np.full((out_h, out_w), bg, dtype=int)
    row = 0
    for crop in crops:
        h, w = crop.shape
        out[row : row + h, :w] = crop
        row += h + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Collect the shape-outlier object from each separated panel into one vertical stack.

    Input structure:
    - The background is black.
    - Full-height grey separator columns divide the grid into 3-4 equal-width vertical panels.
    - Each panel contains 3-4 disjoint 4-connected monochrome objects, with exactly one object whose translation-normalized shape differs from the others.

    Rule:
    - Split the grid at the grey separator columns to recover the individual panels.
    - In each panel, compare object shapes after normalizing away translation and find the unique outlier shape.
    - Crop each outlier to its bounding box, then stack those crops from top to bottom in left-to-right panel order with one black row between consecutive crops.

    Key insight:
    - The separator columns isolate independent "odd one out" shape-comparison subproblems, and the output is the ordered collection of the outlier objects rather than any transformed panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = 0
    panels = _split_panels(g, bg)
    crops = [_panel_outlier(panel, bg) for panel in panels]
    return _stack_crops(crops, bg)


_RNG = np.random.default_rng()
_BG = 0
_SEP = 5
_OBJ_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_MAX_ATTEMPTS = 300
_SEP_W = 1
_MIN_GAP = 1

_BASE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
)


def _variants(base: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    arr = np.array(base, dtype=int)
    out: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for flip in (False, True):
        cur = arr.copy()
        if flip:
            cur[:, 1] *= -1
        for _ in range(4):
            shape = _normalize_shape(cur)
            if shape not in seen:
                seen.add(shape)
                out.append(shape)
            cur = np.stack([cur[:, 1], -cur[:, 0]], axis=1)
    return out


_SHAPES = tuple(
    dict.fromkeys(shape for base in _BASE_SHAPES for shape in _variants(base))
)


def _fit_shapes(panel_w: int) -> list[tuple[tuple[int, int], ...]]:
    max_w = panel_w - 2
    return [
        shape
        for shape in _SHAPES
        if _shape_dims(shape)[1] <= max_w and _shape_dims(shape)[0] <= 4
    ]


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(lo, hi + 1))


def _random_gaps(extra: int, parts: int) -> list[int]:
    if extra <= 0:
        return [0] * parts
    return [int(x) for x in _RNG.multinomial(extra, np.full(parts, 1.0 / parts))]


def _panel_plan(panel_w: int) -> dict | None:
    choices = _fit_shapes(panel_w)
    if len(choices) < 2:
        return None

    n_objs = _randint(3, 4)
    picked = _RNG.choice(len(choices), size=2, replace=False)
    majority = choices[int(picked[0])]
    anomaly = choices[int(picked[1])]
    shapes = [majority] * (n_objs - 1) + [anomaly]
    _RNG.shuffle(shapes)

    colors = [int(_RNG.choice(_OBJ_COLORS)) for _ in range(n_objs)]
    objects = []
    anomaly_crop = None
    for shape, color in zip(shapes, colors):
        h, w = _shape_dims(shape)
        objects.append({"shape": shape, "color": color, "h": h, "w": w})
        if shape == anomaly:
            anomaly_crop = _shape_array(shape, color, _BG)

    assert anomaly_crop is not None
    needed_h = sum(obj["h"] for obj in objects) + _MIN_GAP * (n_objs + 1)
    return {"objects": objects, "needed_h": needed_h, "outlier": anomaly_crop}


def _expected_output(crops: list[np.ndarray]) -> np.ndarray:
    return _stack_crops(crops, _BG)


def _draw_panel(grid: np.ndarray, start_c: int, panel_w: int, panel_h: int, plan: dict) -> None:
    objects = plan["objects"]
    extra = panel_h - sum(obj["h"] for obj in objects) - _MIN_GAP * (len(objects) + 1)
    assert extra >= 0
    extras = _random_gaps(extra, len(objects) + 1)
    gaps = [_MIN_GAP + x for x in extras]

    row = gaps[0]
    for idx, obj in enumerate(objects):
        max_x = panel_w - obj["w"] - 1
        min_x = 1
        assert min_x <= max_x, "object must fit with side margins"
        col = _randint(min_x, max_x)
        for sr, sc in obj["shape"]:
            grid[row + sr, start_c + col + sc] = obj["color"]
        row += obj["h"] + gaps[idx + 1]


def _gen_attempt() -> np.ndarray | None:
    n_panels = int(_RNG.choice((3, 4)))
    if n_panels == 3:
        panel_w = int(_RNG.choice((6, 7, 8)))
    else:
        panel_w = int(_RNG.choice((5, 6)))

    total_w = n_panels * panel_w + (n_panels - 1) * _SEP_W
    assert total_w <= 30

    plans = []
    for _ in range(n_panels):
        plan = _panel_plan(panel_w)
        if plan is None:
            return None
        plans.append(plan)

    panel_h = max(plan["needed_h"] for plan in plans) + _randint(0, 2)
    if panel_h > 30:
        return None

    grid = np.full((panel_h, total_w), _BG, dtype=int)
    expected_crops = [plan["outlier"] for plan in plans]

    for i in range(n_panels - 1):
        sep_c = (i + 1) * panel_w + i * _SEP_W
        grid[:, sep_c : sep_c + _SEP_W] = _SEP

    for i, plan in enumerate(plans):
        start_c = i * (panel_w + _SEP_W)
        _draw_panel(grid, start_c, panel_w, panel_h, plan)

    expected = _expected_output(expected_crops)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
