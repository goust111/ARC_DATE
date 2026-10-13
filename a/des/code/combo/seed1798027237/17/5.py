from __future__ import annotations

import numpy as np


_RNG = np.random.default_rng()

# Layer and canvas sampling.
MIN_LAYERS = 3
MAX_LAYERS = 5
INNER_SIZE_MIN = 1
INNER_SIZE_MAX = 4
CANVAS_MAX = 30
CANVAS_SLACK = 6

# Retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    winners = values[counts == max_count]
    assert len(winners) == 1, "background color must be uniquely dominant"
    return int(winners[0])


def _bbox_of_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == color)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_in_bbox(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1] == color, dtype=bool)


def _is_border_mask(mask: np.ndarray) -> bool:
    h, w = mask.shape
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(mask, border)


def _layer_infos(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        bbox = _bbox_of_color(grid, color)
        mask = _mask_in_bbox(grid, color, bbox)
        if mask.all():
            kind = "solid"
        else:
            assert _is_border_mask(mask), "each non-background color must form a solid rectangle or a 1-cell ring"
            kind = "ring"

        infos.append(
            {
                "color": color,
                "bbox": bbox,
                "kind": kind,
                "count": int(mask.sum()),
            }
        )

    assert infos, "expected at least one non-background layer"
    infos.sort(key=lambda info: info["bbox"])  # outer layers start earlier and end later

    outer_bbox = infos[0]["bbox"]
    assert isinstance(outer_bbox, tuple)
    r0, c0, r1, c1 = outer_bbox
    outer_patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert not np.any(outer_patch == bg), "nested structure must fully fill its outer bounding box"

    outside_mask = np.ones_like(grid, dtype=bool)
    outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    assert not np.any(grid[outside_mask] != bg), "all foreground cells must belong to the nested structure"

    for i, info in enumerate(infos[:-1]):
        curr = info["bbox"]
        nxt = infos[i + 1]["bbox"]
        assert info["kind"] == "ring", "every non-final layer must be a 1-cell ring"
        assert nxt == (curr[0] + 1, curr[1] + 1, curr[2] - 1, curr[3] - 1), "layers must be immediately nested"

    assert infos[-1]["kind"] == "solid", "innermost layer must be solid"
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unroll nested rectangular rings into left-aligned horizontal color bands.

    Input structure:
    - A dominant background surrounds one concentric multi-layer rectangle.
    - Each non-final layer is a 1-cell-thick monochrome rectangular ring.
    - The innermost layer is a solid monochrome rectangle.

    Rule:
    - Read layers from outermost to innermost.
    - Output one row per layer.
    - In row i, paint the layer color for exactly as many cells as that layer has pixels.
    - The output width equals the pixel count of the outermost ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    infos = _layer_infos(g, bg)

    out_h = len(infos)
    out_w = int(infos[0]["count"])
    out = np.full((out_h, out_w), bg, dtype=int)

    for row, info in enumerate(infos):
        count = int(info["count"])
        out[row, :count] = int(info["color"])

    return out


def _ring_cell_count(h: int, w: int) -> int:
    if h == 1 or w == 1:
        return h * w
    return 2 * (h + w) - 4


def _paint_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _expected_output(bg: int, colors: list[int], inner_h: int, inner_w: int) -> np.ndarray:
    n_layers = len(colors)
    counts = []
    for i, color in enumerate(colors):
        h = inner_h + 2 * (n_layers - 1 - i)
        w = inner_w + 2 * (n_layers - 1 - i)
        count = inner_h * inner_w if i == n_layers - 1 else _ring_cell_count(h, w)
        counts.append((color, count))

    out = np.full((n_layers, counts[0][1]), bg, dtype=int)
    for r, (color, count) in enumerate(counts):
        out[r, :count] = color
    return out


def _centered_start(span: int, size: int) -> int:
    base = (span - size) // 2
    jitter = min(2, base, span - size - base)
    if jitter <= 0:
        return base
    return int(base + _RNG.integers(-jitter, jitter + 1))


def _gen_attempt() -> np.ndarray | None:
    n_layers = int(_RNG.integers(MIN_LAYERS, MAX_LAYERS + 1))
    inner_h = int(_RNG.integers(INNER_SIZE_MIN, INNER_SIZE_MAX + 1))
    inner_w = int(_RNG.integers(INNER_SIZE_MIN, INNER_SIZE_MAX + 1))

    outer_h = inner_h + 2 * (n_layers - 1)
    outer_w = inner_w + 2 * (n_layers - 1)

    extra_h_max = min(outer_h + CANVAS_SLACK, CANVAS_MAX - outer_h)
    extra_w_max = min(outer_w + CANVAS_SLACK, CANVAS_MAX - outer_w)
    if extra_h_max < outer_h or extra_w_max < outer_w:
        return None

    grid_h = outer_h + int(_RNG.integers(outer_h, extra_h_max + 1))
    grid_w = outer_w + int(_RNG.integers(outer_w, extra_w_max + 1))

    bg_and_layers = _RNG.choice(np.arange(10), size=n_layers + 1, replace=False)
    bg = int(bg_and_layers[0])
    colors = [int(color) for color in bg_and_layers[1:]]

    r0 = _centered_start(grid_h, outer_h)
    c0 = _centered_start(grid_w, outer_w)
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    for i, color in enumerate(colors):
        inset = i
        bbox = (r0 + inset, c0 + inset, r0 + outer_h - 1 - inset, c0 + outer_w - 1 - inset)
        if i == n_layers - 1:
            br0, bc0, br1, bc1 = bbox
            grid[br0 : br1 + 1, bc0 : bc1 + 1] = color
        else:
            _paint_ring(grid, bbox, color)

    bg_count = int(np.count_nonzero(grid == bg))
    if bg_count <= outer_h * outer_w:
        return None

    expected = _expected_output(bg, colors, inner_h, inner_w)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring input")
