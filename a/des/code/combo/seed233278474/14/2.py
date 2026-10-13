import random

import numpy as np


BG = 0
ARROW = 7
FRAME_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)
EDGES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400
MAX_GRID_SIZE = 30


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    if pts.size == 0:
        raise ValueError("empty mask")
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paint_frame(grid: np.ndarray, color: int, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _border_mask(shape: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _detect_arrow_edge(grid: np.ndarray) -> str:
    h, w = grid.shape
    pts = np.argwhere(grid == ARROW)
    assert pts.shape[0] == 3
    edges = set()
    if np.any(pts[:, 0] == 0):
        edges.add("top")
    if np.any(pts[:, 0] == h - 1):
        edges.add("bottom")
    if np.any(pts[:, 1] == 0):
        edges.add("left")
    if np.any(pts[:, 1] == w - 1):
        edges.add("right")
    assert len(edges) == 1
    return next(iter(edges))


def _extract_layers(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    remaining = (grid != BG) & (grid != ARROW)
    assert remaining.any()

    layers: list[dict[str, int | tuple[int, int, int, int]]] = []
    center2_r = None
    center2_c = None

    while remaining.any():
        r0, c0, r1, c1 = _bbox(remaining)
        border = _border_mask(grid.shape, r0, c0, r1, c1)
        vals = grid[border]
        assert np.all((vals != BG) & (vals != ARROW))
        colors = {int(v) for v in vals}
        assert len(colors) == 1
        color = next(iter(colors))

        if center2_r is None:
            center2_r = r0 + r1
            center2_c = c0 + c1
        else:
            assert center2_r == r0 + r1
            assert center2_c == c0 + c1

        layers.append({"color": color, "bbox": (r0, c0, r1, c1)})
        remaining[border] = False

    return layers


def _stack_layers(
    shape: tuple[int, int],
    edge: str,
    layers: list[dict[str, int | tuple[int, int, int, int]]],
) -> np.ndarray:
    h, w = shape
    outer_r0, outer_c0, outer_r1, outer_c1 = layers[0]["bbox"]  # type: ignore[index]
    center2_r = int(outer_r0) + int(outer_r1)
    center2_c = int(outer_c0) + int(outer_c1)

    out = np.full(shape, BG, dtype=int)

    if edge == "top":
        cursor = 0
        for layer in layers:
            r0, c0, r1, c1 = layer["bbox"]  # type: ignore[index]
            fh = int(r1) - int(r0) + 1
            fw = int(c1) - int(c0) + 1
            top = cursor
            left = (center2_c - fw + 1) // 2
            assert 0 <= left and left + fw <= w and top + fh <= h
            _paint_frame(out, int(layer["color"]), top, left, top + fh - 1, left + fw - 1)
            cursor += fh
    elif edge == "bottom":
        cursor = h
        for layer in layers:
            r0, c0, r1, c1 = layer["bbox"]  # type: ignore[index]
            fh = int(r1) - int(r0) + 1
            fw = int(c1) - int(c0) + 1
            top = cursor - fh
            left = (center2_c - fw + 1) // 2
            assert 0 <= left and left + fw <= w and 0 <= top
            _paint_frame(out, int(layer["color"]), top, left, top + fh - 1, left + fw - 1)
            cursor = top
    elif edge == "left":
        cursor = 0
        for layer in layers:
            r0, c0, r1, c1 = layer["bbox"]  # type: ignore[index]
            fh = int(r1) - int(r0) + 1
            fw = int(c1) - int(c0) + 1
            top = (center2_r - fh + 1) // 2
            left = cursor
            assert 0 <= top and top + fh <= h and left + fw <= w
            _paint_frame(out, int(layer["color"]), top, left, top + fh - 1, left + fw - 1)
            cursor += fw
    else:
        assert edge == "right"
        cursor = w
        for layer in layers:
            r0, c0, r1, c1 = layer["bbox"]  # type: ignore[index]
            fh = int(r1) - int(r0) + 1
            fw = int(c1) - int(c0) + 1
            top = (center2_r - fh + 1) // 2
            left = cursor - fw
            assert 0 <= top and top + fh <= h and 0 <= left
            _paint_frame(out, int(layer["color"]), top, left, top + fh - 1, left + fw - 1)
            cursor = left

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the border arrow, peel the concentric rectangular frames, and stack the
    layers against the indicated edge while preserving each layer's original size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    edge = _detect_arrow_edge(g)
    layers = _extract_layers(g)
    return _stack_layers(g.shape, edge, layers)


def _sample_central_offset(total: int, size: int, rng: random.Random) -> int | None:
    lo = 2
    hi = total - size - 2
    if lo > hi:
        return None
    center = (total - size) // 2
    band_lo = max(lo, center - 3)
    band_hi = min(hi, center + 3)
    return rng.randint(band_lo, band_hi)


def _draw_arrow(grid: np.ndarray, edge: str, rng: random.Random) -> None:
    h, w = grid.shape
    if edge == "top":
        c = rng.randint(2, w - 3)
        cells = [(0, c - 1), (1, c), (0, c + 1)]
    elif edge == "bottom":
        c = rng.randint(2, w - 3)
        cells = [(h - 1, c - 1), (h - 2, c), (h - 1, c + 1)]
    elif edge == "left":
        r = rng.randint(2, h - 3)
        cells = [(r - 1, 0), (r, 1), (r + 1, 0)]
    else:
        r = rng.randint(2, h - 3)
        cells = [(r - 1, w - 1), (r, w - 2), (r + 1, w - 1)]

    for rr, cc in cells:
        assert grid[rr, cc] == BG
        grid[rr, cc] = ARROW


def _make_attempt(rng: random.Random) -> np.ndarray | None:
    edge = rng.choice(EDGES)
    layers = rng.randint(2, 4)

    inner_h = rng.randint(3, 6)
    inner_w = rng.randint(3, 7)
    outer_h = inner_h + 2 * (layers - 1)
    outer_w = inner_w + 2 * (layers - 1)

    heights = [outer_h - 2 * i for i in range(layers)]
    widths = [outer_w - 2 * i for i in range(layers)]
    stack_h = sum(heights)
    stack_w = sum(widths)

    if edge in ("top", "bottom"):
        min_h = max(stack_h, outer_h + 4)
        max_h = min(MAX_GRID_SIZE, min_h + 6)
        min_w = outer_w + 4
        max_w = min(MAX_GRID_SIZE, outer_w + 10)
    else:
        min_h = outer_h + 4
        max_h = min(MAX_GRID_SIZE, outer_h + 10)
        min_w = max(stack_w, outer_w + 4)
        max_w = min(MAX_GRID_SIZE, min_w + 6)

    if min_h > max_h or min_w > max_w:
        return None

    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)

    top = _sample_central_offset(h, outer_h, rng)
    left = _sample_central_offset(w, outer_w, rng)
    if top is None or left is None:
        return None

    colors = rng.sample(FRAME_COLORS, layers)
    grid = np.full((h, w), BG, dtype=int)

    for i, color in enumerate(colors):
        r0 = top + i
        c0 = left + i
        r1 = top + outer_h - 1 - i
        c1 = left + outer_w - 1 - i
        _paint_frame(grid, color, r0, c0, r1, c1)

    _draw_arrow(grid, edge, rng)

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if np.count_nonzero(grid == ARROW) != 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _make_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
