import random
from collections import deque

import numpy as np


BG_COLOR = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 300
MAX_GRID_SIZE = 30


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(pts)
    return parts


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _component_info(points: list[tuple[int, int]]) -> dict:
    r0, c0, r1, c1 = _bbox(points)
    return {
        "points": points,
        "bbox": (r0, c0, r1, c1),
        "height": r1 - r0 + 1,
        "width": c1 - c0 + 1,
        "area": len(points),
    }


def _is_horizontal_strip(info: dict) -> bool:
    return info["height"] == 1 and info["width"] >= 2 and info["area"] == info["width"]


def _classify_rectangular_component(info: dict) -> str | None:
    h = int(info["height"])
    w = int(info["width"])
    area = int(info["area"])
    r0, c0, _, _ = info["bbox"]

    mask = np.zeros((h, w), dtype=bool)
    for r, c in info["points"]:
        mask[r - r0, c - c0] = True

    if area == h * w and bool(mask.all()):
        return "solid"

    if h < 3 or w < 3:
        return None

    if area != 2 * h + 2 * w - 4:
        return None

    if not np.all(mask[0, :]) or not np.all(mask[-1, :]):
        return None
    if not np.all(mask[:, 0]) or not np.all(mask[:, -1]):
        return None
    if bool(mask[1:-1, 1:-1].any()):
        return None
    return "frame"


def _extract_scene(
    grid: np.ndarray,
) -> tuple[list[tuple[tuple[int, int, int, int], str]], list[int], int]:
    bg = _most_common_color(grid)
    infos = [_component_info(points) for points in _components(grid != bg)]

    strip_candidates = [info for info in infos if _is_horizontal_strip(info)]
    assert len(strip_candidates) == 1, "expected exactly one horizontal color strip"
    strip_info = strip_candidates[0]

    sr0, sc0, _, sc1 = strip_info["bbox"]
    strip_colors = [int(grid[sr0, c]) for c in range(sc0, sc1 + 1)]

    layers: list[tuple[tuple[int, int, int, int], str]] = []
    for info in infos:
        if info is strip_info:
            continue
        kind = _classify_rectangular_component(info)
        assert kind is not None, "non-strip components must be rectangular frames or a solid core"
        layers.append((info["bbox"], kind))

    assert len(layers) >= 2, "expected at least one frame and one solid core"
    layers.sort(
        key=lambda item: (
            (item[0][2] - item[0][0] + 1) * (item[0][3] - item[0][1] + 1),
            item[0][0],
            item[0][1],
        ),
        reverse=True,
    )

    for (outer, outer_kind), (inner, _) in zip(layers, layers[1:]):
        assert outer_kind == "frame", "all non-innermost layers must be frames"
        assert (
            inner[0] == outer[0] + 2
            and inner[1] == outer[1] + 2
            and inner[2] == outer[2] - 2
            and inner[3] == outer[3] - 2
        ), "nested layers must be separated by a one-cell background gap"

    assert layers[-1][1] == "solid", "innermost layer must be solid"
    assert len(strip_colors) >= len(layers), "strip must provide enough recoloring entries"
    return layers, strip_colors, bg


def _paint_rectangular_layer(
    canvas: np.ndarray, col0: int, height: int, width: int, color: int, kind: str
) -> None:
    if kind == "solid":
        canvas[:height, col0 : col0 + width] = color
        return

    canvas[0, col0 : col0 + width] = color
    canvas[height - 1, col0 : col0 + width] = color
    canvas[:height, col0] = color
    canvas[:height, col0 + width - 1] = color


def _build_output(
    layers: list[tuple[tuple[int, int, int, int], str]], strip_colors: list[int], bg: int
) -> np.ndarray:
    dims = [
        (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1, kind)
        for bbox, kind in layers
    ]
    out_h = max(h for h, _, _ in dims)
    out_w = sum(w for _, w, _ in dims) + len(dims) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col0 = 0
    for i, (height, width, kind) in enumerate(dims):
        _paint_rectangular_layer(out, col0, height, width, int(strip_colors[i]), kind)
        col0 += width + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unnest a stack of rectangular frames and recolor each depth from a horizontal strip.
    """
    g = np.asarray(grid, dtype=int)
    layers, strip_colors, bg = _extract_scene(g)
    return _build_output(layers, strip_colors, bg)


def _draw_nested_structure(
    grid: np.ndarray,
    top: int,
    left: int,
    layer_count: int,
    inner_h: int,
    inner_w: int,
    layer_colors: list[int],
) -> list[tuple[tuple[int, int, int, int], str]]:
    outer_h = inner_h + 4 * (layer_count - 1)
    outer_w = inner_w + 4 * (layer_count - 1)
    layers: list[tuple[tuple[int, int, int, int], str]] = []

    for depth in range(layer_count):
        r0 = top + 2 * depth
        c0 = left + 2 * depth
        height = outer_h - 4 * depth
        width = outer_w - 4 * depth
        color = int(layer_colors[depth])
        bbox = (r0, c0, r0 + height - 1, c0 + width - 1)

        if depth == layer_count - 1:
            grid[r0 : r0 + height, c0 : c0 + width] = color
            layers.append((bbox, "solid"))
        else:
            grid[r0, c0 : c0 + width] = color
            grid[r0 + height - 1, c0 : c0 + width] = color
            grid[r0 : r0 + height, c0] = color
            grid[r0 : r0 + height, c0 + width - 1] = color
            layers.append((bbox, "frame"))

    return layers


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layer_count = rng.randint(2, 5)
    inner_h = rng.randint(2, 4)
    inner_w = rng.randint(2, 4)
    outer_h = inner_h + 4 * (layer_count - 1)
    outer_w = inner_w + 4 * (layer_count - 1)

    extra_strip = rng.randint(0, 2)
    strip_len = layer_count + extra_strip

    layer_colors = rng.sample(PALETTE, layer_count)
    strip_colors = rng.sample(PALETTE, strip_len)

    place_right = bool(rng.randrange(2))
    grid: np.ndarray
    expected_layers: list[tuple[tuple[int, int, int, int], str]]

    if place_right:
        top = rng.randint(0, 2)
        left = rng.randint(0, 2)
        gap_x = rng.randint(2, 3)
        bottom = rng.randint(0, 2)
        right = rng.randint(0, 2)
        height = top + outer_h + bottom
        width = left + outer_w + gap_x + strip_len + right
        if height > MAX_GRID_SIZE or width > MAX_GRID_SIZE:
            return None

        grid = np.full((height, width), BG_COLOR, dtype=int)
        expected_layers = _draw_nested_structure(
            grid, top, left, layer_count, inner_h, inner_w, layer_colors
        )
        strip_row = top + rng.randint(0, outer_h - 1)
        strip_col = left + outer_w + gap_x
        grid[strip_row, strip_col : strip_col + strip_len] = strip_colors
    else:
        top = rng.randint(0, 2)
        left = rng.randint(0, 2)
        strip_left = rng.randint(0, 2)
        gap_y = rng.randint(2, 3)
        bottom = rng.randint(0, 2)
        right = rng.randint(0, 2)
        height = top + outer_h + gap_y + 1 + bottom
        width = max(left + outer_w, strip_left + strip_len) + right
        if height > MAX_GRID_SIZE or width > MAX_GRID_SIZE:
            return None

        grid = np.full((height, width), BG_COLOR, dtype=int)
        expected_layers = _draw_nested_structure(
            grid, top, left, layer_count, inner_h, inner_w, layer_colors
        )
        strip_row = top + outer_h + gap_y
        grid[strip_row, strip_left : strip_left + strip_len] = strip_colors

    expected = _build_output(expected_layers, strip_colors, BG_COLOR)
    parsed_layers, parsed_strip, parsed_bg = _extract_scene(grid)
    if not np.array_equal(_build_output(parsed_layers, parsed_strip, parsed_bg), expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
