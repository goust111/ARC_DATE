from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, colors_present, get_bg


def _find_divider(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    full_rows = [(r, int(grid[r, 0])) for r in range(h) if np.all(grid[r, :] == grid[r, 0])]
    full_cols = [(c, int(grid[0, c])) for c in range(w) if np.all(grid[:, c] == grid[0, c])]

    matches = [(r, c, row_color) for r, row_color in full_rows for c, col_color in full_cols if row_color == col_color]
    assert len(matches) == 1, "expected exactly one matching full-row/full-col divider"
    return matches[0]


def _extract_segment(line: np.ndarray, bg: int) -> tuple[int, int, int]:
    idx = np.flatnonzero(line != bg)
    assert idx.size > 0, "each rail must contain one colored segment"

    start = int(idx[0])
    end = int(idx[-1])
    color = int(line[start])
    assert color != bg, "segment color must differ from the background"
    assert np.all(line[start : end + 1] == color), "segment must be monochromatic"
    assert idx.size == end - start + 1, "segment must be contiguous"
    assert np.all(line[:start] == bg) and np.all(line[end + 1 :] == bg), "rail must contain exactly one segment"
    return color, start, end


def _parse_layers(grid: np.ndarray) -> tuple[int, list[dict[str, int]], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    div_r, div_c, divider_color = _find_divider(g)
    bg = get_bg(g)
    assert bg != divider_color, "divider color must differ from the background"
    assert np.all(g[div_r + 1 :, div_c + 1 :] == bg), "canvas area must start empty"

    top_rows = [r for r in range(div_r) if np.any(g[r, div_c + 1 :] != bg)]
    left_cols = [c for c in range(div_c) if np.any(g[div_r + 1 :, c] != bg)]
    assert top_rows and left_cols and len(top_rows) == len(left_cols), "top and left rails must exist in matching counts"
    assert np.all(g[:div_r, :div_c] == bg), "top-left corner block must be blank"

    top_rank = {row: rank for rank, row in enumerate(sorted(top_rows, reverse=True))}
    left_rank = {col: rank for rank, col in enumerate(sorted(left_cols, reverse=True))}

    top_segments: dict[int, tuple[int, int, int]] = {}
    for row in top_rows:
        assert np.all(g[row, :div_c] == bg), "top rails must only project into the canvas columns"
        color, c0, c1 = _extract_segment(g[row, div_c + 1 :], bg)
        assert color not in top_segments, "each top-rail color must be unique"
        top_segments[color] = (c0, c1, top_rank[row])

    left_segments: dict[int, tuple[int, int, int]] = {}
    for col in left_cols:
        assert np.all(g[:div_r, col] == bg), "left rails must only project into the canvas rows"
        color, r0, r1 = _extract_segment(g[div_r + 1 :, col], bg)
        assert color not in left_segments, "each left-rail color must be unique"
        left_segments[color] = (r0, r1, left_rank[col])

    assert set(top_segments) == set(left_segments), "top and left rails must use the same colors"
    layers: list[dict[str, int]] = []
    for color in colors_present(g, exclude={bg, divider_color}):
        c0, c1, top_layer = top_segments[color]
        r0, r1, left_layer = left_segments[color]
        assert top_layer == left_layer, "matching colors must occupy rails of the same distance rank"
        layers.append({"color": int(color), "r0": r0, "r1": r1, "c0": c0, "c1": c1, "rank": top_layer})

    return int(bg), layers, (g.shape[0] - div_r - 1, g.shape[1] - div_c - 1)


def _render_layers(
    canvas_shape: tuple[int, int],
    layers: list[dict[str, int]],
    bg: int,
    order: list[int] | None = None,
) -> np.ndarray:
    h, w = canvas_shape
    out = np.full((h, w), int(bg), dtype=int)

    rank_order = {layer["color"]: layer["rank"] for layer in layers}
    rects = {layer["color"]: layer for layer in layers}
    if order is None:
        order = [layer["color"] for layer in sorted(layers, key=lambda layer: layer["rank"], reverse=True)]

    for color in order:
        layer = rects[int(color)]
        out[layer["r0"] : layer["r1"] + 1, layer["c0"] : layer["c1"] + 1] = int(color)

    assert set(rank_order) == set(order), "render order must mention each layer exactly once"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct layered filled rectangles from one-dimensional rail projections.

    Input structure:
    - One full row and one full column of the same divider color split the grid into a top-left blank corner,
      upper and left rail zones, and an empty bottom-right canvas.
    - Each top rail row contains exactly one colored horizontal segment over the canvas columns.
    - Each left rail column contains exactly one same-colored vertical segment over the canvas rows.

    Rule:
    - Pair the top and left segments by color to recover each rectangle's column and row span.
    - Use the rail distance to the canvas, not the numeric color id, as the layer order: rails closer to the canvas
      are painted later and therefore cover farther rails in overlaps.
    - Draw all recovered filled rectangles on the empty bottom-right canvas and return only that canvas.

    Key insight:
    - The rectangles are never shown directly in the input; only their two orthogonal projections appear, while the
      rail positions encode the occlusion stack needed to resolve overlaps correctly.
    """
    bg, layers, canvas_shape = _parse_layers(grid)
    return _render_layers(canvas_shape, layers, bg)


TRACK_COUNT_RANGE = (3, 5)
CANVAS_SIDE_MIN = 7
CANVAS_SIDE_MAX = 20
RECT_SIDE_MIN = 2
RECT_SIDE_MAX_RATIO = 0.78
RECT_SIDE_MIN_RATIO = 0.28
MAX_RECT_RETRIES = 160
GEN_RETRIES = 400


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _sample_rect(h: int, w: int, focus: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
    min_h = min(h, max(RECT_SIDE_MIN, int(np.ceil(h * RECT_SIDE_MIN_RATIO))))
    min_w = min(w, max(RECT_SIDE_MIN, int(np.ceil(w * RECT_SIDE_MIN_RATIO))))
    max_h = max(min_h, min(h, max(RECT_SIDE_MIN, int(np.floor(h * RECT_SIDE_MAX_RATIO)))))
    max_w = max(min_w, min(w, max(RECT_SIDE_MIN, int(np.floor(w * RECT_SIDE_MAX_RATIO)))))

    rect_h = random.randint(min_h, max_h)
    rect_w = random.randint(min_w, max_w)

    if focus is None:
        r0 = random.randint(0, h - rect_h)
        c0 = random.randint(0, w - rect_w)
    else:
        fr, fc = focus
        r0_lo = max(0, fr - rect_h + 1)
        r0_hi = min(fr, h - rect_h)
        c0_lo = max(0, fc - rect_w + 1)
        c0_hi = min(fc, w - rect_w)
        r0 = random.randint(r0_lo, r0_hi)
        c0 = random.randint(c0_lo, c0_hi)
    return (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)


def _render_rect_specs(
    rect_specs: list[dict[str, int]],
    h: int,
    w: int,
    bg: int,
    order: list[int] | None = None,
) -> np.ndarray:
    return _render_layers((h, w), rect_specs, bg, order=order)


def _sample_rect_specs(colors: list[int], h: int, w: int) -> list[dict[str, int]] | None:
    focus = (random.randrange(h), random.randrange(w))
    overlap_indices = set(random.sample(range(len(colors)), random.randint(2, len(colors))))

    for _ in range(MAX_RECT_RETRIES):
        rects: list[tuple[int, int, int, int]] = []
        rect_set: set[tuple[int, int, int, int]] = set()
        for idx in range(len(colors)):
            rect = _sample_rect(h, w, focus if idx in overlap_indices else None)
            if rect in rect_set:
                break
            rect_set.add(rect)
            rects.append(rect)
        if len(rects) != len(colors):
            continue

        coverage = np.zeros((h, w), dtype=int)
        for r0, c0, r1, c1 in rects:
            coverage[r0 : r1 + 1, c0 : c1 + 1] += 1

        if int(coverage.max()) < 2:
            continue
        overlap_cells = int(np.count_nonzero(coverage >= 2))
        if overlap_cells > int(0.35 * h * w):
            continue

        areas = [_rect_area(rect) for rect in rects]
        if areas == sorted(areas) or areas == sorted(areas, reverse=True):
            continue

        rect_specs = []
        for order_idx, (color, rect) in enumerate(zip(colors, rects, strict=True)):
            r0, c0, r1, c1 = rect
            rect_specs.append(
                {
                    "color": int(color),
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "rank": len(colors) - 1 - order_idx,
                }
            )

        return rect_specs
    return None


def _build_input(
    rect_specs: list[dict[str, int]],
    bg: int,
    divider_color: int,
    h: int,
    w: int,
) -> np.ndarray:
    n = len(rect_specs)
    grid = np.full((n + 1 + h, n + 1 + w), int(bg), dtype=int)
    grid[n, :] = int(divider_color)
    grid[:, n] = int(divider_color)

    for rank, layer in enumerate(sorted(rect_specs, key=lambda layer: layer["rank"], reverse=True)):
        color = int(layer["color"])
        grid[rank, n + 1 + layer["c0"] : n + 1 + layer["c1"] + 1] = color
        grid[n + 1 + layer["r0"] : n + 1 + layer["r1"] + 1, rank] = color
    return grid


def _semantic_ok(inp: np.ndarray, rect_specs: list[dict[str, int]], bg: int) -> bool:
    n = len(rect_specs)
    h = inp.shape[0] - n - 1
    w = inp.shape[1] - n - 1
    expected = _render_rect_specs(rect_specs, h, w, bg)

    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    visible_colors = colors_present(actual, exclude={bg})
    if visible_colors != sorted(layer["color"] for layer in rect_specs):
        return False

    saw_partial_occlusion = False
    for layer in rect_specs:
        area = (layer["r1"] - layer["r0"] + 1) * (layer["c1"] - layer["c0"] + 1)
        visible = int(np.count_nonzero(actual == layer["color"]))
        if visible < 2 or visible > area:
            return False
        saw_partial_occlusion |= visible < area
    if not saw_partial_occlusion:
        return False

    correct_order = [layer["color"] for layer in sorted(rect_specs, key=lambda layer: layer["rank"], reverse=True)]
    asc_by_color = sorted(correct_order)
    desc_by_color = sorted(correct_order, reverse=True)
    if np.array_equal(_render_rect_specs(rect_specs, h, w, bg, order=asc_by_color), expected):
        return False
    if np.array_equal(_render_rect_specs(rect_specs, h, w, bg, order=desc_by_color), expected):
        return False
    if np.array_equal(_render_rect_specs(rect_specs, h, w, bg, order=list(reversed(correct_order))), expected):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        n = random.randint(*TRACK_COUNT_RANGE)
        max_canvas_h = min(CANVAS_SIDE_MAX, 30 - (n + 1))
        max_canvas_w = min(CANVAS_SIDE_MAX, 30 - (n + 1))
        if max_canvas_h < CANVAS_SIDE_MIN or max_canvas_w < CANVAS_SIDE_MIN:
            continue

        h = random.randint(CANVAS_SIDE_MIN, max_canvas_h)
        w = random.randint(CANVAS_SIDE_MIN, max_canvas_w)

        bg = Color.BLACK
        palette = [color for color in Color.NOT_BLACK]
        divider_color = random.choice(palette)
        rect_colors = random.sample([color for color in palette if color != divider_color], n)
        random.shuffle(rect_colors)

        rect_specs = _sample_rect_specs(rect_colors, h, w)
        if rect_specs is None:
            continue

        inp = _build_input(rect_specs, bg, divider_color, h, w)
        if _semantic_ok(inp, rect_specs, bg):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 8")
