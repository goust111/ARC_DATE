from __future__ import annotations

from itertools import combinations

import numpy as np


BG_COLOR = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 600
K_RANGE = (2, 5)
CANVAS_H_RANGE = (6, 15)
CANVAS_W_RANGE = (6, 15)
OVERLAP_PROBABILITY = 0.7
VISIBLE_COVERAGE_RANGE = (0.18, 0.78)


def _parse_single_run(values: np.ndarray, bg: int) -> tuple[int, int, int]:
    idx = np.flatnonzero(values != bg)
    assert idx.size > 0, "expected exactly one colored run"
    start = int(idx[0])
    end = int(idx[-1])
    run = values[start : end + 1]
    color = int(run[0])
    assert color != bg, "run color must differ from background"
    assert idx.size == end - start + 1, "run must be contiguous"
    assert bool(np.all(run == color)), "run must be monochrome"
    return color, start, end


def _candidate_layers(grid: np.ndarray, bg: int, k: int) -> list[dict] | None:
    h, w = grid.shape
    if not (1 <= k < h and 1 <= k < w):
        return None
    if np.any(grid[:k, :k] != bg):
        return None
    if np.any(grid[k:, k:] != bg):
        return None

    layers: list[dict] = []
    seen_colors: set[int] = set()
    for layer in range(k):
        if np.any(grid[layer, :k] != bg):
            return None
        if np.any(grid[:k, layer] != bg):
            return None

        try:
            top_color, c0, c1 = _parse_single_run(grid[layer, k:], bg)
            left_color, r0, r1 = _parse_single_run(grid[k:, layer], bg)
        except AssertionError:
            return None

        if top_color != left_color or top_color in seen_colors:
            return None
        seen_colors.add(top_color)
        layers.append(
            {
                "layer": layer,
                "color": int(top_color),
                "r0": int(r0),
                "r1": int(r1),
                "c0": int(c0),
                "c1": int(c1),
            }
        )

    return layers


def _extract_layers(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(g[0, 0])
    candidates: list[tuple[int, list[dict]]] = []
    for k in range(1, min(g.shape)):
        layers = _candidate_layers(g, bg, k)
        if layers is not None:
            candidates.append((k, layers))
    assert len(candidates) == 1, "expected a unique valid margin thickness"
    k, layers = candidates[0]
    return bg, k, layers


def _paint_rectangles(grid: np.ndarray, bg: int, k: int, layers: list[dict], *, outer_last: bool) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    order = range(k - 1, -1, -1) if outer_last else range(k)
    for layer in order:
        spec = layers[layer]
        r0 = k + spec["r0"]
        r1 = k + spec["r1"]
        c0 = k + spec["c0"]
        c1 = k + spec["c1"]
        out[r0 : r1 + 1, c0 : c1 + 1] = spec["color"]
    out[:k, :k] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each hidden canvas rectangle from matching top and left projection bars, with outer layers painted last.

    Input structure:
    - The first k rows are a top margin and the first k columns are a left margin; their k x k corner is background.
    - Every top-margin row contains exactly one contiguous horizontal colored bar in the canvas columns.
    - Every left-margin column contains exactly one contiguous vertical colored bar in the canvas rows.
    - Matching top and left bars share both color and layer index.

    Rule:
    - Infer k from the empty corner and the empty canvas.
    - For each layer, the top bar gives the rectangle's column span and the left bar gives its row span.
    - Fill the Cartesian product of those spans as a solid rectangle of that color.
    - When rectangles overlap, paint from inner layer to outer layer so smaller layer indices overwrite larger ones; keep the margins unchanged.

    Key insight:
    - The visible bars are only orthogonal projections of hidden rectangles, and the margin layer order encodes the overlap depth.
    """
    bg, k, layers = _extract_layers(grid)
    return _paint_rectangles(np.asarray(grid, dtype=int), bg, k, layers, outer_last=True)


def _sample_span(rng: np.random.Generator, length: int) -> tuple[int, int]:
    max_len = length if rng.random() < 0.2 else max(2, length - 1)
    span_len = int(rng.integers(2, max_len + 1))
    start = int(rng.integers(0, length - span_len + 1))
    return start, start + span_len - 1


def _overlap_area(a: dict, b: dict) -> int:
    rows = min(a["r1"], b["r1"]) - max(a["r0"], b["r0"]) + 1
    cols = min(a["c1"], b["c1"]) - max(a["c0"], b["c0"]) + 1
    return max(0, rows) * max(0, cols)


def _count_overlapping_pairs(layers: list[dict]) -> int:
    return sum(1 for a, b in combinations(layers, 2) if _overlap_area(a, b) > 0)


def _build_input(k: int, canvas_h: int, canvas_w: int, layers: list[dict]) -> np.ndarray:
    grid = np.full((k + canvas_h, k + canvas_w), BG_COLOR, dtype=int)
    for spec in layers:
        layer = spec["layer"]
        color = spec["color"]
        grid[layer, k + spec["c0"] : k + spec["c1"] + 1] = color
        grid[k + spec["r0"] : k + spec["r1"] + 1, layer] = color
    return grid


def _make_layers(
    rng: np.random.Generator,
    k: int,
    canvas_h: int,
    canvas_w: int,
) -> tuple[list[dict], bool]:
    colors = rng.choice(np.array(PALETTE, dtype=int), size=k, replace=False).tolist()
    require_overlap = bool(rng.random() < OVERLAP_PROBABILITY)
    layers: list[dict] = []
    for layer, color in enumerate(colors):
        r0, r1 = _sample_span(rng, canvas_h)
        c0, c1 = _sample_span(rng, canvas_w)
        layers.append(
            {
                "layer": layer,
                "color": int(color),
                "r0": int(r0),
                "r1": int(r1),
                "c0": int(c0),
                "c1": int(c1),
            }
        )
    return layers, require_overlap


def _semantic_checks(inp: np.ndarray, layers: list[dict], require_overlap: bool) -> bool:
    try:
        bg, k, parsed = _extract_layers(inp)
    except AssertionError:
        return False

    intended = [(d["color"], d["r0"], d["r1"], d["c0"], d["c1"]) for d in layers]
    recovered = [(d["color"], d["r0"], d["r1"], d["c0"], d["c1"]) for d in parsed]
    if len(layers) != k or intended != recovered:
        return False

    expected = _paint_rectangles(inp, bg, k, parsed, outer_last=True)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    visible_canvas = expected[k:, k:]
    coverage = float(np.count_nonzero(visible_canvas != bg)) / float(visible_canvas.size)
    if not (VISIBLE_COVERAGE_RANGE[0] <= coverage <= VISIBLE_COVERAGE_RANGE[1]):
        return False

    if len({(d["r0"], d["r1"]) for d in layers}) == 1:
        return False
    if len({(d["c0"], d["c1"]) for d in layers}) == 1:
        return False

    if any(not np.any(visible_canvas == d["color"]) for d in layers):
        return False

    overlap_pairs = _count_overlapping_pairs(layers)
    if require_overlap:
        if overlap_pairs == 0:
            return False
        wrong_order = _paint_rectangles(inp, bg, k, parsed, outer_last=False)
        if np.array_equal(expected, wrong_order):
            return False
    elif overlap_pairs != 0:
        return False

    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    k = int(rng.integers(K_RANGE[0], K_RANGE[1] + 1))
    canvas_h = int(rng.integers(CANVAS_H_RANGE[0], CANVAS_H_RANGE[1] + 1))
    canvas_w = int(rng.integers(CANVAS_W_RANGE[0], CANVAS_W_RANGE[1] + 1))
    layers, require_overlap = _make_layers(rng, k, canvas_h, canvas_w)
    inp = _build_input(k, canvas_h, canvas_w, layers)
    if _semantic_checks(inp, layers, require_overlap):
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid margin-projection reconstruction input")
