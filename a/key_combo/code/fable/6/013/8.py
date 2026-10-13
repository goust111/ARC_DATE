from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


MAX_GENERATION_ATTEMPTS = 500
TICK_HEIGHT = 2
LAYER_COUNT_RANGE = (3, 5)
BODY_HEIGHT_RANGE = (4, 8)
MAX_BODY_WIDTH = 9


def _find_body_bbox(patch: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    row_spans: list[tuple[int, int] | None] = []
    for row in np.asarray(patch, dtype=int):
        cols = np.flatnonzero(row != int(bg))
        if cols.size == 0:
            row_spans.append(None)
            continue
        c0 = int(cols[0])
        c1 = int(cols[-1])
        if bool(np.all(row[c0 : c1 + 1] != int(bg))):
            row_spans.append((c0, c1))
        else:
            row_spans.append(None)

    candidates: list[tuple[int, int, int, int, int, int, int, int]] = []
    r = 0
    while r < len(row_spans):
        span = row_spans[r]
        if span is None:
            r += 1
            continue
        start = r
        c0, c1 = span
        while r + 1 < len(row_spans) and row_spans[r + 1] == span:
            r += 1
        end = r
        width = c1 - c0 + 1
        height = end - start + 1
        candidates.append((width * height, width, height, -start, start, end, c0, c1))
        r += 1

    assert candidates, "failed to locate a filled rectangle body"
    _area, width, height, _neg_start, r0, r1, c0, c1 = max(
        candidates,
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    assert width >= 3 and height >= 3, "body rectangle is too small"
    assert r0 >= 1, "expected tally ticks above the rectangle body"
    return int(r0), int(c0), int(r1), int(c1)


def _parse_layer(grid: np.ndarray, component: dict) -> dict[str, object]:
    bg = int(get_bg(grid))
    patch = crop_bbox(grid, component["bbox"])
    body_r0, body_c0, body_r1, body_c1 = _find_body_bbox(patch, bg)
    body = patch[body_r0 : body_r1 + 1, body_c0 : body_c1 + 1].copy()

    values, counts = np.unique(body, return_counts=True)
    base = int(values[np.argmax(counts)])
    assert base != int(bg), "body base color must be non-background"

    tick_mask = patch[:body_r0, body_c0 : body_c1 + 1] != int(bg)
    tick_components = components_mask(tick_mask, connectivity=4)
    order = len(tick_components)
    assert order >= 1, "expected at least one tally tick"

    marker_rows, marker_cols = np.where(body != int(base))
    markers = [
        (int(r), int(c), int(body[int(r), int(c)]))
        for r, c in zip(marker_rows.tolist(), marker_cols.tolist())
    ]
    return {
        "order": int(order),
        "base": int(base),
        "shape": body.shape,
        "markers": markers,
    }


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    layers = [_parse_layer(g, component) for component in components]
    assert len(layers) >= 3, "expected at least three layers"

    shapes = {tuple(layer["shape"]) for layer in layers}
    assert len(shapes) == 1, "all layer rectangles must have the same size"
    out_shape = next(iter(shapes))

    base_counts = Counter(int(layer["base"]) for layer in layers)
    assert len(base_counts) == 2, "expected one shared base color and one negative base color"
    shared_base, shared_count = max(base_counts.items(), key=lambda item: item[1])
    assert shared_count == len(layers) - 1, "expected exactly one negative layer"

    orders = [int(layer["order"]) for layer in layers]
    assert len(set(orders)) == len(orders), "tally counts must be unique"

    ordered = sorted(layers, key=lambda layer: int(layer["order"]))
    out = np.full(out_shape, int(shared_base), dtype=int)
    negative_base = next(base for base in base_counts if int(base) != int(shared_base))

    for layer in ordered:
        base = int(layer["base"])
        markers = layer["markers"]
        if base == int(shared_base):
            for r, c, color in markers:
                out[int(r), int(c)] = int(color)
        else:
            assert base == int(negative_base)
            for r, c, _color in markers:
                out[int(r), int(c)] = int(shared_base)

    details: dict[str, object] = {
        "bg": int(bg),
        "shared_base": int(shared_base),
        "negative_base": int(negative_base),
        "layers": ordered,
    }
    return out, details


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Merge same-sized stamped layers by tally-count order.

    Each connected object contains a filled rectangle body and a set of short ticks
    attached above it. The tick count gives the processing order. Most layers stamp
    their non-base marker pixels onto a shared base-colored canvas. The one layer
    with the odd base color is subtractive: its marker positions erase the current
    canvas back to the shared base color.
    """
    return _solve_details(np.asarray(input_grid, dtype=int))[0]


def _choose_tick_columns(width: int, count: int, rng: np.random.Generator) -> list[int]:
    span = 2 * count - 1
    assert span <= width, "not enough width for separated tally ticks"
    offset = int(rng.integers(0, width - span + 1))
    return [offset + 2 * idx for idx in range(count)]


def _render_layer(
    body_shape: tuple[int, int],
    base: int,
    markers: list[tuple[int, int, int]],
    order: int,
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    body_h, body_w = map(int, body_shape)
    layer = np.full((body_h + TICK_HEIGHT, body_w), int(bg), dtype=int)
    layer[TICK_HEIGHT:, :] = int(base)
    for r, c, color in markers:
        layer[TICK_HEIGHT + int(r), int(c)] = int(color)
    for c in _choose_tick_columns(body_w, int(order), rng):
        layer[:TICK_HEIGHT, int(c)] = int(base)
    return layer


def _all_positions(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    return [(r, c) for r in range(h) for c in range(w)]


def _make_normal_markers(
    current: np.ndarray,
    base: int,
    marker_colors: list[int],
    rng: np.random.Generator,
    *,
    forced: tuple[tuple[int, int], ...] = (),
    banned: frozenset[tuple[int, int]] = frozenset(),
) -> list[tuple[int, int, int]]:
    body_h, body_w = current.shape
    area = body_h * body_w
    allowed = [pos for pos in _all_positions((body_h, body_w)) if pos not in banned]
    assert allowed, "no available positions left for a normal layer"

    low = max(3, len(forced))
    high = min(len(allowed), max(low, area // 3))
    count = int(rng.integers(low, high + 1))

    chosen: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pos in forced:
        if pos not in banned and pos not in seen:
            chosen.append(pos)
            seen.add(pos)

    painted = [tuple(pos) for pos in np.argwhere(current != int(base)) if tuple(pos) not in banned]
    while len(chosen) < count:
        pool = painted if painted and float(rng.random()) < 0.45 else allowed
        candidate = tuple(pool[int(rng.integers(0, len(pool)))])
        if candidate in seen:
            continue
        chosen.append(candidate)
        seen.add(candidate)

    markers: list[tuple[int, int, int]] = []
    for r, c in chosen:
        current_color = int(current[int(r), int(c)])
        color_choices = marker_colors
        if current_color != int(base):
            changed = [color for color in marker_colors if int(color) != current_color]
            if changed and float(rng.random()) < 0.65:
                color_choices = changed
        color = int(color_choices[int(rng.integers(0, len(color_choices)))])
        markers.append((int(r), int(c), color))
    return markers


def _build_layers(
    body_shape: tuple[int, int],
    shared_base: int,
    negative_base: int,
    marker_colors: list[int],
    layer_count: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], np.ndarray]:
    current = np.full(body_shape, int(shared_base), dtype=int)
    negative_index = int(rng.integers(1, layer_count - 1))
    layers: list[dict[str, object]] = []

    for order in range(1, negative_index + 1):
        markers = _make_normal_markers(current, shared_base, marker_colors, rng)
        for r, c, color in markers:
            current[int(r), int(c)] = int(color)
        layers.append({"order": order, "base": int(shared_base), "markers": markers})

    painted_positions = [tuple(pos) for pos in np.argwhere(current != int(shared_base))]
    assert len(painted_positions) >= 3, "need enough painted cells before the negative layer"

    erase_count = int(rng.integers(2, min(len(painted_positions), max(3, len(painted_positions) // 2)) + 1))
    chosen_idx = rng.choice(len(painted_positions), size=erase_count, replace=False)
    erase_positions = [painted_positions[int(idx)] for idx in np.asarray(chosen_idx).tolist()]
    repaint_count = int(rng.integers(1, erase_count))
    repaint_idx = rng.choice(erase_count, size=repaint_count, replace=False)
    repaint_targets = {erase_positions[int(idx)] for idx in np.asarray(repaint_idx).tolist()}
    stay_erased = set(erase_positions) - repaint_targets
    assert repaint_targets and stay_erased, "need both repainted and permanently erased cells"

    negative_markers: list[tuple[int, int, int]] = []
    for r, c in erase_positions:
        color = int(marker_colors[int(rng.integers(0, len(marker_colors)))])
        negative_markers.append((int(r), int(c), color))
        current[int(r), int(c)] = int(shared_base)
    layers.append({"order": negative_index + 1, "base": int(negative_base), "markers": negative_markers})

    forced = tuple(sorted(repaint_targets))
    markers = _make_normal_markers(
        current,
        shared_base,
        marker_colors,
        rng,
        forced=forced,
        banned=frozenset(stay_erased),
    )
    for r, c, color in markers:
        current[int(r), int(c)] = int(color)
    layers.append({"order": negative_index + 2, "base": int(shared_base), "markers": markers})

    for order in range(negative_index + 3, layer_count + 1):
        markers = _make_normal_markers(
            current,
            shared_base,
            marker_colors,
            rng,
            banned=frozenset(stay_erased),
        )
        for r, c, color in markers:
            current[int(r), int(c)] = int(color)
        layers.append({"order": order, "base": int(shared_base), "markers": markers})

    assert all(int(current[r, c]) == int(shared_base) for r, c in stay_erased)
    assert all(int(current[r, c]) != int(shared_base) for r, c in repaint_targets)
    return layers, current.copy()


def _canvas_shape(object_shape: tuple[int, int], layer_count: int, rng: np.random.Generator) -> tuple[int, int]:
    obj_h, obj_w = map(int, object_shape)
    rows = 2
    cols = 2
    height = rows * (obj_h + 2) + 2 + int(rng.integers(0, 3))
    width = cols * (obj_w + 2) + 2 + int(rng.integers(0, 5))
    assert layer_count <= rows * cols
    assert height <= 30 and width <= 30
    return int(height), int(width)


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    layer_count = int(rng.integers(*LAYER_COUNT_RANGE))
    body_h = int(rng.integers(*BODY_HEIGHT_RANGE))
    min_width = max(5, 2 * layer_count - 1)
    if min_width > MAX_BODY_WIDTH:
        return None
    body_w = int(rng.integers(min_width, MAX_BODY_WIDTH + 1))

    palette = rng.choice(np.array(Color.ALL_COLORS, dtype=int), size=7, replace=False)
    bg = int(palette[0])
    shared_base = int(palette[1])
    negative_base = int(palette[2])
    marker_colors = [int(color) for color in palette[3:] if int(color) not in (bg, shared_base, negative_base)]
    if len(marker_colors) < 2:
        return None

    body_shape = (body_h, body_w)
    layers, expected = _build_layers(body_shape, shared_base, negative_base, marker_colors, layer_count, rng)

    object_shape = (body_h + TICK_HEIGHT, body_w)
    grid_h, grid_w = _canvas_shape(object_shape, layer_count, rng)
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)

    rendered_layers = [
        _render_layer(body_shape, int(layer["base"]), list(layer["markers"]), int(layer["order"]), bg, rng)
        for layer in layers
    ]
    rng.shuffle(rendered_layers)

    try:
        for patch in rendered_layers:
            r0, c0 = random_free_location(
                grid,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
            paste(grid, patch, r0, c0, transparent=bg)
    except ValueError:
        return None

    try:
        actual, details = _solve_details(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    layer_specs = details["layers"]
    shared = int(details["shared_base"])
    negative_layer = next(layer for layer in layer_specs if int(layer["base"]) != shared)
    neg_positions = {(int(r), int(c)) for r, c, _color in negative_layer["markers"]}
    if not any(int(expected[r, c]) == shared for r, c in neg_positions):
        return None
    if not any(int(expected[r, c]) != shared for r, c in neg_positions):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _gen_attempt(rng)
        if built is None:
            continue
        grid, _expected = built
        if grid.ndim == 2 and grid.shape[0] <= 30 and grid.shape[1] <= 30:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
