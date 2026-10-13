from __future__ import annotations

import importlib.util
import random
from itertools import combinations
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_fable_8_024_1", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load arc_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC_UTILS = _load_arc_utils()
Color = _ARC_UTILS.Color
bbox_coords = _ARC_UTILS.bbox_coords
paste = _ARC_UTILS.paste

BG = Color.BLACK
SEP = Color.GREY
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

GEN_MAX_ATTEMPTS = 400
CANVAS_H_RANGE = (8, 16)
CANVAS_W_RANGE = (10, 18)
RECT_COUNT_OPTIONS = (3, 3, 2)
LEGEND_EXTRA_ROWS = (0, 1, 2)
MIN_RECT_SIDE = 4
MAX_RECT_SIDE = 9
MIN_BAR_LEN = 2


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if bool(np.all(grid[r] == SEP))]
    assert len(rows) == 1, "expected exactly one full-width gray separator row"
    sep = int(rows[0])
    assert 0 < sep < grid.shape[0] - 1, "separator must split canvas and legend"
    return sep


def _visible_on_every_side(canvas: np.ndarray, color: int, rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    return bool(
        np.any(canvas[r0, c0 : c1 + 1] == color)
        and np.any(canvas[r1, c0 : c1 + 1] == color)
        and np.any(canvas[r0 : r1 + 1, c0] == color)
        and np.any(canvas[r0 : r1 + 1, c1] == color)
    )


def _extract_canvas_rectangles(canvas: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    rects: dict[int, tuple[int, int, int, int]] = {}
    colors = sorted(int(color) for color in np.unique(canvas) if int(color) != BG)
    assert SEP not in colors, "gray is reserved for the separator line"
    for color in colors:
        coords = np.argwhere(canvas == color)
        assert coords.size > 0
        rect = tuple(map(int, bbox_coords(coords)))
        assert _visible_on_every_side(canvas, color, rect), "each rectangle must expose all four edges"
        rects[color] = rect
    assert rects, "expected at least one colored rectangle on the canvas"
    return rects


def _extract_legend_lengths(legend: np.ndarray) -> dict[int, int]:
    lengths: dict[int, int] = {}
    colors = sorted(int(color) for color in np.unique(legend) if int(color) not in {BG, SEP})
    for color in colors:
        coords = np.argwhere(legend == color)
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        assert len(rows) == 1, "each legend entry must be a thin horizontal bar"
        assert len(cols) == int(cols.max()) - int(cols.min()) + 1, "legend bar must be contiguous"
        assert int(coords.shape[0]) == len(cols), "legend bar must be one pixel thick"
        lengths[color] = len(cols)
    assert lengths, "expected at least one legend bar"
    assert len(set(lengths.values())) == len(lengths), "legend bar lengths must be distinct"
    return lengths


def _render_rectangles(
    shape: tuple[int, int],
    rects: dict[int, tuple[int, int, int, int]],
    order: list[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for color in order:
        r0, c0, r1, c1 = rects[color]
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(color), dtype=int)
        paste(out, patch, r0, c0)
    return out


def _overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _has_overlap(rects: dict[int, tuple[int, int, int, int]]) -> bool:
    return any(_overlap_area(rects[a], rects[b]) > 0 for a, b in combinations(rects, 2))


def _extract_scene(grid: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep = _find_separator_row(g)
    canvas = g[:sep, :]
    legend = g[sep + 1 :, :]
    assert canvas.size > 0 and legend.size > 0
    rects = _extract_canvas_rectangles(canvas)
    lengths = _extract_legend_lengths(legend)
    assert set(rects) == set(lengths), "canvas colors and legend colors must match exactly"
    return canvas, rects, lengths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each occluded rectangle, then redraw the canvas using legend-bar length as the depth order.

    Input structure
    - A full-width gray separator row splits the grid into an upper black canvas and a lower legend area.
    - The canvas contains several overlapping solid rectangles of distinct colors.
    - The legend contains one thin horizontal bar per rectangle color, with all bar lengths distinct.

    Rule
    - For each canvas color, recover the full rectangle by taking the bounding box of its visible pixels.
    - Interpret shorter legend bars as deeper layers and longer legend bars as higher layers.
    - Redraw only the canvas from black background, painting the recovered rectangles from shortest bar to longest bar.

    Key insight
    - The visible fragments already reveal every rectangle's four true sides, so the hidden geometry comes from color-wise bounding boxes, while the legend independently overrides the input occlusion order.
    """
    canvas, rects, lengths = _extract_scene(np.asarray(grid, dtype=int))
    order = sorted(rects, key=lambda color: lengths[color])
    return _render_rectangles(canvas.shape, rects, order)


def _sample_canvas_shape(rng: random.Random) -> tuple[int, int]:
    return rng.randint(*CANVAS_H_RANGE), rng.randint(*CANVAS_W_RANGE)


def _sample_target_order(colors: list[int], width: int, rng: random.Random) -> tuple[list[int], dict[int, int]]:
    order = colors[:]
    rng.shuffle(order)
    candidate_lengths = list(range(MIN_BAR_LEN, width))
    chosen = sorted(rng.sample(candidate_lengths, len(colors)))
    lengths = {color: chosen[i] for i, color in enumerate(order)}
    return order, lengths


def _sample_input_order(colors: list[int], target_order: list[int], rng: random.Random) -> list[int]:
    order = colors[:]
    for _ in range(30):
        rng.shuffle(order)
        if order != target_order and order[-1] != target_order[-1]:
            return order[:]
    raise RuntimeError("failed to find a distinct input order")


def _sample_rectangles(
    height: int,
    width: int,
    colors: list[int],
    rng: random.Random,
) -> dict[int, tuple[int, int, int, int]] | None:
    core_h = rng.randint(2, min(4, height - 4))
    core_w = rng.randint(2, min(4, width - 4))
    core_r0 = rng.randint(1, height - core_h - 2)
    core_c0 = rng.randint(1, width - core_w - 2)
    core_r1 = core_r0 + core_h - 1
    core_c1 = core_c0 + core_w - 1

    rects: dict[int, tuple[int, int, int, int]] = {}
    seen = set()
    for color in colors:
        placed = None
        for _ in range(60):
            up = rng.randint(0, core_r0)
            down = rng.randint(0, height - 1 - core_r1)
            left = rng.randint(0, core_c0)
            right = rng.randint(0, width - 1 - core_c1)
            r0 = core_r0 - up
            c0 = core_c0 - left
            r1 = core_r1 + down
            c1 = core_c1 + right
            if r1 - r0 + 1 < MIN_RECT_SIDE or c1 - c0 + 1 < MIN_RECT_SIDE:
                continue
            if r1 - r0 + 1 > MAX_RECT_SIDE or c1 - c0 + 1 > MAX_RECT_SIDE:
                continue
            rect = (r0, c0, r1, c1)
            if rect in seen:
                continue
            placed = rect
            break
        if placed is None:
            return None
        seen.add(placed)
        rects[int(color)] = placed
    return rects


def _build_legend(width: int, lengths: dict[int, int], rng: random.Random) -> np.ndarray:
    height = len(lengths) + rng.choice(LEGEND_EXTRA_ROWS)
    bars = np.full((height, width), BG, dtype=int)
    rows = sorted(rng.sample(range(height), len(lengths)))
    items = list(lengths.items())
    rng.shuffle(items)
    for row, (color, length) in zip(rows, items):
        c0 = rng.randint(0, width - length)
        patch = np.full((1, length), int(color), dtype=int)
        paste(bars, patch, row, c0)
    return bars


def _compose_input(
    canvas_shape: tuple[int, int],
    rects: dict[int, tuple[int, int, int, int]],
    input_order: list[int],
    legend_lengths: dict[int, int],
    rng: random.Random,
) -> np.ndarray:
    canvas = _render_rectangles(canvas_shape, rects, input_order)
    legend = _build_legend(canvas_shape[1], legend_lengths, rng)
    full = np.full((canvas_shape[0] + 1 + legend.shape[0], canvas_shape[1]), BG, dtype=int)
    full[: canvas_shape[0], :] = canvas
    full[canvas_shape[0], :] = SEP
    full[canvas_shape[0] + 1 :, :] = legend
    return full


def _requires_recovery(input_canvas: np.ndarray, output_canvas: np.ndarray, rects: dict[int, tuple[int, int, int, int]]) -> bool:
    changed = np.argwhere(input_canvas != output_canvas)
    for r, c in changed:
        color = int(output_canvas[r, c])
        if color == BG:
            continue
        r0, c0, r1, c1 = rects[color]
        if r0 <= int(r) <= r1 and c0 <= int(c) <= c1 and int(input_canvas[r, c]) != color:
            return True
    return False


def validate_instance(grid: np.ndarray, output: np.ndarray | None = None) -> bool:
    canvas, rects, lengths = _extract_scene(grid)
    assert len(rects) >= 2, "need multiple rectangles"
    assert _has_overlap(rects), "at least one rectangle pair must overlap"
    for color, rect in rects.items():
        assert _visible_on_every_side(canvas, color, rect), "each reconstructed rectangle must expose all sides"
    derived = _render_rectangles(canvas.shape, rects, sorted(rects, key=lambda color: lengths[color]))
    if output is None:
        output = main(grid)
    assert np.array_equal(output, derived), "solver output must match the parsed scene"
    assert not np.array_equal(canvas, output), "reordering must change the canvas"
    assert _requires_recovery(canvas, output, rects), "at least one newly revealed pixel must come from a hidden rectangle area"
    return True


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    canvas_shape = _sample_canvas_shape(rng)
    rect_count = rng.choice(RECT_COUNT_OPTIONS)
    colors = rng.sample(PALETTE, rect_count)
    target_order, legend_lengths = _sample_target_order(colors, canvas_shape[1], rng)
    input_order = _sample_input_order(colors, target_order, rng)
    rects = _sample_rectangles(canvas_shape[0], canvas_shape[1], colors, rng)
    if rects is None or not _has_overlap(rects):
        return None

    canvas_in = _render_rectangles(canvas_shape, rects, input_order)
    for color, rect in rects.items():
        if not _visible_on_every_side(canvas_in, color, rect):
            return None

    expected = _render_rectangles(canvas_shape, rects, target_order)
    if np.array_equal(canvas_in, expected):
        return None
    if not _requires_recovery(canvas_in, expected, rects):
        return None

    full = _compose_input(canvas_shape, rects, input_order, legend_lengths, rng)
    if max(full.shape) > 30 or max(expected.shape) > 30:
        return None
    try:
        if not validate_instance(full, expected):
            return None
    except Exception:
        return None
    return full


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid depth-legend rearrangement input")
