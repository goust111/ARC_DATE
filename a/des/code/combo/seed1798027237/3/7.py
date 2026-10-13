from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))


def _parse_legend(row: np.ndarray) -> list[int]:
    colors = [int(v) for v in np.asarray(row, dtype=int).tolist() if int(v) != BG]
    assert colors, "legend must contain at least one layer color"
    assert len(colors) == len(set(colors)), "legend colors must be unique"
    return colors


def _draw_border(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _border_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    _draw_border(mask, rect, True)
    return mask


def _border_size(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    return 2 * (height + width) - 4


def _compose_body(shape: tuple[int, int], rects: list[tuple[int, int, int, int]], colors: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rect, color in zip(rects, colors, strict=True):
        _draw_border(out, rect, color)
    return out


def _visible_mask(body: np.ndarray, color: int) -> np.ndarray:
    return np.asarray(body == int(color), dtype=bool)


def _visible_bbox_matches(rect: tuple[int, int, int, int], body: np.ndarray, color: int) -> bool:
    visible = np.argwhere(body == int(color))
    if visible.size == 0:
        return False
    r0, c0 = visible.min(axis=0)
    r1, c1 = visible.max(axis=0)
    return (int(r0), int(c0), int(r1), int(c1)) == tuple(map(int, rect))


def _visible_sides(rect: tuple[int, int, int, int], body: np.ndarray, color: int) -> tuple[bool, bool, bool, bool]:
    r0, c0, r1, c1 = rect
    visible = body == int(color)
    return (
        bool(visible[r0, c0 : c1 + 1].any()),
        bool(visible[r1, c0 : c1 + 1].any()),
        bool(visible[r0 : r1 + 1, c0].any()),
        bool(visible[r0 : r1 + 1, c1].any()),
    )


def _rect_contains_visible(
    visible_coords: np.ndarray, r0: int, c0: int, r1: int, c1: int
) -> bool:
    for r, c in visible_coords:
        r = int(r)
        c = int(c)
        if not (r0 <= r <= r1 and c0 <= c <= c1):
            return False
        if r != r0 and r != r1 and c != c0 and c != c1:
            return False
    return True


def _rect_allowed(allowed: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    return (
        bool(allowed[r0, c0 : c1 + 1].all())
        and bool(allowed[r1, c0 : c1 + 1].all())
        and bool(allowed[r0 : r1 + 1, c0].all())
        and bool(allowed[r0 : r1 + 1, c1].all())
    )


def _infer_rectangle(visible: np.ndarray, higher_cover: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(visible)
    assert coords.size > 0, "each layer must retain at least one visible pixel"

    h, w = visible.shape
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    allowed = visible | higher_cover

    candidates: list[tuple[int, int, int, int]] = []
    for r0 in range(int(min_r) + 1):
        for r1 in range(int(max_r), h):
            if r1 - r0 < 2:
                continue
            for c0 in range(int(min_c) + 1):
                for c1 in range(int(max_c), w):
                    if c1 - c0 < 2:
                        continue
                    if not _rect_contains_visible(coords, r0, c0, r1, c1):
                        continue
                    if not _rect_allowed(allowed, r0, c0, r1, c1):
                        continue
                    candidates.append((int(r0), int(c0), int(r1), int(c1)))

    assert len(candidates) == 1, "layer must determine a unique rectangle"
    return candidates[0]


def _recover_rectangles(body: np.ndarray, order: list[int]) -> dict[int, tuple[int, int, int, int]]:
    h, w = body.shape
    higher_cover = np.zeros((h, w), dtype=bool)
    recovered: dict[int, tuple[int, int, int, int]] = {}

    for color in reversed(order):
        visible = _visible_mask(body, color)
        rect = _infer_rectangle(visible, higher_cover)
        recovered[int(color)] = rect
        higher_cover |= _border_mask((h, w), rect)

    return recovered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover fully occluded rectangular frames from a layered composite using the bottom-row legend.

    Input structure:
    - Black background.
    - Several one-pixel-thick hollow rectangular frames with distinct colors.
    - Higher layers overwrite lower layers where frame borders overlap.
    - The bottom row is a color legend from bottom layer to top layer, with black gaps.

    Rule:
    - Read layer order from the legend.
    - Infer each colored frame as a complete axis-aligned rectangle border.
    - Restore the missing pixels of occluded lower layers, then re-compose the full stack.
    - Remove the legend row in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 3

    body = g[:-1, :]
    order = _parse_legend(g[-1, :])
    rects = _recover_rectangles(body, order)

    out = np.zeros_like(body)
    for color in order:
        _draw_border(out, rects[int(color)], int(color))
    return out


# --------------- generator ---------------

MIN_LAYERS = 2
MAX_LAYERS = 4
MIN_SIDE = 4
GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (12, 20)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120


def _sample_rect(h: int, w: int) -> tuple[int, int, int, int]:
    r0 = random.randint(0, h - MIN_SIDE)
    r1 = random.randint(r0 + MIN_SIDE - 1, h - 1)
    c0 = random.randint(0, w - MIN_SIDE)
    c1 = random.randint(c0 + MIN_SIDE - 1, w - 1)
    return int(r0), int(c0), int(r1), int(c1)


def _intersection_count(
    shape: tuple[int, int], a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> int:
    return int(np.logical_and(_border_mask(shape, a), _border_mask(shape, b)).sum())


def _sample_rectangles(h: int, w: int, layers: int) -> list[tuple[int, int, int, int]] | None:
    shape = (h, w)
    rects: list[tuple[int, int, int, int]] = []
    for idx in range(layers):
        for _ in range(PLACEMENT_ATTEMPTS):
            rect = _sample_rect(h, w)
            if rect in rects:
                continue
            if idx > 0 and not any(_intersection_count(shape, rect, prev) > 0 for prev in rects):
                continue
            rects.append(rect)
            break
        else:
            return None
    return rects


def _legend_row(width: int, colors: list[int]) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    layer_count = len(colors)
    min_width = 2 * layer_count - 1
    assert min_width <= width

    gaps = [0] + [1] * (layer_count - 1) + [0]
    extra = width - min_width
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1

    c = gaps[0]
    for idx, color in enumerate(colors):
        row[c] = int(color)
        c += 1
        if idx + 1 < layer_count:
            c += gaps[idx + 1]
    return row


def _occlusion_stats(
    shape: tuple[int, int], rects: list[tuple[int, int, int, int]], colors: list[int]
) -> tuple[np.ndarray, list[int]]:
    body = _compose_body(shape, rects, colors)
    hidden_counts: list[int] = []
    for rect, color in zip(rects, colors, strict=True):
        border = _border_mask(shape, rect)
        hidden = int(border.sum()) - int((body == int(color)).sum() if color == colors[-1] else np.logical_and(border, body == int(color)).sum())
        hidden_counts.append(hidden)
    return body, hidden_counts


def _is_valid_instance(
    shape: tuple[int, int], rects: list[tuple[int, int, int, int]], colors: list[int]
) -> bool:
    body = _compose_body(shape, rects, colors)

    overlap_pairs = 0
    for i, rect in enumerate(rects):
        for other in rects[i + 1 :]:
            if _intersection_count(shape, rect, other) > 0:
                overlap_pairs += 1
    if overlap_pairs == 0:
        return False

    occluded_layers = 0
    for rect, color in zip(rects, colors, strict=True):
        border = _border_mask(shape, rect)
        visible_count = int(np.logical_and(border, body == int(color)).sum())
        if visible_count == 0:
            return False
        if visible_count < _border_size(rect):
            occluded_layers += 1
        if not _visible_bbox_matches(rect, body, color):
            return False
        if not all(_visible_sides(rect, body, color)):
            return False
    if occluded_layers == 0:
        return False

    try:
        recovered = _recover_rectangles(body, colors)
    except AssertionError:
        return False

    for rect, color in zip(rects, colors, strict=True):
        if recovered[int(color)] != tuple(map(int, rect)):
            return False
    return True


def _generate_case() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    layers = random.randint(MIN_LAYERS, MAX_LAYERS)
    colors = random.sample(COLOR_POOL, layers)

    rects = _sample_rectangles(h, w, layers)
    if rects is None:
        return None
    if not _is_valid_instance((h, w), rects, colors):
        return None

    body = _compose_body((h, w), rects, colors)
    legend = _legend_row(w, colors)
    return np.vstack([body, legend])


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
