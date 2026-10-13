from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_order(strip: np.ndarray) -> list[int]:
    cols = np.flatnonzero(strip != BG)
    order = [int(strip[c]) for c in cols]
    assert order, "expected at least one color in the order strip"
    assert len(order) == len(set(order)), "order strip colors must be unique"
    return order


def _compose_output(
    order: list[int],
    rects: dict[int, tuple[int, int, int, int]],
) -> np.ndarray:
    heights = [rects[color][2] - rects[color][0] + 1 for color in order]
    widths = [rects[color][3] - rects[color][1] + 1 for color in order]

    out_h = max(heights)
    out_w = sum(widths) + max(0, len(order) - 1)
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for i, color in enumerate(order):
        h = heights[i]
        w = widths[i]
        out[:h, col : col + w] = int(color)
        col += w
        if i < len(order) - 1:
            out[:, col] = SEP
            col += 1

    return out


def _infer_rectangles(
    scene: np.ndarray,
    order: list[int],
) -> dict[int, tuple[int, int, int, int]]:
    present = {int(v) for v in np.unique(scene) if int(v) != BG}
    assert present == set(order), "scene colors must match the order strip"

    higher_mask = np.zeros_like(scene, dtype=bool)
    rects: dict[int, tuple[int, int, int, int]] = {}

    for color in reversed(order):
        visible = scene == int(color)
        bbox = _bbox_from_mask(visible)
        assert bbox is not None, "each strip color must appear in the scene"
        r0, c0, r1, c1 = bbox

        box = np.zeros_like(scene, dtype=bool)
        box[r0 : r1 + 1, c0 : c1 + 1] = True

        # The visible span is the rectangle's full bbox; any hole inside it must
        # already be covered by a higher layer.
        assert not np.any(box & ~(visible | higher_mask)), (
            "current layer must be a rectangle whose hidden cells are all occluded"
        )

        rects[int(color)] = bbox
        higher_mask |= box

    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Separate overlapping colored rectangles into individual canvases.

    The last row is an order strip: left-to-right gives bottom-to-top drawing
    order. Each colored layer is an axis-aligned solid rectangle. Lower layers
    may be partially hidden, but their visible row/column span still determines
    the full rectangle bbox; missing cells inside that bbox must be covered by
    already-reconstructed higher layers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 2 and g.shape[1] >= 1, "expected scene rows plus one strip row"

    scene = g[:-1, :]
    strip = g[-1, :]
    order = _extract_order(strip)
    rects = _infer_rectangles(scene, order)
    return _compose_output(order, rects)


# --------------- generator ---------------

ARC_MAX = 30
SCENE_H_RANGE = (8, 18)
SCENE_W_RANGE = (10, 24)
RECT_COUNT_RANGE = (2, 5)
PLACEMENT_ATTEMPTS = 240
GENERATION_ATTEMPTS = 600


def _sample_prompt_columns(width: int, count: int) -> list[int]:
    base = sorted(random.sample(range(width - (count - 1)), count))
    return [int(v + i) for i, v in enumerate(base)]


def _choose_rectangle_size(height: int, width: int) -> tuple[int, int]:
    options: list[tuple[int, int, int, int]] = []

    if height >= 2 and width >= 4:
        options.append((2, min(height, 4), 4, min(width, 8)))
    if height >= 4 and width >= 2:
        options.append((4, min(height, 8), 2, min(width, 4)))
    if height >= 2 and width >= 2:
        options.append((2, min(height, 7), 2, min(width, 7)))

    assert options, "scene must be large enough for rectangles"
    h0, h1, w0, w1 = random.choice(options)
    return random.randint(h0, h1), random.randint(w0, w1)


def _sample_bbox_under_mask(
    height: int,
    width: int,
    higher_mask: np.ndarray,
    force_overlap: bool,
) -> tuple[tuple[int, int, int, int] | None, int]:
    for _ in range(PLACEMENT_ATTEMPTS):
        rect_h, rect_w = _choose_rectangle_size(height, width)
        r0 = random.randint(0, height - rect_h)
        c0 = random.randint(0, width - rect_w)
        region = higher_mask[r0 : r0 + rect_h, c0 : c0 + rect_w]

        overlap = int(region.sum())
        if force_overlap and overlap == 0:
            continue

        visible = ~region
        area = rect_h * rect_w
        visible_count = int(visible.sum())
        if visible_count == 0:
            continue
        if visible_count < max(4, (area + 4) // 5):
            continue

        # The visible part must still witness all four bbox sides.
        if not visible[0, :].any():
            continue
        if not visible[-1, :].any():
            continue
        if not visible[:, 0].any():
            continue
        if not visible[:, -1].any():
            continue

        return (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1), overlap

    return None, 0


def _render_scene(
    height: int,
    width: int,
    layers: list[tuple[int, tuple[int, int, int, int]]],
) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]]]:
    scene = np.full((height, width), BG, dtype=int)
    rects: dict[int, tuple[int, int, int, int]] = {}

    for color, (r0, c0, r1, c1) in layers:
        scene[r0 : r1 + 1, c0 : c1 + 1] = int(color)
        rects[int(color)] = (int(r0), int(c0), int(r1), int(c1))

    return scene, rects


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*RECT_COUNT_RANGE)
    scene_h = random.randint(*SCENE_H_RANGE)
    scene_w = random.randint(max(SCENE_W_RANGE[0], 2 * count - 1), SCENE_W_RANGE[1])
    order = random.sample(COLOR_POOL, count)

    higher_mask = np.zeros((scene_h, scene_w), dtype=bool)
    top_down: list[tuple[int, tuple[int, int, int, int]]] = []
    overlap_layers = 0

    for idx, color in enumerate(reversed(order)):
        force_overlap = idx > 0 and random.random() < 0.75
        if idx == count - 1 and overlap_layers == 0:
            force_overlap = True

        bbox, overlap = _sample_bbox_under_mask(scene_h, scene_w, higher_mask, force_overlap)
        if bbox is None:
            return None

        if overlap > 0:
            overlap_layers += 1

        top_down.append((int(color), bbox))
        r0, c0, r1, c1 = bbox
        higher_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    if overlap_layers == 0:
        return None

    layers = list(reversed(top_down))
    scene, rects = _render_scene(scene_h, scene_w, layers)
    expected = _compose_output(order, rects)
    if expected.shape[0] > ARC_MAX or expected.shape[1] > ARC_MAX:
        return None

    strip = np.full(scene_w, BG, dtype=int)
    for col, color in zip(_sample_prompt_columns(scene_w, count), order, strict=False):
        strip[col] = int(color)

    inp = np.vstack([scene, strip[None, :]])
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GENERATION_ATTEMPTS} attempts")
