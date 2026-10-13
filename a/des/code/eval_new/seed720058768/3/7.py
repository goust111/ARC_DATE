from __future__ import annotations

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _parse_legend(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[int]]:
    assert grid.ndim == 2 and grid.shape[1] >= 2, "expected at least two columns"
    legend_col = grid[:, -1]
    legend_rows = np.flatnonzero(legend_col != bg)
    assert legend_rows.size > 0, "missing right-edge color sequence"
    if legend_rows.size > 1:
        diffs = np.diff(legend_rows)
        assert np.all(diffs == 1), "legend cells must form one vertical run"

    legend_colors = [int(legend_col[r]) for r in legend_rows]
    assert len(legend_colors) == len(set(legend_colors)), "legend colors must be unique"

    scene = np.asarray(grid[:, :-1], dtype=int)
    scene_colors = set(map(int, np.unique(scene[scene != bg])))
    assert scene_colors == set(legend_colors), "legend must list every scene rectangle color"
    return scene, legend_colors


def _intersection(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    r0 = max(box_a[0], box_b[0])
    c0 = max(box_a[1], box_b[1])
    r1 = min(box_a[2], box_b[2])
    c1 = min(box_a[3], box_b[3])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _verify_layer_order(
    scene: np.ndarray,
    legend_colors: list[int],
    rects: dict[int, tuple[int, int, int, int]],
) -> None:
    order_index = {color: idx for idx, color in enumerate(legend_colors)}
    for idx_a, color_a in enumerate(legend_colors):
        for color_b in legend_colors[idx_a + 1 :]:
            inter = _intersection(rects[color_a], rects[color_b])
            if inter is None:
                continue
            r0, c0, r1, c1 = inter
            window = scene[r0 : r1 + 1, c0 : c1 + 1]
            a_seen = bool(np.any(window == color_a))
            b_seen = bool(np.any(window == color_b))
            assert a_seen != b_seen, "overlap must expose exactly one rectangle color"
            if a_seen:
                assert order_index[color_a] < order_index[color_b], "legend order contradicts occlusion"
            else:
                assert order_index[color_b] < order_index[color_a], "legend order contradicts occlusion"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the fully hidden extent of the back-most rectangle and output it alone.

    Input structure:
    - The dominant color is the black background.
    - Several unique non-background colors appear as solid axis-aligned rectangles
      that partially overlap, so lower rectangles are only visible as truncated
      L-shapes, U-shapes, or bars.
    - The rightmost grid column contains a vertical sequence of one-pixel color
      markers listing the rectangle colors from front-most at the top to back-most
      at the bottom.

    Rule:
    - Read the right-edge color sequence and separate it from the rectangle scene.
    - For every rectangle color, infer its visible bounding box from the remaining
      pixels and use bbox overlaps to confirm that any visible overlap agrees with
      the legend's front-to-back layer order.
    - Take the last legend color as the bottom rectangle, recover its full extent
      from the minimum and maximum row and column of its visible pixels, and output
      that bounding box filled uniformly with the same color.

    Key insight:
    - The side legend identifies which color belongs to the deepest layer, but the
      rectangle itself is reconstructed from the spatial span of its surviving
      visible pixels rather than from the legend geometry.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    scene, legend_colors = _parse_legend(g, bg)

    rects = {
        color: _bbox_from_mask(scene == color)
        for color in legend_colors
    }
    _verify_layer_order(scene, legend_colors, rects)

    bottom_color = legend_colors[-1]
    r0, c0, r1, c1 = rects[bottom_color]
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bottom_color, dtype=int)
    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

_BG = 0
_PALETTE = tuple(range(1, 10))
_LAYER_COUNT_RANGE = (2, 5)
_TOP_SIZE_RANGE = (3, 7)
_MANDATORY_MARGIN_RANGE = (1, 4)
_OPTIONAL_MARGIN_RANGE = (0, 4)
_SCENE_MAX_SIDE = 24
_SCENE_TOP_EXTRA = 4
_SCENE_LEFT_EXTRA = 4
_SCENE_HEIGHT_EXTRA = 4
_LEGEND_GAP_RANGE = (1, 2)
_MAX_ATTEMPTS = 400
_CORNERS = ("TL", "TR", "BL", "BR")


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(lo, hi + 1))


def _wrap_boxes(
    boxes: list[tuple[int, int, int, int]],
    protected_corner: str,
) -> list[tuple[int, int, int, int]]:
    top = _randint(1, _MANDATORY_MARGIN_RANGE[1]) if protected_corner[0] == "T" else _randint(*_OPTIONAL_MARGIN_RANGE)
    bottom = _randint(1, _MANDATORY_MARGIN_RANGE[1]) if protected_corner[0] == "B" else _randint(*_OPTIONAL_MARGIN_RANGE)
    left = _randint(1, _MANDATORY_MARGIN_RANGE[1]) if protected_corner[1] == "L" else _randint(*_OPTIONAL_MARGIN_RANGE)
    right = _randint(1, _MANDATORY_MARGIN_RANGE[1]) if protected_corner[1] == "R" else _randint(*_OPTIONAL_MARGIN_RANGE)

    shifted = []
    for r0, c0, r1, c1 in boxes:
        shifted.append((r0 + top, c0 + left, r1 + top, c1 + left))

    max_r = max(r1 for _, _, r1, _ in shifted)
    max_c = max(c1 for _, _, _, c1 in shifted)
    outer_box = (0, 0, max_r + bottom, max_c + right)
    shifted.append(outer_box)
    return shifted


def _count_visible_edges(
    scene: np.ndarray,
    box: tuple[int, int, int, int],
    color: int,
) -> int:
    r0, c0, r1, c1 = box
    count = 0
    if np.all(scene[r0, c0 : c1 + 1] == color):
        count += 1
    if np.all(scene[r1, c0 : c1 + 1] == color):
        count += 1
    if np.all(scene[r0 : r1 + 1, c0] == color):
        count += 1
    if np.all(scene[r0 : r1 + 1, c1] == color):
        count += 1
    return count


def _build_nested_boxes(layer_count: int) -> list[tuple[int, int, int, int]] | None:
    top_h = _randint(*_TOP_SIZE_RANGE)
    top_w = _randint(*_TOP_SIZE_RANGE)
    boxes = [(0, 0, top_h - 1, top_w - 1)]

    for _ in range(layer_count - 1):
        boxes = _wrap_boxes(boxes, _RNG.choice(_CORNERS))
        outer_h = boxes[-1][2] + 1
        outer_w = boxes[-1][3] + 1
        if max(outer_h, outer_w) > _SCENE_MAX_SIDE:
            return None
    return boxes


def _render_candidate(
    boxes: list[tuple[int, int, int, int]],
    colors: list[int],
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], int] | None:
    scene_h = boxes[-1][2] + 1
    scene_w = boxes[-1][3] + 1
    gap = _randint(*_LEGEND_GAP_RANGE)

    max_left = 30 - (scene_w + gap + 1)
    if max_left < 0:
        return None
    scene_left = _randint(0, min(_SCENE_LEFT_EXTRA, max_left))
    grid_w = scene_left + scene_w + gap + 1
    if grid_w > 30:
        return None

    min_h = max(scene_h, len(colors))
    max_h = min(30, scene_h + _SCENE_HEIGHT_EXTRA)
    if min_h > max_h:
        return None
    grid_h = _randint(min_h, max_h)
    scene_top = _randint(0, grid_h - scene_h)
    legend_top = _randint(0, grid_h - len(colors))

    grid = np.full((grid_h, grid_w), _BG, dtype=int)
    placed_boxes = []
    for color, box in zip(reversed(colors), reversed(boxes)):
        r0, c0, r1, c1 = box
        rr0 = scene_top + r0
        cc0 = scene_left + c0
        rr1 = scene_top + r1
        cc1 = scene_left + c1
        grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = color
        placed_boxes.append((rr0, cc0, rr1, cc1))
    placed_boxes.reverse()

    for idx, color in enumerate(colors):
        grid[legend_top + idx, -1] = color
    return grid, placed_boxes, gap


def _validate_candidate(
    grid: np.ndarray,
    colors: list[int],
    placed_boxes: list[tuple[int, int, int, int]],
) -> bool:
    scene = grid[:, :-1]
    if set(map(int, np.unique(scene[scene != _BG]))) != set(colors):
        return False

    for color, box in zip(colors, placed_boxes):
        mask = scene == color
        if not np.any(mask):
            return False
        if _bbox_from_mask(mask) != box:
            return False
        if _count_visible_edges(scene, box, color) < 2:
            return False

    bottom_box = placed_boxes[-1]
    expected = np.full(
        (bottom_box[2] - bottom_box[0] + 1, bottom_box[3] - bottom_box[1] + 1),
        colors[-1],
        dtype=int,
    )
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    layer_count = _randint(*_LAYER_COUNT_RANGE)
    colors = list(map(int, _RNG.choice(_PALETTE, size=layer_count, replace=False)))
    boxes = _build_nested_boxes(layer_count)
    if boxes is None:
        return None

    rendered = _render_candidate(boxes, colors)
    if rendered is None:
        return None
    grid, placed_boxes, _ = rendered

    if not _validate_candidate(grid, colors, placed_boxes):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rectangle-layer scene")
