from __future__ import annotations

import numpy as np


BACKGROUND = 0
GREY = 5
RING_COLORS = (1, 2, 3, 4, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _fills_bbox(coords: np.ndarray) -> bool:
    box = _bbox_from_coords(coords)
    return int(coords.shape[0]) == _bbox_h(box) * _bbox_w(box)


def _depth_grid(height: int, width: int) -> np.ndarray:
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    return np.minimum(
        np.minimum(rows, height - 1 - rows),
        np.minimum(cols, width - 1 - cols),
    )


def _extract_input_ring(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[int], np.ndarray]:
    ring_mask = (grid != BACKGROUND) & (grid != GREY)
    ring_box = _bbox_from_mask(ring_mask)
    assert ring_box is not None, "expected one concentric ring object"
    r0, c0, r1, c1 = ring_box
    ring_crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((ring_crop != BACKGROUND) & (ring_crop != GREY)), "ring bbox must be fully colored"

    depth = _depth_grid(*ring_crop.shape)
    assert int(depth.max()) == 2, "expected exactly three concentric layers"

    colors: list[int] = []
    for layer in range(3):
        vals = np.unique(ring_crop[depth == layer])
        assert vals.size == 1, "each ring layer must be monochrome"
        colors.append(int(vals[0]))

    assert len(set(colors)) == 3, "ring layers must use three distinct colors"
    assert set(colors).issubset(set(RING_COLORS)), "ring colors must come from the allowed visible palette"
    return ring_box, colors, depth


def _find_grey_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    comps = _components(grid == GREY)
    assert 1 <= len(comps) <= 2, "expected one or two grey rectangles"

    boxes: list[tuple[int, int, int, int]] = []
    for coords in comps:
        box = _bbox_from_coords(coords)
        r0, c0, r1, c1 = box
        assert _bbox_h(box) >= 3 and _bbox_w(box) >= 3, "grey rectangles must expose corners, edges, and interior"
        assert _fills_bbox(coords), "grey component must be a solid rectangle"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY), "grey rectangle bbox must be fully grey"
        boxes.append(box)
    return sorted(boxes)


def _recolor_rectangle(
    out: np.ndarray,
    box: tuple[int, int, int, int],
    corner_color: int,
    edge_color: int,
    interior_color: int,
) -> None:
    r0, c0, r1, c1 = box
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_h = r in (r0, r1)
            on_v = c in (c0, c1)
            if on_h and on_v:
                out[r, c] = corner_color
            elif on_h or on_v:
                out[r, c] = edge_color
            else:
                out[r, c] = interior_color


def _is_output_rectangle(crop: np.ndarray) -> bool:
    h, w = crop.shape
    if h < 3 or w < 3:
        return False
    corners = {int(crop[0, 0]), int(crop[0, -1]), int(crop[-1, 0]), int(crop[-1, -1])}
    if len(corners) != 1:
        return False
    edge_mask = np.zeros_like(crop, dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    edge_mask[0, 0] = False
    edge_mask[0, -1] = False
    edge_mask[-1, 0] = False
    edge_mask[-1, -1] = False
    edge_vals = np.unique(crop[edge_mask])
    if edge_vals.size != 1:
        return False
    interior_vals = np.unique(crop[1:-1, 1:-1])
    return interior_vals.size == 1


def _is_output_ring(crop: np.ndarray) -> bool:
    if np.any(crop == BACKGROUND):
        return False
    depth = _depth_grid(*crop.shape)
    if int(depth.max()) != 2:
        return False
    colors = []
    for layer in range(3):
        vals = np.unique(crop[depth == layer])
        if vals.size != 1:
            return False
        colors.append(int(vals[0]))
    return len(set(colors)) == 3


def _recover_input_from_output(grid: np.ndarray) -> np.ndarray:
    comps = _components(grid != BACKGROUND)
    assert 2 <= len(comps) <= 3, "output must contain one ring component and one or two rectangles"

    ring_box: tuple[int, int, int, int] | None = None
    rect_boxes: list[tuple[int, int, int, int]] = []
    for coords in comps:
        assert _fills_bbox(coords), "every non-background component must fill its bbox"
        box = _bbox_from_coords(coords)
        crop = grid[box[0] : box[2] + 1, box[1] : box[3] + 1]
        if _is_output_ring(crop):
            assert ring_box is None, "output must contain exactly one three-layer ring object"
            ring_box = box
        else:
            assert _is_output_rectangle(crop), "non-ring components must be classified rectangles"
            rect_boxes.append(box)

    assert ring_box is not None and 1 <= len(rect_boxes) <= 2
    recovered = grid.copy()

    r0, c0, r1, c1 = ring_box
    ring_crop = recovered[r0 : r1 + 1, c0 : c1 + 1]
    depth = _depth_grid(*ring_crop.shape)
    reversed_colors = [int(np.unique(ring_crop[depth == layer])[0]) for layer in range(3)]
    original_colors = reversed_colors[::-1]
    for layer, color in enumerate(original_colors):
        ring_crop[depth == layer] = color

    for box in rect_boxes:
        br0, bc0, br1, bc1 = box
        recovered[br0 : br1 + 1, bc0 : bc1 + 1] = GREY
    return recovered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the left concentric layer colors and use the reversed outer-to-inner trio to recolor the right grey rectangles.

    Input structure:
    - The grid has a black background.
    - On the left there is exactly one solid rectangular block made of three concentric one-cell-thick color layers with no gaps.
    - On the right there are one or two solid grey rectangles, each at least 3x3, separated from the ring area by black columns.

    Rule:
    - Read the left layered rectangle from outermost to innermost as colors [C1, C2, C3].
    - Reverse those layer colors in place so the left object becomes [C3, C2, C1] by depth.
    - Treat the reversed outer layer color as the corner color, the reversed middle layer color as the edge color, and the reversed inner layer color as the interior color.
    - Recolor every grey rectangle on the right by local position class: corners, remaining border, then strict interior.
    - Leave the black background unchanged.

    Key insight:
    - The ring reversal is not an independent effect: its reversed layer order is the lookup table that determines how rectangle corners, edges, and interiors must be colored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    ring_box, ring_colors, depth = _extract_input_ring(g)
    rect_boxes = _find_grey_rectangles(g)

    out = g.copy()

    reversed_colors = ring_colors[::-1]
    r0, c0, r1, c1 = ring_box
    ring_crop = out[r0 : r1 + 1, c0 : c1 + 1]
    for layer, color in enumerate(reversed_colors):
        ring_crop[depth == layer] = color

    for box in rect_boxes:
        _recolor_rectangle(out, box, reversed_colors[0], reversed_colors[1], reversed_colors[2])

    return out


def _random_ring_shape(rng: np.random.Generator) -> tuple[int, int]:
    valid_shapes = [
        (h, w)
        for h in range(5, 11)
        for w in range(5, 11)
        if int(_depth_grid(h, w).max()) == 2
    ]
    idx = int(rng.integers(len(valid_shapes)))
    return valid_shapes[idx]


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _render_ring(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    colors: list[int],
) -> None:
    r0, c0, r1, c1 = box
    depth = _depth_grid(_bbox_h(box), _bbox_w(box))
    patch = np.zeros((_bbox_h(box), _bbox_w(box)), dtype=int)
    for layer, color in enumerate(colors):
        patch[depth == layer] = color
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch


def _sample_rectangles(
    rng: np.random.Generator,
    height: int,
    width: int,
    start_col: int,
    count: int,
) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        placed = False
        for _ in range(100):
            rh = int(rng.integers(3, min(7, height - 1)))
            rw = int(rng.integers(3, min(7, width - start_col)))
            r0 = int(rng.integers(1, height - rh))
            c0 = int(rng.integers(start_col, width - rw))
            box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if all(_boxes_separated(box, prev, gap=1) for prev in boxes):
                boxes.append(box)
                placed = True
                break
        if not placed:
            return None
    return sorted(boxes)


def _expected_output_from_construction(
    inp: np.ndarray,
    ring_box: tuple[int, int, int, int],
    ring_colors: list[int],
    rect_boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = inp.copy()
    reversed_colors = ring_colors[::-1]
    r0, c0, r1, c1 = ring_box
    depth = _depth_grid(_bbox_h(ring_box), _bbox_w(ring_box))
    ring_crop = out[r0 : r1 + 1, c0 : c1 + 1]
    for layer, color in enumerate(reversed_colors):
        ring_crop[depth == layer] = color
    for box in rect_boxes:
        _recolor_rectangle(out, box, reversed_colors[0], reversed_colors[1], reversed_colors[2])
    return out


def _wrong_shortcut_output(
    inp: np.ndarray,
    ring_colors: list[int],
    rect_boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = inp.copy()
    for box in rect_boxes:
        _recolor_rectangle(out, box, ring_colors[0], ring_colors[1], ring_colors[2])
    return out


def _attempt_generate() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(12, 21))
    width = int(rng.integers(18, 27))
    grid = np.zeros((height, width), dtype=int)

    ring_h, ring_w = _random_ring_shape(rng)
    ring_r0 = int(rng.integers(1, height - ring_h))
    ring_c0 = int(rng.integers(1, 4))
    ring_box = (ring_r0, ring_c0, ring_r0 + ring_h - 1, ring_c0 + ring_w - 1)

    rect_count = 1 if rng.random() < 0.5 else 2
    rect_start_col = ring_box[3] + 3
    if width - rect_start_col < 4:
        return None

    rect_boxes = _sample_rectangles(rng, height, width, rect_start_col, rect_count)
    if rect_boxes is None:
        return None

    ring_colors = [int(c) for c in rng.choice(np.array(RING_COLORS), size=3, replace=False)]
    _render_ring(grid, ring_box, ring_colors)
    for box in rect_boxes:
        r0, c0, r1, c1 = box
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY

    expected = _expected_output_from_construction(grid, ring_box, ring_colors, rect_boxes)
    shortcut = _wrong_shortcut_output(grid, ring_colors, rect_boxes)

    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(shortcut, expected):
        return None
    if not np.array_equal(_recover_input_from_output(expected), grid):
        return None

    ring_component_count = len(_components((grid != BACKGROUND) & (grid != GREY)))
    if ring_component_count != 1:
        return None
    if len(_find_grey_rectangles(grid)) != rect_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
