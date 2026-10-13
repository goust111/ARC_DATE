from __future__ import annotations

from collections import Counter

import numpy as np


_RNG = np.random.default_rng()
_COLOR_IDS = np.arange(10, dtype=int)
_MAX_GENERATION_ATTEMPTS = 400

_GRID_MIN = 18
_GRID_MAX_EXCLUSIVE = 27
_OUTER_SIZE_MIN = 13
_OUTER_SIZE_MAX_EXCLUSIVE = 17


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _paint_filled(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _infer_frames(grid: np.ndarray) -> tuple[int, list[int], dict[int, tuple[int, int, int, int]], dict[int, np.ndarray]]:
    bg = _get_bg(grid)
    colors = _colors_present(grid, exclude={bg})
    assert len(colors) >= 3, "expected at least three rectangular frame colors"

    bboxes: dict[int, tuple[int, int, int, int]] = {}
    frames: dict[int, np.ndarray] = {}
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "each foreground color must remain visible"
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        assert r1 > r0 and c1 > c0, "each color must define a non-degenerate rectangle"
        frame = _frame_mask(grid.shape, bbox)
        assert np.all(frame[grid == color]), "visible pixels must lie on the reconstructed frame perimeter"
        bboxes[color] = bbox
        frames[color] = frame
    return bg, colors, bboxes, frames


def _crossing_edges(
    grid: np.ndarray,
    colors: list[int],
    frames: dict[int, np.ndarray],
) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for idx, a in enumerate(colors):
        for b in colors[idx + 1 :]:
            overlap = frames[a] & frames[b]
            if not np.any(overlap):
                continue
            visible = set(map(int, np.unique(grid[overlap])))
            evidence = visible & {a, b}
            assert len(evidence) <= 1, "a frame pair cannot show contradictory crossing directions"
            if len(evidence) == 1:
                top = next(iter(evidence))
                bottom = b if top == a else a
                edges.add((bottom, top))
    assert edges, "expected at least one visible frame crossing"
    return edges


def _reachability(colors: list[int], edges: set[tuple[int, int]]) -> dict[int, set[int]]:
    reach = {color: set() for color in colors}
    for src, dst in edges:
        reach[src].add(dst)

    changed = True
    while changed:
        changed = False
        for src in colors:
            expanded = set(reach[src])
            for mid in tuple(reach[src]):
                expanded |= reach[mid]
            if expanded != reach[src]:
                reach[src] = expanded
                changed = True
    for color in colors:
        assert color not in reach[color], "crossing order must be acyclic"
    return reach


def _unique_total_order(colors: list[int], edges: set[tuple[int, int]]) -> tuple[list[int], dict[int, set[int]]]:
    reach = _reachability(colors, edges)
    for idx, a in enumerate(colors):
        for b in colors[idx + 1 :]:
            assert (b in reach[a]) ^ (a in reach[b]), "crossings plus transitive closure must give a unique order"
    order = sorted(colors, key=lambda color: len(reach[color]), reverse=True)
    return order, reach


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _render_frames(shape: tuple[int, int], bboxes: dict[int, tuple[int, int, int, int]], order: list[int], bg: int) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for color in order:
        grid[_frame_mask(shape, bboxes[color])] = color
    return grid


def _render_solids(shape: tuple[int, int], bboxes: dict[int, tuple[int, int, int, int]], order: list[int], bg: int) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for color in order:
        _paint_filled(grid, bboxes[color], color)
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden stacking order of overlapping rectangular frames, then redraw them as solid stacked rectangles.

    Input structure:
    - The background contains several one-cell-thick axis-aligned rectangular frames, one color per frame.
    - Where two frame perimeters cross, only the upper frame's color remains visible at that crossing.
    - One pair of rectangles is in strict containment, so their perimeters never cross directly.
    - Each color still shows enough perimeter pixels to recover its full bounding box.

    Rule:
    - Reconstruct each rectangle from the bounding box of its visible pixels.
    - Read each visible crossing as a directed "lower frame -> upper frame" relation.
    - Take the transitive closure of those relations to obtain the unique bottom-to-top order, including the contained pair.
    - Fill each reconstructed rectangle solidly and paint them in that order on the original canvas, leaving the background unchanged.

    Key insight:
    - The missing relation is not visible directly: it must be inferred because the contained pair each crosses a third rectangle whose visible-over-hidden colors place that third rectangle between them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, colors, bboxes, frames = _infer_frames(g)
    edges = _crossing_edges(g, colors, frames)
    order, _reach = _unique_total_order(colors, edges)
    return _render_solids(g.shape, bboxes, order, bg)


def _sample_vertical_bridge(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    h, _w = shape
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner

    top_ext = int(_RNG.integers(1, min(4, or0 + 1)))
    bot_ext = int(_RNG.integers(1, min(4, h - or1)))
    br0 = or0 - top_ext
    br1 = or1 + bot_ext

    if bool(_RNG.integers(0, 2)):
        bc0 = int(_RNG.integers(oc0 + 1, ic0 - 1))
        bc1 = int(_RNG.integers(ic0 + 1, ic1))
    else:
        bc0 = int(_RNG.integers(ic0 + 1, ic1))
        bc1 = int(_RNG.integers(ic1 + 2, oc1))
    assert bc1 - bc0 >= 2, "bridge must remain a proper rectangle"
    return br0, bc0, br1, bc1


def _sample_horizontal_bridge(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    _h, w = shape
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner

    left_ext = int(_RNG.integers(1, min(4, oc0 + 1)))
    right_ext = int(_RNG.integers(1, min(4, w - oc1)))
    bc0 = oc0 - left_ext
    bc1 = oc1 + right_ext

    if bool(_RNG.integers(0, 2)):
        br0 = int(_RNG.integers(or0 + 1, ir0 - 1))
        br1 = int(_RNG.integers(ir0 + 1, ir1))
    else:
        br0 = int(_RNG.integers(ir0 + 1, ir1))
        br1 = int(_RNG.integers(ir1 + 2, or1))
    assert br1 - br0 >= 2, "bridge must remain a proper rectangle"
    return br0, bc0, br1, bc1


def _sample_bboxes(shape: tuple[int, int]) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]:
    h, w = shape
    for _ in range(80):
        outer_h = int(_RNG.integers(_OUTER_SIZE_MIN, min(_OUTER_SIZE_MAX_EXCLUSIVE, h - 3)))
        outer_w = int(_RNG.integers(_OUTER_SIZE_MIN, min(_OUTER_SIZE_MAX_EXCLUSIVE, w - 3)))
        or0 = int(_RNG.integers(2, h - outer_h - 1))
        oc0 = int(_RNG.integers(2, w - outer_w - 1))
        outer = (or0, oc0, or0 + outer_h - 1, oc0 + outer_w - 1)

        vertical_gaps = [
            (top_gap, bottom_gap)
            for top_gap in range(3, 6)
            for bottom_gap in range(3, 6)
            if outer_h - top_gap - bottom_gap >= 5
        ]
        horizontal_gaps = [
            (left_gap, right_gap)
            for left_gap in range(3, 6)
            for right_gap in range(3, 6)
            if outer_w - left_gap - right_gap >= 5
        ]
        if vertical_gaps and horizontal_gaps:
            top_gap, bottom_gap = vertical_gaps[int(_RNG.integers(len(vertical_gaps)))]
            left_gap, right_gap = horizontal_gaps[int(_RNG.integers(len(horizontal_gaps)))]
            inner = (
                outer[0] + top_gap,
                outer[1] + left_gap,
                outer[2] - bottom_gap,
                outer[3] - right_gap,
            )
            break
    else:
        raise AssertionError("failed to place a valid contained rectangle")

    if bool(_RNG.integers(0, 2)):
        bridge = _sample_vertical_bridge(outer, inner, shape)
    else:
        bridge = _sample_horizontal_bridge(outer, inner, shape)
    return outer, inner, bridge


def _scene_is_valid(
    inp: np.ndarray,
    expected: np.ndarray,
    outer_color: int,
    bridge_color: int,
    inner_color: int,
    chosen_order: list[int],
) -> bool:
    try:
        bg, colors, bboxes, frames = _infer_frames(inp)
        edges = _crossing_edges(inp, colors, frames)
        order, reach = _unique_total_order(colors, edges)
        pred = main(inp)
    except Exception:
        return False

    if order != chosen_order:
        return False
    if not np.array_equal(pred, expected):
        return False

    outer_bbox = bboxes[outer_color]
    inner_bbox = bboxes[inner_color]
    bridge_bbox = bboxes[bridge_color]
    if not _bbox_contains(outer_bbox, inner_bbox):
        return False
    if _bbox_contains(outer_bbox, bridge_bbox) or _bbox_contains(bridge_bbox, inner_bbox) or _bbox_contains(inner_bbox, bridge_bbox):
        return False

    pair_masks = {
        frozenset((outer_color, inner_color)): frames[outer_color] & frames[inner_color],
        frozenset((outer_color, bridge_color)): frames[outer_color] & frames[bridge_color],
        frozenset((inner_color, bridge_color)): frames[inner_color] & frames[bridge_color],
    }
    if np.any(pair_masks[frozenset((outer_color, inner_color))]):
        return False
    if not np.any(pair_masks[frozenset((outer_color, bridge_color))]):
        return False
    if not np.any(pair_masks[frozenset((inner_color, bridge_color))]):
        return False

    if (outer_color, inner_color) in edges or (inner_color, outer_color) in edges:
        return False

    lower = chosen_order[0]
    middle = chosen_order[1]
    upper = chosen_order[2]
    if middle != bridge_color:
        return False
    if middle not in reach[lower] or upper not in reach[middle]:
        return False
    if upper not in reach[lower]:
        return False

    nested_pairs = 0
    bbox_values = list(bboxes.values())
    for i, a in enumerate(bbox_values):
        for b in bbox_values[i + 1 :]:
            if _bbox_contains(a, b) or _bbox_contains(b, a):
                nested_pairs += 1
    if nested_pairs != 1:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(_GRID_MIN, _GRID_MAX_EXCLUSIVE))
    w = int(_RNG.integers(_GRID_MIN, _GRID_MAX_EXCLUSIVE))
    bg, outer_color, bridge_color, inner_color = map(int, _RNG.choice(_COLOR_IDS, size=4, replace=False))

    outer_bbox, inner_bbox, bridge_bbox = _sample_bboxes((h, w))
    bboxes = {
        outer_color: outer_bbox,
        bridge_color: bridge_bbox,
        inner_color: inner_bbox,
    }

    if bool(_RNG.integers(0, 2)):
        order = [outer_color, bridge_color, inner_color]
    else:
        order = [inner_color, bridge_color, outer_color]

    inp = _render_frames((h, w), bboxes, order, bg)
    expected = _render_solids((h, w), bboxes, order, bg)

    if np.array_equal(inp, expected):
        return None
    if _scene_is_valid(inp, expected, outer_color, bridge_color, inner_color, order):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid containment-via-bridge frame scene")
