from __future__ import annotations

from collections import Counter, deque

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _touches_all_bbox_sides(coords: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        np.any(rows == r0)
        and np.any(rows == r1)
        and np.any(cols == c0)
        and np.any(cols == c1)
    )


def _find_rectangles(grid: np.ndarray, bg: int) -> dict[int, tuple[int, int, int, int]]:
    rectangles: dict[int, tuple[int, int, int, int]] = {}
    for color in colors_present(grid, exclude={bg}):
        coords = np.argwhere(grid == color)
        if len(coords) < 4:
            continue
        box = bbox_coords(coords)
        r0, c0, r1, c1 = box
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        if _touches_all_bbox_sides(coords, box):
            rectangles[int(color)] = box
    return rectangles


def _top_to_bottom_order(
    grid: np.ndarray, rectangles: dict[int, tuple[int, int, int, int]], bg: int
) -> list[int]:
    colors = list(rectangles)
    edges = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}

    for lower, box in rectangles.items():
        r0, c0, r1, c1 = box
        inside = set(map(int, grid[r0 : r1 + 1, c0 : c1 + 1].ravel()))
        inside -= {bg, lower}
        above_colors = inside & rectangles.keys()
        for upper in above_colors:
            if lower not in edges[upper]:
                edges[upper].add(lower)
                indegree[lower] += 1

    queue = deque(sorted(color for color, deg in indegree.items() if deg == 0))
    order: list[int] = []
    while queue:
        color = queue.popleft()
        order.append(color)
        for lower in sorted(edges[color]):
            indegree[lower] -= 1
            if indegree[lower] == 0:
                queue.append(lower)

    if len(order) != len(colors):
        raise AssertionError("rectangle occlusion graph must be acyclic")
    return order


def _ring_output(top_to_bottom: list[int]) -> np.ndarray:
    side = 2 * len(top_to_bottom) - 1
    layers = list(reversed(top_to_bottom))
    out = np.full((side, side), layers[0], dtype=int)
    for layer, color in enumerate(layers[1:], start=1):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the rectangle layer stack and redraw it as concentric square rings.

    Input structure:
    - A dominant background contains several solid axis-aligned colored rectangles
      that overlap by occlusion, plus one or two isolated noise pixels.
    - Each true rectangle color still has visible pixels on all four extreme sides
      of its full bounding box, so the visible-color bbox recovers the full rectangle.
    - Rectangle size and position do not determine depth; only visible overlaps do.

    Rule:
    - Ignore colors that do not look like partially visible rectangles.
    - For each rectangle color, take the bbox of its visible pixels as the full rectangle.
    - If another rectangle color appears inside that bbox, it lies above this rectangle;
      topologically sort these relations from top layer to bottom layer.
    - Output a square of side 2n-1 whose center is the top color and whose outermost
      ring is the bottom color.

    Key insight:
    - The visible fragments define complete rectangle extents, and occlusion depth is
      encoded by which other colors bite into each rectangle's bbox.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = get_bg(g)
    rectangles = _find_rectangles(g, bg)
    assert len(rectangles) >= 2, "expected at least two recoverable rectangles"

    order = _top_to_bottom_order(g, rectangles, bg)
    return _ring_output(order)


_GRID_SIDE_RANGE = (18, 27)
_RECT_COUNT_RANGE = (3, 5)
_RECT_HEIGHT_RANGE = (4, 10)
_RECT_WIDTH_RANGE = (4, 10)
_WITNESS_RADIUS = 4
_BORDER_PAD = 1
_MAX_ATTEMPTS = 400


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _sample_witness(
    rng: np.random.Generator,
    shape: tuple[int, int],
    higher_union: np.ndarray,
    witnesses: list[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = shape
    if not witnesses:
        return _randint(rng, 4, h - 5), _randint(rng, 4, w - 5)

    mean_r = int(round(sum(r for r, _ in witnesses) / len(witnesses)))
    mean_c = int(round(sum(c for _, c in witnesses) / len(witnesses)))
    candidates: list[tuple[int, int]] = []
    for radius in (_WITNESS_RADIUS, _WITNESS_RADIUS + 2, _WITNESS_RADIUS + 4):
        candidates.clear()
        r0 = max(_BORDER_PAD, mean_r - radius)
        r1 = min(h - 1 - _BORDER_PAD, mean_r + radius)
        c0 = max(_BORDER_PAD, mean_c - radius)
        c1 = min(w - 1 - _BORDER_PAD, mean_c + radius)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if not higher_union[r, c]:
                    candidates.append((r, c))
        if candidates:
            return candidates[int(rng.integers(len(candidates)))]
    return None


def _sample_rectangle(
    rng: np.random.Generator,
    shape: tuple[int, int],
    higher_union: np.ndarray,
    points: list[tuple[int, int]],
) -> tuple[int, int, int, int] | None:
    h, w = shape
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    min_height = max(_RECT_HEIGHT_RANGE[0], max_r - min_r + 1)
    max_height = min(_RECT_HEIGHT_RANGE[1], h - 2 * _BORDER_PAD)
    min_width = max(_RECT_WIDTH_RANGE[0], max_c - min_c + 1)
    max_width = min(_RECT_WIDTH_RANGE[1], w - 2 * _BORDER_PAD)
    if min_height > max_height or min_width > max_width:
        return None

    for _ in range(200):
        height = _randint(rng, min_height, max_height)
        width = _randint(rng, min_width, max_width)

        r0_lo = max(_BORDER_PAD, max_r - height + 1)
        r0_hi = min(min_r, h - _BORDER_PAD - height)
        c0_lo = max(_BORDER_PAD, max_c - width + 1)
        c0_hi = min(min_c, w - _BORDER_PAD - width)
        if r0_lo > r0_hi or c0_lo > c0_hi:
            continue

        r0 = _randint(rng, r0_lo, r0_hi)
        c0 = _randint(rng, c0_lo, c0_hi)
        r1 = r0 + height - 1
        c1 = c0 + width - 1

        visible = ~higher_union[r0 : r1 + 1, c0 : c1 + 1]
        if not (visible[0, :].any() and visible[-1, :].any() and visible[:, 0].any() and visible[:, -1].any()):
            continue
        return r0, c0, r1, c1
    return None


def _render_scene(
    shape: tuple[int, int],
    bg: int,
    rects_top_to_bottom: list[tuple[int, tuple[int, int, int, int]]],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for color, box in reversed(rects_top_to_bottom):
        r0, c0, r1, c1 = box
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return grid


def _isolated_bg_cells(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != bg:
                continue
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != bg:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                cells.append((r, c))
    return cells


def _scene_is_valid(
    grid: np.ndarray,
    rects_top_to_bottom: list[tuple[int, tuple[int, int, int, int]]],
    bg: int,
) -> bool:
    recovered = _find_rectangles(grid, bg)
    expected_colors = [color for color, _ in rects_top_to_bottom]
    if set(recovered) != set(expected_colors):
        return False
    if any(recovered[color] != box for color, box in rects_top_to_bottom):
        return False
    recovered_order = _top_to_bottom_order(grid, recovered, bg)
    return recovered_order == expected_colors


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = _randint(rng, *_GRID_SIDE_RANGE)
    w = _randint(rng, *_GRID_SIDE_RANGE)
    n_rects = _randint(rng, *_RECT_COUNT_RANGE)
    noise_count = _randint(rng, 1, 2)

    palette = list(rng.permutation(10))
    bg = int(palette[0])
    rect_colors = [int(c) for c in palette[1 : 1 + n_rects]]
    noise_colors = [int(c) for c in palette[1 + n_rects : 1 + n_rects + noise_count]]

    rects_top_to_bottom: list[tuple[int, tuple[int, int, int, int]]] = []
    witnesses: list[tuple[int, int]] = []
    higher_union = np.zeros((h, w), dtype=bool)

    for color in rect_colors:
        witness = _sample_witness(rng, (h, w), higher_union, witnesses)
        if witness is None:
            return None
        points = [*witnesses, witness]
        box = _sample_rectangle(rng, (h, w), higher_union, points)
        if box is None:
            return None
        rects_top_to_bottom.append((color, box))
        witnesses.append(witness)
        higher_union |= _bbox_mask((h, w), box)

    areas = [(box[2] - box[0] + 1) * (box[3] - box[1] + 1) for _, box in rects_top_to_bottom]
    if areas == sorted(areas) or areas == sorted(areas, reverse=True):
        return None
    if all(
        _bbox_contains(rects_top_to_bottom[j][1], rects_top_to_bottom[i][1])
        for i in range(n_rects)
        for j in range(i + 1, n_rects)
    ):
        return None

    grid = _render_scene((h, w), bg, rects_top_to_bottom)
    noise_cells = _isolated_bg_cells(grid, bg)
    if len(noise_cells) < noise_count:
        return None
    rng.shuffle(noise_cells)
    for color, (r, c) in zip(noise_colors, noise_cells[:noise_count]):
        grid[r, c] = int(color)

    if not _scene_is_valid(grid, rects_top_to_bottom, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
