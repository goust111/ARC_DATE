import random

import numpy as np


BLACK = 0
GREY = 5
LEGEND_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
LAYER_OPTIONS = (3, 4, 5)
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _outline_perimeter(height: int, width: int) -> int:
    return 2 * (height + width) - 4


def _is_rectangular_ring(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    if int(coords.shape[0]) != _outline_perimeter(height, width):
        return False

    rel = coords - np.array([r0, c0], dtype=int)
    on_border = (
        (rel[:, 0] == 0)
        | (rel[:, 0] == height - 1)
        | (rel[:, 1] == 0)
        | (rel[:, 1] == width - 1)
    )
    return bool(np.all(on_border))


def _is_nested_ring_chain(coords_list: list[np.ndarray]) -> bool:
    if not coords_list:
        return False

    boxes = [_bbox(coords) for coords in coords_list]
    areas = [
        (r1 - r0 + 1) * (c1 - c0 + 1)
        for r0, c0, r1, c1 in boxes
    ]
    if len(set(areas)) != len(areas):
        return False

    ordered = [box for _area, box in sorted(zip(areas, boxes), reverse=True)]
    for outer, inner in zip(ordered, ordered[1:]):
        if (
            inner[0] - outer[0],
            inner[1] - outer[1],
            outer[2] - inner[2],
            outer[3] - inner[3],
        ) != (2, 2, 2, 2):
            return False
    return True


def _ordered_rings(grid: np.ndarray, legend_len: int) -> list[np.ndarray]:
    body = grid[:, :-1]
    candidates: list[tuple[int, list[np.ndarray]]] = []
    for color in np.unique(body):
        color = int(color)
        if color == BLACK:
            continue
        comps = [coords for coords in _components(body == color) if _is_rectangular_ring(coords)]
        if len(comps) != legend_len:
            continue
        if not _is_nested_ring_chain(comps):
            continue
        score = sum(int(coords.shape[0]) for coords in comps)
        candidates.append((score, comps))

    assert candidates, "expected one nested ring set"
    comps = max(candidates, key=lambda item: item[0])[1]
    return sorted(
        comps,
        key=lambda coords: (
            -(_bbox(coords)[2] - _bbox(coords)[0] + 1) * (_bbox(coords)[3] - _bbox(coords)[1] + 1),
            _bbox(coords)[0],
            _bbox(coords)[1],
        ),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each nested rectangular outline ring from outer to inner using the right-edge legend, then clear the legend.

    Input structure:
    - The background is black.
    - One group of concentric nested rectangular outline rings appears in the main body; all rings share the same grey color and are separated by one black pixel gap.
    - The rightmost column is a vertical legend made of distinct non-black pixels separated by black gaps.

    Rule:
    - Read the legend colors from top to bottom.
    - Identify the nested ring layers from outermost to innermost by their bounding boxes.
    - Repaint layer 1 with legend color 1, layer 2 with legend color 2, and so on, then erase the legend column to black.

    Key insight:
    - The rings are not distinguished by their input color, so the solver must infer layer order from geometric nesting before applying the legend's top-to-bottom color mapping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert width >= 2

    legend = [int(g[r, width - 1]) for r in range(height) if int(g[r, width - 1]) != BLACK]
    assert 3 <= len(legend) <= 5
    assert len(legend) == len(set(legend))

    rings = _ordered_rings(g, len(legend))

    out = g.copy()
    out[:, width - 1] = BLACK
    for color, coords in zip(legend, rings):
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _sample_rows(height: int, count: int) -> list[int]:
    for _ in range(200):
        rows = sorted(random.sample(range(height), count))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            return rows
    raise RuntimeError("failed to sample legend rows")


def _draw_ring(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = int(color)
    grid[bottom, left : right + 1] = int(color)
    grid[top : bottom + 1, left] = int(color)
    grid[top : bottom + 1, right] = int(color)


def _expected_output(
    shape: tuple[int, int],
    ring_boxes: list[tuple[int, int, int, int]],
    colors: list[int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for color, (top, left, ring_h, ring_w) in zip(colors, ring_boxes):
        _draw_ring(out, top, left, ring_h, ring_w, color)
    return out


def _gen_attempt() -> np.ndarray | None:
    layers = random.choice(LAYER_OPTIONS)

    max_inner_h = min(9, 30 - 4 * (layers - 1))
    max_inner_w = min(11, 28 - 4 * (layers - 1))
    if max_inner_h < 3 or max_inner_w < 3:
        return None

    inner_h = random.randint(3, max_inner_h)
    inner_w = random.randint(3, max_inner_w)
    outer_h = inner_h + 4 * (layers - 1)
    outer_w = inner_w + 4 * (layers - 1)

    min_h = max(outer_h, 2 * layers - 1)
    min_w = outer_w + 2
    if min_h > 30 or min_w > 30:
        return None

    height = random.randint(min_h, min(30, min_h + 6))
    width = random.randint(min_w, min(30, min_w + 6))
    top = random.randint(0, height - outer_h)
    left = random.randint(0, width - outer_w - 2)

    legend_rows = _sample_rows(height, layers)
    legend_colors = random.sample(LEGEND_COLORS, layers)

    grid = np.zeros((height, width), dtype=int)
    ring_boxes: list[tuple[int, int, int, int]] = []
    for layer in range(layers):
        ring_top = top + 2 * layer
        ring_left = left + 2 * layer
        ring_h = outer_h - 4 * layer
        ring_w = outer_w - 4 * layer
        _draw_ring(grid, ring_top, ring_left, ring_h, ring_w, GREY)
        ring_boxes.append((ring_top, ring_left, ring_h, ring_w))

    for row, color in zip(legend_rows, legend_colors):
        grid[row, width - 1] = int(color)

    expected = _expected_output((height, width), ring_boxes, legend_colors)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring input")
