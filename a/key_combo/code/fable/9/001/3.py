from __future__ import annotations

import random
from collections import Counter

import numpy as np


BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _perimeter_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _box_of_color(grid: np.ndarray, color: int) -> BBox:
    coords = np.argwhere(grid == color)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    box = (int(r0), int(c0), int(r1), int(c1))
    assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3
    perim = _perimeter_mask(grid.shape, box)
    assert np.all(perim[coords[:, 0], coords[:, 1]])
    return box


def _frames(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    colors = sorted(int(v) for v in np.unique(grid) if int(v) != bg)
    frames = []
    for color in colors:
        box = _box_of_color(grid, color)
        frames.append(
            {
                "color": color,
                "bbox": box,
                "mask": _perimeter_mask(grid.shape, box),
            }
        )
    return frames


def _direct_relations(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    n = len(frames)
    above = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            overlap = frames[i]["mask"] & frames[j]["mask"]
            if not np.any(overlap):
                continue
            colors_seen = {int(v) for v in np.unique(grid[overlap])}
            ci = frames[i]["color"]
            cj = frames[j]["color"]
            seen_i = ci in colors_seen
            seen_j = cj in colors_seen
            assert seen_i != seen_j
            if seen_i:
                above[i, j] = True
            else:
                above[j, i] = True
    return above


def _transitive_closure(above: np.ndarray) -> np.ndarray:
    reach = above.copy()
    n = reach.shape[0]
    for k in range(n):
        for i in range(n):
            if reach[i, k]:
                reach[i] |= reach[k]
    return reach


def _bottom_index(reach: np.ndarray) -> int:
    n = reach.shape[0]
    candidates = [j for j in range(n) if all(i == j or reach[i, j] for i in range(n))]
    assert len(candidates) == 1
    return candidates[0]


def _paint_perimeter(grid: np.ndarray, box: BBox, color: int) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = box
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the unique globally lowest hollow rectangle and redraw its full frame on top.

    Input structure:
    - The background contains several hollow rectangular frames of distinct colors.
    - Frames overlap only at border crossing cells, where the visible color belongs to the upper frame and the lower frame shows a gap.
    - Not every pair of frames crosses directly, so the comparison graph is only partially observed.

    Rule:
    - Recover each frame's intended bounding rectangle from its visible border fragments.
    - Read each direct upper/lower relation from every border-crossing cell, then propagate those relations transitively to identify the unique globally lowest frame.
    - Redraw only that frame's complete border as the new top layer, filling all of its missing crossing gaps and leaving every other pixel unchanged.

    Key insight:
    - A missing border segment is not damage but occlusion evidence: the visible crossing color tells which frame is above, and chaining those pairwise clues reveals the one frame that is below all others.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frames = _frames(g)
    assert len(frames) >= 2
    reach = _transitive_closure(_direct_relations(g, frames))
    bottom = frames[_bottom_index(reach)]
    return _paint_perimeter(g, bottom["bbox"], bottom["color"])


def _sample_snake_rects(height: int, width: int, n_frames: int) -> list[BBox] | None:
    rects: list[BBox] = []
    direction = random.choice((1, -1))
    row = random.randint(1, 3)
    prev_h_top = prev_h_bottom = None
    v_left = v_right = None

    for idx in range(n_frames):
        if idx % 2 == 0:
            short = random.randint(3, 5)
            long = random.randint(10, min(16, width - 4))
            if idx == 0:
                if direction == 1:
                    left = random.randint(1, 3)
                    right = left + long - 1
                else:
                    right = width - 2 - random.randint(0, 2)
                    left = right - long + 1
            elif direction == 1:
                left = max(1, v_left - random.randint(1, 2))
                right = left + long - 1
            else:
                right = min(width - 2, v_right + random.randint(1, 2))
                left = right - long + 1

            top = row
            bottom = top + short - 1
            if left < 1 or right > width - 2 or bottom > height - 2:
                return None
            rects.append((top, left, bottom, right))
            prev_h_top, prev_h_bottom = top, bottom
            h_left, h_right = left, right
        else:
            short = random.randint(3, 5)
            long = random.randint(10, min(14, height - 4))
            if direction == 1:
                right = h_right - random.randint(0, 2)
                left = right - short + 1
            else:
                left = h_left + random.randint(0, 2)
                right = left + short - 1

            top = max(1, prev_h_top - random.randint(1, 2))
            bottom = top + long - 1
            if left < 1 or right > width - 2 or bottom > height - 2:
                return None
            rects.append((top, left, bottom, right))
            v_left, v_right = left, right
            row = bottom - random.randint(2, 4)
            direction *= -1

    return rects


def _transform_rects(rects: list[BBox], shape: tuple[int, int], mode: int) -> tuple[list[BBox], tuple[int, int]]:
    h, w = shape
    transformed: list[BBox] = []
    swap = mode >= 4
    for r0, c0, r1, c1 in rects:
        if mode == 0:
            nr0, nc0, nr1, nc1 = r0, c0, r1, c1
        elif mode == 1:
            nr0, nc0, nr1, nc1 = r0, w - 1 - c1, r1, w - 1 - c0
        elif mode == 2:
            nr0, nc0, nr1, nc1 = h - 1 - r1, c0, h - 1 - r0, c1
        elif mode == 3:
            nr0, nc0, nr1, nc1 = h - 1 - r1, w - 1 - c1, h - 1 - r0, w - 1 - c0
        elif mode == 4:
            nr0, nc0, nr1, nc1 = c0, r0, c1, r1
        elif mode == 5:
            nr0, nc0, nr1, nc1 = c0, h - 1 - r1, c1, h - 1 - r0
        elif mode == 6:
            nr0, nc0, nr1, nc1 = w - 1 - c1, r0, w - 1 - c0, r1
        else:
            nr0, nc0, nr1, nc1 = w - 1 - c1, h - 1 - r1, w - 1 - c0, h - 1 - r0
        transformed.append((int(nr0), int(nc0), int(nr1), int(nc1)))
    return transformed, (w, h) if swap else (h, w)


def _perimeter_cells(box: BBox) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _intersection_edges(rects: list[BBox]) -> tuple[set[tuple[int, int]], int]:
    perimeters = [_perimeter_cells(box) for box in rects]
    cover_count: Counter[tuple[int, int]] = Counter()
    for cells in perimeters:
        cover_count.update(cells)

    edges: set[tuple[int, int]] = set()
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if perimeters[i] & perimeters[j]:
                edges.add((i, j))
    max_cover = max(cover_count.values(), default=0)
    return edges, max_cover


def _render_scene(shape: tuple[int, int], rects: list[BBox], colors: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for box, color in zip(reversed(rects), reversed(colors)):
        r0, c0, r1, c1 = box
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    return out


def _valid_geometry(rects: list[BBox]) -> bool:
    edges, max_cover = _intersection_edges(rects)
    n = len(rects)
    expected = {(i, i + 1) for i in range(n - 1)}
    if edges != expected:
        return False
    if max_cover > 2:
        return False
    bottom_box = rects[-1]
    bottom_mask = _perimeter_cells(bottom_box)
    visible = bottom_mask.copy()
    for box in rects[:-1]:
        visible -= _perimeter_cells(box)
    if len(visible) < max(6, len(bottom_mask) // 2):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    base_h = random.randint(22, 28)
    base_w = random.randint(22, 28)
    n_frames = random.randint(4, 6)
    rects = _sample_snake_rects(base_h, base_w, n_frames)
    if rects is None or not _valid_geometry(rects):
        return None

    mode = random.randrange(8)
    rects, shape = _transform_rects(rects, (base_h, base_w), mode)
    if not _valid_geometry(rects):
        return None

    colors = random.sample(list(range(1, 10)), n_frames)
    inp = _render_scene(shape, rects, colors)
    out = _paint_perimeter(inp, rects[-1], colors[-1])
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
