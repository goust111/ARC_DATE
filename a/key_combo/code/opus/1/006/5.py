from __future__ import annotations

import random

import numpy as np


BLACK = 0
RANK_FILL_COLORS = (1, 2, 3, 4, 5)
FRAME_PALETTE = (3, 4, 6, 7, 8)

MIN_FRAME_SIZE = 5
MIN_FRAME_COUNT = 3
MAX_FRAME_COUNT = 5
MAX_SCENE_SOLUTIONS = 2

GEN_ATTEMPTS = 400
GEN_STEP_RANGE = (1, 2)
GEN_HEIGHT_RANGE_PAD = (2, 5)
GEN_WIDTH_RANGE_PAD = (2, 5)
GEN_MARGIN_RANGE = (2, 4)


def _border_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _interior_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    mask = np.zeros(shape, dtype=bool)
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _zero_prefixes(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    zeros = (np.asarray(grid, dtype=int) == BLACK).astype(np.int16)
    row_prefix = np.pad(np.cumsum(zeros, axis=1), ((0, 0), (1, 0)))
    col_prefix = np.pad(np.cumsum(zeros, axis=0), ((1, 0), (0, 0)))
    return row_prefix, col_prefix


def _row_zero_count(prefix: np.ndarray, row: int, c0: int, c1: int) -> int:
    return int(prefix[row, c1 + 1] - prefix[row, c0])


def _col_zero_count(prefix: np.ndarray, col: int, r0: int, r1: int) -> int:
    return int(prefix[r1 + 1, col] - prefix[r0, col])


def _border_is_fully_nonzero(
    row_prefix: np.ndarray, col_prefix: np.ndarray, box: tuple[int, int, int, int]
) -> bool:
    r0, c0, r1, c1 = map(int, box)
    zero_count = _row_zero_count(row_prefix, r0, c0, c1) + _row_zero_count(row_prefix, r1, c0, c1)
    if r1 - r0 > 1:
        zero_count += _col_zero_count(col_prefix, c0, r0 + 1, r1 - 1)
        zero_count += _col_zero_count(col_prefix, c1, r0 + 1, r1 - 1)
    return zero_count == 0


def _enumerate_candidates(
    grid: np.ndarray, color: int, row_prefix: np.ndarray, col_prefix: np.ndarray
) -> list[dict[str, object]]:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, f"missing visible pixels for color {color}"

    rows = coords[:, 0]
    cols = coords[:, 1]
    min_r, max_r = int(rows.min()), int(rows.max())
    min_c, max_c = int(cols.min()), int(cols.max())
    height, width = map(int, grid.shape)

    candidates: list[dict[str, object]] = []
    for r0 in range(0, min_r + 1):
        min_r1 = max(max_r, r0 + MIN_FRAME_SIZE - 1)
        for r1 in range(min_r1, height):
            for c0 in range(0, min_c + 1):
                min_c1 = max(max_c, c0 + MIN_FRAME_SIZE - 1)
                for c1 in range(min_c1, width):
                    inside = (r0 <= rows) & (rows <= r1) & (c0 <= cols) & (cols <= c1)
                    on_border = (rows == r0) | (rows == r1) | (cols == c0) | (cols == c1)
                    if not np.all(inside & on_border):
                        continue
                    box = (r0, c0, r1, c1)
                    if not _border_is_fully_nonzero(row_prefix, col_prefix, box):
                        continue
                    candidates.append({"box": box, "mask": _border_mask(grid.shape, box)})
    return candidates


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int] | None:
    successors = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for low, high in edges:
        if high not in successors[low]:
            successors[low].add(high)
            indegree[high] += 1

    remaining = set(colors)
    order: list[int] = []
    while remaining:
        bottoms = sorted(color for color in remaining if indegree[color] == 0)
        if len(bottoms) != 1:
            return None
        color = bottoms[0]
        order.append(color)
        remaining.remove(color)
        for nxt in successors[color]:
            indegree[nxt] -= 1
    return order


def _scene_from_choice(
    grid: np.ndarray, colors: list[int], chosen: dict[int, dict[str, object]]
) -> dict[str, object] | None:
    masks = {color: np.asarray(chosen[color]["mask"], dtype=bool) for color in colors}
    edges: set[tuple[int, int]] = set()

    for r, c in np.argwhere(grid != BLACK):
        r = int(r)
        c = int(c)
        visible = int(grid[r, c])
        covering = [color for color in colors if masks[color][r, c]]
        if not covering or visible not in covering:
            return None
        for color in covering:
            if color != visible:
                edges.add((color, visible))

    order = _unique_topological_order(colors, edges)
    if order is None:
        return None

    return {
        "boxes": {color: tuple(map(int, chosen[color]["box"])) for color in colors},
        "order": order,
    }


def _partial_choice_is_feasible(
    grid: np.ndarray, chosen: dict[int, dict[str, object]], visible_cells: np.ndarray
) -> bool:
    chosen_colors = list(chosen)
    if len(chosen_colors) < 2:
        return True

    masks = {color: np.asarray(chosen[color]["mask"], dtype=bool) for color in chosen_colors}
    edges: set[tuple[int, int]] = set()
    for r, c in visible_cells:
        r = int(r)
        c = int(c)
        visible = int(grid[r, c])
        if visible not in chosen:
            continue
        covering = [color for color in chosen_colors if masks[color][r, c]]
        if visible not in covering:
            return False
        for color in covering:
            if color != visible:
                edges.add((color, visible))

    return _has_acyclic_extension(chosen_colors, edges)


def _has_acyclic_extension(colors: list[int], edges: set[tuple[int, int]]) -> bool:
    successors = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for low, high in edges:
        if high not in successors[low]:
            successors[low].add(high)
            indegree[high] += 1

    remaining = set(colors)
    ready = [color for color in colors if indegree[color] == 0]
    while ready:
        color = ready.pop()
        if color not in remaining:
            continue
        remaining.remove(color)
        for nxt in successors[color]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    return not remaining


def _enumerate_scenes(grid: np.ndarray, limit: int = MAX_SCENE_SOLUTIONS) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(color) for color in np.unique(g) if int(color) != BLACK)
    assert MIN_FRAME_COUNT <= len(colors) <= MAX_FRAME_COUNT, "expected 3-5 visible frame colors"

    row_prefix, col_prefix = _zero_prefixes(g)
    candidates_by_color = {
        color: _enumerate_candidates(g, color, row_prefix, col_prefix)
        for color in colors
    }
    if any(not candidates for candidates in candidates_by_color.values()):
        return []

    search_colors = sorted(colors, key=lambda color: len(candidates_by_color[color]))
    visible_cells = np.argwhere(g != BLACK)
    solutions: list[dict[str, object]] = []
    chosen: dict[int, dict[str, object]] = {}

    def backtrack(idx: int) -> None:
        if len(solutions) >= limit:
            return
        if idx == len(search_colors):
            scene = _scene_from_choice(g, colors, chosen)
            if scene is not None:
                solutions.append(scene)
            return

        color = search_colors[idx]
        for candidate in candidates_by_color[color]:
            chosen[color] = candidate
            if _partial_choice_is_feasible(g, chosen, visible_cells):
                backtrack(idx + 1)
            del chosen[color]
            if len(solutions) >= limit:
                return

    backtrack(0)
    return solutions


def _infer_unique_scene(grid: np.ndarray) -> dict[str, object]:
    solutions = _enumerate_scenes(np.asarray(grid, dtype=int), limit=MAX_SCENE_SOLUTIONS)
    assert len(solutions) == 1, f"expected exactly one valid scene, found {len(solutions)}"
    return solutions[0]


def _render_input(shape: tuple[int, int], boxes: dict[int, tuple[int, int, int, int]], order: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for color in order:
        out[_border_mask(shape, boxes[color])] = int(color)
    return out


def _render_output(shape: tuple[int, int], boxes: dict[int, tuple[int, int, int, int]], order: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rank, color in enumerate(order, start=1):
        fill_color = RANK_FILL_COLORS[rank - 1]
        out[_interior_mask(shape, boxes[color])] = int(fill_color)
    for color in order:
        out[_border_mask(shape, boxes[color])] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the overlapping rectangle-frame draw order from visible occlusions, then fill each frame interior by depth rank.

    Input structure:
    - Black background contains 3 to 5 differently colored one-pixel rectangular frames.
    - Frames are axis-aligned, have non-empty interiors, and partially overlap in the same plane.
    - Later-drawn frame borders overwrite earlier frame borders at shared border cells, so some frame segments are hidden.

    Rule:
    - Reconstruct the full bounding box of each colored frame from its visible border pixels.
    - For every shared border cell, treat the visible color as the highest frame covering that cell and infer the unique bottom-to-top z-order that satisfies all such occlusion constraints.
    - Fill each frame interior, excluding its border, by z-order rank: rank 1 to 5 map to colors blue(1), red(2), green(3), yellow(4), gray(5).
    - If frame interiors overlap, higher-ranked fills overwrite lower-ranked fills; after filling, redraw the original frame borders in the inferred draw order so border colors stay unchanged.

    Key insight:
    - The frame colors themselves do not determine the interior colors; the visible border crossings encode a hidden draw order, and that recovered depth ranking is the only source of the fill-color mapping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    scene = _infer_unique_scene(g)
    boxes = {int(color): tuple(map(int, box)) for color, box in scene["boxes"].items()}
    order = [int(color) for color in scene["order"]]
    return _render_output(g.shape, boxes, order)


def _reflect_vertical(box: tuple[int, int, int, int], height: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, box)
    return (height - 1 - r1, c0, height - 1 - r0, c1)


def _reflect_horizontal(box: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, box)
    return (r0, width - 1 - c1, r1, width - 1 - c0)


def _sample_offsets(count: int) -> list[int]:
    offsets = [0]
    for _ in range(count - 1):
        step = random.randint(*GEN_STEP_RANGE)
        offsets.append(offsets[-1] + step)
    return offsets


def _visible_side_counts(
    grid: np.ndarray, color: int, box: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, box)
    top = int(np.count_nonzero(grid[r0, c0 : c1 + 1] == color))
    bottom = int(np.count_nonzero(grid[r1, c0 : c1 + 1] == color))
    left = int(np.count_nonzero(grid[r0 : r1 + 1, c0] == color))
    right = int(np.count_nonzero(grid[r0 : r1 + 1, c1] == color))
    return top, bottom, left, right


def _hidden_border_count(
    grid: np.ndarray, color: int, box: tuple[int, int, int, int]
) -> int:
    mask = _border_mask(grid.shape, box)
    return int(np.count_nonzero(mask & (grid != color)))


def _semantic_checks(
    grid: np.ndarray, boxes: dict[int, tuple[int, int, int, int]], order: list[int]
) -> bool:
    total_hidden = 0
    for color, box in boxes.items():
        if min(_visible_side_counts(grid, color, box)) < 1:
            return False
        total_hidden += _hidden_border_count(grid, color, box)
    if total_hidden < len(order):
        return False

    scenes = _enumerate_scenes(grid, limit=MAX_SCENE_SOLUTIONS)
    if len(scenes) != 1:
        return False
    scene = scenes[0]
    inferred_boxes = {int(color): tuple(map(int, box)) for color, box in scene["boxes"].items()}
    inferred_order = [int(color) for color in scene["order"]]
    return inferred_boxes == boxes and inferred_order == order


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(MIN_FRAME_COUNT, MAX_FRAME_COUNT)
    row_offsets = _sample_offsets(count)
    col_offsets = _sample_offsets(count)
    total_row = row_offsets[-1]
    total_col = col_offsets[-1]

    min_h = max(MIN_FRAME_SIZE, total_row + GEN_HEIGHT_RANGE_PAD[0])
    max_h = total_row + GEN_HEIGHT_RANGE_PAD[1]
    min_w = max(MIN_FRAME_SIZE, total_col + GEN_WIDTH_RANGE_PAD[0])
    max_w = total_col + GEN_WIDTH_RANGE_PAD[1]
    if min_h > max_h or min_w > max_w:
        return None

    frame_h = random.randint(min_h, max_h)
    frame_w = random.randint(min_w, max_w)
    margin_top = random.randint(*GEN_MARGIN_RANGE)
    margin_bottom = random.randint(*GEN_MARGIN_RANGE)
    margin_left = random.randint(*GEN_MARGIN_RANGE)
    margin_right = random.randint(*GEN_MARGIN_RANGE)

    height = frame_h + total_row + margin_top + margin_bottom
    width = frame_w + total_col + margin_left + margin_right
    if max(height, width) > 30:
        return None

    boxes_by_index: list[tuple[int, int, int, int]] = []
    for dr, dc in zip(row_offsets, col_offsets):
        r0 = margin_top + dr
        c0 = margin_left + dc
        boxes_by_index.append((r0, c0, r0 + frame_h - 1, c0 + frame_w - 1))

    if random.random() < 0.5:
        boxes_by_index = [_reflect_vertical(box, height) for box in boxes_by_index]
    if random.random() < 0.5:
        boxes_by_index = [_reflect_horizontal(box, width) for box in boxes_by_index]

    colors = random.sample(list(FRAME_PALETTE), count)
    boxes = {color: box for color, box in zip(colors, boxes_by_index)}
    order = random.sample(colors, count)

    grid = _render_input((height, width), boxes, order)
    if not _semantic_checks(grid, boxes, order):
        return None

    expected = _render_output(grid.shape, boxes, order)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-frame input")
