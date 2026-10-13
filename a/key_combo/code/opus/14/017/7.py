from __future__ import annotations

import random

import numpy as np


FRAME_COLORS = (1, 2, 3, 6)
TARGET_COLOR = 4
BACKGROUND = 0

GRID_H_RANGE = (14, 17)
GRID_W_RANGE = (14, 17)
FRAME_COUNT_CHOICES = (3, 4)
FRAME_SIZE_RANGE = (4, 8)
PLACEMENT_TRIES = 240
GENERATION_TRIES = 500
TARGET_TRIES = 200


def _sign(x: int) -> int:
    return int((x > 0) - (x < 0))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_pixels(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _paint_pixels(grid: np.ndarray, pixels: set[tuple[int, int]], color: int) -> None:
    if not pixels:
        return
    rows, cols = zip(*sorted(pixels))
    grid[rows, cols] = int(color)


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    above: dict[int, set[int]] = {color: set() for color in colors}
    indegree: dict[int, int] = {color: 0 for color in colors}
    for lower, higher in edges:
        if lower == higher:
            raise AssertionError("self-ordering constraint is invalid")
        if higher not in above[lower]:
            above[lower].add(higher)
            indegree[higher] += 1

    order: list[int] = []
    placed: set[int] = set()
    while len(order) < len(colors):
        available = [color for color in colors if color not in placed and indegree[color] == 0]
        if len(available) != 1:
            raise AssertionError("layer constraints must define a unique order")
        color = available[0]
        order.append(color)
        placed.add(color)
        for higher in above[color]:
            indegree[higher] -= 1
    return order


def _reconstruct_frames(
    grid: np.ndarray,
) -> tuple[list[int], dict[int, tuple[int, int, int, int]], dict[int, set[tuple[int, int]]], tuple[int, int]]:
    target_coords = np.argwhere(grid == TARGET_COLOR)
    assert target_coords.shape == (1, 2), "expected exactly one yellow target pixel"
    target = (int(target_coords[0, 0]), int(target_coords[0, 1]))

    colors = sorted(int(v) for v in np.unique(grid) if v not in (BACKGROUND, TARGET_COLOR))
    assert 3 <= len(colors) <= 4, "expected 3-4 frame colors"
    assert set(colors) <= set(FRAME_COLORS), "unexpected frame color present"

    bboxes: dict[int, tuple[int, int, int, int]] = {}
    pixels_by_color: dict[int, set[tuple[int, int]]] = {}
    for color in colors:
        box = _bbox(grid == color)
        assert box is not None, "each frame color must be visible"
        r0, c0, r1, c1 = box
        assert r1 - r0 + 1 >= 4 and c1 - c0 + 1 >= 4, "each frame bbox must be at least 4x4"
        full_pixels = _frame_pixels(box)
        visible_pixels = set(map(tuple, np.argwhere(grid == color)))
        assert visible_pixels <= full_pixels, "visible pixels must lie on the reconstructed frame border"
        bboxes[color] = box
        pixels_by_color[color] = full_pixels

    return colors, bboxes, pixels_by_color, target


def _infer_order(
    grid: np.ndarray,
    colors: list[int],
    pixels_by_color: dict[int, set[tuple[int, int]]],
) -> list[int]:
    h, w = grid.shape
    edges: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            covering = [color for color in colors if (r, c) in pixels_by_color[color]]
            if len(covering) < 2:
                continue
            visible = int(grid[r, c])
            assert visible in covering, "overlap cell must show one of the covering frames"
            for color in covering:
                if color != visible:
                    edges.add((color, visible))

    assert edges, "overlap constraints are required"
    order = _unique_topological_order(colors, edges)
    return order


def _direction_for_bbox(bbox: tuple[int, int, int, int], target: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    tr, tc = target
    return _sign(2 * tr - (r0 + r1)), _sign(2 * tc - (c0 + c1))


def _shift_pixels(
    pixels: set[tuple[int, int]],
    delta: tuple[int, int],
) -> set[tuple[int, int]]:
    dr, dc = delta
    return {(r + dr, c + dc) for r, c in pixels}


def _simulate_objects(
    shape: tuple[int, int],
    order: list[int],
    pixels_by_color: dict[int, set[tuple[int, int]]],
    bboxes: dict[int, tuple[int, int, int, int]],
    target: tuple[int, int],
    *,
    collision: bool,
    priority_top_down: bool,
) -> tuple[dict[int, set[tuple[int, int]]], dict[int, bool], dict[int, bool], dict[int, tuple[int, int]]]:
    h, w = shape
    final_pixels: dict[int, set[tuple[int, int]]] = {}
    moved: dict[int, bool] = {}
    blocked: dict[int, bool] = {}
    directions = {color: _direction_for_bbox(bboxes[color], target) for color in order}

    occupied: set[tuple[int, int]] = set()
    process_order = list(reversed(order)) if priority_top_down else list(order)
    for color in process_order:
        old_pixels = pixels_by_color[color]
        delta = directions[color]
        shifted = _shift_pixels(old_pixels, delta)
        in_bounds = all(0 <= r < h and 0 <= c < w for r, c in shifted)
        collides = in_bounds and bool(shifted & occupied)
        can_move = in_bounds and (not collision or not collides)

        final_pixels[color] = shifted if can_move else set(old_pixels)
        moved[color] = can_move
        blocked[color] = in_bounds and collision and collides
        occupied |= final_pixels[color]

    return final_pixels, moved, blocked, directions


def _render_output(
    shape: tuple[int, int],
    order: list[int],
    final_pixels: dict[int, set[tuple[int, int]]],
    target: tuple[int, int],
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for color in order:
        _paint_pixels(out, final_pixels[color], color)
    out[target] = TARGET_COLOR
    return out


def _solve_from_scene(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[int], dict[int, tuple[int, int, int, int]], dict[int, set[tuple[int, int]]], tuple[int, int]]:
    colors, bboxes, pixels_by_color, target = _reconstruct_frames(grid)
    order = _infer_order(grid, colors, pixels_by_color)
    final_pixels, _moved, _blocked, _dirs = _simulate_objects(
        grid.shape,
        order,
        pixels_by_color,
        bboxes,
        target,
        collision=True,
        priority_top_down=True,
    )
    out = _render_output(grid.shape, order, final_pixels, target)
    return out, order, bboxes, pixels_by_color, target


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct occluded rectangular frames, infer their stacking order, then move each frame one step toward the yellow target.

    Input structure:
    - The grid is black except for 3-4 colored 1-pixel-thick hollow rectangular frames and exactly one yellow target pixel.
    - Each frame color has some visible border pixels, but overlap regions show only the higher layer's color.
    - The yellow target lies outside every frame border.

    Rule:
    - For each frame color, reconstruct the full hollow rectangle as the border of that color's visible-pixel bounding box.
    - Wherever reconstructed frames overlap, the visible input color is the higher frame there; use all such constraints to infer the unique bottom-to-top layer order.
    - For each frame, compute the sign of the row and column difference from the frame center to the yellow target, giving a step vector in {-1,0,1}^2.
    - Process frames from top to bottom: a frame moves one step by that vector unless its shifted border would leave the grid or overlap a higher frame's already-final border, in which case it stays put.
    - Draw the final scene from bottom layer to top layer and keep the yellow target pixel fixed.

    Key insight:
    - The visible pixels reveal each frame's full bbox, while the overlap colors reveal the hidden layer order; that same order then controls which frames are allowed to move when shifted borders would collide.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    out, _order, _bboxes, _pixels_by_color, _target = _solve_from_scene(g)
    return out


def _render_input(
    shape: tuple[int, int],
    order: list[int],
    bboxes: dict[int, tuple[int, int, int, int]],
    target: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for color in order:
        _paint_pixels(grid, _frame_pixels(bboxes[color]), color)
    grid[target] = TARGET_COLOR
    return grid


def _random_frame_size(h: int, w: int) -> tuple[int, int]:
    max_h = min(FRAME_SIZE_RANGE[1], h - 2)
    max_w = min(FRAME_SIZE_RANGE[1], w - 2)
    return random.randint(FRAME_SIZE_RANGE[0], max_h), random.randint(FRAME_SIZE_RANGE[0], max_w)


def _place_first_bbox(h: int, w: int) -> tuple[int, int, int, int]:
    fh, fw = _random_frame_size(h, w)
    r0 = random.randint(1, h - fh - 1)
    c0 = random.randint(1, w - fw - 1)
    return (r0, c0, r0 + fh - 1, c0 + fw - 1)


def _place_next_bbox(
    h: int,
    w: int,
    prev_bbox: tuple[int, int, int, int],
    prior_sets: list[set[tuple[int, int]]],
) -> tuple[tuple[int, int, int, int], set[tuple[int, int]]] | None:
    pr0, pc0, pr1, pc1 = prev_bbox
    for _ in range(PLACEMENT_TRIES):
        fh, fw = _random_frame_size(h, w)
        rmin = max(1, pr0 - fh + 1)
        rmax = min(h - fh - 1, pr1)
        cmin = max(1, pc0 - fw + 1)
        cmax = min(w - fw - 1, pc1)
        if rmin > rmax or cmin > cmax:
            continue

        r0 = random.randint(rmin, rmax)
        c0 = random.randint(cmin, cmax)
        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        cells = _frame_pixels(bbox)

        if len(cells & prior_sets[-1]) not in (2, 4):
            continue
        if any(len(cells & earlier) not in (0, 2, 4) for earlier in prior_sets[:-1]):
            continue
        return bbox, cells
    return None


def _place_frame_stack(
    h: int,
    w: int,
    order: list[int],
) -> dict[int, tuple[int, int, int, int]] | None:
    first_bbox = _place_first_bbox(h, w)
    bboxes: list[tuple[int, int, int, int]] = [first_bbox]
    cell_sets: list[set[tuple[int, int]]] = [_frame_pixels(first_bbox)]

    for _ in range(1, len(order)):
        placed = _place_next_bbox(h, w, bboxes[-1], cell_sets)
        if placed is None:
            return None
        bbox, cells = placed
        bboxes.append(bbox)
        cell_sets.append(cells)

    return {color: bbox for color, bbox in zip(order, bboxes)}


def _frame_union_mask(shape: tuple[int, int], bboxes: dict[int, tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for bbox in bboxes.values():
        for r, c in _frame_pixels(bbox):
            mask[r, c] = True
    return mask


def _candidate_targets(union_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = union_mask.shape
    isolated: list[tuple[int, int]] = []
    other: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if union_mask[r, c]:
                continue
            clear = True
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    if union_mask[rr, cc]:
                        clear = False
                        break
                if not clear:
                    break
            if clear:
                isolated.append((r, c))
            else:
                other.append((r, c))

    def border_score(cell: tuple[int, int]) -> int:
        r, c = cell
        return int(r in (0, h - 1) or c in (0, w - 1))

    random.shuffle(isolated)
    random.shuffle(other)
    isolated.sort(key=border_score, reverse=True)
    other.sort(key=border_score, reverse=True)
    return isolated + other


def _semantic_target_ok(
    input_grid: np.ndarray,
    order: list[int],
    bboxes: dict[int, tuple[int, int, int, int]],
    target: tuple[int, int],
) -> bool:
    full_pixels = {color: _frame_pixels(bboxes[color]) for color in order}
    final_pixels, moved, blocked, directions = _simulate_objects(
        input_grid.shape,
        order,
        full_pixels,
        bboxes,
        target,
        collision=True,
        priority_top_down=True,
    )
    true_output = _render_output(input_grid.shape, order, final_pixels, target)

    if sum(moved.values()) < 1 or sum(blocked.values()) < 1:
        return False
    if len(set(directions.values())) < 2:
        return False

    no_collision_pixels, _m1, _b1, _d1 = _simulate_objects(
        input_grid.shape,
        order,
        full_pixels,
        bboxes,
        target,
        collision=False,
        priority_top_down=True,
    )
    if np.array_equal(true_output, _render_output(input_grid.shape, order, no_collision_pixels, target)):
        return False

    no_move_output = _render_output(input_grid.shape, order, full_pixels, target)
    if np.array_equal(true_output, no_move_output):
        return False

    reverse_priority_pixels, _m2, _b2, _d2 = _simulate_objects(
        input_grid.shape,
        list(reversed(order)),
        full_pixels,
        bboxes,
        target,
        collision=True,
        priority_top_down=True,
    )
    reverse_priority_output = _render_output(input_grid.shape, list(reversed(order)), reverse_priority_pixels, target)
    if np.array_equal(true_output, reverse_priority_output):
        return False

    reverse_draw_output = _render_output(input_grid.shape, list(reversed(order)), final_pixels, target)
    if np.array_equal(true_output, reverse_draw_output):
        return False

    visible_pixels = {color: set(map(tuple, np.argwhere(input_grid == color))) for color in order}
    visible_move_pixels, _m3, _b3, _d3 = _simulate_objects(
        input_grid.shape,
        order,
        visible_pixels,
        bboxes,
        target,
        collision=True,
        priority_top_down=True,
    )
    visible_move_output = _render_output(input_grid.shape, order, visible_move_pixels, target)
    if np.array_equal(true_output, visible_move_output):
        return False

    return True


def _base_scene_ok(
    input_grid: np.ndarray,
    order: list[int],
    bboxes: dict[int, tuple[int, int, int, int]],
) -> bool:
    visible_bboxes = {color: _bbox(input_grid == color) for color in order}
    if any(visible_bboxes[color] != bboxes[color] for color in order):
        return False

    try:
        solved_output, solved_order, solved_bboxes, _pixels_by_color, _target = _solve_from_scene(input_grid)
    except Exception:
        return False

    del solved_output
    if solved_order != order:
        return False
    if solved_bboxes != bboxes:
        return False

    hidden_exists = any(np.count_nonzero(input_grid == color) < len(_frame_pixels(bboxes[color])) for color in order)
    return hidden_exists


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        frame_count = random.choice(FRAME_COUNT_CHOICES)
        order = random.sample(FRAME_COLORS, frame_count)

        bboxes = _place_frame_stack(h, w, order)
        if bboxes is None:
            continue

        union_mask = _frame_union_mask((h, w), bboxes)
        target_candidates = _candidate_targets(union_mask)
        if not target_candidates:
            continue

        for target in target_candidates[:TARGET_TRIES]:
            base_input = _render_input((h, w), order, bboxes, target)
            if not _base_scene_ok(base_input, order, bboxes):
                continue
            if not _semantic_target_ok(base_input, order, bboxes, target):
                continue

            actual = main(base_input)
            expected, _order, _bboxes, _pixels_by_color, _target = _solve_from_scene(base_input)
            if not np.array_equal(actual, expected):
                continue
            return base_input

    raise RuntimeError("failed to generate a valid puzzle instance")
