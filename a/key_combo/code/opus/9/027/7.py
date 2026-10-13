from __future__ import annotations

import itertools
import random
from typing import Iterable

import numpy as np


BBox = tuple[int, int, int, int]

BG = 0
RECT_COLORS = (1, 2, 3, 4)
RANK_COLORS = (8, 7, 6, 5)  # front -> back
MIN_H, MAX_H = 3, 6
MIN_W, MAX_W = 3, 5
MAX_SOLUTIONS = 2
GEN_RETRIES = 400


def _colors_present(grid: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) != BG)


def _bbox_of_mask(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def _inside(box: BBox, r: int, c: int) -> bool:
    return box[0] <= r <= box[2] and box[1] <= c <= box[3]


def _intersection(box_a: BBox, box_b: BBox) -> BBox | None:
    r0 = max(box_a[0], box_b[0])
    c0 = max(box_a[1], box_b[1])
    r1 = min(box_a[2], box_b[2])
    c1 = min(box_a[3], box_b[3])
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _fill_rect(grid: np.ndarray, box: BBox, color: int, *, overwrite: bool = True) -> None:
    r0, c0, r1, c1 = box
    if overwrite:
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
        return
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    region[region == BG] = int(color)


def _paint_boundary(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _render_input(shape: tuple[int, int], order: list[int], boxes: dict[int, BBox]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for color in reversed(order):
        _fill_rect(out, boxes[color], color, overwrite=True)
    return out


def _render_output(shape: tuple[int, int], order: list[int], boxes: dict[int, BBox]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)

    # Reveal hidden interiors: deeper interiors stay visible through shallower interiors.
    for color in reversed(order):
        r0, c0, r1, c1 = boxes[color]
        if r1 - r0 >= 2 and c1 - c0 >= 2:
            region = out[r0 + 1 : r1, c0 + 1 : c1]
            region[region == BG] = int(color)

    # Then paint ranked boundary rings from back to front so front rings win on conflicts.
    for rank_from_front in range(len(order) - 1, -1, -1):
        color = order[rank_from_front]
        boundary_color = RANK_COLORS[rank_from_front]
        _paint_boundary(out, boxes[color], boundary_color)
    return out


def _candidate_boxes(grid: np.ndarray, color: int, front_colors: Iterable[int]) -> list[BBox]:
    mask = grid == int(color)
    box = _bbox_of_mask(mask)
    assert box is not None
    vr0, vc0, vr1, vc1 = box
    h, w = grid.shape
    allowed = set(int(v) for v in front_colors)
    allowed.add(int(color))
    out: list[BBox] = []

    for rect_h in range(MIN_H, MAX_H + 1):
        r0_lo = max(0, vr1 - rect_h + 1)
        r0_hi = min(vr0, h - rect_h)
        if r0_lo > r0_hi:
            continue
        for rect_w in range(MIN_W, MAX_W + 1):
            c0_lo = max(0, vc1 - rect_w + 1)
            c0_hi = min(vc0, w - rect_w)
            if c0_lo > c0_hi:
                continue
            for r0 in range(r0_lo, r0_hi + 1):
                r1 = r0 + rect_h - 1
                for c0 in range(c0_lo, c0_hi + 1):
                    c1 = c0 + rect_w - 1
                    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
                    values = {int(v) for v in np.unique(patch)}
                    if BG in values:
                        continue
                    if not values <= allowed:
                        continue
                    out.append((int(r0), int(c0), int(r1), int(c1)))
    return out


def _has_pair_witness(
    grid: np.ndarray,
    order: list[int],
    boxes: dict[int, BBox],
    front_index: int,
    back_color: int,
) -> bool:
    front_color = order[front_index]
    overlap = _intersection(boxes[front_color], boxes[back_color])
    if overlap is None:
        return False
    r0, c0, r1, c1 = overlap
    blockers = [boxes[order[i]] for i in range(front_index)]
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if int(grid[r, c]) != int(front_color):
                continue
            if any(_inside(box, r, c) for box in blockers):
                continue
            return True
    return False


def _partial_visible_consistent(
    grid: np.ndarray,
    shape: tuple[int, int],
    order: list[int],
    boxes: dict[int, BBox],
    depth: int,
) -> bool:
    partial = np.full(shape, -1, dtype=int)
    for color in reversed(order[:depth]):
        _fill_rect(partial, boxes[color], color, overwrite=True)
    mask = partial >= 0
    return bool(np.array_equal(partial[mask], grid[mask]))


def _find_solutions(grid: np.ndarray, *, stop_after: int = MAX_SOLUTIONS) -> list[tuple[list[int], dict[int, BBox]]]:
    colors = _colors_present(grid)
    assert len(colors) == 4, "expected exactly four foreground colors"
    assert set(colors) == set(RECT_COLORS), "expected colors {1,2,3,4}"
    h, w = grid.shape
    solutions: list[tuple[list[int], dict[int, BBox]]] = []

    for order_tuple in itertools.permutations(colors):
        order = list(order_tuple)
        candidates: dict[int, list[BBox]] = {}
        valid_perm = True
        for i, color in enumerate(order):
            boxes = _candidate_boxes(grid, color, order[:i])
            if not boxes:
                valid_perm = False
                break
            candidates[color] = boxes
        if not valid_perm:
            continue

        assigned: dict[int, BBox] = {}

        def search(depth: int) -> None:
            if len(solutions) >= stop_after:
                return
            if depth == len(order):
                if not np.array_equal(_render_input((h, w), order, assigned), grid):
                    return
                for back_index in range(1, len(order)):
                    for front_index in range(back_index):
                        if not _has_pair_witness(grid, order, assigned, front_index, order[back_index]):
                            return
                solutions.append((order.copy(), dict(assigned)))
                return

            color = order[depth]
            for box in candidates[color]:
                assigned[color] = box

                ok = True
                for earlier in range(depth):
                    earlier_color = order[earlier]
                    if _intersection(assigned[earlier_color], box) is None:
                        ok = False
                        break
                    if not _has_pair_witness(grid, order, assigned, earlier, color):
                        ok = False
                        break
                if ok and _partial_visible_consistent(grid, (h, w), order, assigned, depth + 1):
                    search(depth + 1)
                del assigned[color]

                if len(solutions) >= stop_after:
                    return

        search(0)
        if len(solutions) >= stop_after:
            break

    return solutions


def _solve_unique(grid: np.ndarray) -> tuple[list[int], dict[int, BBox]]:
    solutions = _find_solutions(np.asarray(grid, dtype=int), stop_after=MAX_SOLUTIONS)
    assert len(solutions) == 1, "puzzle must have a unique rectangle/order reconstruction"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the hidden front-to-back order of overlapping solid rectangles and recolor their recovered borders by depth rank.

    Input structure:
    - Black background with exactly four non-black colors {1,2,3,4}.
    - Each color belongs to one solid axis-aligned rectangle, but only the visible top color is shown where rectangles overlap.
    - Every pair of rectangles overlaps in a positive-area region, and the visible overlap evidence uniquely determines one total depth order.

    Rule:
    - Reconstruct the unique set of four full rectangles and the unique front-to-back order whose opaque rectangle composite exactly matches the input.
    - Accept only reconstructions where each overlapping pair has a direct visible witness cell showing which of the two colors is in front.
    - Rank the recovered rectangles from front to back and recolor their boundary rings as 1->8, 2->7, 3->6, 4->5.
    - Reveal covered interiors by keeping the original color on rectangle interiors while letting only the ranked boundary rings participate in front-over-back conflict resolution.

    Key insight:
    - The visible colors inside overlap regions encode the hidden occlusion order; once the full rectangles and depth ranking are recovered, the same order resolves both boundary repainting and ring-overlap conflicts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    order, boxes = _solve_unique(g)
    return _render_output(g.shape, order, boxes)


def _visible_bbox_differs(grid: np.ndarray, color: int, full_box: BBox) -> bool:
    visible_box = _bbox_of_mask(grid == int(color))
    assert visible_box is not None
    return visible_box != full_box


def _hidden_side_count(grid: np.ndarray, color: int, full_box: BBox) -> int:
    visible_box = _bbox_of_mask(grid == int(color))
    assert visible_box is not None
    return sum(int(a != b) for a, b in zip(visible_box, full_box))


def _sample_depth_boxes(n: int) -> list[BBox] | None:
    ext0 = (1, 1, 1, 1)
    ext1 = (random.choice((2, 3)), random.choice((2, 3)), random.choice((0, 1)), random.choice((1, 2)))
    ext2 = (random.choice((0, 1)), random.choice((2, 3)), random.choice((2, 3)), random.choice((1, 2)))
    ext3 = (random.choice((1, 2)), random.choice((2, 3)), random.choice((2, 3)), random.choice((0, 1)))
    extents = [ext0, ext1, ext2, ext3]

    max_up = max(e[0] for e in extents)
    max_down = max(e[1] for e in extents)
    max_left = max(e[2] for e in extents)
    max_right = max(e[3] for e in extents)
    if max_up + max_down + 1 > n or max_left + max_right + 1 > n:
        return None

    core_r = random.randint(max_up, n - 1 - max_down)
    core_c = random.randint(max_left, n - 1 - max_right)
    boxes = [
        (core_r - up, core_c - left, core_r + down, core_c + right)
        for up, down, left, right in extents
    ]

    # Mild shape jitter for the front rectangle without breaking the witness layout.
    if random.random() < 0.5:
        front = list(boxes[0])
        if random.random() < 0.5 and _bbox_h(tuple(front)) < MAX_H and front[0] > 0:
            front[0] -= 1
        elif _bbox_w(tuple(front)) < MAX_W and front[3] < n - 1:
            front[3] += 1
        boxes[0] = tuple(front)  # type: ignore[assignment]

    if random.random() < 0.5:
        boxes = [(n - 1 - r1, c0, n - 1 - r0, c1) for r0, c0, r1, c1 in boxes]
    if random.random() < 0.5:
        boxes = [(r0, n - 1 - c1, r1, n - 1 - c0) for r0, c0, r1, c1 in boxes]

    if not all(MIN_H <= _bbox_h(box) <= MAX_H and MIN_W <= _bbox_w(box) <= MAX_W for box in boxes):
        return None
    return boxes


def _sample_scene() -> tuple[np.ndarray, list[int], dict[int, BBox]] | None:
    n = random.randint(13, 16)
    depth_boxes = _sample_depth_boxes(n)
    if depth_boxes is None:
        return None

    order = random.sample(list(RECT_COLORS), k=4)
    boxes = {order[i]: depth_boxes[i] for i in range(4)}
    grid = _render_input((n, n), order, boxes)

    if any(np.count_nonzero(grid == color) == 0 for color in RECT_COLORS):
        return None
    if not (18 <= int(np.count_nonzero(grid)) <= 45):
        return None

    hidden_boxes = sum(_visible_bbox_differs(grid, color, boxes[color]) for color in order[1:])
    hidden_sides = sum(_hidden_side_count(grid, color, boxes[color]) for color in order[1:])
    if hidden_boxes < 2 or hidden_sides < 4:
        return None

    output = _render_output(grid.shape, order, boxes)
    if set(int(v) for v in np.unique(output)) != {0, 1, 2, 3, 4, 5, 6, 7, 8}:
        return None

    recovered = _find_solutions(grid, stop_after=MAX_SOLUTIONS)
    if len(recovered) != 1:
        return None
    rec_order, rec_boxes = recovered[0]
    if rec_order != order or rec_boxes != boxes:
        return None

    if not np.array_equal(main(grid), output):
        return None
    return grid, order, boxes


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sampled = _sample_scene()
        if sampled is not None:
            grid, _order, _boxes = sampled
            return grid
    raise RuntimeError("failed to generate a valid unique occlusion-order puzzle")
