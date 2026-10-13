from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
DOT = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 600


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _interior_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}


def _overlap_matrix(boxes: list[tuple[int, int, int, int]]) -> list[list[bool]]:
    borders = [_border_cells(box) for box in boxes]
    n = len(boxes)
    overlap = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            hit = bool(borders[i] & borders[j])
            overlap[i][j] = hit
            overlap[j][i] = hit
    return overlap


def _counts_for_order(
    order: tuple[int, ...], overlap: list[list[bool]]
) -> tuple[int, ...]:
    pos = {idx: depth for depth, idx in enumerate(order)}
    n = len(order)
    counts = [0] * n
    for i in range(n):
        counts[i] = sum(
            1 for j in range(n) if overlap[i][j] and pos[j] > pos[i]
        )
    return tuple(counts)


def _solve_unique_order(
    boxes: list[tuple[int, int, int, int]], counts: list[int]
) -> tuple[int, ...]:
    overlap = _overlap_matrix(boxes)
    n = len(boxes)
    matches = []
    for order in itertools.permutations(range(n)):
        if _counts_for_order(order, overlap) == tuple(counts):
            matches.append(order)
            if len(matches) > 1:
                break
    assert len(matches) == 1, "dot declarations must determine one unique layer order"
    return matches[0]


def _recover_boxes_and_counts(
    grid: np.ndarray,
) -> tuple[list[int], list[tuple[int, int, int, int]], list[int]]:
    colors = [int(color) for color in np.unique(grid) if color not in (BG, DOT)]
    assert colors, "expected at least one colored hollow frame"

    boxes = []
    borders = []
    interiors = []
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.size, "each frame color must appear"
        box = _bbox_from_coords(coords)
        r0, c0, r1, c1 = box
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frames must have nonempty interiors"
        border = _border_cells(box)
        visible = {tuple(map(int, rc)) for rc in coords}
        assert visible <= border, "visible frame pixels must lie on one hollow-rectangle border"
        boxes.append(box)
        borders.append(border)
        interiors.append(_interior_cells(box))

    counts = [0] * len(colors)
    grey_cells = [tuple(map(int, rc)) for rc in np.argwhere(grid == DOT)]
    for cell in grey_cells:
        owners = [i for i, interior in enumerate(interiors) if cell in interior]
        assert len(owners) == 1, "each grey dot must belong to exactly one frame interior"
        counts[owners[0]] += 1

    return colors, boxes, counts


def _render_frames(
    shape: tuple[int, int],
    colors: list[int],
    boxes: list[tuple[int, int, int, int]],
    order: tuple[int, ...],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for idx in order:
        for r, c in _border_cells(boxes[idx]):
            out[r, c] = colors[idx]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Solve the self-referential cover counts and redraw the hollow rectangles in the unique true depth order.

    Input structure:
    - A black background contains several differently colored hollow rectangles whose border overlaps are shown in an arbitrary visible layer order.
    - Grey dots lie only in cells that are inside exactly one rectangle interior, never on a border.
    - Because overlaps hide lower border cells, each colored frame may appear with gaps in the input.

    Rule:
    - For each rectangle, its grey-dot count declares how many overlapping rectangles must lie above it in the correct order.
    - Recover the unique global bottom-to-top order satisfying all declarations, using overlap only between frames whose full borders share cells.
    - Erase all grey dots and redraw every rectangle completely from the bottom layer upward.

    Key insight:
    - The dot count is not a direct depth rank; it is a local covered-by-overlapping-neighbors count, so the answer requires solving the whole overlap graph at once.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    colors, boxes, counts = _recover_boxes_and_counts(g)
    order = _solve_unique_order(boxes, counts)
    return _render_frames(g.shape, colors, boxes, order)


def _connected(overlap: list[list[bool]]) -> bool:
    n = len(overlap)
    seen = {0}
    stack = [0]
    while stack:
        cur = stack.pop()
        for nxt, hit in enumerate(overlap[cur]):
            if hit and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == n


def _interesting_order(order: tuple[int, ...], counts: tuple[int, ...], target: str) -> bool:
    seq = [counts[idx] for idx in order]
    has_rise = any(seq[i] < seq[i + 1] for i in range(len(seq) - 1))
    has_fall = any(seq[i] > seq[i + 1] for i in range(len(seq) - 1))
    if not (has_rise and has_fall):
        return False

    has_repeat = len(set(seq)) < len(seq)
    zero_middle = any(seq[i] == 0 for i in range(1, len(seq) - 1))
    if target == "zero_middle":
        return zero_middle
    if target == "repeat":
        return has_repeat
    return True


def _sample_boxes() -> list[tuple[int, int, int, int]]:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    count = random.choice((4, 5))
    boxes = []
    for _ in range(count):
        box_h = random.randint(5, 10)
        box_w = random.randint(5, 10)
        r0 = random.randint(1, h - box_h - 1)
        c0 = random.randint(1, w - box_w - 1)
        boxes.append((r0, c0, r0 + box_h - 1, c0 + box_w - 1))
    return boxes


def _choose_solution(
    boxes: list[tuple[int, int, int, int]], target: str
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    overlap = _overlap_matrix(boxes)
    if not _connected(overlap):
        return None

    counts_to_orders: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for order in itertools.permutations(range(len(boxes))):
        counts = _counts_for_order(order, overlap)
        counts_to_orders.setdefault(counts, []).append(order)

    unique = []
    for counts, orders in counts_to_orders.items():
        if len(orders) != 1:
            continue
        order = orders[0]
        if _interesting_order(order, counts, target):
            unique.append((order, counts))
    if not unique and target != "mixed":
        for counts, orders in counts_to_orders.items():
            if len(orders) == 1:
                order = orders[0]
                if _interesting_order(order, counts, "mixed"):
                    unique.append((order, counts))
    if not unique:
        return None
    return random.choice(unique)


def _dot_candidates(
    boxes: list[tuple[int, int, int, int]]
) -> list[list[tuple[int, int]]] | None:
    borders = [_border_cells(box) for box in boxes]
    interiors = [_interior_cells(box) for box in boxes]
    all_borders = set().union(*borders)

    candidates = []
    for i, interior in enumerate(interiors):
        unique = set(interior)
        for j, other in enumerate(interiors):
            if i != j:
                unique -= other
        unique -= all_borders
        candidates.append(sorted(unique))
    return candidates


def _visible_bbox_ok(
    grid: np.ndarray,
    colors: list[int],
    boxes: list[tuple[int, int, int, int]],
) -> bool:
    for color, box in zip(colors, boxes):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            return False
        if _bbox_from_coords(coords) != box:
            return False
    return True


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    colors: list[int],
    boxes: list[tuple[int, int, int, int]],
    order: tuple[int, ...],
    counts: tuple[int, ...],
) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    recovered_colors, recovered_boxes, recovered_counts = _recover_boxes_and_counts(inp)
    if recovered_colors != colors:
        return False
    if recovered_boxes != boxes:
        return False
    if tuple(recovered_counts) != counts:
        return False

    solved = _solve_unique_order(recovered_boxes, recovered_counts)
    if solved != order:
        return False

    overlap = _overlap_matrix(boxes)
    if _counts_for_order(order, overlap) != counts:
        return False

    rendered = _render_frames(inp.shape, colors, boxes, order)
    if not np.array_equal(rendered, expected):
        return False

    got = main(inp)
    return np.array_equal(got, expected)


def _attempt_generate() -> np.ndarray | None:
    boxes = _sample_boxes()
    if len(set(boxes)) != len(boxes):
        return None
    target = random.choice(("zero_middle", "repeat", "mixed"))
    chosen = _choose_solution(boxes, target)
    if chosen is None:
        return None
    order, counts = chosen

    candidates = _dot_candidates(boxes)
    if candidates is None:
        return None
    for idx, need in enumerate(counts):
        if len(candidates[idx]) < need:
            return None

    colors = sorted(random.sample(FRAME_COLORS, len(boxes)))
    visible_order = list(range(len(boxes)))
    random.shuffle(visible_order)
    if tuple(visible_order) == order:
        visible_order.reverse()

    shape = (
        max(box[2] for box in boxes) + 2,
        max(box[3] for box in boxes) + 2,
    )
    inp = np.full(shape, BG, dtype=int)
    for idx in visible_order:
        for r, c in _border_cells(boxes[idx]):
            inp[r, c] = colors[idx]

    for idx, need in enumerate(counts):
        if need == 0:
            continue
        for r, c in random.sample(candidates[idx], need):
            inp[r, c] = DOT

    if not _visible_bbox_ok(inp, colors, boxes):
        return None

    expected = _render_frames(shape, colors, boxes, order)
    if not _validate_instance(inp, expected, colors, boxes, order, counts):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-referential overlap-order input")
