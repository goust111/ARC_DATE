from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
BLOCK_COLORS = (1, 4, 5, 6, 7, 8, 9)

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
DIRS = (UP, RIGHT, DOWN, LEFT)
ROTATION_STEPS = {UP: 0, RIGHT: 1, DOWN: 2, LEFT: 3}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _frame_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREEN):
        cells = set(comp)
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if side_h != side_w or side_h < 7 or side_h % 2 == 0:
            continue
        border = set()
        for c in range(c0, c1 + 1):
            border.add((r0, c))
            border.add((r1, c))
        for r in range(r0, r1 + 1):
            border.add((r, c0))
            border.add((r, c1))
        if cells == border:
            frames.append((r0, c0, r1, c1))
    frames.sort()
    return frames


def _pointer_direction(inner: np.ndarray) -> tuple[int, int]:
    n = inner.shape[0]
    center = n // 2
    assert inner[center, center] == RED, "clock center must be red"
    lengths = []
    for dr, dc in DIRS:
        rr, cc = center + dr, center + dc
        length = 0
        while 0 <= rr < n and 0 <= cc < n and inner[rr, cc] == RED:
            length += 1
            rr += dr
            cc += dc
        lengths.append(length)
    best = int(np.argmax(lengths))
    assert lengths[best] > 0, "clock hand must extend away from the center"
    return DIRS[best]


def _pointer_cells(n: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    center = n // 2
    if direction == UP:
        return [(r, center) for r in range(center, -1, -1)]
    if direction == RIGHT:
        return [(center, c) for c in range(center, n)]
    if direction == DOWN:
        return [(r, center) for r in range(center, n)]
    if direction == LEFT:
        return [(center, c) for c in range(center, -1, -1)]
    raise ValueError(direction)


def _first_seen_colors(patch: np.ndarray) -> list[int]:
    colors: list[int] = []
    seen: set[int] = set()
    for value in patch.flat:
        color = int(value)
        if color in (BG, RED, GREEN) or color in seen:
            continue
        seen.add(color)
        colors.append(color)
    return colors


def _color_bbox(patch: np.ndarray, color: int) -> tuple[int, int, int, int]:
    pts = np.argwhere(patch == color)
    assert pts.size > 0
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    return r0, c0, r1, c1


def _render_patch(
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
    colors: list[int],
    order: tuple[int, ...] | list[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for idx in order:
        r0, c0, r1, c1 = rects[int(idx)]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(colors[int(idx)])
    return out


def _matching_orders(
    observed: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    colors: list[int],
) -> list[tuple[int, ...]]:
    matches: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(len(colors))):
        if np.array_equal(_render_patch(observed.shape, rects, colors, perm), observed):
            matches.append(perm)
    return matches


def _rotated_order(order: tuple[int, ...] | list[int], steps: int) -> tuple[int, ...]:
    order = tuple(int(x) for x in order)
    if not order:
        return order
    shift = steps % len(order)
    return order[shift:] + order[:shift]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the overlapped rectangles inside each fenced clock region by rotating their layer stack.

    Input structure:
    - The background is black.
    - One or more separate one-pixel-thick green square fences appear in the grid.
    - Each fence contains a red center pixel and a straight red clock hand pointing up, right, down, or left.
    - Aside from the red hand, the fence interior contains 2 to 4 colored axis-aligned rectangles whose overlaps show only the topmost visible color.

    Rule:
    - Process each green fence independently.
    - Recover each rectangle as the bounding box of the visible cells of one interior color, then find the unique bottom-to-top rectangle order whose redraw exactly matches the observed overlap pattern.
    - Convert the hand direction into a cyclic layer rotation: up=0, right=1, down=2, left=3, where one step moves the former bottom layer to the top.
    - Redraw the same rectangles in the rotated order and erase all red hand pixels to black; keep the green fence and the outer background unchanged.

    Key insight:
    - The red hand does not move shapes; it encodes how far to rotate the hidden layer order that must first be inferred from the visible overlap evidence.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, c0, r1, c1 in _frame_bboxes(g):
        inner = g[r0 + 1 : r1, c0 + 1 : c1].copy()
        direction = _pointer_direction(inner)
        inner[inner == RED] = BG

        colors = _first_seen_colors(inner)
        assert 2 <= len(colors) <= 4, "each fence must contain 2 to 4 visible block colors"
        rects = [_color_bbox(inner, color) for color in colors]

        matches = _matching_orders(inner, rects, colors)
        assert len(matches) == 1, "overlap pattern must imply a unique layer order"

        rotated = _rotated_order(matches[0], ROTATION_STEPS[direction])
        out[r0 + 1 : r1, c0 + 1 : c1] = _render_patch(inner.shape, rects, colors, rotated)

    return out


def _rect_intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _rectangles_connected(rects: list[tuple[int, int, int, int]]) -> bool:
    if not rects:
        return False
    seen = {0}
    stack = [0]
    while stack:
        cur = stack.pop()
        for nxt in range(len(rects)):
            if nxt in seen:
                continue
            if _rect_intersection_area(rects[cur], rects[nxt]) > 0:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(rects)


def _candidate_rectangles(n: int, blocked: set[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for r0 in range(n):
        for c0 in range(n):
            for r1 in range(r0 + 1, n):
                for c1 in range(c0 + 1, n):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area < 4 or area > n * n - len(blocked) - 1:
                        continue
                    bad = False
                    for r in range(r0, r1 + 1):
                        for c in range(c0, c1 + 1):
                            if (r, c) in blocked:
                                bad = True
                                break
                        if bad:
                            break
                    if not bad:
                        rects.append((r0, c0, r1, c1))
    return rects


def _sample_fence_case(rng: random.Random) -> dict[str, object] | None:
    block_count = rng.choice((2, 3, 4))
    side_choices = {
        2: (7, 9, 11),
        3: (7, 9, 11),
        4: (9, 11),
    }
    side = int(rng.choice(side_choices[block_count]))
    inner_size = side - 2
    direction = rng.choice(DIRS)
    blocked = set(_pointer_cells(inner_size, direction))
    candidates = _candidate_rectangles(inner_size, blocked)
    if len(candidates) < block_count:
        return None

    for _ in range(2000):
        rects = rng.sample(candidates, block_count)
        if not _rectangles_connected(rects):
            continue
        colors = list(rng.sample(BLOCK_COLORS, block_count))
        order = list(range(block_count))
        rng.shuffle(order)

        observed = _render_patch((inner_size, inner_size), rects, colors, order)
        if any(np.count_nonzero(observed == color) == 0 for color in colors):
            continue
        if any(_color_bbox(observed, color) != rect for color, rect in zip(colors, rects)):
            continue

        recovered_colors = _first_seen_colors(observed)
        recovered_rects = [_color_bbox(observed, color) for color in recovered_colors]
        matches = _matching_orders(observed, recovered_rects, recovered_colors)
        if len(matches) != 1:
            continue

        rotated = _rotated_order(matches[0], ROTATION_STEPS[direction])
        output_inner = _render_patch((inner_size, inner_size), recovered_rects, recovered_colors, rotated)
        input_inner = observed.copy()
        for r, c in blocked:
            input_inner[r, c] = RED

        return {
            "side": side,
            "direction": direction,
            "input_inner": input_inner,
            "output_inner": output_inner,
        }
    return None


def _draw_frame(grid: np.ndarray, top: int, left: int, side: int, inner: np.ndarray) -> None:
    bottom = top + side - 1
    right = left + side - 1
    grid[top : bottom + 1, left] = GREEN
    grid[top : bottom + 1, right] = GREEN
    grid[top, left : right + 1] = GREEN
    grid[bottom, left : right + 1] = GREEN
    grid[top + 1 : bottom, left + 1 : right] = inner


def _place_cases(
    rng: random.Random,
    shape: tuple[int, int],
    cases: list[dict[str, object]],
) -> list[tuple[int, int]] | None:
    h, w = shape
    occupied = np.zeros(shape, dtype=bool)
    placements: list[tuple[int, int]] = []
    for case in cases:
        side = int(case["side"])
        positions: list[tuple[int, int]] = []
        for top in range(1, h - side):
            for left in range(1, w - side):
                r0 = max(0, top - 1)
                c0 = max(0, left - 1)
                r1 = min(h, top + side + 1)
                c1 = min(w, left + side + 1)
                if not occupied[r0:r1, c0:c1].any():
                    positions.append((top, left))
        if not positions:
            return None
        top, left = rng.choice(positions)
        placements.append((top, left))
        occupied[top : top + side, left : left + side] = True
    return placements


def _build_example(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    fence_count = rng.choice((1, 2, 3))
    cases: list[dict[str, object]] = []
    for _ in range(fence_count):
        case = _sample_fence_case(rng)
        if case is None:
            return None
        cases.append(case)

    min_side = max(int(case["side"]) for case in cases) + 2
    height = rng.randint(max(12, min_side), 30)
    width = rng.randint(max(12, min_side), 30)
    placements = _place_cases(rng, (height, width), cases)
    if placements is None:
        return None

    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    for (top, left), case in zip(placements, cases):
        side = int(case["side"])
        _draw_frame(inp, top, left, side, np.array(case["input_inner"], dtype=int))
        _draw_frame(out, top, left, side, np.array(case["output_inner"], dtype=int))

    return inp, out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    candidate = _build_example(rng)
    if candidate is None:
        return None
    inp, out = candidate
    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
