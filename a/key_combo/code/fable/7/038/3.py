from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


TEAL = 8
BBox = tuple[int, int, int, int]


def _background(grid: np.ndarray) -> int:
    """Return the unique most frequent color, following arc_utils.get_bg."""
    counts = Counter(map(int, grid.ravel()))
    top = max(counts.values())
    winners = [color for color, count in counts.items() if count == top]
    assert len(winners) == 1, "expected a unique majority background"
    return winners[0]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, col)."""
    h, w = mask.shape
    unseen = set(map(tuple, np.argwhere(mask)))
    result: list[np.ndarray] = []
    while unseen:
        seed = unseen.pop()
        queue = deque([seed])
        cells = [seed]
        while queue:
            r, c = queue.popleft()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in unseen:
                    unseen.remove((nr, nc))
                    queue.append((nr, nc))
                    cells.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _intersects(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _shift(box: BBox, direction: tuple[int, int], steps: int = 1) -> BBox:
    dr, dc = direction
    return (
        box[0] + dr * steps,
        box[1] + dc * steps,
        box[2] + dr * steps,
        box[3] + dc * steps,
    )


def _inside(box: BBox, shape: tuple[int, int]) -> bool:
    h, w = shape
    return 0 <= box[0] <= box[2] < h and 0 <= box[1] <= box[3] < w


def _extract(grid: np.ndarray) -> tuple[int, list[dict], int, tuple[int, int]]:
    """Parse solid rectangles and the unique straight teal tail."""
    bg = _background(grid)
    assert bg != TEAL, "teal must be visible against the background"

    tail = np.argwhere(grid == TEAL)
    assert len(tail) >= 1, "expected a teal tail"
    tail_parts = _components(grid == TEAL)
    assert len(tail_parts) == 1 and len(tail_parts[0]) == len(tail), "tail must be connected"
    rows = sorted(set(map(int, tail[:, 0])))
    cols = sorted(set(map(int, tail[:, 1])))
    if len(rows) == 1:
        assert cols == list(range(cols[0], cols[-1] + 1)), "tail must be a solid line"
    elif len(cols) == 1:
        assert rows == list(range(rows[0], rows[-1] + 1)), "tail must be a solid line"
    else:
        raise AssertionError("tail must be one cell wide")

    objects: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg, TEAL}):
        for coords in _components(grid == color):
            box = _bbox(coords)
            r0, c0, r1, c1 = box
            assert len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1), "objects must be filled rectangles"
            assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)
            objects.append({"bbox": box, "color": color})
    assert len(objects) >= 2, "expected several rectangles"

    tail_set = {tuple(map(int, p)) for p in tail}
    endpoints = []
    for cell in tail_set:
        degree = sum((cell[0] + dr, cell[1] + dc) in tail_set for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        if degree <= 1:
            endpoints.append(cell)
    if len(tail_set) == 1:
        endpoints = [next(iter(tail_set))]
    assert len(endpoints) in (1, 2)

    attachments: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for er, ec in endpoints:
        for direction in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = er + direction[0], ec + direction[1]
            for i, obj in enumerate(objects):
                r0, c0, r1, c1 = obj["bbox"]
                if r0 <= nr <= r1 and c0 <= nc <= c1:
                    attachments.append((i, direction, (er, ec)))
    assert len(attachments) == 1, "tail must attach to exactly one rectangle edge"
    active, direction, near_end = attachments[0]

    rectangle_cells: set[tuple[int, int]] = set()
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        rectangle_cells.update((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
    contact_pairs = {
        ((r, c), (r + dr, c + dc))
        for r, c in tail_set
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        if (r + dr, c + dc) in rectangle_cells
    }
    attached_cell = (near_end[0] + direction[0], near_end[1] + direction[1])
    assert contact_pairs == {(near_end, attached_cell)}, "only one tail endpoint may touch a rectangle"

    # Every tail cell must lie directly behind the attached endpoint.
    expected_tail = {
        (near_end[0] - direction[0] * k, near_end[1] - direction[1] * k)
        for k in range(len(tail_set))
    }
    assert tail_set == expected_tail, "tail must point exactly opposite the motion direction"
    return bg, objects, active, direction


def _simulate(
    boxes: list[BBox],
    active: int,
    direction: tuple[int, int],
    budget: int,
    shape: tuple[int, int],
) -> tuple[list[BBox], list[int], list[int]]:
    """Spend one budget unit per successful translation; collisions cost zero."""
    boxes = list(boxes)
    moved = [0] * len(boxes)
    handoffs: list[int] = []
    seen_handoffs: set[tuple[int, int]] = set()
    while budget:
        candidate = _shift(boxes[active], direction)
        assert _inside(candidate, shape), "momentum budget would carry a rectangle beyond the grid"
        hits = [i for i, box in enumerate(boxes) if i != active and _intersects(candidate, box)]
        assert len(hits) <= 1, "the next collision must identify one receiver"
        if hits:
            receiver = hits[0]
            event = (active, receiver)
            assert event not in seen_handoffs, "collision chain must progress forward"
            seen_handoffs.add(event)
            active = receiver
            handoffs.append(receiver)
            continue
        boxes[active] = candidate
        moved[active] += 1
        budget -= 1
    return boxes, moved, handoffs


def main(grid: np.ndarray) -> np.ndarray:
    """Move a rectangle relay chain using the teal tail length as one conserved step budget.

    Input structure:
    - A unique majority background contains several separated monochromatic filled rectangles.
    - Exactly one rectangle has a one-cell-wide straight teal tail attached outside one edge;
      the tail is perpendicular to that edge and its length is the shared momentum quota.

    Rule:
    - Remove the teal tail and move its rectangle one cell at a time opposite the tail direction.
    - A successful translation spends one quota step. If the next translation would overlap one
      rectangle, leave the current rectangle in place and pass all unspent steps to that rectangle.
    - Continue in the same direction until exactly the tail-length quota has been spent; rectangles
      never reached by this collision relay remain unchanged.

    Key insight:
    - The tail encodes both direction and a conserved budget: collisions transfer the remaining
      momentum rather than granting each rectangle its own trip to a neighbor or boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"
    bg, objects, active, direction = _extract(g)
    budget = int(np.count_nonzero(g == TEAL))
    final_boxes, _moved, _handoffs = _simulate(
        [obj["bbox"] for obj in objects], active, direction, budget, g.shape
    )

    out = np.full_like(g, bg)
    for obj, (r0, c0, r1, c1) in zip(objects, final_boxes):
        target = out[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(target == bg), "final rectangles must not overlap"
        target[:] = int(obj["color"])
    return out


def _expanded(box: BBox, pad: int, shape: tuple[int, int]) -> BBox:
    h, w = shape
    return (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(h - 1, box[2] + pad),
        min(w - 1, box[3] + pad),
    )


def _paint_rect(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _gen_rightward() -> np.ndarray | None:
    """Build a right-moving canonical case; callers rotate it for all directions."""
    h = random.randint(15, 24)
    chain_count = random.choices((2, 3, 4), weights=(3, 5, 2))[0]
    heights = [random.randint(2, 4) for _ in range(chain_count)]
    widths = [random.randint(2, 4) for _ in range(chain_count)]
    gaps = [random.randint(1, 2) for _ in range(chain_count - 1)]
    final_steps = random.randint(1, 3)
    budget = sum(gaps) + final_steps

    min_w = 2 + 2 * budget + sum(widths)
    if min_w > 30:
        return None
    w = random.randint(max(19, min_w), 30)
    shape = (h, w)

    # Consecutive receivers overlap in the row projection, but are visibly staggered.
    r0 = random.randint(2, h - heights[0] - 3)
    row_starts = [r0]
    for i in range(1, chain_count):
        prev0 = row_starts[-1]
        prev1 = prev0 + heights[i - 1] - 1
        choices = [
            r
            for r in range(2, h - heights[i] - 2)
            if max(r, prev0) <= min(r + heights[i] - 1, prev1)
        ]
        if not choices:
            return None
        staggered = [r for r in choices if r != prev0]
        row_starts.append(random.choice(staggered or choices))

    c = budget + 1
    chain_boxes: list[BBox] = []
    for i in range(chain_count):
        box = (row_starts[i], c, row_starts[i] + heights[i] - 1, c + widths[i] - 1)
        chain_boxes.append(box)
        if i < chain_count - 1:
            c = box[3] + 1 + gaps[i]
    assert chain_boxes[-1][3] + final_steps <= w - 2

    tail_row = random.randint(chain_boxes[0][0], chain_boxes[0][2])
    tail_cells = [(tail_row, chain_boxes[0][1] - k) for k in range(1, budget + 1)]

    distractor_count = random.randint(1, min(3, 8 - chain_count))
    boxes = list(chain_boxes)
    for _ in range(distractor_count):
        placed = False
        for _try in range(160):
            rh, rw = random.randint(2, 4), random.randint(2, 4)
            rr = random.randint(1, h - rh - 1)
            cc = random.randint(1, w - rw - 1)
            candidate = (rr, cc, rr + rh - 1, cc + rw - 1)
            forbidden = [_expanded(box, 1, shape) for box in boxes]
            if any(_intersects(candidate, box) for box in forbidden):
                continue
            padded_candidate = _expanded(candidate, 1, shape)
            if any(
                padded_candidate[0] <= r <= padded_candidate[2]
                and padded_candidate[1] <= c0 <= padded_candidate[3]
                for r, c0 in tail_cells
            ):
                continue
            boxes.append(candidate)
            placed = True
            break
        if not placed:
            return None

    colors = list(set(range(10)) - {TEAL})
    bg = random.choice(colors)
    colors.remove(bg)
    if len(colors) < len(boxes):
        return None
    object_colors = random.sample(colors, len(boxes))
    grid = np.full(shape, bg, dtype=int)
    for box, color in zip(boxes, object_colors):
        _paint_rect(grid, box, color)
    for r, c0 in tail_cells:
        assert grid[r, c0] == bg
        grid[r, c0] = TEAL

    # Independent semantic oracle: every intended receiver must take the baton,
    # distractors must stay fixed, and exactly D successful translations occur.
    try:
        final_boxes, moved, handoffs = _simulate(boxes, 0, (0, 1), budget, shape)
    except AssertionError:
        return None
    if handoffs != list(range(1, chain_count)):
        return None
    if sum(moved) != budget or any(moved[i] == 0 for i in range(chain_count)):
        return None
    if any(moved[i] != 0 for i in range(chain_count, len(boxes))):
        return None
    terminal = chain_count - 1
    if final_boxes[terminal][3] >= w - 1:
        return None
    after_budget = _shift(final_boxes[terminal], (0, 1))
    if any(_intersects(after_budget, box) for i, box in enumerate(final_boxes) if i != terminal):
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    expected = np.full_like(grid, bg)
    for box, color in zip(final_boxes, object_colors):
        _paint_rect(expected, box, color)
    return grid if np.array_equal(solved, expected) else None


def generate_input() -> np.ndarray:
    """Generate diverse non-degenerate relay cases in all four directions."""
    for _ in range(500):
        grid = _gen_rightward()
        if grid is None:
            continue
        transform = random.randrange(4)
        if transform == 0:       # right
            result = grid
        elif transform == 1:     # down (clockwise)
            result = np.rot90(grid, 3)
        elif transform == 2:     # left
            result = np.fliplr(grid)
        else:                    # up (counter-clockwise)
            result = np.rot90(grid, 1)
        result = np.asarray(result, dtype=int).copy()
        try:
            out = main(result)
        except AssertionError:
            continue
        if result.ndim == 2 and max(result.shape) <= 30 and out.shape == result.shape:
            return result
    raise RuntimeError("failed to generate a valid momentum-relay input")
