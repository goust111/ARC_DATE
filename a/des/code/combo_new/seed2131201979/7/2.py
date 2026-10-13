from __future__ import annotations

import random
from collections import deque

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _intersect_bbox(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return (r0, c0, r1, c1)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)


def _intersection_is_interior(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and inner[2] < outer[2] and inner[3] < outer[3]


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0, r1, c1 = _bbox(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            out.append({"color": color, "bbox": (r0, c0, r1, c1), "mask": mask})
    return out


def _separator_column(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for c in range(1, w - 1):
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) != bg:
            if np.any(grid[:, :c] != bg) and np.any(grid[:, c + 1 :] != bg):
                candidates.append((c, int(col[0])))
    assert len(candidates) == 1, f"expected one separator column, got {candidates}"
    return candidates[0]


def _left_rectangles(left: np.ndarray, bg: int) -> list[dict]:
    rects = []
    for color in sorted(int(x) for x in np.unique(left) if int(x) != bg):
        coords = np.argwhere(left == color)
        assert coords.size > 0, "each left-side color must remain visible"
        rects.append({"color": color, "bbox": _bbox(coords)})
    assert len(rects) >= 2, "expected multiple left rectangles"
    return rects


def _unique_topological(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    outgoing = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for src, dst in edges:
        if dst not in outgoing[src]:
            outgoing[src].add(dst)
            indegree[dst] += 1

    order: list[int] = []
    remaining = set(colors)
    while remaining:
        ready = [color for color in colors if color in remaining and indegree[color] == 0]
        assert len(ready) == 1, f"stack order must be unique, got {ready}"
        color = ready[0]
        remaining.remove(color)
        order.append(color)
        for nxt in outgoing[color]:
            indegree[nxt] -= 1
    return order


def _stack_order(left: np.ndarray, rects: list[dict], bg: int) -> list[int]:
    edges: set[tuple[int, int]] = set()
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a = rects[i]
            b = rects[j]
            inter = _intersect_bbox(a["bbox"], b["bbox"])
            if inter is None:
                continue
            r0, c0, r1, c1 = inter
            visible = set(int(x) for x in np.unique(left[r0 : r1 + 1, c0 : c1 + 1])) - {bg}
            assert len(visible) == 1, "pairwise overlap must reveal a single visible rectangle"
            top = next(iter(visible))
            assert top in (a["color"], b["color"]), "unexpected third rectangle in pairwise overlap"
            if top == a["color"]:
                edges.add((b["color"], a["color"]))
            else:
                edges.add((a["color"], b["color"]))
    return _unique_topological([rect["color"] for rect in rects], edges)


def _right_blocks(right: np.ndarray, bg: int) -> list[dict]:
    blocks = _components(right, bg)
    assert blocks, "expected right-side blocks"
    colors = [block["color"] for block in blocks]
    assert len(colors) == len(set(colors)), "right-side block colors must be unique"
    return blocks


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the left rectangle stack order and use it to vertically reorder the right-side blocks.

    Input structure:
    - The grid is split into a left occlusion scene and a right block scene by one full-height separator column.
    - The left scene contains one solid axis-aligned rectangle per color; rectangles overlap and partially hide one another.
    - The right scene contains the same number of isolated colored blocks, arranged from top to bottom on a shared background.

    Rule:
    - Infer each left rectangle as the full bounding box of all visible pixels of its color.
    - Whenever two inferred rectangles overlap, the single visible color inside that overlap is the upper rectangle; topologically sort these lower-to-upper relations to recover the stack order from bottom to top.
    - Sort the left rectangles by their top row and sort the right blocks by their top row; pair them in that order.
    - Output only the right scene, moving each right block into the vertical slot for its paired rectangle's stack rank while keeping the block shape, color, and horizontal placement unchanged.

    Key insight:
    - The left overlap pattern is a code for layer order, and that hidden order must be transferred through an independent top-to-bottom correspondence to reorder unrelated right-hand blocks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    sep_c, _sep_color = _separator_column(g, bg)
    left = g[:, :sep_c]
    right = g[:, sep_c + 1 :]

    rects = _left_rectangles(left, bg)
    stack = _stack_order(left, rects, bg)
    blocks = _right_blocks(right, bg)
    assert len(rects) == len(blocks), "left and right sides must contain the same number of objects"

    rects_by_top = sorted(rects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    blocks_by_top = sorted(blocks, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    assert len({obj["bbox"][0] for obj in rects_by_top}) == len(rects_by_top), "left top-row order must be unique"
    assert len({obj["bbox"][0] for obj in blocks_by_top}) == len(blocks_by_top), "right top-row order must be unique"

    block_for_left = {rects_by_top[i]["color"]: blocks_by_top[i] for i in range(len(rects_by_top))}
    slot_rows = [block["bbox"][0] for block in blocks_by_top]

    out = np.full(right.shape, bg, dtype=int)
    for slot_index, left_color in enumerate(stack):
        block = block_for_left[left_color]
        r0, c0, _r1, _c1 = block["bbox"]
        mask = block["mask"]
        h, w = mask.shape
        nr0 = slot_rows[slot_index]
        view = out[nr0 : nr0 + h, c0 : c0 + w]
        assert view.shape == mask.shape, "reordered block must stay inside the right region"
        assert not np.any(view[mask] != bg), "reordered blocks must stay disjoint"
        view[mask] = block["color"]
    return out


# --------------- generator ---------------

MAX_GENERATION_ATTEMPTS = 400
RECT_COUNT_CHOICES = (3, 4)
LEFT_W_RANGE = (12, 16)
RIGHT_W_CHOICES = (7, 8)
RIGHT_SLOT_STEP = 6
RIGHT_TOP_MARGIN_CHOICES = (1, 2)
RECT_SIZE_CHOICES = (5, 6)
RECT_STEP = 3


def _paint_rectangles(
    h: int, w: int, bg: int, rects: list[dict]
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = rect["color"]
    return grid


def _visible_bboxes_match(grid: np.ndarray, rects: list[dict]) -> bool:
    for rect in rects:
        coords = np.argwhere(grid == rect["color"])
        if coords.size == 0 or _bbox(coords) != rect["bbox"]:
            return False
    return True


def _place_left_rectangles(h: int, w: int, colors: list[int]) -> list[dict] | None:
    dr = random.choice((-RECT_STEP, RECT_STEP))
    dc = random.choice((-RECT_STEP, RECT_STEP))
    sizes = [(random.choice(RECT_SIZE_CHOICES), random.choice(RECT_SIZE_CHOICES)) for _ in colors]

    row_lower = max(-i * dr for i in range(len(colors)))
    row_upper = min(h - rh - i * dr for i, (rh, _rw) in enumerate(sizes))
    col_lower = max(-i * dc for i in range(len(colors)))
    col_upper = min(w - rw - i * dc for i, (_rh, rw) in enumerate(sizes))
    if row_lower > row_upper or col_lower > col_upper:
        return None

    base_r = random.randint(row_lower, row_upper)
    base_c = random.randint(col_lower, col_upper)
    rects = []
    for i, (color, (rh, rw)) in enumerate(zip(colors, sizes)):
        r0 = base_r + i * dr
        c0 = base_c + i * dc
        rects.append({"color": color, "bbox": (r0, c0, r0 + rh - 1, c0 + rw - 1)})

    if len({rect["bbox"][0] for rect in rects}) != len(rects):
        return None
    for i in range(len(rects) - 1):
        inter = _intersect_bbox(rects[i]["bbox"], rects[i + 1]["bbox"])
        if inter is None or _bbox_area(inter) < 4:
            return None
    for i in range(len(rects)):
        for j in range(i + 2, len(rects)):
            if _intersect_bbox(rects[i]["bbox"], rects[j]["bbox"]) is not None:
                return None

    canvas = _paint_rectangles(h, w, -1, rects)
    if not _visible_bboxes_match(canvas, rects):
        return None
    try:
        inferred = _stack_order(canvas, _left_rectangles(canvas, -1), -1)
    except AssertionError:
        return None
    return rects if inferred == colors else None


def _shape_mask() -> np.ndarray:
    if random.random() < 0.5:
        side = random.choice((2, 3))
        return np.ones((side, side), dtype=bool)
    arm = random.choice((1, 2))
    size = 2 * arm + 1
    mask = np.zeros((size, size), dtype=bool)
    mask[arm, :] = True
    mask[:, arm] = True
    return mask


def _build_right_region(
    h: int, w: int, bg: int, colors: list[int]
) -> tuple[np.ndarray, list[dict]] | None:
    top_margin = random.choice(RIGHT_TOP_MARGIN_CHOICES)
    slot_rows = [top_margin + i * RIGHT_SLOT_STEP for i in range(len(colors))]
    max_size = 5
    if slot_rows[-1] + max_size > h:
        return None

    c0 = random.randint(1, w - max_size)
    grid = np.full((h, w), bg, dtype=int)
    blocks: list[dict] = []
    for color, slot_r0 in zip(colors, slot_rows):
        mask = _shape_mask()
        bh, bw = mask.shape
        if c0 + bw > w:
            return None
        view = grid[slot_r0 : slot_r0 + bh, c0 : c0 + bw]
        if view.shape != mask.shape or np.any(view[mask] != bg):
            return None
        view[mask] = color
        blocks.append({"color": color, "bbox": (slot_r0, c0, slot_r0 + bh - 1, c0 + bw - 1), "mask": mask})
    return grid, blocks


def _render_reordered_right(
    shape: tuple[int, int], bg: int, stack_order: list[int], left_rects: list[dict], blocks: list[dict]
) -> np.ndarray:
    rects_by_top = sorted(left_rects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    blocks_by_top = sorted(blocks, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    mapping = {rects_by_top[i]["color"]: blocks_by_top[i] for i in range(len(rects_by_top))}
    slot_rows = [block["bbox"][0] for block in blocks_by_top]

    out = np.full(shape, bg, dtype=int)
    for slot_index, left_color in enumerate(stack_order):
        block = mapping[left_color]
        r0, c0, _r1, _c1 = block["bbox"]
        mask = block["mask"]
        bh, bw = mask.shape
        nr0 = slot_rows[slot_index]
        out[nr0 : nr0 + bh, c0 : c0 + bw][mask] = block["color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(RECT_COUNT_CHOICES)
    right_w = random.choice(RIGHT_W_CHOICES)
    min_h = random.choice(RIGHT_TOP_MARGIN_CHOICES) + (n - 1) * RIGHT_SLOT_STEP + 5
    h = random.randint(min_h, 28)
    left_w = random.randint(*LEFT_W_RANGE)

    palette = random.sample(list(range(10)), 2 + 2 * n)
    bg = palette[0]
    sep = palette[1]
    left_colors = palette[2 : 2 + n]
    right_colors = palette[2 + n :]

    stack_order = left_colors[:]
    left_rects = _place_left_rectangles(h, left_w, stack_order)
    if left_rects is None:
        return None

    right_result = _build_right_region(h, right_w, bg, right_colors)
    if right_result is None:
        return None
    right_grid, right_blocks = right_result

    left_grid = _paint_rectangles(h, left_w, bg, left_rects)
    inp = np.full((h, left_w + 1 + right_w), bg, dtype=int)
    inp[:, :left_w] = left_grid
    inp[:, left_w] = sep
    inp[:, left_w + 1 :] = right_grid

    if _most_common_color(inp) != bg:
        return None

    expected = _render_reordered_right(right_grid.shape, bg, stack_order, left_rects, right_blocks)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
