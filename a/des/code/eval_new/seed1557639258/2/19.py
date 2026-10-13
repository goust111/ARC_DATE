import random
from collections import deque

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
COPY_COUNT_RANGE = (3, 5)
PATTERN_HEIGHT_RANGE = (4, 8)
PATTERN_WIDTH_RANGE = (4, 8)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    banned = {int(x) for x in (exclude or set())}
    return [int(x) for x in np.unique(grid) if int(x) not in banned]


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _can_remove(mask: np.ndarray, r: int, c: int) -> bool:
    if not mask[r, c] or int(mask.sum()) <= 1:
        return False
    mask[r, c] = False
    ok = _touches_all_sides(mask) and _is_connected(mask)
    mask[r, c] = True
    return ok


def _add_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    er, ec = end
    mask[r, c] = True
    while (r, c) != (er, ec):
        moves = []
        if r != er:
            moves.append((r + (1 if er > r else -1), c))
        if c != ec:
            moves.append((r, c + (1 if ec > c else -1)))
        r, c = random.choice(moves)
        mask[r, c] = True


def _make_base_mask(height: int, width: int) -> np.ndarray | None:
    for _ in range(200):
        mask = np.zeros((height, width), dtype=bool)
        hub_r = random.randint(1, height - 2) if height > 2 else random.randint(0, height - 1)
        hub_c = random.randint(1, width - 2) if width > 2 else random.randint(0, width - 1)
        hub = (hub_r, hub_c)
        anchors = [
            (0, random.randrange(width)),
            (height - 1, random.randrange(width)),
            (random.randrange(height), 0),
            (random.randrange(height), width - 1),
        ]
        for anchor in anchors:
            _add_path(mask, anchor, hub)

        min_target = max(int(mask.sum()) + 2, (height * width) // 3)
        max_target = max(min_target, min(height * width - 1, (height * width * 3) // 4))
        target_area = random.randint(min_target, max_target)
        while int(mask.sum()) < target_area:
            frontier: list[tuple[int, int]] = []
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c), height, width):
                    if not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            frontier = list(dict.fromkeys(frontier))
            rr, cc = random.choice(frontier)
            mask[rr, cc] = True

        removable = [
            tuple(map(int, coord))
            for coord in np.argwhere(mask)
            if _can_remove(mask, int(coord[0]), int(coord[1]))
        ]
        if _touches_all_sides(mask) and _is_connected(mask) and len(removable) >= 4:
            return mask
    return None


def _build_copy_masks(base_mask: np.ndarray, copy_count: int) -> list[np.ndarray] | None:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(base_mask)]
    max_missing = min(6, max(2, len(coords) // 4))
    for _ in range(300):
        copies = [base_mask.copy() for _ in range(copy_count)]
        support = np.zeros_like(base_mask, dtype=int)
        support[base_mask] = copy_count
        targets = [random.randint(1, max_missing) for _ in range(copy_count)]
        removed = [0] * copy_count

        for _round in range(max_missing * 4 + 8):
            progress = False
            order = list(range(copy_count))
            random.shuffle(order)
            for idx in order:
                if removed[idx] >= targets[idx]:
                    continue
                candidates = []
                for r, c in coords:
                    if copies[idx][r, c] and support[r, c] > 1 and _can_remove(copies[idx], r, c):
                        candidates.append((r, c))
                if not candidates:
                    continue
                if random.random() < 0.7:
                    min_support = min(support[r, c] for r, c in candidates)
                    candidates = [(r, c) for r, c in candidates if support[r, c] == min_support]
                r, c = random.choice(candidates)
                copies[idx][r, c] = False
                support[r, c] -= 1
                removed[idx] += 1
                progress = True

            if all(removed[idx] >= targets[idx] for idx in range(copy_count)):
                break
            if not progress:
                break

        if any(count == 0 for count in removed):
            continue
        if any(removed[idx] < targets[idx] for idx in range(copy_count)):
            continue
        union_mask = np.zeros_like(base_mask)
        for copy_mask in copies:
            union_mask |= copy_mask
        if not np.array_equal(union_mask, base_mask):
            continue
        if len({copy_mask.tobytes() for copy_mask in copies}) < 2:
            continue
        return copies
    return None


def _split_nonnegative_sum(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    dividers = sorted(random.sample(range(total + parts - 1), parts - 1))
    pieces = []
    prev = -1
    for divider in dividers + [total + parts - 1]:
        pieces.append(divider - prev - 1)
        prev = divider
    return pieces


def _sample_layout(height: int, width: int, copy_count: int) -> tuple[int, int, list[tuple[int, int, int, int]]] | None:
    for _ in range(120):
        cols = random.randint(1, min(3, copy_count))
        rows = (copy_count + cols - 1) // cols
        gap = random.randint(1, 3)
        min_h = rows * height + (rows - 1) * gap
        min_w = cols * width + (cols - 1) * gap
        if min_h > 30 or min_w > 30:
            continue

        total_h = random.randint(min_h, 30)
        total_w = random.randint(min_w, 30)
        row_pad = _split_nonnegative_sum(total_h - min_h, rows + 1)
        col_pad = _split_nonnegative_sum(total_w - min_w, cols + 1)

        row_starts: list[int] = []
        cur = row_pad[0]
        for r in range(rows):
            row_starts.append(cur)
            cur += height
            if r < rows - 1:
                cur += gap + row_pad[r + 1]

        col_starts: list[int] = []
        cur = col_pad[0]
        for c in range(cols):
            col_starts.append(cur)
            cur += width
            if c < cols - 1:
                cur += gap + col_pad[c + 1]

        slots = [(r, c) for r in range(rows) for c in range(cols)]
        chosen_slots = random.sample(slots, copy_count)
        random.shuffle(chosen_slots)
        boxes = []
        for r, c in chosen_slots:
            r0 = row_starts[r]
            c0 = col_starts[c]
            boxes.append((r0, c0, r0 + height - 1, c0 + width - 1))
        return total_h, total_w, boxes
    return None


def _render_grid(
    grid_shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    masks: list[np.ndarray],
    bg: int,
    color: int,
) -> np.ndarray:
    grid = np.full(grid_shape, bg, dtype=int)
    for (r0, c0, r1, c1), mask in zip(boxes, masks):
        block = grid[r0 : r1 + 1, c0 : c1 + 1]
        block[mask] = color
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the complete pattern by unioning several incomplete translated copies.

    Input structure:
    - The grid has a dominant background color and exactly one foreground color.
    - The foreground pixels form at least three separated connected components.
    - Each component is a damaged copy of the same shape, with the same orientation and the same bounding-box size.

    Rule:
    - Find every foreground component and crop its bounding box.
    - Convert each cropped copy into a local binary mask and take the pixelwise union of all masks.
    - Clear each original bounding box and redraw the union mask there in the shared foreground color.

    Key insight:
    - A pixel missing from one copy still appears in another, so aligning the copies by their common bounding box lets set union reconstruct the full shape.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = _colors_present(g, exclude={bg})
    assert len(colors) == 1
    color = colors[0]

    components = _find_components(g == color)
    assert len(components) >= 3

    boxes: list[tuple[int, int, int, int]] = []
    masks: list[np.ndarray] = []
    shapes = set()
    for coords in components:
        r0, c0, r1, c1 = _bbox(coords)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        boxes.append((r0, c0, r1, c1))
        masks.append(local)
        shapes.add(local.shape)

    assert len(shapes) == 1
    full_mask = np.zeros_like(masks[0])
    for local in masks:
        full_mask |= local

    out = g.copy()
    for r0, c0, r1, c1 in boxes:
        block = out[r0 : r1 + 1, c0 : c1 + 1]
        block[:] = bg
        block[full_mask] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    copy_count = random.randint(*COPY_COUNT_RANGE)
    height = random.randint(*PATTERN_HEIGHT_RANGE)
    width = random.randint(*PATTERN_WIDTH_RANGE)

    base_mask = _make_base_mask(height, width)
    if base_mask is None:
        return None
    copy_masks = _build_copy_masks(base_mask, copy_count)
    if copy_masks is None:
        return None
    layout = _sample_layout(height, width, copy_count)
    if layout is None:
        return None
    grid_h, grid_w, boxes = layout

    bg, color = random.sample(range(10), 2)
    inp = _render_grid((grid_h, grid_w), boxes, copy_masks, bg, color)
    expected = _render_grid((grid_h, grid_w), boxes, [base_mask] * copy_count, bg, color)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 19")
