from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
FRAME_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_HEIGHT_RANGE = (18, 31)
GRID_WIDTH_RANGE = (18, 31)
CONTAINER_COUNT_CHOICES = (2, 3, 4, 5)
CONTAINER_MIN_SIZE = 5
CONTAINER_MAX_SIZE_LOW = 8
CONTAINER_MAX_SIZE_HIGH = 10
CONTAINER_GAP = 1
KEY_BOX_GAP = 1
KEY_KEY_GAP = 1
MAX_GENERATION_ATTEMPTS = 400
MAX_PLACEMENT_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _frame_bbox(g: np.ndarray, coords: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    border_area = 2 * (h + w) - 4
    if len(coords) != border_area:
        return None

    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(sub[border] == color):
        return None
    if np.any(sub[~border] == color):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _find_containers(g: np.ndarray) -> list[dict]:
    containers: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BLACK, GREY)):
        for coords in _components(g == color):
            bbox = _frame_bbox(g, coords, color)
            if bbox is None:
                continue
            containers.append({"color": int(color), "bbox": bbox})

    assert containers, "expected at least one rectangular container"
    seen_boxes = set()
    for container in containers:
        bbox = container["bbox"]
        assert bbox not in seen_boxes, "duplicate container bbox"
        seen_boxes.add(bbox)
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """Open containers whose border color appears among outside keys.

    Input structure
    - Black background.
    - Several one-pixel-thick rectangular frames of different colors.
    - Each frame contains a Grey pattern.
    - Outside the frames, single-pixel colored keys are scattered on the background.

    Rule
    - Collect all key colors outside every container.
    - If a container's border color is present, recolor its Grey interior pattern
      to the border color.
    - Otherwise erase that Grey pattern to Black.
    - Remove all outside keys.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    containers = _find_containers(g)
    box_mask = np.zeros(g.shape, dtype=bool)
    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        box_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    outside = ~box_mask
    active_colors = {
        int(v) for v in np.unique(g[outside]) if int(v) not in (BLACK, GREY)
    }

    out = g.copy()
    key_mask = outside & (g != BLACK) & (g != GREY)
    out[key_mask] = BLACK

    for container in containers:
        color = int(container["color"])
        r0, c0, r1, c1 = container["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        grey_mask = interior == GREY
        interior[grey_mask] = color if color in active_colors else BLACK

    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _random_walk_pattern(h: int, w: int) -> np.ndarray:
    area = h * w
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    min_cells = 1 if area <= 3 else 2
    max_cells = max(min_cells, min(area - 1, max(2, (area * 3) // 5)))
    target = random.randint(min_cells, max_cells)

    while int(mask.sum()) < target:
        coords = np.argwhere(mask)
        r, c = map(int, coords[random.randrange(len(coords))])
        options = []
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                options.append((nr, nc))
        if not options:
            zeros = np.argwhere(~mask)
            if len(zeros) == 0:
                break
            nr, nc = map(int, zeros[random.randrange(len(zeros))])
        else:
            nr, nc = random.choice(options)
        mask[nr, nc] = True
    return mask


def _rectangles_pattern(h: int, w: int) -> np.ndarray:
    area = h * w
    mask = np.zeros((h, w), dtype=bool)
    rect_count = random.randint(1, 3)
    for _ in range(rect_count):
        rh = random.randint(1, max(1, h - (0 if h <= 2 else 1)))
        rw = random.randint(1, max(1, w - (0 if w <= 2 else 1)))
        r0 = random.randint(0, h - rh)
        c0 = random.randint(0, w - rw)
        mask[r0 : r0 + rh, c0 : c0 + rw] = True

    if not mask.any():
        mask[random.randrange(h), random.randrange(w)] = True
    if int(mask.sum()) == area:
        rr, cc = random.choice(list(map(tuple, np.argwhere(mask))))
        mask[rr, cc] = False
    return mask


def _line_pattern(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    segments = random.randint(1, 3)
    for _ in range(segments):
        if random.random() < 0.5:
            r = random.randrange(h)
            c0 = random.randrange(w)
            c1 = random.randrange(w)
            if c0 > c1:
                c0, c1 = c1, c0
            mask[r, c0 : c1 + 1] = True
        else:
            c = random.randrange(w)
            r0 = random.randrange(h)
            r1 = random.randrange(h)
            if r0 > r1:
                r0, r1 = r1, r0
            mask[r0 : r1 + 1, c] = True

    if not mask.any():
        mask[random.randrange(h), random.randrange(w)] = True
    if int(mask.sum()) == h * w:
        rr, cc = random.choice(list(map(tuple, np.argwhere(mask))))
        mask[rr, cc] = False
    return mask


def _random_pattern(h: int, w: int) -> np.ndarray:
    chooser = random.choice((_random_walk_pattern, _rectangles_pattern, _line_pattern))
    mask = chooser(h, w)
    if not mask.any():
        mask[random.randrange(h), random.randrange(w)] = True
    if int(mask.sum()) == h * w:
        rr, cc = random.choice(list(map(tuple, np.argwhere(mask))))
        mask[rr, cc] = False
    return mask


def _sample_container_boxes(h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    max_dim = CONTAINER_MAX_SIZE_HIGH if count <= 3 else CONTAINER_MAX_SIZE_LOW
    dims = []
    for _ in range(count):
        box_h = random.randint(CONTAINER_MIN_SIZE, min(max_dim, h - 2))
        box_w = random.randint(CONTAINER_MIN_SIZE, min(max_dim, w - 2))
        dims.append((box_h, box_w))
    dims.sort(key=lambda hw: hw[0] * hw[1], reverse=True)

    blocked = np.zeros((h, w), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    for box_h, box_w in dims:
        placed = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            if h - box_h - 1 <= 0 or w - box_w - 1 <= 0:
                return None
            r0 = random.randint(1, h - box_h - 1)
            c0 = random.randint(1, w - box_w - 1)
            r1 = r0 + box_h - 1
            c1 = c0 + box_w - 1
            rr0 = max(0, r0 - CONTAINER_GAP)
            cc0 = max(0, c0 - CONTAINER_GAP)
            rr1 = min(h, r1 + CONTAINER_GAP + 1)
            cc1 = min(w, c1 + CONTAINER_GAP + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            blocked[rr0:rr1, cc0:cc1] = True
            boxes.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return None
    return boxes


def _key_block_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in boxes:
        rr0 = max(0, r0 - KEY_BOX_GAP)
        cc0 = max(0, c0 - KEY_BOX_GAP)
        rr1 = min(h, r1 + KEY_BOX_GAP + 1)
        cc1 = min(w, c1 + KEY_BOX_GAP + 1)
        blocked[rr0:rr1, cc0:cc1] = True
    return blocked


def _choose_activation(colors: list[int]) -> set[int]:
    while True:
        active = {color for color in colors if random.random() < 0.5}
        if 0 < len(active) < len(colors):
            return active


def _place_keys(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    key_colors: list[int],
) -> list[tuple[int, int, int]] | None:
    h, w = grid.shape
    blocked = _key_block_mask((h, w), boxes)
    placed: list[tuple[int, int, int]] = []

    for color in key_colors:
        candidates = np.argwhere((grid == BLACK) & (~blocked))
        if len(candidates) == 0:
            return None
        r, c = map(int, candidates[random.randrange(len(candidates))])
        grid[r, c] = color
        placed.append((r, c, color))

        rr0 = max(0, r - KEY_KEY_GAP)
        cc0 = max(0, c - KEY_KEY_GAP)
        rr1 = min(h, r + KEY_KEY_GAP + 1)
        cc1 = min(w, c + KEY_KEY_GAP + 1)
        blocked[rr0:rr1, cc0:cc1] = True

    return placed


def _expected_output(
    inp: np.ndarray,
    containers: list[dict],
    key_positions: list[tuple[int, int, int]],
    active_colors: set[int],
) -> np.ndarray:
    out = inp.copy()
    for r, c, _color in key_positions:
        out[r, c] = BLACK
    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        color = int(container["color"])
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        grey_mask = interior == GREY
        interior[grey_mask] = color if color in active_colors else BLACK
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    count_choices = list(CONTAINER_COUNT_CHOICES)
    if h * w < 450:
        count_choices.remove(5)
    count = random.choice(count_choices)

    boxes = _sample_container_boxes(h, w, count)
    if boxes is None:
        return None

    frame_colors = random.sample(FRAME_COLORS, count)
    active_colors = _choose_activation(frame_colors)
    unused_colors = [color for color in FRAME_COLORS if color not in frame_colors]

    inp = np.full((h, w), BLACK, dtype=int)
    containers: list[dict] = []
    for bbox, color in zip(boxes, frame_colors, strict=False):
        _draw_frame(inp, bbox, color)
        r0, c0, r1, c1 = bbox
        pattern = _random_pattern(r1 - r0 - 1, c1 - c0 - 1)
        inp[r0 + 1 : r1, c0 + 1 : c1][pattern] = GREY
        containers.append({"bbox": bbox, "color": color})

    key_colors: list[int] = []
    for color in frame_colors:
        if color in active_colors:
            key_colors.extend([color] * random.randint(1, 2))

    extra_count = random.randint(0, min(2, len(unused_colors)))
    if extra_count:
        key_colors.extend(random.sample(unused_colors, extra_count))
    random.shuffle(key_colors)

    key_positions = _place_keys(inp, boxes, key_colors)
    if key_positions is None:
        return None

    out = _expected_output(inp, containers, key_positions, active_colors)
    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _build_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
