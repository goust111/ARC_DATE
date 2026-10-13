from __future__ import annotations

import random

import numpy as np


FOUR_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ARC_COLORS = tuple(range(10))
GEN_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 120

SPRITE_TEMPLATES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
)


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(grid.reshape(-1), minlength=10)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for dr, dc in FOUR_NEIGHBORS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boundary_color(rect: np.ndarray, top: int, bottom: int, left: int, right: int) -> int:
    parts = [
        rect[top, left:right],
        rect[bottom - 1, left:right],
        rect[top + 1 : bottom - 1, left],
        rect[top + 1 : bottom - 1, right - 1],
    ]
    boundary = np.concatenate([part.reshape(-1) for part in parts if part.size])
    if boundary.size == 0:
        raise AssertionError("expected a nonempty boundary")
    colors = np.unique(boundary)
    if colors.size != 1:
        raise AssertionError("boundary must be monochromatic")
    return int(colors[0])


def _layer_order(rect: np.ndarray) -> list[int]:
    top, left = 0, 0
    bottom, right = map(int, rect.shape)
    order: list[int] = []

    while top < bottom and left < right:
        color = _boundary_color(rect, top, bottom, left, right)
        order.append(color)
        previous = (top, bottom, left, right)

        while top < bottom and np.all(rect[top, left:right] == color):
            top += 1
        while top < bottom and np.all(rect[bottom - 1, left:right] == color):
            bottom -= 1
        while left < right and top < bottom and np.all(rect[top:bottom, left] == color):
            left += 1
        while left < right and top < bottom and np.all(rect[top:bottom, right - 1] == color):
            right -= 1

        if (top, bottom, left, right) == previous:
            raise AssertionError("layer peeling made no progress")

    if len(order) < 3:
        raise AssertionError("expected at least three layers")
    if len(order) != len(set(order)):
        raise AssertionError("layer colors must be unique")
    return order


def _find_code_rect(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], list[int]]:
    candidates: list[tuple[tuple[int, int, int, int], list[int]]] = []
    for coords in _components(grid != bg):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if coords.shape[0] != area:
            continue
        rect = grid[r0 : r1 + 1, c0 : c1 + 1]
        try:
            order = _layer_order(rect)
        except AssertionError:
            continue
        candidates.append(((r0, c0, r1, c1), order))

    assert len(candidates) == 1, f"expected one nested rectangle, found {len(candidates)}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the unique nested rectangle as a cyclic color legend and apply it to every foreground pixel.

    Input structure
    - The background is the most common color.
    - Exactly one foreground component is a solid rectangle filled by at least
      three nested monochromatic rectangular layers with distinct colors.
    - All other foreground pixels or sprites use only the colors that appear in
      that layered rectangle.

    Rule
    - Read the layered rectangle from the outermost layer to the innermost layer
      to obtain a color order c1, c2, ..., ck.
    - Build the cyclic substitution c1->c2, c2->c3, ..., ck->c1.
    - Recolor every non-background pixel in the whole grid with that same
      substitution, including the rectangle itself, and leave the background
      unchanged.

    Key insight
    - The layered rectangle is not the target object but a radial legend whose
      nesting order defines one global cyclic color permutation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    _bbox, order = _find_code_rect(g, bg)

    foreground_colors = set(map(int, np.unique(g[g != bg])))
    assert foreground_colors.issubset(set(order)), "foreground colors must come from the code rectangle"

    mapping = {color: order[(idx + 1) % len(order)] for idx, color in enumerate(order)}
    out = g.copy()
    for src, dst in mapping.items():
        out[g == src] = dst
    return out


def _sample_thicknesses(rect_h: int, rect_w: int, layer_count: int) -> list[int] | None:
    thicks: list[int] = []
    h, w = rect_h, rect_w
    for layer_idx in range(layer_count - 1):
        remaining = layer_count - layer_idx - 1
        min_after = 2 * (remaining - 1) + 1
        max_t = min(2, (min(h, w) - min_after) // 2)
        if max_t < 1:
            return None
        t = random.randint(1, max_t)
        thicks.append(t)
        h -= 2 * t
        w -= 2 * t
    return thicks


def _build_layered_rect(rect_h: int, rect_w: int, colors: list[int], thicks: list[int]) -> np.ndarray:
    rect = np.full((rect_h, rect_w), colors[0], dtype=int)
    top, left = 0, 0
    bottom, right = rect_h, rect_w
    for idx, color in enumerate(colors):
        rect[top:bottom, left:right] = color
        if idx < len(thicks):
            t = thicks[idx]
            top += t
            left += t
            bottom -= t
            right -= t
    return rect


def _trim_sprite(sprite: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(sprite != bg)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return sprite[r0 : r1 + 1, c0 : c1 + 1]


def _make_sprite(layer_colors: list[int], bg: int) -> np.ndarray:
    mask = random.choice(SPRITE_TEMPLATES)
    mask = np.rot90(mask, k=random.randrange(4))
    cells = np.argwhere(mask)

    sprite = np.full(mask.shape, bg, dtype=int)
    if cells.shape[0] == 1 or random.random() < 0.45:
        color = random.choice(layer_colors)
        sprite[mask] = color
        return sprite

    color_count = min(len(layer_colors), random.randint(2, min(3, cells.shape[0])))
    palette = random.sample(layer_colors, color_count)
    assignments = [random.choice(palette) for _ in range(cells.shape[0])]
    if len(set(assignments)) == 1:
        assignments[0] = palette[0]
        assignments[-1] = palette[-1]
    for (r, c), color in zip(cells, assignments):
        sprite[int(r), int(c)] = int(color)
    return sprite


def _can_place(occupied: np.ndarray, sprite: np.ndarray, bg: int, top: int, left: int, padding: int) -> bool:
    h, w = sprite.shape
    H, W = occupied.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    non_bg = sprite != bg
    if not np.any(non_bg):
        return False
    if np.any(occupied[top : top + h, left : left + w] & non_bg):
        return False

    rr0 = max(0, top - padding)
    rr1 = min(H, top + h + padding)
    cc0 = max(0, left - padding)
    cc1 = min(W, left + w + padding)
    return not np.any(occupied[rr0:rr1, cc0:cc1])


def _place_sprite(grid: np.ndarray, occupied: np.ndarray, sprite: np.ndarray, bg: int, padding: int) -> bool:
    h, w = sprite.shape
    H, W = grid.shape
    row_candidates = list(range(H - h + 1))
    col_candidates = list(range(W - w + 1))
    random.shuffle(row_candidates)
    random.shuffle(col_candidates)

    attempts = 0
    for top in row_candidates:
        for left in col_candidates:
            attempts += 1
            if attempts > PLACEMENT_ATTEMPTS:
                return False
            if not _can_place(occupied, sprite, bg, top, left, padding):
                continue
            patch = sprite != bg
            grid[top : top + h, left : left + w][patch] = sprite[patch]
            occupied[top : top + h, left : left + w][patch] = True
            return True
    return False


def _color_counts(grid: np.ndarray) -> np.ndarray:
    return np.bincount(grid.reshape(-1), minlength=10)


def _expected_output(grid: np.ndarray, order: list[int], bg: int) -> np.ndarray:
    mapping = {color: order[(idx + 1) % len(order)] for idx, color in enumerate(order)}
    out = grid.copy()
    for src, dst in mapping.items():
        out[grid == src] = dst
    return out


def _generate_attempt() -> np.ndarray | None:
    H = random.randint(15, 30)
    W = random.randint(15, 30)
    max_rect_h = min(16, H - 4)
    max_rect_w = min(16, W - 4)
    if max_rect_h < 7 or max_rect_w < 7:
        return None

    rect_h = random.randint(7, max_rect_h)
    rect_w = random.randint(7, max_rect_w)
    max_layers = min(5, (min(rect_h, rect_w) + 1) // 2)
    if max_layers < 3:
        return None
    layer_count = random.randint(3, max_layers)
    thicks = _sample_thicknesses(rect_h, rect_w, layer_count)
    if thicks is None:
        return None

    bg = random.choice(ARC_COLORS)
    layer_colors = random.sample([color for color in ARC_COLORS if color != bg], layer_count)
    rect = _build_layered_rect(rect_h, rect_w, layer_colors, thicks)

    grid = np.full((H, W), bg, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    rect_top = random.randint(1, H - rect_h - 1)
    rect_left = random.randint(1, W - rect_w - 1)
    grid[rect_top : rect_top + rect_h, rect_left : rect_left + rect_w] = rect
    occupied[rect_top : rect_top + rect_h, rect_left : rect_left + rect_w] = True

    mandatory_colors = layer_colors.copy()
    random.shuffle(mandatory_colors)
    for color in mandatory_colors:
        sprite = np.array([[color]], dtype=int)
        if not _place_sprite(grid, occupied, sprite, bg, padding=1):
            return None

    extra_count = random.randint(2, 6)
    for _ in range(extra_count):
        sprite = _trim_sprite(_make_sprite(layer_colors, bg), bg)
        if not _place_sprite(grid, occupied, sprite, bg, padding=1):
            return None

    if np.count_nonzero(grid != bg) >= grid.size // 2:
        return None

    counts = _color_counts(grid)
    if counts[bg] <= max(counts[color] for color in ARC_COLORS if color != bg):
        return None

    rect_bbox, inferred_order = _find_code_rect(grid, bg)
    if inferred_order != layer_colors:
        return None

    expected = _expected_output(grid, layer_colors, bg)
    try:
        solved = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None

    r0, c0, r1, c1 = rect_bbox
    outside = grid.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg
    if np.count_nonzero(outside != bg) < layer_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid radial-code-table input")
