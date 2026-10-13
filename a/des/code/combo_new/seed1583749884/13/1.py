import random
from collections import Counter, deque

import numpy as np


BG = 0
SEPARATOR = 5
TARGET = 9
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8)
GENERATION_RETRIES = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_signature(points: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(points)
    return tuple(sorted((r - r0, c - c0) for r, c in points))


def _corner_background(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return Counter(corners).most_common(1)[0][0]


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates = []
    for c in range(1, w - 1):
        color = int(grid[0, c])
        if color == bg:
            continue
        if not np.all(grid[:, c] == color):
            continue
        if not np.any(grid[:, :c] != bg):
            continue
        if not np.any(grid[:, c + 1 :] != bg):
            continue
        candidates.append(c)
    assert len(candidates) == 1, f"expected one separator column, got {candidates}"
    return candidates[0]


def _extract_source_holes(left: np.ndarray, bg: int) -> list[tuple[tuple[tuple[int, int], ...], int]]:
    objects = _components4(left != bg)
    assert objects, "expected at least one source template"
    result = []
    for comp in objects:
        r0, c0, r1, c1 = _bbox(comp)
        sub = left[r0 : r1 + 1, c0 : c1 + 1]
        colors = {int(left[r, c]) for r, c in comp}
        assert len(colors) == 1, "each source object must be monochrome"
        color = colors.pop()
        assert np.all(sub[0, :] == color)
        assert np.all(sub[-1, :] == color)
        assert np.all(sub[:, 0] == color)
        assert np.all(sub[:, -1] == color)
        hole_mask = sub == bg
        assert hole_mask.any(), "source template must contain a hole"
        hole_components = _components4(hole_mask)
        assert len(hole_components) == 1, "expected exactly one source hole"
        hole = hole_components[0]
        assert all(0 < r < sub.shape[0] - 1 and 0 < c < sub.shape[1] - 1 for r, c in hole)
        result.append((_shape_signature(hole), color))
    signatures = [signature for signature, _ in result]
    assert len(signatures) == len(set(signatures)), "source hole shapes must be unique"
    return result


def _extract_target_holes(right: np.ndarray, bg: int) -> tuple[int, dict[tuple[tuple[int, int], ...], list[tuple[int, int]]]]:
    components = _components4(right != bg)
    assert len(components) == 1, "expected exactly one target rectangle"
    comp = components[0]
    r0, c0, r1, c1 = _bbox(comp)
    sub = right[r0 : r1 + 1, c0 : c1 + 1]
    fill_colors = {int(right[r, c]) for r, c in comp}
    assert len(fill_colors) == 1, "target rectangle must be monochrome"
    fill = fill_colors.pop()
    assert np.all(sub[0, :] == fill)
    assert np.all(sub[-1, :] == fill)
    assert np.all(sub[:, 0] == fill)
    assert np.all(sub[:, -1] == fill)
    hole_mask = sub == bg
    assert hole_mask.any(), "target rectangle must contain holes"
    holes: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
    for hole in _components4(hole_mask):
        assert all(0 < r < sub.shape[0] - 1 and 0 < c < sub.shape[1] - 1 for r, c in hole)
        signature = _shape_signature(hole)
        assert signature not in holes, "target hole shapes must be unique"
        holes[signature] = [(r0 + r, c0 + c) for r, c in hole]
    return fill, holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hole in the target panel using the color of the source template with the same missing-shape mask.

    Input structure:
    - The background is black.
    - A full-height grey vertical separator splits the grid into a left source panel and a right target panel.
    - The left panel contains several monochrome rectangular templates, each with exactly one black interior hole.
    - The right panel contains one large monochrome rectangle with several black interior holes.

    Rule:
    - Find each source template and extract the normalized shape of its interior black hole.
    - Find each black hole inside the large right-panel rectangle and normalize its shape in the same way.
    - Match source holes to target holes by exact shape equality under translation only.
    - Fill each matched target hole with the color of its source template, while leaving the separator, source panel, and target rectangle otherwise unchanged.

    Key insight:
    - The visible colored parts are distractors; the real matching key is the hidden negative-space shape carved out of each source template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _corner_background(g)
    sep = _find_separator_col(g, bg)

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    source_holes = _extract_source_holes(left, bg)
    target_fill, target_holes = _extract_target_holes(right, bg)
    assert target_fill != bg, "target fill color must differ from background"
    assert len(source_holes) == len(target_holes), "source and target hole counts must match"

    out = g.copy()
    source_map = dict(source_holes)
    assert set(source_map) == set(target_holes), "source and target hole shapes must match exactly"
    for signature, coords in target_holes.items():
        color = source_map[signature]
        for r, c in coords:
            out[r, sep + 1 + c] = color
    return out


def _rotate_mask(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, k=turns)


def _connected_mask_from_size(rng: random.Random, h: int, w: int, cells: int) -> np.ndarray:
    start = (rng.randrange(h), rng.randrange(w))
    chosen = {start}
    frontier = {start}
    while len(chosen) < cells:
        candidates = []
        for r, c in frontier:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in chosen:
                    candidates.append((nr, nc))
        if not candidates:
            break
        nxt = rng.choice(candidates)
        chosen.add(nxt)
        frontier.add(nxt)
    if len(chosen) != cells:
        return np.zeros((0, 0), dtype=bool)
    rows = [r for r, _ in chosen]
    cols = [c for _, c in chosen]
    r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in chosen:
        mask[r - r0, c - c0] = True
    return mask


def _random_hole_mask(rng: random.Random) -> np.ndarray:
    modes = ("single", "bar", "rect", "ell", "blob")
    mode = rng.choice(modes)
    if mode == "single":
        return np.array([[True]], dtype=bool)
    if mode == "bar":
        length = rng.randint(2, 4)
        horizontal = rng.choice((False, True))
        if horizontal:
            return np.ones((1, length), dtype=bool)
        return np.ones((length, 1), dtype=bool)
    if mode == "rect":
        h = rng.randint(2, 3)
        w = rng.randint(2, 4)
        return np.ones((h, w), dtype=bool)
    if mode == "ell":
        arm_h = rng.randint(2, 4)
        arm_w = rng.randint(2, 4)
        mask = np.zeros((arm_h, arm_w), dtype=bool)
        mask[:, 0] = True
        mask[-1, :] = True
        return _rotate_mask(mask, rng.randrange(4))
    for _ in range(20):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        upper = min(h * w, 6)
        lower = min(max(3, (h * w) // 2), upper)
        cells = rng.randint(lower, upper)
        mask = _connected_mask_from_size(rng, h, w, cells)
        if mask.size > 0:
            return mask
    return np.ones((2, 2), dtype=bool)


def _place_boxes(
    rng: random.Random,
    canvas_h: int,
    canvas_w: int,
    sizes: list[tuple[int, int]],
    *,
    border: int,
    gap: int,
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    for idx in order:
        box_h, box_w = sizes[idx]
        r_lo = border
        r_hi = canvas_h - box_h - border
        c_lo = border
        c_hi = canvas_w - box_w - border
        if r_lo > r_hi or c_lo > c_hi:
            return None
        candidates = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
        rng.shuffle(candidates)
        placed = None
        for r, c in candidates:
            rr0 = max(0, r - gap)
            rr1 = min(canvas_h, r + box_h + gap)
            cc0 = max(0, c - gap)
            cc1 = min(canvas_w, c + box_w + gap)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[rr0:rr1, cc0:cc1] = True
            placed = (r, c)
            break
        if placed is None:
            return None
        placements[idx] = placed
    return [(r, c) for r, c in placements if r is not None and c is not None]


def _build_source_template(rng: random.Random, hole_mask: np.ndarray, color: int) -> np.ndarray:
    hole_h, hole_w = hole_mask.shape
    top = rng.randint(1, 2)
    bottom = rng.randint(1, 2)
    left = rng.randint(1, 2)
    right = rng.randint(1, 2)
    template = np.full((hole_h + top + bottom, hole_w + left + right), color, dtype=int)
    window = template[top : top + hole_h, left : left + hole_w]
    window[hole_mask] = BG
    return template


def _generate_shapes(rng: random.Random, count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    signatures: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(count):
        for _ in range(50):
            mask = _random_hole_mask(rng)
            points = [tuple(point) for point in np.argwhere(mask)]
            signature = _shape_signature(points)
            if signature in signatures:
                continue
            signatures.add(signature)
            shapes.append(mask)
            break
        else:
            return None
    return shapes


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    source_count = rng.randint(2, 4)
    hole_masks = _generate_shapes(rng, source_count)
    if hole_masks is None:
        return None
    colors = rng.sample(SOURCE_COLORS, source_count)
    templates = [_build_source_template(rng, hole_mask, color) for hole_mask, color in zip(hole_masks, colors)]

    max_template_h = max(template.shape[0] for template in templates)
    max_template_w = max(template.shape[1] for template in templates)
    full_h_min = max(12, max_template_h + 4)
    full_h_max = 22
    if full_h_min > full_h_max:
        return None
    full_h = rng.randint(full_h_min, full_h_max)

    left_w_min = max(9, max_template_w + 4)
    left_w_max = 13
    if left_w_min > left_w_max:
        return None
    left_w = rng.randint(left_w_min, left_w_max)
    source_sizes = [template.shape for template in templates]
    source_positions = _place_boxes(rng, full_h, left_w, source_sizes, border=1, gap=1)
    if source_positions is None:
        return None

    hole_sizes = [hole_mask.shape for hole_mask in hole_masks]
    max_hole_h = max(size[0] for size in hole_sizes)
    max_hole_w = max(size[1] for size in hole_sizes)
    target_h_min = max(7, max_hole_h + 4)
    target_h_max = min(full_h - 2, 15)
    if target_h_min > target_h_max:
        return None

    target_layout = None
    for _ in range(40):
        target_h = rng.randint(target_h_min, target_h_max)
        target_w_min = max(7, max_hole_w + 4)
        target_w_max = 14
        if target_w_min > target_w_max:
            return None
        target_w = rng.randint(target_w_min, target_w_max)
        layout = _place_boxes(rng, target_h, target_w, hole_sizes, border=1, gap=1)
        if layout is not None:
            target_layout = (target_h, target_w, layout)
            break
    if target_layout is None:
        return None
    target_h, target_w, hole_positions = target_layout

    right_w_min = target_w + 2
    right_w_max = 16
    if right_w_min > right_w_max:
        return None
    right_w = rng.randint(right_w_min, right_w_max)
    full_w = left_w + 1 + right_w
    if full_w > 30:
        return None

    target_row = rng.randint(1, full_h - target_h - 1)
    target_col = rng.randint(1, right_w - target_w - 1)

    grid = np.full((full_h, full_w), BG, dtype=int)
    sep = left_w
    grid[:, sep] = SEPARATOR

    for template, (r0, c0) in zip(templates, source_positions):
        h, w = template.shape
        grid[r0 : r0 + h, c0 : c0 + w] = template

    target_left = sep + 1 + target_col
    grid[target_row : target_row + target_h, target_left : target_left + target_w] = TARGET

    expected = grid.copy()
    for idx, (hole_r, hole_c) in enumerate(hole_positions):
        mask = hole_masks[idx]
        hole_h, hole_w = mask.shape
        hole_window = grid[
            target_row + hole_r : target_row + hole_r + hole_h,
            target_left + hole_c : target_left + hole_c + hole_w,
        ]
        hole_window[mask] = BG
        expected_window = expected[
            target_row + hole_r : target_row + hole_r + hole_h,
            target_left + hole_c : target_left + hole_c + hole_w,
        ]
        expected_window[mask] = colors[idx]

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
