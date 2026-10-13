from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
RED = 2
GREY = 5
TEMPLATE_COLORS = (1, 3, 4, 6, 7, 8, 9)

MAX_GENERATION_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (22, 29)
GRID_WIDTH_RANGE = (22, 29)
RECT_COUNT_RANGE = (3, 5)
RECT_SIZE_RANGE = (6, 10)
PLACEMENT_GAP = 1

TEMPLATE_PATTERNS = (
    np.array([[0, 0, 1], [2, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [0, 0, 1]], dtype=int),
    np.array([[0, 0, 1], [2, 1, 1], [2, 1, 0]], dtype=int),
    np.array([[0, 1, 0], [0, 0, 1], [1, 0, 1]], dtype=int),
)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_template_component(components: list[list[tuple[int, int]]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    corner_components = []
    for comp in components:
        r0, c0, r1, c1 = _bbox(comp)
        touches_corner = (
            (r0 == 0 and c0 == 0)
            or (r0 == 0 and c1 == w - 1)
            or (r1 == h - 1 and c0 == 0)
            or (r1 == h - 1 and c1 == w - 1)
        )
        if touches_corner:
            corner_components.append(comp)
    assert len(corner_components) == 1, "expected exactly one corner template component"
    return corner_components[0]


def _template_primary_color(template: np.ndarray) -> int:
    counts = Counter(int(v) for v in template.flat if int(v) != BLACK)
    assert counts, "template must contain colors"
    primary, primary_count = counts.most_common(1)[0]
    assert list(counts.values()).count(primary_count) == 1, "template primary color must be unique"
    return int(primary)


def _match_template(interior: np.ndarray, template: np.ndarray) -> bool:
    for k in range(4):
        rotated = np.rot90(template, k=k)
        th, tw = rotated.shape
        if th > interior.shape[0] or tw > interior.shape[1]:
            continue
        solid = rotated != BLACK
        for r in range(interior.shape[0] - th + 1):
            for c in range(interior.shape[1] - tw + 1):
                window = interior[r : r + th, c : c + tw]
                if np.array_equal(window[solid], rotated[solid]) and np.all(window[~solid] == GREY):
                    return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the corner template to decide whether each grey rectangle keeps its interior pattern or becomes a solid warning block.

    Input structure:
    - The grid has a black background, one small multicolor template tucked into a corner, and 3-5 separated non-black rectangular components.
    - Each large component is a grey filled rectangle whose interior may contain a rotated copy of the template while its border stays grey.
    - The template uses only non-black, non-grey colors and is strictly smaller than every rectangle interior.

    Rule:
    - Extract the corner template and determine its most frequent color.
    - For each non-template rectangle, inspect only the interior region and test whether any 0/90/180/270 degree rotation of the template matches exactly there.
    - If a match exists, recolor the rectangle border to the template's primary color and keep the interior unchanged; otherwise recolor the whole rectangle red. Erase the corner template to black.

    Key insight:
    - Template matching is the branch condition: the same grey rectangle is transformed in two different ways depending on whether its interior contains a rotated template instance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _components(g != BLACK)
    template_comp = _find_template_component(components, g.shape)
    tr0, tc0, tr1, tc1 = _bbox(template_comp)
    template = g[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
    primary = _template_primary_color(template)

    out = g.copy()
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = BLACK

    template_set = set(template_comp)
    for comp in components:
        if set(comp) == template_set:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        rect = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(rect != BLACK), "expected filled non-black rectangles"
        assert rect.shape[0] >= 4 and rect.shape[1] >= 4, "rectangles must have interior regions"
        interior = rect[1:-1, 1:-1]
        if _match_template(interior, template):
            out[r0, c0 : c1 + 1] = primary
            out[r1, c0 : c1 + 1] = primary
            out[r0 : r1 + 1, c0] = primary
            out[r0 : r1 + 1, c1] = primary
        else:
            out[r0 : r1 + 1, c0 : c1 + 1] = RED

    return out


def _pattern_with_colors() -> np.ndarray:
    base = TEMPLATE_PATTERNS[random.randrange(len(TEMPLATE_PATTERNS))]
    color_count = int(base.max()) + 1
    palette = random.sample(TEMPLATE_COLORS, color_count)
    template = np.zeros_like(base)
    for idx, color in enumerate(palette):
        template[base == idx] = color
    template = np.rot90(template, k=random.randrange(4))
    return template


def _corner_position(shape: tuple[int, int], patch_shape: tuple[int, int], corner: int) -> tuple[int, int]:
    h, w = shape
    ph, pw = patch_shape
    if corner == 0:
        return 0, 0
    if corner == 1:
        return 0, w - pw
    if corner == 2:
        return h - ph, 0
    return h - ph, w - pw


def _expand_bbox(box: tuple[int, int, int, int], margin: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return max(0, r0 - margin), max(0, c0 - margin), min(h - 1, r1 + margin), min(w - 1, c1 + margin)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _random_rectangles(
    shape: tuple[int, int],
    count: int,
    template_shape: tuple[int, int],
    blocked: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    min_inner = max(template_shape) + 1
    min_size = max(4, min_inner + 2)
    rects: list[tuple[int, int, int, int]] = []
    forbidden = [blocked]
    for _ in range(count):
        placed = False
        for _ in range(200):
            rh = random.randint(min_size, RECT_SIZE_RANGE[1])
            rw = random.randint(min_size, RECT_SIZE_RANGE[1])
            if rh >= h - 1 or rw >= w - 1:
                continue
            r0 = random.randint(1, h - rh - 1)
            c0 = random.randint(1, w - rw - 1)
            bbox = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            padded = _expand_bbox(bbox, PLACEMENT_GAP, shape)
            if any(_boxes_overlap(padded, other) for other in forbidden):
                continue
            rects.append(bbox)
            forbidden.append(padded)
            placed = True
            break
        if not placed:
            return None
    return rects


def _place_template_instance(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    template: np.ndarray,
    rotation: int,
) -> tuple[int, int, np.ndarray]:
    r0, c0, r1, c1 = rect
    rotated = np.rot90(template, k=rotation)
    th, tw = rotated.shape
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    top = random.randint(r0 + 1, r1 - th)
    left = random.randint(c0 + 1, c1 - tw)
    grid[top : top + th, left : left + tw] = rotated
    return top, left, rotated


def _expected_output(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    matches: set[int],
    primary: int,
    template_box: tuple[int, int, int, int],
) -> np.ndarray:
    out = grid.copy()
    tr0, tc0, tr1, tc1 = template_box
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = BLACK
    for idx, (r0, c0, r1, c1) in enumerate(rects):
        if idx in matches:
            out[r0, c0 : c1 + 1] = primary
            out[r1, c0 : c1 + 1] = primary
            out[r0 : r1 + 1, c0] = primary
            out[r0 : r1 + 1, c1] = primary
        else:
            out[r0 : r1 + 1, c0 : c1 + 1] = RED
    return out


def _validate_semantics(
    grid: np.ndarray,
    template: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    matches: set[int],
    rotations_used: list[int],
    expected: np.ndarray,
) -> bool:
    components = _components(grid != BLACK)
    if len(components) != len(rects) + 1:
        return False

    try:
        template_comp = _find_template_component(components, grid.shape)
    except AssertionError:
        return False

    tr0, tc0, tr1, tc1 = _bbox(template_comp)
    extracted = grid[tr0 : tr1 + 1, tc0 : tc1 + 1]
    if not np.array_equal(extracted, template):
        return False

    counts = Counter(int(v) for v in template.flat if int(v) != BLACK)
    top_count = max(counts.values())
    if list(counts.values()).count(top_count) != 1:
        return False

    if not (1 <= len(matches) < len(rects)):
        return False
    if len(rotations_used) != len(set(rotations_used)):
        return False

    for rect in rects:
        r0, c0, r1, c1 = rect
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            return False
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.shape[0] <= template.shape[0] or inner.shape[1] <= template.shape[1]:
            return False

    detected_matches = {
        idx
        for idx, (r0, c0, r1, c1) in enumerate(rects)
        if _match_template(grid[r0 + 1 : r1, c0 + 1 : c1], template)
    }
    if detected_matches != matches:
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.zeros((h, w), dtype=int)

    template = _pattern_with_colors()
    corner = random.randrange(4)
    tr0, tc0 = _corner_position(grid.shape, template.shape, corner)
    th, tw = template.shape
    grid[tr0 : tr0 + th, tc0 : tc0 + tw] = template
    template_box = (tr0, tc0, tr0 + th - 1, tc0 + tw - 1)
    blocked = _expand_bbox(template_box, PLACEMENT_GAP, grid.shape)

    rect_count = random.randint(*RECT_COUNT_RANGE)
    rects = _random_rectangles(grid.shape, rect_count, template.shape, blocked)
    if rects is None:
        return None

    for r0, c0, r1, c1 in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY

    max_successes = min(rect_count - 1, 4)
    success_count = random.randint(1, max_successes)
    match_indices = set(random.sample(range(rect_count), success_count))
    rotations = random.sample(range(4), success_count)

    for idx, rotation in zip(sorted(match_indices), rotations):
        _place_template_instance(grid, rects[idx], template, rotation)

    primary = _template_primary_color(template)
    expected = _expected_output(grid, rects, match_indices, primary, template_box)

    if not _validate_semantics(grid, template, rects, match_indices, rotations, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
