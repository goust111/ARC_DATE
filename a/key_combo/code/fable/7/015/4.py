"""ARC task: register two masks by their single-cell negative-space anchors."""

from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
OUTPUT_GREEN = 3
_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_RING = {
    (r, c)
    for r in (-1, 0, 1)
    for c in (-1, 0, 1)
    if (r, c) != (0, 0)
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected True components using the common (row, column) convention."""
    h, w = mask.shape
    unseen = mask.astype(bool).copy()
    result: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not unseen[r, c]:
                continue
            unseen[r, c] = False
            queue = deque([(r, c)])
            component: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                component.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and unseen[nr, nc]:
                        unseen[nr, nc] = False
                        queue.append((nr, nc))
            result.append(component)
    return result


def _single_cell_hole(foreground: np.ndarray) -> tuple[int, int]:
    """Find the unique enclosed 4-connected background component of area one."""
    h, w = foreground.shape
    enclosed: list[list[tuple[int, int]]] = []
    for component in _components(~foreground):
        if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in component):
            enclosed.append(component)
    assert len(enclosed) == 1, "each mask must enclose exactly one background region"
    assert len(enclosed[0]) == 1, "the enclosed hole must be a single cell"
    return enclosed[0][0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Register two separated foreground masks by their hidden holes and output their green intersection.

    Input structure:
    - A uniform divider row separates two equal-sized rectangular halves.
    - Each half contains a shared background color and one distinct foreground color.
    - Each foreground mask encloses exactly one single-cell background hole and no other hole.

    Rule:
    - Use the top half's coordinates for an output of one-half size.
    - Translate the bottom mask so its unique enclosed hole coincides with the top mask's hole.
    - Color green exactly the cells foreground in both registered masks; all other cells are background.
      Bottom-mask cells translated outside the output are clipped and therefore count as background.

    Key insight:
    - The registration anchors are never drawn: the unique negative-space holes, not the original grid
      coordinates or the masks' bounding boxes, determine how the two masks overlap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = g.shape
    assert height % 2 == 1 and height >= 7, "divider must leave two equal nonempty halves"

    divider_r = height // 2
    divider = g[divider_r]
    assert np.all(divider == divider[0]), "middle row must be a uniform divider"
    divider_color = int(divider[0])
    top = g[:divider_r]
    bottom = g[divider_r + 1 :]
    assert top.shape == bottom.shape and width > 0

    top_colors = set(map(int, np.unique(top)))
    bottom_colors = set(map(int, np.unique(bottom)))
    common = top_colors & bottom_colors
    assert len(top_colors) == 2 and len(bottom_colors) == 2
    assert len(common) == 1, "the halves must share only their background color"
    background = common.pop()
    assert background != OUTPUT_GREEN, "green must remain visible as the output color"
    assert divider_color not in top_colors | bottom_colors, "divider color must be distinct"

    top_fg_color = next(iter(top_colors - {background}))
    bottom_fg_color = next(iter(bottom_colors - {background}))
    assert top_fg_color != bottom_fg_color
    top_mask = top == top_fg_color
    bottom_mask = bottom == bottom_fg_color

    top_hole = _single_cell_hole(top_mask)
    bottom_hole = _single_cell_hole(bottom_mask)
    dr = top_hole[0] - bottom_hole[0]
    dc = top_hole[1] - bottom_hole[1]

    registered_bottom = np.zeros_like(bottom_mask)
    for r, c in np.argwhere(bottom_mask):
        nr, nc = int(r) + dr, int(c) + dc
        if 0 <= nr < top.shape[0] and 0 <= nc < top.shape[1]:
            registered_bottom[nr, nc] = True

    out = np.full(top.shape, background, dtype=int)
    out[top_mask & registered_bottom] = OUTPUT_GREEN
    return out


def _grow(seed: set[tuple[int, int]], target_size: int) -> set[tuple[int, int]] | None:
    """Grow a connected relative-coordinate polyomino inside a 7x7 workspace."""
    cells = set(seed)
    while len(cells) < target_size:
        frontier = {
            (r + dr, c + dc)
            for r, c in cells
            for dr, dc in _NEIGHBORS4
            if -3 <= r + dr <= 3
            and -3 <= c + dc <= 3
            and (r + dr, c + dc) != (0, 0)
            and (r + dr, c + dc) not in cells
        }
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return cells


def _relative_shape_valid(cells: set[tuple[int, int]]) -> bool:
    """Check connectivity and that relative origin is the sole one-cell hole."""
    canvas = np.zeros((9, 9), dtype=bool)
    for r, c in cells:
        canvas[r + 4, c + 4] = True
    if len(_components(canvas.astype(bool))) != 1:
        return False
    try:
        return _single_cell_hole(canvas) == (4, 4)
    except AssertionError:
        return False


def _sample_masks() -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    shared = _grow(_RING, random.randint(11, 15))
    if shared is None:
        return None
    first = _grow(shared, len(shared) + random.randint(3, 7))
    second = _grow(shared, len(shared) + random.randint(3, 7))
    if first is None or second is None:
        return None
    intersection = first & second
    first_min = (min(r for r, _ in first), min(c for _, c in first))
    second_min = (min(r for r, _ in second), min(c for _, c in second))
    bbox_shift = (first_min[0] - second_min[0], first_min[1] - second_min[1])
    bbox_registered_second = {
        (r + bbox_shift[0], c + bbox_shift[1]) for r, c in second
    }
    if not (
        _relative_shape_valid(first)
        and _relative_shape_valid(second)
        and len(first - second) >= 2
        and len(second - first) >= 2
        and len(intersection - _RING) >= 3
        and first & bbox_registered_second != intersection
    ):
        return None
    return first, second


def _anchor_choices(
    height: int, width: int, cells: set[tuple[int, int]], margin: int = 1
) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    max_r = max(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)
    return [
        (r, c)
        for r in range(margin - min_r, height - margin - max_r)
        for c in range(margin - min_c, width - margin - max_c)
    ]


def _paint_relative(
    half: np.ndarray, cells: set[tuple[int, int]], anchor: tuple[int, int], color: int
) -> None:
    ar, ac = anchor
    for dr, dc in cells:
        half[ar + dr, ac + dc] = color


def _semantic_candidate_valid(
    grid: np.ndarray,
    first: set[tuple[int, int]],
    second: set[tuple[int, int]],
    top_hole: tuple[int, int],
    bottom_hole: tuple[int, int],
    shifted_case: bool,
) -> tuple[bool, np.ndarray | None]:
    """Independent oracle plus anti-shortcut and topology checks for generated scenes."""
    half_h = grid.shape[0] // 2
    width = grid.shape[1]
    expected = np.full((half_h, width), BACKGROUND, dtype=int)
    registered_cells = {
        (top_hole[0] + dr, top_hole[1] + dc) for dr, dc in first & second
    }
    for r, c in registered_cells:
        expected[r, c] = OUTPUT_GREEN

    top_fg = grid[:half_h] != BACKGROUND
    bottom_fg = grid[half_h + 1 :] != BACKGROUND
    try:
        if _single_cell_hole(top_fg) != top_hole or _single_cell_hole(bottom_fg) != bottom_hole:
            return False, None
        solved = main(grid)
    except (AssertionError, ValueError):
        return False, None
    if not np.array_equal(solved, expected):
        return False, None

    raw_first = {(top_hole[0] + r, top_hole[1] + c) for r, c in first}
    raw_second = {(bottom_hole[0] + r, bottom_hole[1] + c) for r, c in second}
    if shifted_case and raw_first & raw_second == registered_cells:
        return False, None
    if len(registered_cells) <= len(_RING) or registered_cells == raw_first or registered_cells == raw_second:
        return False, None
    return True, expected


_seen_output_signatures: set[tuple[tuple[int, int], bytes]] = set()


def generate_input() -> np.ndarray:
    """Generate varied valid scenes, cycling aligned and genuinely shifted anchor cases."""
    # The second sample in each group of three is deliberately coordinate-aligned;
    # the other two require a visible nonzero registration shift.
    aligned_case = len(_seen_output_signatures) % 3 == 1
    palette = [1, 2, 4, 5, 6, 7, 8, 9]

    for _ in range(600):
        half_h = random.randint(10, 14)
        width = random.randint(10, 16)
        masks = _sample_masks()
        if masks is None:
            continue
        first, second = masks

        top_choices = _anchor_choices(half_h, width, first | second)
        bottom_choices = _anchor_choices(half_h, width, second)
        if not top_choices or not bottom_choices:
            continue
        top_hole = random.choice(top_choices)
        if aligned_case:
            if top_hole not in bottom_choices:
                continue
            bottom_hole = top_hole
        else:
            shifted_choices = [
                p
                for p in bottom_choices
                if abs(p[0] - top_hole[0]) + abs(p[1] - top_hole[1]) >= 2
            ]
            if not shifted_choices:
                continue
            bottom_hole = random.choice(shifted_choices)

        top_color, bottom_color, divider_color = random.sample(palette, 3)
        top = np.full((half_h, width), BACKGROUND, dtype=int)
        bottom = np.full((half_h, width), BACKGROUND, dtype=int)
        _paint_relative(top, first, top_hole, top_color)
        _paint_relative(bottom, second, bottom_hole, bottom_color)
        divider = np.full((1, width), divider_color, dtype=int)
        grid = np.vstack((top, divider, bottom))

        valid, expected = _semantic_candidate_valid(
            grid, first, second, top_hole, bottom_hole, not aligned_case
        )
        if not valid or expected is None:
            continue
        signature = (expected.shape, expected.tobytes())
        if signature in _seen_output_signatures:
            continue
        _seen_output_signatures.add(signature)
        return grid
    raise RuntimeError("failed to generate a semantically valid negative-space registration input")
