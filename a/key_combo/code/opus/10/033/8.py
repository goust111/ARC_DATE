from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

LEFT_OBJECT_COLORS = [BLUE, RED, GREEN, YELLOW, ORANGE]
RIGHT_FOREGROUND_COLORS = [BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, 8, 9]

GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 250

SMALL_OBJECT_MASKS = [
    np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalized_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _components_monochrome(grid: np.ndarray, bg: int = BLACK) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return components


def _contiguous_span(indices: np.ndarray) -> tuple[int, int]:
    assert indices.size > 0, "expected a non-empty side run"
    lo = int(indices[0])
    hi = int(indices[-1])
    assert np.array_equal(indices, np.arange(lo, hi + 1)), "expected a contiguous side run"
    return lo, hi


def _centered_span(source_lo: int, source_hi: int, length: int) -> tuple[int, int]:
    center2 = int(source_lo) + int(source_hi)
    lo = (center2 - (int(length) - 1)) // 2
    hi = (center2 + (int(length) - 1)) // 2
    return int(lo), int(hi)


def _find_separator_col(grid: np.ndarray) -> int:
    candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(candidates) == 1, "expected exactly one grey separator column"
    return int(candidates[0])


def _frame_candidates(grid: np.ndarray, separator_col: int) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    candidates = []
    for r in range(1, h - 1):
        for c in range(separator_col + 2, w - 1):
            patch = grid[r - 1 : r + 2, c - 1 : c + 2]
            center = int(patch[1, 1])
            if center in (BLACK, GREY):
                continue
            ring = [
                int(patch[0, 0]),
                int(patch[0, 1]),
                int(patch[0, 2]),
                int(patch[1, 0]),
                int(patch[1, 2]),
                int(patch[2, 0]),
                int(patch[2, 1]),
                int(patch[2, 2]),
            ]
            if len(set(ring)) != 1:
                continue
            ring_color = ring[0]
            if ring_color in (BLACK, GREY) or ring_color == center:
                continue
            candidates.append((r, c, ring_color, center))
    return candidates


def _shrinking_coords_from_mask(mask: np.ndarray, steps: int) -> set[tuple[int, int]]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty selected object"
    h, w = mask.shape
    r0, c0, r1, c1 = _bbox(coords)

    top_lo, top_hi = _contiguous_span(np.flatnonzero(mask[r0, :]))
    bottom_lo, bottom_hi = _contiguous_span(np.flatnonzero(mask[r1, :]))
    left_lo, left_hi = _contiguous_span(np.flatnonzero(mask[:, c0]))
    right_lo, right_hi = _contiguous_span(np.flatnonzero(mask[:, c1]))

    out: set[tuple[int, int]] = set()
    for k in range(1, int(steps) + 1):
        top_len = (top_hi - top_lo + 1) - 2 * k
        if top_len > 0:
            rr = r0 - k
            if 0 <= rr < h:
                cc0, cc1 = _centered_span(top_lo, top_hi, top_len)
                for cc in range(max(0, cc0), min(w - 1, cc1) + 1):
                    if not mask[rr, cc]:
                        out.add((rr, cc))

        bottom_len = (bottom_hi - bottom_lo + 1) - 2 * k
        if bottom_len > 0:
            rr = r1 + k
            if 0 <= rr < h:
                cc0, cc1 = _centered_span(bottom_lo, bottom_hi, bottom_len)
                for cc in range(max(0, cc0), min(w - 1, cc1) + 1):
                    if not mask[rr, cc]:
                        out.add((rr, cc))

        left_len = (left_hi - left_lo + 1) - 2 * k
        if left_len > 0:
            cc = c0 - k
            if 0 <= cc < w:
                rr0, rr1 = _centered_span(left_lo, left_hi, left_len)
                for rr in range(max(0, rr0), min(h - 1, rr1) + 1):
                    if not mask[rr, cc]:
                        out.add((rr, cc))

        right_len = (right_hi - right_lo + 1) - 2 * k
        if right_len > 0:
            cc = c1 + k
            if 0 <= cc < w:
                rr0, rr1 = _centered_span(right_lo, right_hi, right_len)
                for rr in range(max(0, rr0), min(h - 1, rr1) + 1):
                    if not mask[rr, cc]:
                        out.add((rr, cc))

    return out


def _render_output(left_shape: tuple[int, int], selected_coords: np.ndarray, color: int, steps: int) -> np.ndarray:
    out = np.zeros(left_shape, dtype=int)
    mask = np.zeros(left_shape, dtype=bool)
    mask[selected_coords[:, 0], selected_coords[:, 1]] = True
    out[mask] = int(color)
    for r, c in _shrinking_coords_from_mask(mask, steps):
        if out[r, c] == BLACK:
            out[r, c] = MAGENTA
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the left-side object whose color matches the unique framed center, then extend it with count-controlled shrinking caps.

    Input structure:
    - The grid has black background and one full grey separator column splitting a left object zone from a right marker zone.
    - The left zone contains several 4-connected monochrome objects in distinct colors.
    - The right zone contains distractor pixels plus exactly one 3x3 window whose eight border cells share one non-grey foreground color and whose center is a different foreground color.

    Rule:
    - Find the unique 3x3 monochrome frame on the right and read its center color C.
    - In the left zone, keep only the unique object of color C and remove every other object.
    - Let N be the number of removed left-side objects, then extend the kept object outward for N shrinking steps from its top, bottom, left, and right outer side runs; each step moves one cell farther and shortens the run by two cells while staying centered on the original side run.
    - Paint the added extension cells magenta and return only the columns left of the separator.

    Key insight:
    - The framed center is a cross-region lookup key, and the objects not selected still matter because their count becomes the extension depth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_col = _find_separator_col(g)
    frames = _frame_candidates(g, separator_col)
    assert len(frames) == 1, "expected exactly one valid right-side frame"
    chosen_color = int(frames[0][3])

    left = g[:, :separator_col]
    objects = _components_monochrome(left, bg=BLACK)
    assert len(objects) >= 3, "expected at least three left-side objects"

    matches = [obj for obj in objects if int(obj["color"]) == chosen_color]
    assert len(matches) == 1, "expected exactly one left-side object matching the framed center color"
    selected = matches[0]
    steps = len(objects) - 1

    return _render_output(left.shape, selected["coords"], chosen_color, steps)


def _random_selected_mask(height: int, width: int) -> np.ndarray:
    mask = np.ones((int(height), int(width)), dtype=bool)
    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    random.shuffle(interior)
    hole_count = random.randint(1, min(4, len(interior)))
    for r, c in interior[:hole_count]:
        mask[r, c] = False
    return mask


def _place_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> np.ndarray:
    coords = np.argwhere(mask)
    abs_coords = coords + np.array([int(r0), int(c0)])
    grid[abs_coords[:, 0], abs_coords[:, 1]] = int(color)
    return abs_coords


def _distance_ok(occupied: np.ndarray, r: int, c: int, padding: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, r - padding)
    r1 = min(h, r + padding + 1)
    c0 = max(0, c - padding)
    c1 = min(w, c + padding + 1)
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def _can_place_object(
    occupied: np.ndarray,
    forbidden: np.ndarray,
    mask: np.ndarray,
    r0: int,
    c0: int,
    left_width: int,
    edge_gap: int = 3,
    padding: int = 1,
) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    if r0 < edge_gap or c0 < edge_gap:
        return False
    if r0 + mh > h - edge_gap or c0 + mw > left_width - edge_gap:
        return False
    for rr, cc in np.argwhere(mask):
        ar = int(r0 + rr)
        ac = int(c0 + cc)
        if forbidden[ar, ac] or not _distance_ok(occupied, ar, ac, padding):
            return False
    return True


def _shape_candidates_for_count(unselected_count: int, left_width: int) -> tuple[np.ndarray, int, int] | None:
    min_width = max(2 * unselected_count + 1, 7)
    max_width = min(left_width - 6, 11)
    if min_width > max_width:
        return None
    width = random.randint(min_width, max_width)
    height = random.randint(5, 7)
    return _random_selected_mask(height, width), height, width


def _sample_selected_anchor(h: int, left_width: int, mask_shape: tuple[int, int]) -> tuple[int, int]:
    mh, mw = mask_shape
    row_modes = ["top", "bottom", "center"]
    col_modes = ["left", "right", "center"]

    max_r0 = h - mh - 3
    max_c0 = left_width - mw - 3
    assert max_r0 >= 3 and max_c0 >= 3

    row_mode = random.choice(row_modes)
    col_mode = random.choice(col_modes)

    if row_mode == "top":
        r0 = random.randint(3, min(6, max_r0))
    elif row_mode == "bottom":
        r0 = random.randint(max(3, max_r0 - 3), max_r0)
    else:
        center = (h - mh) // 2
        lo = max(3, center - 2)
        hi = min(max_r0, center + 2)
        r0 = random.randint(lo, hi)

    if col_mode == "left":
        c0 = random.randint(3, min(6, max_c0))
    elif col_mode == "right":
        c0 = random.randint(max(3, max_c0 - 3), max_c0)
    else:
        center = (left_width - mw) // 2
        lo = max(3, center - 2)
        hi = min(max_c0, center + 2)
        c0 = random.randint(lo, hi)

    return int(r0), int(c0)


def _right_zone_cells(h: int, w: int, separator_col: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h) for c in range(separator_col + 1, w)]


def _build_distractor_multiset(selected_color: int, ring_color: int) -> list[int]:
    palette = [c for c in RIGHT_FOREGROUND_COLORS if c != GREY]
    counts: dict[int, int] = {}
    counts[selected_color] = random.randint(1, 2)
    counts[ring_color] = random.randint(1, 2)
    other_colors = [c for c in palette if c not in counts]
    random.shuffle(other_colors)
    for color in other_colors[: random.randint(2, 4)]:
        counts[color] = random.randint(1, 2)
    cells: list[int] = []
    for color, count in counts.items():
        cells.extend([int(color)] * int(count))
    random.shuffle(cells)
    return cells


def _place_distractors(grid: np.ndarray, separator_col: int, blocked: np.ndarray, colors: list[int]) -> bool:
    h, w = grid.shape
    for color in colors:
        candidates = []
        for r in range(h):
            for c in range(separator_col + 1, w):
                if blocked[r, c] or int(grid[r, c]) != BLACK:
                    continue
                trial = grid.copy()
                trial[r, c] = int(color)
                if len(_frame_candidates(trial, separator_col)) == 1:
                    candidates.append((r, c))
        if not candidates:
            return False
        r, c = random.choice(candidates)
        grid[r, c] = int(color)
    return True


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(3, 5)
    unselected_count = object_count - 1

    h = random.randint(20, 26)
    left_width = random.randint(15, 18)
    right_width = random.randint(6, 9)
    w = left_width + 1 + right_width

    selected_shape = _shape_candidates_for_count(unselected_count, left_width)
    if selected_shape is None:
        return None
    selected_mask, _, _ = selected_shape
    selected_r0, selected_c0 = _sample_selected_anchor(h, left_width, selected_mask.shape)

    grid = np.zeros((h, w), dtype=int)
    grid[:, left_width] = GREY

    occupied = np.zeros((h, left_width), dtype=bool)
    selected_coords_local = np.argwhere(selected_mask) + np.array([selected_r0, selected_c0])
    selected_whole_mask = np.zeros((h, left_width), dtype=bool)
    selected_whole_mask[selected_coords_local[:, 0], selected_coords_local[:, 1]] = True
    occupied |= selected_whole_mask

    extension_reserved = np.zeros((h, left_width), dtype=bool)
    for r, c in _shrinking_coords_from_mask(selected_whole_mask, unselected_count):
        extension_reserved[r, c] = True
    forbidden_for_small = extension_reserved.copy()

    color_pool = random.sample(LEFT_OBJECT_COLORS, object_count)
    selected_color = random.choice(color_pool)
    other_colors = [c for c in color_pool if c != selected_color]
    grid[selected_coords_local[:, 0], selected_coords_local[:, 1]] = int(selected_color)

    used_keys: set[tuple[tuple[int, int], ...]] = {_normalized_key(selected_mask)}
    small_masks = SMALL_OBJECT_MASKS[:]
    random.shuffle(small_masks)
    chosen_small_masks: list[np.ndarray] = []
    for mask in small_masks:
        key = _normalized_key(mask)
        if key not in used_keys:
            used_keys.add(key)
            chosen_small_masks.append(mask)
        if len(chosen_small_masks) == unselected_count:
            break
    if len(chosen_small_masks) != unselected_count:
        return None

    for color, mask in zip(other_colors, chosen_small_masks):
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            r0 = random.randint(3, h - mask.shape[0] - 3)
            c0 = random.randint(3, left_width - mask.shape[1] - 3)
            if not _can_place_object(occupied, forbidden_for_small, mask, r0, c0, left_width):
                continue
            coords = _place_mask(grid[:, :left_width], mask, r0, c0, color)
            occupied[coords[:, 0], coords[:, 1]] = True
            placed = True
            break
        if not placed:
            return None

    ring_choices = [c for c in RIGHT_FOREGROUND_COLORS if c not in (selected_color, GREY, BLACK)]
    ring_color = random.choice(ring_choices)
    frame_center_r = random.randint(1, h - 2)
    frame_center_c = random.randint(left_width + 2, w - 2)
    blocked = np.zeros((h, w), dtype=bool)
    blocked[frame_center_r - 1 : frame_center_r + 2, frame_center_c - 1 : frame_center_c + 2] = True
    grid[frame_center_r - 1 : frame_center_r + 2, frame_center_c - 1 : frame_center_c + 2] = ring_color
    grid[frame_center_r, frame_center_c] = int(selected_color)

    distractors = _build_distractor_multiset(selected_color, ring_color)
    if not _place_distractors(grid, left_width, blocked, distractors):
        return None

    if len(_frame_candidates(grid, left_width)) != 1:
        return None

    left_components = _components_monochrome(grid[:, :left_width], bg=BLACK)
    if len(left_components) != object_count:
        return None
    if len({int(comp["color"]) for comp in left_components}) != object_count:
        return None

    matches = [comp for comp in left_components if int(comp["color"]) == selected_color]
    if len(matches) != 1:
        return None

    expected = _render_output((h, left_width), matches[0]["coords"], selected_color, unselected_count)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-8 input")
