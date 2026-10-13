from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

PANEL_SIZE = 4
MIN_TEMPLATE_AREA = 6
MAX_TEMPLATE_AREA = 12
MAX_MISSING = 3


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]):
                nxt = (int(nr), int(nc))
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
    return len(seen) == int(mask.sum())


def _bits_to_mask(bits: int) -> np.ndarray:
    mask = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
    for idx in range(PANEL_SIZE * PANEL_SIZE):
        if (bits >> idx) & 1:
            r, c = divmod(idx, PANEL_SIZE)
            mask[r, c] = True
    return mask


_PANEL_MASKS: list[tuple[int, int]] | None = None


def _panel_masks() -> list[tuple[int, int]]:
    global _PANEL_MASKS
    if _PANEL_MASKS is not None:
        return _PANEL_MASKS

    masks: list[tuple[int, int]] = []
    for bits in range(1, 1 << (PANEL_SIZE * PANEL_SIZE)):
        area = bits.bit_count()
        if not 3 <= area <= 8:
            continue
        mask = _bits_to_mask(bits)
        box = _bbox(mask)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        if r0 != 0 or c0 != 0:
            continue
        if r1 == 0 or c1 == 0:
            continue
        if area == (r1 + 1) * (c1 + 1):
            continue
        if not _is_connected(mask):
            continue
        masks.append((bits, area))
    _PANEL_MASKS = masks
    return masks


def _sample_template_split() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = _panel_masks()
    for _ in range(800):
        a_bits, a_area = random.choice(masks)
        partners = [
            (bits, area)
            for bits, area in masks
            if bits & a_bits == 0
            and MIN_TEMPLATE_AREA <= a_area + area <= MAX_TEMPLATE_AREA
            and _bbox(_bits_to_mask(bits | a_bits)) == (0, 0, PANEL_SIZE - 1, PANEL_SIZE - 1)
        ]
        if not partners:
            continue
        b_bits, _b_area = random.choice(partners)
        a_mask = _bits_to_mask(a_bits)
        b_mask = _bits_to_mask(b_bits)
        if random.random() < 0.5:
            a_mask, b_mask = b_mask, a_mask
        return a_mask, b_mask, a_mask | b_mask
    raise RuntimeError("failed to sample complementary panel masks")


def _separator_rows(grid: np.ndarray) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if np.all(np.asarray(grid[r, :], dtype=int) == GREY)]
    assert len(rows) == 2, "expected exactly two grey separator rows"
    return int(rows[0]), int(rows[1])


def _extract_panel(region: np.ndarray, color: int) -> np.ndarray:
    box = _bbox(region == color)
    assert box is not None, "expected a colored panel in the region"
    r0, c0, _r1, _c1 = box
    assert r0 + PANEL_SIZE <= region.shape[0] and c0 + PANEL_SIZE <= region.shape[1], (
        "expected room for a 4x4 panel window anchored at the panel's top-left foreground cell"
    )
    patch = np.asarray(region[r0 : r0 + PANEL_SIZE, c0 : c0 + PANEL_SIZE], dtype=int).copy()
    assert patch.shape == (PANEL_SIZE, PANEL_SIZE), "expected a 4x4 panel window"
    assert np.all((patch == BLACK) | (patch == color)), "panel window should only contain panel color and background"
    return patch == color


def _extract_template(grid: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    sep1, sep2 = _separator_rows(grid)
    top = np.asarray(grid[:sep1, :], dtype=int)
    middle = np.asarray(grid[sep1 + 1 : sep2, :], dtype=int)
    panel_a = _extract_panel(top, BLUE)
    panel_b = _extract_panel(middle, RED)
    template = panel_a | panel_b
    assert MIN_TEMPLATE_AREA <= int(template.sum()) <= MAX_TEMPLATE_AREA, "unexpected template area"
    return (sep1, sep2), template


def _matching_copy_origins(lower: np.ndarray, template: np.ndarray) -> list[tuple[int, int]]:
    th, tw = map(int, template.shape)
    template_area = int(template.sum())
    origins: list[tuple[int, int]] = []
    for r0 in range(lower.shape[0] - th + 1):
        for c0 in range(lower.shape[1] - tw + 1):
            window = np.asarray(lower[r0 : r0 + th, c0 : c0 + tw], dtype=int)
            if np.any((window != BLACK) & (window != GREEN)):
                continue
            green_mask = window == GREEN
            green_count = int(green_mask.sum())
            if green_count == 0:
                continue
            if np.any(green_mask & ~template):
                continue
            missing = template_area - green_count
            if not 1 <= missing <= min(MAX_MISSING, template_area - 1):
                continue
            origins.append((int(r0), int(c0)))
    return origins


def _complete_with_template(grid: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    out = np.asarray(grid, dtype=int).copy()
    (_sep1, sep2), _ = _extract_template(out)
    lower = out[sep2 + 1 :, :]
    origins = _matching_copy_origins(lower, template)
    for r0, c0 in origins:
        patch = lower[r0 : r0 + PANEL_SIZE, c0 : c0 + PANEL_SIZE]
        patch[template] = GREEN
    return out, origins


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the two hidden panel fragments into one 4x4 template, then complete every matching partial copy below.

    Input structure:
    - A black background grid is split by two full-width grey separator rows into top, middle, and bottom regions.
    - The top region contains one blue 4x4 panel fragment, and the middle region contains one red 4x4 panel fragment.
    - Each fragment occupies some foreground cells of its own 4x4 window, and the two windows encode complementary parts of one hidden template.
    - The bottom region contains several non-touching green 4x4 partial copies whose green cells are a true subset of the hidden template.

    Rule:
    - Crop the blue panel window and the red panel window, align them pixel-by-pixel, and take the union of their foreground cells to recover the full 4x4 template.
    - Slide a 4x4 window across the bottom region.
    - A window is a partial copy exactly when it contains only black or green, every green cell lies on a template-foreground position, and the template is missing exactly 1 to 3 of its foreground cells there.
    - For every matching window, fill the template's missing foreground positions with green; keep the separators and both source panels unchanged.

    Key insight:
    - Neither colored panel shows the whole shape on its own; the full completion mask only appears after overlaying the blue and red fragments into a single template.
    """
    g = np.asarray(grid, dtype=int)
    (_sep1, sep2), template = _extract_template(g)
    out = g.copy()
    lower = out[sep2 + 1 :, :]
    for r0, c0 in _matching_copy_origins(lower, template):
        patch = lower[r0 : r0 + PANEL_SIZE, c0 : c0 + PANEL_SIZE]
        patch[template] = GREEN
    return out


def _choose_dimensions(copy_count: int) -> tuple[int, int, int, int, int]:
    for _ in range(400):
        width = random.randint(14, 20)
        top_h = random.randint(5, 7)
        mid_h = random.randint(5, 7)
        low_h = random.randint(8, 12)
        height = top_h + mid_h + low_h + 2
        if height > 30 or width > 30:
            continue
        if width >= copy_count * PANEL_SIZE + (copy_count - 1):
            return height, width, top_h, mid_h, low_h
    raise RuntimeError("failed to choose valid grid dimensions")


def _rectangles_separated(origins: list[tuple[int, int]], r0: int, c0: int) -> bool:
    r1 = r0 + PANEL_SIZE - 1
    c1 = c0 + PANEL_SIZE - 1
    for rr0, cc0 in origins:
        rr1 = rr0 + PANEL_SIZE - 1
        cc1 = cc0 + PANEL_SIZE - 1
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _choose_copy_origins(lower_shape: tuple[int, int], count: int) -> list[tuple[int, int]] | None:
    low_h, width = lower_shape
    candidates = [(r, c) for r in range(low_h - PANEL_SIZE + 1) for c in range(width - PANEL_SIZE + 1)]
    for _ in range(250):
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for r0, c0 in candidates:
            if _rectangles_separated(chosen, r0, c0):
                chosen.append((int(r0), int(c0)))
                if len(chosen) == count:
                    return chosen
    return None


def _copy_patch(template: np.ndarray, missing_coords: list[tuple[int, int]]) -> np.ndarray:
    patch = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    patch[template] = GREEN
    for r, c in missing_coords:
        patch[int(r), int(c)] = BLACK
    return patch


def _sample_missing_sets(
    template: np.ndarray, a_mask: np.ndarray, b_mask: np.ndarray, copy_count: int
) -> list[list[tuple[int, int]]] | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(template)]
    a_coords = [coord for coord in coords if bool(a_mask[coord])]
    b_coords = [coord for coord in coords if bool(b_mask[coord])]
    if not a_coords or not b_coords:
        return None

    for _ in range(300):
        missing_sets: list[list[tuple[int, int]]] = []
        for _copy_idx in range(copy_count):
            miss_n = random.randint(1, min(MAX_MISSING, len(coords) - 1))
            miss = random.sample(coords, miss_n)
            missing_sets.append(sorted(miss))

        missing_union = {coord for miss in missing_sets for coord in miss}
        if not (missing_union & set(a_coords)) or not (missing_union & set(b_coords)):
            continue
        if copy_count >= 2 and len({tuple(miss) for miss in missing_sets}) == 1:
            continue
        return missing_sets
    return None


def _render_instance() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    a_mask, b_mask, template = _sample_template_split()
    copy_count = random.randint(2, 3)
    height, width, top_h, mid_h, low_h = _choose_dimensions(copy_count)
    grid = np.zeros((height, width), dtype=int)
    sep1 = top_h
    sep2 = top_h + 1 + mid_h
    grid[sep1, :] = GREY
    grid[sep2, :] = GREY

    top_r = random.randint(0, top_h - PANEL_SIZE)
    top_c = random.randint(0, width - PANEL_SIZE)
    mid_r = random.randint(sep1 + 1, sep2 - PANEL_SIZE)
    mid_c = random.randint(0, width - PANEL_SIZE)

    grid[top_r : top_r + PANEL_SIZE, top_c : top_c + PANEL_SIZE][a_mask] = BLUE
    grid[mid_r : mid_r + PANEL_SIZE, mid_c : mid_c + PANEL_SIZE][b_mask] = RED

    origins_rel = _choose_copy_origins((low_h, width), copy_count)
    if origins_rel is None:
        raise RuntimeError("failed to place copy windows")

    missing_sets = _sample_missing_sets(template, a_mask, b_mask, copy_count)
    if missing_sets is None:
        raise RuntimeError("failed to sample missing sets")

    expected = grid.copy()
    abs_origins: list[tuple[int, int]] = []
    for (r0, c0), missing in zip(origins_rel, missing_sets):
        patch = _copy_patch(template, missing)
        rr = sep2 + 1 + r0
        abs_origins.append((int(rr), int(c0)))
        grid[rr : rr + PANEL_SIZE, c0 : c0 + PANEL_SIZE] = patch
        expected[rr : rr + PANEL_SIZE, c0 : c0 + PANEL_SIZE][template] = GREEN

    return grid, expected, a_mask, b_mask, abs_origins


def _semantic_check(
    grid: np.ndarray,
    expected: np.ndarray,
    a_mask: np.ndarray,
    b_mask: np.ndarray,
    abs_origins: list[tuple[int, int]],
) -> bool:
    try:
        (sep1, sep2), template = _extract_template(grid)
    except AssertionError:
        return False

    if int(template.sum()) <= max(int(a_mask.sum()), int(b_mask.sum())):
        return False

    lower = grid[sep2 + 1 :, :]
    matches = [(sep2 + 1 + r0, c0) for r0, c0 in _matching_copy_origins(lower, template)]
    if sorted(matches) != sorted(abs_origins):
        return False

    if not np.array_equal(main(grid), expected):
        return False

    a_only_grid, _ = _complete_with_template(grid, a_mask)
    b_only_grid, _ = _complete_with_template(grid, b_mask)
    if np.array_equal(a_only_grid, expected):
        return False
    if np.array_equal(b_only_grid, expected):
        return False

    if np.any(grid[sep1, :] != GREY) or np.any(grid[sep2, :] != GREY):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(600):
        try:
            grid, expected, a_mask, b_mask, abs_origins = _render_instance()
        except RuntimeError:
            continue
        if _semantic_check(grid, expected, a_mask, b_mask, abs_origins):
            return grid
    raise RuntimeError("failed to generate a valid overlay-completion ARC input")
