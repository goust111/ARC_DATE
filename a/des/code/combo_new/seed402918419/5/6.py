import random
from typing import Optional

import numpy as np


GENERATE_ATTEMPTS = 500
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_filled_rectangle(mask: np.ndarray) -> bool:
    if not np.any(mask):
        return False
    r0, c0, r1, c1 = _bbox(mask)
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    return int(mask.sum()) == area and bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))


def _reflect(r: int, c: int, orientation: str, axis_sum: int) -> tuple[int, int]:
    if orientation == "vertical":
        return r, axis_sum - c
    return axis_sum - r, c


def _valid_completions(
    base_bbox: tuple[int, int, int, int],
    hidden_mask: np.ndarray,
    visible_mask: np.ndarray,
) -> list[np.ndarray]:
    r0, c0, r1, c1 = base_bbox
    if np.any(hidden_mask & visible_mask) or not np.any(visible_mask):
        return []

    visible_coords = np.argwhere(visible_mask)
    completions: list[np.ndarray] = []

    for orientation in ("vertical", "horizontal"):
        lo, hi = (2 * c0, 2 * c1) if orientation == "vertical" else (2 * r0, 2 * r1)
        for axis_sum in range(lo, hi + 1):
            full_mask = np.zeros_like(visible_mask, dtype=bool)
            valid = True
            for r, c in visible_coords:
                rr, cc = _reflect(int(r), int(c), orientation, axis_sum)
                if rr < r0 or rr > r1 or cc < c0 or cc > c1:
                    valid = False
                    break
                full_mask[r, c] = True
                full_mask[rr, cc] = True
            if not valid:
                continue
            added = full_mask & ~visible_mask
            if not np.any(added):
                continue
            if np.any(added & ~hidden_mask):
                continue
            completions.append(full_mask)

    return completions


def _recover_pattern_mask(
    base_bbox: tuple[int, int, int, int],
    hidden_mask: np.ndarray,
    visible_mask: np.ndarray,
) -> Optional[np.ndarray]:
    completions = _valid_completions(base_bbox, hidden_mask, visible_mask)
    if len(completions) != 1:
        return None
    return completions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden symmetric pattern from beneath the overlapping occluder and output only that pattern.

    Input structure:
    - A dominant background surrounds one larger filled rectangle whose cells are blue, green, or red.
    - The red cells form a solid rectangular occluder layered above the larger rectangle.
    - Green cells are the visible part of a smaller symmetric pattern originally drawn inside the larger rectangle; the rest of that pattern may be hidden under red.

    Rule:
    - Identify the larger base rectangle and the solid red occluder rectangle.
    - Treat the red cells inside the base rectangle as unknown covered cells and the visible green cells as evidence for the hidden pattern.
    - Enumerate horizontal and vertical reflection axes, keep the unique completion whose mirrored green pixels stay inside the base rectangle and add new green cells only under the red occluder.
    - Output the completed green pattern in its original coordinates and set every other cell to the background color.

    Key insight:
    - The red rectangle does not belong to the target object; it is a mask hiding part of a symmetric green pattern, so solving requires separating the occlusion layer from the mirrored shape underneath.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(colors) == 3, "expected exactly three non-background colors"

    red_candidates = [color for color in colors if _is_filled_rectangle(g == color)]
    assert len(red_candidates) == 1, "expected a unique solid occluder rectangle"
    red_color = red_candidates[0]

    base_mask = g != bg
    base_bbox = _bbox(base_mask)
    r0, c0, r1, c1 = base_bbox
    assert np.all(base_mask[r0 : r1 + 1, c0 : c1 + 1]), "foreground union must form the base rectangle"

    hidden_mask = g == red_color
    remaining = [color for color in colors if color != red_color]
    remaining.sort(key=lambda color: int(np.count_nonzero(g == color)))

    tried: set[int] = set()
    pattern_color = None
    pattern_mask = None
    for green_color in remaining + remaining[::-1]:
        if green_color in tried:
            continue
        tried.add(green_color)
        visible_mask = g == green_color
        recovered = _recover_pattern_mask(base_bbox, hidden_mask, visible_mask)
        if recovered is not None:
            pattern_color = green_color
            pattern_mask = recovered
            break

    assert pattern_mask is not None and pattern_color is not None, "failed to recover a unique hidden pattern"
    out = np.full_like(g, bg)
    out[pattern_mask] = pattern_color
    return out


def _grow_connected_half(height: int, width: int, target: int, anchor: tuple[int, int]) -> np.ndarray:
    cells = {anchor}
    while len(cells) < target:
        r, c = random.choice(tuple(cells))
        dr, dc = random.choice(DIR4)
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            cells.add((nr, nc))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_symmetric_pattern(height: int, width: int, orientation: str) -> Optional[np.ndarray]:
    if orientation == "vertical":
        half_width = (width + 1) // 2
        boundary = half_width - 1
        area = height * half_width
        if area < 4:
            return None
        for _ in range(80):
            start = (random.randrange(height), boundary)
            target = random.randint(max(3, height // 2), max(3, min(area - 1, (3 * area) // 4)))
            half = _grow_connected_half(height, half_width, target, start)
            if not np.any(half[:, boundary]):
                continue
            if width % 2 == 1 and np.all(np.argwhere(half)[:, 1] == boundary):
                continue

            full = np.zeros((height, width), dtype=bool)
            for c in range(half_width):
                full[:, c] |= half[:, c]
                full[:, width - 1 - c] |= half[:, c]

            coords = np.argwhere(full)
            if coords[:, 0].ptp() == 0 or coords[:, 1].ptp() == 0:
                continue
            if np.all(full):
                continue
            return full
        return None

    half_height = (height + 1) // 2
    boundary = half_height - 1
    area = half_height * width
    if area < 4:
        return None
    for _ in range(80):
        start = (boundary, random.randrange(width))
        target = random.randint(max(3, width // 2), max(3, min(area - 1, (3 * area) // 4)))
        half = _grow_connected_half(half_height, width, target, start)
        if not np.any(half[boundary, :]):
            continue
        if height % 2 == 1 and np.all(np.argwhere(half)[:, 0] == boundary):
            continue

        full = np.zeros((height, width), dtype=bool)
        for r in range(half_height):
            full[r, :] |= half[r, :]
            full[height - 1 - r, :] |= half[r, :]

        coords = np.argwhere(full)
        if coords[:, 0].ptp() == 0 or coords[:, 1].ptp() == 0:
            continue
        if np.all(full):
            continue
        return full
    return None


def _hidden_side_mask(shape: tuple[int, int], orientation: str, side: str) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    if orientation == "vertical":
        axis_twice = width - 1
        cols = np.arange(width)
        if side == "left":
            mask[:, cols * 2 < axis_twice] = True
        else:
            mask[:, cols * 2 > axis_twice] = True
        return mask

    axis_twice = height - 1
    rows = np.arange(height)
    if side == "top":
        mask[rows * 2 < axis_twice, :] = True
    else:
        mask[rows * 2 > axis_twice, :] = True
    return mask


def _choose_red_bbox(
    base_height: int,
    base_width: int,
    pattern_top: int,
    pattern_left: int,
    pattern_mask: np.ndarray,
    orientation: str,
    side: str,
) -> Optional[tuple[int, int, int, int]]:
    pattern_height, pattern_width = pattern_mask.shape
    hidden_mask = pattern_mask & _hidden_side_mask(pattern_mask.shape, orientation, side)
    hidden_coords = np.argwhere(hidden_mask)
    if hidden_coords.size == 0:
        return None

    hr_rel, hc_rel = map(int, random.choice(hidden_coords.tolist()))
    if orientation == "vertical":
        half_boundary = (pattern_width - 1) // 2
        if side == "left":
            hidden_end = half_boundary - (1 if pattern_width % 2 == 1 else 0)
            if hidden_end < 0:
                return None
            c1 = random.randint(pattern_left + hc_rel, pattern_left + hidden_end)
            c0_lo = max(0, pattern_left - 2)
            c0_hi = min(c1 - 1, pattern_left + hc_rel)
            if c0_lo > c0_hi:
                c0_lo = c0_hi = max(0, c1 - 1)
            c0 = random.randint(c0_lo, c0_hi)
        else:
            hidden_start = (pattern_width + 1) // 2
            if hidden_start >= pattern_width:
                return None
            c0 = random.randint(pattern_left + hidden_start, pattern_left + hc_rel)
            c1_lo = max(c0 + 1, pattern_left + hc_rel)
            c1_hi = min(base_width - 1, pattern_left + pattern_width - 1 + 2)
            if c1_lo > c1_hi:
                return None
            c1 = random.randint(c1_lo, c1_hi)

        r0 = random.randint(max(0, pattern_top - 2), pattern_top + hr_rel)
        r1_lo = max(r0 + 1, pattern_top + hr_rel)
        r1_hi = min(base_height - 1, pattern_top + pattern_height - 1 + 2)
        if r1_lo > r1_hi:
            return None
        r1 = random.randint(r1_lo, r1_hi)
        return r0, c0, r1, c1

    half_boundary = (pattern_height - 1) // 2
    if side == "top":
        hidden_end = half_boundary - (1 if pattern_height % 2 == 1 else 0)
        if hidden_end < 0:
            return None
        r1 = random.randint(pattern_top + hr_rel, pattern_top + hidden_end)
        r0_lo = max(0, pattern_top - 2)
        r0_hi = min(r1 - 1, pattern_top + hr_rel)
        if r0_lo > r0_hi:
            r0_lo = r0_hi = max(0, r1 - 1)
        r0 = random.randint(r0_lo, r0_hi)
    else:
        hidden_start = (pattern_height + 1) // 2
        if hidden_start >= pattern_height:
            return None
        r0 = random.randint(pattern_top + hidden_start, pattern_top + hr_rel)
        r1_lo = max(r0 + 1, pattern_top + hr_rel)
        r1_hi = min(base_height - 1, pattern_top + pattern_height - 1 + 2)
        if r1_lo > r1_hi:
            return None
        r1 = random.randint(r1_lo, r1_hi)

    c0 = random.randint(max(0, pattern_left - 2), pattern_left + hc_rel)
    c1_lo = max(c0 + 1, pattern_left + hc_rel)
    c1_hi = min(base_width - 1, pattern_left + pattern_width - 1 + 2)
    if c1_lo > c1_hi:
        return None
    c1 = random.randint(c1_lo, c1_hi)
    return r0, c0, r1, c1


def _render_case(
    grid_shape: tuple[int, int],
    colors: tuple[int, int, int, int],
    base_bbox: tuple[int, int, int, int],
    pattern_origin: tuple[int, int],
    pattern_mask: np.ndarray,
    red_bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    bg, blue, red, green = colors
    height, width = grid_shape
    grid = np.full((height, width), bg, dtype=int)

    br0, bc0, br1, bc1 = base_bbox
    grid[br0 : br1 + 1, bc0 : bc1 + 1] = blue

    pr0, pc0 = pattern_origin
    ph, pw = pattern_mask.shape
    pattern_view = grid[pr0 : pr0 + ph, pc0 : pc0 + pw]
    pattern_view[pattern_mask] = green

    rr0, rc0, rr1, rc1 = red_bbox
    grid[rr0 : rr1 + 1, rc0 : rc1 + 1] = red

    expected = np.full_like(grid, bg)
    expected[pr0 : pr0 + ph, pc0 : pc0 + pw][pattern_mask] = green
    return grid, expected


def _gen_attempt() -> Optional[np.ndarray]:
    grid_height = random.randint(14, 26)
    grid_width = random.randint(14, 26)
    palette = random.sample(range(10), 4)
    bg, blue, red, green = palette

    base_height = random.randint(8, min(18, grid_height - 2))
    base_width = random.randint(8, min(18, grid_width - 2))
    if base_height * base_width >= (grid_height * grid_width) // 2:
        return None
    base_top = random.randint(1, grid_height - base_height - 1)
    base_left = random.randint(1, grid_width - base_width - 1)
    base_bbox = (base_top, base_left, base_top + base_height - 1, base_left + base_width - 1)

    orientation = random.choice(["vertical", "horizontal"])
    if orientation == "vertical":
        pattern_height = random.randint(4, min(base_height - 1, 9))
        pattern_width = random.randint(4, min(base_width - 1, 9))
        side = random.choice(["left", "right"])
    else:
        pattern_height = random.randint(4, min(base_height - 1, 9))
        pattern_width = random.randint(4, min(base_width - 1, 9))
        side = random.choice(["top", "bottom"])

    if pattern_height >= base_height or pattern_width >= base_width:
        return None

    pattern_mask = _make_symmetric_pattern(pattern_height, pattern_width, orientation)
    if pattern_mask is None:
        return None

    side_mask = _hidden_side_mask(pattern_mask.shape, orientation, side)
    if not np.any(pattern_mask & side_mask):
        return None

    pattern_top = random.randint(base_top, base_top + base_height - pattern_height)
    pattern_left = random.randint(base_left, base_left + base_width - pattern_width)

    red_local_bbox = _choose_red_bbox(
        base_height,
        base_width,
        pattern_top - base_top,
        pattern_left - base_left,
        pattern_mask,
        orientation,
        side,
    )
    if red_local_bbox is None:
        return None

    lr0, lc0, lr1, lc1 = red_local_bbox
    red_bbox = (base_top + lr0, base_left + lc0, base_top + lr1, base_left + lc1)

    grid, expected = _render_case(
        (grid_height, grid_width),
        (bg, blue, red, green),
        base_bbox,
        (pattern_top, pattern_left),
        pattern_mask,
        red_bbox,
    )

    if np.count_nonzero(grid == bg) <= max(
        np.count_nonzero(grid == blue),
        np.count_nonzero(grid == red),
        np.count_nonzero(grid == green),
    ):
        return None

    visible_green = int(np.count_nonzero(grid == green))
    full_green = int(np.count_nonzero(expected == green))
    visible_blue = int(np.count_nonzero(grid == blue))
    if visible_green < 3 or visible_green >= full_green or visible_blue <= visible_green:
        return None
    if _recover_pattern_mask(base_bbox, grid == red, grid == green) is None:
        return None
    if _recover_pattern_mask(base_bbox, grid == red, grid == blue) is not None:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occlusion-symmetry puzzle")
