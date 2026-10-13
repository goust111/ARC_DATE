from __future__ import annotations

import random
from typing import Iterable

import numpy as np


ORTHO_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for dr, dc in ORTHO_STEPS:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _round_nearest(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _rounded_center(coords: np.ndarray) -> tuple[int, int]:
    mean = coords.mean(axis=0)
    return _round_nearest(float(mean[0])), _round_nearest(float(mean[1]))


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, cell)) for cell in coords}


def _ordered_pair(
    comp_a: np.ndarray, comp_b: np.ndarray
) -> tuple[str, np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    center_a = _rounded_center(comp_a)
    center_b = _rounded_center(comp_b)

    if center_a[0] == center_b[0] and center_a[1] != center_b[1]:
        if center_a[1] < center_b[1]:
            return "horizontal", comp_a.copy(), comp_b.copy(), center_a, center_b
        return "horizontal", comp_b.copy(), comp_a.copy(), center_b, center_a

    if center_a[1] == center_b[1] and center_a[0] != center_b[0]:
        if center_a[0] < center_b[0]:
            return "vertical", comp_a.copy(), comp_b.copy(), center_a, center_b
        return "vertical", comp_b.copy(), comp_a.copy(), center_b, center_a

    ar0, ac0, ar1, ac1 = _bbox(comp_a)
    br0, bc0, br1, bc1 = _bbox(comp_b)
    rows_overlap = _intervals_overlap(ar0, ar1, br0, br1)
    cols_overlap = _intervals_overlap(ac0, ac1, bc0, bc1)

    if rows_overlap and ac1 < bc0:
        return "horizontal", comp_a.copy(), comp_b.copy(), center_a, center_b
    if rows_overlap and bc1 < ac0:
        return "horizontal", comp_b.copy(), comp_a.copy(), center_b, center_a
    if cols_overlap and ar1 < br0:
        return "vertical", comp_a.copy(), comp_b.copy(), center_a, center_b
    if cols_overlap and br1 < ar0:
        return "vertical", comp_b.copy(), comp_a.copy(), center_b, center_a

    if abs(center_a[1] - center_b[1]) >= abs(center_a[0] - center_b[0]):
        if center_a[1] < center_b[1]:
            return "horizontal", comp_a.copy(), comp_b.copy(), center_a, center_b
        return "horizontal", comp_b.copy(), comp_a.copy(), center_b, center_a
    if center_a[0] < center_b[0]:
        return "vertical", comp_a.copy(), comp_b.copy(), center_a, center_b
    return "vertical", comp_b.copy(), comp_a.copy(), center_b, center_a


def _merged_pair(comp_a: np.ndarray, comp_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis, first, second, center_first, center_second = _ordered_pair(comp_a, comp_b)
    midpoint = ((center_first[0] + center_second[0]) / 2.0, (center_first[1] + center_second[1]) / 2.0)

    if axis == "horizontal":
        step_first = np.array([0, 1], dtype=int)
        step_second = np.array([0, -1], dtype=int)
        assert int(first[:, 1].max()) < int(second[:, 1].min()), "horizontal pair must start separated left/right"
    else:
        step_first = np.array([1, 0], dtype=int)
        step_second = np.array([-1, 0], dtype=int)
        assert int(first[:, 0].max()) < int(second[:, 0].min()), "vertical pair must start separated top/bottom"

    merged_first = first.copy()
    merged_second = second.copy()

    while True:
        trial_first = merged_first + step_first
        trial_second = merged_second + step_second

        if _coord_set(trial_first) & _coord_set(trial_second):
            break

        if axis == "horizontal":
            if int(trial_first[:, 1].max()) >= int(trial_second[:, 1].min()):
                break
        else:
            if int(trial_first[:, 0].max()) >= int(trial_second[:, 0].min()):
                break

        merged_first = trial_first
        merged_second = trial_second

    merged_centers = (_rounded_center(merged_first), _rounded_center(merged_second))
    merged_midpoint = (
        (merged_centers[0][0] + merged_centers[1][0]) / 2.0,
        (merged_centers[0][1] + merged_centers[1][1]) / 2.0,
    )
    assert midpoint == merged_midpoint, "equal opposite slides must preserve the pair midpoint"
    return merged_first, merged_second


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge each same-colored fragment pair by sliding the mirrored halves toward their shared midpoint.

    Input structure:
    - The grid has a dominant black background and several non-background colors.
    - Each foreground color appears in exactly two separated 4-connected fragments.
    - For a given color, the two fragments are arranged either left/right with a shared row span or above/below with a shared column span.
    - The two same-colored fragments are mirrored halves of one symmetric figure.

    Rule:
    - For each color, find its two fragments and the cardinal line joining their rounded fragment centers.
    - Slide both fragments synchronously by one cell per step toward the midpoint of those centers, always moving along that line.
    - Stop at the closest placement that still keeps the two halves from crossing or overlapping.
    - Clear the original cells and draw the merged symmetric figure in the same color.

    Key insight:
    - Because each pair consists of mirrored halves that move equal distances in opposite directions, the pair midpoint stays fixed; solving the puzzle means compacting each matched pair to its nearest non-crossing symmetric placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    out = np.full_like(g, int(bg))

    colors = [int(color) for color in np.unique(g) if int(color) != int(bg)]
    assert colors, "expected at least one foreground color"

    for color in colors:
        comps = _components_of_color(g, int(color))
        assert len(comps) == 2, "each color must form exactly two fragments"
        merged_a, merged_b = _merged_pair(comps[0], comps[1])
        for coords in (merged_a, merged_b):
            out[coords[:, 0], coords[:, 1]] = int(color)

    return out


BG_COLOR = 0
GRID_HEIGHT_RANGE = (12, 24)
GRID_WIDTH_RANGE = (12, 24)
PAIR_COUNT_RANGE = (1, 3)
FRAGMENT_SIZE_RANGE = (2, 5)
GAP_RANGE = (1, 4)
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
PLACEMENT_RETRIES = 150
GENERATION_RETRIES = 400


def _neighbors(cell: tuple[int, int], shape: tuple[int, int]) -> Iterable[tuple[int, int]]:
    r, c = cell
    for dr, dc in ORTHO_STEPS:
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
            yield nr, nc


def _random_polyomino(size: int) -> np.ndarray:
    for _ in range(100):
        h = random.randint(1, min(4, int(size)))
        w = random.randint(1, min(4, int(size)))
        if h * w < int(size):
            continue

        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = {nbr for nbr in _neighbors(start, (h, w))}
        while len(cells) < int(size) and frontier:
            nxt = random.choice(tuple(frontier))
            frontier.remove(nxt)
            cells.add(nxt)
            for nbr in _neighbors(nxt, (h, w)):
                if nbr not in cells:
                    frontier.add(nbr)
        if len(cells) != int(size):
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        mask = mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        if random.choice((False, True)):
            mask = np.flipud(mask)
        if random.choice((False, True)):
            mask = np.fliplr(mask)
        return mask.copy()

    raise RuntimeError("failed to sample a connected fragment")


def _mask_coords(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return np.argwhere(mask) + np.array([int(r0), int(c0)], dtype=int)


def _paint(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _expanded_rect(
    top: int, left: int, height: int, width: int, grid_shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    r0 = max(0, int(top) - 1)
    c0 = max(0, int(left) - 1)
    r1 = min(grid_shape[0] - 1, int(top + height))
    c1 = min(grid_shape[1] - 1, int(left + width))
    return r0, c0, r1, c1


def _rect_free(occupied: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    return not bool(occupied[r0 : r1 + 1, c0 : c1 + 1].any())


def _occupy_rect(occupied: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _build_pair_spec(color: int) -> dict:
    orientation = random.choice(("horizontal", "vertical"))
    size = random.randint(FRAGMENT_SIZE_RANGE[0], FRAGMENT_SIZE_RANGE[1])
    base = _random_polyomino(size)
    gap = random.randint(GAP_RANGE[0], GAP_RANGE[1])

    if orientation == "horizontal":
        left_mask = base
        right_mask = np.fliplr(base)
        h, w = map(int, base.shape)
        patch_h = h
        patch_w = 2 * w + 2 * gap
        input_left = _mask_coords(left_mask, 0, 0)
        input_right = _mask_coords(right_mask, 0, w + 2 * gap)
        output_left = _mask_coords(left_mask, 0, gap)
        output_right = _mask_coords(right_mask, 0, gap + w)
    else:
        top_mask = base
        bottom_mask = np.flipud(base)
        h, w = map(int, base.shape)
        patch_h = 2 * h + 2 * gap
        patch_w = w
        input_left = _mask_coords(top_mask, 0, 0)
        input_right = _mask_coords(bottom_mask, h + 2 * gap, 0)
        output_left = _mask_coords(top_mask, gap, 0)
        output_right = _mask_coords(bottom_mask, gap + h, 0)

    return {
        "color": int(color),
        "orientation": orientation,
        "patch_h": int(patch_h),
        "patch_w": int(patch_w),
        "input_a": input_left,
        "input_b": input_right,
        "output_a": output_left,
        "output_b": output_right,
    }


def _place_specs(
    specs: list[dict], shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = shape
    inp = np.full(shape, BG_COLOR, dtype=int)
    out = np.full(shape, BG_COLOR, dtype=int)
    occupied = np.zeros(shape, dtype=bool)

    ordered = sorted(specs, key=lambda spec: spec["patch_h"] * spec["patch_w"], reverse=True)

    for spec in ordered:
        ph = int(spec["patch_h"])
        pw = int(spec["patch_w"])
        if ph > h or pw > w:
            return None

        positions = [(r0, c0) for r0 in range(h - ph + 1) for c0 in range(w - pw + 1)]
        random.shuffle(positions)

        placed = False
        for r0, c0 in positions[:PLACEMENT_RETRIES]:
            rect = _expanded_rect(r0, c0, ph, pw, shape)
            if not _rect_free(occupied, rect):
                continue

            coords_in_a = spec["input_a"] + np.array([int(r0), int(c0)], dtype=int)
            coords_in_b = spec["input_b"] + np.array([int(r0), int(c0)], dtype=int)
            coords_out_a = spec["output_a"] + np.array([int(r0), int(c0)], dtype=int)
            coords_out_b = spec["output_b"] + np.array([int(r0), int(c0)], dtype=int)
            color = int(spec["color"])

            _paint(inp, coords_in_a, color)
            _paint(inp, coords_in_b, color)
            _paint(out, coords_out_a, color)
            _paint(out, coords_out_b, color)
            _occupy_rect(occupied, rect)
            placed = True
            break

        if not placed:
            return None

    return inp, out


def _valid_structure(grid: np.ndarray) -> bool:
    bg = _get_bg(grid)
    colors = [int(color) for color in np.unique(grid) if int(color) != int(bg)]
    if not colors:
        return False
    for color in colors:
        comps = _components_of_color(grid, int(color))
        if len(comps) != 2:
            return False
        if any(not (FRAGMENT_SIZE_RANGE[0] <= int(len(comp)) <= FRAGMENT_SIZE_RANGE[1]) for comp in comps):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1])
    w = random.randint(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1])
    area = h * w
    max_pairs = PAIR_COUNT_RANGE[1]
    if area < 220:
        max_pairs = min(max_pairs, 2)
    pair_count = random.randint(PAIR_COUNT_RANGE[0], max_pairs)
    colors = random.sample(COLOR_POOL, pair_count)
    specs = [_build_pair_spec(color) for color in colors]

    placed = _place_specs(specs, (h, w))
    if placed is None:
        return None
    inp, expected = placed

    if not _valid_structure(inp):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
