from __future__ import annotations

import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _dominant_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values, minlength=10)
    return int(counts.argmax())


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _cropped_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = coords - np.array([r0, c0], dtype=int)
    mask[local[:, 0], local[:, 1]] = True
    return mask


def _coords_from_mask(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(np.asarray(mask, dtype=bool)).astype(int)


def _column_gap(top: np.ndarray, bottom: np.ndarray) -> int:
    top_max: dict[int, int] = {}
    bottom_min: dict[int, int] = {}
    for r, c in top:
        top_max[int(c)] = max(top_max.get(int(c), -10**9), int(r))
    for r, c in bottom:
        bottom_min[int(c)] = min(bottom_min.get(int(c), 10**9), int(r))
    common = set(top_max) & set(bottom_min)
    assert common, "vertically paired fragments must share projected columns"
    gap = min(bottom_min[c] - top_max[c] - 1 for c in common)
    assert gap >= 0, "paired fragments must not overlap"
    return int(gap)


def _row_gap(left: np.ndarray, right: np.ndarray) -> int:
    left_max: dict[int, int] = {}
    right_min: dict[int, int] = {}
    for r, c in left:
        left_max[int(r)] = max(left_max.get(int(r), -10**9), int(c))
    for r, c in right:
        right_min[int(r)] = min(right_min.get(int(r), 10**9), int(c))
    common = set(left_max) & set(right_min)
    assert common, "horizontally paired fragments must share projected rows"
    gap = min(right_min[r] - left_max[r] - 1 for r in common)
    assert gap >= 0, "paired fragments must not overlap"
    return int(gap)


def _vertical_contact_offset(top_mask: np.ndarray, bottom_mask: np.ndarray) -> int:
    top = _coords_from_mask(top_mask)
    bottom = _coords_from_mask(bottom_mask)
    top_max: dict[int, int] = {}
    bottom_min: dict[int, int] = {}
    for r, c in top:
        top_max[int(c)] = max(top_max.get(int(c), -10**9), int(r))
    for r, c in bottom:
        bottom_min[int(c)] = min(bottom_min.get(int(c), 10**9), int(r))
    common = set(top_max) & set(bottom_min)
    assert common
    return int(max(top_max[c] - bottom_min[c] + 1 for c in common))


def _horizontal_contact_offset(left_mask: np.ndarray, right_mask: np.ndarray) -> int:
    left = _coords_from_mask(left_mask)
    right = _coords_from_mask(right_mask)
    left_max: dict[int, int] = {}
    right_min: dict[int, int] = {}
    for r, c in left:
        left_max[int(r)] = max(left_max.get(int(r), -10**9), int(c))
    for r, c in right:
        right_min[int(r)] = min(right_min.get(int(r), 10**9), int(c))
    common = set(left_max) & set(right_min)
    assert common
    return int(max(left_max[r] - right_min[r] + 1 for r in common))


def _solve_pair(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bbox_a = _bbox(first)
    bbox_b = _bbox(second)
    row_sum_a = bbox_a[0] + bbox_a[2]
    row_sum_b = bbox_b[0] + bbox_b[2]
    col_sum_a = bbox_a[1] + bbox_a[3]
    col_sum_b = bbox_b[1] + bbox_b[3]

    if col_sum_a == col_sum_b:
        top, bottom = (first, second) if bbox_a[0] <= bbox_b[0] else (second, first)
        top_mask = _cropped_mask(top)
        bottom_mask = _cropped_mask(bottom)
        assert top_mask.shape == bottom_mask.shape
        assert np.array_equal(bottom_mask, np.flipud(top_mask)), "vertical pair must be horizontal mirrors"
        gap = _column_gap(top, bottom)
        top_shift = gap // 2
        bottom_shift = gap - top_shift
        solved_top = top + np.array([top_shift, 0], dtype=int)
        solved_bottom = bottom + np.array([-bottom_shift, 0], dtype=int)
        return solved_top, solved_bottom

    if row_sum_a == row_sum_b:
        left, right = (first, second) if bbox_a[1] <= bbox_b[1] else (second, first)
        left_mask = _cropped_mask(left)
        right_mask = _cropped_mask(right)
        assert left_mask.shape == right_mask.shape
        assert np.array_equal(right_mask, np.fliplr(left_mask)), "horizontal pair must be vertical mirrors"
        gap = _row_gap(left, right)
        left_shift = gap // 2
        right_shift = gap - left_shift
        solved_left = left + np.array([0, left_shift], dtype=int)
        solved_right = right + np.array([0, -right_shift], dtype=int)
        return solved_left, solved_right

    raise AssertionError("paired fragments must align by row band or column band")


def main(grid: np.ndarray) -> np.ndarray:
    """Pair same-colored fragments and slide each mirrored pair together until they form one symmetric figure.

    Input structure:
    - The background is azure.
    - Several small non-symmetric colored fragments are scattered on the grid, with exactly two fragments per color.
    - The two fragments of a color are mirror images of each other across a horizontal or vertical axis.
    - Each pair is either stacked with the same vertical center line or placed side by side with the same horizontal center line.

    Rule:
    - For each color, identify its two fragments and determine whether they are stacked or side by side.
    - If a pair is stacked, move the upper fragment downward and the lower fragment upward until their boundary pixels become adjacent; if a pair is side by side, move the left fragment rightward and the right fragment leftward until they become adjacent.
    - Redraw only the moved fragments at their merged locations and clear their original positions to the background.

    Key insight:
    - Color identifies the pairing, while shared row-band or column-band alignment reveals the mirror axis and therefore the direction in which the two asymmetric pieces must close toward the middle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _dominant_color(g)
    out = np.full_like(g, bg)

    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    for color in colors:
        components = _find_components(g == color)
        assert len(components) == 2, "each foreground color must appear in exactly two fragments"
        solved_a, solved_b = _solve_pair(components[0], components[1])
        for coords in (solved_a, solved_b):
            assert np.all(coords[:, 0] >= 0) and np.all(coords[:, 0] < g.shape[0])
            assert np.all(coords[:, 1] >= 0) and np.all(coords[:, 1] < g.shape[1])
            out[coords[:, 0], coords[:, 1]] = color
    return out


# --------------- generator ---------------

GEN_BG = 8
GEN_COLOR_POOL = [0, 1, 2, 3, 4, 5, 6, 7, 9]
GEN_GRID_MIN = 16
GEN_GRID_MAX = 28
GEN_PAIR_MIN = 2
GEN_PAIR_MAX = 4
GEN_MASK_SIDE_MIN = 2
GEN_MASK_SIDE_MAX = 5
GEN_MASK_AREA_MIN = 3
GEN_MASK_AREA_MAX = 8
GEN_EXTRA_GAP_MIN = 2
GEN_EXTRA_GAP_MAX = 8
GEN_REGION_PAD = 1
GEN_SHAPE_ATTEMPTS = 200
GEN_PLACE_ATTEMPTS = 250
GEN_TOTAL_ATTEMPTS = 400


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_connected_mask(axis: str) -> np.ndarray:
    for _ in range(GEN_SHAPE_ATTEMPTS):
        h = random.randint(GEN_MASK_SIDE_MIN, GEN_MASK_SIDE_MAX)
        w = random.randint(GEN_MASK_SIDE_MIN, GEN_MASK_SIDE_MAX)
        max_area = min(GEN_MASK_AREA_MAX, h * w - 1)
        if max_area < GEN_MASK_AREA_MIN:
            continue
        target_area = random.randint(GEN_MASK_AREA_MIN, max_area)

        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        while int(mask.sum()) < target_area:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                for dr, dc in DIRS4:
                    nr, nc = int(r) + dr, int(c) + dc
                    if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True

        if int(mask.sum()) != target_area:
            continue
        mask = _trim_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if bool(mask.all()):
            continue
        if np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.fliplr(mask)):
            continue
        if axis == "vertical" and np.array_equal(mask, np.flipud(mask)):
            continue
        if axis == "horizontal" and np.array_equal(mask, np.fliplr(mask)):
            continue
        if len(_find_components(mask)) != 1:
            continue
        return mask
    raise RuntimeError("failed to sample a valid fragment mask")


def _normalize_pair(coords_a: np.ndarray, coords_b: np.ndarray, out_a: np.ndarray, out_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    min_r = int(min(coords_a[:, 0].min(), coords_b[:, 0].min()))
    min_c = int(min(coords_a[:, 1].min(), coords_b[:, 1].min()))
    shift = np.array([-min_r, -min_c], dtype=int)
    in_a = coords_a + shift
    in_b = coords_b + shift
    moved_a = out_a + shift
    moved_b = out_b + shift
    height = int(max(in_a[:, 0].max(), in_b[:, 0].max()) + 1)
    width = int(max(in_a[:, 1].max(), in_b[:, 1].max()) + 1)
    return in_a, in_b, moved_a, moved_b, height, width


def _build_pair(axis: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    mask = _random_connected_mask(axis)
    mirrored = np.flipud(mask) if axis == "vertical" else np.fliplr(mask)
    if random.random() < 0.5:
        first_mask, second_mask = mask, mirrored
    else:
        first_mask, second_mask = mirrored, mask

    first = _coords_from_mask(first_mask)
    second = _coords_from_mask(second_mask)
    extra_gap = random.randint(GEN_EXTRA_GAP_MIN, GEN_EXTRA_GAP_MAX)
    first_shift = extra_gap // 2
    second_shift = extra_gap - first_shift

    if axis == "vertical":
        contact = _vertical_contact_offset(first_mask, second_mask)
        out_first = first.copy()
        out_second = second + np.array([contact, 0], dtype=int)
        in_first = first + np.array([-first_shift, 0], dtype=int)
        in_second = second + np.array([contact + second_shift, 0], dtype=int)
    else:
        contact = _horizontal_contact_offset(first_mask, second_mask)
        out_first = first.copy()
        out_second = second + np.array([0, contact], dtype=int)
        in_first = first + np.array([0, -first_shift], dtype=int)
        in_second = second + np.array([0, contact + second_shift], dtype=int)

    in_a, in_b, out_a, out_b, height, width = _normalize_pair(in_first, in_second, out_first, out_second)
    return (
        np.vstack([in_a, in_b]),
        np.vstack([out_a, out_b]),
        height,
        width,
    )


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - GEN_REGION_PAD)
    c0 = max(0, left - GEN_REGION_PAD)
    r1 = min(h, top + height + GEN_REGION_PAD)
    c1 = min(w, left + width + GEN_REGION_PAD)
    return not bool(occupied[r0:r1, c0:c1].any())


def _occupy(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = occupied.shape
    r0 = max(0, top - GEN_REGION_PAD)
    c0 = max(0, left - GEN_REGION_PAD)
    r1 = min(h, top + height + GEN_REGION_PAD)
    c1 = min(w, left + width + GEN_REGION_PAD)
    occupied[r0:r1, c0:c1] = True


def _render_pair(grid: np.ndarray, coords: np.ndarray, color: int, top: int, left: int) -> None:
    shifted = coords + np.array([top, left], dtype=int)
    grid[shifted[:, 0], shifted[:, 1]] = int(color)


def _generate_example() -> tuple[np.ndarray, np.ndarray]:
    h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    pair_count = random.randint(GEN_PAIR_MIN, GEN_PAIR_MAX)
    colors = random.sample(GEN_COLOR_POOL, pair_count)

    inp = np.full((h, w), GEN_BG, dtype=int)
    out = np.full((h, w), GEN_BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for color in colors:
        axis = random.choice(["vertical", "horizontal"])
        pair_in, pair_out, region_h, region_w = _build_pair(axis)

        placed = False
        for _ in range(GEN_PLACE_ATTEMPTS):
            if region_h > h or region_w > w:
                break
            top = random.randint(0, h - region_h)
            left = random.randint(0, w - region_w)
            if not _can_place(occupied, top, left, region_h, region_w):
                continue
            _occupy(occupied, top, left, region_h, region_w)
            _render_pair(inp, pair_in, color, top, left)
            _render_pair(out, pair_out, color, top, left)
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place all fragment pairs")

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_TOTAL_ATTEMPTS):
        try:
            inp, expected = _generate_example()
        except RuntimeError:
            continue
        try:
            produced = main(inp)
        except AssertionError:
            continue
        if np.array_equal(produced, expected):
            return inp
    raise RuntimeError("failed to generate a valid mirrored-fragment puzzle")
