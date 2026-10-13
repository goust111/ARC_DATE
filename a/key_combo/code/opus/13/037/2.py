from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
MARKER_COLORS = (1, 2, 3, 4)
SPLIT = 5
TEMPLATE_COLOR = 8

GEN_ATTEMPTS = 400
GRID_SIZES = (17, 19, 21, 23, 25)
AREA_RANGE = (8, 15)
MARKER_COUNT_RANGE = (5, 9)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _count_components(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _is_single_component(mask: np.ndarray) -> bool:
    return bool(mask.any()) and _count_components(mask) == 1


def _strict_majority_color(values: np.ndarray) -> int:
    colors = [int(v) for v in values.ravel() if int(v) != BG]
    assert colors, "expected at least one marker"
    counts = Counter(colors)
    top = max(counts.values())
    winners = [color for color, count in counts.items() if count == top]
    assert len(winners) == 1, "expected a unique majority marker color"
    return int(winners[0])


def _validate_template(mask: np.ndarray) -> None:
    area = int(mask.sum())
    assert AREA_RANGE[0] <= area <= AREA_RANGE[1], "template area must be in range"
    assert _is_single_component(mask), "template must be a single 4-connected object"
    assert not np.array_equal(mask, np.fliplr(mask)), "template must not be LR symmetric"
    assert not np.array_equal(mask, np.flipud(mask)), "template must not be UD symmetric"
    assert not np.array_equal(mask, np.rot90(mask, 2)), "template should not collapse under 180-degree rotation"


def _validate_marker_quadrant(quad: np.ndarray) -> int:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(quad != BG)]
    assert MARKER_COUNT_RANGE[0] <= len(coords) <= MARKER_COUNT_RANGE[1], "marker count out of range"
    assert set(int(quad[r, c]) for r, c in coords) <= set(MARKER_COLORS), "unexpected marker colors"
    for i, (r0, c0) in enumerate(coords):
        for r1, c1 in coords[i + 1 :]:
            assert max(abs(r0 - r1), abs(c0 - c1)) >= 2, "markers must be isolated single pixels"
    return _strict_majority_color(quad)


def _split_grid(grid: np.ndarray) -> tuple[int, int]:
    grey_rows = np.where(np.all(grid == SPLIT, axis=1))[0]
    grey_cols = np.where(np.all(grid == SPLIT, axis=0))[0]
    assert len(grey_rows) == 1 and len(grey_cols) == 1, "expected one full grey row and one full grey column"
    mid_r = int(grey_rows[0])
    mid_c = int(grey_cols[0])
    assert mid_r == grid.shape[0] - mid_r - 1, "horizontal split must bisect the grid"
    assert mid_c == grid.shape[1] - mid_c - 1, "vertical split must bisect the grid"
    return mid_r, mid_c


def _expected_output_from_input(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    mid_r, mid_c = _split_grid(g)

    ul = g[:mid_r, :mid_c]
    ur = g[:mid_r, mid_c + 1 :]
    bl = g[mid_r + 1 :, :mid_c]
    br = g[mid_r + 1 :, mid_c + 1 :]

    assert set(np.unique(ul)).issubset({BG, TEMPLATE_COLOR}), "upper-left quadrant must contain only background and template"
    template = ul == TEMPLATE_COLOR
    _validate_template(template)

    ur_color = _validate_marker_quadrant(ur)
    bl_color = _validate_marker_quadrant(bl)
    br_color = _validate_marker_quadrant(br)

    out = np.full(g.shape, BG, dtype=int)
    out[mid_r, :] = SPLIT
    out[:, mid_c] = SPLIT
    out[:mid_r, :mid_c][template] = TEMPLATE_COLOR
    out[:mid_r, mid_c + 1 :][np.fliplr(template)] = ur_color
    out[mid_r + 1 :, :mid_c][np.flipud(template)] = bl_color
    out[mid_r + 1 :, mid_c + 1 :][np.rot90(template, 2)] = br_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the upper-left template into the other quadrants and recolor each copy by that quadrant's majority marker color.

    Input structure:
    - A black background grid is split into four equal quadrants by one full grey horizontal line and one full grey vertical line.
    - The upper-left quadrant contains one 4-connected teal template object.
    - The other three quadrants contain isolated single-pixel markers in blue, red, green, or yellow.

    Rule:
    - Read the teal object in the upper-left quadrant as the template.
    - In each non-template quadrant, count marker colors and take the unique majority color.
    - Stamp a mirrored copy of the template into the upper-right, lower-left, and lower-right quadrants:
      upper-right uses a left-right reflection, lower-left uses an up-down reflection, and lower-right uses both.
    - Recolor each stamped copy to that quadrant's majority color, erase the original markers, and keep the grey split lines and original teal template unchanged.

    Key insight:
    - Shape information comes only from the upper-left template, while color information comes independently from majority voting inside each target quadrant.
    """
    return _expected_output_from_input(grid)


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    cells = {start}
    frontier = set()
    frontier.update((nr, nc) for nr, nc in _neighbors4(*start) if 0 <= nr < h and 0 <= nc < w)
    while len(cells) < area:
        assert frontier, "frontier should not be empty while growing a connected set"
        nr, nc = random.choice(tuple(frontier))
        frontier.remove((nr, nc))
        if (nr, nc) in cells:
            continue
        cells.add((nr, nc))
        mask[nr, nc] = True
        for rr, cc in _neighbors4(nr, nc):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                frontier.add((rr, cc))
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _generate_template(quadrant_size: int) -> np.ndarray:
    for _ in range(200):
        area = random.randint(*AREA_RANGE)
        h = random.randint(3, min(6, quadrant_size))
        w = random.randint(3, min(6, quadrant_size))
        if area > h * w:
            continue
        mask = _crop_mask(_random_connected_mask(h, w, area))
        if mask.shape[0] > quadrant_size or mask.shape[1] > quadrant_size:
            continue
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        try:
            _validate_template(mask)
        except AssertionError:
            continue
        return mask
    raise RuntimeError("failed to generate a non-degenerate template")


def _distribute_positive_sum(total: int, parts: int, cap: int) -> list[int] | None:
    if parts == 0:
        return [] if total == 0 else None
    if total < parts or total > parts * cap:
        return None
    counts = [1] * parts
    total -= parts
    while total:
        idxs = [i for i, value in enumerate(counts) if value < cap]
        idx = random.choice(idxs)
        counts[idx] += 1
        total -= 1
    random.shuffle(counts)
    return counts


def _choose_marker_counts() -> tuple[dict[int, int], int]:
    for _ in range(100):
        total = random.randint(*MARKER_COUNT_RANGE)
        close_margin = random.random() < 0.45
        winner = random.choice(MARKER_COLORS)
        others = [color for color in MARKER_COLORS if color != winner]
        random.shuffle(others)
        used_other_count = random.randint(1, len(others))
        used_others = others[:used_other_count]

        feasible: list[tuple[int, list[int]]] = []
        for winner_count in range(2, total + 1):
            remaining = total - winner_count
            if remaining < used_other_count:
                continue
            dist = _distribute_positive_sum(remaining, used_other_count, winner_count - 1)
            if dist is None:
                continue
            if close_margin and max(dist) != winner_count - 1:
                continue
            feasible.append((winner_count, dist))
        if not feasible:
            continue

        winner_count, dist = random.choice(feasible)
        counts = {winner: winner_count}
        for color, count in zip(used_others, dist):
            counts[color] = count
        return counts, winner
    raise RuntimeError("failed to sample marker counts with a unique majority")


def _sample_isolated_positions(h: int, w: int, n: int) -> list[tuple[int, int]]:
    candidates = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= 2 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == n:
                return chosen
    raise RuntimeError("failed to place isolated marker pixels")


def _build_quadrant_markers(h: int, w: int) -> tuple[np.ndarray, int, dict[int, int]]:
    quad = np.full((h, w), BG, dtype=int)
    counts, winner = _choose_marker_counts()
    positions = _sample_isolated_positions(h, w, sum(counts.values()))
    idx = 0
    items = list(counts.items())
    random.shuffle(items)
    for color, count in items:
        for _ in range(count):
            r, c = positions[idx]
            idx += 1
            quad[r, c] = color
    return quad, winner, counts


def _build_expected(template: np.ndarray, majorities: tuple[int, int, int], size: int, top: int, left: int) -> np.ndarray:
    mid = size // 2
    q = mid
    out = np.full((size, size), BG, dtype=int)
    out[mid, :] = SPLIT
    out[:, mid] = SPLIT

    ul = np.zeros((q, q), dtype=bool)
    h, w = template.shape
    ul[top : top + h, left : left + w] = template

    out[:q, :q][ul] = TEMPLATE_COLOR
    out[:q, mid + 1 :][np.fliplr(ul)] = majorities[0]
    out[mid + 1 :, :q][np.flipud(ul)] = majorities[1]
    out[mid + 1 :, mid + 1 :][np.rot90(ul, 2)] = majorities[2]
    return out


def _semantic_guard(inp: np.ndarray, expected: np.ndarray, majorities: tuple[int, int, int]) -> None:
    out = main(inp)
    assert np.array_equal(out, expected), "solver output does not match the independently constructed target"
    assert len(set(majorities)) >= 2, "at least two target quadrants should use different colors"

    mid = inp.shape[0] // 2
    template = inp[:mid, :mid] == TEMPLATE_COLOR
    assert not np.array_equal(np.fliplr(template), template)
    assert not np.array_equal(np.flipud(template), template)
    assert not np.array_equal(np.rot90(template, 2), template)

    ur_mask = out[:mid, mid + 1 :] != BG
    bl_mask = out[mid + 1 :, :mid] != BG
    br_mask = out[mid + 1 :, mid + 1 :] != BG
    assert np.array_equal(ur_mask, np.fliplr(template)), "UR must be an LR mirror of the template"
    assert np.array_equal(bl_mask, np.flipud(template)), "BL must be a UD mirror of the template"
    assert np.array_equal(br_mask, np.rot90(template, 2)), "BR must be a double-axis mirror of the template"


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(GRID_SIZES)
    q = size // 2

    template = _generate_template(q)
    h, w = template.shape
    top = random.randint(0, q - h)
    left = random.randint(0, q - w)

    grid = np.full((size, size), BG, dtype=int)
    mid = q
    grid[mid, :] = SPLIT
    grid[:, mid] = SPLIT
    grid[top : top + h, left : left + w][template] = TEMPLATE_COLOR

    ur, ur_majority, ur_counts = _build_quadrant_markers(q, q)
    bl, bl_majority, bl_counts = _build_quadrant_markers(q, q)
    br, br_majority, br_counts = _build_quadrant_markers(q, q)

    majorities = (ur_majority, bl_majority, br_majority)
    if len(set(majorities)) < 2:
        return None

    grid[:q, mid + 1 :] = ur
    grid[mid + 1 :, :q] = bl
    grid[mid + 1 :, mid + 1 :] = br

    expected = _build_expected(template, majorities, size, top, left)

    # Require the majority votes to be visually meaningful, not three identical landslides.
    max_counts = [max(ur_counts.values()), max(bl_counts.values()), max(br_counts.values())]
    if all(value >= 4 for value in max_counts) and len(set(max_counts)) == 1 and len(set(majorities)) == 2:
        return None

    try:
        _semantic_guard(grid, expected, majorities)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid quadrant-mirror majority-color input")
