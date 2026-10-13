from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
PAYLOAD_COLORS = (BLUE, GREEN, YELLOW, PINK)

GRID_HALF_CHOICES = (6, 7, 8, 9)
TARGET_TOTAL_RANGE = (3, 5)
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _quadrant_slices(mid: int, size: int) -> list[tuple[slice, slice]]:
    assert size == 2 * mid + 1
    return [
        (slice(0, mid), slice(0, mid)),
        (slice(0, mid), slice(mid + 1, size)),
        (slice(mid + 1, size), slice(0, mid)),
        (slice(mid + 1, size), slice(mid + 1, size)),
    ]


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(rows) == 1 and len(cols) == 1, "expected one full grey cross"
    return int(rows[0]), int(cols[0])


def _mirror_mask_into_output(grid: np.ndarray, mid: int) -> np.ndarray:
    out = grid.copy()
    source = np.argwhere(grid[:mid, :mid] == RED)
    assert 5 <= len(source) <= 10, "upper-left red mask size must be 5..10"
    for r, c in source:
        rr = int(r)
        cc = int(c)
        out[rr, cc] = GREY
        out[rr, 2 * mid - cc] = GREY
        out[2 * mid - rr, cc] = GREY
        out[2 * mid - rr, 2 * mid - cc] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the upper-left mask across a central grey cross, then fill each quadrant's lone hole to equalize color multisets.

    Input structure:
    - The grid is an odd square with black background and one full grey row plus
      one full grey column crossing at the exact center.
    - The upper-left quadrant contains a 4-connected red mask, while every
      quadrant contains a small rectangle of colored pixels and black gaps.
    - In the three non-source quadrants, the future mirrored mask locations are
      currently black.

    Rule:
    - Use the central grey cross as horizontal and vertical reflection axes.
    - Recolor the upper-left red mask to grey and mirror it into the other three
      quadrants, keeping the cross and all mirrored mask pixels grey.
    - In each quadrant, take the tight bbox of all non-black cells after the
      mirror step, count only non-grey, non-black colors, and define the target
      multiset by the per-color maxima across the four quadrants.
    - Each quadrant then has exactly one black cell inside that bbox and is
      missing exactly one copy of exactly one color; fill that hole with the
      missing color.

    Key insight:
    - The mirrored grey mask changes which cells still count as usable blanks,
      so the right hole can only be identified after the symmetry step and then
      completed via cross-quadrant multiset comparison.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 1

    mid_r, mid_c = _find_cross(g)
    assert mid_r == mid_c and h == 2 * mid_r + 1
    mid = mid_r

    mirrored = _mirror_mask_into_output(g, mid)
    out = mirrored.copy()

    counts_by_quadrant: list[Counter[int]] = []
    holes: list[tuple[int, int]] = []
    for row_slice, col_slice in _quadrant_slices(mid, h):
        quad = mirrored[row_slice, col_slice]
        box = _bbox(quad != BG)
        assert box is not None, "each quadrant must have a non-black effective area"
        r0, c0, r1, c1 = box
        patch = quad[r0 : r1 + 1, c0 : c1 + 1]

        black_cells = np.argwhere(patch == BG)
        assert len(black_cells) == 1, "each quadrant must have one black hole after mirroring"
        local_hole_r, local_hole_c = map(int, black_cells[0])
        holes.append((int(row_slice.start + r0 + local_hole_r), int(col_slice.start + c0 + local_hole_c)))

        payload = patch[(patch != BG) & (patch != GREY)]
        counts_by_quadrant.append(Counter(map(int, payload.reshape(-1))))

    target: Counter[int] = Counter()
    for counts in counts_by_quadrant:
        for color, count in counts.items():
            target[color] = max(target[color], count)
    assert target, "expected at least one payload color"

    for counts, (hr, hc) in zip(counts_by_quadrant, holes):
        missing = target - counts
        assert len(missing) == 1, "each quadrant must be missing exactly one color"
        color, amount = next(iter(missing.items()))
        assert amount == 1, "each quadrant must be short by one pixel"
        out[hr, hc] = int(color)

    return out


def _rect_candidates(q: int) -> list[tuple[int, int, int]]:
    options = []
    for height in range(3, min(5, q) + 1):
        for width in range(3, min(5, q) + 1):
            area = height * width
            for target_total in range(TARGET_TOTAL_RANGE[0], TARGET_TOTAL_RANGE[1] + 1):
                mask_size = area - target_total
                if 5 <= mask_size <= 10:
                    options.append((height, width, target_total))
    return options


def _neighbors4(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < height and 0 <= nc < width:
            out.append((nr, nc))
    return out


def _random_connected_mask(height: int, width: int, size: int) -> np.ndarray | None:
    for _ in range(80):
        mask = np.zeros((height, width), dtype=bool)
        seed = (random.randrange(height), random.randrange(width))
        mask[seed] = True
        while int(mask.sum()) < size:
            frontier = set()
            for r, c in np.argwhere(mask):
                frontier.update(
                    (nr, nc)
                    for nr, nc in _neighbors4(int(r), int(c), height, width)
                    if not mask[nr, nc]
                )
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True

        if int(mask.sum()) != size:
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue
        nonmask = ~mask
        interior = nonmask.copy()
        interior[[0, height - 1], :] = False
        interior[:, [0, width - 1]] = False
        if not np.any(interior):
            continue
        return mask
    return None


def _positive_composition(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in [*cuts, total]:
        values.append(cut - prev)
        prev = cut
    random.shuffle(values)
    return values


def _choose_missing_colors(colors: list[int]) -> list[int] | None:
    for _ in range(80):
        missing = [random.choice(colors) for _ in range(4)]
        if len(set(missing)) < 2:
            continue
        if any(all(color == miss for miss in missing) for color in colors):
            continue
        return missing
    return None


def _quadrant_local_cell(q: int, rect_r0: int, rect_c0: int, rr: int, cc: int, quadrant: int) -> tuple[int, int]:
    base_r = rect_r0 + rr
    base_c = rect_c0 + cc
    if quadrant == 0:
        return base_r, base_c
    if quadrant == 1:
        return base_r, q - 1 - base_c
    if quadrant == 2:
        return q - 1 - base_r, base_c
    return q - 1 - base_r, q - 1 - base_c


def _absolute_cell(mid: int, quadrant: int, local_r: int, local_c: int) -> tuple[int, int]:
    if quadrant == 0:
        return local_r, local_c
    if quadrant == 1:
        return local_r, mid + 1 + local_c
    if quadrant == 2:
        return mid + 1 + local_r, local_c
    return mid + 1 + local_r, mid + 1 + local_c


def _build_expected(grid: np.ndarray, missing_colors: list[int]) -> np.ndarray:
    out = main(grid)
    mirrored = _mirror_mask_into_output(grid, (grid.shape[0] - 1) // 2)
    mid = (grid.shape[0] - 1) // 2
    counts = []
    holes = []
    for row_slice, col_slice in _quadrant_slices(mid, grid.shape[0]):
        quad = mirrored[row_slice, col_slice]
        r0, c0, r1, c1 = _bbox(quad != BG)
        patch = quad[r0 : r1 + 1, c0 : c1 + 1]
        hole = np.argwhere(patch == BG)
        holes.append((int(row_slice.start + r0 + hole[0, 0]), int(col_slice.start + c0 + hole[0, 1])))
        counts.append(Counter(map(int, patch[(patch != BG) & (patch != GREY)].reshape(-1))))
    target = Counter()
    for counter in counts:
        for color, amount in counter.items():
            target[color] = max(target[color], amount)
    for quadrant, miss in enumerate(missing_colors):
        diff = target - counts[quadrant]
        assert diff == Counter({miss: 1})
        hr, hc = holes[quadrant]
        assert out[hr, hc] == miss
    return out


def _generate_attempt() -> np.ndarray | None:
    q = random.choice(GRID_HALF_CHOICES)
    rect_h, rect_w, target_total = random.choice(_rect_candidates(q))
    mask = _random_connected_mask(rect_h, rect_w, rect_h * rect_w - target_total)
    if mask is None:
        return None

    rect_r0 = random.randint(0, q - rect_h)
    rect_c0 = random.randint(0, q - rect_w)
    nonmask_cells = [(r, c) for r in range(rect_h) for c in range(rect_w) if not mask[r, c]]
    interior_nonmask = [(r, c) for r, c in nonmask_cells if 0 < r < rect_h - 1 and 0 < c < rect_w - 1]
    if not interior_nonmask:
        return None

    color_count = random.randint(2, min(4, target_total))
    colors = random.sample(PAYLOAD_COLORS, color_count)
    target_counts = Counter(dict(zip(colors, _positive_composition(target_total, color_count))))
    missing_colors = _choose_missing_colors(colors)
    if missing_colors is None:
        return None

    size = 2 * q + 1
    grid = np.full((size, size), BG, dtype=int)
    grid[q, :] = GREY
    grid[:, q] = GREY

    for rr, cc in np.argwhere(mask):
        abs_r, abs_c = _absolute_cell(q, 0, rect_r0 + int(rr), rect_c0 + int(cc))
        grid[abs_r, abs_c] = RED

    expected_missing: list[int] = []
    for quadrant in range(4):
        hole_local = random.choice(interior_nonmask)
        counts = target_counts.copy()
        counts[missing_colors[quadrant]] -= 1
        if counts[missing_colors[quadrant]] == 0:
            del counts[missing_colors[quadrant]]
        values = []
        for color, amount in counts.items():
            values.extend([color] * amount)
        random.shuffle(values)

        cells = [cell for cell in nonmask_cells if cell != hole_local]
        random.shuffle(cells)
        assert len(cells) == len(values)
        for (rr, cc), color in zip(cells, values):
            local_r, local_c = _quadrant_local_cell(q, rect_r0, rect_c0, rr, cc, quadrant)
            abs_r, abs_c = _absolute_cell(q, quadrant, local_r, local_c)
            grid[abs_r, abs_c] = int(color)
        expected_missing.append(missing_colors[quadrant])

    if np.count_nonzero(grid[:q, :q] == RED) != int(mask.sum()):
        return None
    try:
        expected = _build_expected(grid, expected_missing)
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
