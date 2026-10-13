from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append(np.asarray(coords, dtype=int))

    return comps


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    comps = _components(mask)
    labels = np.full(mask.shape, -1, dtype=int)
    for idx, coords in enumerate(comps):
        labels[coords[:, 0], coords[:, 1]] = idx
    return labels, comps


def _separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected exactly one full-height grey separator column"
    sep = cols[0]
    assert sep > 0 and sep < grid.shape[1] - 1, "separator must split the grid"
    assert sep == grid.shape[1] - sep - 1, "separator must leave equal-width halves"
    return sep


def _mirrored_left_col_from_right_local(right_local_c: int, left_w: int) -> int:
    return left_w - 1 - int(right_local_c)


def _mirrored_right_global_from_left(left_c: int, left_w: int) -> int:
    return 2 * left_w - int(left_c)


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror right-half singleton indicators into the left wall partition and flood-fill each reached region with that indicator color.

    Input structure:
    - The grid has black background and a full-height grey separator column splitting it into equal left and right halves.
    - In the left half, green pixels form a single 4-connected wall network that partitions the remaining black cells into three or four 4-connected regions.
    - In the right half, there is exactly one isolated colored indicator pixel for each left-half black region; indicator colors are neither black, grey, nor green.

    Rule:
    - Reflect every right-half indicator horizontally across the grey separator into left-half coordinates.
    - Discard the separator and the original right half; the output canvas is just the left half.
    - For each reflected indicator position, find the 4-connected black region containing that landing cell, treating green wall pixels as barriers.
    - Recolor that entire region with the indicator's color, preserving every green wall pixel.

    Key insight:
    - The right-half pixels are not copied directly; folding them across the separator determines which left-side enclosed region each color should flood-fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _separator_col(g)
    left = g[:, :sep].copy()
    right = g[:, sep + 1 :]
    left_h, left_w = left.shape
    assert right.shape == (left_h, left_w)

    allowed_left = {BLACK, GREEN}
    assert set(map(int, np.unique(left))).issubset(allowed_left), "left half should contain only black background and green walls"

    wall_comps = _components(left == GREEN)
    assert len(wall_comps) == 1, "green wall must be a single 4-connected component"

    labels, regions = _label_components(left == BLACK)
    assert 3 <= len(regions) <= 4, "expected three or four enclosed black regions"

    indicator_coords = np.argwhere((right != BLACK) & (right != GREY) & (right != GREEN))
    assert len(indicator_coords) == len(regions), "indicator count must match region count"
    indicator_colors = [int(right[r, c]) for r, c in indicator_coords]
    assert len(set(indicator_colors)) == len(indicator_colors), "indicator colors must be unique"

    out = left.copy()
    used_regions: set[int] = set()
    for r, right_local_c in indicator_coords:
        color = int(right[r, right_local_c])
        left_c = _mirrored_left_col_from_right_local(int(right_local_c), left_w)
        region_id = int(labels[int(r), left_c])
        assert region_id >= 0, "mirrored indicator must land inside a black region"
        assert region_id not in used_regions, "each region must receive exactly one indicator"
        used_regions.add(region_id)
        coords = regions[region_id]
        out[coords[:, 0], coords[:, 1]] = color

    assert len(used_regions) == len(regions)
    return out


# --------------- generator ---------------

LEFT_W = 6
HEIGHT_MIN = 9
HEIGHT_MAX = 15
REGION_CHOICES = (3, 4)
INDICATOR_COLORS = (BLUE, RED, YELLOW, PINK, ORANGE, TEAL, MAROON)
MIN_REGION_AREA = 4
MAX_GENERATION_ATTEMPTS = 400

_GEN_CALL_COUNT = 0


def _sample_spaced_rows(rng: random.Random, pool: list[int], count: int, min_gap: int) -> list[int] | None:
    rows = pool[:]
    rng.shuffle(rows)
    chosen: list[int] = []
    for row in rows:
        if all(abs(row - prev) >= min_gap for prev in chosen):
            chosen.append(row)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _make_trunk(rng: random.Random, height: int, left_w: int) -> tuple[set[tuple[int, int]], dict[int, int], set[int]]:
    shifts = rng.randint(1, 3)
    shift_rows = _sample_spaced_rows(rng, list(range(1, height - 1)), shifts, 2)
    assert shift_rows is not None

    wall: set[tuple[int, int]] = set()
    trunk_col_by_row: dict[int, int] = {}
    current_col = rng.randint(1, left_w - 2)
    prev_row = 0

    for shift_row in shift_rows:
        for r in range(prev_row, shift_row + 1):
            wall.add((r, current_col))
            trunk_col_by_row[r] = current_col

        choices = [c for c in range(1, left_w - 1) if c != current_col]
        new_col = rng.choice(choices)
        step = 1 if new_col > current_col else -1
        for c in range(current_col, new_col + step, step):
            wall.add((shift_row, c))
        current_col = new_col
        trunk_col_by_row[shift_row] = current_col
        prev_row = shift_row

    for r in range(prev_row, height):
        wall.add((r, current_col))
        trunk_col_by_row[r] = current_col

    return wall, trunk_col_by_row, set(shift_rows)


def _add_branch(wall: set[tuple[int, int]], trunk_col_by_row: dict[int, int], row: int, side: str, left_w: int) -> None:
    anchor = trunk_col_by_row[row]
    if side == "left":
        cols = range(0, anchor + 1)
    else:
        cols = range(anchor, left_w)
    for c in cols:
        wall.add((row, c))


def _build_left_half(
    rng: random.Random, height: int, left_w: int, region_count: int, branch_pattern: tuple[str, ...]
) -> np.ndarray | None:
    wall, trunk_col_by_row, shift_rows = _make_trunk(rng, height, left_w)
    branch_rows = _sample_spaced_rows(
        rng,
        [r for r in range(1, height - 1) if r not in shift_rows],
        len(branch_pattern),
        2,
    )
    if branch_rows is None:
        return None

    for row, side in zip(branch_rows, branch_pattern):
        _add_branch(wall, trunk_col_by_row, row, side, left_w)

    left = np.zeros((height, left_w), dtype=int)
    for r, c in wall:
        left[r, c] = GREEN

    wall_comps = _components(left == GREEN)
    if len(wall_comps) != 1:
        return None

    regions = _components(left == BLACK)
    if len(regions) != region_count:
        return None
    if min(coords.shape[0] for coords in regions) < MIN_REGION_AREA:
        return None

    return left


def _preferred_seed_candidates(coords: np.ndarray, region_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = region_mask.shape
    strong: list[tuple[int, int]] = []
    weak: list[tuple[int, int]] = []

    for r, c in coords.tolist():
        neighbors_in_region = 0
        edge_clear = 0 < r < h - 1 and 0 < c < w - 1
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and region_mask[nr, nc]:
                neighbors_in_region += 1
        if edge_clear and neighbors_in_region == 4:
            strong.append((int(r), int(c)))
        else:
            weak.append((int(r), int(c)))

    random.shuffle(strong)
    random.shuffle(weak)
    return strong + weak


def _pick_seed_cells(rng: random.Random, regions: list[np.ndarray], left_shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, left_w = left_shape
    region_masks = []
    candidate_lists = []
    for coords in regions:
        mask = np.zeros((h, left_w), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        region_masks.append(mask)
        candidate_lists.append(_preferred_seed_candidates(coords, mask))

    order = sorted(range(len(regions)), key=lambda idx: len(candidate_lists[idx]))
    chosen: dict[int, tuple[int, int]] = {}
    mirrored_used: list[tuple[int, int]] = []

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True

        region_idx = order[pos]
        candidates = candidate_lists[region_idx][:]
        rng.shuffle(candidates)
        for r, c in candidates:
            mirrored = (r, left_w - 1 - c)
            if any(max(abs(r - mr), abs(mirrored[1] - mc)) <= 1 for mr, mc in mirrored_used):
                continue
            chosen[region_idx] = (r, c)
            mirrored_used.append(mirrored)
            if dfs(pos + 1):
                return True
            mirrored_used.pop()
            del chosen[region_idx]

        return False

    if not dfs(0):
        return None

    return [chosen[idx] for idx in range(len(regions))]


def _build_expected_output(left: np.ndarray, seeds: list[tuple[int, int]], colors: list[int]) -> np.ndarray:
    labels, regions = _label_components(left == BLACK)
    out = left.copy()
    for color, (r, c) in zip(colors, seeds):
        region_id = int(labels[r, c])
        coords = regions[region_id]
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _branch_mode(call_index: int, rng: random.Random) -> tuple[int, tuple[str, ...]]:
    mode = call_index % 4
    if mode == 0:
        return 3, (rng.choice(("left", "right")),)
    if mode == 1:
        return 4, rng.choice((("left", "right"), ("right", "left")))
    if mode == 2:
        side = rng.choice(("left", "right"))
        return 4, (side, side)

    region_count = rng.choice(REGION_CHOICES)
    if region_count == 3:
        return 3, (rng.choice(("left", "right")),)
    return 4, tuple(rng.choice(("left", "right")) for _ in range(2))


def _validate_scene(grid: np.ndarray) -> None:
    sep = _separator_col(grid)
    left = grid[:, :sep]
    right = grid[:, sep + 1 :]

    assert set(map(int, np.unique(left))).issubset({BLACK, GREEN})
    wall_comps = _components(left == GREEN)
    assert len(wall_comps) == 1

    labels, regions = _label_components(left == BLACK)
    assert 3 <= len(regions) <= 4
    assert min(coords.shape[0] for coords in regions) >= MIN_REGION_AREA

    indicator_coords = np.argwhere((right != BLACK) & (right != GREY) & (right != GREEN))
    assert len(indicator_coords) == len(regions)
    colors = [int(right[r, c]) for r, c in indicator_coords]
    assert len(set(colors)) == len(colors)

    touched_regions: set[int] = set()
    for r, right_local_c in indicator_coords:
        left_c = _mirrored_left_col_from_right_local(int(right_local_c), sep)
        region_id = int(labels[int(r), left_c])
        assert region_id >= 0
        assert region_id not in touched_regions
        touched_regions.add(region_id)

    assert len(touched_regions) == len(regions)

    row_spans = []
    for coords in regions:
        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        row_spans.append((r0, r1))
    overlapping_pair_exists = any(
        max(a0, b0) <= min(a1, b1)
        for i, (a0, a1) in enumerate(row_spans)
        for b0, b1 in row_spans[i + 1 :]
    )
    assert overlapping_pair_exists, "regions should not reduce to a pure top-to-bottom ordering"

    expected = main(grid)
    for r, right_local_c in indicator_coords:
        color = int(right[r, right_local_c])
        left_c = _mirrored_left_col_from_right_local(int(right_local_c), sep)
        region_id = int(labels[int(r), left_c])
        coords = regions[region_id]
        assert np.all(expected[coords[:, 0], coords[:, 1]] == color)
    assert np.array_equal(expected[left == GREEN], left[left == GREEN])


def _gen_attempt(call_index: int) -> np.ndarray | None:
    rng = random.Random()
    height = rng.randint(HEIGHT_MIN, HEIGHT_MAX)
    region_count, branch_pattern = _branch_mode(call_index, rng)

    left = _build_left_half(rng, height, LEFT_W, region_count, branch_pattern)
    if left is None:
        return None

    regions = _components(left == BLACK)
    seeds = _pick_seed_cells(rng, regions, left.shape)
    if seeds is None:
        return None

    colors = rng.sample(list(INDICATOR_COLORS), len(regions))
    rng.shuffle(colors)
    expected = _build_expected_output(left, seeds, colors)

    grid = np.zeros((height, 2 * LEFT_W + 1), dtype=int)
    grid[:, LEFT_W] = GREY
    grid[:, :LEFT_W] = left
    for color, (r, c) in zip(colors, seeds):
        grid[r, _mirrored_right_global_from_left(c, LEFT_W)] = int(color)

    try:
        _validate_scene(grid)
    except AssertionError:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    call_index = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(call_index)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-project-fill ARC input")
