from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
RED = 2
YELLOW = 4
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = (1, 3, 5, 6, 7, 8, 9)

# --------------- solver ---------------


def _find_axis(grid: np.ndarray) -> tuple[str, int]:
    rows = [int(r) for r in range(grid.shape[0]) if bool(np.all(grid[int(r), :] == YELLOW))]
    cols = [int(c) for c in range(grid.shape[1]) if bool(np.all(grid[:, int(c)] == YELLOW))]
    assert (not rows) != (not cols), "expected exactly one yellow separator axis"
    if rows:
        assert len(rows) == 1, "expected a single horizontal separator row"
        return "horizontal", rows[0]
    assert len(cols) == 1, "expected a single vertical separator column"
    return "vertical", cols[0]


def _color_counts(half: np.ndarray) -> Counter[int]:
    counts: Counter[int] = Counter()
    for value in half.ravel():
        color = int(value)
        if color not in (BG, YELLOW):
            counts[color] += 1
    return counts


def _same_color_support(half: np.ndarray, r: int, c: int, color: int) -> int:
    support = 0
    h, w = half.shape
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and int(half[nr, nc]) == color:
            support += 1
    return support


def _neighborhood_count(half: np.ndarray, r: int, c: int, color: int) -> int:
    total = 0
    h, w = half.shape
    for nr in range(max(0, r - 1), min(h, r + 2)):
        for nc in range(max(0, c - 1), min(w, c + 2)):
            if nr == r and nc == c:
                continue
            if int(half[nr, nc]) == color:
                total += 1
    return total


def _choose_defect_side(
    first: np.ndarray,
    second_aligned: np.ndarray,
    counts_first: Counter[int],
    counts_second: Counter[int],
    r: int,
    c: int,
) -> int:
    first_color = int(first[r, c])
    second_color = int(second_aligned[r, c])
    first_count = counts_first.get(first_color, 0)
    second_count = counts_second.get(second_color, 0)
    if first_count < second_count:
        return 0
    if second_count < first_count:
        return 1

    first_support = _same_color_support(first, r, c, first_color)
    second_support = _same_color_support(second_aligned, r, c, second_color)
    if first_support < second_support:
        return 0
    if second_support < first_support:
        return 1

    first_context = _neighborhood_count(first, r, c, second_color)
    second_context = _neighborhood_count(second_aligned, r, c, first_color)
    if first_context > second_context:
        return 0
    if second_context > first_context:
        return 1

    raise AssertionError("ambiguous mismatch; generator should avoid ties")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair mirrored halves by frequency analysis, then mark repaired defects.

    Input structure:
    - Black background with one full yellow separator row or column.
    - The two sides should be mirror images of one another.
    - A few pixels on each side were replaced by rare colors.

    Rule:
    - Align the two halves by reflection and find every mismatched pair.
    - The side whose mismatching color is rarer within its own half is the
      defective side.
    - Replace the defective pixel with the mirrored color from the opposite
      side.
    - Around each repaired pixel, paint red on orthogonally adjacent
      background cells only.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    axis_kind, axis_index = _find_axis(g)
    out = g.copy()
    repairs: list[tuple[int, int]] = []

    if axis_kind == "vertical":
        left = g[:, :axis_index]
        right = g[:, axis_index + 1 :]
        assert left.shape == right.shape, "separator must split the grid into equal halves"
        right_aligned = np.fliplr(right)

        counts_left = _color_counts(left)
        counts_right = _color_counts(right_aligned)
        for r_raw, c_raw in np.argwhere(left != right_aligned):
            r = int(r_raw)
            c = int(c_raw)
            side = _choose_defect_side(left, right_aligned, counts_left, counts_right, r, c)
            if side == 0:
                out[r, c] = int(right_aligned[r, c])
                repairs.append((r, c))
            else:
                full_c = g.shape[1] - 1 - c
                out[r, full_c] = int(left[r, c])
                repairs.append((r, full_c))
    else:
        top = g[:axis_index, :]
        bottom = g[axis_index + 1 :, :]
        assert top.shape == bottom.shape, "separator must split the grid into equal halves"
        bottom_aligned = np.flipud(bottom)

        counts_top = _color_counts(top)
        counts_bottom = _color_counts(bottom_aligned)
        for r_raw, c_raw in np.argwhere(top != bottom_aligned):
            r = int(r_raw)
            c = int(c_raw)
            side = _choose_defect_side(top, bottom_aligned, counts_top, counts_bottom, r, c)
            if side == 0:
                out[r, c] = int(bottom_aligned[r, c])
                repairs.append((r, c))
            else:
                full_r = g.shape[0] - 1 - r
                out[full_r, c] = int(top[r, c])
                repairs.append((full_r, c))

    h, w = out.shape
    for r, c in repairs:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and int(out[nr, nc]) == BG:
                out[nr, nc] = RED

    return out


# --------------- generator ---------------

GEN_RETRIES = 400
HALF_HEIGHT_RANGE = (9, 15)
HALF_WIDTH_RANGE = (8, 13)
BASE_COLOR_COUNT_CHOICES = (2, 3)
MIN_BASE_COLOR_CELLS = 7
PER_COLOR_TOTAL_RANGE = (7, 10)
MAX_HALF_FILL_RATIO = 0.42
MIN_DEFECT_CANDIDATES = 8
BLOB_MIN_SIZE = 3
BLOB_COUNT_RANGE = (1, 2)
HALF_BUILD_RETRIES = 160
BLOB_GROW_RETRIES = 120


def _partition_total(total: int, parts: int, min_each: int) -> list[int]:
    sizes: list[int] = []
    remaining = total
    for idx in range(parts):
        left = parts - idx - 1
        cur_lo = max(min_each, remaining - left * (total - min_each * idx))
        cur_hi = remaining - left * min_each
        size = random.randint(cur_lo, cur_hi)
        sizes.append(size)
        remaining -= size
    random.shuffle(sizes)
    return sizes


def _free_neighbors(r: int, c: int, blocked: np.ndarray, chosen: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = blocked.shape
    neighbors: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and not bool(blocked[nr, nc]) and (nr, nc) not in chosen:
            neighbors.append((nr, nc))
    return neighbors


def _grow_blob(h: int, w: int, blocked: np.ndarray, size: int) -> set[tuple[int, int]] | None:
    free = [(int(r), int(c)) for r, c in np.argwhere(~blocked)]
    if not free:
        return None

    for _ in range(BLOB_GROW_RETRIES):
        seed = random.choice(free)
        chosen = {seed}
        frontier = set(_free_neighbors(seed[0], seed[1], blocked, chosen))
        while len(chosen) < size and frontier:
            cell = random.choice(tuple(frontier))
            frontier.discard(cell)
            chosen.add(cell)
            frontier.update(_free_neighbors(cell[0], cell[1], blocked, chosen))
        if len(chosen) == size:
            return chosen
    return None


def _reserve_with_margin(blocked: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        blocked[r, c] = True
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _candidate_defect_positions(half: np.ndarray) -> list[tuple[int, int]]:
    h, w = half.shape
    candidates: list[tuple[int, int]] = []
    for r_raw, c_raw in np.argwhere(half != BG):
        r = int(r_raw)
        c = int(c_raw)
        if any(
            0 <= r + dr < h
            and 0 <= c + dc < w
            and int(half[r + dr, c + dc]) == BG
            for dr, dc in DIRS4
        ):
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _choose_spaced_positions(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                return chosen
    return None


def _build_base_half(height: int, width: int, colors: list[int]) -> np.ndarray | None:
    for _ in range(HALF_BUILD_RETRIES):
        half = np.full((height, width), BG, dtype=int)
        blocked = np.zeros((height, width), dtype=bool)
        ok = True

        for color in colors:
            total = random.randint(*PER_COLOR_TOTAL_RANGE)
            max_blobs = min(BLOB_COUNT_RANGE[1], total // BLOB_MIN_SIZE)
            n_blobs = random.randint(BLOB_COUNT_RANGE[0], max_blobs)
            for size in _partition_total(total, n_blobs, BLOB_MIN_SIZE):
                cells = _grow_blob(height, width, blocked, size)
                if cells is None:
                    ok = False
                    break
                for r, c in cells:
                    half[r, c] = color
                _reserve_with_margin(blocked, cells)
            if not ok:
                break

        if not ok:
            continue

        counts = _color_counts(half)
        if set(counts) != set(colors):
            continue
        if any(counts[color] < MIN_BASE_COLOR_CELLS for color in colors):
            continue
        if int(np.count_nonzero(half != BG)) > int(height * width * MAX_HALF_FILL_RATIO):
            continue
        if len(_candidate_defect_positions(half)) < MIN_DEFECT_CANDIDATES:
            continue
        return half
    return None


def _defect_groups(defect_count: int, base_color_count: int) -> list[int]:
    if defect_count == 1:
        return [1]
    if defect_count == 2:
        if base_color_count == 2 and random.random() < 0.5:
            return [1, 1]
        return [2]
    assert defect_count == 3 and base_color_count == 2
    return list(random.choice(((2, 1), (1, 2))))


def _defect_plan(defect_count: int, base_colors: list[int]) -> list[int]:
    groups = _defect_groups(defect_count, len(base_colors))
    extras = [color for color in PALETTE if color not in base_colors]
    chosen_colors = random.sample(extras, len(groups))
    plan: list[int] = []
    for color, size in zip(chosen_colors, groups, strict=False):
        plan.extend([color] * size)
    random.shuffle(plan)
    return plan


def _assemble_grid(kind: str, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if kind == "vertical":
        h, w = first.shape
        grid = np.full((h, 2 * w + 1), BG, dtype=int)
        grid[:, :w] = first
        grid[:, w] = YELLOW
        grid[:, w + 1 :] = second
        return grid

    h, w = first.shape
    grid = np.full((2 * h + 1, w), BG, dtype=int)
    grid[:h, :] = first
    grid[h, :] = YELLOW
    grid[h + 1 :, :] = second
    return grid


def _full_position(kind: str, half_shape: tuple[int, int], side: int, pos: tuple[int, int]) -> tuple[int, int]:
    r, c = pos
    half_h, half_w = half_shape
    if kind == "vertical":
        return (r, c) if side == 0 else (r, 2 * half_w - c)
    return (r, c) if side == 0 else (2 * half_h - r, c)


def _validate_case(
    grid: np.ndarray,
    expected: np.ndarray,
    kind: str,
    canonical_first_defects: set[tuple[int, int]],
    canonical_second_defects: set[tuple[int, int]],
) -> bool:
    if kind == "vertical":
        axis = grid.shape[1] // 2
        first = grid[:, :axis]
        second_aligned = np.fliplr(grid[:, axis + 1 :])
    else:
        axis = grid.shape[0] // 2
        first = grid[:axis, :]
        second_aligned = np.flipud(grid[axis + 1 :, :])

    if first.shape != second_aligned.shape:
        return False

    for half in (first, second_aligned):
        colors = {int(v) for v in half.ravel() if int(v) != BG}
        if not (2 <= len(colors) <= 4):
            return False

    mismatches = {tuple(map(int, pt)) for pt in np.argwhere(first != second_aligned)}
    if mismatches != canonical_first_defects | canonical_second_defects:
        return False

    counts_first = _color_counts(first)
    counts_second = _color_counts(second_aligned)
    for r, c in mismatches:
        left_count = counts_first.get(int(first[r, c]), 0)
        right_count = counts_second.get(int(second_aligned[r, c]), 0)
        if (r, c) in canonical_first_defects:
            if not (left_count < right_count):
                return False
        else:
            if not (right_count < left_count):
                return False

    return bool(np.array_equal(main(grid), expected))


def _gen_attempt() -> np.ndarray | None:
    kind = random.choice(("vertical", "horizontal"))
    if kind == "vertical":
        half_h = random.randint(*HALF_HEIGHT_RANGE)
        half_w = random.randint(*HALF_WIDTH_RANGE)
    else:
        half_h = random.randint(*HALF_WIDTH_RANGE)
        half_w = random.randint(*HALF_HEIGHT_RANGE)

    base_color_count = random.choice(BASE_COLOR_COUNT_CHOICES)
    base_colors = random.sample(PALETTE, base_color_count)
    base_half = _build_base_half(half_h, half_w, base_colors)
    if base_half is None:
        return None

    max_defects = 2 if base_color_count == 3 else 3
    first_count = random.randint(1, max_defects)
    second_count = random.randint(1, max_defects)

    candidates = _candidate_defect_positions(base_half)
    chosen = _choose_spaced_positions(candidates, first_count + second_count)
    if chosen is None:
        return None
    random.shuffle(chosen)
    first_positions = chosen[:first_count]
    second_positions = chosen[first_count:]

    first_plan = _defect_plan(first_count, base_colors)
    second_plan = _defect_plan(second_count, base_colors)

    if kind == "vertical":
        first_half = base_half.copy()
        second_half = np.fliplr(base_half).copy()
        for (r, c), color in zip(first_positions, first_plan, strict=False):
            first_half[r, c] = color
        for (r, c), color in zip(second_positions, second_plan, strict=False):
            second_half[r, half_w - 1 - c] = color
    else:
        first_half = base_half.copy()
        second_half = np.flipud(base_half).copy()
        for (r, c), color in zip(first_positions, first_plan, strict=False):
            first_half[r, c] = color
        for (r, c), color in zip(second_positions, second_plan, strict=False):
            second_half[half_h - 1 - r, c] = color

    grid = _assemble_grid(kind, first_half, second_half)
    expected = grid.copy()
    repairs: list[tuple[int, int]] = []

    for pos in first_positions:
        full_pos = _full_position(kind, base_half.shape, 0, pos)
        expected[full_pos] = int(base_half[pos])
        repairs.append(full_pos)
    for pos in second_positions:
        full_pos = _full_position(kind, base_half.shape, 1, pos)
        expected[full_pos] = int(base_half[pos])
        repairs.append(full_pos)

    h, w = expected.shape
    for r, c in repairs:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and int(expected[nr, nc]) == BG:
                expected[nr, nc] = RED

    if not _validate_case(grid, expected, kind, set(first_positions), set(second_positions)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
