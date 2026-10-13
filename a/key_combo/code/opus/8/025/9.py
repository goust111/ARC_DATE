from __future__ import annotations

import math
import random

import numpy as np


class Color:
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


PALETTE = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]

PANEL_SIZE_RANGE = (9, 13)
SHARED_HOLE_RANGE = (5, 8)
PANEL_HOLE_RATE_RANGE = (0.26, 0.36)
MAX_GENERATION_ATTEMPTS = 400


def _separator_column(g: np.ndarray) -> int:
    height, width = map(int, g.shape)
    candidates = [c for c in range(width) if np.all(g[:, c] == Color.GREY)]
    assert len(candidates) == 1, "expected exactly one full-height grey divider column"
    sep = int(candidates[0])
    assert 0 < sep < width - 1, "divider must split two nonempty panels"
    assert sep * 2 + 1 == width, "divider must split equal-width panels"
    return sep


def _antidiagonal_groups(shape: tuple[int, int]) -> dict[int, list[tuple[int, int]]]:
    h, w = map(int, shape)
    groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            groups.setdefault(r + c, []).append((r, c))
    return groups


def _merge_panels(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    overlap = (left != Color.BLACK) & (right != Color.BLACK)
    assert np.all(left[overlap] == right[overlap]), "panel overlaps must agree"
    return np.where(left != Color.BLACK, left, right)


def _fill_antidiagonals(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    diag_groups = _antidiagonal_groups(out.shape)
    diag_color: dict[int, int] = {}
    for total, coords in diag_groups.items():
        colors = {int(out[r, c]) for r, c in coords if int(out[r, c]) != Color.BLACK}
        if colors:
            assert len(colors) == 1, f"anti-diagonal {total} is inconsistent"
            diag_color[total] = colors.pop()

    for r, c in np.argwhere(out == Color.BLACK):
        total = int(r + c)
        assert total in diag_color, f"anti-diagonal {total} has no visible color"
        out[r, c] = diag_color[total]

    return out


def _solve_spec(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep = _separator_column(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape, "panels must have equal shape"
    assert not np.any(left == Color.GREY), "grey must be reserved for the divider"
    assert not np.any(right == Color.GREY), "grey must be reserved for the divider"

    merged = _merge_panels(left, right)
    return _fill_antidiagonals(merged)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge complementary side panels, then repair the remaining holes from anti-diagonal color constancy.

    Input structure:
    - A single full-height grey divider column splits the grid into two equal panels.
    - The left and right panels are aligned views of the same square color pattern, with black cells marking missing pixels.
    - In the complete pattern, every anti-diagonal (cells with the same row+col sum) is monochromatic.

    Rule:
    - Remove the grey divider and align the two panels cell by cell.
    - First perform a transparent union merge: keep the left color when present, otherwise use the right color.
    - If any merged cells are still black, fill each one with the unique visible non-black color already present on the same anti-diagonal.

    Key insight:
    - The two panels recover most missing pixels by complementing each other, but a few doubly missing cells can only be inferred after the merge reveals the anti-diagonal stripe structure clearly enough.
    """
    return _solve_spec(grid)


def _random_run_lengths(total: int, run_count: int) -> list[int]:
    lengths = [1] * run_count
    remaining = total - run_count
    while remaining > 0:
        idx = random.randrange(run_count)
        if lengths[idx] < 4:
            lengths[idx] += 1
            remaining -= 1
    return lengths


def _random_diag_colors(size: int, palette: list[int]) -> list[int]:
    total = 2 * size - 1
    min_runs = max(len(palette) + 2, math.ceil(total / 4))
    max_runs = min(total, max(min_runs, len(palette) + 6))
    run_count = random.randint(min_runs, max_runs)
    run_lengths = _random_run_lengths(total, run_count)

    run_colors = random.sample(palette, len(palette))
    while len(run_colors) < run_count:
        prev = run_colors[-1]
        choices = [color for color in palette if color != prev]
        run_colors.append(random.choice(choices))

    diag_colors: list[int] = []
    for color, length in zip(run_colors, run_lengths, strict=False):
        diag_colors.extend([color] * length)

    assert len(diag_colors) == total
    assert len(set(diag_colors)) == len(palette)
    return diag_colors


def _build_complete_panel(size: int) -> np.ndarray:
    palette_size = random.randint(3, 4)
    palette = random.sample(PALETTE, palette_size)
    diag_colors = _random_diag_colors(size, palette)
    panel = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            panel[r, c] = diag_colors[r + c]
    return panel


def _coord_list_to_index(coords: set[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    items = np.array(sorted(coords), dtype=int)
    return items[:, 0], items[:, 1]


def _sample_without(coords: list[tuple[int, int]], banned: set[tuple[int, int]], count: int) -> set[tuple[int, int]] | None:
    choices = [coord for coord in coords if coord not in banned]
    if len(choices) < count:
        return None
    return set(random.sample(choices, count))


def _make_hole_sets(size: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]], int, int] | None:
    diag_groups = _antidiagonal_groups((size, size))
    eligible_diags = [total for total, coords in diag_groups.items() if len(coords) >= 3]
    if len(eligible_diags) < 4:
        return None

    left_blackout, right_blackout = random.sample(eligible_diags, 2)
    shared_count = random.randint(*SHARED_HOLE_RANGE)
    shared_diag_count = min(random.randint(4, 6), len(eligible_diags), shared_count)
    chosen_diags = {left_blackout, right_blackout}
    remaining_diags = [d for d in eligible_diags if d not in chosen_diags]
    if len(chosen_diags) < shared_diag_count:
        chosen_diags.update(random.sample(remaining_diags, shared_diag_count - len(chosen_diags)))

    shared: set[tuple[int, int]] = set()
    shared_per_diag = {total: 0 for total in diag_groups}
    for total in chosen_diags:
        coord = random.choice(diag_groups[total])
        shared.add(coord)
        shared_per_diag[total] += 1

    all_coords = [(r, c) for r in range(size) for c in range(size)]
    extra_candidates = all_coords.copy()
    random.shuffle(extra_candidates)
    for coord in extra_candidates:
        if len(shared) >= shared_count:
            break
        total = coord[0] + coord[1]
        if coord in shared:
            continue
        if shared_per_diag[total] >= len(diag_groups[total]) - 1:
            continue
        shared.add(coord)
        shared_per_diag[total] += 1

    if len(shared) != shared_count:
        return None

    left_holes = set(shared)
    right_holes = set(shared)
    left_holes.update(diag_groups[left_blackout])
    right_holes.update(diag_groups[right_blackout])

    area = size * size
    low = math.ceil(PANEL_HOLE_RATE_RANGE[0] * area)
    high = math.floor(PANEL_HOLE_RATE_RANGE[1] * area)
    left_target_low = max(low, len(left_holes))
    right_target_low = max(low, len(right_holes))
    if left_target_low > high or right_target_low > high:
        return None

    left_target = random.randint(left_target_low, high)
    right_target = random.randint(right_target_low, high)

    left_extra = _sample_without(all_coords, left_holes | right_holes, left_target - len(left_holes))
    if left_extra is None:
        return None
    left_holes.update(left_extra)

    right_extra = _sample_without(all_coords, left_holes | right_holes, right_target - len(right_holes))
    if right_extra is None:
        return None
    right_holes.update(right_extra)

    assert left_holes & right_holes == shared
    return left_holes, right_holes, left_blackout, right_blackout


def _render_input(full: np.ndarray, left_holes: set[tuple[int, int]], right_holes: set[tuple[int, int]]) -> np.ndarray:
    left = full.copy()
    right = full.copy()
    if left_holes:
        rr, cc = _coord_list_to_index(left_holes)
        left[rr, cc] = Color.BLACK
    if right_holes:
        rr, cc = _coord_list_to_index(right_holes)
        right[rr, cc] = Color.BLACK

    size = full.shape[0]
    grid = np.full((size, 2 * size + 1), Color.BLACK, dtype=int)
    grid[:, :size] = left
    grid[:, size] = Color.GREY
    grid[:, size + 1 :] = right
    return grid


def _single_panel_visible_on_diag(panel: np.ndarray, total: int) -> int:
    count = 0
    h, w = panel.shape
    for r in range(h):
        c = total - r
        if 0 <= c < w and int(panel[r, c]) != Color.BLACK:
            count += 1
    return count


def _validate_instance(
    inp: np.ndarray,
    full: np.ndarray,
    left_holes: set[tuple[int, int]],
    right_holes: set[tuple[int, int]],
    left_blackout: int,
    right_blackout: int,
) -> bool:
    try:
        out = _solve_spec(inp)
    except Exception:
        return False
    if not np.array_equal(out, full):
        return False

    sep = _separator_column(inp)
    left = inp[:, :sep]
    right = inp[:, sep + 1 :]
    merged = _merge_panels(left, right)
    shared = left_holes & right_holes

    if np.count_nonzero(merged == Color.BLACK) != len(shared):
        return False
    if len(shared) < SHARED_HOLE_RANGE[0]:
        return False
    if len({r + c for r, c in shared}) < 4:
        return False

    diag_groups = _antidiagonal_groups(full.shape)
    for total, coords in diag_groups.items():
        visible = [merged[r, c] for r, c in coords if int(merged[r, c]) != Color.BLACK]
        if not visible:
            return False
        if len(set(map(int, visible))) != 1:
            return False

    left_visible = _single_panel_visible_on_diag(left, left_blackout)
    right_visible = _single_panel_visible_on_diag(right, right_blackout)
    if left_visible != 0 or right_visible != 0:
        return False
    if not any((r + c) == left_blackout for r, c in shared):
        return False
    if not any((r + c) == right_blackout for r, c in shared):
        return False

    if len(left_holes) - len(shared) < 8 or len(right_holes) - len(shared) < 8:
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    size = random.randint(*PANEL_SIZE_RANGE)
    full = _build_complete_panel(size)
    hole_info = _make_hole_sets(size)
    if hole_info is None:
        return None
    left_holes, right_holes, left_blackout, right_blackout = hole_info
    inp = _render_input(full, left_holes, right_holes)
    if not _validate_instance(inp, full, left_holes, right_holes, left_blackout, right_blackout):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-panel anti-diagonal puzzle")
