from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
TEAL = 8
MAROON = 9

HALF_HEIGHT_RANGE = (7, 16)
HALF_WIDTH_RANGE = (5, 11)
GREEN_COUNT_RANGE = (2, 6)
MAX_GENERATION_ATTEMPTS = 800


def _reflection_orbit(row: int, col: int, height: int, width: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {
                (row, col),
                (row, width - 1 - col),
                (height - 1 - row, col),
                (height - 1 - row, width - 1 - col),
            }
        )
    )


def _find_separator(grid: np.ndarray) -> int:
    height, width = map(int, grid.shape)
    cols = [col for col in range(width) if np.all(grid[:, col] == BLUE)]
    assert len(cols) == 1, "expected exactly one full-height blue separator"
    col = int(cols[0])
    assert 0 < col < width - 1, "separator must split the grid into two non-empty halves"
    return col


def _repair_half(half: np.ndarray) -> np.ndarray:
    height, width = map(int, half.shape)
    out = np.array(half, dtype=int, copy=True)
    seen: set[tuple[tuple[int, int], ...]] = set()

    for row in range(height):
        for col in range(width):
            orbit = _reflection_orbit(row, col, height, width)
            if orbit in seen:
                continue
            seen.add(orbit)

            visible = {int(half[r, c]) for r, c in orbit if int(half[r, c]) != GREEN}
            assert len(visible) == 1, "each damaged orbit must expose exactly one visible color"
            fill = visible.pop()
            assert fill in (BLACK, MAROON), "half-masks should only contain black, green, and maroon"

            for r, c in orbit:
                if int(out[r, c]) == GREEN:
                    out[r, c] = fill

    return out


def _nor_masks(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.zeros(left.shape, dtype=int)
    out[(left != MAROON) & (right != MAROON)] = TEAL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair two damaged symmetric masks and return their pixelwise logical NOR.

    Input structure:
    - A full-height blue vertical separator splits the grid into equal-sized left and right halves.
    - Each half is a black background with a maroon foreground mask that is symmetric under both
      left-right and top-bottom reflection once green placeholder cells are repaired.
    - Green cells mark damaged pixels whose values must be inferred within their own half from
      the corresponding reflection orbit.

    Rule:
    - Locate the blue separator and slice out the two same-sized halves.
    - Independently repair every green cell in each half by finding its horizontal/vertical
      reflection orbit and copying the unique visible non-green value from that orbit.
    - Treat the repaired halves as aligned binary masks where maroon is active and black is inactive.
    - Output a half-sized grid with teal where both repaired masks are inactive, and black elsewhere.

    Key insight:
    - The green cells are not part of the final masks: they are recoverable missing values, and the
      NOR result is only determined after each half is restored to full two-axis symmetry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    c_bar = _find_separator(g)
    left = g[:, :c_bar]
    right = g[:, c_bar + 1 :]
    assert left.shape == right.shape, "separator must divide the grid into equal halves"

    repaired_left = _repair_half(left)
    repaired_right = _repair_half(right)
    return _nor_masks(repaired_left, repaired_right)


def _paint_rect(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, value: bool) -> None:
    mask[r0 : r1 + 1, c0 : c1 + 1] = value


def _paint_diamond(mask: np.ndarray, center_r: int, center_c: int, radius: int, value: bool) -> None:
    h, w = mask.shape
    rr, cc = np.indices((h, w))
    mask[np.abs(rr - center_r) + np.abs(cc - center_c) <= radius] = value


def _paint_plus(mask: np.ndarray, center_r: int, center_c: int, arm_r: int, arm_c: int, value: bool) -> None:
    r0 = max(0, center_r - arm_r)
    r1 = min(mask.shape[0] - 1, center_r + arm_r)
    c0 = max(0, center_c - arm_c)
    c1 = min(mask.shape[1] - 1, center_c + arm_c)
    mask[r0 : r1 + 1, center_c] = value
    mask[center_r, c0 : c1 + 1] = value


def _reflect_quarter(quarter: np.ndarray, height: int, width: int) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    qh, qw = quarter.shape
    for row in range(qh):
        for col in range(qw):
            value = MAROON if bool(quarter[row, col]) else BLACK
            for r, c in _reflection_orbit(row, col, height, width):
                out[r, c] = value
    return out


def _make_symmetric_mask(rng: np.random.Generator, height: int, width: int) -> np.ndarray | None:
    qh = (height + 1) // 2
    qw = (width + 1) // 2

    for _ in range(40):
        quarter = np.zeros((qh, qw), dtype=bool)
        op_count = int(rng.integers(2, 6))
        for _ in range(op_count):
            mode = str(rng.choice(["rect", "diamond", "plus", "hband", "vband"]))
            if mode == "rect":
                rh = int(rng.integers(1, qh + 1))
                rw = int(rng.integers(1, qw + 1))
                r0 = int(rng.integers(0, qh - rh + 1))
                c0 = int(rng.integers(0, qw - rw + 1))
                _paint_rect(quarter, r0, c0, r0 + rh - 1, c0 + rw - 1, True)
            elif mode == "diamond":
                cr = int(rng.integers(0, qh))
                cc = int(rng.integers(0, qw))
                radius = int(rng.integers(1, max(2, qh + qw - 1)))
                _paint_diamond(quarter, cr, cc, radius, True)
            elif mode == "plus":
                cr = int(rng.integers(0, qh))
                cc = int(rng.integers(0, qw))
                arm_r = int(rng.integers(0, max(1, qh)))
                arm_c = int(rng.integers(0, max(1, qw)))
                _paint_plus(quarter, cr, cc, arm_r, arm_c, True)
            elif mode == "hband":
                r0 = int(rng.integers(0, qh))
                r1 = int(rng.integers(r0, qh))
                c0 = 0 if rng.random() < 0.65 else int(rng.integers(0, qw))
                c1 = qw - 1 if rng.random() < 0.65 else int(rng.integers(c0, qw))
                _paint_rect(quarter, r0, c0, r1 - 1 if r1 > r0 else r0, c1 - 1 if c1 > c0 else c0, True)
            else:
                c0 = int(rng.integers(0, qw))
                c1 = int(rng.integers(c0, qw))
                r0 = 0 if rng.random() < 0.65 else int(rng.integers(0, qh))
                r1 = qh - 1 if rng.random() < 0.65 else int(rng.integers(r0, qh))
                _paint_rect(quarter, r0, c0, r1 - 1 if r1 > r0 else r0, c1 - 1 if c1 > c0 else c0, True)

        if rng.random() < 0.45:
            mode = str(rng.choice(["rect", "diamond"]))
            if mode == "rect":
                rh = int(rng.integers(1, qh + 1))
                rw = int(rng.integers(1, qw + 1))
                r0 = int(rng.integers(0, qh - rh + 1))
                c0 = int(rng.integers(0, qw - rw + 1))
                _paint_rect(quarter, r0, c0, r0 + rh - 1, c0 + rw - 1, False)
            else:
                cr = int(rng.integers(0, qh))
                cc = int(rng.integers(0, qw))
                radius = int(rng.integers(1, max(2, min(qh, qw) + 1)))
                _paint_diamond(quarter, cr, cc, radius, False)

        if rng.random() < 0.2:
            quarter = ~quarter

        full = _reflect_quarter(quarter, height, width)
        fill_ratio = float(np.mean(full == MAROON))
        if 0.18 <= fill_ratio <= 0.82 and np.any(full == MAROON) and np.any(full == BLACK):
            return full
    return None


def _collect_orbits(mask: np.ndarray) -> list[dict[str, object]]:
    height, width = mask.shape
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[dict[str, object]] = []
    for row in range(height):
        for col in range(width):
            orbit = _reflection_orbit(row, col, height, width)
            if orbit in seen:
                continue
            seen.add(orbit)
            value = int(mask[orbit[0]])
            orbits.append({"cells": orbit, "value": value})
    return orbits


def _side_damage_candidates(
    complete_self: np.ndarray, complete_other: np.ndarray
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int], int], list[dict[str, object]]]:
    orbits = _collect_orbits(complete_self)
    orbit_id_by_cell: dict[tuple[int, int], int] = {}
    by_value = {BLACK: [], MAROON: []}
    both_zero: list[tuple[int, int]] = []
    active_against_zero: list[tuple[int, int]] = []

    for orbit_id, orbit in enumerate(orbits):
        cells = orbit["cells"]
        value = int(orbit["value"])
        for cell in cells:
            orbit_id_by_cell[cell] = orbit_id
        if len(cells) <= 1:
            continue
        for row, col in cells:
            coord = (int(row), int(col))
            by_value[value].append(coord)
            other_value = int(complete_other[coord])
            if value == BLACK and other_value == BLACK:
                both_zero.append(coord)
            if value == MAROON and other_value == BLACK:
                active_against_zero.append(coord)

    return (
        by_value[BLACK],
        by_value[MAROON],
        both_zero,
        active_against_zero,
        orbit_id_by_cell,
        orbits,
    )


def _try_add_cell(
    coord: tuple[int, int],
    selected: set[tuple[int, int]],
    selected_per_orbit: dict[int, int],
    orbit_id_by_cell: dict[tuple[int, int], int],
    orbit_sizes: dict[int, int],
) -> bool:
    if coord in selected:
        return False
    orbit_id = orbit_id_by_cell[coord]
    if selected_per_orbit.get(orbit_id, 0) >= orbit_sizes[orbit_id] - 1:
        return False
    selected.add(coord)
    selected_per_orbit[orbit_id] = selected_per_orbit.get(orbit_id, 0) + 1
    return True


def _sample_side_damage(
    rng: np.random.Generator,
    quota: int,
    candidates0: list[tuple[int, int]],
    candidates9: list[tuple[int, int]],
    forced_coords: list[tuple[int, int]],
    orbit_id_by_cell: dict[tuple[int, int], int],
    orbits: list[dict[str, object]],
) -> set[tuple[int, int]] | None:
    orbit_sizes = {idx: len(orbit["cells"]) for idx, orbit in enumerate(orbits)}
    selected: set[tuple[int, int]] = set()
    selected_per_orbit: dict[int, int] = {}

    for coord in forced_coords:
        if not _try_add_cell(coord, selected, selected_per_orbit, orbit_id_by_cell, orbit_sizes):
            return None
    if len(selected) > quota:
        return None

    shuffled0 = list(candidates0)
    shuffled9 = list(candidates9)
    rng.shuffle(shuffled0)
    rng.shuffle(shuffled9)

    if not any(coord in selected for coord in candidates0):
        for coord in shuffled0:
            if _try_add_cell(coord, selected, selected_per_orbit, orbit_id_by_cell, orbit_sizes):
                break
        else:
            return None

    if not any(coord in selected for coord in candidates9):
        for coord in shuffled9:
            if _try_add_cell(coord, selected, selected_per_orbit, orbit_id_by_cell, orbit_sizes):
                break
        else:
            return None

    remaining = [coord for coord in candidates0 + candidates9 if coord not in selected]
    rng.shuffle(remaining)
    for coord in remaining:
        if len(selected) >= quota:
            break
        _try_add_cell(coord, selected, selected_per_orbit, orbit_id_by_cell, orbit_sizes)

    if len(selected) != quota:
        return None
    return selected


def _damage_half(complete_half: np.ndarray, green_cells: set[tuple[int, int]]) -> np.ndarray:
    damaged = np.array(complete_half, copy=True)
    for row, col in green_cells:
        damaged[row, col] = GREEN
    return damaged


def _make_complete_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    height = int(rng.integers(*HALF_HEIGHT_RANGE))
    width = int(rng.integers(*HALF_WIDTH_RANGE))
    left = _make_symmetric_mask(rng, height, width)
    right = _make_symmetric_mask(rng, height, width)
    if left is None or right is None:
        return None
    if np.array_equal(left, right):
        return None
    nor = _nor_masks(left, right)
    if not np.any(nor == TEAL) or not np.any(nor == BLACK):
        return None
    return left, right


def _assemble_input(rng: np.random.Generator, left: np.ndarray, right: np.ndarray) -> np.ndarray | None:
    (
        left0,
        left9,
        left_both0,
        left_active_vs0,
        left_orbit_id,
        left_orbits,
    ) = _side_damage_candidates(left, right)
    (
        right0,
        right9,
        right_both0,
        right_active_vs0,
        right_orbit_id,
        right_orbits,
    ) = _side_damage_candidates(right, left)

    if not left0 or not left9 or not right0 or not right9:
        return None
    if not (left_both0 or right_both0):
        return None
    if not (left_active_vs0 or right_active_vs0):
        return None

    force_both0_side = "L" if left_both0 and (not right_both0 or rng.random() < 0.5) else "R"
    force_active_side = "L" if left_active_vs0 and (not right_active_vs0 or rng.random() < 0.5) else "R"

    forced_left: list[tuple[int, int]] = []
    forced_right: list[tuple[int, int]] = []
    if force_both0_side == "L":
        forced_left.append(left_both0[int(rng.integers(len(left_both0)))])
    else:
        forced_right.append(right_both0[int(rng.integers(len(right_both0)))])
    if force_active_side == "L":
        forced_left.append(left_active_vs0[int(rng.integers(len(left_active_vs0)))])
    else:
        forced_right.append(right_active_vs0[int(rng.integers(len(right_active_vs0)))])

    left_quota = int(rng.integers(*GREEN_COUNT_RANGE))
    right_quota = int(rng.integers(*GREEN_COUNT_RANGE))

    left_selected = _sample_side_damage(
        rng, left_quota, left0, left9, forced_left, left_orbit_id, left_orbits
    )
    right_selected = _sample_side_damage(
        rng, right_quota, right0, right9, forced_right, right_orbit_id, right_orbits
    )
    if left_selected is None or right_selected is None:
        return None

    damaged_left = _damage_half(left, left_selected)
    damaged_right = _damage_half(right, right_selected)
    return np.concatenate(
        [damaged_left, np.full((left.shape[0], 1), BLUE, dtype=int), damaged_right], axis=1
    )


def _naive_nor_with_green(grid: np.ndarray, green_is_active: bool) -> np.ndarray:
    c_bar = _find_separator(grid)
    left = grid[:, :c_bar]
    right = grid[:, c_bar + 1 :]
    left_active = left == MAROON
    right_active = right == MAROON
    if green_is_active:
        left_active |= left == GREEN
        right_active |= right == GREEN
    out = np.zeros(left.shape, dtype=int)
    out[~left_active & ~right_active] = TEAL
    return out


def _semantic_checks(grid: np.ndarray) -> bool:
    c_bar = _find_separator(grid)
    left = grid[:, :c_bar]
    right = grid[:, c_bar + 1 :]
    if left.shape != right.shape:
        return False
    if not (2 <= int(np.count_nonzero(left == GREEN)) <= 5):
        return False
    if not (2 <= int(np.count_nonzero(right == GREEN)) <= 5):
        return False

    repaired_left = _repair_half(left)
    repaired_right = _repair_half(right)
    if np.array_equal(repaired_left, repaired_right):
        return False
    if np.any((repaired_left != BLACK) & (repaired_left != MAROON)):
        return False
    if np.any((repaired_right != BLACK) & (repaired_right != MAROON)):
        return False

    correct = _nor_masks(repaired_left, repaired_right)
    if not np.any(correct == TEAL) or not np.any(correct == BLACK):
        return False

    naive_black = _naive_nor_with_green(grid, green_is_active=False)
    naive_maroon = _naive_nor_with_green(grid, green_is_active=True)
    if np.array_equal(correct, naive_black):
        return False
    if np.array_equal(correct, naive_maroon):
        return False

    if not np.array_equal(main(grid), correct):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        pair = _make_complete_pair(rng)
        if pair is None:
            continue
        left, right = pair
        grid = _assemble_input(rng, left, right)
        if grid is None:
            continue
        if max(grid.shape) > 30:
            continue
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
