from __future__ import annotations

import random
from typing import Sequence

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GRAY = 5

MAX_GENERATION_ATTEMPTS = 500
GRID_RANGE = (14, 19)
STEP_CHOICES = (3, 4)
GRAY_COUNT_CHOICES = (2, 3)
RED_COUNT_CHOICES = (1, 2)
MIN_MANHATTAN = 4


def _full_line_rows(grid: np.ndarray) -> list[int]:
    return [
        r
        for r in range(grid.shape[0])
        if np.all(np.isin(grid[r, :], (BLUE, GRAY)))
    ]


def _full_line_cols(grid: np.ndarray) -> list[int]:
    return [
        c
        for c in range(grid.shape[1])
        if np.all(np.isin(grid[:, c], (BLUE, GRAY)))
    ]


def _regular_step(indices: Sequence[int]) -> int:
    assert len(indices) >= 2, "expected at least two lattice lines"
    diffs = {b - a for a, b in zip(indices, indices[1:])}
    assert len(diffs) == 1, "lattice lines must be equally spaced"
    step = diffs.pop()
    assert step in STEP_CHOICES, "lattice spacing must be 3 or 4"
    return step


def _lattice_mask(shape: tuple[int, int], row_lines: Sequence[int], col_lines: Sequence[int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[list(row_lines), :] = True
    mask[:, list(col_lines)] = True
    return mask


def _window(shape: tuple[int, int], r: int, c: int) -> tuple[slice, slice]:
    h, w = shape
    return slice(max(0, r - 1), min(h, r + 2)), slice(max(0, c - 1), min(w, c + 2))


def _has_mixed_membership(mask: np.ndarray, r: int, c: int) -> bool:
    rs, cs = _window(mask.shape, r, c)
    block = mask[rs, cs]
    return bool(np.any(block) and np.any(~block))


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _halos_disjoint(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) >= 3 or abs(a[1] - b[1]) >= 3


def _well_separated(pos: tuple[int, int], others: Sequence[tuple[int, int]]) -> bool:
    return all(_manhattan(pos, other) >= MIN_MANHATTAN and _halos_disjoint(pos, other) for other in others)


def _gray_kind(r: int, c: int, row_set: set[int], col_set: set[int]) -> str:
    return "intersection" if r in row_set and c in col_set else "line"


def _red_kind(r: int, c: int, row_lines: Sequence[int], col_lines: Sequence[int]) -> str:
    near_row = any(abs(r - rr) == 1 for rr in row_lines)
    near_col = any(abs(c - cc) == 1 for cc in col_lines)
    if near_row and near_col:
        return "corner"
    if near_row or near_col:
        return "single"
    return "deep"


def _expected_output(
    grid: np.ndarray, row_lines: Sequence[int], col_lines: Sequence[int]
) -> tuple[np.ndarray, list[set[tuple[int, int]]], list[set[tuple[int, int]]]]:
    lattice = _lattice_mask(grid.shape, row_lines, col_lines)
    out = grid.copy()
    gray_effects: list[set[tuple[int, int]]] = []
    red_effects: list[set[tuple[int, int]]] = []

    for r, c in np.argwhere(grid == GRAY):
        rs, cs = _window(grid.shape, int(r), int(c))
        local = lattice[rs, cs]
        rows = range(rs.start, rs.stop)
        cols = range(cs.start, cs.stop)
        effect = {
            (rr, cc)
            for rr in rows
            for cc in cols
            if local[rr - rs.start, cc - cs.start] and grid[rr, cc] == BLUE
        }
        block = out[rs, cs].copy()
        block[local] = GRAY
        out[rs, cs] = block
        gray_effects.append(effect)

    for r, c in np.argwhere(grid == RED):
        rs, cs = _window(grid.shape, int(r), int(c))
        local_fill = (~lattice[rs, cs]) & (grid[rs, cs] == BLACK)
        rows = range(rs.start, rs.stop)
        cols = range(cs.start, cs.stop)
        effect = {
            (rr, cc)
            for rr in rows
            for cc in cols
            if local_fill[rr - rs.start, cc - cs.start]
        }
        block = out[rs, cs].copy()
        block[local_fill] = RED
        out[rs, cs] = block
        red_effects.append(effect)

    out[grid == GRAY] = GRAY
    out[grid == RED] = RED
    return out, gray_effects, red_effects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each marker through only the part of its 3x3 neighborhood that matches its lattice membership.

    Input structure:
    - The background is black.
    - Blue pixels form a regular one-pixel lattice made from full horizontal and vertical lines.
    - Gray single-pixel markers lie on lattice cells, while red single-pixel markers lie strictly off the lattice.

    Rule:
    - Infer the lattice from the full blue-or-gray rows and columns.
    - For each gray marker, recolor only the lattice-member cells in its clipped 3x3 neighborhood to gray.
    - For each red marker, recolor only the non-lattice black cells in its clipped 3x3 neighborhood to red, leaving lattice cells untouched.

    Key insight:
    - The same local 3x3 expansion is filtered by an unseen membership mask: gray markers spread along the lattice, red markers spread only through the lattice gaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED, GRAY}

    row_lines = _full_line_rows(g)
    col_lines = _full_line_cols(g)
    row_step = _regular_step(row_lines)
    col_step = _regular_step(col_lines)
    assert row_step in STEP_CHOICES and col_step in STEP_CHOICES

    lattice = _lattice_mask(g.shape, row_lines, col_lines)
    gray_markers = [tuple(map(int, rc)) for rc in np.argwhere(g == GRAY)]
    red_markers = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    assert 2 <= len(gray_markers) <= 3
    assert 1 <= len(red_markers) <= 2

    for marker in gray_markers:
        assert lattice[marker], "gray marker must be on the lattice"
        assert _has_mixed_membership(lattice, *marker), "gray marker neighborhood must mix lattice and non-lattice cells"
    for marker in red_markers:
        assert not lattice[marker], "red marker must be off the lattice"
        assert _has_mixed_membership(lattice, *marker), "red marker neighborhood must mix lattice and non-lattice cells"

    markers = gray_markers + red_markers
    for i, a in enumerate(markers):
        for b in markers[i + 1 :]:
            assert _manhattan(a, b) >= MIN_MANHATTAN

    out = g.copy()
    for r, c in gray_markers:
        rs, cs = _window(g.shape, r, c)
        local_lattice = lattice[rs, cs]
        block = out[rs, cs].copy()
        block[local_lattice] = GRAY
        out[rs, cs] = block

    for r, c in red_markers:
        rs, cs = _window(g.shape, r, c)
        local_fill = (~lattice[rs, cs]) & (g[rs, cs] == BLACK)
        block = out[rs, cs].copy()
        block[local_fill] = RED
        out[rs, cs] = block

    out[g == GRAY] = GRAY
    out[g == RED] = RED
    return out


def _line_offsets(length: int, step: int) -> list[int]:
    return [offset for offset in range(step) if len(range(offset, length, step)) >= 3]


def _collect_candidates(
    shape: tuple[int, int], row_lines: Sequence[int], col_lines: Sequence[int]
) -> tuple[list[dict], list[dict]]:
    h, w = shape
    row_set = set(row_lines)
    col_set = set(col_lines)
    lattice = _lattice_mask(shape, row_lines, col_lines)
    gray_candidates: list[dict] = []
    red_candidates: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not _has_mixed_membership(lattice, r, c):
                continue
            if lattice[r, c]:
                gray_candidates.append(
                    {"pos": (r, c), "kind": _gray_kind(r, c, row_set, col_set)}
                )
            else:
                kind = _red_kind(r, c, row_lines, col_lines)
                if kind != "deep":
                    red_candidates.append({"pos": (r, c), "kind": kind})

    return gray_candidates, red_candidates


def _pick_candidates(
    candidates: list[dict],
    count: int,
    occupied: Sequence[tuple[int, int]],
    min_kind_counts: dict[str, int],
) -> list[dict] | None:
    pool = candidates[:]
    random.shuffle(pool)

    def rec(index: int, chosen: list[dict], counts: dict[str, int]) -> list[dict] | None:
        if len(chosen) == count:
            return chosen[:] if all(counts.get(kind, 0) >= need for kind, need in min_kind_counts.items()) else None
        if index >= len(pool):
            return None
        remaining_slots = count - len(chosen)
        if len(pool) - index < remaining_slots:
            return None

        possible: dict[str, int] = {}
        for cand in pool[index:]:
            possible[cand["kind"]] = possible.get(cand["kind"], 0) + 1
        for kind, need in min_kind_counts.items():
            if counts.get(kind, 0) + possible.get(kind, 0) < need:
                return None

        for j in range(index, len(pool)):
            cand = pool[j]
            pos = cand["pos"]
            placed = [item["pos"] for item in chosen]
            if not _well_separated(pos, [*occupied, *placed]):
                continue
            chosen.append(cand)
            counts[cand["kind"]] = counts.get(cand["kind"], 0) + 1
            result = rec(j + 1, chosen, counts)
            if result is not None:
                return result
            counts[cand["kind"]] -= 1
            if counts[cand["kind"]] == 0:
                del counts[cand["kind"]]
            chosen.pop()
        return None

    return rec(0, [], {})


def _gray_requirements(gray_count: int, gray_candidates: list[dict]) -> dict[str, int]:
    line_avail = sum(c["kind"] == "line" for c in gray_candidates)
    inter_avail = sum(c["kind"] == "intersection" for c in gray_candidates)
    mode = random.random()
    if gray_count >= 2 and line_avail >= 1 and inter_avail >= 1 and mode < 0.60:
        return {"line": 1, "intersection": 1}
    if line_avail >= min(gray_count, 2) and mode < 0.85:
        return {"line": 1}
    if inter_avail >= 1:
        return {"intersection": 1}
    return {}


def _red_requirements(red_count: int, red_candidates: list[dict]) -> dict[str, int]:
    single_avail = sum(c["kind"] == "single" for c in red_candidates)
    corner_avail = sum(c["kind"] == "corner" for c in red_candidates)
    mode = random.random()
    if red_count == 2 and single_avail >= 1 and corner_avail >= 1 and mode < 0.70:
        return {"single": 1, "corner": 1}
    if corner_avail >= 1 and mode < 0.50:
        return {"corner": 1}
    if single_avail >= 1:
        return {"single": 1}
    return {}


def _validate_generated(
    grid: np.ndarray, row_lines: Sequence[int], col_lines: Sequence[int]
) -> bool:
    try:
        expected, gray_effects, red_effects = _expected_output(grid, row_lines, col_lines)
        actual = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False

    gray_markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == GRAY)]
    red_markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    if not gray_markers or not red_markers:
        return False

    if any(len(effect) == 0 for effect in gray_effects):
        return False
    if any(len(effect) == 0 for effect in red_effects):
        return False

    lattice = _lattice_mask(grid.shape, row_lines, col_lines)
    for marker in gray_markers + red_markers:
        if not _has_mixed_membership(lattice, *marker):
            return False

    all_markers = gray_markers + red_markers
    for i, a in enumerate(all_markers):
        for b in all_markers[i + 1 :]:
            if _manhattan(a, b) < MIN_MANHATTAN or not _halos_disjoint(a, b):
                return False

    if not np.any((grid == BLUE) & (actual == BLUE)):
        return False
    if not np.any((grid == BLACK) & (actual == BLACK)):
        return False
    if not np.any((grid == BLUE) & (actual == GRAY)):
        return False
    if not np.any((grid == BLACK) & (actual == RED)):
        return False

    gray_sets = list(gray_effects)
    red_sets = list(red_effects)
    for i, effect in enumerate(gray_effects):
        others = set().union(*gray_sets[:i], *gray_sets[i + 1 :], *red_sets)
        if effect & others:
            return False
    for i, effect in enumerate(red_effects):
        others = set().union(*red_sets[:i], *red_sets[i + 1 :], *gray_sets)
        if effect & others:
            return False

    gray_kinds = {_gray_kind(r, c, set(row_lines), set(col_lines)) for r, c in gray_markers}
    red_kinds = {_red_kind(r, c, row_lines, col_lines) for r, c in red_markers}
    if not gray_kinds <= {"intersection", "line"}:
        return False
    if not red_kinds <= {"single", "corner"}:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_RANGE)
    width = random.randint(*GRID_RANGE)
    row_step = random.choice(STEP_CHOICES)
    col_step = random.choice(STEP_CHOICES)

    row_offsets = _line_offsets(height, row_step)
    col_offsets = _line_offsets(width, col_step)
    if not row_offsets or not col_offsets:
        return None

    row_lines = list(range(random.choice(row_offsets), height, row_step))
    col_lines = list(range(random.choice(col_offsets), width, col_step))
    if len(row_lines) < 3 or len(col_lines) < 3:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    grid[row_lines, :] = BLUE
    grid[:, col_lines] = BLUE

    gray_candidates, red_candidates = _collect_candidates(grid.shape, row_lines, col_lines)
    if len(gray_candidates) < 3 or len(red_candidates) < 2:
        return None

    gray_count = random.choice(GRAY_COUNT_CHOICES)
    red_count = random.choice(RED_COUNT_CHOICES)
    gray_choice = _pick_candidates(
        gray_candidates,
        gray_count,
        occupied=[],
        min_kind_counts=_gray_requirements(gray_count, gray_candidates),
    )
    if gray_choice is None:
        return None

    red_choice = _pick_candidates(
        red_candidates,
        red_count,
        occupied=[cand["pos"] for cand in gray_choice],
        min_kind_counts=_red_requirements(red_count, red_candidates),
    )
    if red_choice is None:
        return None

    for cand in gray_choice:
        r, c = cand["pos"]
        grid[r, c] = GRAY
    for cand in red_choice:
        r, c = cand["pos"]
        grid[r, c] = RED

    if not _validate_generated(grid, row_lines, col_lines):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lattice-membership dilation task")
