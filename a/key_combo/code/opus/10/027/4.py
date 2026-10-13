from __future__ import annotations

import random

import numpy as np


BG = 0
FG = 2
MASK = 8
PLUS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
CANDIDATES = ((2, 1), (2, 2), (3, 1), (3, 2))


def _shift_row(row: np.ndarray, dc: int, bg: int = BG) -> np.ndarray:
    out = np.full(row.shape, int(bg), dtype=int)
    cols = np.flatnonzero(row != int(bg))
    dst = cols + int(dc)
    keep = (0 <= dst) & (dst < row.size)
    out[dst[keep]] = row[cols[keep]]
    return out


def _plus_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr, dc in PLUS_OFFSETS]


def _assert_plus_masks(grid: np.ndarray) -> None:
    mask = grid == MASK
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one cross mask"

    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = 0
    for r0, c0 in map(tuple, coords):
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue
        stack = [(r0, c0)]
        seen[r0, c0] = True
        comp: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        cr = (min(rows) + max(rows)) // 2
        cc = (min(cols) + max(cols)) // 2
        assert set(comp) == set(_plus_cells((cr, cc))), "each mask must be a 3x3 plus"
        components += 1

    assert 1 <= components <= 2, "expected one or two cross masks"


def _clean_rows(grid: np.ndarray) -> np.ndarray:
    return ~np.any(grid == MASK, axis=1)


def _candidate_support(grid: np.ndarray, period: int, shift: int) -> list[int]:
    clean = _clean_rows(grid)
    rows = []
    for r in range(grid.shape[0] - period):
        if not (clean[r] and clean[r + period]):
            continue
        if np.array_equal(_shift_row(grid[r], shift), grid[r + period]):
            rows.append(int(r))
        else:
            return []
    return rows


def _infer_period_and_shift(grid: np.ndarray) -> tuple[int, int]:
    valid: list[tuple[int, int]] = []
    for period, shift in CANDIDATES:
        support = _candidate_support(grid, period, shift)
        if len(support) >= 2:
            valid.append((int(period), int(shift)))
    assert len(valid) == 1, "expected a unique (period, shift) from clean row pairs"
    return valid[0]


def _resolve_cell(
    grid: np.ndarray,
    r: int,
    c: int,
    period: int,
    shift: int,
    memo: dict[tuple[int, int], int],
    visiting: set[tuple[int, int]],
) -> int:
    key = (int(r), int(c))
    if key in memo:
        return memo[key]

    value = int(grid[r, c])
    if value != MASK:
        memo[key] = value
        return value

    assert key not in visiting, "mask dependency should not cycle"
    visiting.add(key)
    src_r = int(r - period) if r - period >= 0 else int(r + period)
    src_c = int(c - shift)
    if src_c < 0:
        result = BG
    else:
        result = _resolve_cell(grid, src_r, src_c, period, shift, memo, visiting)
    visiting.remove(key)
    memo[key] = int(result)
    return int(result)


def _repair_with_params(grid: np.ndarray, period: int, shift: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    memo: dict[tuple[int, int], int] = {}
    for r, c in map(tuple, np.argwhere(out == MASK)):
        out[r, c] = _resolve_cell(grid, int(r), int(c), period, shift, memo, set())
    return out


def _full_pattern_holds(grid: np.ndarray, period: int, shift: int) -> bool:
    for r in range(period, grid.shape[0]):
        if not np.array_equal(grid[r], _shift_row(grid[r - period], shift)):
            return False
    return True


def _semantic_solutions(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    sols: list[tuple[int, int, np.ndarray]] = []
    for period, shift in CANDIDATES:
        support = _candidate_support(grid, period, shift)
        if len(support) < 2:
            continue
        repaired = _repair_with_params(grid, period, shift)
        if np.any(repaired == MASK):
            continue
        visible = grid != MASK
        if not np.array_equal(repaired[visible], grid[visible]):
            continue
        if not _full_pattern_holds(repaired, period, shift):
            continue
        sols.append((int(period), int(shift), repaired))
    return sols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the row-translation period from undamaged rows and repair cross-masked pixels.

    Input structure:
    - The grid uses black background, red row-pattern pixels, and one or two gray 3x3 plus masks.
    - Rows follow a periodic translation: for one unique period P in {2,3} and shift S in {1,2},
      row r equals row r-P shifted right by S, with off-grid pixels clipped away.
    - Mask color 8 marks damaged cells whose original values were either black or red.

    Rule:
    - Treat every gray pixel as unknown and ignore every row containing gray when inferring the pattern.
    - Find the unique (P, S) such that all clean row pairs separated by P satisfy the right-shift relation.
    - Repair each gray cell (r, c) from row r-P when available, otherwise from row r+P, reading column c-S;
      if c-S is outside the grid, fill black, and if that reference cell is also gray, recurse by another period step.

    Key insight:
    - The masks hide local values, but the untouched rows still reveal the only consistent vertical period and horizontal drift,
      which then determines every masked pixel by tracing backward through the row-translation chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    _assert_plus_masks(g)
    period, shift = _infer_period_and_shift(g)
    out = _repair_with_params(g, period, shift)
    assert _full_pattern_holds(out, period, shift), "repaired grid must obey the inferred pattern"
    return out


def _build_output(height: int, width: int, period: int, shift: int) -> np.ndarray | None:
    out = np.zeros((height, width), dtype=int)
    for base_r in range(period):
        max_steps = (height - 1 - base_r) // period
        max_col = width - 1 - shift * (max_steps + 1)
        if max_col < 1:
            return None

        count = random.randint(1, min(4, max_col + 1))
        if count == 1:
            cols = [random.randint(0, max_col)]
        elif random.random() < 0.5:
            start = random.randint(0, max_col - count + 1)
            cols = list(range(start, start + count))
        else:
            cols = sorted(random.sample(range(max_col + 1), count))
        out[base_r, cols] = FG

    for r in range(period, height):
        out[r] = _shift_row(out[r - period], shift)

    if not np.any(out == FG):
        return None
    for base_r in range(period):
        if np.array_equal(out[base_r], out[(base_r + 1) % period]):
            continue
    return out


def _branch_flags(grid: np.ndarray, period: int, shift: int) -> set[str]:
    flags: set[str] = set()
    mask = grid == MASK
    for r, c in map(tuple, np.argwhere(mask)):
        r = int(r)
        c = int(c)
        if r < period:
            flags.add("top")
        if c < shift:
            flags.add("left_black")
        src_r = int(r - period) if r - period >= 0 else int(r + period)
        src_c = int(c - shift)
        if src_c >= 0 and 0 <= src_r < grid.shape[0] and mask[src_r, src_c]:
            flags.add("recursive")
    return flags


def _stamp_masks(output: np.ndarray, centers: list[tuple[int, int]]) -> np.ndarray | None:
    h, w = output.shape
    cells: set[tuple[int, int]] = set()
    for center in centers:
        local = _plus_cells(center)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in local):
            return None
        for cell in local:
            if cell in cells:
                return None
            cells.add(cell)
    inp = output.copy()
    for r, c in cells:
        inp[r, c] = MASK
    return inp


def _scenario_targets(scenario: str) -> set[str]:
    if scenario == "single_top_left":
        return {"top", "left_black"}
    if scenario == "double_recursive":
        return {"recursive"}
    if scenario == "double_recursive_top":
        return {"recursive", "top"}
    return set()


def _sample_centers(
    height: int,
    width: int,
    period: int,
    shift: int,
    scenario: str,
) -> list[tuple[int, int]] | None:
    if scenario == "single_random":
        return [(random.randint(1, height - 2), random.randint(1, width - 2))]

    if scenario == "single_top_left":
        return [(random.randint(1, min(period, height - 2)), 1)]

    if scenario == "double_recursive":
        r0 = random.randint(1, height - period - 2)
        c0 = random.randint(1, width - shift - 2)
        return [(r0, c0), (r0 + period, c0 + shift)]

    if scenario == "double_recursive_top":
        if 1 + period > height - 2 or 1 + shift > width - 2:
            return None
        r0 = random.randint(1, min(period, height - period - 2))
        c0 = 1
        return [(r0, c0), (r0 + period, c0 + shift)]

    return None


def _attempt() -> np.ndarray | None:
    period, shift = random.choice(CANDIDATES)
    height = random.randint(11, 14)
    min_width = max(12, shift * (((height - 1) // period) + 1) + 4)
    width = random.randint(min_width, min(20, min_width + 3))
    clean = _build_output(height, width, period, shift)
    if clean is None:
        return None

    scenario = random.choice(
        ("single_random", "single_top_left", "double_recursive", "double_recursive_top")
    )
    centers = _sample_centers(height, width, period, shift, scenario)
    if not centers:
        return None

    inp = _stamp_masks(clean, centers)
    if inp is None:
        return None

    if not _scenario_targets(scenario).issubset(_branch_flags(inp, period, shift)):
        return None

    sols = _semantic_solutions(inp)
    if len(sols) != 1:
        return None

    got_period, got_shift, repaired = sols[0]
    if (got_period, got_shift) != (period, shift):
        return None
    if not np.array_equal(repaired, clean):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, clean):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-mask input")
