from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color


def _find_spine(red_coords: list[tuple[int, int]]) -> tuple[str, int, int, int]:
    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    for r, c in red_coords:
        rows[int(r)].append(int(c))
        cols[int(c)].append(int(r))

    row_spines = [(r, sorted(cs)) for r, cs in rows.items() if len(cs) == 2]
    col_spines = [(c, sorted(rs)) for c, rs in cols.items() if len(rs) == 2]
    assert len(row_spines) + len(col_spines) == 1, "expected exactly one aligned two-red spine"

    if row_spines:
        spine_row, spine_cols = row_spines[0]
        return "row", int(spine_row), int(spine_cols[0]), int(spine_cols[1])

    spine_col, spine_rows = col_spines[0]
    return "col", int(spine_col), int(spine_rows[0]), int(spine_rows[1])


def _paint_green_if_black(out: np.ndarray, r: int, c: int) -> None:
    value = int(out[r, c])
    assert value in (Color.BLACK, Color.RED, Color.GREEN, Color.GREY), "unexpected color on connector path"
    if value == Color.BLACK:
        out[r, c] = Color.GREEN


def _draw_spine(out: np.ndarray, axis: str, fixed: int, lo: int, hi: int) -> None:
    if axis == "row":
        for c in range(lo, hi + 1):
            _paint_green_if_black(out, fixed, c)
        return

    for r in range(lo, hi + 1):
        _paint_green_if_black(out, r, fixed)


def _cast_branch(out: np.ndarray, axis: str, fixed: int, red_coord: tuple[int, int]) -> None:
    r, c = red_coord
    if axis == "row":
        assert r != fixed
        step = 1 if r < fixed else -1
        rr = r + step
        while True:
            value = int(out[rr, c])
            if value == Color.GREY:
                break
            _paint_green_if_black(out, rr, c)
            if rr == fixed:
                break
            rr += step
        return

    assert c != fixed
    step = 1 if c < fixed else -1
    cc = c + step
    while True:
        value = int(out[r, cc])
        if value == Color.GREY:
            break
        _paint_green_if_black(out, r, cc)
        if cc == fixed:
            break
        cc += step


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a green spine between the unique aligned red pair, while grey scaffold lines truncate perpendicular rays.

    Input structure:
    - A black background contains a regular lattice of full grey rows and full grey columns.
    - Red control pixels lie strictly inside scaffold cells, never on grey lines.
    - Exactly one red pair shares a row or a column; that aligned pair defines the spine segment.

    Rule:
    - Paint the black cells between the two spine endpoints green, but leave grey scaffold cells grey so the spine visually passes through them.
    - For every other red pixel, cast a ray perpendicular to the spine toward it.
    - Recolor each visited black cell green until the ray reaches the spine or its next step would enter a grey scaffold cell; red and grey pixels remain unchanged.

    Key insight:
    - The aligned red pair sets the connector direction, and the grey lattice independently decides how far each perpendicular ray is allowed to travel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    assert bool(np.any(g == Color.BLACK)), "expected black background cells"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.RED, Color.GREY}, "unexpected input colors"

    red_coords = [(int(r), int(c)) for r, c in np.argwhere(g == Color.RED)]
    assert len(red_coords) >= 3, "expected a spine pair and at least one extra red control"

    axis, fixed, lo, hi = _find_spine(red_coords)
    out = g.copy()
    _draw_spine(out, axis, fixed, lo, hi)

    if axis == "row":
        for r, c in red_coords:
            if r == fixed:
                continue
            assert lo <= c <= hi, "off-spine red pixels must project onto the horizontal spine span"
            _cast_branch(out, axis, fixed, (r, c))
        return out

    for r, c in red_coords:
        if c == fixed:
            continue
        assert lo <= r <= hi, "off-spine red pixels must project onto the vertical spine span"
        _cast_branch(out, axis, fixed, (r, c))
    return out


# --------------- generator ---------------

_GEN_STEPS = (4, 5)
_GEN_BRANCH_COUNT_RANGE = (3, 6)
_GEN_MAX_ATTEMPTS = 400
_GEN_MIXED_BRANCH_PROB = 0.45


def _band_positions(step: int, band: int) -> list[int]:
    return list(range(int(band * step + 1), int((band + 1) * step)))


def _build_lattice(row_cells: int, col_cells: int, step: int) -> np.ndarray:
    h = int(row_cells * step + 1)
    w = int(col_cells * step + 1)
    grid = np.zeros((h, w), dtype=int)
    grid[::step, :] = Color.GREY
    grid[:, ::step] = Color.GREY
    return grid


def _expected_horizontal(
    grid: np.ndarray,
    spine_row: int,
    left_col: int,
    right_col: int,
    branches: list[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()

    for c in range(left_col, right_col + 1):
        if int(out[spine_row, c]) == Color.BLACK:
            out[spine_row, c] = Color.GREEN

    for r, c in branches:
        step = 1 if r < spine_row else -1
        rr = r + step
        while True:
            if int(out[rr, c]) == Color.GREY:
                break
            if rr != spine_row and int(out[rr, c]) == Color.BLACK:
                out[rr, c] = Color.GREEN
            if rr == spine_row:
                break
            rr += step

    return out


def _regular_gap(indices: list[int]) -> int | None:
    if len(indices) < 2:
        return None
    diffs = [int(b - a) for a, b in zip(indices, indices[1:])]
    return diffs[0] if len(set(diffs)) == 1 else None


def _validate_instance(grid: np.ndarray) -> dict[str, int | str]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.RED, Color.GREY}
    assert bool(np.any(g == Color.BLACK)), "expected black background cells"

    grey_rows = [int(r) for r in range(g.shape[0]) if np.all(g[r, :] == Color.GREY)]
    grey_cols = [int(c) for c in range(g.shape[1]) if np.all(g[:, c] == Color.GREY)]
    step_r = _regular_gap(grey_rows)
    step_c = _regular_gap(grey_cols)
    assert step_r is not None and step_r == step_c, "expected a regular scaffold lattice"

    grey_row_set = set(grey_rows)
    grey_col_set = set(grey_cols)
    red_coords = [(int(r), int(c)) for r, c in np.argwhere(g == Color.RED)]
    assert len(red_coords) >= 3
    for r, c in red_coords:
        assert r not in grey_row_set and c not in grey_col_set, "reds must sit inside cell interiors"

    axis, fixed, lo, hi = _find_spine(red_coords)
    if axis == "row":
        assert any(lo < c < hi for c in grey_cols), "spine endpoints must lie in different scaffold cells"
    else:
        assert any(lo < r < hi for r in grey_rows), "spine endpoints must lie in different scaffold cells"

    blocked = 0
    clear = 0
    for r, c in red_coords:
        if axis == "row":
            if r == fixed:
                continue
            assert lo <= c <= hi
            step = 1 if r < fixed else -1
            visible = 0
            rr = r + step
            while True:
                if int(g[rr, c]) == Color.GREY:
                    blocked += 1
                    assert visible >= 1, "blocked rays must visibly stop at a scaffold line"
                    break
                if rr != fixed:
                    visible += 1
                if rr == fixed:
                    clear += 1
                    assert visible >= 1, "clear rays must visibly reach the spine"
                    break
                rr += step
        else:
            if c == fixed:
                continue
            assert lo <= r <= hi
            step = 1 if c < fixed else -1
            visible = 0
            cc = c + step
            while True:
                if int(g[r, cc]) == Color.GREY:
                    blocked += 1
                    assert visible >= 1, "blocked rays must visibly stop at a scaffold line"
                    break
                if cc != fixed:
                    visible += 1
                if cc == fixed:
                    clear += 1
                    assert visible >= 1, "clear rays must visibly reach the spine"
                    break
                cc += step

    assert blocked >= 1, "each sample must exhibit visible truncation"

    out = main(g)
    assert np.all(out[g == Color.GREY] == Color.GREY), "scaffold lines must stay grey"
    assert np.all(out[g == Color.RED] == Color.RED), "control pixels must stay red"
    changed = out != g
    assert np.all(g[changed] == Color.BLACK), "only background cells may be recolored"
    assert np.all(out[changed] == Color.GREEN), "all painted cells must be green"
    assert bool(np.any(changed)), "output must differ from input"

    return {"axis": axis, "blocked": blocked, "clear": clear, "step": int(step_r)}


def _gen_horizontal_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    step = random.choice(_GEN_STEPS)
    max_cells = 29 // step
    row_cells = random.randint(4, max_cells)
    col_cells = random.randint(5, max_cells)
    if row_cells < 4 or col_cells < 5:
        return None

    spine_band = random.randint(1, row_cells - 2)
    spine_band_rows = _band_positions(step, spine_band)
    spine_row = random.choice((spine_band_rows[0], spine_band_rows[-1]))

    left_band = random.randint(0, col_cells - 3)
    right_band = random.randint(left_band + 2, col_cells - 1)
    left_col = random.choice(_band_positions(step, left_band))
    right_col = random.choice(_band_positions(step, right_band))
    if left_col >= right_col:
        return None

    attach_cols = [c for c in range(left_col + 1, right_col) if c % step != 0]
    if len(attach_cols) < _GEN_BRANCH_COUNT_RANGE[0]:
        return None

    same_band_rows = [r for r in spine_band_rows if abs(r - spine_row) >= 2]
    blocked_up = [r for b in range(spine_band) for r in _band_positions(step, b)[:-1]]
    blocked_down = [r for b in range(spine_band + 1, row_cells) for r in _band_positions(step, b)[1:]]
    blocked_pool = blocked_up + blocked_down
    all_rows = blocked_pool + same_band_rows

    max_branches = min(_GEN_BRANCH_COUNT_RANGE[1], len(attach_cols), len(all_rows))
    if max_branches < _GEN_BRANCH_COUNT_RANGE[0]:
        return None
    branch_count = random.randint(_GEN_BRANCH_COUNT_RANGE[0], max_branches)

    if same_band_rows and branch_count >= 2 and random.random() < _GEN_MIXED_BRANCH_PROB:
        max_unblocked = min(len(same_band_rows), branch_count - 1)
        n_unblocked = random.randint(1, max_unblocked)
    else:
        n_unblocked = 0
    blocked_count = branch_count - n_unblocked
    if blocked_count < 1 or blocked_count > len(blocked_pool):
        return None

    chosen_rows: list[int] = []
    used_rows: set[int] = set()

    if blocked_count >= 2 and blocked_up and blocked_down:
        row_up = random.choice(blocked_up)
        row_down = random.choice(blocked_down)
        chosen_rows.extend([row_up, row_down])
        used_rows.update(chosen_rows)

    remaining_blocked = blocked_count - len(chosen_rows)
    remaining_pool = [r for r in blocked_pool if r not in used_rows]
    if remaining_blocked > len(remaining_pool):
        return None
    chosen_rows.extend(random.sample(remaining_pool, remaining_blocked))

    if n_unblocked:
        chosen_rows.extend(random.sample(same_band_rows, n_unblocked))

    if len(set(chosen_rows)) != branch_count:
        return None

    branch_cols = random.sample(attach_cols, branch_count)
    random.shuffle(chosen_rows)
    branches = list(zip(chosen_rows, branch_cols))

    inp = _build_lattice(row_cells, col_cells, step)
    inp[spine_row, left_col] = Color.RED
    inp[spine_row, right_col] = Color.RED
    for r, c in branches:
        inp[r, c] = Color.RED

    expected = _expected_horizontal(inp, spine_row, left_col, right_col, branches)
    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    sample = _gen_horizontal_attempt()
    if sample is None:
        return None
    inp, expected = sample

    if random.random() < 0.5:
        inp = inp.T.copy()
        expected = expected.T.copy()

    info = _validate_instance(inp)
    if info["clear"] == 0 and random.random() < _GEN_MIXED_BRANCH_PROB:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid scaffold-ray truncation input")
