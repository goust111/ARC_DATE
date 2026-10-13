from __future__ import annotations

import random
from collections import Counter

import numpy as np


PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _get_grid_color(grid: np.ndarray, bg: int) -> int:
    fg = grid[grid != bg]
    assert fg.size > 0
    counts = np.bincount(fg.astype(int), minlength=10)
    counts[int(bg)] = -1
    return int(np.argmax(counts))


def _infer_periodic_lines(grid: np.ndarray, bg: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    fg = grid != int(bg)
    H, W = grid.shape
    row_counts = fg.sum(axis=1)
    col_counts = fg.sum(axis=0)

    row_lines = tuple(int(r) for r in np.where(row_counts > W // 2)[0])
    col_lines = tuple(int(c) for c in np.where(col_counts > H // 2)[0])

    assert len(row_lines) >= 3 and len(col_lines) >= 3
    row_diffs = {int(b - a) for a, b in zip(row_lines, row_lines[1:])}
    col_diffs = {int(b - a) for a, b in zip(col_lines, col_lines[1:])}
    assert len(row_diffs) == 1 and len(col_diffs) == 1
    return row_lines, col_lines


def _template_mask(shape: tuple[int, int], row_lines: tuple[int, ...], col_lines: tuple[int, ...]) -> np.ndarray:
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    mask[np.array(row_lines, dtype=int), :] = True
    mask[:, np.array(col_lines, dtype=int)] = True
    return mask


def _detect_defects(
    grid: np.ndarray,
    bg: int,
    row_lines: tuple[int, ...],
    col_lines: tuple[int, ...],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    expected = _template_mask(grid.shape, row_lines, col_lines)

    extras = sorted((int(r), int(c)) for r, c in np.argwhere((grid != int(bg)) & ~expected))
    holes = sorted((int(r), int(c)) for r, c in np.argwhere((grid == int(bg)) & expected))

    assert extras and len(extras) == len(holes)
    return extras, holes


def _build_candidates(
    extras: list[tuple[int, int]],
    holes: list[tuple[int, int]],
) -> list[list[int]]:
    candidates: list[list[int]] = []
    for er, ec in extras:
        opts = [idx for idx, (hr, hc) in enumerate(holes) if hr == er or hc == ec]
        assert opts
        candidates.append(opts)

    for hid in range(len(holes)):
        assert any(hid in opts for opts in candidates)
    return candidates


def _solve_unique_matching(candidates: list[list[int]]) -> list[int]:
    n = len(candidates)
    assert n > 0

    def recurse(
        remaining: tuple[int, ...],
        available: frozenset[int],
        partial: dict[int, int],
    ) -> tuple[int, dict[int, int] | None]:
        rem = list(remaining)
        avail = set(available)
        cur = dict(partial)

        while True:
            changed = False
            for eid in list(rem):
                opts = [hid for hid in candidates[eid] if hid in avail]
                if not opts:
                    return 0, None
                if len(opts) == 1:
                    hid = opts[0]
                    cur[eid] = hid
                    rem.remove(eid)
                    avail.remove(hid)
                    changed = True
                    break
            if changed:
                continue

            for hid in list(avail):
                owners = [eid for eid in rem if hid in candidates[eid]]
                if not owners:
                    return 0, None
                if len(owners) == 1:
                    eid = owners[0]
                    cur[eid] = hid
                    rem.remove(eid)
                    avail.remove(hid)
                    changed = True
                    break
            if not changed:
                break

        if not rem:
            return 1, cur

        eid = min(rem, key=lambda idx: len([hid for hid in candidates[idx] if hid in avail]))
        total = 0
        first_solution: dict[int, int] | None = None
        for hid in [h for h in candidates[eid] if h in avail]:
            next_partial = dict(cur)
            next_partial[eid] = hid
            count, sol = recurse(
                tuple(x for x in rem if x != eid),
                frozenset(avail - {hid}),
                next_partial,
            )
            total += count
            if first_solution is None and sol is not None:
                first_solution = sol
            if total > 1:
                return total, first_solution
        return total, first_solution

    count, solution = recurse(tuple(range(n)), frozenset(range(n)), {})
    assert count == 1 and solution is not None
    return [int(solution[idx]) for idx in range(n)]


def _render_output(
    shape: tuple[int, int],
    bg: int,
    grid_color: int,
    row_lines: tuple[int, ...],
    col_lines: tuple[int, ...],
    extras: list[tuple[int, int]],
    holes: list[tuple[int, int]],
    assignment: list[int],
    extra_colors: list[int],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    out[_template_mask(shape, row_lines, col_lines)] = int(grid_color)
    for eid, hid in enumerate(assignment):
        hr, hc = holes[hid]
        out[int(hr), int(hc)] = int(extra_colors[eid])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each surplus pixel onto a uniquely matched gap to restore a periodic grid skeleton.

    Input structure:
    - The background is the dominant color; the main structure is a periodic skeleton made from evenly spaced full rows and full columns in one grid color.
    - Some expected skeleton cells are missing and show background, while some template-background cells contain isolated surplus pixels in various colors.
    - A surplus pixel may share a row or a column with one or more gaps, and some surplus pixels can even use the same color as the grid.

    Rule:
    - Infer the periodic row-lines and column-lines of the intact skeleton, then mark every non-template foreground cell as a surplus pixel and every template cell turned to background as a gap.
    - Match surplus pixels to gaps so that each pixel moves only along a shared row or shared column and every gap receives exactly one pixel; the full assignment is uniquely determined by constraint propagation.
    - Clear each surplus pixel from its original background position and fill its matched gap with that pixel's color.

    Key insight:
    - The hard part is not spotting colorful dots, but recovering the periodic lattice first so even grid-colored impostors are exposed and must participate in one global unique matching.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    grid_color = _get_grid_color(g, bg)
    row_lines, col_lines = _infer_periodic_lines(g, bg)
    extras, holes = _detect_defects(g, bg, row_lines, col_lines)
    candidates = _build_candidates(extras, holes)
    assignment = _solve_unique_matching(candidates)
    extra_colors = [int(g[r, c]) for r, c in extras]

    return _render_output(
        g.shape,
        bg,
        grid_color,
        row_lines,
        col_lines,
        extras,
        holes,
        assignment,
        extra_colors,
    )


def _sample_spaced(values: list[int], count: int, min_gap: int) -> list[int] | None:
    if len(values) < count:
        return None
    for gap in range(min_gap, -1, -1):
        for _ in range(80):
            pick = sorted(random.sample(values, count))
            if all(b - a >= gap for a, b in zip(pick, pick[1:])):
                return pick
    return None


def _line_positions(period: int, offset: int, n_lines: int) -> tuple[int, ...]:
    return tuple(int(offset + step * period) for step in range(n_lines))


def _make_template_params() -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    while True:
        row_period = random.randint(4, 6)
        col_period = random.randint(4, 6)
        n_row_lines = random.randint(3, 5)
        n_col_lines = random.randint(3, 5)
        row_offset = random.randint(0, row_period - 1)
        col_offset = random.randint(0, col_period - 1)
        row_tail = random.randint(0, row_period - 1)
        col_tail = random.randint(0, col_period - 1)

        H = row_offset + row_period * (n_row_lines - 1) + 1 + row_tail
        W = col_offset + col_period * (n_col_lines - 1) + 1 + col_tail
        if 16 <= H <= 30 and 16 <= W <= 30:
            return H, W, _line_positions(row_period, row_offset, n_row_lines), _line_positions(col_period, col_offset, n_col_lines)


def _make_staircase_extras(
    bg_rows: list[int],
    bg_cols: list[int],
    length: int,
    start_axis: str,
) -> list[tuple[int, int]] | None:
    row_changes = sum(1 for step in range(1, length) if (start_axis == "row") == (step % 2 == 1))
    col_changes = (length - 1) - row_changes
    rows_needed = 1 + row_changes
    cols_needed = 1 + col_changes

    rows = _sample_spaced(bg_rows, rows_needed, 2)
    cols = _sample_spaced(bg_cols, cols_needed, 2)
    if rows is None or cols is None:
        return None

    ridx = 1
    cidx = 1
    cur_r = rows[0]
    cur_c = cols[0]
    extras = [(int(cur_r), int(cur_c))]

    for step in range(1, length):
        change_row = (start_axis == "row") == (step % 2 == 1)
        if change_row:
            cur_r = rows[ridx]
            ridx += 1
        else:
            cur_c = cols[cidx]
            cidx += 1
        extras.append((int(cur_r), int(cur_c)))
    return extras


def _make_chain_holes(
    extras: list[tuple[int, int]],
    row_lines: tuple[int, ...],
    col_lines: tuple[int, ...],
) -> list[tuple[int, int]] | None:
    needed_line_rows = 0
    needed_line_cols = 0
    for prev, cur in zip(extras, extras[1:]):
        if prev[1] == cur[1]:
            needed_line_rows += 1
        else:
            needed_line_cols += 1

    last_prev, last_cur = extras[-2], extras[-1]
    if last_prev[1] == last_cur[1]:
        needed_line_cols += 1
    else:
        needed_line_rows += 1

    picked_line_rows = _sample_spaced(list(row_lines), needed_line_rows, 1)
    picked_line_cols = _sample_spaced(list(col_lines), needed_line_cols, 1)
    if picked_line_rows is None or picked_line_cols is None:
        return None

    row_iter = iter(picked_line_rows)
    col_iter = iter(picked_line_cols)
    holes: list[tuple[int, int]] = []

    for prev, cur in zip(extras, extras[1:]):
        if prev[1] == cur[1]:
            holes.append((int(next(row_iter)), int(prev[1])))
        else:
            holes.append((int(prev[0]), int(next(col_iter))))

    if last_prev[1] == last_cur[1]:
        holes.append((int(last_cur[0]), int(next(col_iter))))
    else:
        holes.append((int(next(row_iter)), int(last_cur[1])))

    return holes


def _make_extra_colors(length: int, bg: int, grid_color: int) -> list[int]:
    others = [c for c in PALETTE if c not in (bg, grid_color)]
    colors = [int(random.choice(others)) for _ in range(length)]

    if length >= 3 and random.random() < 0.8:
        hidden_idx = random.randint(1, length - 2)
        colors[hidden_idx] = int(grid_color)
        if random.random() < 0.3:
            second_idx = random.choice([idx for idx in range(length) if idx != hidden_idx])
            colors[second_idx] = int(grid_color)

    if all(color == int(grid_color) for color in colors):
        colors[random.randrange(length)] = int(random.choice(others))
    return colors


def _build_input_and_expected(
    shape: tuple[int, int],
    bg: int,
    grid_color: int,
    row_lines: tuple[int, ...],
    col_lines: tuple[int, ...],
    extras: list[tuple[int, int]],
    holes: list[tuple[int, int]],
    extra_colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    template = np.full(shape, int(bg), dtype=int)
    template[_template_mask(shape, row_lines, col_lines)] = int(grid_color)

    inp = template.copy()
    for hr, hc in holes:
        inp[int(hr), int(hc)] = int(bg)
    for (er, ec), color in zip(extras, extra_colors):
        inp[int(er), int(ec)] = int(color)

    assignment = _solve_unique_matching(_build_candidates(extras, holes))
    expected = _render_output(
        shape,
        bg,
        grid_color,
        row_lines,
        col_lines,
        extras,
        holes,
        assignment,
        extra_colors,
    )
    return inp, expected


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    bg: int,
    grid_color: int,
    row_lines: tuple[int, ...],
    col_lines: tuple[int, ...],
    extras: list[tuple[int, int]],
    holes: list[tuple[int, int]],
    extra_colors: list[int],
) -> bool:
    try:
        got_rows, got_cols = _infer_periodic_lines(inp, bg)
        if got_rows != row_lines or got_cols != col_lines:
            return False

        got_extras, got_holes = _detect_defects(inp, bg, got_rows, got_cols)
        if got_extras != sorted(extras) or got_holes != sorted(holes):
            return False

        candidates = _build_candidates(got_extras, got_holes)
        if not any(len(opts) > 1 for opts in candidates):
            return False
        if max(sum(1 for opts in candidates if hid in opts) for hid in range(len(got_holes))) < 2:
            return False

        solved = _solve_unique_matching(candidates)
        rendered = _render_output(
            inp.shape,
            bg,
            grid_color,
            got_rows,
            got_cols,
            got_extras,
            got_holes,
            solved,
            [int(inp[r, c]) for r, c in got_extras],
        )
        if not np.array_equal(rendered, expected):
            return False
        if not np.array_equal(main(inp), expected):
            return False

        if not any(color != int(grid_color) for color in extra_colors):
            return False
        if not any(int(inp[r, c]) == int(grid_color) for r, c in extras):
            return False
        if np.array_equal(inp, expected):
            return False
        return True
    except AssertionError:
        return False


def _generate_attempt() -> np.ndarray | None:
    H, W, row_lines, col_lines = _make_template_params()
    bg, grid_color = random.sample(PALETTE, 2)

    bg_rows = [r for r in range(H) if r not in row_lines]
    bg_cols = [c for c in range(W) if c not in col_lines]
    if len(bg_rows) < 3 or len(bg_cols) < 3:
        return None

    start_axis = random.choice(["row", "col"])
    max_len = min(
        6,
        1 + len(bg_rows) + len(bg_cols),
        len(row_lines) + len(col_lines),
    )
    lengths = [length for length in range(3, max_len + 1)]
    random.shuffle(lengths)

    for length in lengths:
        extras = _make_staircase_extras(bg_rows, bg_cols, length, start_axis)
        if extras is None:
            continue
        holes = _make_chain_holes(extras, row_lines, col_lines)
        if holes is None:
            continue

        extra_colors = _make_extra_colors(length, bg, grid_color)
        inp, expected = _build_input_and_expected(
            (H, W),
            bg,
            grid_color,
            row_lines,
            col_lines,
            extras,
            holes,
            extra_colors,
        )
        if _validate_instance(
            inp,
            expected,
            bg,
            grid_color,
            row_lines,
            col_lines,
            extras,
            holes,
            extra_colors,
        ):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after repeated semantic validation")
