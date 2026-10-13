from __future__ import annotations

import random

import numpy as np


BG = 0
PLACEHOLDER = 3
DIVIDER = 5
TARGET = 8
PALETTE_CHOICES = (1, 2, 4, 6, 7, 9)

HEIGHT_CHOICES = (10, 12, 14, 16)
LEFT_WIDTH_CHOICES = (8, 10, 12)
PLACEHOLDER_COUNT_RANGE = (3, 5)
WIDTH_RANGE = (2, 3)
HEIGHT_MARGIN = 2
GROUP_GAP_CHOICES = (0, 1)
OUTER_PAD_CHOICES = (1, 2)
MAX_GENERATION_ATTEMPTS = 400
LAYOUT_ATTEMPTS = 200


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


def _find_divider(grid: np.ndarray) -> int:
    cols = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]
    assert len(cols) == 1, "expected exactly one full-height gray divider"
    divider = cols[0]
    assert 0 < divider < grid.shape[1] - 1, "divider must split left and right panels"
    return divider


def _repair_left_panel(panel: np.ndarray) -> tuple[np.ndarray, list[int]]:
    height, width = map(int, panel.shape)
    out = panel.copy()
    greens = [tuple(map(int, rc)) for rc in np.argwhere(panel == PLACEHOLDER)]
    assert greens, "expected green placeholders on the left panel"
    greens.sort()

    palette: list[int] = []
    seen_orbits: set[tuple[tuple[int, int], ...]] = set()
    for row, col in greens:
        orbit = _reflection_orbit(row, col, height, width)
        assert orbit not in seen_orbits, "each placeholder must occupy a distinct orbit"
        seen_orbits.add(orbit)
        values = [int(panel[r, c]) for r, c in orbit]
        assert values.count(PLACEHOLDER) == 1, "each damaged orbit must contain one green cell"
        visible = {value for value in values if value != PLACEHOLDER}
        assert len(visible) == 1, "each damaged orbit must reveal one visible color"
        fill = visible.pop()
        palette.append(fill)
        for r, c in orbit:
            if out[r, c] == PLACEHOLDER:
                out[r, c] = fill
    return out, palette


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _column_interval(mask: np.ndarray, col: int) -> tuple[int, int]:
    rows = np.flatnonzero(mask[:, col])
    assert rows.size > 0, "target columns must be nonempty"
    r0, r1 = int(rows[0]), int(rows[-1])
    assert np.array_equal(rows, np.arange(r0, r1 + 1)), "each target column must be one contiguous segment"
    return r0, r1


def _right_runs(panel: np.ndarray) -> tuple[list[tuple[int, int, int, int]], int, int]:
    bg = _most_common_color(panel)
    colors = {int(v) for v in np.unique(panel)}
    colors.discard(bg)
    assert len(colors) == 1, "right panel must contain one target drawing color"
    target = colors.pop()
    mask = panel == target

    cols = [int(c) for c in np.flatnonzero(mask.any(axis=0))]
    assert cols, "expected at least one target column"

    runs: list[tuple[int, int, int, int]] = []
    cur_r0, cur_r1 = _column_interval(mask, cols[0])
    cur_c0 = cur_c1 = cols[0]
    for col in cols[1:]:
        r0, r1 = _column_interval(mask, col)
        if col == cur_c1 + 1 and (r0, r1) == (cur_r0, cur_r1):
            cur_c1 = col
        else:
            runs.append((cur_r0, cur_c0, cur_r1, cur_c1))
            cur_r0, cur_r1 = r0, r1
            cur_c0 = cur_c1 = col
    runs.append((cur_r0, cur_c0, cur_r1, cur_c1))
    return runs, bg, target


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the left damaged symmetric pattern into a palette, then recolor the right column-run groups in order.

    Input structure:
    - A full-height gray vertical line splits the grid into left and right panels.
    - The left panel is horizontally and vertically symmetric once green placeholders are ignored; each green cell is the lone damaged member of its reflection orbit.
    - The right panel contains one single-color drawing whose occupied columns each hold one contiguous vertical span; adjacent columns with the same span form one rectangular column-run group.

    Rule:
    - On the left panel, repair every green placeholder with the unique visible color from its two-axis reflection orbit.
    - Read the repaired placeholder colors in placeholder reading order: top-to-bottom, then left-to-right.
    - On the right panel, recolor the column-run groups from left to right with that repaired color sequence; keep the divider unchanged.

    Key insight:
    - The palette is hidden inside the left symmetry repair, and the placeholder positions determine the recoloring order rather than the geometry of the right panel itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider = _find_divider(g)
    left = g[:, :divider]
    right = g[:, divider + 1 :]

    repaired_left, palette = _repair_left_panel(left)
    runs, right_bg, _target = _right_runs(right)
    assert len(runs) == len(palette), "placeholder count must match right-side run count"

    out = g.copy()
    out[:, :divider] = repaired_left
    out[:, divider + 1 :] = right_bg

    for (r0, c0, r1, c1), color in zip(runs, palette):
        out[r0 : r1 + 1, divider + 1 + c0 : divider + 1 + c1 + 1] = color

    return out


def _reflect_quadrant(quadrant: np.ndarray) -> np.ndarray:
    return np.block(
        [
            [quadrant, quadrant[:, ::-1]],
            [quadrant[::-1, :], quadrant[::-1, ::-1]],
        ]
    )


def _pick_placeholder_positions(rng: random.Random, qh: int, qw: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(qh) for c in range(qw)]
    for _ in range(LAYOUT_ATTEMPTS):
        chosen = rng.sample(cells, count)
        if len({r for r, _c in chosen}) < 2 or len({c for _r, c in chosen}) < 2:
            continue
        if any(max(abs(r1 - r2), abs(c1 - c2)) <= 1 for i, (r1, c1) in enumerate(chosen) for r2, c2 in chosen[i + 1 :]):
            continue
        row_major = sorted(chosen)
        col_major = sorted(chosen, key=lambda rc: (rc[1], rc[0]))
        if row_major == col_major:
            continue
        return row_major
    return None


def _build_left_output(
    rng: random.Random, height: int, left_width: int, palette: list[int]
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    qh, qw = height // 2, left_width // 2
    positions = _pick_placeholder_positions(rng, qh, qw, len(palette))
    if positions is None:
        raise ValueError("failed to pick placeholder positions")

    extra_pool = [color for color in PALETTE_CHOICES if color not in palette]
    extra_count = min(len(extra_pool), rng.randint(2, 3))
    active_colors = [BG, *palette, *rng.sample(extra_pool, extra_count)]

    quadrant = np.zeros((qh, qw), dtype=int)
    for row in range(qh):
        for col in range(qw):
            neighbors = []
            if row > 0:
                neighbors.append(int(quadrant[row - 1, col]))
            if col > 0:
                neighbors.append(int(quadrant[row, col - 1]))
            if row > 0 and col > 0:
                neighbors.append(int(quadrant[row - 1, col - 1]))
            if neighbors and rng.random() < 0.45:
                quadrant[row, col] = rng.choice(neighbors)
            else:
                quadrant[row, col] = rng.choice(active_colors)

    for (row, col), color in zip(positions, palette):
        quadrant[row, col] = color

    out = _reflect_quadrant(quadrant)
    return out, positions


def _sample_spans(
    rng: random.Random, height: int, gaps: list[int]
) -> list[tuple[int, int]] | None:
    spans: list[tuple[int, int]] = []
    max_h = height - HEIGHT_MARGIN
    for i in range(len(gaps) + 1):
        for _ in range(LAYOUT_ATTEMPTS):
            run_h = rng.randint(2, max_h)
            r0 = rng.randint(0, height - run_h)
            span = (r0, r0 + run_h - 1)
            if i > 0 and gaps[i - 1] == 0 and span == spans[-1]:
                continue
            spans.append(span)
            break
        else:
            return None
    if len({span for span in spans}) == 1:
        return None
    return spans


def _build_right_panels(
    rng: random.Random, height: int, palette: list[int], max_width: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]] | None:
    count = len(palette)
    for _ in range(LAYOUT_ATTEMPTS):
        widths = [rng.randint(*WIDTH_RANGE) for _ in range(count)]
        gaps = [rng.choice(GROUP_GAP_CHOICES) for _ in range(count - 1)]
        if gaps:
            gaps[rng.randrange(len(gaps))] = 0
        spans = _sample_spans(rng, height, gaps)
        if spans is None:
            continue

        left_pad = rng.choice(OUTER_PAD_CHOICES)
        right_pad = rng.choice(OUTER_PAD_CHOICES)
        used_width = left_pad + right_pad + sum(widths) + sum(gaps)
        if used_width > max_width:
            continue
        right_width = used_width + rng.randint(0, max_width - used_width)

        out = np.full((height, right_width), BG, dtype=int)
        inp = np.full_like(out, BG)
        runs: list[tuple[int, int, int, int]] = []
        col = left_pad
        for i, color in enumerate(palette):
            r0, r1 = spans[i]
            width = widths[i]
            out[r0 : r1 + 1, col : col + width] = color
            inp[r0 : r1 + 1, col : col + width] = TARGET
            runs.append((r0, col, r1, col + width - 1))
            col += width
            if i < len(gaps):
                col += gaps[i]

        target_area = int(np.count_nonzero(inp == TARGET))
        bg_area = int(inp.size - target_area)
        if bg_area <= target_area:
            continue
        touching = [
            i
            for i in range(len(runs) - 1)
            if runs[i][3] + 1 == runs[i + 1][1] and (runs[i][0], runs[i][2]) != (runs[i + 1][0], runs[i + 1][2])
        ]
        if not touching:
            continue
        return inp, out, runs
    return None


def _assemble_grid(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [left, np.full((left.shape[0], 1), DIVIDER, dtype=int), right],
        axis=1,
    )


def _validate_instance(
    inp: np.ndarray,
    out: np.ndarray,
    left_positions: list[tuple[int, int]],
    palette: list[int],
    right_runs: list[tuple[int, int, int, int]],
) -> bool:
    if not np.array_equal(main(inp), out):
        return False

    if int(np.count_nonzero(inp == DIVIDER)) != inp.shape[0]:
        return False
    if len(set(palette)) != len(palette):
        return False
    if len({r for r, _c in left_positions}) < 2 or len({c for _r, c in left_positions}) < 2:
        return False
    if left_positions == sorted(left_positions, key=lambda rc: (rc[1], rc[0])):
        return False

    divider = _find_divider(inp)
    repaired_left, repaired_palette = _repair_left_panel(inp[:, :divider])
    if repaired_palette != palette:
        return False
    if not np.array_equal(repaired_left, out[:, :divider]):
        return False

    runs, bg, target = _right_runs(inp[:, divider + 1 :])
    if target != TARGET or len(runs) != len(palette) or runs != right_runs:
        return False
    if bg != BG:
        return False

    out_right = out[:, divider + 1 :]
    if np.any((inp[:, divider + 1 :] != BG) & (inp[:, divider + 1 :] != TARGET)):
        return False
    for (r0, c0, r1, c1), color in zip(runs, palette):
        if not np.all(out_right[r0 : r1 + 1, c0 : c1 + 1] == color):
            return False
    touching = [
        i
        for i in range(len(runs) - 1)
        if runs[i][3] + 1 == runs[i + 1][1] and (runs[i][0], runs[i][2]) != (runs[i + 1][0], runs[i + 1][2])
    ]
    return bool(touching)


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    height = rng.choice(HEIGHT_CHOICES)
    left_width = rng.choice(LEFT_WIDTH_CHOICES)
    count = rng.randint(*PLACEHOLDER_COUNT_RANGE)
    palette = rng.sample(PALETTE_CHOICES, count)

    left_out, left_positions = _build_left_output(rng, height, left_width, palette)
    left_in = left_out.copy()
    for row, col in left_positions:
        left_in[row, col] = PLACEHOLDER

    max_right_width = 30 - left_width - 1
    right_panels = _build_right_panels(rng, height, palette, max_right_width)
    if right_panels is None:
        raise ValueError("failed to build right panel")
    right_in, right_out, right_runs = right_panels

    inp = _assemble_grid(left_in, right_in)
    out = _assemble_grid(left_out, right_out)
    if not _validate_instance(inp, out, left_positions, palette, right_runs):
        raise ValueError("semantic validation failed")
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random()
        try:
            inp, _out = _generate_case(rng)
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
