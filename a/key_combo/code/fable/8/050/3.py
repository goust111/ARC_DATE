from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
RED = 2
GREEN = 3
YELLOW = 4
SEPARATOR = 5
CYAN = 8

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _groups(indices: list[int]) -> list[tuple[int, int]]:
    assert indices, "expected separator lines"
    spans: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            spans.append((start, prev))
            start = prev = idx
    spans.append((start, prev))
    return spans


def _spans_between(groups: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for line_start, line_end in groups:
        if start < line_start:
            spans.append((start, line_start - 1))
        start = line_end + 1
    if start < limit:
        spans.append((start, limit - 1))
    assert spans, "expected cell interiors between separator lines"
    return spans


def _separator_color(grid: np.ndarray) -> int:
    row_colors = {
        int(grid[r, 0])
        for r in range(grid.shape[0])
        if np.all(grid[r, :] == grid[r, 0])
    }
    col_colors = {
        int(grid[0, c])
        for c in range(grid.shape[1])
        if np.all(grid[:, c] == grid[0, c])
    }
    candidates = row_colors & col_colors
    assert len(candidates) == 1, "expected one separator color"
    return candidates.pop()


def _cell_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    sep = _separator_color(grid)
    row_lines = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == sep)]
    col_lines = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == sep)]
    row_spans = _spans_between(_groups(row_lines), grid.shape[0])
    col_spans = _spans_between(_groups(col_lines), grid.shape[1])
    heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1 and heights == widths, "cells must be square and uniform"
    return row_spans, col_spans, heights.pop()


def _background_color(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> int:
    vals: list[int] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            vals.extend(map(int, grid[r0 : r1 + 1, c0 : c1 + 1].ravel()))
    return Counter(vals).most_common(1)[0][0]


def _decode_glyph(patch: np.ndarray, color: int) -> tuple[int, int, int]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(patch == color)]
    assert len(coords) >= 4, "glyphs need an arm of at least length 3 plus one head pixel"
    coord_set = set(coords)

    matches: list[tuple[int, int, int]] = []
    rows = sorted({r for r, _c in coords})
    cols = sorted({c for _r, c in coords})

    for arm_r in rows:
        arm_cols = sorted(c for r, c in coords if r == arm_r)
        if len(arm_cols) + 1 != len(coords):
            continue
        if arm_cols != list(range(arm_cols[0], arm_cols[-1] + 1)):
            continue
        if len(arm_cols) < 3:
            continue
        extras = [rc for rc in coords if rc[0] != arm_r]
        if len(extras) != 1:
            continue
        head_r, head_c = extras[0]
        if abs(head_r - arm_r) != 1:
            continue
        if head_c == arm_cols[0]:
            matches.append((0, -1, len(arm_cols)))
        elif head_c == arm_cols[-1]:
            matches.append((0, 1, len(arm_cols)))

    for arm_c in cols:
        arm_rows = sorted(r for r, c in coords if c == arm_c)
        if len(arm_rows) + 1 != len(coords):
            continue
        if arm_rows != list(range(arm_rows[0], arm_rows[-1] + 1)):
            continue
        if len(arm_rows) < 3:
            continue
        extras = [rc for rc in coords if rc[1] != arm_c]
        if len(extras) != 1:
            continue
        head_r, head_c = extras[0]
        if abs(head_c - arm_c) != 1:
            continue
        if head_r == arm_rows[0]:
            matches.append((-1, 0, len(arm_rows)))
        elif head_r == arm_rows[-1]:
            matches.append((1, 0, len(arm_rows)))

    matches = list(dict.fromkeys(matches))
    assert len(matches) == 1, f"ambiguous or invalid glyph: {coord_set}"
    return matches[0]


def _classify_patch(patch: np.ndarray, bg: int) -> dict[str, object]:
    colors = sorted(set(map(int, np.unique(patch))) - {bg})
    if not colors:
        return {"kind": "empty"}
    assert len(colors) == 1, "each cell must contain at most one foreground color"
    color = colors[0]
    if color == GREEN:
        assert np.all(patch == GREEN), "goal cell must be fully green"
        return {"kind": "goal"}
    assert color in {RED, CYAN}, "only red/cyan glyph cells are allowed"
    return {"kind": "glyph", "color": color, "instr": _decode_glyph(patch, color)}


def _parse_cells(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], np.ndarray, int]:
    row_spans, col_spans, _cell_size = _cell_layout(grid)
    bg = _background_color(grid, row_spans, col_spans)
    cells = np.empty((len(row_spans), len(col_spans)), dtype=object)
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            cells[r, c] = _classify_patch(patch, bg)
    return row_spans, col_spans, cells, bg


def _paint_cell(out: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], r: int, c: int, color: int) -> None:
    r0, r1 = row_spans[r]
    c0, c1 = col_spans[c]
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _trace_program(cells: np.ndarray) -> tuple[list[tuple[int, int]], set[tuple[int, int]], list[tuple[int, int]]]:
    start_positions: list[tuple[int, int]] = []
    goal_positions: list[tuple[int, int]] = []
    glyph_positions: list[tuple[int, int]] = []
    for r in range(cells.shape[0]):
        for c in range(cells.shape[1]):
            cell = cells[r, c]
            if cell["kind"] == "glyph":
                glyph_positions.append((r, c))
                if cell["color"] == RED:
                    start_positions.append((r, c))
            elif cell["kind"] == "goal":
                goal_positions.append((r, c))

    assert len(start_positions) == 1, "expected exactly one red start glyph"
    assert len(goal_positions) == 1, "expected exactly one green goal cell"
    assert len(glyph_positions) >= 2, "expected at least one decoy or one next instruction"

    start = start_positions[0]
    goal = goal_positions[0]
    chain = [start]
    painted: list[tuple[int, int]] = []
    painted_set: set[tuple[int, int]] = set()
    seen = {start}
    current = start

    while True:
        cell = cells[current]
        assert cell["kind"] == "glyph"
        dr, dc, step = cell["instr"]
        for k in range(1, step):
            rr = current[0] + dr * k
            cc = current[1] + dc * k
            assert 0 <= rr < cells.shape[0] and 0 <= cc < cells.shape[1], "jump left the lattice"
            assert cells[rr, cc]["kind"] == "empty", "intermediate cells must be empty"
            if (rr, cc) not in painted_set:
                painted_set.add((rr, cc))
                painted.append((rr, cc))
        landing = (current[0] + dr * step, current[1] + dc * step)
        assert 0 <= landing[0] < cells.shape[0] and 0 <= landing[1] < cells.shape[1], "landing left the lattice"
        landing_cell = cells[landing]
        if landing == goal:
            assert landing_cell["kind"] == "goal"
            return chain, painted_set, painted
        assert landing_cell["kind"] == "glyph", "landing must be the next glyph or the goal"
        assert landing_cell["color"] == CYAN, "only the first instruction may be red"
        assert landing not in seen, "program must not cycle"
        seen.add(landing)
        chain.append(landing)
        current = landing


def main(grid: np.ndarray) -> np.ndarray:
    """Trace the routed instruction chain from the red glyph and fill every jumped-through empty cell yellow.

    Input structure
    - A single-color lattice of separator lines partitions the canvas into equal square cells.
    - Exactly one cell contains a red direction glyph, exactly one cell is filled solid green, and some other cells contain cyan glyphs of the same family.
    - Each glyph is a straight arm plus one side head pixel attached to one arm endpoint; all remaining cells are empty background.

    Rule
    - Decode each glyph as one instruction: the travel direction is from the arm tail toward the endpoint carrying the side head, and the step count is the arm length excluding the head pixel.
    - Start from the red glyph, jump through the cell lattice by that many cells, and fill every intermediate empty cell completely yellow.
    - If the landing cell is a cyan glyph, continue executing from there; if it is the solid green cell, stop. Unreached glyphs are decoys and stay unchanged.

    Key insight
    - The glyphs are not drawings to move; each one is a compound "direction plus distance" instruction, so the solution must execute the chain in cell coordinates rather than transform pixels geometrically.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_spans, col_spans, cells, _bg = _parse_cells(g)
    _chain, painted_set, painted_order = _trace_program(cells)

    out = g.copy()
    for r, c in painted_order:
        assert cells[r, c]["kind"] == "empty"
        _paint_cell(out, row_spans, col_spans, r, c, YELLOW)
    assert painted_set, "the routed program must paint at least one cell"
    return out


def _make_glyph_patch(cell_size: int, direction: tuple[int, int], step: int, color: int) -> np.ndarray:
    patch = np.full((cell_size, cell_size), BACKGROUND, dtype=int)
    dr, dc = direction
    if dr == 0:
        row = random.randint(1, cell_size - 2)
        head_side = random.choice([-1, 1])
        start_col = random.randint(0, cell_size - step)
        cols = list(range(start_col, start_col + step))
        patch[row, cols] = color
        head_col = cols[-1] if dc == 1 else cols[0]
        patch[row + head_side, head_col] = color
    else:
        col = random.randint(1, cell_size - 2)
        head_side = random.choice([-1, 1])
        start_row = random.randint(0, cell_size - step)
        rows = list(range(start_row, start_row + step))
        patch[rows, col] = color
        head_row = rows[-1] if dr == 1 else rows[0]
        patch[head_row, col + head_side] = color
    return patch


def _path_options(pos: tuple[int, int], n_cells: int, used: set[tuple[int, int]], max_step: int) -> list[tuple[tuple[int, int], int, tuple[int, int], list[tuple[int, int]]]]:
    options = []
    directions = DIRS[:]
    random.shuffle(directions)
    steps = list(range(3, max_step + 1))
    random.shuffle(steps)
    for dr, dc in directions:
        for step in steps:
            landing = (pos[0] + dr * step, pos[1] + dc * step)
            if not (0 <= landing[0] < n_cells and 0 <= landing[1] < n_cells):
                continue
            mids = [(pos[0] + dr * k, pos[1] + dc * k) for k in range(1, step)]
            if any(cell in used for cell in mids + [landing]):
                continue
            options.append(((dr, dc), step, landing, mids))
    random.shuffle(options)
    return options


def _build_chain(n_cells: int, max_step: int, instr_count: int) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], int]], set[tuple[int, int]], list[tuple[int, int]]] | None:
    require_turn = instr_count >= 2 and random.random() < 0.8
    require_mixed_step = instr_count >= 2 and random.random() < 0.5

    def dfs(
        positions: list[tuple[int, int]],
        instrs: list[tuple[tuple[int, int], int]],
        used: set[tuple[int, int]],
        painted: list[tuple[int, int]],
    ) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], int]], set[tuple[int, int]], list[tuple[int, int]]] | None:
        if len(instrs) == instr_count:
            axes = {0 if dr != 0 else 1 for (dr, _dc), _step in instrs}
            step_values = {step for _direction, step in instrs}
            if require_turn and len(axes) == 1:
                return None
            if require_mixed_step and len(step_values) == 1:
                return None
            return positions, instrs, used, painted

        pos = positions[-1]
        options = _path_options(pos, n_cells, used, max_step)
        def score(option: tuple[tuple[int, int], int, tuple[int, int], list[tuple[int, int]]]) -> tuple[int, int, float]:
            direction, step, _landing, _mids = option
            turn_bonus = 0
            if instrs and direction != instrs[-1][0]:
                turn_bonus = 1
            return (turn_bonus, step, random.random())
        options.sort(key=score, reverse=True)

        for direction, step, landing, mids in options:
            res = dfs(
                positions + [landing],
                instrs + [(direction, step)],
                used | set(mids) | {landing},
                painted + mids,
            )
            if res is not None:
                return res
        return None

    starts = [(r, c) for r in range(n_cells) for c in range(n_cells)]
    random.shuffle(starts)
    for start in starts:
        res = dfs([start], [], {start}, [])
        if res is not None:
            return res
    return None


def _render_abstract(
    n_cells: int,
    cell_size: int,
    positions: list[tuple[int, int]],
    instrs: list[tuple[tuple[int, int], int]],
    decoys: dict[tuple[int, int], tuple[tuple[int, int], int]],
) -> np.ndarray:
    side = n_cells * cell_size + (n_cells - 1)
    grid = np.full((side, side), BACKGROUND, dtype=int)
    for k in range(1, n_cells):
        idx = k * cell_size + (k - 1)
        grid[idx, :] = SEPARATOR
        grid[:, idx] = SEPARATOR

    for i, pos in enumerate(positions[:-1]):
        color = RED if i == 0 else CYAN
        patch = _make_glyph_patch(cell_size, instrs[i][0], instrs[i][1], color)
        r0 = pos[0] * (cell_size + 1)
        c0 = pos[1] * (cell_size + 1)
        grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = patch

    goal = positions[-1]
    r0 = goal[0] * (cell_size + 1)
    c0 = goal[1] * (cell_size + 1)
    grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = GREEN

    for pos, (direction, step) in decoys.items():
        patch = _make_glyph_patch(cell_size, direction, step, CYAN)
        r0 = pos[0] * (cell_size + 1)
        c0 = pos[1] * (cell_size + 1)
        grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = patch

    return grid


def _choose_decoys(
    n_cells: int,
    used: set[tuple[int, int]],
    max_step: int,
) -> dict[tuple[int, int], tuple[tuple[int, int], int]]:
    available = [(r, c) for r in range(n_cells) for c in range(n_cells) if (r, c) not in used]
    random.shuffle(available)
    upper = min(4, len(available))
    assert upper >= 1, "need room for at least one decoy"
    count = random.randint(1, upper)
    decoys: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    for pos in available:
        options = []
        for direction in DIRS:
            for step in range(3, max_step + 1):
                landing = (pos[0] + direction[0] * step, pos[1] + direction[1] * step)
                if 0 <= landing[0] < n_cells and 0 <= landing[1] < n_cells:
                    options.append((direction, step))
        if not options:
            continue
        decoys[pos] = random.choice(options)
        if len(decoys) == count:
            return decoys
    raise AssertionError("failed to place enough decoys")


def _expected_painted_cells(positions: list[tuple[int, int]], instrs: list[tuple[tuple[int, int], int]]) -> list[tuple[int, int]]:
    painted: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pos, (direction, step) in zip(positions[:-1], instrs):
        dr, dc = direction
        for k in range(1, step):
            cell = (pos[0] + dr * k, pos[1] + dc * k)
            if cell not in seen:
                painted.append(cell)
                seen.add(cell)
    return painted


def _validate_generation(
    inp: np.ndarray,
    positions: list[tuple[int, int]],
    instrs: list[tuple[tuple[int, int], int]],
    decoys: dict[tuple[int, int], tuple[tuple[int, int], int]],
) -> bool:
    try:
        row_spans, col_spans, cells, bg = _parse_cells(inp)
        chain, painted_set, painted_order = _trace_program(cells)
    except AssertionError:
        return False

    if bg != BACKGROUND:
        return False
    if chain != positions[:-1]:
        return False
    expected_painted = _expected_painted_cells(positions, instrs)
    if painted_order != expected_painted or painted_set != set(expected_painted):
        return False

    parsed_glyphs = {
        (r, c): tuple(cells[r, c]["instr"])
        for r in range(cells.shape[0])
        for c in range(cells.shape[1])
        if cells[r, c]["kind"] == "glyph" and cells[r, c]["color"] == CYAN
    }
    intended_chain = {pos: (direction[0], direction[1], step) for pos, (direction, step) in zip(positions[1:-1], instrs[1:])}
    intended_decoys = {pos: (direction[0], direction[1], step) for pos, (direction, step) in decoys.items()}
    if parsed_glyphs != intended_chain | intended_decoys:
        return False

    out = main(inp)
    yellow_cells = set()
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            block = out[r0 : r1 + 1, c0 : c1 + 1]
            if np.all(block == YELLOW):
                yellow_cells.add((r, c))
            elif (r, c) in painted_set:
                return False
    if yellow_cells != painted_set:
        return False
    if np.any(inp == YELLOW):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    cell_size = random.choice([4, 5])
    n_cells = 6 if cell_size == 4 else 5
    max_step = min(4, cell_size, n_cells - 1)
    instr_options = [2, 3, 4] if n_cells == 6 else [2, 3]
    instr_count = random.choice(instr_options)

    chain = _build_chain(n_cells, max_step, instr_count)
    if chain is None:
        return None
    positions, instrs, used, _painted = chain
    decoys = _choose_decoys(n_cells, used, max_step)
    inp = _render_abstract(n_cells, cell_size, positions, instrs, decoys)

    if not _validate_generation(inp, positions, instrs, decoys):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid glyph routing input")
