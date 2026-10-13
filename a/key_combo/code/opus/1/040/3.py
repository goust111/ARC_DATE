from __future__ import annotations

import random

import numpy as np


SEP = 4
BG = 0
RED = 2
OBSTACLE = 5
INERT_A = 1
INERT_B = 3
ACTIVE_COLORS = {BG, RED, OBSTACLE}

CELL_ROWS = 4
CELL_COLS = 4
ROW_SIZE_CHOICES = (4, 5)
COL_SIZE_CHOICES = (4, 5, 6)
ACTIVE_COUNT_RANGE = (6, 8)
MAX_GENERATION_ATTEMPTS = 400


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for cut in separators:
        assert start < cut, "adjacent separator lines would create an empty cell band"
        bands.append((start, cut - 1))
        start = cut + 1
    assert start < size, "separator lines cannot consume the full grid"
    bands.append((start, size - 1))
    return bands


def _layout_from_sizes(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    bands: list[tuple[int, int]] = []
    separators: list[int] = []
    cursor = 0
    for i, size in enumerate(sizes):
        bands.append((cursor, cursor + size - 1))
        cursor += size
        if i != len(sizes) - 1:
            separators.append(cursor)
            cursor += 1
    return bands, separators, cursor


def _simulate_active_patch(patch: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    patch = np.asarray(patch, dtype=int)
    h, w = patch.shape
    out = patch.copy()
    starts = [int(c) for c in np.flatnonzero(patch[-1] == RED)]
    events: list[dict[str, object]] = []

    for sc in starts:
        r = h - 1
        c = sc
        turns = 0
        trace = [(r, c)]
        while True:
            if r == 0:
                stop = "top"
                break
            above = int(out[r - 1, c])
            if above == BG:
                r -= 1
                out[r, c] = RED
                trace.append((r, c))
                continue
            if above == OBSTACLE:
                if c + 1 >= w or int(out[r, c + 1]) != BG:
                    stop = "right_blocked"
                    break
                c += 1
                turns += 1
                out[r, c] = RED
                trace.append((r, c))
                continue
            stop = "occupied_above"
            break
        events.append(
            {
                "start_col": sc,
                "turns": turns,
                "stop": stop,
                "trace": trace,
            }
        )
    return out, events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow red control paths only inside palette-valid separator-lattice cells, turning right under gray blockers.

    Input structure:
    - Full yellow rows and columns split the grid into rectangular cells.
    - Active cells use only black, red, and gray; inert cells contain at least one other color, often blue/green noise.
    - Red controls, when present, lie on the bottom row of an active cell; inert cells may still contain decorative red pixels that are not controls.

    Rule:
    - Preserve every separator pixel and every inert cell exactly as they are.
    - In each active cell, process the bottom-row red controls from left to right.
    - A path repeatedly moves up through black cells; if the cell above is gray, it steps one cell right on the current row if that cell is black, then resumes trying to move up.
    - Stop a path when it reaches the cell's top edge or when its required destination is not black. Earlier painted red path cells block later paths in the same cell.

    Key insight:
    - The lattice supplies local cell boundaries, but the full cell palette decides whether the path-growth rule is allowed to run there at all.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape

    sep_rows = [r for r in range(h) if np.all(g[r, :] == SEP)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == SEP)]
    assert sep_rows and sep_cols, "expected full separator rows and columns"

    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    out = g.copy()

    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            cell_colors = {int(v) for v in np.unique(cell)}
            if not cell_colors <= ACTIVE_COLORS:
                continue

            starts = [c0 + int(c) for c in np.flatnonzero(cell[-1] == RED)]
            for start_c in starts:
                r = r1
                c = start_c
                while True:
                    if r - 1 < r0:
                        break
                    above = int(out[r - 1, c])
                    if above == BG:
                        r -= 1
                        out[r, c] = RED
                    elif above == OBSTACLE:
                        if c + 1 > c1 or int(out[r, c + 1]) != BG:
                            break
                        c += 1
                        out[r, c] = RED
                    else:
                        break

    return out


def _scatter_color(
    patch: np.ndarray,
    color: int,
    count: int,
    forbidden: set[tuple[int, int]] | None = None,
) -> None:
    h, w = patch.shape
    blocked = forbidden or set()
    coords = [(r, c) for r in range(h) for c in range(w) if (r, c) not in blocked]
    random.shuffle(coords)
    for r, c in coords[:count]:
        patch[r, c] = color


def _make_active_empty(h: int, w: int) -> tuple[np.ndarray, dict[str, object]]:
    patch = np.full((h, w), BG, dtype=int)
    _scatter_color(patch, OBSTACLE, random.randint(0, min(3, h * w // 4)))
    expected, events = _simulate_active_patch(patch)
    assert not events
    return patch, {"kind": "active_empty", "expected": expected, "events": events}


def _make_active_straight(h: int, w: int, controls: int) -> tuple[np.ndarray, dict[str, object]]:
    assert 1 <= controls <= min(2, w)
    patch = np.full((h, w), BG, dtype=int)
    cols = sorted(random.sample(range(w), controls))
    for c in cols:
        patch[-1, c] = RED

    forbidden = {(r, c) for c in cols for r in range(h)}
    _scatter_color(patch, OBSTACLE, random.randint(0, min(2, h * w // 5)), forbidden)
    expected, events = _simulate_active_patch(patch)
    assert len(events) == controls
    assert all(int(e["turns"]) == 0 for e in events)
    return patch, {"kind": "active_straight", "expected": expected, "events": events}


def _make_active_turn(h: int, w: int) -> tuple[np.ndarray, dict[str, object]]:
    assert h >= 4 and w >= 2
    patch = np.full((h, w), BG, dtype=int)

    max_turns = min(3, h - 1, w - 1)
    turns = random.randint(1, max_turns)
    start_c = random.randint(0, w - 1 - turns)
    turn_rows = sorted(random.sample(list(range(1, h)), turns), reverse=True)

    patch[-1, start_c] = RED
    current_r = h - 1
    current_c = start_c
    path_cells = {(current_r, current_c)}
    blocker_cells: set[tuple[int, int]] = set()

    for turn_r in turn_rows:
        while current_r > turn_r:
            current_r -= 1
            path_cells.add((current_r, current_c))
        patch[current_r - 1, current_c] = OBSTACLE
        blocker_cells.add((current_r - 1, current_c))
        current_c += 1
        path_cells.add((current_r, current_c))

    while current_r > 0:
        current_r -= 1
        path_cells.add((current_r, current_c))

    forbidden = path_cells | blocker_cells
    _scatter_color(patch, OBSTACLE, random.randint(0, 1), forbidden)
    expected, events = _simulate_active_patch(patch)
    assert len(events) == 1
    assert int(events[0]["turns"]) == turns
    return patch, {"kind": "active_turn", "expected": expected, "events": events}


def _make_active_multi_order(h: int, w: int) -> tuple[np.ndarray, dict[str, object]]:
    assert h >= 4 and w >= 2
    patch = np.full((h, w), BG, dtype=int)
    left = random.randint(0, w - 2)
    right = left + 1
    patch[-1, left] = RED
    patch[-1, right] = RED
    patch[h - 3, left] = OBSTACLE

    expected, events = _simulate_active_patch(patch)
    assert len(events) == 2
    assert int(events[0]["turns"]) >= 1
    assert events[1]["stop"] == "occupied_above"
    assert expected[h - 2, right] == RED
    return patch, {"kind": "active_multi_order", "expected": expected, "events": events}


def _make_inert_patch(h: int, w: int, lure: bool) -> tuple[np.ndarray, dict[str, object]]:
    rr, cc = np.indices((h, w))
    style = random.choice(("checker", "diag", "vstripe", "hstripe"))
    phase = random.randint(0, 2)

    if style == "checker":
        patch = np.where((rr + cc + phase) % 2 == 0, INERT_A, INERT_B)
    elif style == "diag":
        patch = np.where((rr + 2 * cc + phase) % 3 == 0, INERT_A, INERT_B)
    elif style == "vstripe":
        patch = np.where((cc + phase) % 3 == 0, INERT_A, INERT_B)
    else:
        patch = np.where((rr + phase) % 3 == 0, INERT_A, INERT_B)
    patch = patch.astype(int)

    extras = [(BG, random.randint(0, 1)), (OBSTACLE, random.randint(0, 1))]
    for color, count in extras:
        if count == 0:
            continue
        _scatter_color(patch, color, count)

    lure_info: dict[str, object] = {}
    if lure:
        c = random.randint(0, w - 2)
        patch[-1, c] = RED
        patch[-2, c] = BG
        patch[-2, c + 1] = BG
        patch[-3, c] = OBSTACLE
        lure_info = {"lure_col": c}

    assert INERT_A in patch and INERT_B in patch
    assert any(int(v) not in ACTIVE_COLORS for v in np.unique(patch))
    return patch, {"kind": "inert_lure" if lure else "inert", "lure": lure, **lure_info}


def _build_cell_patch(kind: str, h: int, w: int) -> tuple[np.ndarray, dict[str, object]]:
    if kind == "active_empty":
        return _make_active_empty(h, w)
    if kind == "active_straight":
        return _make_active_straight(h, w, controls=1)
    if kind == "active_double":
        return _make_active_straight(h, w, controls=2 if w >= 2 else 1)
    if kind == "active_turn":
        return _make_active_turn(h, w)
    if kind == "active_multi_order":
        return _make_active_multi_order(h, w)
    if kind == "inert":
        return _make_inert_patch(h, w, lure=False)
    if kind == "inert_lure":
        return _make_inert_patch(h, w, lure=True)
    raise AssertionError(f"unknown cell kind: {kind}")


def _semantic_check(
    grid: np.ndarray,
    expected: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    cell_meta: dict[tuple[int, int], dict[str, object]],
) -> bool:
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False

    active_count = 0
    inert_count = 0
    saw_turn = False
    saw_multi_order = False
    saw_straight = False
    saw_lure = False

    for pos, meta in cell_meta.items():
        r0, r1 = row_bands[pos[0]]
        c0, c1 = col_bands[pos[1]]
        inp_cell = grid[r0 : r1 + 1, c0 : c1 + 1]
        out_cell = solved[r0 : r1 + 1, c0 : c1 + 1]
        colors = {int(v) for v in np.unique(inp_cell)}
        is_active = colors <= ACTIVE_COLORS
        kind = str(meta["kind"])

        if is_active:
            active_count += 1
            if not np.array_equal(out_cell, np.asarray(meta["expected"], dtype=int)):
                return False
        else:
            inert_count += 1
            if not np.array_equal(out_cell, inp_cell):
                return False

        if kind == "active_turn":
            events = meta["events"]
            if int(events[0]["turns"]) < 1:
                return False
            saw_turn = True
        elif kind == "active_multi_order":
            events = meta["events"]
            if len(events) != 2 or events[1]["stop"] != "occupied_above":
                return False
            saw_multi_order = True
        elif kind == "active_straight":
            events = meta["events"]
            if not events or any(int(e["turns"]) != 0 for e in events):
                return False
            saw_straight = True
        elif kind == "inert_lure":
            lure_sim, _ = _simulate_active_patch(inp_cell)
            if np.array_equal(lure_sim, inp_cell):
                return False
            if not np.array_equal(out_cell, inp_cell):
                return False
            if not np.any(inp_cell[-1] == RED):
                return False
            saw_lure = True

    if not (ACTIVE_COUNT_RANGE[0] <= active_count <= ACTIVE_COUNT_RANGE[1]):
        return False
    if inert_count != CELL_ROWS * CELL_COLS - active_count:
        return False
    if not (saw_turn and saw_multi_order and saw_straight and saw_lure):
        return False
    if not np.array_equal(grid[grid == SEP], expected[grid == SEP]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    row_sizes = [random.choice(ROW_SIZE_CHOICES) for _ in range(CELL_ROWS)]
    col_sizes = [random.choice(COL_SIZE_CHOICES) for _ in range(CELL_COLS)]

    row_bands, sep_rows, height = _layout_from_sizes(row_sizes)
    col_bands, sep_cols, width = _layout_from_sizes(col_sizes)
    if height > 30 or width > 30:
        return None

    positions = [(r, c) for r in range(CELL_ROWS) for c in range(CELL_COLS)]
    active_count = random.randint(*ACTIVE_COUNT_RANGE)

    special_positions = random.sample(positions, 4)
    straight_pos, turn_pos, multi_pos, empty_pos = special_positions
    active_positions = set(special_positions)
    remaining = [pos for pos in positions if pos not in active_positions]
    while len(active_positions) < active_count:
        active_positions.add(random.choice(remaining))
        remaining = [pos for pos in remaining if pos not in active_positions]

    inert_positions = [pos for pos in positions if pos not in active_positions]
    lure_pos = random.choice(inert_positions)

    grid = np.full((height, width), SEP, dtype=int)
    expected = grid.copy()
    cell_meta: dict[tuple[int, int], dict[str, object]] = {}

    for r_idx, (r0, r1) in enumerate(row_bands):
        for c_idx, (c0, c1) in enumerate(col_bands):
            pos = (r_idx, c_idx)
            cell_h = r1 - r0 + 1
            cell_w = c1 - c0 + 1

            if pos == straight_pos:
                kind = "active_straight"
            elif pos == turn_pos:
                kind = "active_turn"
            elif pos == multi_pos:
                kind = "active_multi_order"
            elif pos == empty_pos:
                kind = "active_empty"
            elif pos == lure_pos:
                kind = "inert_lure"
            elif pos in active_positions:
                kind = random.choice(("active_empty", "active_straight", "active_double", "active_turn"))
            else:
                kind = "inert"

            patch, meta = _build_cell_patch(kind, cell_h, cell_w)
            grid[r0 : r1 + 1, c0 : c1 + 1] = patch
            if kind.startswith("active"):
                expected[r0 : r1 + 1, c0 : c1 + 1] = np.asarray(meta["expected"], dtype=int)
            else:
                expected[r0 : r1 + 1, c0 : c1 + 1] = patch
            cell_meta[pos] = meta

    if any(np.all(grid[r, :] == SEP) != (r in sep_rows) for r in range(height)):
        return None
    if any(np.all(grid[:, c] == SEP) != (c in sep_cols) for c in range(width)):
        return None
    if not _semantic_check(grid, expected, row_bands, col_bands, cell_meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
