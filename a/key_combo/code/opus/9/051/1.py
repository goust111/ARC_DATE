from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
WALL = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MODE_FULL = "full"
MODE_CHECKER = "checker"
MODE_NONE = "none"


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not mask[sr, sc]:
                continue
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _shape_key(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _mode_from_parity(r: int, c: int) -> str:
    if (r & 1) == 0 and (c & 1) == 0:
        return MODE_FULL
    if (r & 1) == 1 and (c & 1) == 1:
        return MODE_CHECKER
    return MODE_NONE


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each wall-bounded chamber according to the parity of its single marker.

    Input structure
    - Black background cells are partitioned by one-pixel-thick grey walls into 4-connected closed chambers.
    - Every chamber contains exactly one non-grey, non-black marker pixel, and the rest of the chamber is black.
    - Grey wall pixels remain barriers and are not part of any chamber.

    Rule
    - Find each 4-connected non-wall chamber and locate its unique marker at absolute coordinate `(r, c)`.
    - If `(r % 2, c % 2) == (0, 0)`, recolor every black cell in that chamber to the marker color.
    - If `(r % 2, c % 2) == (1, 1)`, recolor only chamber black cells whose row parity and column parity both match the marker.
    - If the marker parity is `(0, 1)` or `(1, 0)`, leave the chamber unchanged apart from the original marker.

    Key insight
    - The wall geometry determines which cells belong to a chamber, but the marker's absolute row/column parity determines which of three fill modes that chamber uses.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for cells in _component_cells(g != WALL):
        markers = [(r, c) for r, c in cells if g[r, c] not in (BG, WALL)]
        assert len(markers) == 1, "every chamber must contain exactly one marker"
        mr, mc = markers[0]
        color = int(g[mr, mc])
        mode = _mode_from_parity(mr, mc)
        for r, c in cells:
            if g[r, c] != BG:
                continue
            if mode == MODE_FULL:
                out[r, c] = color
            elif mode == MODE_CHECKER and (r & 1) == (mr & 1) and (c & 1) == (mc & 1):
                out[r, c] = color
    return out


def _build_two_col_layout(width: int, heights: list[int]) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]], list[int]]:
    rows = len(heights)
    h = sum(heights) + rows + 1
    w = 2 * width + 3
    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    vertical_wall = width + 1
    grid[:, vertical_wall] = WALL
    col_spans = [(1, width), (width + 2, 2 * width + 1)]

    row_spans: list[tuple[int, int]] = []
    wall_rows: list[int] = []
    cur = 1
    for i, room_h in enumerate(heights):
        row_spans.append((cur, cur + room_h - 1))
        cur += room_h
        if i != rows - 1:
            grid[cur, :] = WALL
            wall_rows.append(cur)
            cur += 1
    return grid, row_spans, col_spans, wall_rows


def _open_doorway(grid: np.ndarray, wall_row: int, col_span: tuple[int, int]) -> None:
    c0, c1 = col_span
    doorway_cols = list(range(c0 + 1, c1))
    if not doorway_cols:
        doorway_cols = list(range(c0, c1 + 1))
    grid[wall_row, random.choice(doorway_cols)] = BG


def _add_spur(grid: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int]) -> bool:
    r0, r1 = row_span
    c0, c1 = col_span
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    options: list[tuple[str, int, int]] = []

    if width >= 4:
        for rr in range(r0 + 1, r1):
            for length in (1, 2):
                if c0 + length - 1 <= c1 - 1:
                    options.append(("from_left", rr, length))
                if c1 - length + 1 >= c0 + 1:
                    options.append(("from_right", rr, length))
    if height >= 4:
        for cc in range(c0 + 1, c1):
            for length in (1, 2):
                if r0 + length - 1 <= r1 - 1:
                    options.append(("from_top", cc, length))
                if r1 - length + 1 >= r0 + 1:
                    options.append(("from_bottom", cc, length))

    random.shuffle(options)
    for kind, anchor, length in options:
        changed: list[tuple[int, int]] = []
        if kind == "from_left":
            rr = anchor
            for cc in range(c0, c0 + length):
                changed.append((rr, cc))
        elif kind == "from_right":
            rr = anchor
            for cc in range(c1 - length + 1, c1 + 1):
                changed.append((rr, cc))
        elif kind == "from_top":
            cc = anchor
            for rr in range(r0, r0 + length):
                changed.append((rr, cc))
        else:
            cc = anchor
            for rr in range(r1 - length + 1, r1 + 1):
                changed.append((rr, cc))

        if any(grid[r, c] != BG for r, c in changed):
            continue
        for r, c in changed:
            grid[r, c] = WALL
        if len(_component_cells(grid != WALL)) >= 4:
            return True
        for r, c in changed:
            grid[r, c] = BG
    return False


def _generate_layout() -> np.ndarray:
    chamber_count = random.choices([4, 5, 6], weights=[2, 2, 3], k=1)[0]

    if chamber_count == 4:
        width = random.choice([4, 5])
        heights = list(random.choice([(4, 4), (3, 4), (4, 3)]))
        grid, row_spans, col_spans, _ = _build_two_col_layout(width, heights)
        if random.random() < 0.6:
            row_idx = random.randrange(len(row_spans))
            col_idx = random.randrange(2)
            _add_spur(grid, row_spans[row_idx], col_spans[col_idx])
        return grid

    if chamber_count == 5:
        heights = list(random.choice([(3, 3, 4), (4, 3, 3)]))
        grid, row_spans, col_spans, wall_rows = _build_two_col_layout(4, heights)
        sep_idx = 0 if heights[0] == heights[1] == 3 else 1
        _open_doorway(grid, wall_rows[sep_idx], random.choice(col_spans))
        return grid

    width = random.choice([4, 5])
    height_choices = [list(t) for t in [(3, 3, 4), (3, 4, 3), (4, 3, 3), (4, 4, 3), (3, 4, 4)]]
    if width == 5:
        height_choices.append([4, 3, 4])
    grid, row_spans, col_spans, _ = _build_two_col_layout(width, random.choice(height_choices))
    if random.random() < 0.5:
        available_rows = list(range(len(row_spans)))
        random.shuffle(available_rows)
        for row_idx in available_rows[: random.choice([1, 1, 2])]:
            col_idx = random.randrange(2)
            _add_spur(grid, row_spans[row_idx], col_spans[col_idx])
    return grid


def _candidate_cells_by_mode(cells: list[tuple[int, int]]) -> dict[str, list[tuple[int, int]]]:
    cell_set = set(cells)
    area = len(cells)
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    if area < 4:
        return out

    for r, c in cells:
        if not all((nr, nc) in cell_set for nr, nc in neighbors4(r, c)):
            continue
        mode = _mode_from_parity(r, c)
        if mode == MODE_CHECKER:
            matching = sum(((rr & 1) == (r & 1) and (cc & 1) == (c & 1)) for rr, cc in cells) - 1
            remaining = area - 1 - matching
            if matching >= 1 and remaining >= 2:
                out[mode].append((r, c))
        else:
            if area >= 4:
                out[mode].append((r, c))
    return out


def _same_shape_different_modes(shape_keys: list[tuple[tuple[int, int], ...]], assignments: dict[int, str]) -> bool:
    for i in range(len(shape_keys)):
        for j in range(i + 1, len(shape_keys)):
            if shape_keys[i] == shape_keys[j] and assignments.get(i) != assignments.get(j):
                return True
    return False


def _search_mode_assignment(mode_options: list[list[str]], shape_keys: list[tuple[tuple[int, int], ...]]) -> dict[int, str] | None:
    order = sorted(range(len(mode_options)), key=lambda idx: (len(mode_options[idx]), idx))
    assignments: dict[int, str] = {}

    def rec(pos: int) -> dict[int, str] | None:
        if pos == len(order):
            used = set(assignments.values())
            if {MODE_FULL, MODE_CHECKER, MODE_NONE} <= used and _same_shape_different_modes(shape_keys, assignments):
                return dict(assignments)
            return None

        idx = order[pos]
        choices = mode_options[idx][:]
        random.shuffle(choices)
        for mode in choices:
            assignments[idx] = mode
            result = rec(pos + 1)
            if result is not None:
                return result
            del assignments[idx]
        return None

    return rec(0)


def _place_markers(layout: np.ndarray) -> np.ndarray | None:
    chambers = _component_cells(layout == BG)
    if not (4 <= len(chambers) <= 6):
        return None

    chamber_infos = []
    for cells in chambers:
        shape = _shape_key(cells)
        mode_to_cells = _candidate_cells_by_mode(cells)
        if not mode_to_cells:
            return None
        chamber_infos.append((cells, shape, mode_to_cells))

    mode_options = [sorted(info[2]) for info in chamber_infos]
    if not all(mode_options):
        return None

    assignments = _search_mode_assignment(mode_options, [info[1] for info in chamber_infos])
    if assignments is None:
        return None

    colors = random.sample(list(MARKER_COLORS), len(chamber_infos))
    grid = layout.copy()
    order = list(range(len(chamber_infos)))
    random.shuffle(order)
    for idx, color in zip(order, colors):
        _, _, mode_to_cells = chamber_infos[idx]
        r, c = sorted(mode_to_cells[assignments[idx]])[0]
        grid[r, c] = int(color)
    return grid


def _validate_generated(inp: np.ndarray) -> None:
    assert isinstance(inp, np.ndarray)
    assert inp.ndim == 2
    assert max(inp.shape) <= 30
    assert np.all(np.isin(inp, list(range(10))))

    chambers = _component_cells(inp != WALL)
    assert 4 <= len(chambers) <= 6

    colors_seen: set[int] = set()
    shape_keys = [_shape_key(cells) for cells in chambers]
    mode_by_idx: dict[int, str] = {}

    out = main(inp)
    assert np.array_equal(out[inp == WALL], inp[inp == WALL])
    for idx, cells in enumerate(chambers):
        markers = [(r, c) for r, c in cells if inp[r, c] not in (BG, WALL)]
        assert len(markers) == 1
        mr, mc = markers[0]
        color = int(inp[mr, mc])
        assert color not in colors_seen
        colors_seen.add(color)
        assert 0 < color < 10 and color != WALL
        assert all(inp[nr, nc] != WALL for nr, nc in neighbors4(mr, mc))
        assert 8 <= len(cells) <= 25

        mode = _mode_from_parity(mr, mc)
        mode_by_idx[idx] = mode

        if mode == MODE_FULL:
            changed = sum(inp[r, c] == BG for r, c in cells)
            assert changed >= 3
        elif mode == MODE_CHECKER:
            changed = 0
            same_parity = 0
            other_parity = 0
            for r, c in cells:
                if inp[r, c] == BG and (r & 1) == (mr & 1) and (c & 1) == (mc & 1):
                    changed += 1
                    same_parity += 1
                elif inp[r, c] == BG:
                    other_parity += 1
            assert changed >= 1
            assert other_parity >= 2
        else:
            assert sum(inp[r, c] == BG for r, c in cells) >= 3

    assert {MODE_FULL, MODE_CHECKER, MODE_NONE} <= set(mode_by_idx.values())
    assert _same_shape_different_modes(shape_keys, mode_by_idx)

    for idx, cells in enumerate(chambers):
        markers = [(r, c) for r, c in cells if inp[r, c] not in (BG, WALL)]
        mr, mc = markers[0]
        color = int(inp[mr, mc])
        mode = mode_by_idx[idx]
        if mode == MODE_FULL:
            assert all(out[r, c] == color for r, c in cells)
        elif mode == MODE_CHECKER:
            for r, c in cells:
                if (r, c) == (mr, mc):
                    assert out[r, c] == color
                elif (r & 1) == (mr & 1) and (c & 1) == (mc & 1):
                    assert out[r, c] == color
                else:
                    assert out[r, c] == BG
        else:
            assert out[mr, mc] == color
            assert sum(out[r, c] == color for r, c in cells) == 1


def generate_input() -> np.ndarray:
    for _ in range(500):
        layout = _generate_layout()
        grid = _place_markers(layout)
        if grid is None:
            continue
        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid parity-gated chamber instance")
