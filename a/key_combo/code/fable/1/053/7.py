from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


POWER = 5
WIRE = 8
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = [0, 1, 2, 3, 4, 6, 7, 9]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(h: int, w: int, r: int, c: int):
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(h, w, rr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _powered_network_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    conductive = (grid == POWER) | (grid == WIRE)
    starts = list(map(tuple, np.argwhere(grid == POWER)))
    assert starts, "expected one grey power block"
    reached = np.zeros_like(conductive, dtype=bool)
    q = deque(starts)
    for cell in starts:
        reached[cell] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(h, w, r, c):
            if conductive[nr, nc] and not reached[nr, nc]:
                reached[nr, nc] = True
                q.append((nr, nc))
    return reached


def _extract_dominoes(grid: np.ndarray, bg: int) -> list[dict]:
    dominoes: list[dict] = []
    colors = sorted(set(map(int, np.unique(grid))) - {bg, POWER, WIRE})
    for color in colors:
        for comp in _components(grid == color):
            assert len(comp) == 2, "foreground markers must be 2-pixel dominoes"
            comp = sorted(comp)
            rows = sorted({r for r, _c in comp})
            cols = sorted({c for _r, c in comp})
            if len(rows) == 1 and cols[1] == cols[0] + 1:
                orient = "h"
                axis = rows[0]
            elif len(cols) == 1 and rows[1] == rows[0] + 1:
                orient = "v"
                axis = cols[0]
            else:
                raise AssertionError("domino must be horizontal or vertical")
            dominoes.append(
                {
                    "color": int(color),
                    "cells": tuple(comp),
                    "orient": orient,
                    "axis": int(axis),
                }
            )
    assert dominoes, "expected colored domino markers"
    return dominoes


def _domino_is_powered(domino: dict, powered_mask: np.ndarray) -> bool:
    h, w = powered_mask.shape
    for r, c in domino["cells"]:
        for nr, nc in _neighbors4(h, w, r, c):
            if powered_mask[nr, nc]:
                return True
    return False


def _render_output(
    grid: np.ndarray,
    active_dominoes: list[dict],
    inactive_dominoes: list[dict],
    bg: int,
) -> np.ndarray:
    out = grid.copy()
    for domino in inactive_dominoes:
        for r, c in domino["cells"]:
            out[r, c] = bg

    col_colors: dict[int, int] = {}
    row_colors: dict[int, int] = {}
    for domino in active_dominoes:
        color = int(domino["color"])
        axis = int(domino["axis"])
        if domino["orient"] == "v":
            previous = col_colors.setdefault(axis, color)
        else:
            previous = row_colors.setdefault(axis, color)
        assert previous == color, "an output bar cannot receive two colors"

    for c, color in col_colors.items():
        out[:, c] = color
    for r, color in row_colors.items():
        out[r, :] = color
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)

    power_components = _components(g == POWER)
    assert len(power_components) == 1, "expected one power component"
    power_cells = power_components[0]
    power_rows = {r for r, _c in power_cells}
    power_cols = {c for _r, c in power_cells}
    assert len(power_cells) == 4 and len(power_rows) == 2 and len(power_cols) == 2, "power block must be 2x2"

    powered_mask = _powered_network_mask(g)
    dominoes = _extract_dominoes(g, bg)

    active_dominoes = [domino for domino in dominoes if _domino_is_powered(domino, powered_mask)]
    inactive_dominoes = [domino for domino in dominoes if not _domino_is_powered(domino, powered_mask)]
    return _render_output(g, active_dominoes, inactive_dominoes, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend only the powered domino markers into full bars and erase the unpowered ones.

    Input structure:
    - The background is the dominant color.
    - One gray 2x2 power block and one or more teal wire paths are present.
    - Every other foreground object is a same-color two-pixel domino, either horizontal or vertical.

    Rule:
    - Follow 4-connected teal-wire connectivity from the gray power block.
    - A domino executes only when at least one of its cells touches that powered wire component.
    - Powered vertical dominoes paint their whole column in their own color, powered horizontal dominoes paint their whole row, and rows are painted after columns.
    - Unpowered dominoes are erased to the background, while wires and the power block stay visible wherever a later bar does not cover them.

    Key insight:
    - Domino orientation gives the instruction, but the hidden gating condition is electrical connectivity to the power network; decoy dominoes can look identical and even touch disconnected wires.
    """
    return _solve(grid)


def _domino_cells(r: int, c: int, orient: str) -> list[tuple[int, int]]:
    if orient == "h":
        return [(r, c), (r, c + 1)]
    return [(r, c), (r + 1, c)]


def _can_place_with_padding(grid: np.ndarray, cells: list[tuple[int, int]], bg: int) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != bg:
            return False
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cell_set and int(grid[rr, cc]) != bg:
                    return False
    return True


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _shortest_wire_path(grid: np.ndarray, bg: int, target: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    start_cells = [tuple(cell) for cell in np.argwhere((grid == POWER) | (grid == WIRE))]
    if target in start_cells:
        return [target]

    passable = (grid == bg) | (grid == POWER) | (grid == WIRE)
    passable[target] = True
    prev: dict[tuple[int, int], tuple[int, int] | None] = {}
    q = deque()
    for cell in start_cells:
        prev[cell] = None
        q.append(cell)

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(h, w, r, c):
            nxt = (nr, nc)
            if nxt in prev or not passable[nr, nc]:
                continue
            prev[nxt] = (r, c)
            if nxt == target:
                path = [nxt]
                cur = prev[nxt]
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                return path
            q.append(nxt)
    return None


def _add_wire_path(grid: np.ndarray, bg: int, path: list[tuple[int, int]]) -> None:
    for r, c in path:
        if int(grid[r, c]) == bg:
            grid[r, c] = WIRE


def _place_active_horizontal(
    grid: np.ndarray,
    bg: int,
    row: int,
    color: int,
    side: str,
) -> dict | None:
    h, w = grid.shape
    starts = list(range(0, w - 1))
    random.shuffle(starts)
    for c in starts:
        cells = _domino_cells(row, c, "h")
        if not _can_place_with_padding(grid, cells, bg):
            continue
        anchor_c = random.choice([c, c + 1])
        target = (row + 1, anchor_c) if side == "top" else (row - 1, anchor_c)
        if not (0 <= target[0] < h and 0 <= target[1] < w) or int(grid[target]) != bg:
            continue
        _paint_cells(grid, cells, color)
        return {
            "color": color,
            "cells": tuple(cells),
            "orient": "h",
            "axis": row,
            "powered": True,
            "stubbed": False,
            "target": target,
        }
    return None


def _place_active_vertical(
    grid: np.ndarray,
    bg: int,
    col: int,
    color: int,
    side: str,
) -> dict | None:
    h, _w = grid.shape
    starts = list(range(0, h - 1))
    random.shuffle(starts)
    for r in starts:
        cells = _domino_cells(r, col, "v")
        if not _can_place_with_padding(grid, cells, bg):
            continue
        anchor_r = random.choice([r, r + 1])
        target = (anchor_r, col + 1) if side == "left" else (anchor_r, col - 1)
        if not (0 <= target[0] < h and 0 <= target[1] < grid.shape[1]) or int(grid[target]) != bg:
            continue
        _paint_cells(grid, cells, color)
        return {
            "color": color,
            "cells": tuple(cells),
            "orient": "v",
            "axis": col,
            "powered": True,
            "stubbed": False,
            "target": target,
        }
    return None


def _inactive_candidates(
    grid: np.ndarray,
    bg: int,
    orient: str,
    active_rows: set[int],
    active_cols: set[int],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    row_limit = h if orient == "h" else h - 1
    col_limit = w - 1 if orient == "h" else w
    for r in range(row_limit):
        for c in range(col_limit):
            cells = _domino_cells(r, c, orient)
            if any(rr in active_rows or cc in active_cols for rr, cc in cells):
                continue
            if _can_place_with_padding(grid, cells, bg):
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _stub_for_domino(
    grid: np.ndarray,
    bg: int,
    cells: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    domino_set = set(cells)
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    options: list[list[tuple[int, int]]] = []
    for r, c in cells:
        for dr, dc in DIR4:
            first = (r + dr, c + dc)
            key = (first, (dr, dc))
            if first in domino_set or key in seen:
                continue
            seen.add(key)
            for length in (1, 2, 3):
                stub = [(r + dr * k, c + dc * k) for k in range(1, length + 1)]
                ok = True
                for rr, cc in stub:
                    if not (0 <= rr < h and 0 <= cc < w) or int(grid[rr, cc]) != bg:
                        ok = False
                        break
                if not ok:
                    continue
                stub_set = set(stub)
                allowed = domino_set | stub_set
                for rr, cc in stub:
                    for nr, nc in _neighbors4(h, w, rr, cc):
                        if int(grid[nr, nc]) != bg and (nr, nc) not in allowed:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    options.append(stub)
    random.shuffle(options)
    return options[0] if options else None


def _place_inactive_domino(
    grid: np.ndarray,
    bg: int,
    orient: str,
    color: int,
    active_rows: set[int],
    active_cols: set[int],
    *,
    with_stub: bool,
) -> dict | None:
    for r, c in _inactive_candidates(grid, bg, orient, active_rows, active_cols):
        cells = _domino_cells(r, c, orient)
        stub = _stub_for_domino(grid, bg, cells) if with_stub else None
        if with_stub and stub is None:
            continue
        _paint_cells(grid, cells, color)
        if stub is not None:
            _paint_cells(grid, stub, WIRE)
        return {
            "color": color,
            "cells": tuple(cells),
            "orient": orient,
            "axis": r if orient == "h" else c,
            "powered": False,
            "stubbed": bool(stub),
        }
    return None


def _render_expected_from_meta(
    grid: np.ndarray,
    bg: int,
    active_dominoes: list[dict],
    inactive_dominoes: list[dict],
) -> np.ndarray:
    return _render_output(grid, active_dominoes, inactive_dominoes, bg)


def _render_all_dominoes(
    grid: np.ndarray,
    bg: int,
    active_dominoes: list[dict],
    inactive_dominoes: list[dict],
) -> np.ndarray:
    return _render_output(grid, active_dominoes + inactive_dominoes, [], bg)


def _render_any_wired_dominoes(
    grid: np.ndarray,
    bg: int,
    active_dominoes: list[dict],
    inactive_dominoes: list[dict],
) -> np.ndarray:
    active = list(active_dominoes)
    active.extend(domino for domino in inactive_dominoes if domino["stubbed"])
    inactive = [domino for domino in inactive_dominoes if not domino["stubbed"]]
    return _render_output(grid, active, inactive, bg)


def _wire_components_excluding_dominoes(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    return _components((grid == POWER) | (grid == WIRE))


def _semantic_checks(
    inp: np.ndarray,
    bg: int,
    active_dominoes: list[dict],
    inactive_dominoes: list[dict],
) -> bool:
    out = main(inp)
    expected = _render_expected_from_meta(inp, bg, active_dominoes, inactive_dominoes)
    if not np.array_equal(out, expected):
        return False

    active_rows = {int(domino["axis"]) for domino in active_dominoes if domino["orient"] == "h"}
    active_cols = {int(domino["axis"]) for domino in active_dominoes if domino["orient"] == "v"}
    if not active_rows or not active_cols or not inactive_dominoes:
        return False

    try:
        if np.array_equal(out, _render_all_dominoes(inp, bg, active_dominoes, inactive_dominoes)):
            return False
    except AssertionError:
        pass
    try:
        if np.array_equal(out, _render_any_wired_dominoes(inp, bg, active_dominoes, inactive_dominoes)):
            return False
    except AssertionError:
        pass

    if np.count_nonzero((out == POWER) | (out == WIRE)) < 3:
        return False

    crossing_ok = False
    for h_domino in active_dominoes:
        if h_domino["orient"] != "h":
            continue
        for v_domino in active_dominoes:
            if v_domino["orient"] != "v":
                continue
            if int(h_domino["color"]) != int(v_domino["color"]) and int(out[h_domino["axis"], v_domino["axis"]]) == int(h_domino["color"]):
                crossing_ok = True
    if not crossing_ok:
        return False

    if not any(domino["stubbed"] for domino in inactive_dominoes):
        return False

    untouched_inactive = 0
    for domino in inactive_dominoes:
        cells = domino["cells"]
        if all(int(out[r, c]) == bg for r, c in cells):
            untouched_inactive += 1
    if untouched_inactive == 0:
        return False

    if len(_wire_components_excluding_dominoes(inp)) < 2:
        return False

    return not np.array_equal(inp, out)


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(16, 22)
    w = random.randint(16, 22)
    bg = random.choice(PALETTE)
    grid = np.full((h, w), bg, dtype=int)

    power_r = random.randint(4, h - 6)
    power_c = random.randint(4, w - 6)
    grid[power_r : power_r + 2, power_c : power_c + 2] = POWER

    count_options = [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 1),
        (1, 2, 2),
        (2, 1, 1),
        (2, 1, 2),
        (2, 2, 1),
    ]
    n_h, n_v, n_inactive = random.choice(count_options)
    total_dominoes = n_h + n_v + n_inactive

    color_pool = [color for color in PALETTE if color != bg]
    if total_dominoes > len(color_pool):
        return None
    colors = random.sample(color_pool, total_dominoes)
    color_index = 0

    active_dominoes: list[dict] = []
    inactive_dominoes: list[dict] = []

    h_sides = random.sample(["top", "bottom"], n_h) if n_h == 1 else ["top", "bottom"]
    v_sides = random.sample(["left", "right"], n_v) if n_v == 1 else ["left", "right"]

    top_rows = list(range(1, power_r - 2))
    bottom_rows = list(range(power_r + 4, h - 1))
    left_cols = list(range(1, power_c - 2))
    right_cols = list(range(power_c + 4, w - 1))
    if len(top_rows) < ("top" in h_sides) or len(bottom_rows) < ("bottom" in h_sides):
        return None
    if len(left_cols) < ("left" in v_sides) or len(right_cols) < ("right" in v_sides):
        return None

    row_by_side = {}
    col_by_side = {}
    if "top" in h_sides:
        row_by_side["top"] = random.choice(top_rows)
    if "bottom" in h_sides:
        row_by_side["bottom"] = random.choice(bottom_rows)
    if "left" in v_sides:
        col_by_side["left"] = random.choice(left_cols)
    if "right" in v_sides:
        col_by_side["right"] = random.choice(right_cols)

    for side in h_sides:
        domino = _place_active_horizontal(grid, bg, row_by_side[side], colors[color_index], side)
        color_index += 1
        if domino is None:
            return None
        active_dominoes.append(domino)

    for side in v_sides:
        domino = _place_active_vertical(grid, bg, col_by_side[side], colors[color_index], side)
        color_index += 1
        if domino is None:
            return None
        active_dominoes.append(domino)

    random.shuffle(active_dominoes)
    for domino in active_dominoes:
        path = _shortest_wire_path(grid, bg, domino["target"])
        if path is None:
            return None
        _add_wire_path(grid, bg, path)

    active_rows = {int(domino["axis"]) for domino in active_dominoes if domino["orient"] == "h"}
    active_cols = {int(domino["axis"]) for domino in active_dominoes if domino["orient"] == "v"}

    inactive_specs = [True] + [random.choice([False, True]) for _ in range(n_inactive - 1)]
    random.shuffle(inactive_specs)
    for with_stub in inactive_specs:
        orient = random.choice(["h", "v"])
        domino = _place_inactive_domino(
            grid,
            bg,
            orient,
            colors[color_index],
            active_rows,
            active_cols,
            with_stub=with_stub,
        )
        color_index += 1
        if domino is None:
            return None
        inactive_dominoes.append(domino)

    if not _semantic_checks(grid, bg, active_dominoes, inactive_dominoes):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid powered-domino ARC input")
