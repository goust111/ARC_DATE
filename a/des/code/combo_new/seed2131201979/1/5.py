from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]


# Fixed ARC palette used by the puzzle statement.
BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7

PRIMARY_COLORS = (BLUE, RED, YELLOW)
PRIMARY_PAIRS = ((RED, BLUE), (RED, YELLOW), (BLUE, YELLOW))

COLOR_TO_BASIS = {
    BLUE: frozenset({"b"}),
    RED: frozenset({"r"}),
    YELLOW: frozenset({"y"}),
    GREEN: frozenset({"b", "y"}),
    MAGENTA: frozenset({"b", "r"}),
    ORANGE: frozenset({"r", "y"}),
}
BASIS_TO_COLOR = {basis: color for color, basis in COLOR_TO_BASIS.items()}

SCENARIOS = ("isolated", "same_cross", "mixed_cross", "multi_cross")
GEN_RETRIES = 400
POSITION_RETRIES = 120
MIN_LINE_GAP = 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each source color along its path and recolor downstream segments after gray crossings.

    Input structure:
    - The background is black and paths are one-pixel-wide 4-connected colored routes.
    - Each route starts from a same-colored 2x2 source block and ends at a same-colored
      2x1 or 1x2 terminal bar.
    - Crossings are marked by gray pixels shared by one horizontal and one vertical path.

    Rule:
    - Trace every path from its source block toward its terminal bar.
    - When the traced path passes through a gray crossing, keep the gray pixel unchanged
      but absorb the original color of the perpendicular path at that crossing.
    - Recolor all later path cells and the terminal bar with the cumulative mixed color;
      source blocks stay unchanged and paths with no effective color change stay as they are.

    Key insight:
    - The endpoint shapes identify direction, and each gray crossing is resolved locally by
      continuing straight through the crossing while mixing with the perpendicular path's
      original color for all downstream cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert 1 < height <= 30 and 1 < width <= 30, f"grid exceeds ARC size limit: {g.shape}"

    sources = _find_sources(g)
    terminals = _find_terminals(g, sources)
    assert sources, "expected at least one source block"
    assert len(sources) == len(terminals), "source/terminal count mismatch"

    source_lookup = _region_lookup(sources)
    terminal_lookup = _region_lookup(terminals)

    out = g.copy()
    visit_counts: dict[Coord, int] = {}
    matched_terminals: set[int] = set()

    for source in sources:
        basis = set(COLOR_TO_BASIS[source["color"]])
        prev = source["port_inside"]
        cur = source["first_outside"]
        steps = 0

        while True:
            steps += 1
            assert steps <= height * width * 4, "path trace did not terminate"

            if cur in terminal_lookup:
                terminal = terminals[terminal_lookup[cur]]
                assert terminal["color"] == source["color"], "terminal color must match its source color"
                assert terminal["id"] not in matched_terminals, "terminal reused by multiple sources"
                matched_terminals.add(terminal["id"])
                final_color = BASIS_TO_COLOR[frozenset(basis)]
                for cell in terminal["cells"]:
                    out[cell] = final_color
                    visit_counts[cell] = visit_counts.get(cell, 0) + 1
                break

            assert cur not in source_lookup, "path unexpectedly re-entered a source block"

            value = int(g[cur])
            assert value != BG, "path trace left the network"

            if value == GRAY:
                visit_counts[cur] = visit_counts.get(cur, 0) + 1
                direction = (cur[0] - prev[0], cur[1] - prev[1])
                assert abs(direction[0]) + abs(direction[1]) == 1, "invalid trace direction"
                other_color = _crossing_color(g, cur, direction)
                basis = _merge_bases(basis, COLOR_TO_BASIS[other_color])
                nxt = (cur[0] + direction[0], cur[1] + direction[1])
                assert _occupied(g, nxt), "crossing must continue straight through"
                prev, cur = cur, nxt
                continue

            out[cur] = BASIS_TO_COLOR[frozenset(basis)]
            visit_counts[cur] = visit_counts.get(cur, 0) + 1
            next_cells = [nbr for nbr in _occupied_neighbors(g, *cur) if nbr != prev]
            assert len(next_cells) == 1, f"expected one forward continuation at {cur}, got {next_cells}"
            prev, cur = cur, next_cells[0]

    assert len(matched_terminals) == len(terminals), "not all terminals were reached"
    _validate_visit_counts(g, sources, terminals, visit_counts)
    return out


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _occupied(grid: np.ndarray, cell: Coord) -> bool:
    r, c = cell
    return _in_bounds(grid.shape, r, c) and int(grid[r, c]) != BG


def _occupied_neighbors(grid: np.ndarray, r: int, c: int) -> list[Coord]:
    return [cell for cell in _neighbors(r, c) if _occupied(grid, cell)]


def _external_neighbors(grid: np.ndarray, cells: set[Coord]) -> set[Coord]:
    found: set[Coord] = set()
    for r, c in cells:
        for cell in _neighbors(r, c):
            if cell in cells:
                continue
            if _occupied(grid, cell):
                found.add(cell)
    return found


def _find_sources(grid: np.ndarray) -> list[dict[str, object]]:
    height, width = grid.shape
    sources: list[dict[str, object]] = []
    used_cells: set[Coord] = set()

    for r in range(height - 1):
        for c in range(width - 1):
            cells = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
            if cells & used_cells:
                continue
            color = int(grid[r, c])
            if color not in COLOR_TO_BASIS:
                continue
            if any(int(grid[rr, cc]) != color for rr, cc in cells):
                continue
            outside = _external_neighbors(grid, cells)
            if len(outside) != 1:
                continue
            first_outside = next(iter(outside))
            if int(grid[first_outside]) != color:
                continue
            port_cells = [cell for cell in cells if first_outside in _neighbors(*cell)]
            if len(port_cells) != 1:
                continue
            used_cells |= cells
            sources.append(
                {
                    "id": len(sources),
                    "color": color,
                    "cells": frozenset(cells),
                    "port_inside": port_cells[0],
                    "first_outside": first_outside,
                }
            )

    sources.sort(key=lambda item: tuple(sorted(item["cells"])[0]))
    for idx, source in enumerate(sources):
        source["id"] = idx
    return sources


def _find_terminals(grid: np.ndarray, sources: list[dict[str, object]]) -> list[dict[str, object]]:
    source_cells = {cell for source in sources for cell in source["cells"]}
    height, width = grid.shape
    terminals: list[dict[str, object]] = []
    used_cells: set[Coord] = set()

    def maybe_add(cells: set[Coord]) -> None:
        if cells & source_cells or cells & used_cells:
            return
        first = next(iter(cells))
        color = int(grid[first])
        if color not in COLOR_TO_BASIS:
            return
        if any(int(grid[cell]) != color for cell in cells):
            return
        outside = _external_neighbors(grid, cells)
        if len(outside) != 1:
            return
        first_outside = next(iter(outside))
        if int(grid[first_outside]) != color:
            return
        port_cells = [cell for cell in cells if first_outside in _neighbors(*cell)]
        if len(port_cells) != 1:
            return
        used_cells.update(cells)
        terminals.append(
            {
                "id": len(terminals),
                "color": color,
                "cells": frozenset(cells),
                "port_inside": port_cells[0],
                "first_outside": first_outside,
            }
        )

    for r in range(height):
        for c in range(width - 1):
            maybe_add({(r, c), (r, c + 1)})
    for r in range(height - 1):
        for c in range(width):
            maybe_add({(r, c), (r + 1, c)})

    terminals.sort(key=lambda item: tuple(sorted(item["cells"])[0]))
    for idx, terminal in enumerate(terminals):
        terminal["id"] = idx
    return terminals


def _region_lookup(regions: list[dict[str, object]]) -> dict[Coord, int]:
    lookup: dict[Coord, int] = {}
    for region in regions:
        for cell in region["cells"]:
            assert cell not in lookup, "overlapping marker regions are not allowed"
            lookup[cell] = int(region["id"])
    return lookup


def _crossing_color(grid: np.ndarray, cell: Coord, direction: Coord) -> int:
    r, c = cell
    dr, dc = direction
    if dr != 0:
        perpendicular = [(r, c - 1), (r, c + 1)]
    else:
        perpendicular = [(r - 1, c), (r + 1, c)]
    colors = {int(grid[pos]) for pos in perpendicular if _occupied(grid, pos)}
    assert len(colors) == 1, f"crossing at {cell} must have one perpendicular color, got {colors}"
    other_color = next(iter(colors))
    assert other_color in COLOR_TO_BASIS and other_color != GRAY, "invalid crossing color"
    return other_color


def _merge_bases(current: set[str], added: frozenset[str]) -> set[str]:
    merged = set(current)
    merged.update(added)
    assert frozenset(merged) in BASIS_TO_COLOR, "unsupported three-color mix"
    return merged


def _validate_visit_counts(
    grid: np.ndarray,
    sources: list[dict[str, object]],
    terminals: list[dict[str, object]],
    visit_counts: dict[Coord, int],
) -> None:
    source_cells = {cell for source in sources for cell in source["cells"]}
    terminal_cells = {cell for terminal in terminals for cell in terminal["cells"]}

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) == BG or (r, c) in source_cells:
                continue
            expected = 2 if int(grid[r, c]) == GRAY else 1
            if (r, c) in terminal_cells:
                expected = 1
            actual = visit_counts.get((r, c), 0)
            assert actual == expected, f"unexpected visit count at {(r, c)}: expected {expected}, got {actual}"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid puzzle instance")


def _gen_attempt() -> np.ndarray | None:
    scenario = random.choice(SCENARIOS)

    if scenario == "isolated":
        axis = random.choice(("h", "v"))
        count = random.randint(2, 4)
        n_h = count if axis == "h" else 0
        n_v = count if axis == "v" else 0
        h_colors = [random.choice(PRIMARY_COLORS) for _ in range(n_h)]
        v_colors = [random.choice(PRIMARY_COLORS) for _ in range(n_v)]
    elif scenario == "same_cross":
        color = random.choice(PRIMARY_COLORS)
        n_h, n_v = 1, 1
        h_colors = [color]
        v_colors = [color]
    elif scenario == "mixed_cross":
        a, b = random.choice(PRIMARY_PAIRS)
        n_h, n_v = 1, 1
        if random.random() < 0.5:
            h_colors, v_colors = [a], [b]
        else:
            h_colors, v_colors = [b], [a]
    else:
        n_h, n_v = random.choice(((1, 2), (2, 1), (2, 2), (1, 3), (3, 1)))
        a, b = random.choice(PRIMARY_PAIRS)
        templates = [
            ([a] * n_h, [b] * n_v),
            ([a] * n_h, [a] + [b] * (n_v - 1)) if n_v >= 2 else None,
            ([a] + [b] * (n_h - 1), [b] * n_v) if n_h >= 2 else None,
            ([a] + [b] * (n_h - 1), [a] + [b] * (n_v - 1)),
        ]
        valid_templates = [tpl for tpl in templates if tpl is not None]
        h_colors, v_colors = random.choice(valid_templates)

    min_height = max(9, 9 + 2 * max(0, n_h - 1))
    min_width = max(9, 9 + 2 * max(0, n_v - 1))
    max_height = min(28, min_height + 8)
    max_width = min(28, min_width + 8)
    height = random.randint(min_height, max_height)
    width = random.randint(min_width, max_width)

    h_rows = _sample_positions(4, height - 4, n_h, MIN_LINE_GAP)
    v_cols = _sample_positions(4, width - 4, n_v, MIN_LINE_GAP)
    if h_rows is None or v_cols is None:
        return None

    paths: list[dict[str, object]] = []
    for row, color in zip(h_rows, h_colors):
        paths.append(_make_horizontal_path(row, width, color, source_left=bool(random.getrandbits(1))))
    for col, color in zip(v_cols, v_colors):
        paths.append(_make_vertical_path(col, height, color, source_top=bool(random.getrandbits(1))))

    grid = _render_grid(height, width, paths)
    if not _scenario_is_valid(scenario, grid, h_colors, v_colors):
        return None

    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def _sample_positions(low: int, high: int, count: int, min_gap: int) -> list[int] | None:
    if count == 0:
        return []
    choices = list(range(low, high + 1))
    if len(choices) < count:
        return None
    for _ in range(POSITION_RETRIES):
        picked = sorted(random.sample(choices, count))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _make_horizontal_path(row: int, width: int, color: int, source_left: bool) -> dict[str, object]:
    if source_left:
        source = frozenset({(row - 1, 1), (row - 1, 2), (row, 1), (row, 2)})
        terminal = frozenset({(row - 1, width - 2), (row, width - 2)})
        body = tuple((row, c) for c in range(3, width - 2))
    else:
        source = frozenset({(row - 1, width - 3), (row - 1, width - 2), (row, width - 3), (row, width - 2)})
        terminal = frozenset({(row - 1, 1), (row, 1)})
        body = tuple((row, c) for c in range(2, width - 3))
    return {
        "axis": "h",
        "line": row,
        "color": color,
        "source": source,
        "terminal": terminal,
        "body": body,
        "body_set": frozenset(body),
    }


def _make_vertical_path(col: int, height: int, color: int, source_top: bool) -> dict[str, object]:
    if source_top:
        source = frozenset({(1, col - 1), (1, col), (2, col - 1), (2, col)})
        terminal = frozenset({(height - 2, col - 1), (height - 2, col)})
        body = tuple((r, col) for r in range(3, height - 2))
    else:
        source = frozenset({(height - 3, col - 1), (height - 3, col), (height - 2, col - 1), (height - 2, col)})
        terminal = frozenset({(1, col - 1), (1, col)})
        body = tuple((r, col) for r in range(2, height - 3))
    return {
        "axis": "v",
        "line": col,
        "color": color,
        "source": source,
        "terminal": terminal,
        "body": body,
        "body_set": frozenset(body),
    }


def _render_grid(height: int, width: int, paths: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    body_axis_at: dict[Coord, str] = {}

    for path in paths:
        color = int(path["color"])
        for cell in path["source"] | path["terminal"]:
            assert int(grid[cell]) == BG, f"marker overlap at {cell}"
            grid[cell] = color
        for cell in path["body"]:
            if int(grid[cell]) == BG:
                grid[cell] = color
                body_axis_at[cell] = str(path["axis"])
                continue
            assert cell in body_axis_at, f"body unexpectedly overlaps a marker at {cell}"
            assert body_axis_at[cell] != path["axis"], f"same-axis body overlap at {cell}"

    horizontals = [path for path in paths if path["axis"] == "h"]
    verticals = [path for path in paths if path["axis"] == "v"]
    for h_path in horizontals:
        for v_path in verticals:
            cross = (int(h_path["line"]), int(v_path["line"]))
            if cross in h_path["body_set"] and cross in v_path["body_set"]:
                grid[cross] = GRAY
    return grid


def _scenario_is_valid(
    scenario: str,
    grid: np.ndarray,
    h_colors: list[int],
    v_colors: list[int],
) -> bool:
    gray_count = int(np.count_nonzero(grid == GRAY))
    if scenario == "isolated":
        return gray_count == 0
    if scenario == "same_cross":
        return gray_count == 1 and h_colors[0] == v_colors[0]
    if scenario == "mixed_cross":
        return gray_count == 1 and h_colors[0] != v_colors[0]
    if gray_count < 2:
        return False
    return max(len(h_colors), len(v_colors)) >= 2 and any(h != v for h in h_colors for v in v_colors)
