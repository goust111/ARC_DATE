from __future__ import annotations

import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
GREY = 5
ALL_COLORS = list(range(10))
GEN_ATTEMPTS = 300


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr, dc in DIR8:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_palette(grid: np.ndarray) -> tuple[int, list[int]]:
    bg = _mode_color(grid)
    palette: list[int] = []
    for value in map(int, grid[0]):
        if value == bg:
            break
        palette.append(value)
    assert palette, "expected a non-empty palette run in the top row"
    assert len(set(palette)) == len(palette), "palette colors must be unique"
    assert np.all(grid[0, len(palette) :] == bg), "top row suffix must be background"
    return bg, palette


def _ordered_path(grid: np.ndarray, comp: list[tuple[int, int]], palette_set: set[int]) -> list[tuple[int, int]]:
    comp_set = set(comp)
    degrees = {
        cell: sum((nbr in comp_set) for nbr in _neighbors4(*cell))
        for cell in comp
    }
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "each component must be a simple path"

    seed_ends = [cell for cell in endpoints if int(grid[cell]) in palette_set]
    assert len(seed_ends) == 1, "each path must have exactly one colored endpoint"
    start = seed_ends[0]

    ordered = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [nbr for nbr in _neighbors4(*cur) if nbr in comp_set and nbr != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "simple path should have one forward continuation"
        prev, cur = cur, nxts[0]
        ordered.append(cur)
    assert len(ordered) == len(comp), "path walk must cover the whole component"
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey snake path by walking from its colored endpoint and cycling through the top-row palette.

    Input structure:
    - The left side of the top row is a contiguous palette of distinct colors, and the rest of that row is background.
    - Below it are one or more non-touching one-cell-wide grey simple paths that may bend but never branch.
    - Each path has exactly one endpoint recolored to one of the palette colors.

    Rule:
    - For each path, start at its colored endpoint and trace the path cell by cell.
    - Keep the endpoint color, then assign the next palette color at each subsequent step, wrapping cyclically after the last palette entry.
    - Leave every non-path cell, including the palette row, unchanged.

    Key insight:
    - The color at each cell depends on arc length along the traced path, not on geometric position; the seed endpoint only chooses the starting phase in the cyclic palette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, palette = _parse_palette(g)
    palette_set = set(palette)
    palette_index = {color: idx for idx, color in enumerate(palette)}

    field_mask = g != bg
    field_mask[0, :] = False
    comps = _components(field_mask)
    assert comps, "expected at least one snake path"

    out = g.copy()
    for comp in comps:
        ordered = _ordered_path(g, comp, palette_set)
        seed_color = int(g[ordered[0]])
        start_idx = palette_index[seed_color]
        for step, (r, c) in enumerate(ordered):
            out[r, c] = palette[(start_idx + step) % len(palette)]
    return out


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        turns += (dr1, dc1) != (dr2, dc2)
    return turns


def _cell_clear(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = cell
    if cell in occupied:
        return False
    return all((nr, nc) not in occupied for nr, nc in _neighbors8(r, c))


def _expand_nodes(
    nodes: list[tuple[int, int]],
    row_pos: list[int],
    col_pos: list[int],
) -> list[tuple[int, int]]:
    def actual(node: tuple[int, int]) -> tuple[int, int]:
        return row_pos[node[0]], col_pos[node[1]]

    cells = [actual(nodes[0])]
    for a, b in zip(nodes, nodes[1:]):
        ar, ac = actual(a)
        br, bc = actual(b)
        cells.append(((ar + br) // 2, (ac + bc) // 2))
        cells.append((br, bc))
    return cells


def _build_path_nodes(
    n_rows: int,
    n_cols: int,
    row_pos: list[int],
    col_pos: list[int],
    occupied: set[tuple[int, int]],
    target_nodes: int,
    min_turns: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    starts = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    rng.shuffle(starts)

    def step_cells(cur: tuple[int, int], nxt: tuple[int, int]) -> list[tuple[int, int]]:
        return _expand_nodes([cur, nxt], row_pos, col_pos)[1:]

    def dfs(
        path: list[tuple[int, int]],
        visited: set[tuple[int, int]],
        turns: int,
        prev_dir: tuple[int, int] | None,
    ) -> list[tuple[int, int]] | None:
        if len(path) == target_nodes:
            return path if turns >= min_turns else None

        remaining_edges = target_nodes - len(path)
        if turns + remaining_edges < min_turns:
            return None

        cur = path[-1]
        candidates: list[tuple[float, tuple[int, int], int, tuple[int, int]]] = []
        for dr, dc in DIR4:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < n_rows and 0 <= nxt[1] < n_cols):
                continue
            if nxt in visited:
                continue
            new_cells = step_cells(cur, nxt)
            if not all(_cell_clear(cell, occupied) for cell in new_cells):
                continue
            extra_turn = int(prev_dir is not None and prev_dir != (dr, dc))
            score = rng.random()
            if turns < min_turns and extra_turn:
                score -= 0.4
            if turns < min_turns and not extra_turn:
                score += 0.15
            candidates.append((score, nxt, turns + extra_turn, (dr, dc)))

        candidates.sort(key=lambda item: item[0])
        for _, nxt, nxt_turns, nxt_dir in candidates:
            visited.add(nxt)
            path.append(nxt)
            result = dfs(path, visited, nxt_turns, nxt_dir)
            if result is not None:
                return result
            path.pop()
            visited.remove(nxt)
        return None

    for start in starts:
        anchor = _expand_nodes([start], row_pos, col_pos)[0]
        if not _cell_clear(anchor, occupied):
            continue
        result = dfs([start], {start}, 0, None)
        if result is not None:
            return result
    return None


def _path_specs(palette_len: int, path_count: int, rng: random.Random) -> list[tuple[int, int]]:
    specs: list[tuple[int, int]] = []
    wrap_nodes = max(4, (palette_len + 3) // 2)
    specs.append((rng.randint(wrap_nodes, wrap_nodes + 2), 2))
    for _ in range(path_count - 1):
        nodes = rng.randint(2, 6)
        min_turns = rng.choice([0, 1, 2])
        if nodes < 4:
            min_turns = min(min_turns, 1)
        specs.append((nodes, min_turns))
    specs.sort(key=lambda item: (-item[0], -item[1]))
    return specs


def _validate_candidate(grid: np.ndarray) -> None:
    bg, palette = _parse_palette(grid)
    palette_set = set(palette)

    field_mask = grid != bg
    field_mask[0, :] = False
    comps = _components(field_mask)
    assert 1 <= len(comps) <= 4

    field_colors = {int(v) for v in grid[1:][grid[1:] != bg]}
    assert len(field_colors - palette_set) == 1, "expected one shared grey path color"
    grey = next(iter(field_colors - palette_set))
    assert grey == GREY, "generator keeps path bodies grey for visual consistency"

    comp_index = {}
    path_cells: set[tuple[int, int]] = set()
    for idx, comp in enumerate(comps):
        ordered = _ordered_path(grid, comp, palette_set)
        path_cells.update(comp)
        for cell in comp:
            comp_index[cell] = idx
        colors = [int(grid[cell]) for cell in comp]
        assert sum(color in palette_set for color in colors) == 1
        assert all(color == grey for color in colors[1:])
        if len(comp) > len(palette):
            break
    else:
        raise AssertionError("expected at least one path long enough to wrap the palette")

    for cell, idx in comp_index.items():
        for nbr in _neighbors8(*cell):
            if nbr in comp_index:
                assert comp_index[nbr] == idx or nbr in _neighbors4(*cell)

    out = main(grid)
    assert np.array_equal(out[0], grid[0]), "palette row must stay unchanged"
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if (r, c) not in path_cells and r != 0:
                assert int(out[r, c]) == int(grid[r, c]), "non-path cells must remain unchanged"

    palette_index = {color: idx for idx, color in enumerate(palette)}
    for comp in comps:
        ordered = _ordered_path(grid, comp, palette_set)
        start_idx = palette_index[int(grid[ordered[0]])]
        for step, cell in enumerate(ordered):
            expect = palette[(start_idx + step) % len(palette)]
            assert int(out[cell]) == expect


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        bg = rng.choice([0, 1, 8, 9])
        palette_pool = [color for color in ALL_COLORS if color not in {bg, GREY}]
        palette_len = rng.randint(3, 6)
        palette = rng.sample(palette_pool, palette_len)

        path_count = rng.choice([1, 2, 2, 3, 3, 4])
        coarse_rows = rng.randint(4 + (path_count >= 3), 9)
        coarse_cols = rng.randint(max(5, path_count + 2), 13)
        height = 2 * coarse_rows + 2
        width = 2 * coarse_cols + 1
        if height > 30 or width > 30 or palette_len >= width:
            continue

        row_pos = list(range(2, 2 * coarse_rows + 1, 2))
        col_pos = list(range(1, 2 * coarse_cols, 2))
        specs = _path_specs(palette_len, path_count, rng)

        occupied: set[tuple[int, int]] = set()
        actual_paths: list[list[tuple[int, int]]] = []
        ok = True
        for target_nodes, min_turns in specs:
            target_nodes = min(target_nodes, len(row_pos) * len(col_pos))
            nodes = _build_path_nodes(
                len(row_pos),
                len(col_pos),
                row_pos,
                col_pos,
                occupied,
                target_nodes,
                min_turns,
                rng,
            )
            if nodes is None:
                ok = False
                break
            cells = _expand_nodes(nodes, row_pos, col_pos)
            if _turn_count(cells) < min_turns:
                ok = False
                break
            actual_paths.append(cells)
            occupied.update(cells)
        if not ok:
            continue

        grid = np.full((height, width), bg, dtype=int)
        grid[0, :palette_len] = palette

        if path_count <= palette_len:
            seed_colors = rng.sample(palette, path_count)
        else:
            seed_colors = [rng.choice(palette) for _ in range(path_count)]

        for cells, seed_color in zip(actual_paths, seed_colors):
            ordered = cells if rng.random() < 0.5 else list(reversed(cells))
            for r, c in ordered:
                grid[r, c] = GREY
            grid[ordered[0]] = seed_color

        try:
            _validate_candidate(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid snake-phase recoloring input")
