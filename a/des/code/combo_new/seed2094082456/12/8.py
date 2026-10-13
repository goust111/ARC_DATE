from __future__ import annotations

import random

import numpy as np


BG = 0
GREEN = 3
PORTAL_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GEN_ATTEMPTS = 400


def _in_bounds(r: int, c: int, shape: tuple[int, int]) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _on_boundary(r: int, c: int, shape: tuple[int, int]) -> bool:
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _trace_green_path(grid: np.ndarray) -> list[tuple[int, int]]:
    coords = {tuple(map(int, rc)) for rc in np.argwhere(grid == GREEN)}
    assert coords, "expected a green path"

    degrees: dict[tuple[int, int], int] = {}
    for r, c in coords:
        degree = 0
        for dr, dc in DIR4:
            if (r + dr, c + dc) in coords:
                degree += 1
        assert degree <= 2, "green path must not branch"
        degrees[(r, c)] = degree

    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "green component must be a simple path"
    boundary_endpoints = [cell for cell in endpoints if _on_boundary(cell[0], cell[1], grid.shape)]
    assert len(boundary_endpoints) == 1, "exactly one green endpoint must lie on the boundary"

    start = boundary_endpoints[0]
    ordered = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = []
        for dr, dc in DIR4:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in coords and nxt != prev:
                nxts.append(nxt)
        if not nxts:
            break
        assert len(nxts) == 1, "green path order must be unambiguous"
        prev, cur = cur, nxts[0]
        ordered.append(cur)

    assert len(ordered) == len(coords), "green cells must form a single connected path"
    assert len(ordered) >= 2, "path must have at least two cells"
    return ordered


def _portal_pairs(grid: np.ndarray) -> tuple[dict[int, tuple[tuple[int, int], tuple[int, int]]], dict[tuple[int, int], int]]:
    pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    by_coord: dict[tuple[int, int], int] = {}

    for color in sorted(int(v) for v in np.unique(grid)):
        if color in (BG, GREEN):
            continue
        pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        assert len(pts) == 2, "every non-black non-green color must form one portal pair"
        assert pts[0] != pts[1]
        assert abs(pts[0][0] - pts[1][0]) + abs(pts[0][1] - pts[1][1]) != 1, "portal twins must be isolated single pixels"
        pairs[color] = (pts[0], pts[1])
        by_coord[pts[0]] = color
        by_coord[pts[1]] = color

    assert pairs, "expected at least one portal pair"
    return pairs, by_coord


def main(grid: np.ndarray) -> np.ndarray:
    """Extend the green path through matched-color teleport portals while preserving direction.

    Input structure:
    - The background is Black.
    - One 4-connected Green path of width 1 starts at the grid boundary and ends inside the grid.
    - Several other colors appear as isolated single-pixel pairs, and the path's final Green pixel faces one member of such a pair.

    Rule:
    - Trace the Green path from its boundary endpoint to its internal endpoint and read the travel direction from the last two Green pixels.
    - When the cell directly ahead of the current path endpoint is a portal color C, jump to C's matching pixel.
    - From the matched portal, keep drawing Green one cell at a time in the same direction until the next step would leave the grid or hit any non-Black cell; if the new endpoint again faces a portal, repeat the teleport.
    - Portal pixels themselves never change; only new Green pixels are added.

    Key insight:
    The paired same-color singletons act as teleport entrances and exits, and the path's direction is carried through every jump, so only the forward-facing portal matters at each stage.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    ordered = _trace_green_path(g)
    direction = (
        ordered[-1][0] - ordered[-2][0],
        ordered[-1][1] - ordered[-2][1],
    )
    assert direction in DIR4

    pairs, by_coord = _portal_pairs(g)
    out = g.copy()
    endpoint = ordered[-1]
    used_colors: set[int] = set()

    while True:
        ahead = (endpoint[0] + direction[0], endpoint[1] + direction[1])
        if not _in_bounds(ahead[0], ahead[1], out.shape):
            break

        ahead_val = int(out[ahead])
        if ahead_val in (BG, GREEN):
            break

        color = by_coord[ahead]
        assert color == ahead_val
        assert color not in used_colors, "portal chain should not loop back to a used color"
        used_colors.add(color)

        a, b = pairs[color]
        exit_portal = b if ahead == a else a
        cursor = exit_portal
        drew = False

        while True:
            nxt = (cursor[0] + direction[0], cursor[1] + direction[1])
            if not _in_bounds(nxt[0], nxt[1], out.shape):
                break
            if int(out[nxt]) != BG:
                break
            out[nxt] = GREEN
            endpoint = nxt
            cursor = nxt
            drew = True
            if _on_boundary(nxt[0], nxt[1], out.shape):
                break

        if not drew or _on_boundary(endpoint[0], endpoint[1], out.shape):
            break

    return out


def _build_initial_path(height: int, width: int, end_row: int, end_col: int) -> list[tuple[int, int]] | None:
    del width
    max_turn_col = min(2, end_col - 1)
    if max_turn_col < 1:
        return None

    turn_col = random.randint(1, max_turn_col)
    start_side = random.choice(("top", "bottom", "left"))
    path: list[tuple[int, int]] = []

    if start_side == "top":
        path.extend((r, turn_col) for r in range(0, end_row + 1))
        path.extend((end_row, c) for c in range(turn_col + 1, end_col + 1))
        return path

    if start_side == "bottom":
        path.extend((r, turn_col) for r in range(height - 1, end_row - 1, -1))
        path.extend((end_row, c) for c in range(turn_col + 1, end_col + 1))
        return path

    if random.random() < 0.35:
        start_row = end_row
    else:
        choices = [r for r in range(1, height - 1) if r != end_row]
        start_row = random.choice(choices)

    path.extend((start_row, c) for c in range(0, turn_col + 1))
    if start_row != end_row:
        step = 1 if end_row > start_row else -1
        path.extend((r, turn_col) for r in range(start_row + step, end_row + step, step))
    path.extend((end_row, c) for c in range(turn_col + 1, end_col + 1))
    return path


def _touch4(cell: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = cell
    touched = {(r, c)}
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if _in_bounds(rr, cc, shape):
            touched.add((rr, cc))
    return touched


def _place_extra_pairs(
    shape: tuple[int, int],
    colors: list[int],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, tuple[int, int], tuple[int, int]]] | None:
    blocked: set[tuple[int, int]] = set()
    for cell in reserved:
        blocked.update(_touch4(cell, shape))

    candidates = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
    random.shuffle(candidates)
    placed: list[tuple[int, tuple[int, int], tuple[int, int]]] = []

    for color in colors:
        chosen: list[tuple[int, int]] = []
        for cell in candidates:
            if cell in blocked:
                continue
            chosen.append(cell)
            blocked.update(_touch4(cell, shape))
            if len(chosen) == 2:
                placed.append((color, chosen[0], chosen[1]))
                break
        if len(chosen) != 2:
            return None

    return placed


def _collect_added_cells(
    ray_rows: list[int],
    exit_cols: list[int],
    entry_cols: list[int],
    width: int,
) -> set[tuple[int, int]]:
    added: set[tuple[int, int]] = set()
    for i in range(len(ray_rows) - 1):
        row = ray_rows[i]
        for col in range(exit_cols[i] + 1, entry_cols[i + 1]):
            added.add((row, col))
    final_row = ray_rows[-1]
    for col in range(exit_cols[-1] + 1, width):
        added.add((final_row, col))
    return added


def _build_canonical() -> np.ndarray | None:
    chain_len = random.randint(1, 4)
    height = random.randint(max(11, 2 * chain_len + 5), 22)
    width = random.randint(12, 24)

    row_choices = list(range(1, height - 1, 2))
    if len(row_choices) < chain_len + 1:
        return None
    special_rows = random.sample(row_choices, chain_len + 1)
    init_row = special_rows[0]
    ray_rows = special_rows[1:]

    entry_cols = [random.randint(6, width - 3)]
    exit_cols: list[int] = []
    for _ in range(chain_len - 1):
        exit_col = random.randint(3, width - 5)
        seg_len = random.randint(1, width - exit_col - 3)
        exit_cols.append(exit_col)
        entry_cols.append(exit_col + seg_len + 1)
    exit_cols.append(random.randint(3, width - 2))

    path = _build_initial_path(height, width, init_row, entry_cols[0] - 1)
    if path is None:
        return None

    extra_pair_count = random.randint(0, 2)
    total_pairs = chain_len + extra_pair_count
    colors = random.sample(PORTAL_COLORS, total_pairs)
    chain_colors = colors[:chain_len]
    extra_colors = colors[chain_len:]

    grid = np.full((height, width), BG, dtype=int)
    reserved_output = set(path)

    for r, c in path:
        grid[r, c] = GREEN

    for i, color in enumerate(chain_colors):
        source = (init_row, entry_cols[0]) if i == 0 else (ray_rows[i - 1], entry_cols[i])
        exit_portal = (ray_rows[i], exit_cols[i])
        assert int(grid[source]) == BG and int(grid[exit_portal]) == BG
        grid[source] = color
        grid[exit_portal] = color
        reserved_output.add(source)
        reserved_output.add(exit_portal)

    reserved_output.update(_collect_added_cells(ray_rows, exit_cols, entry_cols, width))

    extras = _place_extra_pairs(grid.shape, extra_colors, reserved_output)
    if extras is None:
        return None
    for color, a, b in extras:
        assert int(grid[a]) == BG and int(grid[b]) == BG
        grid[a] = color
        grid[b] = color

    return grid


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    if canonical is None:
        return None

    rotated = np.rot90(canonical, random.randrange(4)).copy()
    out = main(rotated)
    if np.array_equal(out, rotated):
        return None
    return rotated


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid portal-chain input")
