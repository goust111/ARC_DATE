from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BLUE = 1
MAROON = 9
INPUT_COLORS = (2, 3, 4, 7)
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GAP_CELLS = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}
SIDE_CELLS = tuple(GAP_CELLS.values())
CORNER_CELLS = ((0, 0), (0, 2), (2, 0), (2, 2))

GEN_ATTEMPTS = 400
GEN_SIZE_RANGE = (4, 7)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return components


def _decode_symbol(grid: np.ndarray, component: dict, bg: int) -> dict:
    color = int(component["color"])
    coords = component["coords"]
    r0, c0, r1, c1 = component["bbox"]
    assert len(coords) == 7, "each symbol must have 7 colored pixels"
    assert r1 - r0 + 1 == 3 and c1 - c0 + 1 == 3, "each symbol must occupy a 3x3 box"

    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    assert int(patch[1, 1]) == bg, "center must be hollow"

    for rr, cc in CORNER_CELLS:
        assert int(patch[rr, cc]) == color, "corners must remain present"

    missing_dirs = [name for name, (rr, cc) in GAP_CELLS.items() if int(patch[rr, cc]) == bg]
    assert len(missing_dirs) == 1, "exactly one side-center gap is required"
    gap_dir = missing_dirs[0]

    for name, (rr, cc) in GAP_CELLS.items():
        if name == gap_dir:
            continue
        assert int(patch[rr, cc]) == color, "all non-gap side centers must be present"

    non_bg = patch[patch != bg]
    assert non_bg.size == 7 and np.all(non_bg == color), "symbol must be monochrome"
    return {
        "color": color,
        "input_top_left": (r0, c0),
        "dir_name": gap_dir,
        "dir": DIR_VECS[gap_dir],
    }


def _extract_symbols(grid: np.ndarray) -> tuple[list[dict], int, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_color_components(g, bg)
    assert components, "expected at least one symbol"

    symbols = [_decode_symbol(g, comp, bg) for comp in components]
    all_coords = {coord for comp in components for coord in comp["coords"]}
    non_bg_coords = {tuple(map(int, rc)) for rc in np.argwhere(g != bg)}
    assert all_coords == non_bg_coords, "all foreground pixels must belong to symbols"

    rows = sorted({sym["input_top_left"][0] for sym in symbols})
    cols = sorted({sym["input_top_left"][1] for sym in symbols})
    assert rows and cols
    assert all(b - a == 4 for a, b in zip(rows, rows[1:])), "symbol rows must follow the 4x4 coarse lattice"
    assert all(b - a == 4 for a, b in zip(cols, cols[1:])), "symbol cols must follow the 4x4 coarse lattice"

    row_index = {r: i for i, r in enumerate(rows)}
    col_index = {c: i for i, c in enumerate(cols)}
    for sym in symbols:
        rr = row_index[sym["input_top_left"][0]]
        cc = col_index[sym["input_top_left"][1]]
        sym["out_rc"] = (rr, cc)

    return symbols, bg, (len(rows), len(cols))


def _simulate(symbols: list[dict], out_shape: tuple[int, int], bg: int = BG) -> tuple[np.ndarray, dict]:
    h, w = out_shape
    out = np.full((h, w), bg, dtype=int)
    point_positions = {tuple(sym["out_rc"]) for sym in symbols}
    point_lookup = {tuple(sym["out_rc"]): sym for sym in symbols}
    order = sorted(symbols, key=lambda sym: (sym["out_rc"][0], sym["out_rc"][1]))
    for sym in symbols:
        rr, cc = sym["out_rc"]
        out[rr, cc] = int(sym["color"])

    visits: dict[tuple[int, int], list[tuple[int, int]]] = {}
    hits: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for sym in order:
        sr, sc = sym["out_rc"]
        dr, dc = sym["dir"]
        color = int(sym["color"])
        r, c = sr, sc
        while True:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                break
            if (nr, nc) in point_positions and (nr, nc) != (sr, sc):
                out[nr, nc] = MAROON
                hits.append(((sr, sc), (nr, nc)))
                break
            visits.setdefault((nr, nc), []).append((sr, sc))
            if len(visits[(nr, nc)]) >= 2:
                out[nr, nc] = BLUE
            else:
                out[nr, nc] = color
            r, c = nr, nc

    return out, {"visits": visits, "hits": hits, "points": point_lookup}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each hollow symbol to one coarse-grid pixel, then emit rays from the gap direction.

    Input structure:
    - The background is black and the foreground consists of monochrome hollow 3x3 border symbols.
    - Each symbol has a black center and exactly one missing side-center border pixel, so the gap points up, down, left, or right.
    - Symbols sit on a 4x4 coarse lattice: occupied 3x3 boxes are separated by one row and one column of background.

    Rule:
    - Replace every hollow 3x3 symbol with one coarse-grid pixel of the same color at the symbol's lattice cell; the output size is the occupied coarse-grid size.
    - The symbol's missing side-center determines the ray direction.
    - Process compressed pixels in row-major order. A ray advances cell by cell until it leaves the output grid or hits another compressed pixel.
    - Empty cells take the ray color, a later ray entering an already colored ray cell turns it blue, blue stays blue, and a ray that hits another compressed pixel recolors that pixel maroon and stops.

    Key insight:
    - The same hollow symbol encodes both the compressed point and the ray orientation, so solving requires extracting direction from the gap before simulating the coarse-grid ray interactions.
    """
    symbols, bg, out_shape = _extract_symbols(np.asarray(grid, dtype=int))
    out, _meta = _simulate(symbols, out_shape, bg=bg)
    return out


def _path_cells(start: tuple[int, int], dir_name: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = DIR_VECS[dir_name]
    h, w = shape
    r, c = start
    cells: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if r < 0 or r >= h or c < 0 or c >= w:
            return cells
        cells.append((r, c))


def _pair_corridor(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        row = a[0]
        c0, c1 = sorted((a[1], b[1]))
        return [(row, c) for c in range(c0 + 1, c1)]
    col = a[1]
    r0, r1 = sorted((a[0], b[0]))
    return [(r, col) for r in range(r0 + 1, r1)]


def _render_symbol(color: int, dir_name: str) -> np.ndarray:
    patch = np.full((3, 3), color, dtype=int)
    patch[1, 1] = BG
    gr, gc = GAP_CELLS[dir_name]
    patch[gr, gc] = BG
    return patch


def _render_layout(rows: int, cols: int, items: list[dict]) -> np.ndarray:
    grid = np.zeros((4 * rows - 1, 4 * cols - 1), dtype=int)
    for item in items:
        r, c = item["pos"]
        patch = _render_symbol(int(item["color"]), item["dir_name"])
        r0 = 4 * r
        c0 = 4 * c
        grid[r0 : r0 + 3, c0 : c0 + 3] = patch
    return grid


def _choose_pair(rng: random.Random, rows: int, cols: int) -> tuple[tuple[int, int], tuple[int, int], dict[tuple[int, int], str]]:
    orientations = []
    if cols >= 3:
        orientations.append("h")
    if rows >= 3:
        orientations.append("v")
    orient = rng.choice(orientations)
    if orient == "h":
        row = rng.randrange(rows)
        c0 = rng.randrange(cols - 2)
        c1 = rng.randrange(c0 + 2, cols)
        a = (row, c0)
        b = (row, c1)
        return a, b, {a: "right", b: "left"}
    col = rng.randrange(cols)
    r0 = rng.randrange(rows - 2)
    r1 = rng.randrange(r0 + 2, rows)
    a = (r0, col)
    b = (r1, col)
    return a, b, {a: "down", b: "up"}


def _has_safe_dir(
    pos: tuple[int, int],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
    forbidden_paths: set[tuple[int, int]],
) -> bool:
    for dir_name in DIR_VECS:
        path = _path_cells(pos, dir_name, shape)
        if not path:
            continue
        if any(cell in occupied for cell in path):
            continue
        if any(cell in forbidden_paths for cell in path):
            continue
        return True
    return False


def _choose_positions(
    rng: random.Random,
    rows: int,
    cols: int,
    pair_positions: tuple[tuple[int, int], tuple[int, int]],
) -> list[tuple[int, int]]:
    shape = (rows, cols)
    corridor = set(_pair_corridor(pair_positions[0], pair_positions[1]))
    positions = set(pair_positions)

    while True:
        covered_rows = {r for r, _ in positions}
        covered_cols = {c for _, c in positions}
        uncovered_rows = set(range(rows)) - covered_rows
        uncovered_cols = set(range(cols)) - covered_cols
        if not uncovered_rows and not uncovered_cols and len(positions) >= 4:
            break

        best_score = None
        choices: list[tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                pos = (r, c)
                if pos in positions:
                    continue
                if not _has_safe_dir(pos, positions, shape, corridor):
                    continue
                score = (
                    3 * int(r in uncovered_rows) + 3 * int(c in uncovered_cols),
                    int(r not in (0, rows - 1) or c not in (0, cols - 1)),
                    int(r in (1, rows - 2) or c in (1, cols - 2)),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    choices = [pos]
                elif score == best_score:
                    choices.append(pos)
        if not choices:
            raise RuntimeError("failed to cover all coarse rows and cols")
        positions.add(rng.choice(choices))
    return sorted(positions)


def _assign_dirs(
    rng: random.Random,
    positions: list[tuple[int, int]],
    fixed_dirs: dict[tuple[int, int], str],
    shape: tuple[int, int],
    pair_positions: tuple[tuple[int, int], tuple[int, int]],
) -> dict[tuple[int, int], str] | None:
    dirs = dict(fixed_dirs)
    used_paths = set(_pair_corridor(pair_positions[0], pair_positions[1]))
    occupied = set(positions)
    fillers = [pos for pos in positions if pos not in dirs]
    rng.shuffle(fillers)

    for pos in fillers:
        candidates = []
        for dir_name in DIR_VECS:
            path = _path_cells(pos, dir_name, shape)
            if not path:
                continue
            if any(cell in occupied for cell in path):
                continue
            overlap = len(used_paths.intersection(path))
            if overlap:
                continue
            candidates.append((len(path), rng.random(), dir_name, path))
        if not candidates:
            return None
        _, _, dir_name, path = max(candidates)
        dirs[pos] = dir_name
        used_paths.update(path)
    return dirs


def _assign_colors(rng: random.Random, positions: list[tuple[int, int]], pair_positions: tuple[tuple[int, int], tuple[int, int]]) -> dict[tuple[int, int], int]:
    colors = {pos: rng.choice(INPUT_COLORS) for pos in positions}
    if rng.random() < 0.75:
        colors[pair_positions[0]], colors[pair_positions[1]] = rng.sample(INPUT_COLORS, 2)
    if len(set(colors.values())) == 1:
        one = positions[0]
        replacement = [color for color in INPUT_COLORS if color != colors[one]][0]
        colors[one] = replacement
    return colors


def _layout_items(
    positions: list[tuple[int, int]],
    dirs: dict[tuple[int, int], str],
    colors: dict[tuple[int, int], int],
    pair_positions: tuple[tuple[int, int], tuple[int, int]],
) -> list[dict]:
    pair_set = set(pair_positions)
    items = []
    for pos in positions:
        items.append(
            {
                "pos": pos,
                "dir_name": dirs[pos],
                "color": int(colors[pos]),
                "role": "pair" if pos in pair_set else "filler",
            }
        )
    return items


def _validate_generated(rows: int, cols: int, items: list[dict]) -> bool:
    grid = _render_layout(rows, cols, items)
    parsed, bg, out_shape = _extract_symbols(grid)
    if bg != BG or out_shape != (rows, cols) or len(parsed) != len(items):
        return False

    out, meta = _simulate(parsed, out_shape, bg=bg)
    points = {tuple(sym["out_rc"]) for sym in parsed}
    blue_cells = {cell for cell, shooters in meta["visits"].items() if len(shooters) >= 2}
    colored_trails = [
        cell
        for cell in meta["visits"]
        if cell not in points and int(out[cell]) in INPUT_COLORS
    ]

    pair_positions = [item["pos"] for item in items if item["role"] == "pair"]
    assert len(pair_positions) == 2
    pair_a, pair_b = pair_positions
    corridor = set(_pair_corridor(pair_a, pair_b))
    hit_targets = {dst for _src, dst in meta["hits"]}

    if {r for r, _ in points} != set(range(rows)):
        return False
    if {c for _, c in points} != set(range(cols)):
        return False
    if len(points) >= rows * cols:
        return False
    if blue_cells != corridor:
        return False
    if hit_targets != {pair_a, pair_b} or len(meta["hits"]) != 2:
        return False
    if int(out[pair_a]) != MAROON or int(out[pair_b]) != MAROON:
        return False
    if not colored_trails:
        return False

    mutated = [dict(item) for item in items]
    for item in mutated:
        if item["role"] == "pair":
            alternatives = [name for name in DIR_VECS if name != item["dir_name"]]
            item["dir_name"] = alternatives[0]
            break
    changed = main(_render_layout(rows, cols, mutated))
    if np.array_equal(out, changed):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rows = random.randint(*GEN_SIZE_RANGE)
        cols = random.randint(*GEN_SIZE_RANGE)
        try:
            pair_a, pair_b, fixed_dirs = _choose_pair(random, rows, cols)
            positions = _choose_positions(random, rows, cols, (pair_a, pair_b))
        except RuntimeError:
            continue
        dirs = _assign_dirs(random, positions, fixed_dirs, (rows, cols), (pair_a, pair_b))
        if dirs is None:
            continue
        colors = _assign_colors(random, positions, (pair_a, pair_b))
        items = _layout_items(positions, dirs, colors, (pair_a, pair_b))
        if not _validate_generated(rows, cols, items):
            continue
        return _render_layout(rows, cols, items)
    raise RuntimeError("failed to generate a valid hollow-symbol ray puzzle")
