import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 300

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

SHAPE_LIBRARY = [
    [(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)],  # T
    [(-1, 0), (0, 0), (1, 0), (-1, 1)],  # L
    [(-1, 0), (0, 0), (1, 0), (1, 1), (0, 1)],  # hook
    [(-1, -1), (-1, 0), (0, 0), (1, 0), (1, 1)],  # zig
    [(-2, 0), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1)],  # tall flag
    [(-1, -1), (-1, 0), (0, 0), (1, 0), (0, 1)],  # bent spear
]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _rotate_coords(coords: list[tuple[int, int]], turns: int) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in coords]
    for _ in range(turns % 4):
        pts = [(c, -r) for r, c in pts]
    return pts


def _components_of_color(grid: np.ndarray, color: int, c0: int = 0) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(c0, w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _separator_column(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    for c in range(w):
        color = int(grid[0, c])
        if color != BG and np.all(grid[:, c] == color):
            return c, color
    raise AssertionError("missing legend separator column")


def _legend_mapping(grid: np.ndarray) -> tuple[int, int, dict[int, list[tuple[int, int]]]]:
    sep_col, sep_color = _separator_column(grid)
    legend = grid[:, :sep_col]
    full_rows = [r for r in range(legend.shape[0]) if legend.shape[1] > 0 and np.all(legend[r, :] == sep_color)]
    bands = []
    start = 0
    for r in full_rows + [legend.shape[0]]:
        if start < r:
            bands.append((start, r))
        start = r + 1

    mapping: dict[int, list[tuple[int, int]]] = {}
    for r0, r1 in bands:
        cell = legend[r0:r1, :]
        pts = np.argwhere((cell != BG) & (cell != sep_color))
        if pts.size == 0:
            continue
        colors = set(int(v) for v in cell[(cell != BG) & (cell != sep_color)])
        assert len(colors) == 1, "legend cell must contain one colored shape"
        color = next(iter(colors))
        center_r = cell.shape[0] // 2
        center_c = cell.shape[1] // 2
        rel = []
        for r, c in np.argwhere(cell == color):
            rel.append((int(r - center_r), int(c - center_c)))
        assert (0, 0) in rel, "legend shape must occupy its center anchor"
        mapping[color] = rel
    assert mapping, "legend mapping is empty"
    return sep_col, sep_color, mapping


def _clearance(grid: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    h, w = grid.shape
    steps = 0
    nr, nc = r + dr, c + dc
    while 0 <= nr < h and 0 <= nc < w:
        if int(grid[nr, nc]) != BG:
            return steps
        steps += 1
        nr += dr
        nc += dc
    return steps


def _best_direction(grid: np.ndarray, r: int, c: int) -> int:
    scores = [_clearance(grid, r, c, dr, dc) for dr, dc in DIRS]
    return max(range(4), key=lambda i: (scores[i], -((i - 0) % 4)))


def _marker_components(grid: np.ndarray, scene_c0: int, mapping: dict[int, list[tuple[int, int]]]) -> list[tuple[int, int, int]]:
    markers = []
    for color in mapping:
        for comp in _components_of_color(grid, color, c0=scene_c0):
            if len(comp) == 1:
                r, c = comp[0]
                markers.append((r, c, color))
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep_col, sep_color, mapping = _legend_mapping(g)
    markers = _marker_components(g, sep_col + 1, mapping)
    out = g.copy()
    for r, c, _color in markers:
        out[r, c] = BG

    painted = np.zeros(g.shape, dtype=bool)
    for r, c, color in markers:
        turns = _best_direction(g, r, c)
        rel = _rotate_coords(mapping[color], turns)
        for dr, dc in rel:
            rr, cc = r + dr, c + dc
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "shape placement out of bounds"
            assert int(g[rr, cc]) != sep_color, "shape may not cover grey structure"
            assert not painted[rr, cc], "shapes may not overlap"
            painted[rr, cc] = True
            out[rr, cc] = color
    return out


def _render_shape(coords: list[tuple[int, int]], color: int, cell_h: int = 7, cell_w: int = 7) -> np.ndarray:
    canvas = np.zeros((cell_h, cell_w), dtype=int)
    cr = cell_h // 2
    cc = cell_w // 2
    for dr, dc in coords:
        canvas[cr + dr, cc + dc] = color
    return canvas


def _add_bar(grid: np.ndarray, rng: random.Random, scene_c0: int) -> None:
    h, w = grid.shape
    orient = rng.choice(["h", "v", "rect"])
    if orient == "h":
        width = rng.randint(4, max(4, w - scene_c0 - 4))
        r = rng.randint(2, h - 3)
        c = rng.randint(scene_c0 + 1, w - width - 2)
        t = rng.randint(1, 2)
        grid[r : r + t, c : c + width] = GREY
    elif orient == "v":
        height = rng.randint(4, max(4, h - 4))
        r = rng.randint(2, h - height - 2)
        c = rng.randint(scene_c0 + 1, w - 3)
        t = rng.randint(1, 2)
        grid[r : r + height, c : c + t] = GREY
    else:
        rh = rng.randint(2, min(5, h - 4))
        rw = rng.randint(2, min(6, w - scene_c0 - 4))
        r = rng.randint(2, h - rh - 2)
        c = rng.randint(scene_c0 + 1, w - rw - 2)
        grid[r : r + rh, c : c + rw] = GREY


def _shape_pixels_at(shape: list[tuple[int, int]], center: tuple[int, int], turns: int) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr, dc in _rotate_coords(shape, turns)]


def _markers_valid(grid: np.ndarray, sep_col: int, mapping: dict[int, list[tuple[int, int]]], markers: list[tuple[int, int, int]]) -> bool:
    used = set()
    for r, c, color in markers:
        if int(grid[r, c]) != color:
            return False
        if _best_direction(grid, r, c) not in range(4):
            return False
        turns = _best_direction(grid, r, c)
        for rr, cc in _shape_pixels_at(mapping[color], (r, c), turns):
            if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
                return False
            if int(grid[rr, cc]) == GREY:
                return False
            if (rr, cc) in used:
                return False
            used.add((rr, cc))
    return True


def _candidate_positions(
    grid: np.ndarray,
    sep_col: int,
    mapping: dict[int, list[tuple[int, int]]],
    markers: list[tuple[int, int, int]],
    color: int,
    desired_dir: int,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    out = []
    for r in range(h):
        for c in range(sep_col + 1, w):
            if int(grid[r, c]) != BG:
                continue
            if any(abs(r - mr) <= 1 and abs(c - mc) <= 1 for mr, mc, _ in markers):
                continue
            temp = grid.copy()
            temp[r, c] = color
            if _best_direction(temp, r, c) != desired_dir:
                continue
            trial = markers + [(r, c, color)]
            if _markers_valid(temp, sep_col, mapping, trial):
                out.append((r, c))
    return out


def _build_legend(rng: random.Random) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    count = rng.randint(4, 5)
    colors = rng.sample(COLOR_POOL, count)
    shapes = rng.sample(SHAPE_LIBRARY, count)
    mapping = {color: shape for color, shape in zip(colors, shapes)}

    cell_h = 7
    cell_w = 7
    legend_h = count * cell_h + (count - 1)
    legend = np.zeros((legend_h, cell_w), dtype=int)
    row = 0
    for i, color in enumerate(colors):
        legend[row : row + cell_h, :] = _render_shape(mapping[color], color, cell_h, cell_w)
        row += cell_h
        if i != count - 1:
            legend[row, :] = GREY
            row += 1
    return legend, mapping


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    legend, mapping = _build_legend(rng)
    h = legend.shape[0]
    scene_w = rng.randint(18, 28)
    sep_col = legend.shape[1]
    w = sep_col + 1 + scene_w
    grid = np.zeros((h, w), dtype=int)
    grid[:, :sep_col] = legend
    grid[:, sep_col] = GREY

    for _ in range(rng.randint(2, 5)):
        _add_bar(grid, rng, sep_col)
    if np.all(grid[:, sep_col + 1 :] == BG):
        return None
    if any(np.all(grid[:, c] == GREY) for c in range(sep_col + 1, w)):
        return None

    colors = list(mapping.keys())
    desired_dirs = [0, 1, 2, 3]
    extra = rng.randint(0, 3)
    desired_dirs.extend(rng.choice([0, 1, 2, 3]) for _ in range(extra))
    rng.shuffle(desired_dirs)

    markers: list[tuple[int, int, int]] = []
    for desired_dir in desired_dirs:
        rng.shuffle(colors)
        placed = False
        for color in colors:
            candidates = _candidate_positions(grid, sep_col, mapping, markers, color, desired_dir)
            if not candidates:
                continue
            r, c = rng.choice(candidates)
            grid[r, c] = color
            markers.append((r, c, color))
            placed = True
            break
        if not placed:
            return None

    if not _markers_valid(grid, sep_col, mapping, markers):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
