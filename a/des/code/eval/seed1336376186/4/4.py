import random

import numpy as np


BG = 0
GREY = 5
MIX = 3
PATH_COLORS = (1, 2, 4, 6, 7, 8, 9)
SIZE_CHOICES = (5, 6)
SCALE_CHOICES = (2, 3, 4)
MODE_CHOICES = ("none", "single", "double")
GEN_RETRY_LIMIT = 600


def _most_common_color(arr: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert pts.size > 0, "expected a non-empty mask"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = [points[0]]
    for r1, c1 in points[1:]:
        r0, c0 = cells[-1]
        assert r0 == r1 or c0 == c1, "polyline segments must be axis-aligned"
        dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
        dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
        r, c = r0, c0
        while (r, c) != (r1, c1):
            r += dr
            c += dc
            cells.append((r, c))
    return cells


def _on_boundary(cell: tuple[int, int], size: int) -> bool:
    r, c = cell
    return r == 0 or c == 0 or r == size - 1 or c == size - 1


def _path_orientation_map(path: list[tuple[int, int]]) -> dict[tuple[int, int], str]:
    orientation: dict[tuple[int, int], str] = {}
    for i, cell in enumerate(path):
        if i == 0 or i == len(path) - 1:
            orientation[cell] = "end"
            continue
        pr, pc = path[i - 1]
        nr, nc = path[i + 1]
        if pr == nr:
            orientation[cell] = "h"
        elif pc == nc:
            orientation[cell] = "v"
        else:
            orientation[cell] = "corner"
    return orientation


def _find_divider(g: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = g.shape
    candidates = []
    for c in range(w):
        color = int(g[0, c])
        if color == bg or not np.all(g[:, c] == color):
            continue
        if c + 1 >= w:
            continue
        if np.all(g[:, c + 1 :] == bg):
            candidates.append((c, color))
    assert len(candidates) == 1, "expected one full-height divider with a blank canvas on its right"
    return candidates[0]


def _recover_paths(circuit: np.ndarray, bg: int, grey: int) -> dict[int, np.ndarray]:
    h, w = circuit.shape
    colors = [int(color) for color in np.unique(circuit) if int(color) not in (bg, grey)]
    recovered: dict[int, np.ndarray] = {}

    for color in colors:
        visible = circuit == color
        hidden = np.zeros_like(visible, dtype=bool)
        other_path = (circuit != bg) & (circuit != grey) & (circuit != color)

        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if not other_path[r, c]:
                    continue
                if visible[r - 1, c] and visible[r + 1, c]:
                    hidden[r, c] = True
                if visible[r, c - 1] and visible[r, c + 1]:
                    hidden[r, c] = True

        recovered[color] = visible | hidden

    return recovered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover crossed color paths from the small circuit, then draw a scaled copy
    on the blank canvas. Cells where multiple recovered paths overlap become green.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    divider_col, grey = _find_divider(g, bg)
    left = g[:, :divider_col]
    right = g[:, divider_col + 1 :]

    mask = (left != bg) & (left != grey)
    r0, c0, r1, c1 = _bbox(mask)
    circuit = left[r0 : r1 + 1, c0 : c1 + 1]
    small_h, small_w = circuit.shape

    canvas_h, canvas_w = right.shape
    assert canvas_h % small_h == 0 and canvas_w % small_w == 0, "canvas must be an integer scale of the circuit"
    scale_h = canvas_h // small_h
    scale_w = canvas_w // small_w
    assert scale_h == scale_w, "canvas scaling must match in both dimensions"
    scale = int(scale_h)

    recovered = _recover_paths(circuit, bg, grey)

    out = g.copy()
    out[:, divider_col + 1 :] = bg
    for r in range(small_h):
        for c in range(small_w):
            colors_here = [color for color, cells in recovered.items() if cells[r, c]]
            if not colors_here:
                continue
            block_color = MIX if len(colors_here) > 1 else colors_here[0]
            rr0 = r * scale
            cc0 = divider_col + 1 + c * scale
            out[rr0 : rr0 + scale, cc0 : cc0 + scale] = block_color

    return out


def _random_lr_path(size: int) -> list[tuple[int, int]]:
    if random.random() < 0.45:
        row = random.randrange(size)
        path = _polyline([(row, 0), (row, size - 1)])
    else:
        row0 = random.randrange(size)
        row1 = random.randrange(size)
        while row1 == row0:
            row1 = random.randrange(size)
        bend_col = random.randint(1, size - 2)
        path = _polyline([(row0, 0), (row0, bend_col), (row1, bend_col), (row1, size - 1)])
    if random.random() < 0.5:
        path = path[::-1]
    return path


def _random_tb_path(size: int) -> list[tuple[int, int]]:
    if random.random() < 0.45:
        col = random.randrange(size)
        path = _polyline([(0, col), (size - 1, col)])
    else:
        col0 = random.randrange(size)
        col1 = random.randrange(size)
        while col1 == col0:
            col1 = random.randrange(size)
        bend_row = random.randint(1, size - 2)
        path = _polyline([(0, col0), (bend_row, col0), (bend_row, col1), (size - 1, col1)])
    if random.random() < 0.5:
        path = path[::-1]
    return path


def _random_corner_path(size: int, family: str) -> list[tuple[int, int]]:
    r = random.randint(1, size - 2)
    c = random.randint(1, size - 2)
    if family == "lt":
        path = _polyline([(r, 0), (r, c), (0, c)])
    elif family == "tr":
        path = _polyline([(0, c), (r, c), (r, size - 1)])
    elif family == "rb":
        path = _polyline([(r, size - 1), (r, c), (size - 1, c)])
    elif family == "bl":
        path = _polyline([(size - 1, c), (r, c), (r, 0)])
    else:
        raise ValueError(f"unknown corner family: {family}")
    if random.random() < 0.5:
        path = path[::-1]
    return path


def _validate_paths(paths: list[list[tuple[int, int]]], size: int, crossings: int) -> bool:
    endpoint_cells: set[tuple[int, int]] = set()
    path_sets = [set(path) for path in paths]
    orientations = [_path_orientation_map(path) for path in paths]
    overlap_map: dict[tuple[int, int], list[int]] = {}

    for path in paths:
        if len(path) != len(set(path)):
            return False
        if not _on_boundary(path[0], size) or not _on_boundary(path[-1], size):
            return False
        if path[0] in endpoint_cells or path[-1] in endpoint_cells:
            return False
        endpoint_cells.add(path[0])
        endpoint_cells.add(path[-1])

    for idx, path in enumerate(paths):
        for cell in path:
            overlap_map.setdefault(cell, []).append(idx)

    overlap_cells = [cell for cell, owners in overlap_map.items() if len(owners) > 1]
    if len(overlap_cells) != crossings:
        return False

    for cell in overlap_cells:
        owners = overlap_map[cell]
        if len(owners) != 2:
            return False
        orient_a = orientations[owners[0]][cell]
        orient_b = orientations[owners[1]][cell]
        if orient_a not in {"h", "v"} or orient_b not in {"h", "v"} or orient_a == orient_b:
            return False
        if not (0 < cell[0] < size - 1 and 0 < cell[1] < size - 1):
            return False

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            set_i = path_sets[i]
            set_j = path_sets[j]
            for r, c in set_i:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (nr, nc) in set_j and (nr, nc) != (r, c):
                        return False

    for path in paths:
        overlap_indices = sorted(i for i, cell in enumerate(path) if cell in overlap_cells)
        if any(b - a <= 1 for a, b in zip(overlap_indices, overlap_indices[1:], strict=False)):
            return False

    union = set().union(*path_sets)
    if not any(r == 0 for r, _ in union):
        return False
    if not any(r == size - 1 for r, _ in union):
        return False
    if not any(c == 0 for _, c in union):
        return False
    if not any(c == size - 1 for _, c in union):
        return False

    return True


def _sample_paths(size: int, mode: str) -> list[list[tuple[int, int]]] | None:
    for _ in range(250):
        if mode == "none":
            if random.random() < 0.5:
                paths = [_random_corner_path(size, "lt"), _random_corner_path(size, "rb")]
            else:
                paths = [_random_corner_path(size, "tr"), _random_corner_path(size, "bl")]
            if _validate_paths(paths, size, crossings=0):
                return paths
            continue

        if mode == "single":
            paths = [_random_lr_path(size), _random_tb_path(size)]
            if random.random() < 0.45:
                paths.append(_random_corner_path(size, random.choice(("lt", "tr", "rb", "bl"))))
            if _validate_paths(paths, size, crossings=1):
                return paths
            continue

        if mode == "double":
            if random.random() < 0.5:
                paths = [_random_lr_path(size), _random_tb_path(size), _random_tb_path(size)]
            else:
                paths = [_random_tb_path(size), _random_lr_path(size), _random_lr_path(size)]
            if _validate_paths(paths, size, crossings=2):
                return paths
            continue

        raise ValueError(f"unknown mode: {mode}")
    return None


def _render_input(paths: list[list[tuple[int, int]]], colors: list[int], size: int, scale: int, top_offset: int) -> np.ndarray:
    height = size * scale
    width = size + 1 + size * scale
    grid = np.full((height, width), BG, dtype=int)
    grid[:, size] = GREY

    draw_order = list(range(len(paths)))
    random.shuffle(draw_order)
    for idx in draw_order:
        color = colors[idx]
        for r, c in paths[idx]:
            grid[top_offset + r, c] = color

    return grid


def _expected_output(inp: np.ndarray, paths: list[list[tuple[int, int]]], colors: list[int], size: int, scale: int) -> np.ndarray:
    out = inp.copy()
    out[:, size + 1 :] = BG

    path_sets = [set(path) for path in paths]
    for r in range(size):
        for c in range(size):
            cell_colors = [colors[i] for i, cells in enumerate(path_sets) if (r, c) in cells]
            if not cell_colors:
                continue
            block_color = MIX if len(cell_colors) > 1 else cell_colors[0]
            rr0 = r * scale
            cc0 = size + 1 + c * scale
            out[rr0 : rr0 + scale, cc0 : cc0 + scale] = block_color

    return out


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(SIZE_CHOICES)
    scale = random.choice(SCALE_CHOICES)
    mode = random.choice(MODE_CHOICES)
    paths = _sample_paths(size, mode)
    if paths is None:
        return None

    colors = random.sample(PATH_COLORS, len(paths))
    top_offset = random.randint(0, size * scale - size)
    inp = _render_input(paths, colors, size, scale, top_offset)
    expected = _expected_output(inp, paths, colors, size, scale)

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid circuit-scaling input")
