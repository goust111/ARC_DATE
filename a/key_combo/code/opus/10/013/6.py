import random

import numpy as np


BACKGROUND = 0
DIAGONAL_MARK = 1
INPUT_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 400
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _neighbor_count_map(mask: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(mask):
        total = 0
        for rr, cc in _neighbors4(mask.shape, int(r), int(c)):
            if mask[rr, cc]:
                total += 1
        counts[(int(r), int(c))] = total
    return counts


def _render_output(grid: np.ndarray, counts: dict[tuple[int, int], int], scale: int) -> np.ndarray:
    h, w = grid.shape
    out = np.zeros((h * scale, w * scale), dtype=int)
    diag = np.arange(scale)
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND:
                continue
            r0 = r * scale
            c0 = c * scale
            if counts[(r, c)] == scale:
                out[r0 : r0 + scale, c0 : c0 + scale] = color
            else:
                out[r0 + diag, c0 + diag] = DIAGONAL_MARK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale the grid by the maximum 4-neighbor count and turn lower-degree foreground pixels into blue diagonals.

    Input structure:
    - Black (0) is the background.
    - Non-black cells form a single small 4-connected foreground shape.
    - Foreground colors may vary, but neighbor counting depends only on foreground vs background.

    Rule:
    - For each foreground pixel, count how many of its up/down/left/right neighbors are also foreground.
    - Let M be the maximum such count over all foreground pixels.
    - Produce an output of shape (M * H, M * W), where each input cell maps to an MxM block.
    - Background cells stay black; foreground cells with count M become solid same-color blocks; foreground cells with count below M become black blocks with only a blue main diagonal.

    Key insight:
    - The same local 4-neighbor counts simultaneously determine the global upscale factor and which foreground pixels keep their color versus collapse to blue diagonal markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mask = g != BACKGROUND
    assert bool(mask.any()), "expected at least one foreground pixel"

    counts = _neighbor_count_map(mask)
    scale = max(counts.values())
    assert scale >= 1, "expected foreground pixels to have at least one 4-neighbor"

    return _render_output(g, counts, scale)


def _normalize(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return {(r - min_r, c - min_c) for r, c in coords}


def _transform_coords(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    pts = set(coords)
    if random.random() < 0.5:
        pts = {(r, -c) for r, c in pts}
    rotation = random.randint(0, 3)
    for _ in range(rotation):
        pts = {(c, -r) for r, c in pts}
    return _normalize(pts)


def _bbox_shape(coords: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return max(rows) + 1, max(cols) + 1


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    start = next(iter(coords))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(coords)


def _shape_is_valid(coords: set[tuple[int, int]], target_scale: int) -> bool:
    if not 3 <= len(coords) <= 8:
        return False
    if not _is_connected(coords):
        return False

    shape_h, shape_w = _bbox_shape(coords)
    if shape_h > 6 or shape_w > 6:
        return False
    if shape_h == 1 or shape_w == 1:
        return False
    if len(coords) == shape_h * shape_w:
        return False

    mask = np.zeros((shape_h, shape_w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True

    counts = _neighbor_count_map(mask)
    values = list(counts.values())
    if min(values) < 1:
        return False
    if max(values) != target_scale:
        return False
    if len(set(values)) < 2:
        return False
    return True


def _generate_path_shape() -> set[tuple[int, int]] | None:
    length = random.randint(3, 8)
    for _ in range(80):
        path = [(0, 0)]
        used = {(0, 0)}
        while len(path) < length:
            r, c = path[-1]
            choices = []
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if nxt not in used:
                    choices.append(nxt)
            if not choices:
                break
            nxt = random.choice(choices)
            path.append(nxt)
            used.add(nxt)

        if len(path) != length:
            continue

        coords = _transform_coords(set(path))
        if _shape_is_valid(coords, target_scale=2):
            return coords
    return None


def _random_composition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = [cuts[0]]
    for i in range(1, len(cuts)):
        values.append(cuts[i] - cuts[i - 1])
    values.append(total - cuts[-1])
    random.shuffle(values)
    return values


def _generate_t_shape() -> set[tuple[int, int]] | None:
    target_size = random.randint(4, 8)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for _ in range(120):
        missing = random.randrange(4)
        active_dirs = [d for i, d in enumerate(directions) if i != missing]
        lengths = _random_composition(target_size - 1, 3)
        coords = {(0, 0)}
        for (dr, dc), arm_len in zip(active_dirs, lengths):
            for step in range(1, arm_len + 1):
                coords.add((dr * step, dc * step))
        coords = _transform_coords(coords)
        if _shape_is_valid(coords, target_scale=3):
            return coords
    return None


def _generate_plus_shape() -> set[tuple[int, int]] | None:
    target_size = random.randint(5, 8)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for _ in range(120):
        lengths = _random_composition(target_size - 1, 4)
        coords = {(0, 0)}
        for (dr, dc), arm_len in zip(directions, lengths):
            for step in range(1, arm_len + 1):
                coords.add((dr * step, dc * step))
        coords = _transform_coords(coords)
        if _shape_is_valid(coords, target_scale=4):
            return coords
    return None


def _sample_shape(target_scale: int) -> set[tuple[int, int]] | None:
    if target_scale == 2:
        return _generate_path_shape()
    if target_scale == 3:
        return _generate_t_shape()
    return _generate_plus_shape()


def _paint_shape(coords: set[tuple[int, int]]) -> np.ndarray:
    shape_h, shape_w = _bbox_shape(coords)
    grid_h = random.randint(max(4, shape_h), 6)
    grid_w = random.randint(max(4, shape_w), 6)
    row_offset = random.randint(0, grid_h - shape_h)
    col_offset = random.randint(0, grid_w - shape_w)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    ordered = list(coords)
    random.shuffle(ordered)

    n_colors = random.randint(1, min(3, len(ordered)))
    palette = random.sample(INPUT_COLORS, n_colors)
    colors = palette[:] + [random.choice(palette) for _ in range(len(ordered) - n_colors)]
    random.shuffle(colors)

    for (r, c), color in zip(ordered, colors):
        grid[row_offset + r, col_offset + c] = color
    return grid


def _input_is_valid(grid: np.ndarray, target_scale: int) -> bool:
    if grid.ndim != 2:
        return False
    h, w = grid.shape
    if not (4 <= h <= 6 and 4 <= w <= 6):
        return False

    mask = grid != BACKGROUND
    fg_count = int(mask.sum())
    if not (3 <= fg_count <= 8):
        return False

    norm_coords = _normalize({(int(r), int(c)) for r, c in np.argwhere(mask)})
    if not _shape_is_valid(norm_coords, target_scale):
        return False

    colors = {int(v) for v in grid[mask]}
    if not (1 <= len(colors) <= 3):
        return False

    counts = _neighbor_count_map(mask)
    values = list(counts.values())
    if max(values) != target_scale:
        return False
    if len(set(values)) < 2:
        return False

    out = _render_output(grid, counts, target_scale)
    return max(out.shape) <= 30


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        target_scale = random.choice([2, 3, 4])
        coords = _sample_shape(target_scale)
        if coords is None:
            continue
        grid = _paint_shape(coords)
        if _input_is_valid(grid, target_scale):
            return grid
    raise RuntimeError("failed to generate a valid input")
