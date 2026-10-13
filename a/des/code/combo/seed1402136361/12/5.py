import random
from collections import Counter

import numpy as np


# --------------- shared helpers ---------------

_MAX_ATTEMPTS = 400
_GRID_H_RANGE = (10, 18)
_GRID_W_RANGE = (12, 20)
_GROUP_COUNT_RANGE = (2, 5)
_MIN_BBOX_SIDE = 2
_MAX_BBOX_SIDE = 6
_MIN_BBOX_AREA = 4
_PLACE_RETRIES = 200


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _contiguous_runs(rows: list[int]) -> list[list[int]]:
    if not rows:
        return []
    runs = [[rows[0]]]
    for row in rows[1:]:
        if row == runs[-1][-1] + 1:
            runs[-1].append(row)
        else:
            runs.append([row])
    return runs


def _groups_from_grid(grid: np.ndarray, bg: int, palette_mask: np.ndarray) -> list[dict]:
    groups = []
    colors = sorted(set(map(int, grid[~palette_mask].ravel())) - {bg})
    for color in colors:
        coords = np.argwhere((grid == color) & (~palette_mask))
        if coords.size == 0:
            continue
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        groups.append(
            {
                "color": int(color),
                "coords": coords,
                "r0": int(r0),
                "c0": int(c0),
                "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
            }
        )
    return groups


def _find_palette(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    color_counts = Counter(map(int, grid.ravel()))
    candidates = []

    for col in (0, w - 1):
        edge_rows = [r for r in range(h) if int(grid[r, col]) != bg]
        for rows in _contiguous_runs(edge_rows):
            colors = [int(grid[r, col]) for r in rows]
            if len(colors) != len(set(colors)):
                continue
            if any(color_counts[color] != 1 for color in colors):
                continue

            mask = np.zeros(grid.shape, dtype=bool)
            mask[rows, col] = True
            groups = _groups_from_grid(grid, bg, mask)
            if len(colors) != max(0, len(groups) - 1):
                continue

            inner_col = 1 if col == 0 else w - 2
            inner_gap = sum(int(grid[r, inner_col] == bg) for r in rows) if w >= 2 else 0
            candidates.append((len(rows), inner_gap, -col, rows, colors, mask))

    assert candidates, "palette legend not found"
    candidates.sort(reverse=True)
    _length, _gap, _edge_bias, _rows, colors, mask = candidates[0]
    return list(colors), mask


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    palette_colors, palette_mask = _find_palette(g, bg)
    groups = _groups_from_grid(g, bg, palette_mask)
    assert len(groups) >= 2, "expected at least two color groups"

    groups.sort(key=lambda group: (-int(group["area"]), int(group["r0"]), int(group["c0"])))
    assert len(palette_colors) >= len(groups) - 1, "palette is too short for the ranking"

    out = g.copy()
    out[palette_mask] = bg
    for rank, group in enumerate(groups):
        new_color = int(group["color"]) if rank == 0 else int(palette_colors[rank - 1])
        coords = group["coords"]
        out[coords[:, 0], coords[:, 1]] = new_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each color group by descending bounding-box area and remove the edge legend."""
    return _transform(grid)


# --------------- generator ---------------

def _sample_group_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    area = height * width
    style = rng.choice(["corners", "scatter", "diag", "edge"])

    if style == "corners":
        cells = {(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)}
    elif style == "diag":
        steps = max(height, width)
        cells = set()
        for i in range(steps):
            denom = max(1, steps - 1)
            r = int(round(i * (height - 1) / denom))
            c = int(round(i * (width - 1) / denom))
            cells.add((r, c))
        if rng.random() < 0.5:
            for i in range(steps):
                denom = max(1, steps - 1)
                r = int(round(i * (height - 1) / denom))
                c = int(round((steps - 1 - i) * (width - 1) / denom))
                cells.add((r, c))
        cells.update(
            {
                (0, rng.randrange(width)),
                (height - 1, rng.randrange(width)),
                (rng.randrange(height), 0),
                (rng.randrange(height), width - 1),
            }
        )
    elif style == "edge":
        cells = {
            (0, rng.randrange(width)),
            (height - 1, rng.randrange(width)),
            (rng.randrange(height), 0),
            (rng.randrange(height), width - 1),
        }
        if rng.random() < 0.5:
            row = rng.choice([0, height - 1])
            span = rng.randint(2, min(width, 4))
            for col in rng.sample(list(range(width)), span):
                cells.add((row, col))
        else:
            col = rng.choice([0, width - 1])
            span = rng.randint(2, min(height, 4))
            for row in rng.sample(list(range(height)), span):
                cells.add((row, col))
    else:
        cells = {
            (0, rng.randrange(width)),
            (height - 1, rng.randrange(width)),
            (rng.randrange(height), 0),
            (rng.randrange(height), width - 1),
        }

    min_cells = max(4, area // 4)
    max_cells = max(min_cells, min(area, (2 * area) // 3))
    target = rng.randint(min_cells, max_cells)
    target = max(target, len(cells))

    all_cells = [(r, c) for r in range(height) for c in range(width)]
    rng.shuffle(all_cells)
    for cell in all_cells:
        if len(cells) >= target:
            break
        cells.add(cell)

    for r, c in cells:
        mask[r, c] = True

    assert mask[0, :].any()
    assert mask[-1, :].any()
    assert mask[:, 0].any()
    assert mask[:, -1].any()
    return mask


def _bbox_dims(h: int, w: int) -> dict[int, list[tuple[int, int]]]:
    area_to_dims: dict[int, list[tuple[int, int]]] = {}
    max_h = min(_MAX_BBOX_SIDE, h - 2)
    max_w = min(_MAX_BBOX_SIDE, w - 3)
    for box_h in range(_MIN_BBOX_SIDE, max_h + 1):
        for box_w in range(_MIN_BBOX_SIDE, max_w + 1):
            area = box_h * box_w
            if area < _MIN_BBOX_AREA:
                continue
            area_to_dims.setdefault(area, []).append((box_h, box_w))
    return area_to_dims


def _validate_input(grid: np.ndarray) -> bool:
    bg = _bg_color(grid)
    try:
        palette_colors, palette_mask = _find_palette(grid, bg)
        groups = _groups_from_grid(grid, bg, palette_mask)
        if len(groups) < 2:
            return False
        if len(palette_colors) != len(groups) - 1:
            return False
        areas = [int(group["area"]) for group in groups]
        if len(areas) != len(set(areas)):
            return False
        out = _transform(grid)
    except AssertionError:
        return False

    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False
    if np.any(out[palette_mask] != bg):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*_GRID_H_RANGE)
    w = rng.randint(*_GRID_W_RANGE)
    n_groups = rng.randint(*_GROUP_COUNT_RANGE)
    if 1 + n_groups + (n_groups - 1) > 10:
        return None

    palette_side = rng.choice(["left", "right"])
    legend_col = 0 if palette_side == "left" else w - 1
    protected_col = 1 if palette_side == "left" else w - 2
    col_min = 2 if palette_side == "left" else 1
    col_max = w - 2 if palette_side == "left" else w - 3
    if col_max - col_min + 1 < _MIN_BBOX_SIDE:
        return None
    if h - 2 < _MIN_BBOX_SIDE:
        return None

    area_to_dims = _bbox_dims(h, w)
    if len(area_to_dims) < n_groups:
        return None

    colors = list(range(10))
    rng.shuffle(colors)
    bg = int(colors[0])
    group_colors = [int(color) for color in colors[1 : 1 + n_groups]]
    legend_colors = [int(color) for color in colors[1 + n_groups : 1 + n_groups + n_groups - 1]]

    grid = np.full((h, w), bg, dtype=int)
    legend_len = len(legend_colors)
    legend_start = rng.randint(0, h - legend_len)
    for i, color in enumerate(legend_colors):
        grid[legend_start + i, legend_col] = int(color)

    occupied = np.zeros((h, w), dtype=bool)
    occupied[:, legend_col] = True
    occupied[:, protected_col] = True
    occupied[0, :] = True
    occupied[-1, :] = True

    chosen_areas = rng.sample(list(area_to_dims.keys()), n_groups)
    placements = []
    for area, color in sorted(zip(chosen_areas, group_colors), reverse=True):
        box_h, box_w = rng.choice(area_to_dims[int(area)])
        placed = False
        for _ in range(_PLACE_RETRIES):
            r0 = rng.randint(1, h - box_h - 1)
            c0 = rng.randint(col_min, col_max - box_w + 1)
            r1 = r0 + box_h - 1
            c1 = c0 + box_w - 1

            pad_r0 = max(0, r0 - 1)
            pad_r1 = min(h - 1, r1 + 1)
            pad_c0 = max(0, c0 - 1)
            pad_c1 = min(w - 1, c1 + 1)
            if occupied[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1].any():
                continue

            mask = _sample_group_mask(box_h, box_w, rng)
            coords = np.argwhere(mask)
            grid[r0 + coords[:, 0], c0 + coords[:, 1]] = int(color)
            occupied[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1] = True
            placements.append((int(area), int(color), int(r0), int(c0)))
            placed = True
            break
        if not placed:
            return None

    if len(placements) != n_groups:
        return None
    if not _validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
