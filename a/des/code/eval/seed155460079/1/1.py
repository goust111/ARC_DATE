from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
COLOR_POOL = [1, 2, 3, 6, 7, 8, 9]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask.astype(bool))


def _rotation_count(mask: np.ndarray) -> int:
    return len({_mask_key(np.rot90(mask, k)) for k in range(4)})


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(int(v) for v in coords[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == len(coords)


def _rotate_offsets(offsets: list[tuple[int, int]], direction: int) -> list[tuple[int, int]]:
    rotated: list[tuple[int, int]] = []
    for dr, dc in offsets:
        if direction == 0:
            rr, cc = dr, dc
        elif direction == 1:
            rr, cc = dc, -dr
        elif direction == 2:
            rr, cc = -dr, -dc
        else:
            rr, cc = -dc, dr
        rotated.append((rr, cc))
    return rotated


def _cell_offsets(cell: np.ndarray, color: int) -> list[tuple[int, int]]:
    anchor_r = cell.shape[0] // 2
    anchor_c = cell.shape[1] // 2
    coords = np.argwhere(cell == color)
    assert len(coords) > 0
    return [(int(r) - anchor_r, int(c) - anchor_c) for r, c in coords]


def _parse_templates(grid: np.ndarray) -> tuple[int, dict[int, list[tuple[int, int]]]]:
    grey_rows = np.flatnonzero(np.any(grid == GREY, axis=1))
    assert len(grey_rows) > 0 and int(grey_rows[0]) == 0
    legend_h = int(grey_rows[-1]) + 1
    assert np.array_equal(grey_rows, np.arange(legend_h))

    band = grid[:legend_h]
    sep_cols = [c for c in range(grid.shape[1]) if np.all(band[:, c] == GREY)]
    assert len(sep_cols) >= 3

    templates: dict[int, list[tuple[int, int]]] = {}
    inner_h = legend_h - 2
    assert inner_h > 0 and inner_h % 2 == 1
    for c0, c1 in zip(sep_cols, sep_cols[1:]):
        if c1 - c0 <= 1:
            continue
        cell = band[1:-1, c0 + 1 : c1]
        assert cell.shape[0] == inner_h and cell.shape[1] % 2 == 1
        colors = [int(v) for v in np.unique(cell) if int(v) not in {BLACK, GREY}]
        assert len(colors) == 1
        color = colors[0]
        center = (cell.shape[0] // 2, cell.shape[1] // 2)
        assert cell[center] == color
        mask = cell == color
        assert _is_connected(mask)
        templates[color] = _cell_offsets(cell, color)
    assert templates
    return legend_h, templates


def _seed_direction(grid: np.ndarray, r: int, c: int) -> int:
    directions = [idx for idx, (dr, dc) in enumerate(DIRS) if grid[r + dr, c + dc] == YELLOW]
    assert len(directions) == 1
    return directions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Look up a color-coded template, decode the adjacent yellow direction marker,
    rotate the template around its center anchor, and project it from the seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    legend_h, templates = _parse_templates(g)
    out = g.copy()
    out[legend_h:][out[legend_h:] == YELLOW] = BLACK

    seeds: list[tuple[int, int, int, int]] = []
    for r in range(legend_h, g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color == BLACK or color == YELLOW:
                continue
            assert color in templates
            same_color_neighbors = sum(
                1
                for rr, cc in _neighbors4(r, c)
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and g[rr, cc] == color
            )
            assert same_color_neighbors == 0
            seeds.append((r, c, color, _seed_direction(g, r, c)))

    occupied = {(r, c) for r, c, _, _ in seeds}
    for r, c, color, direction in sorted(seeds):
        for dr, dc in _rotate_offsets(templates[color], direction):
            rr = r + dr
            cc = c + dc
            assert legend_h <= rr < g.shape[0]
            assert 0 <= cc < g.shape[1]
            existing = int(out[rr, cc])
            assert existing in {BLACK, color}
            occupied.add((rr, cc))
            out[rr, cc] = color
    return out


def _random_shape(side: int) -> np.ndarray | None:
    center = (side // 2, side // 2)
    min_cells, max_cells = ((4, 6) if side == 3 else (6, 11))
    for _ in range(SHAPE_ATTEMPTS):
        target = random.randint(min_cells, max_cells)
        mask = np.zeros((side, side), dtype=bool)
        mask[center] = True
        active = {center}
        while len(active) < target:
            frontier = set()
            for r, c in active:
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < side and 0 <= cc < side and not mask[rr, cc]:
                        frontier.add((rr, cc))
            if not frontier:
                break
            choice = random.choice(list(frontier))
            mask[choice] = True
            active.add(choice)
        if int(mask.sum()) != target:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if _rotation_count(mask) != 4:
            continue
        if not _is_connected(mask):
            continue
        return mask
    return None


def _embed_shape(mask: np.ndarray, cell_side: int, color: int) -> np.ndarray:
    cell = np.zeros((cell_side, cell_side), dtype=int)
    top = (cell_side - mask.shape[0]) // 2
    left = (cell_side - mask.shape[1]) // 2
    cell[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = color
    return cell


def _make_templates(count: int, cell_side: int) -> list[dict[str, object]] | None:
    colors = random.sample(COLOR_POOL, count)
    used_shapes: set[tuple[str, ...]] = set()
    templates: list[dict[str, object]] = []
    for color in colors:
        for _ in range(SHAPE_ATTEMPTS):
            side = random.choice((3, 5))
            mask = _random_shape(side)
            if mask is None:
                continue
            key = _mask_key(mask)
            if key in used_shapes:
                continue
            used_shapes.add(key)
            cell = _embed_shape(mask, cell_side, color)
            templates.append({"color": color, "cell": cell, "offsets": _cell_offsets(cell, color)})
            break
        else:
            return None
    return templates


def _balanced_directions(count: int) -> list[int]:
    all_dirs = [0, 1, 2, 3]
    if count <= 4:
        return random.sample(all_dirs, count)
    directions = all_dirs * (count // 4)
    directions.extend(random.sample(all_dirs, count % 4))
    random.shuffle(directions)
    return directions


def _fits(cells: set[tuple[int, int]], height: int, width: int) -> bool:
    return all(0 <= r < height and 0 <= c < width for r, c in cells)


def _gen_attempt() -> np.ndarray | None:
    template_count = random.randint(2, 5)
    cell_side = random.choice((5, 7))
    templates = _make_templates(template_count, cell_side)
    if templates is None:
        return None

    legend_h = cell_side + 2
    width = 1 + template_count * (cell_side + 1)
    seed_count = random.randint(template_count, min(template_count + 3, 8))
    min_canvas_h = max(10, cell_side + seed_count + 2)
    max_canvas_h = min(21, min_canvas_h + 7)
    canvas_h = random.randint(min_canvas_h, max_canvas_h)

    grid = np.full((legend_h + canvas_h, width), BLACK, dtype=int)
    grid[:legend_h, :] = BLACK
    grid[0, :] = GREY
    grid[legend_h - 1, :] = GREY
    for idx in range(template_count + 1):
        col = idx * (cell_side + 1)
        grid[:legend_h, col] = GREY
    for idx, template in enumerate(templates):
        col0 = idx * (cell_side + 1) + 1
        grid[1 : legend_h - 1, col0 : col0 + cell_side] = template["cell"]

    by_color = {int(template["color"]): template for template in templates}
    seed_colors = [int(template["color"]) for template in templates]
    for _ in range(seed_count - template_count):
        seed_colors.append(random.choice(seed_colors))
    random.shuffle(seed_colors)
    directions = _balanced_directions(seed_count)

    specs = []
    for color, direction in zip(seed_colors, directions):
        offsets = _rotate_offsets(by_color[color]["offsets"], direction)
        specs.append({"color": color, "direction": direction, "offsets": offsets})
    random.shuffle(specs)
    specs.sort(key=lambda item: -len(item["offsets"]))

    occupied: set[tuple[int, int]] = set()
    placements: list[tuple[int, int, int, int]] = []
    for spec in specs:
        direction = int(spec["direction"])
        marker_dr, marker_dc = DIRS[direction]
        offsets = spec["offsets"]
        candidates: list[tuple[int, int, set[tuple[int, int]], tuple[int, int]]] = []
        for r in range(canvas_h):
            for c in range(width):
                shape = {(r + dr, c + dc) for dr, dc in offsets}
                marker = (r + marker_dr, c + marker_dc)
                cells = set(shape)
                cells.add(marker)
                if not _fits(cells, canvas_h, width):
                    continue
                if cells & occupied:
                    continue
                candidates.append((r, c, shape, marker))
        if not candidates:
            return None
        r, c, shape, marker = random.choice(candidates)
        occupied |= shape
        occupied.add(marker)
        placements.append((r, c, int(spec["color"]), direction))

    for r, c, color, direction in placements:
        grid[legend_h + r, c] = color
        dr, dc = DIRS[direction]
        grid[legend_h + r + dr, c + dc] = YELLOW

    try:
        main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
