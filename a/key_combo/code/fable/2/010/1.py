from __future__ import annotations

from itertools import permutations
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg


ALLOWED_DELTAS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
MOVE_DELTAS = ALLOWED_DELTAS[1:]
ALL_FOREGROUND_COLORS = tuple(range(1, 10))
CORNERS = ("tl", "tr", "bl", "br")
OUTWARD_BY_CORNER = {
    "tl": {(1, 0), (0, 1)},
    "tr": {(1, 0), (0, -1)},
    "bl": {(-1, 0), (0, 1)},
    "br": {(-1, 0), (0, -1)},
}


def _normalized_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0 = coords.min(axis=0)
    rel = coords - np.array([r0, c0])
    return tuple(sorted(map(tuple, rel.tolist())))


def _component_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _ranked_layout(components: list[dict]) -> tuple[list[dict], list[tuple[int, int]], int, int]:
    row_keys = sorted({int(comp["bbox"][0] + comp["bbox"][2]) for comp in components})
    col_keys = sorted({int(comp["bbox"][1] + comp["bbox"][3]) for comp in components})
    row_rank = {key: idx for idx, key in enumerate(row_keys)}
    col_rank = {key: idx for idx, key in enumerate(col_keys)}

    enriched = []
    for comp in components:
        bbox = comp["bbox"]
        rr = row_rank[int(bbox[0] + bbox[2])]
        cc = col_rank[int(bbox[1] + bbox[3])]
        enriched.append((rr, cc, int(bbox[0]), int(bbox[1]), comp))

    enriched.sort()
    ordered = [comp for *_rest, comp in enriched]
    ideal_points = [(rr, cc) for rr, cc, _r0, _c0, _comp in enriched]
    return ordered, ideal_points, len(row_keys), len(col_keys)


def _decode_legend(
    ideal_points: list[tuple[int, int]],
    actual_coords: np.ndarray,
) -> list[tuple[int, int]] | None:
    if len(ideal_points) != int(actual_coords.shape[0]):
        return None

    actual = [tuple(map(int, rc)) for rc in actual_coords.tolist()]
    translation_candidates = set()
    for ir, ic in ideal_points:
        for ar, ac in actual:
            for dr, dc in ALLOWED_DELTAS:
                translation_candidates.add((ar - ir - dr, ac - ic - dc))

    best_cost: int | None = None
    best_solutions: set[tuple[tuple[int, int], ...]] = set()
    for tr, tc in translation_candidates:
        local_solutions: set[tuple[tuple[int, int], ...]] = set()
        for perm in permutations(actual):
            deltas = []
            valid = True
            for (ir, ic), (ar, ac) in zip(ideal_points, perm):
                delta = (ar - (tr + ir), ac - (tc + ic))
                if delta not in ALLOWED_DELTAS:
                    valid = False
                    break
                deltas.append(delta)
            if valid:
                local_solutions.add(tuple(deltas))
            if len(local_solutions) > 4:
                break

        for solution in local_solutions:
            cost = sum(abs(dr) + abs(dc) for dr, dc in solution)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_solutions = {solution}
            elif cost == best_cost:
                best_solutions.add(solution)

    if len(best_solutions) != 1:
        return None
    return list(next(iter(best_solutions)))


def _corner_score(coords: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h, w = shape
    distances = [
        int(r0 + c0),
        int(r0 + (w - 1 - c1)),
        int((h - 1 - r1) + c0),
        int((h - 1 - r1) + (w - 1 - c1)),
    ]
    bbox_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    return min(distances), bbox_area, int(r0 + c0)


def _identify_roles(
    grid: np.ndarray,
) -> tuple[int, int, list[dict], list[tuple[int, int]]]:
    bg = int(get_bg(grid))
    components = find_components(grid, bg=bg, connectivity=8, monochromatic=True)
    by_color: dict[int, list[dict]] = {}
    for comp in components:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    candidates = []
    foreground_colors = sorted(by_color)
    for sprite_color in foreground_colors:
        sprite_components = by_color[sprite_color]
        if len(sprite_components) < 2:
            continue
        if any(int(comp["area"]) <= 1 for comp in sprite_components):
            continue
        shapes = {_normalized_shape(comp["coords"]) for comp in sprite_components}
        if len(shapes) != 1:
            continue

        ordered, ideal_points, _rows, _cols = _ranked_layout(sprite_components)
        legend_options = []
        for legend_color in foreground_colors:
            if legend_color == sprite_color:
                continue
            legend_coords = np.argwhere(grid == legend_color)
            if int(legend_coords.shape[0]) != len(ordered):
                continue
            directions = _decode_legend(ideal_points, legend_coords)
            if directions is None:
                continue
            legend_options.append(
                (_corner_score(legend_coords, grid.shape), int(legend_color), directions)
            )

        if legend_options:
            legend_options.sort()
            corner_score, legend_color, directions = legend_options[0]
            candidates.append(
                (
                    len(legend_options),
                    -len(ordered),
                    corner_score,
                    int(sprite_color),
                    int(legend_color),
                    ordered,
                    directions,
                )
            )

    assert candidates, "failed to identify repeated sprite set and distorted legend"
    candidates.sort()
    _num_options, _neg_count, _score, sprite_color, legend_color, ordered, directions = candidates[0]
    return bg, legend_color, ordered, directions


def _slide_distance(
    coords: np.ndarray,
    direction: tuple[int, int],
    obstacle_mask: np.ndarray,
) -> int:
    dr, dc = direction
    if (dr, dc) == (0, 0):
        return 0

    h, w = obstacle_mask.shape
    steps = 0
    while True:
        shifted = coords + np.array([(steps + 1) * dr, (steps + 1) * dc])
        rows = shifted[:, 0]
        cols = shifted[:, 1]
        if int(rows.min()) < 0 or int(cols.min()) < 0 or int(rows.max()) >= h or int(cols.max()) >= w:
            break
        if bool(obstacle_mask[rows, cols].any()):
            break
        steps += 1
    return steps


def _apply_moves(
    grid: np.ndarray,
    sprite_color: int,
    sprites: list[dict],
    directions: list[tuple[int, int]],
) -> np.ndarray:
    bg = int(get_bg(grid))
    obstacle_mask = (grid != bg) & (grid != sprite_color)
    out = grid.copy()
    out[grid == sprite_color] = bg

    for comp, direction in zip(sprites, directions):
        coords = comp["coords"]
        steps = _slide_distance(coords, direction, obstacle_mask)
        dr, dc = direction
        shifted = coords + np.array([steps * dr, steps * dc])
        rows = shifted[:, 0]
        cols = shifted[:, 1]
        assert np.all(out[rows, cols] == bg), "moved sprites must not overlap"
        out[rows, cols] = sprite_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Decode each legend pixel's one-cell distortion as a slide direction and move the matching sprite to the nearest wall or edge.

    Input structure:
    - The background is black; several same-color, same-shape sprites are scattered in a ranked row/column layout.
    - A compact legend of a third color sits in one corner, while solid wall bars of another color act as static blockers.
    - If the sprite layout is compressed to a minimal rank grid, each legend pixel belongs to one layout slot but may be shifted by at most one cell up, down, left, or right.

    Rule:
    - Identify the repeated sprite set and order sprites by bbox-center row rank then column rank.
    - Match the corner legend to that same rank layout; each legend pixel's offset from its slot gives the corresponding sprite's slide direction, and an unshifted pixel means no movement.
    - Remove the original sprites, then slide each sprite straight along its decoded direction until the next step would hit a wall, the legend, or the grid boundary; keep walls and legend unchanged.

    Key insight:
    - The legend is not a shape template to copy: the only information that matters is each pixel's tiny deviation from the normalized sprite-layout thumbnail.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, legend_color, sprites, directions = _identify_roles(g)
    sprite_color = int(g[sprites[0]["coords"][0, 0], sprites[0]["coords"][0, 1]])
    assert legend_color != sprite_color != bg
    return _apply_moves(g, sprite_color, sprites, directions)


def _random_shape() -> np.ndarray:
    for _ in range(200):
        h, w = random.choice(((2, 2), (2, 3), (3, 2), (3, 3)))
        max_area = min(h * w, 6)
        min_area = max(3, max(h, w))
        area = random.randint(min_area, max_area)
        cells = [(r, c) for r in range(h) for c in range(w)]
        chosen = {random.choice(cells)}
        while len(chosen) < area:
            frontier = [
                cell
                for cell in cells
                if cell not in chosen
                and any(
                    max(abs(cell[0] - rr), abs(cell[1] - cc)) == 1
                    for rr, cc in chosen
                )
            ]
            chosen.add(random.choice(frontier))

        if len({r for r, _c in chosen}) != h or len({c for _r, c in chosen}) != w:
            continue
        mask = np.zeros((h, w), dtype=int)
        for r, c in chosen:
            mask[r, c] = 1
        return mask
    raise RuntimeError("failed to sample a sprite shape")


def _sample_layout() -> tuple[list[tuple[int, int]], int, int]:
    for _ in range(200):
        row_count = random.randint(2, 3)
        col_count = random.randint(2, 3)
        cells = [(r, c) for r in range(row_count) for c in range(col_count)]
        count = random.randint(max(4, max(row_count, col_count)), min(6, len(cells)))
        sample = random.sample(cells, count)
        if len({r for r, _c in sample}) != row_count:
            continue
        if len({c for _r, c in sample}) != col_count:
            continue
        sample.sort()
        return sample, row_count, col_count
    raise RuntimeError("failed to sample a rank layout")


def _legend_origin(corner: str, h: int, w: int, row_count: int, col_count: int) -> tuple[int, int]:
    r = 1 if "t" in corner else h - row_count - 1
    c = 1 if "l" in corner else w - col_count - 1
    return int(r), int(c)


def _matrix_base(
    corner: str,
    h: int,
    w: int,
    matrix_h: int,
    matrix_w: int,
) -> tuple[int, int]:
    r = h - matrix_h - 2 if "t" in corner else 2
    c = w - matrix_w - 2 if "l" in corner else 2
    return int(r), int(c)


def _outward_feasible(
    corner: str,
    direction: tuple[int, int],
    row_rank: int,
    col_rank: int,
    row_count: int,
    col_count: int,
) -> bool:
    dr, dc = direction
    if (dr, dc) not in OUTWARD_BY_CORNER[corner]:
        return False
    if (dr, dc) == (-1, 0):
        return row_rank == 0
    if (dr, dc) == (1, 0):
        return row_rank == row_count - 1
    if (dr, dc) == (0, -1):
        return col_rank == 0
    return col_rank == col_count - 1


def _wall_step_limit(
    direction: tuple[int, int],
    row_rank: int,
    col_rank: int,
    row_count: int,
    col_count: int,
    r0: int,
    c0: int,
    shape_h: int,
    shape_w: int,
    grid_h: int,
    grid_w: int,
    pitch_r: int,
    pitch_c: int,
) -> int:
    dr, dc = direction
    if (dr, dc) == (0, 1):
        boundary_limit = grid_w - (c0 + shape_w) - 1
        gap_limit = 3 if col_rank < col_count - 1 else min(4, boundary_limit)
    elif (dr, dc) == (0, -1):
        boundary_limit = c0 - 1
        gap_limit = 3 if col_rank > 0 else min(4, boundary_limit)
    elif (dr, dc) == (1, 0):
        boundary_limit = grid_h - (r0 + shape_h) - 1
        gap_limit = 3 if row_rank < row_count - 1 else min(4, boundary_limit)
    elif (dr, dc) == (-1, 0):
        boundary_limit = r0 - 1
        gap_limit = 3 if row_rank > 0 else min(4, boundary_limit)
    else:
        return 0

    if (dr, dc) in {(0, 1), (0, -1)}:
        gap_limit = min(gap_limit, pitch_c - shape_w - 1 if 0 < col_rank < col_count - 1 else gap_limit)
    else:
        gap_limit = min(gap_limit, pitch_r - shape_h - 1 if 0 < row_rank < row_count - 1 else gap_limit)
    return int(max(0, min(boundary_limit, gap_limit)))


def _build_components_from_shape(
    shape_mask: np.ndarray,
    layout_cells: list[tuple[int, int]],
    base_r: int,
    base_c: int,
    pitch_r: int,
    pitch_c: int,
) -> list[dict]:
    coords_rel = np.argwhere(shape_mask == 1)
    h, w = shape_mask.shape
    components = []
    for row_rank, col_rank in layout_cells:
        r0 = base_r + row_rank * pitch_r
        c0 = base_c + col_rank * pitch_c
        coords = coords_rel + np.array([r0, c0])
        components.append(
            {
                "coords": coords.astype(int),
                "bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
                "row_rank": int(row_rank),
                "col_rank": int(col_rank),
            }
        )
    return components


def _sample_directions(
    layout_cells: list[tuple[int, int]],
    legend_r0: int,
    legend_c0: int,
) -> list[tuple[int, int]] | None:
    actual = []
    directions = []
    for row_rank, col_rank in layout_cells:
        direction = random.choices(ALLOWED_DELTAS, weights=(2, 1, 1, 1, 1), k=1)[0]
        directions.append(direction)
        actual.append((legend_r0 + row_rank + direction[0], legend_c0 + col_rank + direction[1]))

    if all(direction == (0, 0) for direction in directions):
        idx = random.randrange(len(directions))
        directions[idx] = random.choice(MOVE_DELTAS)
        row_rank, col_rank = layout_cells[idx]
        actual[idx] = (legend_r0 + row_rank + directions[idx][0], legend_c0 + col_rank + directions[idx][1])

    if len(set(actual)) != len(actual):
        return None

    decoded = _decode_legend(layout_cells, np.array(actual, dtype=int))
    if decoded is None or tuple(decoded) != tuple(directions):
        return None
    return directions


def _build_grid(
    shape_mask: np.ndarray,
    layout_cells: list[tuple[int, int]],
    row_count: int,
    col_count: int,
    corner: str,
    sprite_color: int,
    wall_color: int,
    legend_color: int,
) -> tuple[np.ndarray, list[dict], list[tuple[int, int]]] | None:
    shape_h, shape_w = shape_mask.shape
    pitch_r = shape_h + 4
    pitch_c = shape_w + 4
    matrix_h = (row_count - 1) * pitch_r + shape_h
    matrix_w = (col_count - 1) * pitch_c + shape_w
    legend_h = row_count + 2
    legend_w = col_count + 2
    grid_h = matrix_h + legend_h + 2 + random.randint(2, 3)
    grid_w = matrix_w + legend_w + 2 + random.randint(2, 3)
    if max(grid_h, grid_w) > 30:
        return None

    legend_r0, legend_c0 = _legend_origin(corner, grid_h, grid_w, row_count, col_count)
    base_r, base_c = _matrix_base(corner, grid_h, grid_w, matrix_h, matrix_w)
    sprite_components = _build_components_from_shape(
        shape_mask, layout_cells, base_r, base_c, pitch_r, pitch_c
    )

    directions = None
    for _ in range(80):
        directions = _sample_directions(layout_cells, legend_r0, legend_c0)
        if directions is not None:
            break
    if directions is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    legend_cells = []
    for (row_rank, col_rank), (dr, dc) in zip(layout_cells, directions):
        rr = legend_r0 + row_rank + dr
        cc = legend_c0 + col_rank + dc
        if grid[rr, cc] != 0:
            return None
        grid[rr, cc] = legend_color
        legend_cells.append((rr, cc))

    wall_cells: set[tuple[int, int]] = set()
    moving_indices = [idx for idx, direction in enumerate(directions) if direction != (0, 0)]
    if not moving_indices:
        return None

    mode_by_index: dict[int, str] = {}
    for idx, comp in enumerate(sprite_components):
        direction = directions[idx]
        if direction == (0, 0):
            mode_by_index[idx] = "stay"
            continue
        feasible_boundary = _outward_feasible(
            corner,
            direction,
            comp["row_rank"],
            comp["col_rank"],
            row_count,
            col_count,
        )
        mode_by_index[idx] = "boundary" if feasible_boundary and random.random() < 0.35 else "wall"

    if all(mode_by_index[idx] != "wall" for idx in moving_indices):
        mode_by_index[random.choice(moving_indices)] = "wall"

    for idx, comp in enumerate(sprite_components):
        direction = directions[idx]
        if mode_by_index[idx] != "wall":
            continue

        r0, c0, _r1, _c1 = comp["bbox"]
        limit = _wall_step_limit(
            direction,
            comp["row_rank"],
            comp["col_rank"],
            row_count,
            col_count,
            r0,
            c0,
            shape_h,
            shape_w,
            grid_h,
            grid_w,
            pitch_r,
            pitch_c,
        )
        if limit <= 0:
            return None
        steps = random.randint(1, limit)

        if direction == (0, 1):
            wall_positions = [(rr, c0 + shape_w + steps) for rr in range(r0, r0 + shape_h)]
        elif direction == (0, -1):
            wall_positions = [(rr, c0 - steps - 1) for rr in range(r0, r0 + shape_h)]
        elif direction == (1, 0):
            wall_positions = [(r0 + shape_h + steps, cc) for cc in range(c0, c0 + shape_w)]
        else:
            wall_positions = [(r0 - steps - 1, cc) for cc in range(c0, c0 + shape_w)]

        for rr, cc in wall_positions:
            if not (0 <= rr < grid_h and 0 <= cc < grid_w):
                return None
            if grid[rr, cc] != 0 or (rr, cc) in wall_cells:
                return None
        for rr, cc in wall_positions:
            wall_cells.add((rr, cc))
            grid[rr, cc] = wall_color

    for comp in sprite_components:
        coords = comp["coords"]
        rows = coords[:, 0]
        cols = coords[:, 1]
        if bool((grid[rows, cols] != 0).any()):
            return None
        grid[rows, cols] = sprite_color

    expected = _apply_moves(grid, sprite_color, sprite_components, directions)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, sprite_components, directions


def generate_input() -> np.ndarray:
    for _ in range(400):
        shape_mask = _random_shape()
        layout_cells, row_count, col_count = _sample_layout()
        sprite_color, wall_color, legend_color = random.sample(ALL_FOREGROUND_COLORS, 3)
        corner = random.choice(CORNERS)
        built = _build_grid(
            shape_mask,
            layout_cells,
            row_count,
            col_count,
            corner,
            sprite_color,
            wall_color,
            legend_color,
        )
        if built is not None:
            grid, _components, _directions = built
            return grid
    raise RuntimeError("failed to generate a valid distorted-legend sliding puzzle")
