import random
from collections import Counter, defaultdict

import numpy as np


BG_COLOR = 0
CROSS_COLOR = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)

DIR_RIGHT = (0, 1)
DIR_DOWN = (1, 0)
DIR_UP = (-1, 0)
DIR_LEFT = (0, -1)
ALL_DIRS = (DIR_RIGHT, DIR_DOWN, DIR_UP, DIR_LEFT)

MAX_GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _direction_for_seed(row: int, col: int, height: int, width: int) -> tuple[int, int]:
    center_row = height // 2
    center_col = width // 2
    # Half-open quadrant split:
    # center-column cells count as the right half, center-row cells as the lower half.
    if col < center_col:
        return DIR_RIGHT if row < center_row else DIR_UP
    return DIR_DOWN if row < center_row else DIR_LEFT


def _ray_cells(
    row: int, col: int, direction: tuple[int, int], steps: int, height: int, width: int
) -> list[tuple[int, int]]:
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    r, c = row, col
    for _ in range(int(steps)):
        r += dr
        c += dc
        if not (0 <= r < height and 0 <= c < width):
            break
        cells.append((int(r), int(c)))
    return cells


def _distance_to_edge(
    row: int, col: int, direction: tuple[int, int], height: int, width: int
) -> int:
    dr, dc = direction
    if dr == 1:
        return int(height - 1 - row)
    if dr == -1:
        return int(row)
    if dc == 1:
        return int(width - 1 - col)
    return int(col)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw quadrant-directed rays whose lengths equal each color's seed count, marking cross-color collisions in yellow.

    Input structure:
    - The background is black and the foreground consists only of isolated single-pixel colored seeds.
    - The same non-background color may appear on multiple seeds, and those repeated seeds are the only clue for ray length.
    - The visual center at (rows//2, cols//2) implicitly splits the canvas into directional quadrants.

    Rule:
    - Each seed grows one straight ray: top-left -> right, top-right -> down, bottom-left -> up, bottom-right -> left.
    - To break center-line ties, seeds on the center column count as the right half and seeds on the center row count as the lower half.
    - Every ray length equals the total number of seeds of that same color; rays advance step by step, same-color overlaps are allowed, and any cross-color contact cell becomes yellow while the newly arriving ray stops there.

    Key insight:
    - Position decides direction, same-color multiplicity decides length, and yellow appears only where different-colored rays interfere.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    height, width = g.shape
    bg = _most_common_color(g)
    seeds = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere(g != bg)]
    if not seeds:
        return g.copy()

    color_counts = Counter(color for _r, _c, color in seeds)
    rays = []
    for row, col, color in seeds:
        rays.append(
            {
                "color": int(color),
                "row": int(row),
                "col": int(col),
                "dir": _direction_for_seed(int(row), int(col), height, width),
                "remaining": int(color_counts[int(color)]),
                "active": True,
            }
        )

    out = g.copy()
    while True:
        proposals: dict[tuple[int, int], list[int]] = defaultdict(list)
        next_pos: dict[int, tuple[int, int]] = {}

        for idx, ray in enumerate(rays):
            if not ray["active"] or int(ray["remaining"]) <= 0:
                continue
            dr, dc = ray["dir"]
            nr = int(ray["row"] + dr)
            nc = int(ray["col"] + dc)
            if not (0 <= nr < height and 0 <= nc < width):
                ray["active"] = False
                continue
            next_pos[idx] = (int(nr), int(nc))
            proposals[(int(nr), int(nc))].append(int(idx))

        if not proposals:
            break

        advancing: set[int] = set()
        stopping: set[int] = set()

        for cell, idxs in proposals.items():
            row, col = cell
            occ = int(out[row, col])
            proposal_colors = {int(rays[idx]["color"]) for idx in idxs}

            if occ == int(bg):
                if len(proposal_colors) == 1:
                    out[row, col] = int(next(iter(proposal_colors)))
                    advancing.update(int(idx) for idx in idxs)
                else:
                    out[row, col] = CROSS_COLOR
                    stopping.update(int(idx) for idx in idxs)
                continue

            if occ == CROSS_COLOR:
                out[row, col] = CROSS_COLOR
                stopping.update(int(idx) for idx in idxs)
                continue

            same = [int(idx) for idx in idxs if int(rays[idx]["color"]) == occ]
            different = [int(idx) for idx in idxs if int(rays[idx]["color"]) != occ]
            if different:
                out[row, col] = CROSS_COLOR
                stopping.update(different)
            advancing.update(same)

        for idx in sorted(advancing):
            row, col = next_pos[idx]
            ray = rays[idx]
            ray["row"] = int(row)
            ray["col"] = int(col)
            ray["remaining"] = int(ray["remaining"] - 1)
            if int(ray["remaining"]) <= 0:
                ray["active"] = False

        for idx in sorted(stopping):
            if idx in advancing:
                continue
            row, col = next_pos[idx]
            ray = rays[idx]
            ray["row"] = int(row)
            ray["col"] = int(col)
            ray["remaining"] = int(ray["remaining"] - 1)
            ray["active"] = False

    return out


def _candidate_positions(
    height: int,
    width: int,
    direction: tuple[int, int],
    length: int,
    *,
    require_full: bool,
    require_clip: bool,
    force_line: bool,
) -> list[tuple[int, int]]:
    center_row = height // 2
    center_col = width // 2
    positions: list[tuple[int, int]] = []
    for row in range(height):
        for col in range(width):
            if _direction_for_seed(row, col, height, width) != direction:
                continue
            on_line = row == center_row or col == center_col
            if force_line and not on_line:
                continue
            dist = _distance_to_edge(row, col, direction, height, width)
            if dist <= 0:
                continue
            if require_full and dist < length:
                continue
            if require_clip and dist >= length:
                continue
            positions.append((int(row), int(col)))
    random.shuffle(positions)
    return positions


def _can_place_seed(
    height: int,
    width: int,
    row: int,
    col: int,
    color: int,
    length: int,
    seed_cells: set[tuple[int, int]],
    path_map: dict[tuple[int, int], set[int]],
    *,
    allow_collision: tuple[int, int] | None = None,
) -> tuple[bool, list[tuple[int, int]]]:
    if (row, col) in seed_cells:
        return False, []
    if (row, col) in path_map:
        return False, []

    direction = _direction_for_seed(row, col, height, width)
    path = _ray_cells(row, col, direction, length, height, width)
    if not path:
        return False, []

    saw_allowed_collision = allow_collision is None
    for cell in path:
        if cell in seed_cells:
            return False, []
        existing = path_map.get(cell)
        if not existing:
            continue
        if allow_collision is not None and cell == allow_collision and any(
            int(other) != int(color) for other in existing
        ):
            if len(existing) != 1:
                return False, []
            saw_allowed_collision = True
            continue
        return False, []

    if not saw_allowed_collision:
        return False, []
    return True, path


def _commit_seed(
    seeds: list[dict[str, int | tuple[int, int] | list[tuple[int, int]]]],
    seed_cells: set[tuple[int, int]],
    path_map: dict[tuple[int, int], set[int]],
    height: int,
    width: int,
    row: int,
    col: int,
    color: int,
    path: list[tuple[int, int]],
) -> dict[str, int | tuple[int, int] | list[tuple[int, int]]]:
    record = {
        "row": int(row),
        "col": int(col),
        "color": int(color),
        "dir": _direction_for_seed(row, col, height, width),
        "path": list(path),
    }
    seeds.append(record)
    seed_cells.add((int(row), int(col)))
    for cell in path:
        path_map.setdefault(cell, set()).add(int(color))
    return record


def _place_seed_with_direction(
    seeds: list[dict[str, int | tuple[int, int] | list[tuple[int, int]]]],
    seed_cells: set[tuple[int, int]],
    path_map: dict[tuple[int, int], set[int]],
    height: int,
    width: int,
    color: int,
    length: int,
    direction: tuple[int, int],
    *,
    require_full: bool,
    require_clip: bool,
    force_line: bool,
) -> dict[str, int | tuple[int, int] | list[tuple[int, int]]] | None:
    positions = _candidate_positions(
        height,
        width,
        direction,
        length,
        require_full=require_full,
        require_clip=require_clip,
        force_line=force_line,
    )
    for row, col in positions:
        ok, path = _can_place_seed(
            height, width, row, col, color, length, seed_cells, path_map
        )
        if ok:
            return _commit_seed(
                seeds, seed_cells, path_map, height, width, row, col, color, path
            )
    return None


def _place_collision_pair(
    seeds: list[dict[str, int | tuple[int, int] | list[tuple[int, int]]]],
    seed_cells: set[tuple[int, int]],
    path_map: dict[tuple[int, int], set[int]],
    height: int,
    width: int,
    color_a: int,
    length_a: int,
    color_b: int,
    length_b: int,
    *,
    want_line: bool,
) -> list[dict[str, int | tuple[int, int] | list[tuple[int, int]]]] | None:
    center_row = height // 2
    center_col = width // 2

    pair_types = ["right_half", "left_half"]
    random.shuffle(pair_types)

    for pair_type in pair_types:
        for _ in range(120):
            dist_a = random.randint(1, int(length_a))
            dist_b = random.randint(1, int(length_b))

            if pair_type == "right_half":
                line_modes = ["none"]
                if want_line:
                    line_modes += ["row_center", "col_center"]
                line_mode = random.choice(line_modes)

                r_min = center_row
                r_max = min(height - 1, center_row + dist_b - 1)
                c_min = center_col
                c_max = width - 1 - dist_a
                if r_min > r_max or c_min > c_max:
                    continue
                if line_mode == "row_center":
                    r_int = center_row
                    if not (r_min <= r_int <= r_max):
                        continue
                    c_int = random.randint(c_min, c_max)
                elif line_mode == "col_center":
                    c_int = center_col
                    if not (c_min <= c_int <= c_max):
                        continue
                    r_int = random.randint(r_min, r_max)
                else:
                    r_int = random.randint(r_min, r_max)
                    c_int = random.randint(c_min, c_max)

                row_a, col_a = r_int, c_int + dist_a
                row_b, col_b = r_int - dist_b, c_int
            else:
                line_modes = ["none"]
                if want_line:
                    line_modes += ["seed_up_on_center_row"]
                line_mode = random.choice(line_modes)

                r_min = max(0, center_row - dist_b)
                r_max = center_row - 1
                c_min = dist_a
                c_max = center_col - 1
                if r_min > r_max or c_min > c_max:
                    continue
                if line_mode == "seed_up_on_center_row":
                    r_int = center_row - dist_b
                    if not (r_min <= r_int <= r_max):
                        continue
                    c_int = random.randint(c_min, c_max)
                else:
                    r_int = random.randint(r_min, r_max)
                    c_int = random.randint(c_min, c_max)

                row_a, col_a = r_int, c_int - dist_a
                row_b, col_b = r_int + dist_b, c_int

            dir_a = _direction_for_seed(row_a, col_a, height, width)
            dir_b = _direction_for_seed(row_b, col_b, height, width)
            expected_a = DIR_LEFT if pair_type == "right_half" else DIR_RIGHT
            expected_b = DIR_DOWN if pair_type == "right_half" else DIR_UP
            if dir_a != expected_a or dir_b != expected_b:
                continue
            if _distance_to_edge(row_a, col_a, dir_a, height, width) < int(length_a):
                continue
            if _distance_to_edge(row_b, col_b, dir_b, height, width) < int(length_b):
                continue

            ok_a, path_a = _can_place_seed(
                height, width, row_a, col_a, color_a, length_a, seed_cells, path_map
            )
            if not ok_a:
                continue

            tmp_seed_cells = set(seed_cells)
            tmp_path_map = {cell: set(colors) for cell, colors in path_map.items()}
            tmp_seed_cells.add((int(row_a), int(col_a)))
            for cell in path_a:
                tmp_path_map.setdefault(cell, set()).add(int(color_a))

            intersection = (int(r_int), int(c_int))
            ok_b, path_b = _can_place_seed(
                height,
                width,
                row_b,
                col_b,
                color_b,
                length_b,
                tmp_seed_cells,
                tmp_path_map,
                allow_collision=intersection,
            )
            if not ok_b:
                continue

            rec_a = _commit_seed(
                seeds, seed_cells, path_map, height, width, row_a, col_a, color_a, path_a
            )
            rec_b = _commit_seed(
                seeds, seed_cells, path_map, height, width, row_b, col_b, color_b, path_b
            )
            return [rec_a, rec_b]

    return None


def _build_grid(
    height: int,
    width: int,
    seeds: list[dict[str, int | tuple[int, int] | list[tuple[int, int]]]],
) -> np.ndarray:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    for seed in seeds:
        grid[int(seed["row"]), int(seed["col"])] = int(seed["color"])
    return grid


def _validate_grid(
    grid: np.ndarray,
    *,
    expect_collision: bool,
    expect_line: bool,
    expect_multi: bool,
) -> bool:
    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.count_nonzero(out != grid) == 0:
        return False

    bg = _most_common_color(grid)
    seeds = np.argwhere(grid != bg)
    center_row = grid.shape[0] // 2
    center_col = grid.shape[1] // 2
    counts = Counter(int(grid[r, c]) for r, c in seeds)

    if expect_collision and not np.any(out == CROSS_COLOR):
        return False
    if expect_line and not any(
        int(r) == center_row or int(c) == center_col for r, c in seeds
    ):
        return False
    if expect_multi and not any(int(v) > 1 for v in counts.values()):
        return False
    return True


def _gen_standard() -> np.ndarray | None:
    height = random.randint(15, 25)
    width = random.randint(15, 25)
    num_colors = random.randint(2, 4)
    colors = random.sample(PALETTE, num_colors)

    counts = {int(color): random.randint(1, 3) for color in colors}
    if not any(int(v) > 1 for v in counts.values()):
        chosen = random.choice(colors)
        counts[int(chosen)] = random.randint(2, 3)

    seeds: list[dict[str, int | tuple[int, int] | list[tuple[int, int]]]] = []
    seed_cells: set[tuple[int, int]] = set()
    path_map: dict[tuple[int, int], set[int]] = {}
    remaining = {int(color): int(count) for color, count in counts.items()}

    direction_usage: Counter[tuple[int, int]] = Counter()
    expect_collision = random.random() < 0.6
    want_line = random.random() < 0.45
    line_used = False

    if expect_collision:
        color_a, color_b = random.sample(colors, 2)
        pair = _place_collision_pair(
            seeds,
            seed_cells,
            path_map,
            height,
            width,
            int(color_a),
            int(counts[int(color_a)]),
            int(color_b),
            int(counts[int(color_b)]),
            want_line=want_line,
        )
        if pair is None:
            return None
        for rec in pair:
            remaining[int(rec["color"])] -= 1
            direction_usage[rec["dir"]] += 1
            if int(rec["row"]) == height // 2 or int(rec["col"]) == width // 2:
                line_used = True

    extra_colors: list[int] = []
    for color, count in remaining.items():
        extra_colors.extend([int(color)] * int(count))
    random.shuffle(extra_colors)
    extra_colors.sort(key=lambda color: -counts[int(color)])

    if want_line and not line_used and extra_colors:
        line_color = random.choice(extra_colors)
        line_dirs = [DIR_DOWN, DIR_UP, DIR_LEFT]
        random.shuffle(line_dirs)
        placed = None
        for direction in line_dirs:
            placed = _place_seed_with_direction(
                seeds,
                seed_cells,
                path_map,
                height,
                width,
                int(line_color),
                int(counts[int(line_color)]),
                direction,
                require_full=True,
                require_clip=False,
                force_line=True,
            )
            if placed is not None:
                break
        if placed is None:
            return None
        remaining[int(line_color)] -= 1
        direction_usage[placed["dir"]] += 1
        extra_colors.remove(int(line_color))
        line_used = True

    for color in list(extra_colors):
        length = int(counts[int(color)])
        directions = list(ALL_DIRS)
        random.shuffle(directions)
        directions.sort(key=lambda d: int(direction_usage[d]))
        placed = None
        for direction in directions:
            placed = _place_seed_with_direction(
                seeds,
                seed_cells,
                path_map,
                height,
                width,
                int(color),
                length,
                direction,
                require_full=True,
                require_clip=False,
                force_line=False,
            )
            if placed is not None:
                break
        if placed is None:
            return None
        remaining[int(color)] -= 1
        direction_usage[placed["dir"]] += 1

    if any(int(v) != 0 for v in remaining.values()):
        return None

    grid = _build_grid(height, width, seeds)
    if not _validate_grid(
        grid,
        expect_collision=expect_collision,
        expect_line=want_line,
        expect_multi=True,
    ):
        return None
    return grid


def _gen_boundary() -> np.ndarray | None:
    size = random.choice((15, 17, 19))
    height = size
    width = size
    center_row = height // 2
    center_col = width // 2

    heavy_color = random.choice(PALETTE)
    extra_palette = [color for color in PALETTE if int(color) != int(heavy_color)]
    extra_count = random.choice((1, 1, 2))
    extra_colors = random.sample(extra_palette, random.choice((1, 2)))

    counts = {int(heavy_color): int(center_row + 1)}
    for color in extra_colors:
        counts[int(color)] = random.randint(1, extra_count)

    seeds: list[dict[str, int | tuple[int, int] | list[tuple[int, int]]]] = []
    seed_cells: set[tuple[int, int]] = set()
    path_map: dict[tuple[int, int], set[int]] = {}
    direction_usage: Counter[tuple[int, int]] = Counter()

    clip_positions = [(center_row, center_col)]
    rows = list(range(center_row + 1, height))
    random.shuffle(rows)
    full_cols = list(range(center_col + 1, width))
    random.shuffle(full_cols)
    for row, col in zip(rows, full_cols):
        clip_positions.append((int(row), int(col)))

    if len(clip_positions) != int(counts[int(heavy_color)]):
        return None

    for row, col in clip_positions:
        ok, path = _can_place_seed(
            height,
            width,
            int(row),
            int(col),
            int(heavy_color),
            int(counts[int(heavy_color)]),
            seed_cells,
            path_map,
        )
        if not ok:
            return None
        rec = _commit_seed(
            seeds,
            seed_cells,
            path_map,
            height,
            width,
            int(row),
            int(col),
            int(heavy_color),
            path,
        )
        direction_usage[rec["dir"]] += 1

    for color in extra_colors:
        for _ in range(int(counts[int(color)])):
            directions = list(ALL_DIRS)
            random.shuffle(directions)
            directions.sort(key=lambda d: int(direction_usage[d]))
            placed = None
            for direction in directions:
                placed = _place_seed_with_direction(
                    seeds,
                    seed_cells,
                    path_map,
                    height,
                    width,
                    int(color),
                    int(counts[int(color)]),
                    direction,
                    require_full=True,
                    require_clip=False,
                    force_line=False,
                )
                if placed is not None:
                    break
            if placed is None:
                return None
            direction_usage[placed["dir"]] += 1

    grid = _build_grid(height, width, seeds)
    if not _validate_grid(
        grid,
        expect_collision=False,
        expect_line=True,
        expect_multi=True,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        if random.random() < 0.3:
            grid = _gen_boundary()
        else:
            grid = _gen_standard()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
