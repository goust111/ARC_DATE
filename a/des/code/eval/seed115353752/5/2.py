from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GRID_H_RANGE = (13, 20)
GRID_W_RANGE = (13, 20)
BAR_LENGTH_RANGE = (2, 6)
GEN_ATTEMPTS = 400

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
VERTICAL_DIRS = (UP, DOWN)
HORIZONTAL_DIRS = (LEFT, RIGHT)
DIRS = (UP, DOWN, LEFT, RIGHT)


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _scan_runs(line: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    idx = 0
    n = int(line.size)
    while idx < n:
        color = int(line[idx])
        if color == bg:
            idx += 1
            continue
        j = idx + 1
        while j < n and int(line[j]) == color:
            j += 1
        if j - idx >= 2:
            runs.append((idx, j - idx, color))
        idx = j
    return runs


def _extract_nutrient_lengths(grid: np.ndarray, bg: int) -> dict[int, int]:
    by_color: dict[int, int] = {}
    edge_lines = (
        grid[0, :],
        grid[-1, :],
        grid[:, 0],
        grid[:, -1],
    )
    for line in edge_lines:
        for _start, length, color in _scan_runs(line, bg):
            if color in by_color:
                assert by_color[color] == length, "matching edge bars must share a length"
            else:
                by_color[color] = int(length)
    assert by_color, "expected at least one nutrient bar"
    return by_color


def _parse_seeds(grid: np.ndarray, bg: int, nutrient_lengths: dict[int, int]) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    nutrient_colors = set(nutrient_lengths)
    seeds: list[dict[str, object]] = []
    seen_colors: set[int] = set()

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(grid[r, c])
            if color == bg or color in nutrient_colors:
                continue

            non_bg_neighbors = [(nr, nc) for nr, nc in _neighbors4(r, c) if int(grid[nr, nc]) != bg]
            assert len(non_bg_neighbors) == 1, "each seed must have exactly one colored 4-neighbor"
            mr, mc = non_bg_neighbors[0]
            marker_color = int(grid[mr, mc])
            assert marker_color in nutrient_colors, "seed marker must match an edge-bar color"
            assert color not in seen_colors, "seed colors must be unique"
            seen_colors.add(color)

            seeds.append(
                {
                    "seed": (int(r), int(c)),
                    "seed_color": color,
                    "marker": (int(mr), int(mc)),
                    "marker_color": marker_color,
                    "direction": (int(mr - r), int(mc - c)),
                }
            )

    assert seeds, "expected at least one seed"
    return seeds


def _ray_cells(
    seed: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], bool]:
    h, w = map(int, shape)
    dr, dc = map(int, direction)
    cells: list[tuple[int, int]] = []
    max_step = 0

    for step in range(1, int(length) + 1):
        rr = int(seed[0] + dr * step)
        cc = int(seed[1] + dc * step)
        if not (0 <= rr < h and 0 <= cc < w):
            break
        max_step = step
        if step >= 2:
            cells.append((rr, cc))

    return cells, max_step < int(length)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a seed-colored ray in the direction of each adjacent marker, using the
    matching edge bar's length as the ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _mode_color(g)
    nutrient_lengths = _extract_nutrient_lengths(g, bg)
    seeds = sorted(_parse_seeds(g, bg, nutrient_lengths), key=lambda item: item["seed"])

    out = g.copy()
    for spec in seeds:
        seed_r, seed_c = spec["seed"]
        dr, dc = spec["direction"]
        length = int(nutrient_lengths[int(spec["marker_color"])])
        seed_color = int(spec["seed_color"])

        for step in range(1, length + 1):
            rr = int(seed_r + dr * step)
            cc = int(seed_c + dc * step)
            if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                break
            if int(g[rr, cc]) == bg:
                out[rr, cc] = seed_color

    return out


def _bar_coords(edge: str, start: int, length: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    if edge == "top":
        return [(0, c) for c in range(start, start + length)]
    if edge == "bottom":
        return [(h - 1, c) for c in range(start, start + length)]
    if edge == "left":
        return [(r, 0) for r in range(start, start + length)]
    if edge == "right":
        return [(r, w - 1) for r in range(start, start + length)]
    raise ValueError(edge)


def _paint_bar(grid: np.ndarray, edge: str, start: int, length: int, color: int) -> None:
    for r, c in _bar_coords(edge, start, length, grid.shape):
        grid[r, c] = int(color)


def _is_interior(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    r, c = map(int, cell)
    return 1 <= r < h - 1 and 1 <= c < w - 1


def _can_place_seed_marker(
    seed: tuple[int, int],
    marker: tuple[int, int],
    immutable: np.ndarray,
) -> bool:
    if not _is_interior(seed, immutable.shape) or not _is_interior(marker, immutable.shape):
        return False
    if immutable[seed] or immutable[marker]:
        return False
    if abs(int(seed[0] - marker[0])) + abs(int(seed[1] - marker[1])) != 1:
        return False

    for nr, nc in _neighbors4(*seed):
        if (nr, nc) == marker:
            continue
        if immutable[nr, nc]:
            return False
    for nr, nc in _neighbors4(*marker):
        if (nr, nc) == seed:
            continue
        if immutable[nr, nc]:
            return False
    return True


def _candidate_spec(
    seed: tuple[int, int],
    direction: tuple[int, int],
    nutrient_color: int,
    length: int,
    immutable: np.ndarray,
    occupied_path: set[tuple[int, int]],
    require_truncation: bool,
) -> dict[str, object] | None:
    marker = (int(seed[0] + direction[0]), int(seed[1] + direction[1]))
    if not _can_place_seed_marker(seed, marker, immutable):
        return None

    path, truncated = _ray_cells(seed, direction, int(length), immutable.shape)
    if truncated != bool(require_truncation):
        return None
    if any(immutable[cell] for cell in path):
        return None
    if any(cell in occupied_path for cell in path):
        return None

    return {
        "seed": (int(seed[0]), int(seed[1])),
        "marker": marker,
        "direction": (int(direction[0]), int(direction[1])),
        "nutrient_color": int(nutrient_color),
        "length": int(length),
        "path": set(path),
        "truncated": bool(truncated),
    }


def _try_place_single(
    nutrient_lengths: dict[int, int],
    immutable: np.ndarray,
    occupied_path: set[tuple[int, int]],
    require_truncation: bool,
) -> dict[str, object] | None:
    h, w = map(int, immutable.shape)
    seeds = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(seeds)

    bars = list(nutrient_lengths.items())
    random.shuffle(bars)
    directions = list(DIRS)
    random.shuffle(directions)

    for nutrient_color, length in bars:
        if require_truncation and int(length) < 3:
            continue
        for direction in directions:
            for seed in seeds:
                spec = _candidate_spec(
                    seed,
                    direction,
                    int(nutrient_color),
                    int(length),
                    immutable,
                    occupied_path,
                    require_truncation,
                )
                if spec is not None:
                    return spec
    return None


def _try_place_cross_pair(
    nutrient_lengths: dict[int, int],
    immutable: np.ndarray,
) -> list[dict[str, object]] | None:
    h, w = map(int, immutable.shape)
    cross_cells = [(r, c) for r in range(2, h - 2) for c in range(2, w - 2)]
    if not cross_cells:
        return None

    bar_colors = list(nutrient_lengths)
    for _ in range(250):
        vertical_color, horizontal_color = random.sample(bar_colors, 2)
        vertical_length = int(nutrient_lengths[vertical_color])
        horizontal_length = int(nutrient_lengths[horizontal_color])
        vertical_dir = random.choice(VERTICAL_DIRS)
        horizontal_dir = random.choice(HORIZONTAL_DIRS)
        vertical_step = random.randint(2, vertical_length)
        horizontal_step = random.randint(2, horizontal_length)

        positions = cross_cells[:]
        random.shuffle(positions)
        for cross in positions:
            seed_v = (
                int(cross[0] - vertical_dir[0] * vertical_step),
                int(cross[1] - vertical_dir[1] * vertical_step),
            )
            seed_h = (
                int(cross[0] - horizontal_dir[0] * horizontal_step),
                int(cross[1] - horizontal_dir[1] * horizontal_step),
            )
            marker_v = (int(seed_v[0] + vertical_dir[0]), int(seed_v[1] + vertical_dir[1]))
            marker_h = (int(seed_h[0] + horizontal_dir[0]), int(seed_h[1] + horizontal_dir[1]))

            if len({seed_v, marker_v, seed_h, marker_h}) != 4:
                continue
            if not _can_place_seed_marker(seed_v, marker_v, immutable):
                continue

            temp_immutable = immutable.copy()
            temp_immutable[seed_v] = True
            temp_immutable[marker_v] = True
            if not _can_place_seed_marker(seed_h, marker_h, temp_immutable):
                continue

            path_v, trunc_v = _ray_cells(seed_v, vertical_dir, vertical_length, immutable.shape)
            path_h, trunc_h = _ray_cells(seed_h, horizontal_dir, horizontal_length, immutable.shape)
            if trunc_v or trunc_h:
                continue

            set_v = set(path_v)
            set_h = set(path_h)
            pair_points = {seed_v, marker_v, seed_h, marker_h}
            if pair_points & (set_v | set_h):
                continue
            if any(immutable[cell] for cell in (set_v | set_h)):
                continue
            if (set_v & set_h) != {cross}:
                continue

            return [
                {
                    "seed": seed_v,
                    "marker": marker_v,
                    "direction": vertical_dir,
                    "nutrient_color": int(vertical_color),
                    "length": vertical_length,
                    "path": set_v,
                    "truncated": False,
                },
                {
                    "seed": seed_h,
                    "marker": marker_h,
                    "direction": horizontal_dir,
                    "nutrient_color": int(horizontal_color),
                    "length": horizontal_length,
                    "path": set_h,
                    "truncated": False,
                },
            ]

    return None


def _apply_spec(grid: np.ndarray, spec: dict[str, object]) -> None:
    seed_r, seed_c = spec["seed"]
    marker_r, marker_c = spec["marker"]
    grid[seed_r, seed_c] = int(spec["seed_color"])
    grid[marker_r, marker_c] = int(spec["nutrient_color"])


def _validate_attempt(
    grid: np.ndarray,
    specs: list[dict[str, object]],
    want_cross: bool,
    want_truncation: bool,
) -> bool:
    if _mode_color(grid) != BG:
        return False

    edge_lines = (grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1])
    if not all(_scan_runs(line, BG) for line in edge_lines):
        return False

    try:
        nutrient_lengths = _extract_nutrient_lengths(grid, BG)
        parsed = _parse_seeds(grid, BG, nutrient_lengths)
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape or np.array_equal(out, grid):
        return False
    if len(nutrient_lengths) != 4 or len(parsed) != len(specs):
        return False

    parsed_by_seed = {tuple(item["seed"]): item for item in parsed}
    path_sets: list[set[tuple[int, int]]] = []
    trunc_flags: list[bool] = []
    for spec in specs:
        parsed_spec = parsed_by_seed.get(tuple(spec["seed"]))
        if parsed_spec is None:
            return False
        if tuple(parsed_spec["marker"]) != tuple(spec["marker"]):
            return False
        if tuple(parsed_spec["direction"]) != tuple(spec["direction"]):
            return False
        if int(parsed_spec["marker_color"]) != int(spec["nutrient_color"]):
            return False

        path, truncated = _ray_cells(
            tuple(spec["seed"]),
            tuple(spec["direction"]),
            int(spec["length"]),
            grid.shape,
        )
        path_sets.append(set(path))
        trunc_flags.append(bool(truncated))

    has_cross = any(path_sets[i] & path_sets[j] for i in range(len(path_sets)) for j in range(i + 1, len(path_sets)))
    if has_cross != bool(want_cross):
        return False
    if any(trunc_flags) != bool(want_truncation):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    bar_colors = random.sample(PALETTE, 4)
    seed_colors = [color for color in PALETTE if color not in bar_colors]
    nutrient_lengths = {
        bar_colors[0]: random.randint(*BAR_LENGTH_RANGE),
        bar_colors[1]: random.randint(*BAR_LENGTH_RANGE),
        bar_colors[2]: random.randint(*BAR_LENGTH_RANGE),
        bar_colors[3]: random.randint(*BAR_LENGTH_RANGE),
    }
    bar_specs = (
        ("top", bar_colors[0], nutrient_lengths[bar_colors[0]], random.randint(1, w - nutrient_lengths[bar_colors[0]] - 1)),
        ("bottom", bar_colors[1], nutrient_lengths[bar_colors[1]], random.randint(1, w - nutrient_lengths[bar_colors[1]] - 1)),
        ("left", bar_colors[2], nutrient_lengths[bar_colors[2]], random.randint(1, h - nutrient_lengths[bar_colors[2]] - 1)),
        ("right", bar_colors[3], nutrient_lengths[bar_colors[3]], random.randint(1, h - nutrient_lengths[bar_colors[3]] - 1)),
    )

    for edge, color, length, start in bar_specs:
        _paint_bar(grid, edge, int(start), int(length), int(color))

    want_cross = random.random() < 0.5
    want_truncation = random.random() < 0.5
    if want_truncation and not any(length >= 3 for length in nutrient_lengths.values()):
        return None

    required = (2 if want_cross else 0) + (1 if want_truncation else 0)
    total_seeds = random.randint(max(2, required), len(seed_colors))
    chosen_seed_colors = random.sample(seed_colors, total_seeds)

    immutable = grid != BG
    occupied_path: set[tuple[int, int]] = set()
    specs: list[dict[str, object]] = []

    if want_cross:
        pair = _try_place_cross_pair(nutrient_lengths, immutable)
        if pair is None:
            return None
        for spec in pair:
            seed_r, seed_c = spec["seed"]
            marker_r, marker_c = spec["marker"]
            immutable[seed_r, seed_c] = True
            immutable[marker_r, marker_c] = True
            occupied_path.update(spec["path"])
        specs.extend(pair)

    if want_truncation:
        spec = _try_place_single(nutrient_lengths, immutable, occupied_path, require_truncation=True)
        if spec is None:
            return None
        seed_r, seed_c = spec["seed"]
        marker_r, marker_c = spec["marker"]
        immutable[seed_r, seed_c] = True
        immutable[marker_r, marker_c] = True
        occupied_path.update(spec["path"])
        specs.append(spec)

    while len(specs) < total_seeds:
        spec = _try_place_single(nutrient_lengths, immutable, occupied_path, require_truncation=False)
        if spec is None:
            return None
        seed_r, seed_c = spec["seed"]
        marker_r, marker_c = spec["marker"]
        immutable[seed_r, seed_c] = True
        immutable[marker_r, marker_c] = True
        occupied_path.update(spec["path"])
        specs.append(spec)

    for spec, seed_color in zip(specs, chosen_seed_colors):
        spec["seed_color"] = int(seed_color)
        _apply_spec(grid, spec)

    if not _validate_attempt(grid, specs, want_cross, want_truncation):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
