import random

import numpy as np


LINE_COLOR = 5
PALETTE = [0, 1, 2, 3, 4, 6, 7, 8, 9]
CELL_SIZES = (4, 5)
ROW_RANGE = (3, 6)
COL_RANGE = (3, 6)
MAX_GENERATION_ATTEMPTS = 400

DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
}
DIR_ORDER = ("up", "right", "left", "down")
BLOCKED_BY_LEGEND = {
    "up": (1, 0),
    "left": (0, 1),
}
MODE_OPTIONS = ("free", "blocked", "edge", "mixed")

SHAPES = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 1), (2, 2)),
    ((0, 2), (1, 1), (2, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
)


def _dominant_color(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[counts.argmax()])


def _periodic_cell_size(indices: np.ndarray, extent: int) -> int | None:
    if indices.size == 0:
        return None
    cell_size = int(indices[0])
    if cell_size <= 0:
        return None
    expected = np.arange(cell_size, extent, cell_size + 1)
    if not np.array_equal(indices, expected):
        return None
    if extent != cell_size * (len(indices) + 1) + len(indices):
        return None
    return cell_size


def _grid_spec(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    for color in np.unique(g):
        row_lines = np.flatnonzero(np.all(g == color, axis=1))
        col_lines = np.flatnonzero(np.all(g == color, axis=0))
        if row_lines.size == 0 or col_lines.size == 0:
            continue
        row_size = _periodic_cell_size(row_lines, g.shape[0])
        col_size = _periodic_cell_size(col_lines, g.shape[1])
        if row_size is None or col_size is None or row_size != col_size:
            continue
        return int(color), row_size, row_lines.astype(int), col_lines.astype(int)
    raise AssertionError("failed to detect separator grid")


def _cell_view(grid: np.ndarray, cell_r: int, cell_c: int, cell_size: int) -> np.ndarray:
    r0 = cell_r * (cell_size + 1)
    c0 = cell_c * (cell_size + 1)
    return grid[r0 : r0 + cell_size, c0 : c0 + cell_size]


def _background_color(grid: np.ndarray, row_lines: np.ndarray, col_lines: np.ndarray) -> int:
    mask = np.ones(grid.shape, dtype=bool)
    mask[row_lines, :] = False
    mask[:, col_lines] = False
    return _dominant_color(grid[mask])


def _direction_map(legend_cell: np.ndarray) -> dict[int, tuple[int, int]]:
    mapping = {
        int(legend_cell[0, 0]): DIRS["up"],
        int(legend_cell[0, -1]): DIRS["right"],
        int(legend_cell[-1, 0]): DIRS["left"],
        int(legend_cell[-1, -1]): DIRS["down"],
    }
    assert len(mapping) == 4, "legend must use four distinct colors"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each single-color seed pattern one cell in the direction encoded by the legend color.

    The input is partitioned by one-pixel grey separators into equal square cells. The top-left
    cell is a direction legend whose corner colors mean up, right, left, and down. Every other
    non-empty cell contains a single-color pattern; that color looks up a direction in the legend.
    The pattern is copied to the adjacent cell in that direction if that target cell is empty.
    """
    g = np.asarray(grid, dtype=int)
    _line_color, cell_size, row_lines, col_lines = _grid_spec(g)
    bg = _background_color(g, row_lines, col_lines)
    n_rows = len(row_lines) + 1
    n_cols = len(col_lines) + 1

    legend = _cell_view(g, 0, 0, cell_size)
    dir_by_color = _direction_map(legend)

    occupied: dict[tuple[int, int], tuple[int, np.ndarray]] = {}
    for cell_r in range(n_rows):
        for cell_c in range(n_cols):
            if (cell_r, cell_c) == (0, 0):
                continue
            cell = _cell_view(g, cell_r, cell_c, cell_size)
            pts = np.argwhere(cell != bg)
            if pts.size == 0:
                continue
            colors = np.unique(cell[cell != bg])
            assert len(colors) == 1, "each seed cell must contain exactly one foreground color"
            color = int(colors[0])
            assert color in dir_by_color, "seed color must appear in the legend"
            occupied[(cell_r, cell_c)] = (color, pts.astype(int))

    out = g.copy()
    requests: list[tuple[int, int, int, np.ndarray]] = []
    target_cells: set[tuple[int, int]] = set()
    for (cell_r, cell_c), (color, pts) in occupied.items():
        dr, dc = dir_by_color[color]
        target_r = cell_r + dr
        target_c = cell_c + dc
        if not (0 <= target_r < n_rows and 0 <= target_c < n_cols):
            continue
        target = _cell_view(g, target_r, target_c, cell_size)
        if np.any(target != bg):
            continue
        target_key = (target_r, target_c)
        assert target_key not in target_cells, "multiple copies into one empty target are ambiguous"
        target_cells.add(target_key)
        requests.append((target_r, target_c, color, pts))

    for target_r, target_c, color, pts in requests:
        cell = _cell_view(out, target_r, target_c, cell_size)
        cell[pts[:, 0], pts[:, 1]] = color

    return out


def _free_candidates(n_rows: int, n_cols: int, direction: str) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    out = []
    for cell_r in range(n_rows):
        for cell_c in range(n_cols):
            if (cell_r, cell_c) == (0, 0):
                continue
            target = (cell_r + dr, cell_c + dc)
            if not (0 <= target[0] < n_rows and 0 <= target[1] < n_cols):
                continue
            if target == (0, 0):
                continue
            out.append((cell_r, cell_c))
    return out


def _edge_candidates(n_rows: int, n_cols: int, direction: str) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    out = []
    for cell_r in range(n_rows):
        for cell_c in range(n_cols):
            if (cell_r, cell_c) == (0, 0):
                continue
            target = (cell_r + dr, cell_c + dc)
            if not (0 <= target[0] < n_rows and 0 <= target[1] < n_cols):
                out.append((cell_r, cell_c))
    return out


def _shape_points(cell_size: int) -> np.ndarray:
    while True:
        shape = random.choice(SHAPES)
        height = max(r for r, _ in shape) + 1
        width = max(c for _, c in shape) + 1
        if height <= cell_size and width <= cell_size:
            break
    off_r = random.randint(0, cell_size - height)
    off_c = random.randint(0, cell_size - width)
    pts = np.array([(off_r + r, off_c + c) for r, c in shape], dtype=int)
    return pts


def _mode_statuses() -> dict[str, str]:
    statuses = {direction: "free" for direction in DIR_ORDER}
    mode = random.choice(MODE_OPTIONS)
    if mode in {"blocked", "mixed"}:
        statuses[random.choice(("up", "left"))] = "blocked"
    if mode in {"edge", "mixed"}:
        choices = [d for d in DIR_ORDER if statuses[d] != "blocked"]
        statuses[random.choice(choices)] = "edge"
    return statuses


def _assign_core_positions(n_rows: int, n_cols: int) -> dict[str, tuple[int, int]] | None:
    statuses = _mode_statuses()
    positions: dict[str, tuple[int, int]] = {}
    used_positions: set[tuple[int, int]] = set()
    reserved_empty: set[tuple[int, int]] = set()

    blocked_dirs = [d for d in DIR_ORDER if statuses[d] == "blocked"]
    if blocked_dirs:
        blocked_dir = blocked_dirs[0]
        blocked_pos = BLOCKED_BY_LEGEND[blocked_dir]
        if blocked_pos[0] >= n_rows or blocked_pos[1] >= n_cols:
            return None
        positions[blocked_dir] = blocked_pos
        used_positions.add(blocked_pos)

    edge_dirs = [d for d in DIR_ORDER if statuses[d] == "edge"]
    random.shuffle(edge_dirs)
    for direction in edge_dirs:
        candidates = _edge_candidates(n_rows, n_cols, direction)
        random.shuffle(candidates)
        chosen = None
        for pos in candidates:
            if pos in used_positions or pos in reserved_empty:
                continue
            chosen = pos
            break
        if chosen is None:
            return None
        positions[direction] = chosen
        used_positions.add(chosen)

    free_dirs = [d for d in DIR_ORDER if statuses[d] == "free"]
    random.shuffle(free_dirs)
    for direction in free_dirs:
        dr, dc = DIRS[direction]
        candidates = _free_candidates(n_rows, n_cols, direction)
        random.shuffle(candidates)
        chosen = None
        for pos in candidates:
            target = (pos[0] + dr, pos[1] + dc)
            if pos in used_positions or pos in reserved_empty:
                continue
            if target in used_positions or target in reserved_empty:
                continue
            chosen = pos
            break
        if chosen is None:
            return None
        target = (chosen[0] + dr, chosen[1] + dc)
        positions[direction] = chosen
        used_positions.add(chosen)
        reserved_empty.add(target)

    return positions


def _add_extra_positions(
    positions: dict[str, tuple[int, int]],
    n_rows: int,
    n_cols: int,
) -> dict[str, tuple[int, int]]:
    used_positions = set(positions.values())
    reserved_empty = set()
    for direction, pos in positions.items():
        dr, dc = DIRS[direction]
        target = (pos[0] + dr, pos[1] + dc)
        if 0 <= target[0] < n_rows and 0 <= target[1] < n_cols and target != (0, 0) and target not in used_positions:
            reserved_empty.add(target)

    extras: dict[str, tuple[int, int]] = {}
    all_positions = [
        (cell_r, cell_c)
        for cell_r in range(n_rows)
        for cell_c in range(n_cols)
        if (cell_r, cell_c) != (0, 0)
    ]
    random.shuffle(all_positions)
    extra_quota = random.randint(0, min(3, max(0, len(all_positions) - len(used_positions) - len(reserved_empty))))
    for pos in all_positions:
        if len(extras) >= extra_quota:
            break
        if pos in used_positions or pos in reserved_empty:
            continue
        directions = list(DIR_ORDER)
        random.shuffle(directions)
        chosen = None
        for direction in directions:
            dr, dc = DIRS[direction]
            target = (pos[0] + dr, pos[1] + dc)
            if 0 <= target[0] < n_rows and 0 <= target[1] < n_cols:
                if target == (0, 0):
                    chosen = direction
                    break
                if target in reserved_empty:
                    continue
                if target not in used_positions:
                    chosen = direction
                    break
                chosen = direction
                break
            chosen = direction
            break
        if chosen is None:
            continue
        extras[f"extra_{len(extras)}:{chosen}"] = pos
        used_positions.add(pos)
        dr, dc = DIRS[chosen]
        target = (pos[0] + dr, pos[1] + dc)
        if 0 <= target[0] < n_rows and 0 <= target[1] < n_cols and target != (0, 0) and target not in used_positions:
            reserved_empty.add(target)
    return extras


def _extract_direction(label: str) -> str:
    if label in DIRS:
        return label
    return label.split(":", 1)[1]


def _render_grid(
    n_rows: int,
    n_cols: int,
    cell_size: int,
    bg: int,
    legend_colors: dict[str, int],
    placements: dict[str, tuple[int, int]],
) -> np.ndarray:
    height = n_rows * cell_size + (n_rows - 1)
    width = n_cols * cell_size + (n_cols - 1)
    grid = np.full((height, width), bg, dtype=int)

    for row_line in range(cell_size, height, cell_size + 1):
        grid[row_line, :] = LINE_COLOR
    for col_line in range(cell_size, width, cell_size + 1):
        grid[:, col_line] = LINE_COLOR

    legend = _cell_view(grid, 0, 0, cell_size)
    split = cell_size // 2
    legend[:split, :split] = legend_colors["up"]
    legend[:split, split:] = legend_colors["right"]
    legend[split:, :split] = legend_colors["left"]
    legend[split:, split:] = legend_colors["down"]

    for label, (cell_r, cell_c) in placements.items():
        direction = _extract_direction(label)
        color = legend_colors[direction]
        cell = _cell_view(grid, cell_r, cell_c, cell_size)
        pts = _shape_points(cell_size)
        cell[pts[:, 0], pts[:, 1]] = color

    return grid


def _count_statuses(
    placements: dict[str, tuple[int, int]],
    n_rows: int,
    n_cols: int,
) -> tuple[int, int, int]:
    used_positions = set(placements.values())
    free = blocked = edge = 0
    for label, pos in placements.items():
        direction = _extract_direction(label)
        dr, dc = DIRS[direction]
        target = (pos[0] + dr, pos[1] + dc)
        if not (0 <= target[0] < n_rows and 0 <= target[1] < n_cols):
            edge += 1
        elif target == (0, 0) or target in used_positions:
            blocked += 1
        else:
            free += 1
    return free, blocked, edge


def _gen_attempt() -> np.ndarray | None:
    cell_size = random.choice(CELL_SIZES)
    n_rows = random.randint(*ROW_RANGE)
    n_cols = random.randint(*COL_RANGE)
    bg = random.choice(PALETTE)
    legend_palette = random.sample([c for c in PALETTE if c != bg], 4)
    legend_colors = dict(zip(DIR_ORDER, legend_palette))

    core = _assign_core_positions(n_rows, n_cols)
    if core is None:
        return None
    placements = {**core, **_add_extra_positions(core, n_rows, n_cols)}

    free, blocked, edge = _count_statuses(placements, n_rows, n_cols)
    if free == 0:
        return None

    grid = _render_grid(n_rows, n_cols, cell_size, bg, legend_colors, placements)
    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    if blocked == 0 and edge == 0 and len(placements) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
