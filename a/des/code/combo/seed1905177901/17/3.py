from __future__ import annotations

import random

import numpy as np


BLACK = 0
FRAME = 5
PALETTE = tuple(c for c in range(10) if c not in (BLACK, FRAME))

GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (16, 24)
LEGEND_LEN_RANGE = (4, 6)
CONTAINER_COUNT_RANGE = (2, 5)
MARKER_COUNT_RANGE = (2, 4)
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as Nx2 coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_hollow_rectangle(grid: np.ndarray, coords: np.ndarray, color: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    border_size = 2 * (h + w) - 4
    if len(coords) != border_size:
        return False
    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(window[border] == color):
        return False
    if np.any(window[~border] == color):
        return False
    return True


def _find_containers(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    containers: list[tuple[int, int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BLACK:
            continue
        for coords in _components(grid == color):
            if _is_hollow_rectangle(grid, coords, color):
                r0, c0, r1, c1 = _bbox(coords)
                containers.append((r0, c0, r1, c1, color))
    containers.sort()
    return containers


def _find_legend_order(grid: np.ndarray, containers: list[tuple[int, int, int, int, int]]) -> list[int]:
    limit = min((r0 for r0, _c0, _r1, _c1, _color in containers), default=grid.shape[0])
    best_row = None
    best_cols: np.ndarray | None = None
    for r in range(limit):
        cols = np.flatnonzero(grid[r] != BLACK)
        if len(cols) >= 2:
            best_row = r
            best_cols = cols
            break
    if best_row is None or best_cols is None:
        for r in range(grid.shape[0]):
            cols = np.flatnonzero(grid[r] != BLACK)
            if len(cols) >= 1:
                best_row = r
                best_cols = cols
                break
    if best_row is None or best_cols is None:
        return []

    order: list[int] = []
    seen: set[int] = set()
    for c in best_cols.tolist():
        color = int(grid[best_row, int(c)])
        if color != BLACK and color not in seen:
            seen.add(color)
            order.append(color)
    return order


def _axis_positions(length: int, count: int) -> list[int]:
    if count == 1:
        return [length // 2]
    vals = np.rint(np.linspace(0, length - 1, count)).astype(int).tolist()
    assert len(set(vals)) == count
    return [int(v) for v in vals]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder each container's colored markers to match the legend order.

    The legend is the topmost row of colored pixels. Each hollow rectangular frame is a container.
    Horizontal containers are reordered left-to-right; vertical containers are reordered top-to-bottom.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    containers = _find_containers(g)
    legend_order = _find_legend_order(g, containers)
    legend_rank = {color: i for i, color in enumerate(legend_order)}

    for r0, c0, r1, c1, frame_color in containers:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_coords = np.argwhere((inner != BLACK) & (inner != frame_color))
        if len(marker_coords) == 0:
            continue

        present = [int(inner[r, c]) for r, c in marker_coords]
        unique_present = sorted(set(present), key=lambda color: legend_rank.get(color, 10**9))

        inner_out = out[r0 + 1 : r1, c0 + 1 : c1]
        inner_out[(inner_out != BLACK) & (inner_out != frame_color)] = BLACK

        inner_h, inner_w = inner.shape
        if (c1 - c0 + 1) > (r1 - r0 + 1):
            cols = _axis_positions(inner_w, len(unique_present))
            row = inner_h // 2
            for color, col in zip(unique_present, cols, strict=False):
                inner_out[row, col] = int(color)
        else:
            rows = _axis_positions(inner_h, len(unique_present))
            col = inner_w // 2
            for color, row in zip(unique_present, rows, strict=False):
                inner_out[row, col] = int(color)

    return out


def _sample_legend_columns(width: int, n_colors: int) -> list[int]:
    start = random.randint(1, 3)
    cols = [start]
    while len(cols) < n_colors:
        cols.append(cols[-1] + random.randint(1, 2))
    if cols[-1] >= width - 1:
        shift = cols[-1] - (width - 2)
        cols = [c - shift for c in cols]
    return cols


def _container_dims(orientation: str, marker_count: int) -> tuple[int, int]:
    if orientation == "h":
        inner_h = random.randint(2, 4)
        inner_w = random.randint(max(5, marker_count), 9)
        if inner_w <= inner_h:
            inner_w = inner_h + 1
    else:
        inner_w = random.randint(2, 4)
        inner_h = random.randint(max(5, marker_count), 9)
        if inner_h <= inner_w:
            inner_h = inner_w + 1
    return inner_h + 2, inner_w + 2


def _target_cells(r0: int, c0: int, h: int, w: int, colors: list[int]) -> dict[int, tuple[int, int]]:
    inner_h = h - 2
    inner_w = w - 2
    if w > h:
        cols = _axis_positions(inner_w, len(colors))
        row = r0 + 1 + inner_h // 2
        return {color: (row, c0 + 1 + col) for color, col in zip(colors, cols, strict=False)}
    rows = _axis_positions(inner_h, len(colors))
    col = c0 + 1 + inner_w // 2
    return {color: (r0 + 1 + row, col) for color, row in zip(colors, rows, strict=False)}


def _placeable(used: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    H, W = used.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(H, r0 + h + 1)
    cc1 = min(W, c0 + w + 1)
    return not used[rr0:rr1, cc0:cc1].any()


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    grid[r0, c0 : c0 + w] = FRAME
    grid[r0 + h - 1, c0 : c0 + w] = FRAME
    grid[r0 : r0 + h, c0] = FRAME
    grid[r0 : r0 + h, c0 + w - 1] = FRAME


def _render_expected(
    grid: np.ndarray,
    specs: list[dict[str, object]],
    legend_order: list[int],
) -> np.ndarray:
    out = grid.copy()
    legend_rank = {color: i for i, color in enumerate(legend_order)}
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        h = int(spec["h"])
        w = int(spec["w"])
        placements = dict(spec["placements"])
        for r, c in placements.values():
            out[int(r), int(c)] = BLACK
        ordered_colors = sorted(placements, key=lambda color: legend_rank[color])
        for color, (r, c) in _target_cells(r0, c0, h, w, ordered_colors).items():
            out[r, c] = color
    return out


def _sample_container_specs(height: int, width: int, legend_colors: list[int]) -> list[dict[str, object]] | None:
    used = np.zeros((height, width), dtype=bool)
    used[0:2, :] = True

    container_count = random.randint(*CONTAINER_COUNT_RANGE)
    orientations = ["h", "v"]
    while len(orientations) < container_count:
        orientations.append(random.choice(("h", "v")))
    random.shuffle(orientations)

    specs: list[dict[str, object]] = []
    for orientation in orientations:
        marker_count = random.randint(MARKER_COUNT_RANGE[0], min(MARKER_COUNT_RANGE[1], len(legend_colors)))
        h, w = _container_dims(orientation, marker_count)
        candidates: list[tuple[int, int]] = []
        for r0 in range(3, height - h + 1):
            for c0 in range(1, width - w):
                if _placeable(used, r0, c0, h, w):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        used[r0 : r0 + h, c0 : c0 + w] = True

        chosen = random.sample(legend_colors, marker_count)
        ordered = [color for color in legend_colors if color in chosen]
        target_by_color = _target_cells(r0, c0, h, w, ordered)

        interior_cells = [(r, c) for r in range(r0 + 1, r0 + h - 1) for c in range(c0 + 1, c0 + w - 1)]
        placements: dict[int, tuple[int, int]] | None = None
        for _ in range(80):
            colors = ordered[:]
            random.shuffle(colors)
            cells = random.sample(interior_cells, marker_count)
            trial = {color: cell for color, cell in zip(colors, cells, strict=False)}
            if any(trial[color] != target_by_color[color] for color in ordered):
                placements = trial
                break
        if placements is None:
            return None

        specs.append(
            {
                "r0": r0,
                "c0": c0,
                "h": h,
                "w": w,
                "placements": placements,
            }
        )
    return specs


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    legend_len = random.randint(*LEGEND_LEN_RANGE)
    legend_colors = random.sample(list(PALETTE), legend_len)

    specs = _sample_container_specs(height, width, legend_colors)
    if specs is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    legend_cols = _sample_legend_columns(width, legend_len)
    for color, col in zip(legend_colors, legend_cols, strict=False):
        grid[0, col] = color

    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        h = int(spec["h"])
        w = int(spec["w"])
        _draw_frame(grid, r0, c0, h, w)
        for color, (r, c) in dict(spec["placements"]).items():
            grid[int(r), int(c)] = int(color)

    expected = _render_expected(grid, specs, legend_colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
