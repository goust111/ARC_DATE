import math
from typing import Iterable

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder each container's bottom cargo row using a shared corner legend and a top/bottom direction tag.

    Input structure:
    - The background is black.
    - One corner contains a vertical legend of distinct single pixels; from top to bottom it defines a global color-priority ladder.
    - Several hollow rectangular containers are drawn with one border color each, and each container has a same-color one-pixel tag attached just above or just below the frame.
    - The bottom interior row of each container holds a set of colored cargo pixels on black cells.

    Rule:
    - Read the corner legend from top to bottom to obtain the priority order, with the top legend color as the highest priority.
    - For each container, inspect whether its same-color tag is above the frame or below it.
    - If the tag is above the frame, sort the cargo from low priority to high priority from left to right.
    - If the tag is below the frame, sort the cargo from high priority to low priority from left to right.
    - Rewrite the sorted cargo onto the same bottom interior row, packed against the left inner wall; keep the frames, tags, legend, and all other black cells unchanged.

    Key insight:
    - The corner legend supplies one global ranking shared by every container, while the tag position encodes whether that ranking should be read forward or backward for that specific container.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    priority = _find_legend_priority(g, bg)
    assert priority, "expected a corner legend"

    containers = []
    for color in _colors_present(g, exclude={bg}):
        for component in _components_of_color(g, color):
            info = _parse_container(g, component, bg)
            if info is not None:
                containers.append(info)
    assert containers, "expected at least one tagged container"

    out = g.copy()
    containers.sort(key=lambda item: (item["frame"][0], item["frame"][1]))
    for info in containers:
        frame_r0, frame_c0, _frame_r1, frame_c1 = info["frame"]
        cargo_row = info["cargo_row"]
        cargo_colors = info["cargo_colors"]
        assert cargo_colors
        assert all(color in priority for color in cargo_colors), "cargo colors must appear in the legend"

        reverse = info["direction"] == "asc"
        ordered = sorted(cargo_colors, key=lambda color: priority[color], reverse=reverse)

        out[cargo_row, frame_c0 + 1 : frame_c1] = bg
        out[cargo_row, frame_c0 + 1 : frame_c0 + 1 + len(ordered)] = ordered

    return out


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    exclude = exclude or set()
    return [int(color) for color in np.unique(grid) if int(color) not in exclude]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            components.append(cells)

    return components


def _corner_legend_candidate(
    grid: np.ndarray,
    bg: int,
    start_row: int,
    step: int,
    col: int,
    min_len: int = 4,
) -> list[tuple[int, int]] | None:
    h, _w = grid.shape
    run: list[tuple[int, int]] = []
    row = start_row

    while 0 <= row < h and int(grid[row, col]) != bg:
        run.append((row, col))
        row += step

    if len(run) < min_len:
        return None

    if step < 0:
        run.reverse()

    colors = [int(grid[r, c]) for r, c in run]
    if len(set(colors)) != len(colors):
        return None

    run_set = set(run)
    for r, c in run:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                continue
            if (nr, nc) in run_set:
                continue
            if int(grid[nr, nc]) != bg:
                return None

    return run


def _find_legend_priority(grid: np.ndarray, bg: int) -> dict[int, int]:
    h, w = grid.shape
    candidates = [
        _corner_legend_candidate(grid, bg, 0, 1, 0),
        _corner_legend_candidate(grid, bg, 0, 1, w - 1),
        _corner_legend_candidate(grid, bg, h - 1, -1, 0),
        _corner_legend_candidate(grid, bg, h - 1, -1, w - 1),
    ]
    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        return {}

    legend = max(candidates, key=len)
    return {int(grid[r, c]): idx for idx, (r, c) in enumerate(legend)}


def _is_hollow_rectangle(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 4 or w < 4:
        return False

    expected = np.zeros_like(mask, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(mask, expected)


def _parse_container(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    bg: int,
) -> dict[str, object] | None:
    if len(cells) < 10:
        return None

    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        patch[r - r0, c - c0] = True

    options: list[tuple[str, tuple[int, int, int, int]]] = []
    if patch.shape[0] >= 5 and patch[0].sum() == 1:
        tag_col = int(np.flatnonzero(patch[0])[0])
        if 0 < tag_col < patch.shape[1] - 1 and _is_hollow_rectangle(patch[1:, :]):
            options.append(("asc", (r0 + 1, c0, r1, c1)))

    if patch.shape[0] >= 5 and patch[-1].sum() == 1:
        tag_col = int(np.flatnonzero(patch[-1])[0])
        if 0 < tag_col < patch.shape[1] - 1 and _is_hollow_rectangle(patch[:-1, :]):
            options.append(("desc", (r0, c0, r1 - 1, c1)))

    if len(options) != 1:
        return None

    direction, frame = options[0]
    frame_r0, frame_c0, frame_r1, frame_c1 = frame
    cargo_row = frame_r1 - 1
    cargo_colors = [
        int(grid[cargo_row, c])
        for c in range(frame_c0 + 1, frame_c1)
        if int(grid[cargo_row, c]) != bg
    ]
    if not cargo_colors:
        return None

    return {
        "direction": direction,
        "frame": frame,
        "cargo_row": cargo_row,
        "cargo_colors": cargo_colors,
    }


# --------------- generator ---------------

_GEN_BG = 0
_CARGO_POOL = (1, 2, 3, 4, 6, 7)
_BORDER_POOL = (5, 8, 9)
_GEN_RETRIES = 300


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    legend_len = int(rng.integers(4, 7))
    legend_colors = [int(x) for x in rng.choice(_CARGO_POOL, size=legend_len, replace=False)]

    frame_h = int(rng.integers(4, 7))
    frame_w = int(rng.integers(5, 9))
    inner_w = frame_w - 2
    if inner_w < 2:
        return None

    container_count = int(rng.integers(2, 5))
    comp_h = frame_h + 1
    comp_w = frame_w

    layouts = []
    for rows in range(1, container_count + 1):
        cols = math.ceil(container_count / rows)
        min_h = rows * comp_h + (rows + 1) * 2
        min_w = cols * comp_w + (cols + 1) * 2
        if min_h <= 30 and min_w <= 30:
            layouts.append((rows, cols, min_h, min_w))
    if not layouts:
        return None

    _rows, _cols, min_h, min_w = layouts[int(rng.integers(len(layouts)))]
    grid_h = int(rng.integers(min_h, min(30, min_h + 6) + 1))
    grid_w = int(rng.integers(min_w, min(30, min_w + 8) + 1))

    grid = np.full((grid_h, grid_w), _GEN_BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    corner = ("tl", "tr", "bl", "br")[int(rng.integers(4))]
    legend_cells = _legend_cells(grid_h, grid_w, legend_len, corner)
    for color, (r, c) in zip(legend_colors, legend_cells):
        grid[r, c] = color
    _reserve_cells(occupied, legend_cells)

    directions = ["asc", "desc"]
    while len(directions) < container_count:
        directions.append("asc" if int(rng.integers(2)) == 0 else "desc")
    rng.shuffle(directions)

    rank = {color: idx for idx, color in enumerate(legend_colors)}
    any_change = False

    for idx in range(container_count):
        border_color = int(rng.choice(_BORDER_POOL))
        direction = directions[idx]
        cargo_count = int(rng.integers(2, min(inner_w, legend_len) + 1))
        cargo_colors = [int(x) for x in rng.choice(legend_colors, size=cargo_count, replace=False)]
        target = _sorted_cargo(cargo_colors, rank, direction)
        start_cols, start_colors = _scrambled_start_state(rng, cargo_colors, inner_w, target)
        if start_cols is None or start_colors is None:
            return None

        placed = False
        for _ in range(200):
            r = int(rng.integers(1, grid_h - comp_h))
            c = int(rng.integers(1, grid_w - comp_w))
            if not _bbox_is_free(occupied, r, c, comp_h, comp_w):
                continue

            tag_col = int(rng.integers(1, frame_w - 1))
            _draw_container(
                grid,
                r,
                c,
                frame_h,
                frame_w,
                border_color,
                direction,
                tag_col,
                start_cols,
                start_colors,
            )
            _reserve_bbox(occupied, r, c, comp_h, comp_w)
            placed = True
            any_change = True
            break

        if not placed:
            return None

    if not any_change:
        return None

    solved = main(grid)
    if np.array_equal(grid, solved):
        return None
    return grid


def _legend_cells(grid_h: int, grid_w: int, legend_len: int, corner: str) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(r, 0) for r in range(legend_len)]
    if corner == "tr":
        return [(r, grid_w - 1) for r in range(legend_len)]
    if corner == "bl":
        return [(grid_h - legend_len + r, 0) for r in range(legend_len)]
    return [(grid_h - legend_len + r, grid_w - 1) for r in range(legend_len)]


def _reserve_cells(occupied: np.ndarray, cells: Iterable[tuple[int, int]]) -> None:
    for r, c in cells:
        r0 = max(0, r - 1)
        r1 = min(occupied.shape[0], r + 2)
        c0 = max(0, c - 1)
        c1 = min(occupied.shape[1], c + 2)
        occupied[r0:r1, c0:c1] = True


def _bbox_is_free(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    r0 = max(0, r - 1)
    r1 = min(occupied.shape[0], r + h + 1)
    c0 = max(0, c - 1)
    c1 = min(occupied.shape[1], c + w + 1)
    return not occupied[r0:r1, c0:c1].any()


def _reserve_bbox(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    r0 = max(0, r - 1)
    r1 = min(occupied.shape[0], r + h + 1)
    c0 = max(0, c - 1)
    c1 = min(occupied.shape[1], c + w + 1)
    occupied[r0:r1, c0:c1] = True


def _sorted_cargo(colors: list[int], rank: dict[int, int], direction: str) -> list[int]:
    reverse = direction == "asc"
    return sorted(colors, key=lambda color: rank[color], reverse=reverse)


def _scrambled_start_state(
    rng: np.random.Generator,
    cargo_colors: list[int],
    inner_w: int,
    target: list[int],
) -> tuple[list[int] | None, list[int] | None]:
    cargo_count = len(cargo_colors)
    target_cols = list(range(cargo_count))

    for _ in range(100):
        cols = sorted(int(x) for x in rng.choice(inner_w, size=cargo_count, replace=False))
        colors = [int(x) for x in rng.permutation(cargo_colors)]
        if cols != target_cols or colors != target:
            return cols, colors
    return None, None


def _draw_container(
    grid: np.ndarray,
    comp_r0: int,
    comp_c0: int,
    frame_h: int,
    frame_w: int,
    border_color: int,
    direction: str,
    tag_col: int,
    cargo_cols: list[int],
    cargo_colors: list[int],
) -> None:
    if direction == "asc":
        frame_r0 = comp_r0 + 1
        tag_r = comp_r0
    else:
        frame_r0 = comp_r0
        tag_r = comp_r0 + frame_h

    frame_r1 = frame_r0 + frame_h - 1
    frame_c1 = comp_c0 + frame_w - 1

    grid[frame_r0, comp_c0 : frame_c1 + 1] = border_color
    grid[frame_r1, comp_c0 : frame_c1 + 1] = border_color
    grid[frame_r0 : frame_r1 + 1, comp_c0] = border_color
    grid[frame_r0 : frame_r1 + 1, frame_c1] = border_color
    grid[tag_r, comp_c0 + tag_col] = border_color

    cargo_row = frame_r1 - 1
    for pos, color in zip(cargo_cols, cargo_colors):
        grid[cargo_row, comp_c0 + 1 + pos] = color
