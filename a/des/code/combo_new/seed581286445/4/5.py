from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SEP = 5
SIDE_NAMES = ("up", "down", "left", "right")
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
CELL_SIZES = (5, 7)
MAX_GENERATION_ATTEMPTS = 400


def _find_separator_color(grid: np.ndarray) -> int:
    row_counts: Counter[int] = Counter()
    col_counts: Counter[int] = Counter()

    for r in range(grid.shape[0]):
        if bool(np.all(grid[r, :] == grid[r, 0])):
            row_counts[int(grid[r, 0])] += 1
    for c in range(grid.shape[1]):
        if bool(np.all(grid[:, c] == grid[0, c])):
            col_counts[int(grid[0, c])] += 1

    candidates = [color for color in row_counts if col_counts[color] > 0]
    assert candidates, "expected a separator color with full rows and columns"
    candidates.sort(
        key=lambda color: (
            row_counts[color] + col_counts[color],
            row_counts[color] * col_counts[color],
        ),
        reverse=True,
    )
    return int(candidates[0])


def _non_separator_runs(grid: np.ndarray, axis: int, sep_color: int) -> list[slice]:
    if axis == 0:
        mask = np.all(grid == sep_color, axis=1)
    else:
        mask = np.all(grid == sep_color, axis=0)

    runs: list[slice] = []
    start: int | None = None
    for idx, is_sep in enumerate(mask.tolist()):
        if not is_sep and start is None:
            start = int(idx)
        elif is_sep and start is not None:
            runs.append(slice(start, int(idx)))
            start = None
    if start is not None:
        runs.append(slice(start, len(mask)))

    assert runs, "expected cell runs between separator lines"
    return runs


def _classify_cell(patch: np.ndarray) -> dict[str, object]:
    values, counts = np.unique(patch, return_counts=True)
    area = int(patch.size)
    if len(values) == 2 and int(np.max(counts)) == area - 1 and int(np.min(counts)) == 1:
        signal_color = int(values[int(np.argmin(counts))])
        signal_pos = np.argwhere(patch == signal_color)
        assert len(signal_pos) == 1, "signal cell must contain exactly one signal pixel"
        center = np.array([patch.shape[0] // 2, patch.shape[1] // 2], dtype=int)
        delta = tuple((signal_pos[0] - center).tolist())
        direction = next((name for name, vec in DIRS.items() if vec == delta), None)
        assert direction is not None, "signal pixel must be offset by one step from center"
        return {"kind": "B", "direction": direction, "patch": patch}
    return {"kind": "A", "patch": patch}


def _edge_stripe_color(patch: np.ndarray, side: str) -> int:
    if side == "up":
        edge = patch[0, :]
    elif side == "down":
        edge = patch[-1, :]
    elif side == "left":
        edge = patch[:, 0]
    else:
        assert side == "right"
        edge = patch[:, -1]

    color_counts = Counter(map(int, edge.reshape(-1).tolist()))
    color, count = color_counts.most_common(1)[0]
    assert count == edge.size, "targeted edge must be a full-width or full-height stripe"
    return int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each signal-marked blank cell with the stripe color copied from its pointed neighbor.

    Input structure:
    - Black background cells and full grey separator rows/columns partition the grid
      into equal rectangular cells.
    - Each A cell is filled by one main color and has a one-pixel stripe color on one
      or more edges.
    - Each B cell is otherwise black and contains exactly one signal pixel shifted by
      one step from the cell center.

    Rule:
    - Detect the separator color, then slice the grid into equal cell interiors between
      separator lines.
    - For every B cell, read the signal pixel's offset direction from the center.
    - Move to the directly adjacent cell in that direction and read the stripe color on
      the neighbor edge that faces back toward the B cell.
    - Fill the entire B cell with that stripe color. Keep A cells and grey separators
      unchanged.

    Key insight:
    - The signal pixel encodes a direction, and that direction tells you which neighbor
      to inspect; the answer is not the neighbor's fill color, but the inward-facing edge stripe.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_color = _find_separator_color(g)
    row_slices = _non_separator_runs(g, axis=0, sep_color=sep_color)
    col_slices = _non_separator_runs(g, axis=1, sep_color=sep_color)

    row_sizes = {row_slice.stop - row_slice.start for row_slice in row_slices}
    col_sizes = {col_slice.stop - col_slice.start for col_slice in col_slices}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "all cells must share one size"
    cell_h = row_slices[0].stop - row_slices[0].start
    cell_w = col_slices[0].stop - col_slices[0].start
    assert cell_h % 2 == 1 and cell_w % 2 == 1, "cell centers must be well-defined"

    infos: list[list[dict[str, object]]] = []
    for row_slice in row_slices:
        row_infos: list[dict[str, object]] = []
        for col_slice in col_slices:
            row_infos.append(_classify_cell(g[row_slice, col_slice]))
        infos.append(row_infos)

    b_count = sum(info["kind"] == "B" for row in infos for info in row)
    a_count = sum(info["kind"] == "A" for row in infos for info in row)
    assert b_count > 0 and b_count < a_count, "expected fewer B cells than A cells"

    out = g.copy()
    for r, row_infos in enumerate(infos):
        for c, info in enumerate(row_infos):
            if info["kind"] != "B":
                continue
            direction = str(info["direction"])
            dr, dc = DIRS[direction]
            nr, nc = r + dr, c + dc
            assert 0 <= nr < len(row_slices) and 0 <= nc < len(col_slices), "signal must point to a neighbor"
            neighbor = infos[nr][nc]
            assert neighbor["kind"] == "A", "signal must point to an A cell"
            stripe_color = _edge_stripe_color(np.asarray(neighbor["patch"]), OPPOSITE[direction])
            out[row_slices[r], col_slices[c]] = int(stripe_color)

    return out


def _grid_extent(count: int, cell_size: int, framed: bool) -> int:
    return count * cell_size + (count + 1 if framed else count - 1)


def _cell_slices(count: int, cell_size: int, framed: bool) -> list[slice]:
    offset = 1 if framed else 0
    step = cell_size + 1
    return [
        slice(offset + idx * step, offset + idx * step + cell_size)
        for idx in range(count)
    ]


def _sample_layout() -> tuple[int, int, int, int, bool]:
    for _ in range(100):
        rows = random.randint(2, 4)
        cols = random.randint(2, 4)
        cell_h = random.choice(CELL_SIZES)
        cell_w = random.choice(CELL_SIZES)
        framed = random.random() < 0.5
        if _grid_extent(rows, cell_h, framed) <= 30 and _grid_extent(cols, cell_w, framed) <= 30:
            return rows, cols, cell_h, cell_w, framed
    raise RuntimeError("failed to sample a valid layout")


def _sample_directions(count: int) -> list[str]:
    directions: list[str] = []
    while len(directions) < count:
        block = list(SIDE_NAMES)
        random.shuffle(block)
        directions.extend(block)
    return directions[:count]


def _place_b_cells(rows: int, cols: int, directions: list[str]) -> dict[tuple[int, int], str] | None:
    b_to_dir: dict[tuple[int, int], str] = {}
    forbidden_b_positions: set[tuple[int, int]] = set()

    for direction in directions:
        dr, dc = DIRS[direction]
        candidates: list[tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                pos = (r, c)
                target = (r + dr, c + dc)
                if pos in b_to_dir or pos in forbidden_b_positions:
                    continue
                if not (0 <= target[0] < rows and 0 <= target[1] < cols):
                    continue
                if target in b_to_dir:
                    continue
                candidates.append(pos)
        if not candidates:
            return None
        pos = random.choice(candidates)
        target = (pos[0] + dr, pos[1] + dc)
        b_to_dir[pos] = direction
        forbidden_b_positions.add(target)
    return b_to_dir


def _paint_a_patch(
    cell_h: int,
    cell_w: int,
    fill_color: int,
    stripe_color: int,
    stripe_sides: set[str],
) -> np.ndarray:
    patch = np.full((cell_h, cell_w), int(fill_color), dtype=int)
    if "up" in stripe_sides:
        patch[0, :] = int(stripe_color)
    if "down" in stripe_sides:
        patch[-1, :] = int(stripe_color)
    if "left" in stripe_sides:
        patch[:, 0] = int(stripe_color)
    if "right" in stripe_sides:
        patch[:, -1] = int(stripe_color)
    return patch


def _paint_b_patch(cell_h: int, cell_w: int, signal_color: int, direction: str) -> np.ndarray:
    patch = np.full((cell_h, cell_w), BG, dtype=int)
    center_r = cell_h // 2
    center_c = cell_w // 2
    dr, dc = DIRS[direction]
    patch[center_r + dr, center_c + dc] = int(signal_color)
    return patch


def _gen_attempt() -> np.ndarray | None:
    rows, cols, cell_h, cell_w, framed = _sample_layout()
    total_cells = rows * cols
    max_b = (total_cells - 1) // 2
    if max_b < 1:
        return None

    b_count = random.randint(1, max_b)
    b_directions = _sample_directions(b_count)
    b_to_dir = _place_b_cells(rows, cols, b_directions)
    if b_to_dir is None:
        return None

    required_sides: dict[tuple[int, int], set[str]] = {}
    for (r, c), direction in b_to_dir.items():
        dr, dc = DIRS[direction]
        target = (r + dr, c + dc)
        required_sides.setdefault(target, set()).add(OPPOSITE[direction])

    signal_color = random.choice([color for color in range(10) if color not in (BG, SEP)])
    a_palette = [color for color in range(10) if color not in (BG, SEP, signal_color)]

    a_specs: dict[tuple[int, int], dict[str, object]] = {}
    for r in range(rows):
        for c in range(cols):
            pos = (r, c)
            if pos in b_to_dir:
                continue
            fill_color = random.choice(a_palette)
            stripe_choices = [color for color in a_palette if color != fill_color]
            stripe_color = random.choice(stripe_choices)

            stripe_sides = set(required_sides.get(pos, set()))
            if stripe_sides:
                for side in SIDE_NAMES:
                    if side not in stripe_sides and random.random() < 0.35:
                        stripe_sides.add(side)
            else:
                stripe_sides = set(random.sample(SIDE_NAMES, random.randint(1, len(SIDE_NAMES))))

            a_specs[pos] = {
                "fill_color": fill_color,
                "stripe_color": stripe_color,
                "stripe_sides": stripe_sides,
            }

    height = _grid_extent(rows, cell_h, framed)
    width = _grid_extent(cols, cell_w, framed)
    inp = np.full((height, width), SEP, dtype=int)
    expected = inp.copy()
    row_slices = _cell_slices(rows, cell_h, framed)
    col_slices = _cell_slices(cols, cell_w, framed)

    for r, row_slice in enumerate(row_slices):
        for c, col_slice in enumerate(col_slices):
            pos = (r, c)
            if pos in b_to_dir:
                patch = _paint_b_patch(cell_h, cell_w, signal_color, b_to_dir[pos])
                inp[row_slice, col_slice] = patch
                dr, dc = DIRS[b_to_dir[pos]]
                target = (r + dr, c + dc)
                expected[row_slice, col_slice] = int(a_specs[target]["stripe_color"])
            else:
                spec = a_specs[pos]
                patch = _paint_a_patch(
                    cell_h,
                    cell_w,
                    int(spec["fill_color"]),
                    int(spec["stripe_color"]),
                    set(spec["stripe_sides"]),
                )
                inp[row_slice, col_slice] = patch
                expected[row_slice, col_slice] = patch

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, inp):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
